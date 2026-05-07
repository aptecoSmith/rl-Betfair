"""Supervised probe: can ``direction_prob_head`` actually learn the
labels in `data/direction_labels/<date>/...`?

Phase-13 follow-up. The S06 cohort showed direction-prob BCE flat
at ~1.04 across 4 generations. To distinguish "head can't learn the
label" from "PPO update isn't moving the head" we strip PPO entirely
and train ``direction_prob_head`` directly via supervised BCE-with-
logits on the cached labels. If the head can fit the labels in pure
supervised mode, the bottleneck is somewhere downstream
(representation, capacity, PPO reaching the head). If it CAN'T fit,
the label / feature pairing is the bottleneck.

Runs in ~minute on GPU per (date, horizon) tuple. Two label caches
(horizon=60, horizon=6) compared head-to-head with identical seed +
identical policy init, so the only difference is the label.

Usage::

    python -m tools.direction_head_supervised_probe \
        --date 2026-05-03 \
        --horizons 60,6 \
        --thresholds 5,2 \
        --device cuda \
        --steps 500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from agents_v2.env_shim import DEFAULT_SCORER_DIR, DiscreteActionShim
from agents_v2.discrete_policy import DiscreteLSTMPolicy
from data.episode_builder import load_day
from env.betfair_env import (
    AGENT_STATE_DIM,
    POSITION_DIM,
    SCALPING_AGENT_STATE_DIM,
    SCALPING_POSITION_DIM,
    BetfairEnv,
)
from training_v2.arb_oracle import _load_config
from training_v2.direction_label_scan import load_labels
from training_v2.scorer.feature_extractor import FeatureExtractor


def build_per_tick_obs(date: str, data_dir: Path, config: dict):
    """Walk *date* and return ``(obs_per_tick, runner_map_per_race,
    pre_race_tick_to_race_idx)``.

    ``obs_per_tick`` is shape ``(N_pre_race_ticks, shim.obs_dim)``
    float32. Indexing matches the global pre-race tick numbering used
    by the offline label scan.
    """
    day = load_day(date, data_dir)
    if not day.races:
        raise RuntimeError(f"No races on {date}")
    env = BetfairEnv(day, config, scalping_mode=True)
    shim = DiscreteActionShim(env, scorer_dir=DEFAULT_SCORER_DIR)
    max_runners = env.max_runners

    agent_state_dim = AGENT_STATE_DIM + SCALPING_AGENT_STATE_DIM
    position_dim = max_runners * (POSITION_DIM + SCALPING_POSITION_DIM)
    zero_agent = np.zeros(agent_state_dim, dtype=np.float32)
    zero_agent[1] = 1.0  # budget_frac
    zero_pos = np.zeros(position_dim, dtype=np.float32)

    obs_list = []
    for race_idx, race in enumerate(day.races):
        env._race_idx = race_idx
        shim._feature_extractor = FeatureExtractor()
        for tick_idx, tick in enumerate(race.ticks):
            env._tick_idx = tick_idx
            shim._update_history_for_current_tick()
            if tick.in_play:
                continue
            static = env._static_obs[race_idx][tick_idx]
            base = np.concatenate([static, zero_agent, zero_pos])
            ext = shim.compute_extended_obs(base)
            obs_list.append(ext.astype(np.float32))

    obs_arr = np.stack(obs_list)
    return obs_arr, max_runners, shim.obs_dim


def run_probe(
    *,
    date: str,
    data_dir: Path,
    config: dict,
    horizon_ticks: int,
    threshold_ticks: int,
    fc_secs: float,
    device: str,
    n_steps: int,
    batch_size: int,
    log_every: int,
    seed: int,
):
    print(f"\n=== probe horizon={horizon_ticks} thresh={threshold_ticks} ===")
    t0 = time.monotonic()

    # Build day obs.
    obs_arr, max_runners, obs_dim = build_per_tick_obs(
        date, data_dir, config,
    )
    n_ticks = obs_arr.shape[0]
    print(f"  pre-race ticks: {n_ticks}, obs_dim: {obs_dim}, "
          f"max_runners: {max_runners}, "
          f"build_wall: {time.monotonic() - t0:.1f}s")

    # Load labels.
    labels = load_labels(
        date, data_dir,
        direction_horizon_ticks=horizon_ticks,
        direction_threshold_ticks=threshold_ticks,
        force_close_before_off_seconds=fc_secs,
        strict=True,
    )
    print(f"  labels loaded: {len(labels)}")

    # Build the per-(tick, runner) sample arrays. Drop labels whose
    # tick_index falls outside the obs array AND whose runner_idx
    # exceeds the env's max_runners (the scan's own runner-slot map
    # includes inactive / removed runners which the env discards;
    # this is a known bug in `direction_label_scan._runner_slot_map`
    # that the cohort also silently dropped — fix tracked separately).
    valid = [
        r for r in labels
        if 0 <= r.tick_index < n_ticks
        and 0 <= r.runner_idx < max_runners
    ]
    n_dropped_tick = sum(
        1 for r in labels if not (0 <= r.tick_index < n_ticks)
    )
    n_dropped_runner = sum(
        1 for r in labels if not (0 <= r.runner_idx < max_runners)
    )
    if n_dropped_tick or n_dropped_runner:
        print(f"  dropped: {n_dropped_tick} tick-OOB, "
              f"{n_dropped_runner} runner-OOB")
    if len(valid) < len(labels):
        print(f"  WARNING: dropped {len(labels) - len(valid)} labels"
              " with out-of-range tick_index")
    tick_idx = np.array([r.tick_index for r in valid], dtype=np.int64)
    runner_idx = np.array(
        [r.runner_idx for r in valid], dtype=np.int64,
    )
    label_back = np.array(
        [r.label_back for r in valid], dtype=np.float32,
    )
    label_lay = np.array(
        [r.label_lay for r in valid], dtype=np.float32,
    )
    n_samples = tick_idx.shape[0]
    pos_back = float(label_back.mean())
    pos_lay = float(label_lay.mean())
    print(f"  positive density: back={pos_back:.4f} lay={pos_lay:.4f}")

    # Class-balance pos_weight (same recipe as the trainer).
    pw_back = (1.0 - pos_back) / max(pos_back, 1e-6)
    pw_lay = (1.0 - pos_lay) / max(pos_lay, 1e-6)
    print(f"  pos_weight: back={pw_back:.3f} lay={pw_lay:.3f}")

    # Build a fresh policy with the same default config the cohort
    # uses. Move to device. Same seed for both probes so init is
    # bit-identical.
    torch.manual_seed(int(seed))
    from agents_v2.action_space import DiscreteActionSpace
    action_space = DiscreteActionSpace(max_runners=max_runners)
    policy = DiscreteLSTMPolicy(
        obs_dim=obs_dim,
        action_space=action_space,
        hidden_size=128,
    ).to(device)

    # Freeze everything except direction_prob_head. The probe asks
    # "can THIS specific head learn THESE labels given the
    # representation the LSTM produces from a fresh init?" — same
    # capacity question the cohort exposes.
    for n, p in policy.named_parameters():
        if "direction_prob_head" in n:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    target_params = [
        p for p in policy.parameters() if p.requires_grad
    ]
    opt = torch.optim.Adam(target_params, lr=3e-4)

    obs_t = torch.from_numpy(obs_arr).to(device)
    pw_back_t = torch.tensor(pw_back, dtype=torch.float32, device=device)
    pw_lay_t = torch.tensor(pw_lay, dtype=torch.float32, device=device)
    rng = np.random.default_rng(int(seed))

    print(f"  step  loss   bce_back  bce_lay  acc_back  acc_lay  wall")
    bce_history = []
    train_t0 = time.monotonic()
    for step in range(n_steps):
        idx = rng.integers(0, n_samples, size=batch_size)
        b_tick = torch.from_numpy(tick_idx[idx]).to(device)
        b_runner = torch.from_numpy(runner_idx[idx]).to(device)
        b_lback = torch.from_numpy(label_back[idx]).to(device)
        b_llay = torch.from_numpy(label_lay[idx]).to(device)

        b_obs = obs_t.index_select(0, b_tick)  # (B, obs_dim)
        # Run through the policy. The policy's forward expects
        # (batch, obs_dim) which it lifts to (batch, 1, obs_dim) and
        # initialises hidden=zero. That's exactly what we want for a
        # tick-independent supervised probe.
        out = policy(b_obs)
        # (B, max_runners) per side; gather the column for this row's
        # runner_idx.
        rows = torch.arange(b_obs.shape[0], device=device)
        back_logits = out.direction_back_logits_per_runner[rows, b_runner]
        lay_logits = out.direction_lay_logits_per_runner[rows, b_runner]

        bce_back = nn.functional.binary_cross_entropy_with_logits(
            back_logits, b_lback, pos_weight=pw_back_t,
        )
        bce_lay = nn.functional.binary_cross_entropy_with_logits(
            lay_logits, b_llay, pos_weight=pw_lay_t,
        )
        loss = bce_back + bce_lay

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == n_steps - 1:
            with torch.no_grad():
                p_back = torch.sigmoid(back_logits)
                p_lay = torch.sigmoid(lay_logits)
                acc_back = ((p_back > 0.5) == (b_lback > 0.5)).float().mean()
                acc_lay = ((p_lay > 0.5) == (b_llay > 0.5)).float().mean()
            bce_history.append({
                "step": step,
                "loss": float(loss.item()),
                "bce_back": float(bce_back.item()),
                "bce_lay": float(bce_lay.item()),
                "acc_back": float(acc_back.item()),
                "acc_lay": float(acc_lay.item()),
            })
            print(
                f"  {step:4d}  {loss.item():.4f}  "
                f"{bce_back.item():.4f}    {bce_lay.item():.4f}   "
                f"{acc_back.item():.4f}    {acc_lay.item():.4f}   "
                f"{time.monotonic() - train_t0:.1f}s"
            )

    # Held-out eval over the FULL day (no shuffling, no minibatching) —
    # tells us calibration + accuracy against every priceable
    # (tick, runner). With the same data the head was trained on this
    # is in-sample, but it gives a clean read on whether the head's
    # output is meaningful at all.
    with torch.no_grad():
        b_tick = torch.from_numpy(tick_idx).to(device)
        b_runner = torch.from_numpy(runner_idx).to(device)
        b_lback = torch.from_numpy(label_back).to(device)
        b_llay = torch.from_numpy(label_lay).to(device)
        # Process in chunks to avoid one-shot OOM.
        chunk = 4096
        all_back_logits = []
        all_lay_logits = []
        for s in range(0, n_samples, chunk):
            e = min(s + chunk, n_samples)
            sub_obs = obs_t.index_select(0, b_tick[s:e])
            out = policy(sub_obs)
            rows = torch.arange(e - s, device=device)
            all_back_logits.append(
                out.direction_back_logits_per_runner[rows, b_runner[s:e]]
            )
            all_lay_logits.append(
                out.direction_lay_logits_per_runner[rows, b_runner[s:e]]
            )
        back_logits = torch.cat(all_back_logits)
        lay_logits = torch.cat(all_lay_logits)
        p_back = torch.sigmoid(back_logits)
        p_lay = torch.sigmoid(lay_logits)
        acc_back = ((p_back > 0.5) == (b_lback > 0.5)).float().mean()
        acc_lay = ((p_lay > 0.5) == (b_llay > 0.5)).float().mean()

        # Calibration: split into 5 quantile bins of predicted
        # probability and report mean(label) per bin.
        def calib(p, lab, n_bins=5):
            sorted_idx = torch.argsort(p)
            chunks = torch.chunk(sorted_idx, n_bins)
            return [
                (
                    float(p[c].mean().item()),
                    float(lab[c].mean().item()),
                )
                for c in chunks
            ]

        cal_back = calib(p_back, b_lback)
        cal_lay = calib(p_lay, b_llay)

    print(f"\n  Final accuracy (in-sample, full day):")
    print(f"    acc_back: {acc_back.item():.4f}  acc_lay: {acc_lay.item():.4f}")
    print(f"    base-rate (always-predict-majority): "
          f"{max(pos_back, 1 - pos_back):.4f} / "
          f"{max(pos_lay, 1 - pos_lay):.4f}")
    print(f"  Calibration (predicted P → realised P, 5 quantile bins):")
    for i, ((pb, rb), (pl, rl)) in enumerate(zip(cal_back, cal_lay)):
        print(f"    bin {i}: back P_pred={pb:.3f} P_real={rb:.3f}  "
              f"lay P_pred={pl:.3f} P_real={rl:.3f}")

    return {
        "horizon": horizon_ticks,
        "threshold": threshold_ticks,
        "history": bce_history,
        "final_acc_back": float(acc_back.item()),
        "final_acc_lay": float(acc_lay.item()),
        "pos_back": pos_back,
        "pos_lay": pos_lay,
        "calibration_back": cal_back,
        "calibration_lay": cal_lay,
        "wall_total_sec": time.monotonic() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-05-03")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--horizons", default="60,6")
    ap.add_argument("--thresholds", default="5,2")
    ap.add_argument("--fc-secs", type=float, default=60.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    horizons = [int(s) for s in args.horizons.split(",")]
    thresholds = [int(s) for s in args.thresholds.split(",")]
    if len(horizons) != len(thresholds):
        raise ValueError("--horizons and --thresholds must have same length")

    config = _load_config()
    data_dir = Path(args.data_dir)

    results = []
    for h, t in zip(horizons, thresholds):
        results.append(run_probe(
            date=args.date,
            data_dir=data_dir,
            config=config,
            horizon_ticks=h,
            threshold_ticks=t,
            fc_secs=args.fc_secs,
            device=args.device,
            n_steps=args.steps,
            batch_size=args.batch_size,
            log_every=args.log_every,
            seed=args.seed,
        ))

    print("\n=== summary ===")
    for r in results:
        h0_loss = r["history"][0]["loss"]
        h_last_loss = r["history"][-1]["loss"]
        h0_back = r["history"][0]["bce_back"]
        h_last_back = r["history"][-1]["bce_back"]
        h0_lay = r["history"][0]["bce_lay"]
        h_last_lay = r["history"][-1]["bce_lay"]
        baseline_acc_back = max(r["pos_back"], 1 - r["pos_back"])
        baseline_acc_lay = max(r["pos_lay"], 1 - r["pos_lay"])
        gain_back = r["final_acc_back"] - baseline_acc_back
        gain_lay = r["final_acc_lay"] - baseline_acc_lay
        print(
            f"  horizon={r['horizon']} thresh={r['threshold']}: "
            f"loss {h0_loss:.4f} → {h_last_loss:.4f}  |  "
            f"bce_back {h0_back:.4f} → {h_last_back:.4f}  |  "
            f"bce_lay {h0_lay:.4f} → {h_last_lay:.4f}  |  "
            f"acc_back {r['final_acc_back']:.4f} (base {baseline_acc_back:.4f}, "
            f"gain {gain_back:+.4f})  |  "
            f"acc_lay {r['final_acc_lay']:.4f} (base {baseline_acc_lay:.4f}, "
            f"gain {gain_lay:+.4f})  |  "
            f"wall {r['wall_total_sec']:.1f}s"
        )


if __name__ == "__main__":
    main()
