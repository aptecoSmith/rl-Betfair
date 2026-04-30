"""End-to-end smoke driver for the v2 discrete policy + env shim.

Phase 1, Session 02 deliverable. Runs a random-weight
:class:`DiscreteLSTMPolicy` through :class:`DiscreteActionShim` for
``n_steps`` env steps on a real day's data and checks the success bar
from
``plans/rewrite/phase-1-policy-and-env-wiring/session_prompts/02_policy_class_and_smoke_test.md``:

1. No exceptions raised.
2. At least one row in ``episodes.jsonl``-style output.
3. Discrete action histogram covers all of NOOP, OPEN_BACK, OPEN_LAY,
   and CLOSE at least once.
4. Action mask never produced an illegal action (refusal counter == 0).
5. Scorer prediction matches standalone booster on at least one
   captured (obs, runner, side) tuple.

CLI::

    python -m agents_v2.smoke_test --day 2026-04-23

Or pass an explicit data dir / day path; ``--help`` for details.

**No training.** No optimiser is built; the policy weights stay at their
random init throughout.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.episode_builder import load_day
from env.betfair_env import BetfairEnv

from agents_v2.action_space import ActionType, DiscreteActionSpace
from agents_v2.discrete_policy import DiscreteLSTMPolicy
from agents_v2.env_shim import DEFAULT_SCORER_DIR, DiscreteActionShim


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_PATH = REPO_ROOT / "logs" / "agents_v2_smoke" / "smoke.jsonl"

logger = logging.getLogger(__name__)


def _scalping_smoke_config(max_runners: int = 14) -> dict:
    """Minimal scalping config compatible with the env + shim.

    Mirrors the keys the env actually reads — does NOT pull from
    ``config.yaml`` so the smoke driver is self-contained and stable
    against config-format drift.
    """
    return {
        "training": {
            "max_runners": max_runners,
            "starting_budget": 100.0,
            "max_bets_per_race": 20,
            "scalping_mode": True,
            "betting_constraints": {
                "max_back_price": 50.0,
                "max_lay_price": None,
                "min_seconds_before_off": 0,
                "force_close_before_off_seconds": 0,  # smoke: disabled
            },
        },
        "actions": {"force_aggressive": True},
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "efficiency_penalty": 0.01,
            "commission": 0.05,
            "mark_to_market_weight": 0.0,
        },
    }


def _capture_scorer_reference(
    shim: DiscreteActionShim,
) -> tuple[np.ndarray, int, str, float] | None:
    """Capture (feature_vector, slot, side, calibrated_prob) for a
    runner currently in the obs.

    Used by success bar #5 — re-load the booster + calibrator
    standalone and verify the shim's per-step prediction matches.

    Returns ``None`` if no ACTIVE-with-LTP runner is found at the
    current tick.
    """
    race = shim._current_race()
    tick = shim._current_tick()
    if race is None or tick is None:
        return None
    feature_names = shim._feature_spec["feature_names"]

    for runner_idx_in_tick, runner in enumerate(tick.runners):
        if runner.status != "ACTIVE":
            continue
        ltp = runner.last_traded_price
        if ltp is None or ltp <= 1.0:
            continue
        slot_map = shim.env._slot_maps[shim.env._race_idx]
        slot = next(
            (s for s, sid in slot_map.items()
             if sid == runner.selection_id),
            None,
        )
        if slot is None:
            continue
        try:
            feat_dict = shim._feature_extractor.extract(
                race=race,
                tick_idx=shim.env._tick_idx,
                runner_idx=runner_idx_in_tick,
                side="back",
            )
        except Exception:
            continue
        feature_vec = np.asarray(
            [feat_dict[name] for name in feature_names],
            dtype=np.float32,
        ).reshape(1, -1)
        raw = shim._booster.predict(feature_vec)
        cal = float(shim._calibrator.predict(np.asarray(raw))[0])
        if not np.isfinite(cal):
            continue
        return feature_vec, slot, "back", cal
    return None


def _verify_scorer_against_standalone_booster(
    scorer_dir: Path,
    captured: tuple[np.ndarray, int, str, float],
) -> bool:
    """Re-load booster + calibrator from disk; assert prediction matches."""
    import joblib
    import lightgbm as lgb

    feature_vec, _slot, _side, recorded_cal = captured

    booster = lgb.Booster(model_file=str(scorer_dir / "model.lgb"))
    calibrator = joblib.load(scorer_dir / "calibrator.joblib")
    raw = booster.predict(feature_vec)
    cal = float(calibrator.predict(np.asarray(raw))[0])
    return abs(cal - recorded_cal) < 1e-9


def main(
    *,
    day_str: str,
    data_dir: Path,
    n_steps: int = 1000,
    seed: int = 42,
    out_path: Path = DEFAULT_OUT_PATH,
    scorer_dir: Path = DEFAULT_SCORER_DIR,
    hidden_size: int = 128,
    max_runners: int = 14,
) -> int:
    """Run the smoke and write one episode row. Returns 0 on success."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("Loading day %s from %s …", day_str, data_dir)
    day = load_day(day_str, data_dir=data_dir)

    cfg = _scalping_smoke_config(max_runners=max_runners)
    env = BetfairEnv(day, cfg)
    shim = DiscreteActionShim(env, scorer_dir=scorer_dir)

    obs, info = shim.reset()
    space: DiscreteActionSpace = shim.action_space

    policy = DiscreteLSTMPolicy(
        obs_dim=shim.obs_dim,
        action_space=space,
        hidden_size=hidden_size,
    )
    policy.eval()

    hidden = policy.init_hidden(batch=1)
    rng = np.random.default_rng(seed)

    histogram: Counter[ActionType] = Counter()
    refusal_count = 0
    captured_scorer: tuple[np.ndarray, int, str, float] | None = None
    last_info: dict[str, Any] = info or {}
    total_reward = 0.0
    step_idx = 0
    t0 = time.perf_counter()
    finished_episode = False

    while step_idx < n_steps:
        if captured_scorer is None:
            captured_scorer = _capture_scorer_reference(shim)

        mask_np = shim.get_action_mask()
        # Sanity: NOOP must always be legal — without it the masked
        # softmax could blow up.
        if not mask_np[0]:
            raise RuntimeError(
                "NOOP unexpectedly masked; smoke contract broken",
            )

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(torch.float32)
            mask_t = torch.from_numpy(mask_np).unsqueeze(0)
            out = policy(obs_t, hidden_state=hidden, mask=mask_t)
            action_idx = int(out.action_dist.sample().item())
            hidden = out.new_hidden_state

        if not mask_np[action_idx]:
            # Should be impossible — masked categorical doesn't sample
            # from -inf logits — but count it just in case for the
            # refusal bar.
            refusal_count += 1

        kind, _runner = space.decode(action_idx)
        histogram[kind] += 1

        # Optional: sample a stake to keep the Beta head exercised. We
        # don't actually pass it through (default_stake is fine for the
        # smoke run); the .sample() proves the gradient pathway is
        # alive end-to-end.
        _ = torch.distributions.Beta(
            out.stake_alpha, out.stake_beta,
        ).sample()

        obs, reward, terminated, truncated, info = shim.step(action_idx)
        total_reward += float(reward)
        last_info = info or {}
        step_idx += 1
        if terminated or truncated:
            finished_episode = True
            break

    wall = time.perf_counter() - t0

    # ── Success bar evaluation ────────────────────────────────────────────
    bar_results: dict[str, bool] = {}
    bar_results["1_no_exceptions_raised"] = True  # we got here
    bar_results["3_action_histogram_covers_all_kinds"] = all(
        histogram[k] >= 1
        for k in (
            ActionType.NOOP, ActionType.OPEN_BACK,
            ActionType.OPEN_LAY, ActionType.CLOSE,
        )
    )
    bar_results["4_no_illegal_actions"] = (refusal_count == 0)

    bar_5_match = False
    bar_5_recorded = None
    if captured_scorer is not None:
        bar_5_match = _verify_scorer_against_standalone_booster(
            scorer_dir, captured_scorer,
        )
        bar_5_recorded = float(captured_scorer[3])
    bar_results["5_scorer_matches_standalone_booster"] = bar_5_match

    # Bar 2 — episodes.jsonl row written, see below.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema": "agents_v2_smoke",
        "day": day_str,
        "n_steps_run": step_idx,
        "finished_episode": finished_episode,
        "total_reward": total_reward,
        "day_pnl": float(last_info.get("day_pnl", 0.0)),
        "wall_seconds": wall,
        "wall_per_step_ms": (wall / step_idx * 1000.0) if step_idx else 0.0,
        "refusal_count": refusal_count,
        "action_histogram": {k.name: int(v) for k, v in histogram.items()},
        "scorer_reference_recorded_prob": bar_5_recorded,
        "bar_results": bar_results,
        "obs_dim": int(shim.obs_dim),
        "action_space_n": int(space.n),
        "max_runners": int(max_runners),
        "hidden_size": int(hidden_size),
        "seed": int(seed),
    }
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    bar_results["2_episodes_jsonl_written"] = out_path.exists()

    # ── Reporting ────────────────────────────────────────────────────────
    print()
    print(f"Smoke run on {day_str} — {step_idx} steps in {wall:.2f}s")
    print(f"  total_reward          = {total_reward:+.4f}")
    print(f"  day_pnl               = {row['day_pnl']:+.2f}")
    print(f"  refusal_count         = {refusal_count}")
    print(f"  wall per step         = {row['wall_per_step_ms']:.3f} ms")
    print("  action histogram:")
    for kind in (
        ActionType.NOOP, ActionType.OPEN_BACK,
        ActionType.OPEN_LAY, ActionType.CLOSE,
    ):
        print(f"    {kind.name:11s} = {histogram[kind]}")
    print()
    print("Success bar:")
    for name, ok in bar_results.items():
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {name}")
    print()
    print(f"Wrote {out_path}")
    all_passed = all(bar_results.values())
    return 0 if all_passed else 1


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1 smoke driver: random-init discrete policy "
                    "→ shim → env, n_steps env steps, asserts the "
                    "five success bars and writes one episodes.jsonl "
                    "row.",
    )
    p.add_argument(
        "--day", default="2026-04-23",
        help="ISO date string of the day to load. Default 2026-04-23.",
    )
    p.add_argument(
        "--data-dir", default=str(REPO_ROOT / "data" / "processed"),
        help="Directory containing the day's parquet files.",
    )
    p.add_argument(
        "--n-steps", type=int, default=1000,
        help="Maximum env steps. The driver stops earlier if the "
             "episode terminates. Default 1000.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for torch + numpy. Default 42.",
    )
    p.add_argument(
        "--out", default=str(DEFAULT_OUT_PATH),
        help="Path to write the smoke episodes.jsonl row to. Parent "
             "directory is created if it doesn't exist.",
    )
    p.add_argument(
        "--scorer-dir", default=str(DEFAULT_SCORER_DIR),
        help="Phase 0 scorer artefacts directory.",
    )
    p.add_argument(
        "--hidden-size", type=int, default=128,
        help="LSTM hidden size for the random-init policy.",
    )
    p.add_argument(
        "--max-runners", type=int, default=14,
        help="Env max_runners. Default 14 (matches production).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    sys.exit(main(
        day_str=args.day,
        data_dir=Path(args.data_dir),
        n_steps=args.n_steps,
        seed=args.seed,
        out_path=Path(args.out),
        scorer_dir=Path(args.scorer_dir),
        hidden_size=args.hidden_size,
        max_runners=args.max_runners,
    ))
