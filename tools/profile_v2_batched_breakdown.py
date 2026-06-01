"""Step-0 per-phase wall profiler for the REAL --batched cohort path.

training-speedup-v2, Step 0. Supersedes the Phase-3 lean-obs cProfile
numbers (which used a different config and cProfile's ~50%-inflated
per-step wall). This one:

* Runs the ACTUAL ``BatchedRolloutCollector`` (the code the cohort runs)
  on N arch-identical agents over one real training day.
* Builds envs EXACTLY as ``train_cluster_batched`` does (so the profiled
  config is byte-for-byte the production batched path — which, as Step 0
  verified, runs predictors OFF / no feature_cache / no input_norm).
* Times each per-tick sub-phase with monkeypatched wall accumulators
  (NOT cProfile — faithful absolute numbers), plus env-build and the
  per-agent PPO update.

Phases (per-tick, inside the rollout):
  - policy_forward     : DiscreteLSTMPolicy.forward            (Step 3A lever)
  - scorer_obs         : DiscreteActionShim.compute_extended_obs(Step 2 prime)
  - env_step_total     : BetfairEnv.step                       (Step 3C parent)
      sub: base_obs    : BetfairEnv._get_obs
      sub: matching    : BetfairEnv._process_action
      sub: settle      : BetfairEnv._settle_current_race
      sub: get_info    : BetfairEnv._get_info
  - attribution        : BatchedRolloutCollector._attribute_step_reward
  - collector_other    : rollout_wall - the above tops (obs/mask copy,
                         sampling, log_prob, RNG save/restore, sidecars)

Reconciliation target: cluster-day wall vs the operator's measured
~867s/agent-train-day (which is the per-day wall of a ~11-agent batched
cluster, averaged over the cohort's training days — see EXPERIMENTS.md
2026-06-01 c1 entry).

Usage:
  python -m tools.profile_v2_batched_breakdown --n-agents 2 --day 2026-05-09 \
      --hidden 128 --device cuda --predictors off
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from agents_v2.discrete_policy import DiscreteLSTMPolicy
from agents_v2.env_shim import DEFAULT_SCORER_DIR, DiscreteActionShim
from env.betfair_env import BetfairEnv
from training_v2.cohort.worker import _build_env_for_day, scalping_train_config
from training_v2.discrete_ppo.batched_rollout import BatchedRolloutCollector
from training_v2.discrete_ppo.trainer import DiscretePPOTrainer


REPO_ROOT = Path(__file__).resolve().parents[1]

CHAMP = REPO_ROOT.parent / "betfair-predictors" / "production" / "race-outcome" / "manifest.json"
RANK = REPO_ROOT.parent / "betfair-predictors" / "production" / "race-outcome-ranker" / "manifest.json"
DIRM = REPO_ROOT.parent / "betfair-predictors" / "production" / "direction-predictor" / "manifest.json"

# c2 reward pins (launch_c2_stable.sh)
REWARD_OVERRIDES = {
    "per_pair_reward_at_resolution": True,
    "locked_pnl_reward_weight": 9.0,
}

# ── timing accumulators ──────────────────────────────────────────────────
ACC: dict[str, list] = defaultdict(lambda: [0.0, 0])  # name -> [total_s, count]


def _wrap(obj, name, label, *, sync_cuda=False):
    """Monkeypatch ``obj.name`` to accumulate wall time under ``label``."""
    orig = getattr(obj, name)

    def timed(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            ACC[label][0] += dt
            ACC[label][1] += 1

    setattr(obj, name, timed)
    return orig


def build_bundle():
    from predictors import PredictorBundle
    return PredictorBundle.from_manifests(
        champion_manifest=CHAMP, ranker_manifest=RANK, direction_manifest=DIRM,
    )


def build_env(day, data_dir, predictors, bundle):
    cfg = scalping_train_config()
    cfg["training"]["strategy_mode"] = "arb"
    if predictors == "on":
        cfg.setdefault("observations", {})
        cfg["observations"]["use_race_outcome_predictor"] = True
        cfg["observations"]["use_direction_predictor"] = True
        return _build_env_for_day(
            day_str=day, data_dir=data_dir, cfg=cfg, scorer_dir=DEFAULT_SCORER_DIR,
            reward_overrides=REWARD_OVERRIDES, scalping_overrides=None,
            predictor_bundle=bundle,
            use_race_outcome_predictor=True, use_direction_predictor=True,
            predictor_lean_obs=False,
            predictor_p_win_back_threshold=0.20,
            predictor_p_win_lay_threshold=0.40,
        )
    # predictors OFF — exactly what train_cluster_batched passes.
    return _build_env_for_day(
        day_str=day, data_dir=data_dir, cfg=cfg, scorer_dir=DEFAULT_SCORER_DIR,
        reward_overrides=REWARD_OVERRIDES, scalping_overrides=None,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", default="2026-05-09")
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "processed"))
    ap.add_argument("--n-agents", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--predictors", choices=["off", "on"], default="off")
    ap.add_argument("--update", action="store_true",
                    help="Also run the per-agent PPO update (adds ~minutes).")
    ap.add_argument("--decompose-other", action="store_true",
                    help="Split collector_other into RNG-juggle vs sampling.")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    data_dir = Path(args.data_dir)
    N = args.n_agents

    bundle = None
    if args.predictors == "on":
        print("[setup] loading predictor bundle ...", flush=True)
        bundle = build_bundle()

    # ── env build (timed per agent) ──────────────────────────────────────
    print(f"[setup] building {N} envs (predictors={args.predictors}) ...", flush=True)
    envs, shims = [], []
    env_build_walls = []
    for i in range(N):
        t0 = time.perf_counter()
        env_i, shim_i = build_env(args.day, data_dir, args.predictors, bundle)
        env_build_walls.append(time.perf_counter() - t0)
        envs.append(env_i)
        shims.append(shim_i)
    obs_dim = shims[0].obs_dim
    print(f"[setup] obs_dim={obs_dim} env_build mean={np.mean(env_build_walls):.1f}s "
          f"(each={[f'{w:.1f}' for w in env_build_walls]})", flush=True)

    # ── policies (faithful to batched path: NO input_norm) ───────────────
    policies = []
    for i in range(N):
        torch.manual_seed(1000 + i)
        p = DiscreteLSTMPolicy(
            obs_dim=obs_dim, action_space=shims[i].action_space,
            hidden_size=args.hidden,
        )
        p.to(args.device)
        policies.append(p)

    # ── install timers ───────────────────────────────────────────────────
    _wrap(DiscreteLSTMPolicy, "forward", "policy_forward", sync_cuda=True)
    _wrap(DiscreteActionShim, "compute_extended_obs", "scorer_obs")
    _wrap(BetfairEnv, "step", "env_step_total")
    _wrap(BetfairEnv, "_get_obs", "  env.base_obs")
    _wrap(BetfairEnv, "_process_action", "  env.matching")
    _wrap(BetfairEnv, "_settle_current_race", "  env.settle")
    _wrap(BetfairEnv, "_get_info", "  env.get_info")
    _wrap(BatchedRolloutCollector, "_attribute_step_reward", "attribution")

    if args.decompose_other:
        # Decompose collector_other: per-agent RNG save/restore vs
        # distribution sampling/log_prob. Both are batch=1 GPU work in
        # the current batched collector — directly the Step 3A lever.
        _wrap(torch, "get_rng_state", "[other] rng_cpu")
        _wrap(torch, "set_rng_state", "[other] rng_cpu")
        if torch.cuda.is_available():
            _wrap(torch.cuda, "get_rng_state", "[other] rng_cuda", sync_cuda=True)
            _wrap(torch.cuda, "set_rng_state", "[other] rng_cuda", sync_cuda=True)
        _wrap(torch.distributions.Categorical, "sample", "[other] sampling", sync_cuda=True)
        _wrap(torch.distributions.Categorical, "log_prob", "[other] sampling", sync_cuda=True)
        _wrap(torch.distributions.Beta, "sample", "[other] sampling", sync_cuda=True)
        _wrap(torch.distributions.Beta, "log_prob", "[other] sampling", sync_cuda=True)

    seeds = [(424242 ^ (i + 1) * 0x9E3779B9) & 0x7FFFFFFF for i in range(N)]
    collector = BatchedRolloutCollector(
        shims=shims, policies=policies, device=args.device, seeds=seeds,
    )

    print(f"[rollout] collecting N={N} agents on {args.day} ...", flush=True)
    t0 = time.perf_counter()
    transitions = collector.collect_episode_batch()
    rollout_wall = time.perf_counter() - t0
    n_steps = sum(len(t) for t in transitions)
    print(f"[rollout] done: rollout_wall={rollout_wall:.1f}s "
          f"total_agent_ticks={n_steps}", flush=True)

    # ── optional PPO update (per agent) ──────────────────────────────────
    update_walls = []
    if args.update:
        for i in range(N):
            tr = DiscretePPOTrainer(
                policy=policies[i], shim=shims[i],
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, entropy_coeff=0.01, value_coeff=0.5,
                ppo_epochs=4, mini_batch_size=64, device=args.device,
            )
            t0 = time.perf_counter()
            tr.update_from_rollout(
                transitions=transitions[i], last_info=collector.last_infos[i],
            )
            update_walls.append(time.perf_counter() - t0)
        print(f"[update] per-agent PPO update mean={np.mean(update_walls):.1f}s", flush=True)

    # ── report ───────────────────────────────────────────────────────────
    ticks_per_agent = n_steps / N
    print("\n" + "=" * 72)
    print(f"STEP-0 BREAKDOWN  day={args.day} N={N} hidden={args.hidden} "
          f"device={args.device} predictors={args.predictors}")
    print(f"obs_dim={obs_dim}  ticks/agent={ticks_per_agent:.0f}  "
          f"agent-ticks={n_steps}")
    print("=" * 72)
    print(f"{'phase':<26} {'total_s':>9} {'calls':>9} "
          f"{'us/call':>9} {'% rollout':>9}")
    order = [
        "policy_forward", "scorer_obs", "env_step_total",
        "  env.base_obs", "  env.matching", "  env.settle", "  env.get_info",
        "attribution",
    ]
    top_level = {"policy_forward", "scorer_obs", "env_step_total", "attribution"}
    summed_top = 0.0
    for name in order:
        tot, cnt = ACC[name]
        us = (tot / cnt * 1e6) if cnt else 0.0
        pct = tot / rollout_wall * 100.0 if rollout_wall else 0.0
        print(f"{name:<26} {tot:>9.1f} {cnt:>9d} {us:>9.1f} {pct:>8.1f}%")
        if name in top_level:
            summed_top += tot
    other = rollout_wall - summed_top
    print(f"{'collector_other':<26} {other:>9.1f} {'':>9} {'':>9} "
          f"{other/rollout_wall*100.0:>8.1f}%")
    for name in sorted(k for k in ACC if k.startswith("[other]")):
        tot, cnt = ACC[name]
        us = (tot / cnt * 1e6) if cnt else 0.0
        pct = tot / rollout_wall * 100.0 if rollout_wall else 0.0
        print(f"{name:<26} {tot:>9.1f} {cnt:>9d} {us:>9.1f} {pct:>8.1f}%")
    print("-" * 72)
    print(f"{'rollout_wall (N agents)':<26} {rollout_wall:>9.1f}")
    print(f"{'rollout per agent':<26} {rollout_wall/N:>9.1f}")
    print(f"{'env_build per agent':<26} {np.mean(env_build_walls):>9.1f}")
    if update_walls:
        print(f"{'ppo_update per agent':<26} {np.mean(update_walls):>9.1f}")

    # cluster-day reconciliation (11-agent cluster, this day's tick count)
    per_agent_rollout = rollout_wall / N
    per_agent_build = float(np.mean(env_build_walls))
    per_agent_update = float(np.mean(update_walls)) if update_walls else 0.0
    for ncl in (N, 11):
        cluster_day = ncl * (per_agent_build + per_agent_rollout + per_agent_update)
        print(f"[reconcile] est cluster-day wall @ N={ncl}: {cluster_day:.0f}s "
              f"(build {ncl*per_agent_build:.0f} + rollout {ncl*per_agent_rollout:.0f} "
              f"+ update {ncl*per_agent_update:.0f})")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
