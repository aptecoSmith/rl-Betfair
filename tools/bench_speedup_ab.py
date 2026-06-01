"""A/B timing: baseline vs Step-2/3A optimised batched cluster-day wall.

training-speedup-v2. Measures the two big bit-identical wins on the real
day, predictors-OFF (matching the 867s as-ran baseline), then reports the
extrapolated cluster-day wall before vs after:

  BASELINE  = CUDA rollout + NO feature_cache  (the c1/c2 path)
  OPTIMISED = CPU  rollout + feature_cache       (this branch)

(extract_array is in BOTH rollout measurements since it's already
committed; its separate ~2-3 s/agent-day win is on top of the baseline
that used the dict path — measured in bench_extract_array.)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from agents_v2.discrete_policy import DiscreteLSTMPolicy
from training_v2.cohort.worker import _build_env_for_day, scalping_train_config
from training_v2.discrete_ppo.batched_rollout import BatchedRolloutCollector


REPO_ROOT = Path(__file__).resolve().parents[1]
REWARD_OVERRIDES = {"per_pair_reward_at_resolution": True,
                    "locked_pnl_reward_weight": 9.0}


def build_env(day, data_dir, cache):
    cfg = scalping_train_config()
    cfg["training"]["strategy_mode"] = "arb"
    return _build_env_for_day(
        day_str=day, data_dir=data_dir, cfg=cfg,
        scorer_dir=__import__("agents_v2.env_shim", fromlist=["DEFAULT_SCORER_DIR"]).DEFAULT_SCORER_DIR,
        reward_overrides=REWARD_OVERRIDES, scalping_overrides=None,
        feature_cache=cache,
    )


def time_build(day, data_dir, n, *, shared_cache):
    walls = []
    cache: dict = {} if shared_cache else None
    for i in range(n):
        c = cache if shared_cache else {}  # fresh dict each → never hits
        t0 = time.perf_counter()
        build_env(day, data_dir, c)
        walls.append(time.perf_counter() - t0)
    return walls


def time_rollout(day, data_dir, n, device, hidden):
    envs, shims = [], []
    for _ in range(n):
        _, shim = build_env(day, data_dir, {})  # independent build (untimed)
        shims.append(shim); envs.append(shim.env)
    policies = []
    for i in range(n):
        torch.manual_seed(7 + i)
        p = DiscreteLSTMPolicy(obs_dim=shims[i].obs_dim,
                               action_space=shims[i].action_space,
                               hidden_size=hidden)
        p.to(device); policies.append(p)
    seeds = [(99 ^ (i + 1) * 0x9E3779B9) & 0x7FFFFFFF for i in range(n)]
    coll = BatchedRolloutCollector(shims=shims, policies=policies,
                                   device=device, seeds=seeds)
    t0 = time.perf_counter()
    tr = coll.collect_episode_batch()
    wall = time.perf_counter() - t0
    steps = sum(len(t) for t in tr)
    return wall, steps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", default="2026-05-09")
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "processed"))
    ap.add_argument("--n-build", type=int, default=3)
    ap.add_argument("--n-rollout", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--cluster", type=int, default=11,
                    help="cluster size to extrapolate the cluster-day wall to")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    cuda = torch.cuda.is_available()

    print(f"=== env-build A/B (n={args.n_build}, predictors=off) ===", flush=True)
    nocache = time_build(args.day, data_dir, args.n_build, shared_cache=False)
    cached = time_build(args.day, data_dir, args.n_build, shared_cache=True)
    print(f"  no-cache builds : {[f'{w:.1f}' for w in nocache]}  "
          f"mean={np.mean(nocache):.1f}s")
    print(f"  cached  builds  : {[f'{w:.1f}' for w in cached]}  "
          f"(first miss, rest hit)")

    print(f"\n=== rollout A/B (n={args.n_rollout}, predictors=off) ===", flush=True)
    cpu_wall, steps = time_rollout(args.day, data_dir, args.n_rollout, "cpu", args.hidden)
    print(f"  CPU  rollout: {cpu_wall:.1f}s  ({cpu_wall/args.n_rollout:.1f}s/agent, "
          f"{steps} agent-ticks)")
    if cuda:
        cuda_wall, _ = time_rollout(args.day, data_dir, args.n_rollout, "cuda", args.hidden)
        print(f"  CUDA rollout: {cuda_wall:.1f}s  ({cuda_wall/args.n_rollout:.1f}s/agent)")
    else:
        cuda_wall = cpu_wall
        print("  CUDA unavailable — using CPU number for both (no rollout delta).")

    # ── extrapolated cluster-day wall ────────────────────────────────────
    C = args.cluster
    build_nocache_per = float(np.mean(nocache))
    # cache: first agent pays full build; the rest pay the cached-build cost.
    build_cache_total = cached[0] + sum(cached[1:]) if len(cached) > 1 else cached[0]
    build_cache_per_after = float(np.mean(cached[1:])) if len(cached) > 1 else cached[0]
    cpu_per = cpu_wall / args.n_rollout
    cuda_per = cuda_wall / args.n_rollout
    update_per = 6.0  # measured earlier (~5.5s/agent PPO update)

    baseline = C * build_nocache_per + C * cuda_per + C * update_per
    optimised = (cached[0] + (C - 1) * build_cache_per_after) + C * cpu_per + C * update_per

    print(f"\n=== extrapolated CLUSTER-DAY wall @ cluster={C} (predictors-off) ===")
    print(f"  BASELINE  (CUDA rollout, no cache): "
          f"build {C*build_nocache_per:.0f} + rollout {C*cuda_per:.0f} "
          f"+ update {C*update_per:.0f} = {baseline:.0f}s")
    print(f"  OPTIMISED (CPU rollout, feature_cache): "
          f"build {cached[0] + (C-1)*build_cache_per_after:.0f} + rollout {C*cpu_per:.0f} "
          f"+ update {C*update_per:.0f} = {optimised:.0f}s")
    if baseline > 0:
        print(f"  --> {(1 - optimised/baseline)*100:.1f}% faster "
              f"({baseline-optimised:.0f}s/cluster-day saved); "
              f"{baseline/optimised:.2f}x speedup")
    print(f"  (c1 log measured baseline for this day: ~1101s — sanity check)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
