"""Calibrate the optimal --parallel-agents worker count for this machine.

The multiprocess cohort path (training-speedup-v2 R5) is memory-bandwidth-
bound: more worker processes finish more agents per wave, but each runs
slower under contention, so THROUGHPUT (agents/sec) rises, peaks, then
plateaus/falls well below the core count. This sweep measures
throughput = K / wall for several worker counts K (each worker trains one
agent over a cached day, so it's pure training-under-contention, not
feature engineering) and prints the peak — pin ``--parallel-agents`` to it.

Run once per machine (re-run if the hardware changes). Example:

    python -m tools.measure_optimal_n --ks 4,8,12,16,20 \
        --train-day 2026-05-09 --eval-day 2026-05-13

Uses NON-held-out days by default. The numbers are wall-clock only — the
bit-identity of the multiprocess path is gated separately by
tests/test_v2_multiproc_cluster.py + the R5 parity probes.
"""
from __future__ import annotations

import os

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import time
from pathlib import Path


def _spec(i: int, train_day: str, eval_day: str, data_dir: Path,
          *, day_paths: dict | None = None,
          static_obs_paths: dict | None = None, manifests=None) -> dict:
    from training_v2.cohort.genes import CohortGenes
    base = dict(learning_rate=3e-4, entropy_coeff=0.01, clip_range=0.2,
                gae_lambda=0.95, value_coeff=0.5, mini_batch_size=64,
                hidden_size=128)
    spec = dict(
        agent_id=f"r{i}", genes=CohortGenes(**base),
        days_to_train=[train_day], eval_days=[eval_day],
        data_dir=data_dir, device="cpu", seed=42 + i,
        model_store=None, generation=0,
        reward_overrides={"per_pair_reward_at_resolution": True,
                          "locked_pnl_reward_weight": 9.0},
    )
    if static_obs_paths is not None:
        # predictors-ON via the SHARED static_obs path (the production path
        # post shared-memory-day-cache): predictors are BAKED into the shared
        # memmap, so workers do NO per-tick inference here — throughput
        # reflects pure training-under-contention on the real path. The
        # bundle is still rebuilt per worker (env ctor) from the manifests.
        spec["use_race_outcome_predictor"] = True
        spec["use_direction_predictor"] = True
        spec["_static_obs_day_paths"] = static_obs_paths
        spec["_predictor_manifests"] = tuple(manifests)
    else:
        # legacy dict-cache path (predictors-OFF, or predictors-ON pre-fix
        # where workers ran per-tick inference).
        spec["_feature_cache_day_paths"] = day_paths
        if manifests:
            spec["use_race_outcome_predictor"] = True
            spec["_predictor_manifests"] = tuple(manifests)
    return spec


def main(argv: list[str] | None = None) -> int:
    import torch
    torch.set_num_threads(1)
    from training_v2.cohort.multiproc_worker import (
        train_cluster_multiproc, prebuild_feature_cache,
        save_shared_cache_per_day, prebuild_static_obs_cache, make_pool,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ks", default="4,8,12,16,20",
                   help="comma-separated worker counts to sweep")
    p.add_argument("--train-day", default="2026-05-09")
    p.add_argument("--eval-day", default="2026-05-13")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--cache-dir", default="logs/optn_cache")
    p.add_argument("--warm", action="store_true",
                   help="warm each pool with a throwaway run before timing "
                        "(isolates steady-state training contention; ~2x "
                        "slower to run)")
    p.add_argument("--predictor-manifests", nargs=3, default=None,
                   metavar=("CHAMP", "RANK", "DIR"),
                   help="calibrate predictors-ON: pass champion, ranker, "
                        "direction manifest paths. Workers rebuild the bundle "
                        "from these; throughput then reflects per-tick "
                        "inference cost (differs from the predictors-OFF "
                        "curve). Tip: include --ks 1,16 to read off the "
                        "K=16-vs-K=1 multiprocess speedup factor directly.")
    args = p.parse_args(argv)

    ks = [int(x) for x in args.ks.split(",")]
    data_dir = Path(args.data_dir)
    days = [args.train_day, args.eval_day]
    manifests = args.predictor_manifests

    day_paths = static_obs_paths = None
    if manifests:
        # Production path post shared-memory-day-cache: bake the shared
        # static_obs memmap (predictors baked in) ONCE, workers consume it.
        from predictors import PredictorBundle
        bundle = PredictorBundle.from_manifests(
            champion_manifest=manifests[0], ranker_manifest=manifests[1],
            direction_manifest=manifests[2],
        )
        static_obs_paths = prebuild_static_obs_cache(
            days, data_dir=data_dir,
            cache_dir=Path(args.cache_dir) / "static_obs",
            predictor_bundle=bundle,
            use_race_outcome_predictor=True, use_direction_predictor=True,
        )
    else:
        cache = prebuild_feature_cache(days, data_dir=data_dir)
        day_paths = save_shared_cache_per_day(cache, Path(args.cache_dir), days)

    print(f"[optn] cores={os.cpu_count()}  Ks={ks}  warm={args.warm}  "
          f"predictors={'ON (shared static_obs)' if manifests else 'off'}",
          flush=True)
    rows = []
    for k in ks:
        pool = make_pool(k)
        try:
            specs = [_spec(i, args.train_day, args.eval_day, data_dir,
                           day_paths=day_paths,
                           static_obs_paths=static_obs_paths,
                           manifests=manifests) for i in range(k)]
            if args.warm:
                train_cluster_multiproc(specs, executor=pool)
            t0 = time.perf_counter()
            train_cluster_multiproc(specs, executor=pool)
            wall = time.perf_counter() - t0
        finally:
            pool.shutdown(wait=True)
        thru = k / wall
        rows.append((k, wall, thru))
        print(f"[optn] K={k:2d}  wall={wall:5.0f}s  throughput={thru:.3f} ag/s",
              flush=True)

    best = max(rows, key=lambda r: r[2])
    print(f"\n[optn] OPTIMAL N = {best[0]}  "
          f"(throughput {best[2]:.3f} ag/s, {best[1]:.0f}s/full wave)")
    print("[optn] pin --parallel-agents to N; the pool waves through any "
          "cohort size at that concurrency.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
