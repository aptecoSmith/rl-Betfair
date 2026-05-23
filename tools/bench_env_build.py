"""Break down env build's wall and verify feature_cache benefit.

phase_3 Option F.1. The per-day env build takes ~16-18s in the cohort
(steady-state). Cohort runs N agents per gen through the SAME date
set so any pure (date) work is cacheable.

This measures:
  1. cold env build (no cache)
  2. warm env build with feature_cache pre-populated (existing
     mechanism — engineer_day only)
  3. delta = engineer_day cost
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

from agents_v2.env_shim import DEFAULT_SCORER_DIR, DiscreteActionShim
from data.episode_builder import load_day
from env.betfair_env import BetfairEnv
from training_v2.cohort.worker import scalping_train_config
from tools.profile_v2_full_agent import build_predictor_bundle


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", default="2026-04-16")
    ap.add_argument("--data-dir", default="data/processed", type=Path)
    args = ap.parse_args()

    cfg = scalping_train_config()
    cfg["training"]["strategy_mode"] = "arb"
    cfg.setdefault("observations", {})
    cfg["observations"]["use_race_outcome_predictor"] = True
    cfg["observations"]["use_direction_predictor"] = True

    print("[bench] loading predictor bundle (one-time worker cost) ...", flush=True)
    t0 = time.perf_counter()
    bundle = build_predictor_bundle()
    print(f"[bench] bundle loaded in {time.perf_counter()-t0:.2f}s\n", flush=True)

    env_kwargs = dict(
        reward_overrides={
            "force_close_before_off_seconds": 120.0,
            "close_feasibility_max_spread_pct": 0.05,
        },
        predictor_bundle=bundle,
        use_race_outcome_predictor=True,
        use_direction_predictor=True,
        predictor_lean_obs=True,
        predictor_p_win_back_threshold=0.20,
        predictor_p_win_lay_threshold=0.40,
        race_confidence_threshold=0.50,
        lay_price_max=20.0,
        emit_debug_features=False,
    )

    # Warm-up build to mute first-time import/JIT effects.
    print("[bench] warm-up build (discarded) ...", flush=True)
    day_wu = load_day(args.day, data_dir=args.data_dir)
    BetfairEnv(day_wu, cfg, **env_kwargs)

    # === Cold build (no cache) ===
    print("\n[bench] COLD build (no feature_cache) ...", flush=True)
    t0 = time.perf_counter()
    day = load_day(args.day, data_dir=args.data_dir)
    t_load = time.perf_counter() - t0

    feature_cache: dict[str, list] = {}
    t0 = time.perf_counter()
    env = BetfairEnv(day, cfg, feature_cache=feature_cache, **env_kwargs)
    t_env_cold = time.perf_counter() - t0
    shim = DiscreteActionShim(env, scorer_dir=DEFAULT_SCORER_DIR)
    print(
        f"  load_day               {t_load:.2f}s\n"
        f"  BetfairEnv ctor (cold) {t_env_cold:.2f}s\n"
        f"  -> feature_cache populated: {bool(feature_cache)}",
        flush=True,
    )

    # === Warm build with feature_cache ===
    print("\n[bench] WARM build (feature_cache hit) ...", flush=True)
    t0 = time.perf_counter()
    day2 = load_day(args.day, data_dir=args.data_dir)
    t_load2 = time.perf_counter() - t0

    t0 = time.perf_counter()
    env2 = BetfairEnv(day2, cfg, feature_cache=feature_cache, **env_kwargs)
    t_env_warm = time.perf_counter() - t0
    print(
        f"  load_day               {t_load2:.2f}s\n"
        f"  BetfairEnv ctor (warm) {t_env_warm:.2f}s\n"
        f"  delta env saved by cache   {t_env_cold - t_env_warm:.2f}s "
        f"({(1 - t_env_warm/t_env_cold)*100:.0f}% reduction)",
        flush=True,
    )

    # === Now share Day across builds too ===
    print("\n[bench] WARM build (feature_cache hit + Day reused) ...", flush=True)
    t0 = time.perf_counter()
    env3 = BetfairEnv(day, cfg, feature_cache=feature_cache, **env_kwargs)
    t_env_reused = time.perf_counter() - t0
    print(
        f"  BetfairEnv ctor (Day+cache) {t_env_reused:.2f}s\n"
        f"  delta vs cold                   {t_env_cold - t_env_reused:.2f}s",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
