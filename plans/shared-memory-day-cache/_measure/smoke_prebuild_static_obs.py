"""Production-path smoke for the static_obs writer (Step 1).

The golden test exercises the env consume-path via the test helper
``golden_cases.build_env``. This smoke exercises the PRODUCTION writer
``prebuild_static_obs_cache`` → ``_build_env_for_day(static_obs_cache=)`` —
the exact path the cohort runner uses — and asserts:

  1. The writer bakes PREDICTORS (champion_p_win column non-zero) — a silent
     predictor-OFF prebuild would be an HC#5 feature drop the golden test
     (which builds its own predictors-ON env) cannot catch.
  2. The artifact's static_obs == a from-scratch predictors-ON worker env,
     bit-for-bit (writer matches worker).
  3. Gate caches (_race_p_win_by_race, _tick_drift_fires_by_race) match.

Run: python plans/shared-memory-day-cache/_measure/smoke_prebuild_static_obs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DATA_DIR = REPO / "data" / "processed"
PRED = REPO.parent / "betfair-predictors" / "production"
MANIFESTS = (
    str(PRED / "race-outcome" / "manifest.json"),
    str(PRED / "race-outcome-ranker" / "manifest.json"),
    str(PRED / "direction-predictor" / "manifest.json"),
)
DAY = "2026-04-15"


def main() -> None:
    from env.betfair_env import MARKET_DIM, RUNNER_KEYS, VELOCITY_DIM
    from predictors import PredictorBundle
    from training_v2.cohort.multiproc_worker import prebuild_static_obs_cache
    from training_v2.cohort.static_obs_cache import DayStaticObs
    from training_v2.cohort.worker import (
        _build_env_for_day,
        scalping_train_config,
    )

    bundle = PredictorBundle.from_manifests(
        champion_manifest=MANIFESTS[0],
        ranker_manifest=MANIFESTS[1],
        direction_manifest=MANIFESTS[2],
    )
    cache_dir = Path(REPO) / "plans" / "shared-memory-day-cache" / "_measure" / "_smoke_cache"

    print("[1/3] prebuild_static_obs_cache (production writer, predictors-ON)…")
    paths = prebuild_static_obs_cache(
        [DAY], data_dir=DATA_DIR, cache_dir=cache_dir,
        predictor_bundle=bundle,
        use_race_outcome_predictor=True,
        use_direction_predictor=True,
        predictor_lean_obs=False,
    )
    npy, side = paths[DAY]
    artifact = DayStaticObs.load(npy, side, mmap=True)
    print(f"      artifact: {artifact.static_obs_flat.shape} "
          f"{artifact.static_obs_flat.dtype}, obs_dim={artifact.obs_dim}")

    # (1) predictors baked? champion_p_win column for runner slot 0.
    cp_idx = MARKET_DIM + VELOCITY_DIM + RUNNER_KEYS.index("champion_p_win")
    cp_col = np.asarray(artifact.static_obs_flat[:, cp_idx])
    n_nonzero = int((cp_col != 0.0).sum())
    print(f"      champion_p_win[slot0] col idx={cp_idx}: "
          f"{n_nonzero}/{cp_col.size} ticks non-zero, max={cp_col.max():.4f}")
    assert n_nonzero > 0, (
        "PREDICTORS NOT BAKED — champion_p_win column is all zero. "
        "HC#5 silent feature drop in prebuild_static_obs_cache."
    )

    print("[2/3] from-scratch predictors-ON worker env (_build_env_for_day)…")
    cfg = scalping_train_config()
    cfg.setdefault("observations", {})["use_race_outcome_predictor"] = True
    cfg["observations"]["use_direction_predictor"] = True
    env_scratch, _ = _build_env_for_day(
        day_str=DAY, data_dir=DATA_DIR, cfg=cfg,
        scorer_dir=__import__(
            "training_v2.cohort.worker", fromlist=["DEFAULT_SCORER_DIR"],
        ).DEFAULT_SCORER_DIR,
        predictor_bundle=bundle,
        predictor_lean_obs=False,
    )

    print("[3/3] compare artifact (writer) vs from-scratch worker env…")
    so_scratch = env_scratch._static_obs
    views = artifact.race_views()
    assert len(views) == len(so_scratch), (len(views), len(so_scratch))
    n_ticks = 0
    for r, (vr, sr) in enumerate(zip(views, so_scratch)):
        assert len(sr) == vr.shape[0], (r, len(sr), vr.shape)
        for t, arr in enumerate(sr):
            if not np.array_equal(np.asarray(vr[t]), arr):
                raise AssertionError(
                    f"static_obs MISMATCH at race {r} tick {t}: "
                    f"max|Δ|={np.abs(np.asarray(vr[t]) - arr).max()}"
                )
            n_ticks += 1
    print(f"      static_obs bit-identical across {n_ticks} ticks ✓")

    # gate caches
    assert artifact.race_p_win_by_race == env_scratch._race_p_win_by_race, (
        "race_p_win_by_race mismatch"
    )
    assert (artifact.tick_drift_fires_by_race
            == env_scratch._tick_drift_fires_by_race), (
        "tick_drift_fires_by_race mismatch"
    )
    print("      gate caches bit-identical ✓")
    print("\n=== SMOKE PASS — production writer bakes predictors & matches "
          "from-scratch worker env ===")


if __name__ == "__main__":
    main()
