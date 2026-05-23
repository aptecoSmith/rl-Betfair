"""Regression guards for the V3/V4 direction-feature builder.

The env's :func:`data.predictor_features.build_direction_windows_for_race`
must produce values that match the predictor repo's training-time
``scripts/predictor/build_dataset.py`` builder COLUMN-FOR-COLUMN, else
the loaded predictor receives features at a different distribution from
what it was trained on and outputs garbage.

Two pre-existing V2 mismatches were caught and fixed alongside the V3/V4
port (2026-05-22):

1. V2 ``ltp_w32_*`` window stats: env was including the current tick;
   predictor training appends ltp to its history deque AFTER computing
   feat, so the window covers the PREVIOUS 32 ticks only.
2. V4 ``rank_in_market``: env used float32 arrays for searchsorted;
   training uses np.asarray default (float64). float32-truncated values
   can cause an off-by-one rank when the lookup equals a stored value
   exactly.

These tests compare env-built windows against the dataset shards in the
predictor repo (``../betfair-predictors/data/predictor_dataset/<date>.parquet``)
which were built by the canonical training-time builder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BETFAIR_PREDICTORS = Path(__file__).resolve().parents[2] / "betfair-predictors"
_DATASET_DIR = _BETFAIR_PREDICTORS / "data" / "predictor_dataset"
_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# The fixture day must have:
#   1. A shard in betfair-predictors/data/predictor_dataset/
#   2. The matching raw parquet in rl-betfair/data/processed/
# 2026-05-07 satisfies both (predictor dataset built for the V4 retrain).
FIXTURE_DAY = "2026-05-07"

pytestmark = pytest.mark.skipif(
    not (_DATASET_DIR / f"{FIXTURE_DAY}.parquet").exists()
    or not (_DATA_DIR / f"{FIXTURE_DAY}.parquet").exists(),
    reason=(
        "fixture parquets missing: requires "
        f"{_DATASET_DIR / (FIXTURE_DAY + '.parquet')} (predictor dataset) "
        f"and {_DATA_DIR / (FIXTURE_DAY + '.parquet')} (raw)."
    ),
)


def _load_race_and_shard():
    import pandas as pd
    from data.episode_builder import load_day
    day = load_day(FIXTURE_DAY, data_dir=_DATA_DIR)
    race = day.races[0]
    shard = pd.read_parquet(_DATASET_DIR / f"{FIXTURE_DAY}.parquet")
    shard = shard[shard["market_id"] == race.market_id].copy()
    return race, shard


def _compare_at(race, shard, target_t: int, target_sid: int, variant: str) -> tuple[int, list[tuple[str, float, float]]]:
    """Return (n_matching, list of (col_name, env_val, predictor_val) for diffs)."""
    import pandas as pd
    from data.predictor_features import (
        build_direction_windows_for_race, DIR_VARIANT_COLS,
    )

    w, idx = build_direction_windows_for_race(race, variant=variant)
    pos = idx.index((target_t, target_sid))
    env_features = w[pos, -1]

    sub = shard[shard["selection_id"] == target_sid].sort_values("tick_idx").reset_index(drop=True)
    row = sub.iloc[target_t]

    cols = DIR_VARIANT_COLS[variant]
    matching = 0
    diffs: list[tuple[str, float, float]] = []
    for col_i, col_name in enumerate(cols):
        e = float(env_features[col_i])
        p = (
            float(row[col_name])
            if (col_name in row.index and pd.notna(row[col_name]))
            else 0.0
        )
        tol = max(0.001 * max(abs(p), 1.0), 1e-4)
        if abs(e - p) < tol:
            matching += 1
        else:
            diffs.append((col_name, e, p))
    return matching, diffs


def test_v4_features_match_training_time_builder_first_runner():
    """All 39 V4 features match the predictor training-time builder for
    a representative (race, tick, runner)."""
    race, shard = _load_race_and_shard()
    target_t = len(race.ticks) // 2
    # First runner with a non-empty traded_volume_ladder at the target tick.
    target_sid = None
    for r in race.ticks[target_t].runners:
        if r.traded_volume_ladder:
            target_sid = r.selection_id
            break
    assert target_sid is not None, "no runner has TVL at the middle tick"

    matching, diffs = _compare_at(race, shard, target_t, target_sid, variant="V4")
    assert not diffs, f"feature divergence: {diffs}"
    assert matching == 39


def test_v4_features_match_at_tick_zero():
    """Edge case: tick 0 has no V2 window history but V3+V4 still
    well-defined."""
    race, shard = _load_race_and_shard()
    # First runner with TVL at tick 0
    target_sid = None
    for r in race.ticks[0].runners:
        if r.traded_volume_ladder:
            target_sid = r.selection_id
            break
    if target_sid is None:
        pytest.skip("no runner has TVL at tick 0 in fixture race")

    matching, diffs = _compare_at(race, shard, 0, target_sid, variant="V4")
    # V2 window stats should be all zero (no history) — matches predictor's
    # NaN→0 fill in our env output.
    assert not diffs, f"feature divergence at tick 0: {diffs}"
    assert matching == 39


def test_v2_window_stats_exclude_current_tick():
    """The V2 ``ltp_w32_*`` window stats cover the PREVIOUS 32 ticks
    exclusive — matches the predictor's deque append-after-compute order.
    Regression guard for the 2026-05-22 fix.
    """
    race, shard = _load_race_and_shard()
    # Find any priceable (tick, sid).
    target_t = 10  # need enough history for a non-empty window
    target_sid = race.ticks[target_t].runners[0].selection_id
    matching, diffs = _compare_at(race, shard, target_t, target_sid, variant="V2")
    # V2 window stats are the 6 cols ltp_w32_*. If any of those diverge,
    # the include-current-tick regression has come back.
    window_stat_diffs = [d for d in diffs if d[0].startswith("ltp_w32_")]
    assert not window_stat_diffs, (
        f"V2 window stats diverge from training-time builder: {window_stat_diffs}"
    )
