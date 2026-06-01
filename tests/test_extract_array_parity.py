"""Byte-equality guard for the Step-2 ``extract_array`` fast path.

deferred.md Option B-big risk: "feature ordering drift could silently
break the LightGBM scorer's expected input shape — load-bearing
byte-equality test is the mitigation." This is that test.

For a battery of (tick, runner, side) opportunities spanning early-race
(NaN velocity windows) and mid-race (populated windows), assert that

    extract_array(out_f32)  ==  np.asarray([extract()[name]
                                            for name in FEATURE_NAMES],
                                           dtype=np.float32)

byte-for-byte (NaN positions included). If this holds, the dict path and
the array path are interchangeable and the obs fed to the booster is
unchanged — which the golden harness then confirms end-to-end.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA = REPO_ROOT / "data" / "processed"

pytestmark = pytest.mark.skipif(
    not _DATA.exists(), reason="processed data not present",
)

from data.episode_builder import load_day  # noqa: E402
from training_v2.scorer.feature_extractor import (  # noqa: E402
    FEATURE_NAMES,
    FeatureExtractor,
)

_N = len(FEATURE_NAMES)


def _rekey_dict(d: dict) -> np.ndarray:
    """The exact re-key the OLD compute_extended_obs consumer did."""
    return np.asarray([d[name] for name in FEATURE_NAMES], dtype=np.float32)


def test_extract_array_byte_identical_to_dict_rekey():
    day = load_day("2026-05-09", data_dir=_DATA)
    races = day.races[:2]
    fx = FeatureExtractor()

    out = np.empty(_N, dtype=np.float32)
    compared = 0
    nan_seen = False
    populated_seen = False

    for race in races:
        # Walk ticks in order, updating rolling history; sample
        # opportunities every few ticks across the race so both
        # early-race (NaN windows) and mid-race (populated) are hit.
        for tick_idx, tick in enumerate(race.ticks):
            fx.update_history(race, tick)
            if tick_idx % 7 != 0:
                continue
            for runner_idx, runner in enumerate(tick.runners):
                ltp = runner.last_traded_price
                if ltp is None or ltp <= 1.0:
                    continue
                for side in ("back", "lay"):
                    d = fx.extract(
                        race=race, tick_idx=tick_idx,
                        runner_idx=runner_idx, side=side,
                    )
                    rekeyed = _rekey_dict(d)
                    fx.extract_array(
                        race=race, tick_idx=tick_idx,
                        runner_idx=runner_idx, side=side, out=out,
                    )
                    # Byte-identical, NaN positions included.
                    assert np.array_equal(out, rekeyed, equal_nan=True), (
                        f"mismatch at race={race.market_id} tick={tick_idx} "
                        f"runner={runner_idx} side={side}:\n"
                        f"  array={out}\n  rekey={rekeyed}"
                    )
                    compared += 1
                    if np.isnan(rekeyed).any():
                        nan_seen = True
                    else:
                        populated_seen = True

    assert compared > 50, f"battery too small ({compared} comparisons)"
    assert nan_seen, "expected at least one NaN-bearing feature vector"
    assert populated_seen, "expected at least one fully-finite feature vector"


def test_extract_array_writes_every_position():
    """A sentinel-poisoned buffer must be fully overwritten — proves
    _extract_into is exhaustive (no stale/garbage position survives)."""
    day = load_day("2026-05-09", data_dir=_DATA)
    race = day.races[0]
    fx = FeatureExtractor()
    for tick_idx, tick in enumerate(race.ticks[:40]):
        fx.update_history(race, tick)

    out = np.full(_N, 123456.0, dtype=np.float32)  # poison
    # find a priceable runner on tick 39
    tick = race.ticks[39]
    runner_idx = next(
        i for i, r in enumerate(tick.runners)
        if r.last_traded_price and r.last_traded_price > 1.0
    )
    fx.extract_array(race=race, tick_idx=39, runner_idx=runner_idx,
                     side="back", out=out)
    assert not np.any(out == 123456.0), "a position was left unwritten"
