"""Tests for the Phase 6 Session 03 closed-form ``_spread_in_ticks``.

The load-bearing test is ``test_closed_form_matches_walk_on_10k_random_pairs``
which generates 10 000 random ``(best_back, best_lay)`` pairs covering the
full ladder range and asserts byte-equality between the closed-form
implementation in ``training_v2/scorer/feature_extractor.py`` and a private
oracle kept inside this test file (the pre-S03 iterative walk). NaN-aware
equality: a pair is "equal" if both outputs are NaN OR both are equal floats.

Six smaller unit tests cover hand-constructed cases at boundaries
(zero spread, single-band, cross-band, cap, MIN/MAX clamping).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from env.tick_ladder import MAX_PRICE, MIN_PRICE, tick_offset
from training_v2.scorer.feature_extractor import _spread_in_ticks


def _spread_in_ticks_walk_oracle(best_back: float, best_lay: float) -> float:
    """Oracle: the pre-S03 iterative walk, kept private to this test file.

    Mirrors ``training_v2/scorer/feature_extractor._spread_in_ticks`` as
    it stood pre-S03 (commit ``1875936``): walks ``best_back`` upward by
    one tick at a time using ``env.tick_ladder.tick_offset`` until the
    walked price reaches or exceeds ``best_lay - 1e-9``.
    """
    if best_lay <= best_back:
        return 0.0
    for n in range(1, 50):
        p = tick_offset(best_back, n, +1)
        if p >= best_lay - 1e-9:
            return float(n)
    return math.nan


def test_closed_form_matches_walk_on_10k_random_pairs():
    """Bit-identity guard: closed form == walk on 10k random pairs."""
    rng = np.random.default_rng(42)
    backs = rng.uniform(MIN_PRICE, MAX_PRICE, size=10_000)
    spread_ticks = rng.integers(0, 60, size=10_000)
    lays = backs + spread_ticks * rng.uniform(0.005, 0.5, size=10_000)
    lays = np.clip(lays, MIN_PRICE, MAX_PRICE)

    oracle = np.asarray([
        _spread_in_ticks_walk_oracle(b, l) for b, l in zip(backs, lays)
    ])
    closed = np.asarray([
        _spread_in_ticks(b, l) for b, l in zip(backs, lays)
    ])

    # NaN-aware byte equality: a pair is OK if both NaN OR both equal floats.
    both_nan = np.isnan(oracle) & np.isnan(closed)
    both_eq = (oracle == closed) & ~np.isnan(oracle) & ~np.isnan(closed)
    ok = both_nan | both_eq
    if not ok.all():
        bad = np.where(~ok)[0]
        first = int(bad[0])
        raise AssertionError(
            f"closed-form diverges from walk on {len(bad)}/10000 pairs; "
            f"first divergent pair (back={backs[first]!r}, "
            f"lay={lays[first]!r}): oracle={oracle[first]!r} "
            f"closed={closed[first]!r}"
        )


def test_zero_spread_returns_zero():
    """best_back >= best_lay → 0.0."""
    assert _spread_in_ticks(5.0, 5.0) == 0.0
    assert _spread_in_ticks(5.0, 4.0) == 0.0
    assert _spread_in_ticks(2.10, 2.10) == 0.0
    assert _spread_in_ticks(1000.0, 999.99) == 0.0


def test_single_band_spread():
    """5 ticks of 0.02 in the [2, 3] band."""
    assert _spread_in_ticks(2.10, 2.20) == 5.0


def test_cross_band_spread():
    """5 ticks of 0.01 from 1.95→2.00 then 5 ticks of 0.02 from 2.00→2.10."""
    assert _spread_in_ticks(1.95, 2.10) == 10.0


def test_spread_above_cap_returns_nan():
    """Spread > 49 ticks → NaN. (1.50, 3.00) is 50 ticks of 0.01 then more."""
    result = _spread_in_ticks(1.50, 3.00)
    assert math.isnan(result), f"expected NaN, got {result!r}"


def test_min_price_clamping():
    """1.005 snaps up to MIN_PRICE = 1.01, then 4 ticks of 0.01 to 1.05."""
    assert _spread_in_ticks(1.005, 1.05) == 4.0


def test_max_price_clamping():
    """995.0 snaps to 1000 (banker's rounding: round(89.5)=90 → 1000.0).

    best_lay = 1005 is above MAX_PRICE = 1000. The original walks ticks
    from MAX; every tick stays at MAX (per ``tick_offset``'s clamp at
    line 106 of env/tick_ladder.py). MAX < 1005 - 1e-9, so the loop
    exhausts and returns NaN.
    """
    result = _spread_in_ticks(995.0, 1005.0)
    assert math.isnan(result), f"expected NaN, got {result!r}"
