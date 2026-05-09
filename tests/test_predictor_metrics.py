"""Unit tests for scripts/predictor/eval_metrics.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.predictor.eval_metrics import (
    calibration_gap,
    coverage,
    directional_accuracy,
    lag1_autocorr_per_group,
    mae,
    naive_backtest_pnl,
    pinball_loss,
)


# ---------------------------------------------------------------- pinball


def test_pinball_zero_when_pred_equals_truth():
    y = np.array([1.0, -2.0, 3.0])
    assert pinball_loss(y, y, 0.5) == 0.0
    assert pinball_loss(y, y, 0.1) == 0.0
    assert pinball_loss(y, y, 0.9) == 0.0


def test_pinball_asymmetric_q10_punishes_overprediction():
    y = np.array([0.0])
    over = pinball_loss(y, np.array([1.0]), 0.1)
    under = pinball_loss(y, np.array([-1.0]), 0.1)
    # q=0.1: under-predicting (y > pred) costs 0.1 * 1 = 0.1
    # q=0.1: over-predicting (y < pred) costs 0.9 * 1 = 0.9
    assert over > under
    assert under == pytest.approx(0.1)
    assert over == pytest.approx(0.9)


def test_pinball_q50_is_symmetric():
    y = np.array([0.0])
    a = pinball_loss(y, np.array([+1.0]), 0.5)
    b = pinball_loss(y, np.array([-1.0]), 0.5)
    assert a == pytest.approx(b)


def test_pinball_rejects_out_of_range_quantile():
    with pytest.raises(ValueError):
        pinball_loss(np.array([0.0]), np.array([0.0]), 0.0)
    with pytest.raises(ValueError):
        pinball_loss(np.array([0.0]), np.array([0.0]), 1.0)


# ---------------------------------------------------------------- mae / coverage


def test_mae_zero_on_identical():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0


def test_mae_basic():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([2.0, 2.0, 2.0])
    # |1-2| + |2-2| + |3-2| = 2; mean = 2/3
    assert mae(y, p) == pytest.approx(2 / 3)


def test_coverage_perfect_inside():
    y = np.array([0.0, 1.0, 2.0])
    lo = np.array([-1.0, 0.0, 1.0])
    hi = np.array([1.0, 2.0, 3.0])
    assert coverage(y, lo, hi) == 1.0


def test_coverage_perfect_outside():
    y = np.array([5.0, 5.0])
    lo = np.array([0.0, 0.0])
    hi = np.array([1.0, 1.0])
    assert coverage(y, lo, hi) == 0.0


def test_calibration_gap_zero_when_observed_matches_nominal():
    # 80% inside -> gap 0 vs nominal 0.8.
    y = np.array([0.5] * 80 + [10.0] * 20)
    lo = np.array([0.0] * 100)
    hi = np.array([1.0] * 100)
    assert calibration_gap(y, lo, hi, nominal_coverage=0.8) == 0.0


# ---------------------------------------------------------------- directional


def test_directional_no_fires_returns_nan_acc():
    y = np.zeros(10)
    q10 = np.full(10, -10.0)
    q50 = np.full(10, 0.0)
    q90 = np.full(10, 10.0)
    out = directional_accuracy(y, q10, q50, q90, k_ticks=5)
    assert out["n_total_fires"] == 0
    assert np.isnan(out["total_acc"])
    assert out["fire_rate"] == 0.0


def test_directional_short_fires_correct_when_y_positive():
    # Predict drift (q50=+10, q10=+5) in 3 cases; truth is +1, +5, -3.
    # short_fires triggers (q50>=+5 AND q10>=0): all 3.
    # correct (y > 0): first two.
    y = np.array([1.0, 5.0, -3.0])
    q10 = np.array([5.0, 5.0, 5.0])
    q50 = np.array([10.0, 10.0, 10.0])
    q90 = np.array([15.0, 15.0, 15.0])
    out = directional_accuracy(y, q10, q50, q90, k_ticks=5)
    assert out["n_short_fires"] == 3
    assert out["n_long_fires"] == 0
    assert out["short_acc"] == pytest.approx(2 / 3)


def test_directional_long_fires_correct_when_y_negative():
    y = np.array([-1.0, -5.0, +3.0])
    # q50=-10, q90=0: long_fires.
    q10 = np.array([-15.0, -15.0, -15.0])
    q50 = np.array([-10.0, -10.0, -10.0])
    q90 = np.array([0.0, 0.0, 0.0])
    out = directional_accuracy(y, q10, q50, q90, k_ticks=5)
    assert out["n_long_fires"] == 3
    assert out["long_acc"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------- autocorr


def test_lag1_autocorr_constant_trajectory_is_one():
    g1 = np.full(20, 5.0)
    assert lag1_autocorr_per_group([g1]) == 1.0


def test_lag1_autocorr_random_trajectory_near_zero():
    rng = np.random.default_rng(0)
    # Many independent groups of pure noise -> median lag-1 ~ 0.
    groups = [rng.normal(size=200) for _ in range(50)]
    val = lag1_autocorr_per_group(groups)
    assert -0.2 < val < 0.2


def test_lag1_autocorr_smooth_trajectory_high():
    # Strongly autocorrelated AR(1) with phi=0.9 -> high lag-1 autocorr.
    rng = np.random.default_rng(1)
    groups = []
    for _ in range(20):
        x = np.zeros(500)
        for t in range(1, 500):
            x[t] = 0.9 * x[t - 1] + rng.normal()
        groups.append(x)
    val = lag1_autocorr_per_group(groups)
    assert val > 0.7


def test_lag1_autocorr_short_trajectory_skipped():
    g1 = np.array([1.0, 2.0])  # too short
    g2 = np.full(20, 5.0)
    assert lag1_autocorr_per_group([g1, g2]) == 1.0


# ---------------------------------------------------------------- backtest


def test_backtest_zero_fires_means_zero_pnl_zero_winrate_nan():
    y = np.zeros(10)
    q10 = np.full(10, -10.0)
    q50 = np.zeros(10)
    q90 = np.full(10, 10.0)
    ltp = np.full(10, 5.0)
    out = naive_backtest_pnl(y, q10, q50, q90, ltp, k_ticks=5)
    assert out["n_fires"] == 0
    assert out["total_pnl"] == 0.0
    assert np.isnan(out["mean_pnl_per_fire"])


def test_backtest_short_fire_profits_when_drift_correct():
    # short-fire (q50 >= +k AND q10 >= 0) = predict drift = action lay-at-open.
    # y > 0 means price drifted UP -> lay at low closes back at high -> profit.
    y = np.array([10.0])  # +10 ticks of drift
    q10 = np.array([5.0])
    q50 = np.array([10.0])
    q90 = np.array([15.0])
    ltp = np.array([5.0])
    out = naive_backtest_pnl(y, q10, q50, q90, ltp, k_ticks=5)
    assert out["n_fires"] == 1
    assert out["total_pnl"] > 0
    assert out["win_rate"] == 1.0


def test_backtest_short_fire_loses_when_drift_wrong():
    # short-fire predicted but y < 0 -> price actually shortened.
    # Lay at low + close-back at lower price = loss.
    y = np.array([-10.0])
    q10 = np.array([5.0])
    q50 = np.array([10.0])
    q90 = np.array([15.0])
    ltp = np.array([5.0])
    out = naive_backtest_pnl(y, q10, q50, q90, ltp, k_ticks=5)
    assert out["n_fires"] == 1
    assert out["total_pnl"] < 0


def test_backtest_long_fire_profits_when_shorten_correct():
    # long-fire (q50 <= -k AND q90 <= 0) = predict shorten = action back-at-open.
    # y < 0 means price shortened -> back at high closes lay at low -> profit.
    y = np.array([-10.0])
    q10 = np.array([-15.0])
    q50 = np.array([-10.0])
    q90 = np.array([0.0])
    ltp = np.array([5.0])
    out = naive_backtest_pnl(y, q10, q50, q90, ltp, k_ticks=5)
    assert out["n_fires"] == 1
    assert out["total_pnl"] > 0
    assert out["win_rate"] == 1.0
