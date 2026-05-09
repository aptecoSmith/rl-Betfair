"""scripts/predictor/eval_metrics.py - pure metric functions.

Imported by train_one.py / run_matrix.py / backtest. Every function
here takes numpy arrays and returns a number or a small dict; no
parquet I/O, no torch dependency. Tested in tests/test_predictor_metrics.py.

The scoreboard reads:

  pinball_loss(y, q10, q50, q90)
  mae(y, q50)
  calibration_gap(y, q10, q90, nominal=0.8)
  directional_accuracy(y, q10, q50, q90, k_ticks)
  lag1_autocorr(predictions_per_market_runner_tick)
  naive_backtest_pnl(y, q10, q50, q90, ltp_now, k_ticks, side)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
) -> float:
    """Standard pinball / quantile loss.

    Lower is better. Asymmetric penalty: under-predicts cost
    `quantile * error`, over-predicts cost `(1-quantile) * error`.
    """
    if quantile <= 0 or quantile >= 1:
        raise ValueError(f"quantile must be in (0,1), got {quantile}")
    err = y_true - y_pred
    return float(np.mean(np.maximum(quantile * err, (quantile - 1) * err)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def calibration_gap(
    y_true: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
    nominal_coverage: float = 0.8,
) -> float:
    """Absolute gap between observed and nominal interval coverage.

    Returns |observed_coverage - nominal_coverage|. 0 = perfectly
    calibrated; larger values are worse. The interval is `[q_low,
    q_high]`; for a 10/90 prediction the nominal is 0.8.
    """
    inside = (y_true >= q_low) & (y_true <= q_high)
    observed = float(np.mean(inside))
    return abs(observed - nominal_coverage)


def coverage(
    y_true: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
) -> float:
    """Fraction of `y_true` inside `[q_low, q_high]`."""
    return float(np.mean((y_true >= q_low) & (y_true <= q_high)))


def directional_accuracy(
    y_true: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    k_ticks: int,
) -> dict[str, float]:
    """Operator-relevant decision rule: how often is the model right
    when it's confident.

    Two firing rules, evaluated symmetrically:
    - LONG-shorten signal: q50 <= -k AND q90 <= 0
        (median says will shorten by >=k; 90th percentile still
         non-positive). Realised "correct" iff y_true < 0.
    - SHORT-drift signal: q50 >= +k AND q10 >= 0
        (median says will drift by >=k; 10th percentile still
         non-negative). Realised "correct" iff y_true > 0.

    Returns:
      n_long_fires, long_acc, n_short_fires, short_acc, n_total_fires,
      total_acc, fire_rate.

    A model that never fires (overcautious) returns acc = NaN for
    that side. A model that fires often but is right 50% of the
    time isn't useful.
    """
    n = len(y_true)
    long_fires = (q50 <= -k_ticks) & (q90 <= 0)
    short_fires = (q50 >= +k_ticks) & (q10 >= 0)
    long_correct = long_fires & (y_true < 0)
    short_correct = short_fires & (y_true > 0)
    nl = int(long_fires.sum())
    ns = int(short_fires.sum())
    return {
        "n_long_fires": nl,
        "long_acc": float(long_correct.sum() / nl) if nl > 0 else float("nan"),
        "n_short_fires": ns,
        "short_acc": float(short_correct.sum() / ns) if ns > 0 else float("nan"),
        "n_total_fires": nl + ns,
        "total_acc": (
            float((long_correct.sum() + short_correct.sum()) / (nl + ns))
            if (nl + ns) > 0 else float("nan")
        ),
        "fire_rate": float((nl + ns) / n) if n > 0 else 0.0,
    }


def lag1_autocorr_per_group(
    values_by_group: Iterable[np.ndarray],
) -> float:
    """Median lag-1 autocorrelation across (market, runner) trajectories.

    `values_by_group` yields one 1-D array per (market, runner) trajectory
    of model predictions through time. We compute lag-1 autocorr per
    trajectory, then return the median across trajectories. Trajectories
    of length < 3 contribute NaN and are excluded.

    >= 0.7 = stable (operator's threshold). Closer to 1 = predictions
    barely change tick-to-tick. Below 0 = oscillating wildly.
    """
    accs: list[float] = []
    for arr in values_by_group:
        a = np.asarray(arr, dtype=float)
        if len(a) < 3:
            continue
        x = a[:-1]
        y = a[1:]
        # Remove NaN pairs.
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            continue
        x, y = x[mask], y[mask]
        if x.std() < 1e-9 or y.std() < 1e-9:
            # Constant trajectory; perfectly stable but uninformative.
            accs.append(1.0)
            continue
        accs.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.median(accs)) if accs else float("nan")


def naive_backtest_pnl(
    y_true_ticks: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    ltp_now: np.ndarray,
    k_ticks: int,
    commission: float = 0.05,
    stake_per_open: float = 10.0,
) -> dict[str, float]:
    """Sanity-check P&L using a direction-correct simplification.

    Fire rules match `directional_accuracy`:
    - SHORT-DRIFT signal (q50>=+k AND q10>=0): action = lay-at-LTP,
      close-back-at-future-LTP. Profits when price drifts UP
      (y_true_ticks > 0).
    - LONG-SHORTEN signal (q50<=-k AND q90<=0): action = back-at-LTP,
      close-lay-at-future-LTP. Profits when price shortens
      (y_true_ticks < 0).

    P&L per fire is computed using the standard
    open-then-close trade math:

      back side: pnl = stake * (P_open - P_close) / P_close
      lay side:  pnl = stake * (P_close - P_open) / P_close

    Future price is approximated as
      `P_open * (1 + y_true_ticks * per_tick_frac)`
    with `per_tick_frac=0.01` (rough average across the Betfair
    ladder; S08 uses real future LTP from the dataset).

    Commission applies to NET winnings only (Betfair convention)
    and to the per-trade P&L when positive. This is a smoke metric;
    do NOT tune against it.
    """
    n = len(y_true_ticks)
    short_fires = (q50 >= +k_ticks) & (q10 >= 0)
    long_fires = (q50 <= -k_ticks) & (q90 <= 0)
    per_tick_price_frac = 0.01

    total = 0.0
    n_fires = 0
    n_wins = 0
    for i in range(n):
        if not (np.isfinite(y_true_ticks[i]) and np.isfinite(ltp_now[i])):
            continue
        p_open = float(ltp_now[i])
        if p_open <= 1.0:
            continue
        p_close = p_open * (1.0 + float(y_true_ticks[i]) * per_tick_price_frac)
        if p_close <= 1.0:
            continue

        pnl: float | None = None
        if short_fires[i]:
            # Predict drift -> lay at open, close-back at future.
            # Profitable when p_close > p_open.
            pnl = stake_per_open * (p_close - p_open) / p_close
        elif long_fires[i]:
            # Predict shorten -> back at open, close-lay at future.
            # Profitable when p_close < p_open.
            pnl = stake_per_open * (p_open - p_close) / p_close

        if pnl is None:
            continue
        if pnl > 0:
            pnl *= (1.0 - commission)
        total += pnl
        n_fires += 1
        n_wins += int(pnl > 0)
    return {
        "n_fires": n_fires,
        "total_pnl": total,
        "mean_pnl_per_fire": (total / n_fires) if n_fires > 0 else float("nan"),
        "win_rate": (n_wins / n_fires) if n_fires > 0 else float("nan"),
    }
