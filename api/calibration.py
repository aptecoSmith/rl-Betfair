"""Calibration-card computation for the model-detail endpoint.

Scalping-active-management §05. Group ``EvaluationBetRecord``s from a
single eval run into pairs (by ``pair_id``), reuse the shared bucket
and MACE math from ``registry.calibration`` (§06 refactor), and emit
the API response shape: reliability buckets + MACE + risk-vs-realised
scatter.

Reads only from the eval-run parquet — no DB lookups here. Hard
constraints §13: calibration is reported on held-out test days only,
and the eval-run parquet is already test-day-only (evaluator writes
one file per test day).
"""

from __future__ import annotations

from dataclasses import dataclass

from api.schemas import CalibrationStats, ReliabilityBucket, RiskScatterPoint
from registry.calibration import (
    MIN_BUCKETS_FOR_MACE,
    MIN_BUCKET_SIZE,
    compute_bucket_outcomes,
    compute_mace,
)
from registry.model_store import EvaluationBetRecord


# -- Tunables --------------------------------------------------------

# Commission applied to the winning leg of a completed scalp pair.
# Matches ``BetManager.get_paired_positions`` default.
_COMMISSION = 0.05


# -- Helpers ---------------------------------------------------------


def _realised_locked_pnl(legs: list[EvaluationBetRecord]) -> float:
    """Compute a completed pair's realised locked P&L.

    Inlined from ``BetManager.get_paired_positions`` — the API layer
    reads parquet records, not live BetManagers, so the math is
    duplicated here rather than reconstructing a BetManager per pair.
    The formula is the guaranteed-floor lock amount: the minimum of
    the two settle-outcome P&Ls, clamped to zero (an "equal-stake"
    pair locks nothing — the min of win_pnl and lose_pnl is negative).
    """
    backs = [b for b in legs if b.action == "back"]
    lays = [b for b in legs if b.action == "lay"]
    if not backs or not lays:
        return 0.0
    back = max(backs, key=lambda b: b.price)
    lay = min(lays, key=lambda b: b.price)
    win_pnl = (
        back.matched_size * (back.price - 1.0) * (1.0 - _COMMISSION)
        - lay.matched_size * (lay.price - 1.0)
    )
    lose_pnl = (
        -back.matched_size
        + lay.matched_size * (1.0 - _COMMISSION)
    )
    return max(0.0, min(win_pnl, lose_pnl))


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in [0, 1]).

    Pure-Python to avoid pulling numpy into the API layer just for two
    percentile calls — the run's stddev list is O(pairs), not big.
    """
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


@dataclass
class _ScatterPair:
    """Minimal per-pair summary for the risk-vs-realised scatter."""

    predicted_pnl: float
    predicted_stddev: float
    realised_pnl: float


def _collect_scatter_pairs(
    bets: list[EvaluationBetRecord],
) -> list[_ScatterPair]:
    """Return one ``_ScatterPair`` per completed pair that has both a
    predicted-pnl and a predicted-stddev prediction. Naked pairs are
    excluded — the scatter shows realised lock only for pairs that
    actually completed.
    """
    by_pair: dict[str, list[EvaluationBetRecord]] = {}
    for bet in bets:
        if bet.pair_id is None:
            continue
        by_pair.setdefault(bet.pair_id, []).append(bet)

    scatter: list[_ScatterPair] = []
    for legs in by_pair.values():
        if len(legs) < 2:
            continue  # naked — no realised locked P&L to plot
        aggressive = min(legs, key=lambda b: b.tick_timestamp)
        predicted_pnl = aggressive.predicted_locked_pnl_at_placement
        predicted_stddev = aggressive.predicted_locked_stddev_at_placement
        if predicted_pnl is None or predicted_stddev is None:
            for leg in legs:
                if predicted_pnl is None:
                    predicted_pnl = leg.predicted_locked_pnl_at_placement
                if predicted_stddev is None:
                    predicted_stddev = (
                        leg.predicted_locked_stddev_at_placement
                    )
        if predicted_pnl is None or predicted_stddev is None:
            continue
        scatter.append(_ScatterPair(
            predicted_pnl=float(predicted_pnl),
            predicted_stddev=float(predicted_stddev),
            realised_pnl=_realised_locked_pnl(legs),
        ))
    return scatter


def _has_any_fillprob_pair(bets: list[EvaluationBetRecord]) -> bool:
    """True if at least one pair-tagged bet carries a fill-prob
    prediction. Used to distinguish "directional run" (return None)
    from "scalping run without enough data yet" (return insufficient
    stats)."""
    for bet in bets:
        if (
            bet.pair_id is not None
            and bet.fill_prob_at_placement is not None
        ):
            return True
    return False


def compute_calibration_stats(
    bets: list[EvaluationBetRecord],
) -> CalibrationStats | None:
    """Compute the calibration card payload for one evaluation run.

    Returns ``None`` when the run has no scalping pairs with
    fill-prob predictions at all (directional models, pre-Session-02
    runs) — the frontend hides the whole card in that case.
    Returns a ``CalibrationStats`` with ``insufficient_data=True`` and
    ``mace=None`` when fewer than two buckets clear the
    ``MIN_BUCKET_SIZE`` threshold.
    """
    if not _has_any_fillprob_pair(bets):
        return None

    bucket_outcomes = compute_bucket_outcomes(bets)
    buckets = [
        ReliabilityBucket(
            bucket_label=b.label,
            predicted_midpoint=b.predicted_midpoint,
            observed_rate=b.observed_rate,
            count=b.count,
            abs_calibration_error=b.abs_calibration_error,
        )
        for b in bucket_outcomes
    ]

    mace = compute_mace(bets)
    # ``compute_mace`` returns None under the same condition we flag
    # here as ``insufficient_data`` — fewer than two dense buckets.
    # We recompute the flag from the raw bucket counts rather than
    # derive it from ``mace is None`` so that a future change to
    # ``compute_mace``'s null policy (e.g. a threshold tweak) doesn't
    # accidentally flip the UI's "insufficient data" empty state.
    dense_buckets = sum(
        1 for b in bucket_outcomes if b.count >= MIN_BUCKET_SIZE
    )
    insufficient_data = dense_buckets < MIN_BUCKETS_FOR_MACE

    scatter_pairs = _collect_scatter_pairs(bets)
    stddevs_sorted = sorted(p.predicted_stddev for p in scatter_pairs)
    p25 = _percentile(stddevs_sorted, 0.25)
    p75 = _percentile(stddevs_sorted, 0.75)
    scatter: list[RiskScatterPoint] = []
    for pair in scatter_pairs:
        # Strict inequalities so that when every stddev is identical
        # (p25 == p75 == value), all points land in "med".
        if pair.predicted_stddev < p25:
            bucket = "low"
        elif pair.predicted_stddev > p75:
            bucket = "high"
        else:
            bucket = "med"
        scatter.append(RiskScatterPoint(
            predicted_pnl=pair.predicted_pnl,
            realised_pnl=pair.realised_pnl,
            stddev_bucket=bucket,
        ))

    return CalibrationStats(
        reliability_buckets=buckets,
        mace=mace,
        scatter=scatter,
        insufficient_data=insufficient_data,
    )
