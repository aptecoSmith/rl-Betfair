"""Calibration-card computation for the model-detail endpoint.

Scalping-active-management §05. Group ``EvaluationBetRecord``s from a
single eval run into pairs (by ``pair_id``), bucket the aggressive
leg's fill-probability prediction, compute observed completion rates
and MACE, and emit the per-pair risk-vs-realised scatter.

Reads only from the eval-run parquet — no DB lookups here. Hard
constraints §13: calibration is reported on held-out test days only,
and the eval-run parquet is already test-day-only (evaluator writes
one file per test day).
"""

from __future__ import annotations

from dataclasses import dataclass

from api.schemas import CalibrationStats, ReliabilityBucket, RiskScatterPoint
from registry.model_store import EvaluationBetRecord


# -- Tunables --------------------------------------------------------

# Bucket edges for the fill-prob reliability diagram. Four fixed
# buckets: <0.25, 0.25-0.5, 0.5-0.75, >0.75. Midpoints are used as
# each bucket's predicted-value proxy.
_BUCKET_EDGES = (0.25, 0.50, 0.75)
_BUCKET_MIDPOINTS = (0.125, 0.375, 0.625, 0.875)
_BUCKET_LABELS = ("<25%", "25-50%", "50-75%", ">75%")

# Minimum pair count for a bucket to contribute to MACE. Fewer than
# this and the observed_rate is too noisy to trust — the bucket still
# appears in the reliability diagram so operators can see the sparse
# bin, but it's excluded from the summary number.
_MIN_BUCKET_COUNT = 20

# If fewer than this many buckets clear ``_MIN_BUCKET_COUNT``, MACE is
# None and the whole card flags ``insufficient_data = True``.
_MIN_BUCKETS_FOR_MACE = 2

# Commission applied to the winning leg of a completed scalp pair.
# Matches ``BetManager.get_paired_positions`` default.
_COMMISSION = 0.05


# -- Helpers ---------------------------------------------------------


def _bucket_for(fill_prob: float) -> int:
    """Return the bucket index (0-3) that ``fill_prob`` lands in."""
    if fill_prob < _BUCKET_EDGES[0]:
        return 0
    if fill_prob < _BUCKET_EDGES[1]:
        return 1
    if fill_prob < _BUCKET_EDGES[2]:
        return 2
    return 3


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
class _PairSummary:
    fill_prob: float
    completed: bool
    predicted_pnl: float | None
    predicted_stddev: float | None
    realised_pnl: float  # 0.0 for naked pairs — they don't land on the scatter


def _summarise_pairs(
    bets: list[EvaluationBetRecord],
) -> list[_PairSummary]:
    """Group ``bets`` by ``pair_id`` and produce one ``_PairSummary``
    per pair that has a fill-prob prediction.

    Pairs without any fill-prob record (pre-Session-02 bets) are
    dropped — they can't be bucketed. The summary picks the first
    non-null fill-prob / predicted-pnl / predicted-stddev across the
    legs; by Session-02 design the passive inherits the aggressive
    leg's predictions so either leg's value works (the prompt specifies
    "use the aggressive for consistency with the trainer's
    pair_to_transition mapping", but the values match either way).
    """
    by_pair: dict[str, list[EvaluationBetRecord]] = {}
    for bet in bets:
        if bet.pair_id is None:
            continue
        by_pair.setdefault(bet.pair_id, []).append(bet)

    summaries: list[_PairSummary] = []
    for legs in by_pair.values():
        # Aggressive leg is the one that placed first (lowest
        # tick_timestamp). The passive's prediction is inherited
        # from it (Session 02 design) so picking the aggressive is
        # canonical but both legs carry the same value.
        aggressive = min(legs, key=lambda b: b.tick_timestamp)
        fill_prob = aggressive.fill_prob_at_placement
        if fill_prob is None:
            # Fall back to any leg that has a prediction — handles
            # the theoretical case where the aggressive is the naked
            # leg of a pre-Session-02 bet that later got a Session-02
            # passive inheriting nothing.
            for leg in legs:
                if leg.fill_prob_at_placement is not None:
                    fill_prob = leg.fill_prob_at_placement
                    break
        if fill_prob is None:
            continue

        predicted_pnl = aggressive.predicted_locked_pnl_at_placement
        predicted_stddev = aggressive.predicted_locked_stddev_at_placement
        if predicted_pnl is None or predicted_stddev is None:
            for leg in legs:
                if predicted_pnl is None:
                    predicted_pnl = leg.predicted_locked_pnl_at_placement
                if predicted_stddev is None:
                    predicted_stddev = leg.predicted_locked_stddev_at_placement

        completed = len(legs) >= 2
        realised = _realised_locked_pnl(legs) if completed else 0.0
        summaries.append(
            _PairSummary(
                fill_prob=float(fill_prob),
                completed=completed,
                predicted_pnl=(
                    float(predicted_pnl) if predicted_pnl is not None else None
                ),
                predicted_stddev=(
                    float(predicted_stddev)
                    if predicted_stddev is not None
                    else None
                ),
                realised_pnl=realised,
            )
        )
    return summaries


def compute_calibration_stats(
    bets: list[EvaluationBetRecord],
) -> CalibrationStats | None:
    """Compute the calibration card payload for one evaluation run.

    Returns ``None`` when the run has no scalping pairs with
    fill-prob predictions at all (directional models, pre-Session-02
    runs) — the frontend hides the whole card in that case.
    Returns a ``CalibrationStats`` with ``insufficient_data=True`` and
    ``mace=None`` when fewer than two buckets clear the
    ``_MIN_BUCKET_COUNT`` threshold.
    """
    summaries = _summarise_pairs(bets)
    if not summaries:
        return None

    # Reliability buckets. Every pair (completed or naked) lands in
    # one bucket by its fill-prob prediction; the bucket's observed
    # rate is completed/(completed+naked) for that bucket.
    completed_per_bucket = [0, 0, 0, 0]
    total_per_bucket = [0, 0, 0, 0]
    for s in summaries:
        idx = _bucket_for(s.fill_prob)
        total_per_bucket[idx] += 1
        if s.completed:
            completed_per_bucket[idx] += 1

    buckets: list[ReliabilityBucket] = []
    errors_over_threshold: list[float] = []
    for i in range(4):
        count = total_per_bucket[i]
        observed_rate = (
            completed_per_bucket[i] / count if count > 0 else 0.0
        )
        err = abs(_BUCKET_MIDPOINTS[i] - observed_rate)
        buckets.append(
            ReliabilityBucket(
                bucket_label=_BUCKET_LABELS[i],
                predicted_midpoint=_BUCKET_MIDPOINTS[i],
                observed_rate=observed_rate,
                count=count,
                abs_calibration_error=err,
            )
        )
        if count >= _MIN_BUCKET_COUNT:
            errors_over_threshold.append(err)

    insufficient_data = len(errors_over_threshold) < _MIN_BUCKETS_FOR_MACE
    mace: float | None = (
        None
        if insufficient_data
        else sum(errors_over_threshold) / len(errors_over_threshold)
    )

    # Scatter: one point per completed pair that has both a
    # predicted-pnl and a predicted-stddev prediction. Stddev bucket
    # is scaled off this run's 25th/75th percentiles.
    completed_with_preds = [
        s for s in summaries
        if s.completed
        and s.predicted_pnl is not None
        and s.predicted_stddev is not None
    ]
    stddevs_sorted = sorted(s.predicted_stddev for s in completed_with_preds)
    p25 = _percentile(stddevs_sorted, 0.25)
    p75 = _percentile(stddevs_sorted, 0.75)
    scatter: list[RiskScatterPoint] = []
    for s in completed_with_preds:
        # Use strict inequalities against p25/p75 so that when every
        # stddev is identical (p25 == p75 == value), all points land
        # in "med" — the test covers this single-value-run case.
        if s.predicted_stddev < p25:
            bucket = "low"
        elif s.predicted_stddev > p75:
            bucket = "high"
        else:
            bucket = "med"
        scatter.append(
            RiskScatterPoint(
                predicted_pnl=s.predicted_pnl,
                realised_pnl=s.realised_pnl,
                stddev_bucket=bucket,
            )
        )

    return CalibrationStats(
        reliability_buckets=buckets,
        mace=mace,
        scatter=scatter,
        insufficient_data=insufficient_data,
    )
