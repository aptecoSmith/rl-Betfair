"""Shared fill-probability calibration math.

Scalping-active-management §06. The model-detail calibration card
(Session 05, ``api/calibration.py``) and the scoreboard's MACE column
(Session 06, ``registry/scoreboard.py``) both need to reduce a run's
``EvaluationBetRecord``s into a bucketed observed-vs-predicted fill
rate and a single MACE summary number. Keeping the math here — in
``registry/`` rather than ``api/`` — means the scoreboard can import it
without crossing the registry → api dependency boundary, and keeps a
single source of truth so the two surfaces can never drift.

Exports the bucket edges, midpoints, and minimum-count constants so
they are greppable and overridable from tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from registry.model_store import EvaluationBetRecord


# -- Bucket definition ----------------------------------------------

BUCKET_EDGES: tuple[float, float, float] = (0.25, 0.50, 0.75)
BUCKET_MIDPOINTS: tuple[float, float, float, float] = (
    0.125, 0.375, 0.625, 0.875,
)
BUCKET_LABELS: tuple[str, str, str, str] = (
    "<25%", "25-50%", "50-75%", ">75%",
)

# Minimum pair count for a bucket to contribute to MACE. Fewer than
# this and the bucket's observed rate is too noisy to trust.
MIN_BUCKET_SIZE: int = 20

# If fewer than this many buckets clear ``MIN_BUCKET_SIZE``, MACE is
# None — there aren't enough dense buckets to average a meaningful
# calibration error.
MIN_BUCKETS_FOR_MACE: int = 2


@dataclass(frozen=True)
class BucketOutcome:
    """Per-bucket completion summary.

    ``abs_calibration_error`` is the gap between the bucket's
    ``predicted_midpoint`` and the empirical ``observed_rate``.
    """

    label: str
    predicted_midpoint: float
    observed_rate: float
    count: int
    abs_calibration_error: float


def _bucket_for(fill_prob: float) -> int:
    if fill_prob < BUCKET_EDGES[0]:
        return 0
    if fill_prob < BUCKET_EDGES[1]:
        return 1
    if fill_prob < BUCKET_EDGES[2]:
        return 2
    return 3


def _collect_pair_outcomes(
    bets: list[EvaluationBetRecord],
) -> list[tuple[float, bool]]:
    """Group ``bets`` by ``pair_id`` and emit one
    ``(fill_prob, completed)`` per pair.

    Pairs without a ``pair_id`` or without any fill-prob prediction on
    either leg are dropped — they can't be bucketed. The aggressive
    leg's prediction is preferred (lowest ``tick_timestamp``); the
    passive inherits the aggressive's value at placement time by
    Session-02 design, so either leg's value works but picking the
    aggressive is canonical.
    """
    by_pair: dict[str, list[EvaluationBetRecord]] = {}
    for bet in bets:
        if bet.pair_id is None:
            continue
        by_pair.setdefault(bet.pair_id, []).append(bet)

    outcomes: list[tuple[float, bool]] = []
    for legs in by_pair.values():
        aggressive = min(legs, key=lambda b: b.tick_timestamp)
        fill_prob: float | None = aggressive.fill_prob_at_placement
        if fill_prob is None:
            for leg in legs:
                if leg.fill_prob_at_placement is not None:
                    fill_prob = leg.fill_prob_at_placement
                    break
        if fill_prob is None:
            continue
        outcomes.append((float(fill_prob), len(legs) >= 2))
    return outcomes


def compute_bucket_outcomes(
    bets: list[EvaluationBetRecord],
) -> list[BucketOutcome]:
    """Reduce ``bets`` to four fill-prob buckets, one ``BucketOutcome``
    per bucket. Empty buckets are reported with ``count=0`` and
    ``observed_rate=0.0``.
    """
    outcomes = _collect_pair_outcomes(bets)
    completed_per_bucket = [0, 0, 0, 0]
    total_per_bucket = [0, 0, 0, 0]
    for fp, completed in outcomes:
        idx = _bucket_for(fp)
        total_per_bucket[idx] += 1
        if completed:
            completed_per_bucket[idx] += 1

    buckets: list[BucketOutcome] = []
    for i in range(4):
        count = total_per_bucket[i]
        observed_rate = (
            completed_per_bucket[i] / count if count > 0 else 0.0
        )
        buckets.append(BucketOutcome(
            label=BUCKET_LABELS[i],
            predicted_midpoint=BUCKET_MIDPOINTS[i],
            observed_rate=observed_rate,
            count=count,
            abs_calibration_error=abs(
                BUCKET_MIDPOINTS[i] - observed_rate
            ),
        ))
    return buckets


def compute_mace(
    bets: list[EvaluationBetRecord],
    *,
    min_bucket_size: int = MIN_BUCKET_SIZE,
) -> float | None:
    """Return mean absolute calibration error, or ``None`` if fewer
    than ``MIN_BUCKETS_FOR_MACE`` buckets clear ``min_bucket_size``.

    Shared between the model-detail calibration card (session 05) and
    the scoreboard column (session 06) — keep the math in one place so
    the two surfaces can never drift. The ``min_bucket_size`` kwarg is
    for tests; production callers should use the default.
    """
    buckets = compute_bucket_outcomes(bets)
    errors = [
        b.abs_calibration_error
        for b in buckets
        if b.count >= min_bucket_size
    ]
    if len(errors) < MIN_BUCKETS_FOR_MACE:
        return None
    return sum(errors) / len(errors)
