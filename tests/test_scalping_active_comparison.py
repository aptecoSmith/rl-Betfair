"""Unit tests for ``scripts/scalping_active_comparison.py``.

Covers the pure-function math (Spearman, bucketing, null handling) on
synthetic bets — the CLI layer that reads from ModelStore is exercised
by the existing model-store tests and integration in the activation
playbook's real runs.
"""

from __future__ import annotations

from registry.model_store import EvaluationBetRecord
from scripts.scalping_active_comparison import (
    compute_activation_metrics,
    spearman_rho,
)


def _bet(
    *,
    pair_id: str,
    tick_timestamp: str,
    action: str,
    pnl: float,
    fill_prob: float | None = None,
    pred_pnl: float | None = None,
    pred_stddev: float | None = None,
    matched_size: float = 10.0,
    price: float = 3.0,
) -> EvaluationBetRecord:
    """One leg of a scalp pair. Defaults line up so two legs with
    opposite ``action`` values form a valid pair that
    ``_realised_locked_pnl`` can evaluate to a non-zero floor when both
    are matched."""
    return EvaluationBetRecord(
        run_id="test-run",
        date="2026-04-17",
        market_id="1.1",
        tick_timestamp=tick_timestamp,
        seconds_to_off=120.0,
        runner_id=1,
        runner_name="R",
        action=action,
        price=price,
        stake=matched_size,
        matched_size=matched_size,
        outcome="won",
        pnl=pnl,
        pair_id=pair_id,
        fill_prob_at_placement=fill_prob,
        predicted_locked_pnl_at_placement=pred_pnl,
        predicted_locked_stddev_at_placement=pred_stddev,
    )


# -- spearman_rho --------------------------------------------------------


class TestSpearmanRho:
    def test_perfect_monotone(self):
        rho = spearman_rho([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
        assert rho == 1.0

    def test_perfect_inverse(self):
        rho = spearman_rho([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0])
        assert rho == -1.0

    def test_none_when_too_few_points(self):
        assert spearman_rho([], []) is None
        assert spearman_rho([1.0], [2.0]) is None

    def test_none_when_zero_variance(self):
        """Constant input has undefined rank correlation."""
        assert spearman_rho([1.0, 1.0, 1.0], [4.0, 5.0, 6.0]) is None

    def test_mismatched_lengths_raises(self):
        import pytest
        with pytest.raises(ValueError):
            spearman_rho([1.0, 2.0], [1.0])

    def test_handles_ties_via_average_ranks(self):
        """Two tied xs, perfectly ordered ys — rho should still be near
        +1 (ties lower it slightly; we just assert sign + magnitude)."""
        rho = spearman_rho([1.0, 1.0, 2.0, 3.0], [10.0, 20.0, 30.0, 40.0])
        assert rho is not None and rho > 0.8


# -- compute_activation_metrics -----------------------------------------


class TestComputeActivationMetrics:
    def test_empty_bets_yields_nulls(self):
        m = compute_activation_metrics([], run_id="empty")
        assert m.n_bets == 0
        assert m.n_pairs_with_fill_prob == 0
        assert m.n_completed_pairs_with_risk == 0
        assert m.fill_prob_mace is None
        assert m.risk_spearman_rho is None
        assert len(m.bucket_rows) == 4

    def test_directional_bets_no_pair_id_ignored(self):
        bets = [
            _bet(pair_id=None, tick_timestamp="t1", action="back", pnl=5.0),
        ]
        # None pair_id → the helper skips; the payload has no fill-prob
        # data so everything is null.
        # (Note: pair_id=None is mutually exclusive with passing it in
        # the factory; we construct directly instead.)
        bets[0].pair_id = None
        m = compute_activation_metrics(bets)
        assert m.n_pairs_with_fill_prob == 0
        assert m.fill_prob_mace is None
        assert m.risk_spearman_rho is None

    def test_well_calibrated_head_gives_low_mace(self):
        """Construct 160 pairs spread across 4 buckets with empirical
        completion rates matching the bucket midpoints exactly. MACE
        should be ≈ 0. 40 pairs per bucket so each clears
        ``MIN_BUCKET_SIZE = 20``."""
        bucket_midpoints = [0.125, 0.375, 0.625, 0.875]
        pair_ix = 0
        bets: list[EvaluationBetRecord] = []
        per_bucket = 40
        for midpoint in bucket_midpoints:
            n_completed = int(round(midpoint * per_bucket))
            for i in range(per_bucket):
                pair_ix += 1
                bets.append(_bet(
                    pair_id=f"p{pair_ix}",
                    tick_timestamp=f"a{pair_ix}",
                    action="back",
                    pnl=1.0,
                    fill_prob=midpoint,
                    price=3.0,
                    matched_size=10.0,
                ))
                if i < n_completed:
                    bets.append(_bet(
                        pair_id=f"p{pair_ix}",
                        tick_timestamp=f"b{pair_ix}",
                        action="lay",
                        pnl=-0.5,
                        fill_prob=midpoint,
                        price=2.5,
                        matched_size=10.0,
                    ))
        m = compute_activation_metrics(bets)
        assert m.fill_prob_mace is not None
        assert m.fill_prob_mace < 0.01  # near-perfect calibration

    def test_uncalibrated_head_gives_high_mace(self):
        """All pairs predict 0.9 but only half complete — big error."""
        bets: list[EvaluationBetRecord] = []
        for i in range(20):
            bets.append(_bet(
                pair_id=f"p{i}", tick_timestamp=f"a{i}", action="back",
                pnl=1.0, fill_prob=0.9, price=3.0, matched_size=10.0,
            ))
            if i < 10:
                bets.append(_bet(
                    pair_id=f"p{i}", tick_timestamp=f"b{i}", action="lay",
                    pnl=-0.5, fill_prob=0.9, price=2.5, matched_size=10.0,
                ))
        m = compute_activation_metrics(bets)
        # Only one bucket has any data (the top bucket), so MACE needs
        # at least two filled buckets to compute — returns None.
        # Move to a two-bucket case to exercise the error itself.
        assert m.fill_prob_mace is None
        assert m.bucket_rows[3]["count"] == 20
        assert abs(m.bucket_rows[3]["observed_rate"] - 0.5) < 1e-9

    def test_mace_reports_when_two_or_more_buckets_filled(self):
        bets: list[EvaluationBetRecord] = []
        # Bucket 0 (midpoint 0.125): predict 0.1, observe 1.0 → error 0.875
        for i in range(25):
            bets.append(_bet(
                pair_id=f"low{i}", tick_timestamp=f"a{i}", action="back",
                pnl=1.0, fill_prob=0.1, price=3.0, matched_size=10.0,
            ))
            bets.append(_bet(
                pair_id=f"low{i}", tick_timestamp=f"b{i}", action="lay",
                pnl=-0.5, fill_prob=0.1, price=2.5, matched_size=10.0,
            ))
        # Bucket 3 (midpoint 0.875): predict 0.9, observe 0.0 → error 0.875
        for i in range(25):
            bets.append(_bet(
                pair_id=f"hi{i}", tick_timestamp=f"c{i}", action="back",
                pnl=1.0, fill_prob=0.9, price=3.0, matched_size=10.0,
            ))
        m = compute_activation_metrics(bets)
        assert m.fill_prob_mace is not None
        assert abs(m.fill_prob_mace - 0.875) < 1e-6

    def test_risk_spearman_positive_when_stddev_tracks_error(self):
        """High predicted-stddev pairs have high |realised − predicted|.

        All three pairs realise locked-pnl = 0 given back.price=3.0 /
        lay.price=2.5 / matched=10.0 (see ``_realised_locked_pnl``'s
        floor-at-zero clamp). So ``|realised - predicted|`` reduces to
        ``|predicted|``, and we set predicted_pnl so the ordering
        matches stddev exactly."""
        bets: list[EvaluationBetRecord] = []
        # Pair 1: low stddev, small |predicted|
        bets.extend([
            _bet(pair_id="p1", tick_timestamp="a1", action="back",
                 pnl=1.0, fill_prob=0.5, pred_pnl=0.1, pred_stddev=0.1,
                 price=3.0, matched_size=10.0),
            _bet(pair_id="p1", tick_timestamp="b1", action="lay",
                 pnl=-0.5, fill_prob=0.5, price=2.5, matched_size=10.0),
        ])
        # Pair 2: medium stddev, medium |predicted|
        bets.extend([
            _bet(pair_id="p2", tick_timestamp="a2", action="back",
                 pnl=1.0, fill_prob=0.5, pred_pnl=1.0, pred_stddev=0.5,
                 price=3.0, matched_size=10.0),
            _bet(pair_id="p2", tick_timestamp="b2", action="lay",
                 pnl=-0.5, fill_prob=0.5, price=2.5, matched_size=10.0),
        ])
        # Pair 3: high stddev, large |predicted|
        bets.extend([
            _bet(pair_id="p3", tick_timestamp="a3", action="back",
                 pnl=1.0, fill_prob=0.5, pred_pnl=5.0, pred_stddev=2.0,
                 price=3.0, matched_size=10.0),
            _bet(pair_id="p3", tick_timestamp="b3", action="lay",
                 pnl=-0.5, fill_prob=0.5, price=2.5, matched_size=10.0),
        ])
        m = compute_activation_metrics(bets)
        assert m.n_completed_pairs_with_risk == 3
        # Monotone mapping of three points → rho = 1.0.
        assert m.risk_spearman_rho == 1.0

    def test_risk_spearman_none_when_no_risk_predictions(self):
        bets = [
            _bet(pair_id="p1", tick_timestamp="a1", action="back",
                 pnl=1.0, fill_prob=0.5, price=3.0, matched_size=10.0),
            _bet(pair_id="p1", tick_timestamp="b1", action="lay",
                 pnl=-0.5, fill_prob=0.5, price=2.5, matched_size=10.0),
        ]
        m = compute_activation_metrics(bets)
        assert m.n_completed_pairs_with_risk == 0
        assert m.risk_spearman_rho is None
