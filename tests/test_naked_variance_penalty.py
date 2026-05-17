"""Tests for the L2 naked-variance penalty (scalping-tight-naked-variance
Phase 2A, 2026-05-15).

See plans/scalping-tight-naked-variance/hard_constraints.md §7-§11.
"""

from __future__ import annotations

import pytest

from env.betfair_env import (
    NAKED_VARIANCE_PENALTY_BETA_MAX,
    _compute_scalping_reward_terms,
)


class TestNakedVariancePenalty:
    """Five guards on the L2 variance-penalty contribution."""

    def test_beta_zero_is_byte_identical_on_shaped_term(self):
        """beta=0.0 → shaped is exactly the pre-plan formula
        (naked_winner_clip + close_bonus). The variance penalty
        contributes nothing.
        """
        # 2 winners and 1 loser — winners get the 95% clip, loser doesn't.
        # close_bonus = +£3 for three close_signal successes.
        raw_a, shaped_a = _compute_scalping_reward_terms(
            race_pnl=20.0,
            naked_per_pair=[+50.0, +30.0, -60.0],
            n_close_signal_successes=3,
            naked_variance_penalty_beta=0.0,
        )
        raw_b, shaped_b = _compute_scalping_reward_terms(
            race_pnl=20.0,
            naked_per_pair=[+50.0, +30.0, -60.0],
            n_close_signal_successes=3,
            # implicit default
        )
        assert raw_a == pytest.approx(raw_b)
        assert shaped_a == pytest.approx(shaped_b)
        # Sanity: shaped = -0.95 * (50+30) + 1.0 * 3 = -76 + 3 = -73
        assert shaped_a == pytest.approx(-73.0)

    def test_penalty_scales_quadratically(self):
        """At beta=0.005, per_pair=[+50, -50] → variance contribution
        is −beta × (50² + 50²) = −0.005 × 5000 = −£25.
        """
        # First measure shaped at beta=0 to get the baseline
        _, shaped_zero = _compute_scalping_reward_terms(
            race_pnl=0.0,
            naked_per_pair=[+50.0, -50.0],
            n_close_signal_successes=0,
            naked_variance_penalty_beta=0.0,
        )
        # Then at beta=0.005
        _, shaped_nonzero = _compute_scalping_reward_terms(
            race_pnl=0.0,
            naked_per_pair=[+50.0, -50.0],
            n_close_signal_successes=0,
            naked_variance_penalty_beta=0.005,
        )
        delta = shaped_nonzero - shaped_zero
        # Expected: penalty = -0.005 * (50² + 50²) = -25
        assert delta == pytest.approx(-25.0)

    def test_penalty_symmetric_on_pair_pnl_sign(self):
        """Hard_constraints §10. A +£100 winner and a -£100 loser
        contribute IDENTICALLY to the variance penalty: each pays
        beta × 10000. The penalty does not discriminate by sign.
        """
        beta = 0.005
        # Race A: only a +£100 winner
        _, shaped_winner = _compute_scalping_reward_terms(
            race_pnl=100.0,
            naked_per_pair=[+100.0],
            n_close_signal_successes=0,
            naked_variance_penalty_beta=beta,
        )
        _, shaped_winner_b0 = _compute_scalping_reward_terms(
            race_pnl=100.0,
            naked_per_pair=[+100.0],
            n_close_signal_successes=0,
            naked_variance_penalty_beta=0.0,
        )
        winner_penalty = shaped_winner - shaped_winner_b0

        # Race B: only a -£100 loser
        _, shaped_loser = _compute_scalping_reward_terms(
            race_pnl=-100.0,
            naked_per_pair=[-100.0],
            n_close_signal_successes=0,
            naked_variance_penalty_beta=beta,
        )
        _, shaped_loser_b0 = _compute_scalping_reward_terms(
            race_pnl=-100.0,
            naked_per_pair=[-100.0],
            n_close_signal_successes=0,
            naked_variance_penalty_beta=0.0,
        )
        loser_penalty = shaped_loser - shaped_loser_b0

        # Both penalties must be equal in magnitude AND sign (both -£50)
        assert winner_penalty == pytest.approx(-50.0)
        assert loser_penalty == pytest.approx(-50.0)
        assert winner_penalty == pytest.approx(loser_penalty)

    def test_penalty_does_not_touch_raw_pnl(self):
        """Hard_constraints §9. The variance penalty lives in the
        SHAPED channel only. ``race_reward_pnl`` (raw) is unchanged
        across beta values.
        """
        for beta in [0.0, 0.001, 0.005]:
            raw, _ = _compute_scalping_reward_terms(
                race_pnl=20.0,
                naked_per_pair=[+50.0, +30.0, -60.0],
                n_close_signal_successes=3,
                naked_variance_penalty_beta=beta,
            )
            # race_pnl = +20, naked_loss_scale = 1.0 (default) → raw == race_pnl
            assert raw == pytest.approx(20.0), (
                f"raw P&L should be invariant under beta, beta={beta}"
            )

    def test_invariant_raw_plus_shaped_total_under_nonzero_beta(self):
        """Hard_constraints §9 load-bearing guard — the
        ``raw + shaped`` total at beta>0 EQUALS the raw + shaped at
        beta=0 MINUS the L2 penalty. No accounting drift.
        """
        beta = 0.005
        pnls = [+100.0, -50.0, +30.0, -20.0]
        race_pnl = sum(pnls)
        # Closed-form variance contribution
        expected_penalty = -beta * sum(p * p for p in pnls)
        # = -0.005 * (10000 + 2500 + 900 + 400) = -0.005 * 13800 = -69.0
        assert expected_penalty == pytest.approx(-69.0)

        raw_b0, shaped_b0 = _compute_scalping_reward_terms(
            race_pnl=race_pnl,
            naked_per_pair=pnls,
            n_close_signal_successes=0,
            naked_variance_penalty_beta=0.0,
        )
        raw_b, shaped_b = _compute_scalping_reward_terms(
            race_pnl=race_pnl,
            naked_per_pair=pnls,
            n_close_signal_successes=0,
            naked_variance_penalty_beta=beta,
        )
        # raw is unchanged (above test).
        assert raw_b == pytest.approx(raw_b0)
        # shaped should differ by exactly expected_penalty.
        assert shaped_b - shaped_b0 == pytest.approx(expected_penalty)
        # Total = raw + shaped: drift equals exactly the penalty (no
        # double-counting in raw).
        total_b0 = raw_b0 + shaped_b0
        total_b = raw_b + shaped_b
        assert total_b - total_b0 == pytest.approx(expected_penalty)


class TestNakedVariancePenaltyBetaClamp:
    """Range / max-constant guard."""

    def test_beta_max_constant_matches_plan_spec(self):
        """Bumped 0.05 → 0.10 (2026-05-17, tnv3). tnv2's GA saturated
        at β=0.05 trying to find more variance pressure than the
        range allowed; widen so the gene's exploration isn't
        clipped."""
        assert NAKED_VARIANCE_PENALTY_BETA_MAX == pytest.approx(0.10)
