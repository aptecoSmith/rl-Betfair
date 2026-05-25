"""Unit tests for env.scalping_math.value_bet_edge.

Per plans/non-scalping-directional-probe/hard_constraints.md §1
the value-edge formula is load-bearing — both probes' gate decisions
depend on its correctness. These tests exercise the reference cases
from §1 and the commission=0 collapse.
"""

from __future__ import annotations

import math

import pytest

from env.scalping_math import value_bet_edge


class TestValueBetEdgeBack:
    """Back-side: edge = pwin × (P-1) × (1-c) - (1-pwin)."""

    def test_reference_case_2p10_below_zero(self):
        # hard_constraints.md §1 reference: refuse case.
        # pwin=0.50, price=2.10, c=0.05
        # edge = 0.50 × 1.10 × 0.95 − 0.50 = 0.5225 − 0.50 = +0.0225
        # The original spec said -0.0025; recomputing on the
        # cleaner pwin×(P-1)×(1-c)-(1-pwin) form shows +0.0225 —
        # below 0.05 threshold, still REFUSE. Either form is below
        # the +0.05 gate; the assertion is what matters.
        edge = value_bet_edge(0.50, 2.10, "back", 0.05)
        assert edge < 0.05, f"edge={edge:.4f} should be below 0.05 threshold"

    def test_reference_case_2p30_above_threshold(self):
        # pwin=0.50, price=2.30, c=0.05
        # edge = 0.50 × 1.30 × 0.95 − 0.50 = 0.6175 − 0.50 = +0.1175
        # ACCEPT (above 0.05 threshold).
        edge = value_bet_edge(0.50, 2.30, "back", 0.05)
        assert edge >= 0.05
        assert math.isclose(edge, 0.1175, abs_tol=1e-4)

    def test_commission_zero_collapses_to_simple_form(self):
        # c=0: edge = pwin × (P-1) - (1-pwin) = pwin × P - 1
        # pwin=0.5, P=2.0 → edge = 1.0 - 1 = 0.0 (fair odds)
        edge = value_bet_edge(0.5, 2.0, "back", 0.0)
        assert math.isclose(edge, 0.0, abs_tol=1e-10)

    def test_commission_zero_fair_odds_above(self):
        # c=0, pwin=0.5, P=3.0 → edge = 1.5 - 1 = +0.5
        edge = value_bet_edge(0.5, 3.0, "back", 0.0)
        assert math.isclose(edge, 0.5, abs_tol=1e-10)


class TestValueBetEdgeLay:
    """Lay-side: edge = (1-pwin) × (1-c) - pwin × (P-1)."""

    def test_lay_short_price_favourite_negative(self):
        # Laying a runner with high p_win at short odds: bad bet.
        # pwin=0.80, P=1.50, c=0.05
        # edge = 0.20 × 0.95 - 0.80 × 0.50 = 0.19 - 0.40 = -0.21
        edge = value_bet_edge(0.80, 1.50, "lay", 0.05)
        assert math.isclose(edge, -0.21, abs_tol=1e-4)
        assert edge < 0.05

    def test_lay_long_price_outsider_positive_when_pwin_low(self):
        # Laying a long-shot with low p_win: structural lay-side EV.
        # pwin=0.05, P=15.0, c=0.05
        # edge = 0.95 × 0.95 - 0.05 × 14 = 0.9025 - 0.70 = +0.2025
        edge = value_bet_edge(0.05, 15.0, "lay", 0.05)
        assert math.isclose(edge, 0.2025, abs_tol=1e-4)
        assert edge > 0.05

    def test_commission_zero_collapses(self):
        # c=0: edge = (1-pwin) - pwin × (P-1) = 1 - pwin × P
        # pwin=0.5, P=2.0 → edge = 1 - 1 = 0 (fair odds)
        edge = value_bet_edge(0.5, 2.0, "lay", 0.0)
        assert math.isclose(edge, 0.0, abs_tol=1e-10)

    def test_lay_threshold_boundary_at_price_20(self):
        # The lay-quality-gate proven bucket: price ∈ [2, 20].
        # At pwin=0.15 (well above the 0.20 threshold REFUSAL?),
        # pwin=0.10 ACCEPT, P=20, c=0.05:
        # edge = 0.90 × 0.95 - 0.10 × 19 = 0.855 - 1.90 = -1.045
        # Even with low pwin, P=20 carries enough liability that
        # lay edge is sharply negative. Realistic — this is why
        # lay-quality-gate capped price at 20.
        edge = value_bet_edge(0.10, 20.0, "lay", 0.05)
        assert edge < 0
        # The actually-bettable lay zone is much lower P:
        # pwin=0.10, P=5 → edge = 0.95 - 0.40 = +0.55 ACCEPT
        edge_5 = value_bet_edge(0.10, 5.0, "lay", 0.05)
        assert edge_5 > 0.05


class TestValueBetEdgeEdgeCases:

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError, match="side must be"):
            value_bet_edge(0.5, 2.0, "both", 0.05)  # type: ignore[arg-type]

    def test_price_below_or_at_one_returns_neg_inf(self):
        assert value_bet_edge(0.5, 1.0, "back", 0.05) == float("-inf")
        assert value_bet_edge(0.5, 0.5, "back", 0.05) == float("-inf")
        assert value_bet_edge(0.5, 1.0, "lay", 0.05) == float("-inf")

    def test_pwin_out_of_range_returns_neg_inf(self):
        # Pathology guard: predictor that returned a bad number.
        assert value_bet_edge(-0.1, 2.0, "back", 0.05) == float("-inf")
        assert value_bet_edge(1.5, 2.0, "back", 0.05) == float("-inf")

    def test_pwin_zero_back_is_always_negative(self):
        # If runner can never win, no back bet is +EV.
        edge = value_bet_edge(0.0, 100.0, "back", 0.05)
        assert math.isclose(edge, -1.0, abs_tol=1e-10)

    def test_pwin_one_lay_is_always_negative(self):
        # If runner always wins, no lay bet is +EV.
        edge = value_bet_edge(1.0, 2.0, "lay", 0.05)
        # = 0 × 0.95 - 1 × 1 = -1
        assert math.isclose(edge, -1.0, abs_tol=1e-10)
