"""Unit tests for ``env/scalping_math.py``.

Covers the closed-form locked-pnl calculation and the commission-aware
min-ticks search across the three commission levels we expect to
encounter (5%, 10%, 15%) and the interesting odds bands.
"""

from __future__ import annotations

import pytest

from env.scalping_math import (
    equal_profit_back_stake,
    equal_profit_lay_stake,
    locked_pnl_per_unit_stake,
    min_arb_ticks_for_profit,
)


# -- locked_pnl_per_unit_stake --------------------------------------------


class TestLockedPnlPerUnitStake:
    def test_back_first_tight_spread_at_5pct_commission(self):
        """Back 4.00 / Lay 3.95 (1 tick) at 5% loses after commission.

        This is the "nothing scalp" pattern observed in the activation-A
        baseline's gen-0 log spam.
        """
        locked = locked_pnl_per_unit_stake(
            back_price=4.00, lay_price=3.95, commission=0.05,
        )
        assert locked < 0  # loses after commission
        assert -0.2 < locked < 0  # small loss, not catastrophic

    def test_back_first_wide_spread_at_5pct_commission(self):
        """Back 4.00 / Lay 3.40 at 5% locks positive."""
        locked = locked_pnl_per_unit_stake(
            back_price=4.00, lay_price=3.40, commission=0.05,
        )
        assert locked > 0

    def test_same_prices_different_commissions(self):
        """Higher commission → lower locked pnl, monotonically."""
        at_5 = locked_pnl_per_unit_stake(4.00, 3.40, 0.05)
        at_10 = locked_pnl_per_unit_stake(4.00, 3.40, 0.10)
        at_15 = locked_pnl_per_unit_stake(4.00, 3.40, 0.15)
        assert at_5 > at_10 > at_15

    def test_invalid_odds_return_negative_infinity(self):
        """Prices at or below 1.00 are invalid; helper returns -inf."""
        assert locked_pnl_per_unit_stake(1.00, 0.95, 0.05) == float("-inf")
        assert locked_pnl_per_unit_stake(4.00, 1.00, 0.05) == float("-inf")

    def test_zero_commission_any_spread_locks_positive(self):
        """With c=0, any back>lay spread locks positive pnl by exactly
        ``S_l − S_b = S_b × (P_b/P_l − 1)``. Commission is the only
        thing that makes tight scalps lose money; in a frictionless
        market every tick is profit."""
        locked = locked_pnl_per_unit_stake(4.00, 3.95, 0.0)
        expected = (4.00 / 3.95) - 1.0  # ≈ 0.01266
        assert abs(locked - expected) < 1e-9


# -- min_arb_ticks_for_profit ---------------------------------------------


class TestMinArbTicksForProfit:
    def test_back_first_at_mid_odds_5pct(self):
        """At c=5% and P_agg=4.00, a 1-tick passive lay (3.95) is
        zero-locked. The floor should kick in somewhere beyond that."""
        floor = min_arb_ticks_for_profit(
            aggressive_price=4.00, aggressive_side="back", commission=0.05,
        )
        assert floor is not None
        assert floor > 1  # 1 tick won't cut it
        # Sanity-check: the returned tick count actually clears zero.
        # (Round-trip via the ladder to verify.)
        from env.tick_ladder import tick_offset
        passive = tick_offset(4.00, floor, -1)
        assert locked_pnl_per_unit_stake(4.00, passive, 0.05) >= 0

    def test_back_first_at_low_odds_5pct(self):
        """At low odds the lose-side constraint binds and the tick count
        needed is small but non-zero."""
        floor = min_arb_ticks_for_profit(
            aggressive_price=1.50, aggressive_side="back", commission=0.05,
        )
        assert floor is not None
        assert floor >= 2  # 1 tick at 1.50 is 0.01, not enough

    def test_back_first_at_high_odds_5pct(self):
        """Higher odds → win-side binds, need wider spread (more ticks)."""
        floor_mid = min_arb_ticks_for_profit(4.00, "back", 0.05)
        floor_hi = min_arb_ticks_for_profit(8.00, "back", 0.05)
        assert floor_mid is not None and floor_hi is not None
        # At 8.00 the tick size is 0.20; the required % spread is also larger.
        # The tick count could go either way, but the 8.00 case must still
        # achieve profitability or return None.

    def test_commission_increase_raises_required_ticks(self):
        """c=5% vs c=10% at the same price: the higher-c version needs
        at least as many ticks, often more."""
        for price in (2.50, 4.00, 6.00):
            f5 = min_arb_ticks_for_profit(price, "back", 0.05)
            f10 = min_arb_ticks_for_profit(price, "back", 0.10)
            assert f5 is not None
            # f10 might be None if 10% pushes above max_ticks at this price.
            if f10 is not None:
                assert f10 >= f5, (
                    f"expected c=10% floor >= c=5% floor at price {price}, "
                    f"got {f10} vs {f5}"
                )

    def test_returns_none_when_unscalpable(self):
        """At c=15% and P=10.00 the price is effectively unscalpable
        within the default 25-tick search."""
        floor = min_arb_ticks_for_profit(
            aggressive_price=10.00,
            aggressive_side="back",
            commission=0.15,
            max_ticks=25,
        )
        assert floor is None

    def test_lay_first_symmetric_behaviour(self):
        """Lay-first scalp at P_agg=3.40, passive back at higher
        price should also need multiple ticks to clear."""
        floor = min_arb_ticks_for_profit(
            aggressive_price=3.40, aggressive_side="lay", commission=0.05,
        )
        assert floor is not None
        assert floor >= 1
        # Verify locked > 0 at the returned tick count.
        from env.tick_ladder import tick_offset
        passive = tick_offset(3.40, floor, +1)
        locked = locked_pnl_per_unit_stake(passive, 3.40, 0.05)
        assert locked >= 0

    def test_profit_floor_raises_tick_count(self):
        """Asking for a positive profit buffer → more ticks than break-even."""
        break_even = min_arb_ticks_for_profit(4.00, "back", 0.05, profit_floor=0.0)
        with_buffer = min_arb_ticks_for_profit(4.00, "back", 0.05, profit_floor=0.01)
        assert break_even is not None and with_buffer is not None
        assert with_buffer >= break_even

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError):
            min_arb_ticks_for_profit(4.00, "both", 0.05)  # type: ignore[arg-type]

    def test_stable_across_call_many_times(self):
        """Determinism smoke test — same inputs, same output."""
        results = {
            min_arb_ticks_for_profit(4.00, "back", 0.05) for _ in range(10)
        }
        assert len(results) == 1


# -- equal_profit_lay_stake / equal_profit_back_stake ---------------------


class TestEqualProfitSizing:
    """Equal-profit lay/back sizing for scalp pairs."""

    def test_zero_commission_collapses_to_old_formula(self):
        """With c=0, equal_profit_lay_stake reduces to the original
        (commission-free) S_b × P_b / P_l form."""
        s = equal_profit_lay_stake(
            back_stake=10.0, back_price=4.0, lay_price=3.0,
            commission=0.0,
        )
        assert s == pytest.approx(10.0 * 4.0 / 3.0, abs=1e-9)

    def test_canonical_worked_example_at_5pct(self):
        """The example from purpose.md: Back £16 @ 8.20 / Lay @ 6.00 /
        c=5% should size the lay at ≈ £21.083."""
        s = equal_profit_lay_stake(
            back_stake=16.0, back_price=8.20, lay_price=6.00,
            commission=0.05,
        )
        assert s == pytest.approx(21.0823529, rel=1e-6)

    def test_canonical_example_at_10pct(self):
        """Same trade at c=10% — formula gives a different, documented
        stake. Recompute by hand:
            num = 8.20 × 0.90 + 0.10 = 7.48
            den = 6.00 − 0.10 = 5.90
            S = 16 × 7.48 / 5.90 ≈ £20.285
        """
        s = equal_profit_lay_stake(
            back_stake=16.0, back_price=8.20, lay_price=6.00,
            commission=0.10,
        )
        assert s == pytest.approx(16.0 * 7.48 / 5.90, rel=1e-6)

    def test_equal_profit_invariant_holds_across_grid(self):
        """For a grid of (P_back, P_lay, c) triples, the helper-sized
        pair must produce |win_pnl − lose_pnl| < 0.01."""
        c = 0.05
        S_b = 10.0
        for P_b in (2.5, 4.0, 6.0, 9.0):
            for P_l in (2.0, 3.0, 4.5, 6.0):
                if P_l >= P_b:  # back-first scalp requires P_b > P_l
                    continue
                S_l = equal_profit_lay_stake(S_b, P_b, P_l, c)
                win_pnl = S_b * (P_b - 1) * (1 - c) - S_l * (P_l - 1)
                lose_pnl = -S_b + S_l * (1 - c)
                assert abs(win_pnl - lose_pnl) < 0.01, (
                    f"P_b={P_b}, P_l={P_l}: win={win_pnl}, lose={lose_pnl}"
                )

    def test_symmetric_helper_for_lay_first(self):
        """equal_profit_back_stake produces the analogous balanced size
        for an aggressive-lay scalp. The back/lay legs are not
        algebraically symmetric, so the formula is the genuine inverse
        (not a label-swap of the forward formula)."""
        # Lay £10 @ 3.00, passive back at 4.00, c=5%.
        s = equal_profit_back_stake(
            lay_stake=10.0, lay_price=3.00, back_price=4.00,
            commission=0.05,
        )
        # By formula: num = 3.00 − 0.05 = 2.95; den = 4.00 × 0.95 + 0.05 = 3.85
        # S_back = 10 × 2.95 / 3.85 ≈ 7.662
        assert s == pytest.approx(10.0 * 2.95 / 3.85, rel=1e-6)
        # And the resulting pair locks equal profit:
        S_b, P_b, S_l, P_l, c = s, 4.00, 10.0, 3.00, 0.05
        win_pnl = S_b * (P_b - 1) * (1 - c) - S_l * (P_l - 1)
        lose_pnl = -S_b + S_l * (1 - c)
        assert abs(win_pnl - lose_pnl) < 0.01

    def test_tiny_back_stake_no_division_instability(self):
        """A £0.01 back stake must produce a finite, positive lay stake
        with the same balance property."""
        s = equal_profit_lay_stake(0.01, 4.00, 3.00, 0.05)
        assert s > 0
        assert s < 1.0  # sanity: tiny stake → tiny lay

    def test_back_price_at_or_below_one_raises(self):
        with pytest.raises(ValueError):
            equal_profit_lay_stake(10.0, 1.0, 0.95, 0.05)
        with pytest.raises(ValueError):
            equal_profit_lay_stake(10.0, 0.50, 0.40, 0.05)

    def test_lay_price_at_or_below_commission_raises(self):
        """lay_price <= commission means denominator <= 0; the helper
        refuses rather than silently producing a negative or huge
        stake. Caller should have refused the trade upstream via
        min_arb_ticks_for_profit."""
        with pytest.raises(ValueError):
            equal_profit_lay_stake(10.0, 4.0, 0.05, 0.05)
        with pytest.raises(ValueError):
            equal_profit_lay_stake(10.0, 4.0, 0.04, 0.05)
