"""Tests for P4b: passive-fill triggering and budget reservation (session 26).

Verifies that PassiveOrderBook correctly:
1.  Fills when traded volume since placement crosses the queue-ahead threshold.
2.  Does not fill when volume is insufficient.
3.  Reserves budget immediately at placement (back: deduct stake; lay: reserve liability).
4.  Does not double-subtract budget on fill.
5.  Applies passive self-depletion: a second order at the same price requires
    extra traded volume equal to the first order's filled stake.
6.  Fill price equals the queue price, not the opposite-side top.
7.  Filled passives settle with the race identically to aggressive bets.
8.  A junk-filtered rest price (LTP drifted away) suppresses fill on that tick.
9.  Aggressive bets still work correctly alongside passive orders.
10. raw + shaped ≈ total reward invariant is unbroken.

All CPU, all fast.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from data.episode_builder import PriceSize, RunnerSnap, Tick
from env.bet_manager import BetManager, BetSide, BetOutcome


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ps(price: float, size: float) -> PriceSize:
    return PriceSize(price=price, size=size)


def _runner(
    selection_id: int = 1001,
    ltp: float = 4.0,
    back_levels: list[tuple[float, float]] | None = None,
    lay_levels: list[tuple[float, float]] | None = None,
    total_matched: float = 1000.0,
    status: str = "ACTIVE",
) -> RunnerSnap:
    atb = [_ps(p, s) for p, s in (back_levels or [])]
    atl = [_ps(p, s) for p, s in (lay_levels or [])]
    return RunnerSnap(
        selection_id=selection_id,
        status=status,
        last_traded_price=ltp,
        total_matched=total_matched,
        starting_price_near=0.0,
        starting_price_far=0.0,
        adjustment_factor=None,
        bsp=None,
        sort_priority=1,
        removal_date=None,
        available_to_back=atb,
        available_to_lay=atl,
    )


_TS = datetime(2026, 4, 10, 14, 0, 0, tzinfo=timezone.utc)


def _tick(runners: list[RunnerSnap]) -> Tick:
    return Tick(
        market_id="1.99999",
        timestamp=_TS,
        sequence_number=1,
        venue="Test",
        market_start_time=_TS,
        number_of_active_runners=len(runners),
        traded_volume=0.0,
        in_play=False,
        winner_selection_id=None,
        race_status=None,
        temperature=None,
        precipitation=None,
        wind_speed=None,
        wind_direction=None,
        humidity=None,
        weather_code=None,
        runners=runners,
    )


def _mgr(budget: float = 500.0) -> BetManager:
    return BetManager(starting_budget=budget)


# ── Test 1: Rest and fill ─────────────────────────────────────────────────────


class TestRestAndFill:
    """Order rests until traded volume crosses the queue-ahead threshold."""

    def test_fills_after_threshold_crossed(self):
        mgr = _mgr()
        # LTP=4.0; back level at 3.9 with 200 queue-ahead.
        snap = _runner(ltp=4.0, back_levels=[(3.9, 200.0)], total_matched=1000.0)
        order = mgr.passive_book.place(snap, stake=10.0, side=BetSide.BACK,
                                       market_id="1.99", tick_index=0)
        assert order is not None

        # Tick 1: 150 traded — below threshold (200).
        t1 = _tick([_runner(ltp=4.0, back_levels=[(3.9, 200.0)], total_matched=1150.0)])
        mgr.passive_book.on_tick(t1)
        assert len(mgr.passive_book.orders) == 1  # still open
        assert len(mgr.bets) == 0

        # Tick 2: 60 more traded (total delta = 210) — crosses 200 threshold.
        t2 = _tick([_runner(ltp=4.0, back_levels=[(3.9, 200.0)], total_matched=1210.0)])
        mgr.passive_book.on_tick(t2)
        assert len(mgr.passive_book.orders) == 0  # filled → removed
        assert len(mgr.bets) == 1

    def test_filled_bet_has_queue_price_not_opposite_side(self):
        """Fill price == queue price (3.9), not the lay top (4.1)."""
        mgr = _mgr()
        snap = _runner(ltp=4.0, back_levels=[(3.9, 100.0)], lay_levels=[(4.1, 100.0)],
                       total_matched=1000.0)
        mgr.passive_book.place(snap, stake=10.0, side=BetSide.BACK,
                               market_id="1.99", tick_index=0)

        # Cross threshold in one tick.
        t = _tick([_runner(ltp=4.0, back_levels=[(3.9, 100.0)], lay_levels=[(4.1, 100.0)],
                           total_matched=1101.0)])
        mgr.passive_book.on_tick(t)
        assert len(mgr.bets) == 1
        assert mgr.bets[0].average_price == pytest.approx(3.9)


# ── Test 2: Rest and not-fill ─────────────────────────────────────────────────


class TestRestAndNotFill:
    """Order stays open when total traded volume never reaches the threshold."""

    def test_unfilled_when_volume_insufficient(self):
        mgr = _mgr()
        snap = _runner(ltp=4.0, back_levels=[(3.9, 200.0)], total_matched=1000.0)
        mgr.passive_book.place(snap, stake=10.0, side=BetSide.BACK,
                               market_id="1.99", tick_index=0)

        # Three ticks accumulate only 150 traded — never crosses 200.
        for cum in [1050.0, 1100.0, 1150.0]:
            t = _tick([_runner(ltp=4.0, back_levels=[(3.9, 200.0)], total_matched=cum)])
            mgr.passive_book.on_tick(t)

        assert len(mgr.passive_book.orders) == 1  # still in open list
        assert len(mgr.bets) == 0


# ── Test 3: Budget reservation at placement ───────────────────────────────────


class TestBudgetReservationAtPlacement:
    """Stake is deducted (back) or liability reserved (lay) immediately on placement."""

    def test_back_budget_drops_at_placement(self):
        mgr = _mgr(budget=500.0)
        snap = _runner(ltp=4.0, back_levels=[(3.9, 200.0)])
        budget_before = mgr.available_budget
        order = mgr.passive_book.place(snap, stake=50.0, side=BetSide.BACK,
                                       market_id="1.99", tick_index=0)
        assert order is not None
        assert mgr.available_budget == pytest.approx(budget_before - 50.0)
        # No ticks yet — the deduction is purely at placement.
        assert len(mgr.bets) == 0

    def test_lay_liability_reserved_at_placement(self):
        mgr = _mgr(budget=500.0)
        # Lay level at 4.1: liability = stake × (4.1 − 1) = 50 × 3.1 = 155
        snap = _runner(ltp=4.0, lay_levels=[(4.1, 200.0)])
        budget_before = mgr.available_budget
        order = mgr.passive_book.place(snap, stake=50.0, side=BetSide.LAY,
                                       market_id="1.99", tick_index=0)
        assert order is not None
        expected_liability = 50.0 * (4.1 - 1.0)
        assert mgr.available_budget == pytest.approx(budget_before - expected_liability)
        assert mgr.open_liability == pytest.approx(expected_liability)

    def test_back_refused_when_budget_insufficient(self):
        mgr = _mgr(budget=30.0)
        snap = _runner(ltp=4.0, back_levels=[(3.9, 200.0)])
        order = mgr.passive_book.place(snap, stake=50.0, side=BetSide.BACK,
                                       market_id="1.99", tick_index=0)
        assert order is None
        assert mgr.available_budget == pytest.approx(30.0)  # unchanged


# ── Test 4: No double-subtraction on fill ─────────────────────────────────────


class TestNoDoubleSubtractionOnFill:
    """Fill conversion does not touch budget — it was already reserved at placement."""

    def test_available_budget_unchanged_between_pre_fill_and_post_fill(self):
        mgr = _mgr(budget=500.0)
        snap = _runner(ltp=4.0, back_levels=[(3.9, 100.0)], total_matched=1000.0)
        mgr.passive_book.place(snap, stake=50.0, side=BetSide.BACK,
                               market_id="1.99", tick_index=0)

        # Record available_budget immediately after placement (stake already reserved).
        budget_after_placement = mgr.available_budget

        # Trigger fill.
        t = _tick([_runner(ltp=4.0, back_levels=[(3.9, 100.0)], total_matched=1101.0)])
        mgr.passive_book.on_tick(t)
        assert len(mgr.bets) == 1

        # Budget must not change further on fill.
        assert mgr.available_budget == pytest.approx(budget_after_placement)


# ── Test 5: Passive self-depletion ────────────────────────────────────────────


class TestPassiveSelfDepletion:
    """Two orders at the same price: second requires extra traded volume."""

    def test_second_order_delayed_by_first_fill(self):
        mgr = _mgr()
        # Both orders at price 3.9, queue-ahead 15, stake 10.
        snap = _runner(ltp=4.0, back_levels=[(3.9, 15.0)], total_matched=1000.0)
        o1 = mgr.passive_book.place(snap, stake=10.0, side=BetSide.BACK,
                                    market_id="1.99", tick_index=0)
        o2 = mgr.passive_book.place(snap, stake=10.0, side=BetSide.BACK,
                                    market_id="1.99", tick_index=0)
        assert o1 is not None and o2 is not None

        # After 15 traded: o1 threshold=15+0=15 → fills. o2 threshold=15+10=25 → not yet.
        t1 = _tick([_runner(ltp=4.0, back_levels=[(3.9, 15.0)], total_matched=1015.0)])
        mgr.passive_book.on_tick(t1)
        assert len(mgr.bets) == 1               # o1 filled
        assert len(mgr.passive_book.orders) == 1  # o2 still open

        # After 10 more (total delta=25): o2 threshold=15+10=25 → fills.
        t2 = _tick([_runner(ltp=4.0, back_levels=[(3.9, 15.0)], total_matched=1025.0)])
        mgr.passive_book.on_tick(t2)
        assert len(mgr.bets) == 2               # o2 also filled
        assert len(mgr.passive_book.orders) == 0


# ── Test 6: Fill price is the queue price ────────────────────────────────────


class TestFillPriceIsQueuePrice:
    """Bet.average_price must equal the resting queue price, not the opposite-side top."""

    def test_back_fill_price_equals_back_queue_price(self):
        mgr = _mgr()
        # Queue price 3.8; opposite-side lay top 4.2.
        snap = _runner(ltp=4.0, back_levels=[(3.8, 50.0)], lay_levels=[(4.2, 50.0)],
                       total_matched=1000.0)
        mgr.passive_book.place(snap, stake=10.0, side=BetSide.BACK,
                               market_id="1.99", tick_index=0)
        t = _tick([_runner(ltp=4.0, back_levels=[(3.8, 50.0)], lay_levels=[(4.2, 50.0)],
                           total_matched=1051.0)])
        mgr.passive_book.on_tick(t)
        assert len(mgr.bets) == 1
        # Must be queue price 3.8, NOT lay top 4.2.
        assert mgr.bets[0].average_price == pytest.approx(3.8)
        assert mgr.bets[0].average_price != pytest.approx(4.2)

    def test_lay_fill_price_equals_lay_queue_price(self):
        mgr = _mgr()
        # Lay queue price 4.2; opposite-side back top 3.8.
        snap = _runner(ltp=4.0, back_levels=[(3.8, 50.0)], lay_levels=[(4.2, 50.0)],
                       total_matched=1000.0)
        mgr.passive_book.place(snap, stake=10.0, side=BetSide.LAY,
                               market_id="1.99", tick_index=0)
        t = _tick([_runner(ltp=4.0, back_levels=[(3.8, 50.0)], lay_levels=[(4.2, 50.0)],
                           total_matched=1051.0)])
        mgr.passive_book.on_tick(t)
        assert len(mgr.bets) == 1
        # Must be lay queue price 4.2, NOT back top 3.8.
        assert mgr.bets[0].average_price == pytest.approx(4.2)
        assert mgr.bets[0].average_price != pytest.approx(3.8)


# ── Test 7: Filled passives settle with the race ─────────────────────────────


class TestFilledPassivesSettleWithRace:
    """A filled passive Bet settles identically to an aggressive Bet."""

    def test_back_fill_settles_as_winner(self):
        mgr = _mgr(budget=500.0)
        snap = _runner(selection_id=1001, ltp=4.0, back_levels=[(3.9, 50.0)],
                       total_matched=1000.0)
        mgr.passive_book.place(snap, stake=10.0, side=BetSide.BACK,
                               market_id="1.99", tick_index=0)
        # Fill.
        t = _tick([_runner(selection_id=1001, ltp=4.0, back_levels=[(3.9, 50.0)],
                           total_matched=1051.0)])
        mgr.passive_book.on_tick(t)
        assert len(mgr.bets) == 1
        bet = mgr.bets[0]
        assert bet.outcome is BetOutcome.UNSETTLED

        # Settle: runner 1001 wins.
        budget_before = mgr.budget
        pnl = mgr.settle_race(winning_selection_ids=1001, market_id="1.99")
        assert bet.outcome is BetOutcome.WON
        expected_profit = 10.0 * (3.9 - 1.0)  # stake × (price − 1)
        assert pnl == pytest.approx(expected_profit, rel=1e-6)
        assert mgr.budget == pytest.approx(budget_before + 10.0 + expected_profit)

    def test_back_fill_settles_as_loser(self):
        mgr = _mgr(budget=500.0)
        snap = _runner(selection_id=1001, ltp=4.0, back_levels=[(3.9, 50.0)],
                       total_matched=1000.0)
        mgr.passive_book.place(snap, stake=10.0, side=BetSide.BACK,
                               market_id="1.99", tick_index=0)
        t = _tick([_runner(selection_id=1001, ltp=4.0, back_levels=[(3.9, 50.0)],
                           total_matched=1051.0)])
        mgr.passive_book.on_tick(t)

        # Settle: runner 9999 wins (1001 loses).
        pnl = mgr.settle_race(winning_selection_ids=9999, market_id="1.99")
        assert mgr.bets[0].outcome is BetOutcome.LOST
        assert pnl == pytest.approx(-10.0)

    def test_filled_passive_contributes_to_realised_pnl(self):
        mgr = _mgr(budget=500.0)
        snap = _runner(selection_id=1001, ltp=4.0, back_levels=[(3.9, 50.0)],
                       total_matched=1000.0)
        mgr.passive_book.place(snap, stake=10.0, side=BetSide.BACK,
                               market_id="1.99", tick_index=0)
        t = _tick([_runner(selection_id=1001, ltp=4.0, back_levels=[(3.9, 50.0)],
                           total_matched=1051.0)])
        mgr.passive_book.on_tick(t)
        mgr.settle_race(winning_selection_ids=1001, market_id="1.99")
        expected_profit = 10.0 * (3.9 - 1.0)
        assert mgr.realised_pnl == pytest.approx(expected_profit, rel=1e-6)


# ── Test 8: Junk-filtered rest price does not fill ────────────────────────────


class TestJunkFilteredRestPriceDoesNotFill:
    """If LTP has drifted so that the resting price is outside tolerance, no fill."""

    def test_order_does_not_fill_when_ltp_drifts_away(self):
        mgr = _mgr()
        # Place at price 3.9 with LTP=4.0 (within ±50% → range [2.0, 6.0]).
        snap = _runner(ltp=4.0, back_levels=[(3.9, 50.0)], total_matched=1000.0)
        mgr.passive_book.place(snap, stake=10.0, side=BetSide.BACK,
                               market_id="1.99", tick_index=0)

        # LTP drifts to 10.0 → junk range [5.0, 15.0]. Price 3.9 is now outside.
        # Traded volume is 100, far above queue-ahead=50, but junk filter blocks fill.
        t_junk = _tick([_runner(ltp=10.0, back_levels=[(3.9, 50.0)], total_matched=1100.0)])
        mgr.passive_book.on_tick(t_junk)
        assert len(mgr.bets) == 0         # junk-filtered: no fill
        assert len(mgr.passive_book.orders) == 1  # order still open

    def test_order_fills_again_when_ltp_returns(self):
        """After drifting out, if LTP returns to where price is in-range, it can fill."""
        mgr = _mgr()
        snap = _runner(ltp=4.0, back_levels=[(3.9, 50.0)], total_matched=1000.0)
        mgr.passive_book.place(snap, stake=10.0, side=BetSide.BACK,
                               market_id="1.99", tick_index=0)

        # Tick 1: LTP drifts → junk, no fill despite volume crossing threshold.
        t_junk = _tick([_runner(ltp=10.0, back_levels=[(3.9, 50.0)], total_matched=1100.0)])
        mgr.passive_book.on_tick(t_junk)
        assert len(mgr.bets) == 0

        # Tick 2: LTP returns to 4.0 → price 3.9 back in range, threshold already met.
        t_back = _tick([_runner(ltp=4.0, back_levels=[(3.9, 50.0)], total_matched=1100.0)])
        mgr.passive_book.on_tick(t_back)
        assert len(mgr.bets) == 1  # now fills


# ── Test 9: Aggressive regression ────────────────────────────────────────────


class TestAggressiveRegression:
    """Mixed aggressive + passive in one race — aggressive tests unaffected."""

    def test_aggressive_back_still_works_alongside_passive(self):
        mgr = _mgr(budget=500.0)
        snap = _runner(ltp=4.0,
                       back_levels=[(3.9, 100.0)],
                       lay_levels=[(4.1, 100.0)],
                       total_matched=1000.0)
        # Passive back at 3.9.
        p_order = mgr.passive_book.place(snap, stake=20.0, side=BetSide.BACK,
                                         market_id="1.99", tick_index=0)
        assert p_order is not None
        # Aggressive back — matches on back side at 3.9.
        agg = mgr.place_back(snap, stake=15.0, market_id="1.99")
        assert agg is not None
        assert agg.average_price == pytest.approx(3.9)
        assert len(mgr.bets) == 1            # aggressive bet in bets
        assert len(mgr.passive_book.orders) == 1  # passive still open

        # Fill passive.
        t = _tick([_runner(ltp=4.0, back_levels=[(3.9, 100.0)], lay_levels=[(4.1, 100.0)],
                           total_matched=1101.0)])
        mgr.passive_book.on_tick(t)
        assert len(mgr.bets) == 2  # aggressive + filled passive

    def test_passive_self_depletion_does_not_affect_aggressive_matcher(self):
        """Passive fills do NOT reduce the available size seen by aggressive bets."""
        mgr = _mgr(budget=500.0)
        snap = _runner(ltp=4.0, back_levels=[(3.9, 50.0)], lay_levels=[(4.1, 100.0)],
                       total_matched=1000.0)
        # Place and fill a passive back.
        mgr.passive_book.place(snap, stake=10.0, side=BetSide.BACK,
                               market_id="1.99", tick_index=0)
        t = _tick([_runner(ltp=4.0, back_levels=[(3.9, 50.0)], lay_levels=[(4.1, 100.0)],
                           total_matched=1051.0)])
        mgr.passive_book.on_tick(t)
        assert len(mgr.bets) == 1

        # Aggressive back should see the full back-side liquidity (50).
        agg = mgr.place_back(snap, stake=40.0, market_id="1.99")
        assert agg is not None
        assert agg.matched_stake == pytest.approx(40.0)


# ── Test 10: raw + shaped ≈ total reward invariant ────────────────────────────


class TestRewardInvariant:
    """raw + shaped ≈ total_reward must hold across passive fill mechanics.

    This test exercises BetManager in isolation (no full env step), verifying
    that the passive fill path produces Bets that settle exactly like aggressive
    Bets so the invariant cannot be broken at the BetManager level.
    """

    def test_passive_fill_pnl_equals_equivalent_aggressive_pnl(self):
        """Passive fill at price P produces the same P&L as an aggressive fill at P."""
        # Passive path.
        mgr_p = _mgr(budget=500.0)
        snap = _runner(selection_id=1001, ltp=4.0,
                       back_levels=[(3.9, 50.0)], lay_levels=[(4.1, 50.0)],
                       total_matched=1000.0)
        mgr_p.passive_book.place(snap, stake=10.0, side=BetSide.BACK,
                                 market_id="1.99", tick_index=0)
        t = _tick([_runner(selection_id=1001, ltp=4.0,
                           back_levels=[(3.9, 50.0)], lay_levels=[(4.1, 50.0)],
                           total_matched=1051.0)])
        mgr_p.passive_book.on_tick(t)
        pnl_passive = mgr_p.settle_race(winning_selection_ids=1001, market_id="1.99")

        # Aggressive path at the same price.  We need to artificially construct
        # a runner where the lay top is 3.9 (matching the passive queue price).
        mgr_a = _mgr(budget=500.0)
        snap_a = _runner(selection_id=1001, ltp=4.0, back_levels=[(3.9, 100.0)])
        mgr_a.place_back(snap_a, stake=10.0, market_id="1.99")
        pnl_aggressive = mgr_a.settle_race(winning_selection_ids=1001, market_id="1.99")

        assert pnl_passive == pytest.approx(pnl_aggressive, rel=1e-6)

    def test_passive_and_aggressive_bets_settle_independently(self):
        """Mixed passive + aggressive bets in one race both settle correctly."""
        mgr = _mgr(budget=500.0)
        snap = _runner(selection_id=1001, ltp=4.0,
                       back_levels=[(3.9, 50.0)], lay_levels=[(4.1, 50.0)],
                       total_matched=1000.0)
        # Passive back £10 at 3.9.
        mgr.passive_book.place(snap, stake=10.0, side=BetSide.BACK,
                               market_id="1.99", tick_index=0)
        # Aggressive back £15 at 4.1.
        mgr.place_back(snap, stake=15.0, market_id="1.99")

        # Fill passive.
        t = _tick([_runner(selection_id=1001, ltp=4.0,
                           back_levels=[(3.9, 50.0)], lay_levels=[(4.1, 50.0)],
                           total_matched=1051.0)])
        mgr.passive_book.on_tick(t)
        assert len(mgr.bets) == 2

        # Settle: runner 1001 wins.
        pnl = mgr.settle_race(winning_selection_ids=1001, market_id="1.99")
        # Passive fill at 3.9, aggressive fill at 3.9 (both match on back side).
        expected = 10.0 * (3.9 - 1.0) + 15.0 * (3.9 - 1.0)
        assert pnl == pytest.approx(expected, rel=1e-6)
