"""Tests for P4a: passive order queue-snapshot bookkeeping (session 25).

Verifies that PassiveOrderBook correctly:
- Snapshots queue_ahead_at_placement from the own-side top level.
- Refuses placement when the top level is junk-filtered out.
- Accumulates traded_volume_since_placement from runner total_matched deltas.
- Ignores trading on runners that don't match the passive order's selection_id.
- Starts empty on each fresh BetManager (race-reset isolation).
- Does not affect aggressive bet behaviour or budget.

All tests are CPU-only and run in milliseconds.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from data.episode_builder import PriceSize, RunnerSnap, Tick
from env.bet_manager import BetManager, BetSide, PassiveOrderBook


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
    """Build a minimal Tick containing the given runners."""
    return Tick(
        market_id="1.23456",
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


def _mgr() -> BetManager:
    return BetManager(starting_budget=500.0)


# ── Test 1: Place a passive back order ────────────────────────────────────────


class TestPassiveBackPlacement:
    """queue_ahead_at_placement captures the own-side (back) top-level size."""

    def test_queue_ahead_equals_top_back_size(self):
        mgr = _mgr()
        # Back side: best price is the highest back price (3.9), size 42.
        snap = _runner(ltp=4.0, back_levels=[(3.9, 42.0), (3.8, 100.0)])
        order = mgr.passive_book.place(snap, stake=10.0, side=BetSide.BACK,
                                       market_id="1.23", tick_index=0)
        assert order is not None
        assert order.queue_ahead_at_placement == pytest.approx(42.0)
        assert order.price == pytest.approx(3.9)
        assert order.side is BetSide.BACK
        assert order.requested_stake == pytest.approx(10.0)
        assert order.traded_volume_since_placement == pytest.approx(0.0)

    def test_order_appears_in_book(self):
        mgr = _mgr()
        snap = _runner(ltp=4.0, back_levels=[(3.9, 42.0)])
        mgr.passive_book.place(snap, stake=10.0, side=BetSide.BACK,
                               market_id="1.23", tick_index=0)
        assert len(mgr.passive_book.orders) == 1


# ── Test 2: Place a passive lay order ─────────────────────────────────────────


class TestPassiveLayPlacement:
    """queue_ahead_at_placement captures the own-side (lay) top-level size."""

    def test_queue_ahead_equals_top_lay_size(self):
        mgr = _mgr()
        # Lay side: best price is the lowest lay price (4.1), size 55.
        snap = _runner(ltp=4.0, lay_levels=[(4.1, 55.0), (4.2, 200.0)])
        order = mgr.passive_book.place(snap, stake=10.0, side=BetSide.LAY,
                                       market_id="1.23", tick_index=0)
        assert order is not None
        assert order.queue_ahead_at_placement == pytest.approx(55.0)
        assert order.price == pytest.approx(4.1)
        assert order.side is BetSide.LAY
        assert order.traded_volume_since_placement == pytest.approx(0.0)


# ── Test 3: Junk-filtered placement refused ───────────────────────────────────


class TestJunkFilteredPlacementRefused:
    """Placement into a level outside the LTP ±50% junk filter returns None."""

    def test_back_junk_level_refused(self):
        mgr = _mgr()
        # LTP = 4.0; junk filter keeps [2.0, 6.0]. Price 1.0 is outside.
        snap = _runner(ltp=4.0, back_levels=[(1.0, 100.0)])
        order = mgr.passive_book.place(snap, stake=10.0, side=BetSide.BACK,
                                       market_id="1.23", tick_index=0)
        assert order is None
        assert len(mgr.passive_book.orders) == 0

    def test_lay_junk_level_refused(self):
        mgr = _mgr()
        # LTP = 4.0; junk filter keeps [2.0, 6.0]. Price 10.0 is outside.
        snap = _runner(ltp=4.0, lay_levels=[(10.0, 100.0)])
        order = mgr.passive_book.place(snap, stake=10.0, side=BetSide.LAY,
                                       market_id="1.23", tick_index=0)
        assert order is None
        assert len(mgr.passive_book.orders) == 0

    def test_no_ltp_refused(self):
        mgr = _mgr()
        snap = _runner(ltp=0.0, back_levels=[(3.9, 100.0)])
        order = mgr.passive_book.place(snap, stake=10.0, side=BetSide.BACK,
                                       market_id="1.23", tick_index=0)
        assert order is None


# ── Test 4: Traded volume accumulates across ticks ────────────────────────────


class TestTradedVolumeAccumulates:
    """traded_volume_since_placement grows with each tick's total_matched delta."""

    def test_volume_accumulates_over_k_ticks(self):
        mgr = _mgr()
        # Queue-ahead=500 so the order never fills during the test
        # (total traded delta = 60 << 500).
        snap = _runner(selection_id=1001, ltp=4.0, back_levels=[(3.9, 500.0)],
                       total_matched=1000.0)
        mgr.passive_book.place(snap, stake=10.0, side=BetSide.BACK,
                               market_id="1.23", tick_index=0)

        # Three ticks with cumulative total_matched: 1020, 1045, 1060.
        for cum_vol in [1020.0, 1045.0, 1060.0]:
            tick = _tick([_runner(selection_id=1001, ltp=4.0,
                                  back_levels=[(3.9, 500.0)],
                                  total_matched=cum_vol)])
            mgr.passive_book.on_tick(tick)

        order = mgr.passive_book.orders[0]
        # Total delta = (1020-1000) + (1045-1020) + (1060-1045) = 60
        assert order.traded_volume_since_placement == pytest.approx(60.0)

    def test_volume_baseline_seeded_from_placement_snap(self):
        """placement() seeds _last_total_matched so the first on_tick delta
        is computed vs the runner's total_matched at placement time.
        Per open_questions.md Q4: 'compute at runtime by snapshotting at
        placement and subtracting.'
        """
        mgr = _mgr()
        snap = _runner(selection_id=1001, ltp=4.0, back_levels=[(3.9, 42.0)],
                       total_matched=1000.0)
        mgr.passive_book.place(snap, stake=10.0, side=BetSide.BACK,
                               market_id="1.23", tick_index=0)

        # First tick: 1010 total matched. Baseline = 1000 (seeded at placement).
        # Delta = 10.
        tick1 = _tick([_runner(1001, ltp=4.0, back_levels=[(3.9, 42.0)],
                               total_matched=1010.0)])
        mgr.passive_book.on_tick(tick1)

        # Second tick: 1025. Delta vs 1010 = 15.
        tick2 = _tick([_runner(1001, ltp=4.0, back_levels=[(3.9, 42.0)],
                               total_matched=1025.0)])
        mgr.passive_book.on_tick(tick2)

        order = mgr.passive_book.orders[0]
        # Total = 10 + 15 = 25.
        assert order.traded_volume_since_placement == pytest.approx(25.0)


# ── Test 5: Volume at other runners is ignored ────────────────────────────────


class TestVolumeAtOtherRunnersIgnored:
    """Traded volume for runner B does not affect a passive order on runner A."""

    def test_different_runner_volume_not_attributed(self):
        mgr = _mgr()
        # Passive order on runner 1001.
        snap_a = _runner(selection_id=1001, ltp=4.0, back_levels=[(3.9, 42.0)],
                         total_matched=1000.0)
        mgr.passive_book.place(snap_a, stake=10.0, side=BetSide.BACK,
                               market_id="1.23", tick_index=0)

        # Tick: runner 1001 unchanged, runner 1002 trades heavily.
        tick = _tick([
            _runner(selection_id=1001, ltp=4.0, back_levels=[(3.9, 42.0)],
                    total_matched=1000.0),  # no change
            _runner(selection_id=1002, ltp=3.0, back_levels=[(2.9, 100.0)],
                    total_matched=5000.0),  # lots of trading on a different runner
        ])
        mgr.passive_book.on_tick(tick)
        mgr.passive_book.on_tick(tick)  # call twice to confirm no drift

        order = mgr.passive_book.orders[0]
        assert order.traded_volume_since_placement == pytest.approx(0.0)


# ── Test 6: Race reset empties the book ───────────────────────────────────────


class TestRaceResetEmptiesBook:
    """A fresh BetManager (new race) starts with an empty PassiveOrderBook."""

    def test_new_betmanager_has_empty_passive_book(self):
        # Race A: place a passive order.
        mgr_a = _mgr()
        snap = _runner(ltp=4.0, back_levels=[(3.9, 42.0)])
        mgr_a.passive_book.place(snap, stake=10.0, side=BetSide.BACK,
                                 market_id="race_a", tick_index=0)
        assert len(mgr_a.passive_book.orders) == 1

        # Race B: fresh BetManager — passive book must be empty.
        mgr_b = BetManager(starting_budget=500.0)
        assert len(mgr_b.passive_book.orders) == 0

    def test_new_betmanager_volume_accumulator_is_empty(self):
        """_last_total_matched is also clean on a fresh BetManager."""
        mgr_a = _mgr()
        snap = _runner(selection_id=1001, ltp=4.0, back_levels=[(3.9, 42.0)],
                       total_matched=1000.0)
        mgr_a.passive_book.place(snap, stake=10.0, side=BetSide.BACK,
                                 market_id="race_a", tick_index=0)
        tick = _tick([_runner(1001, ltp=4.0, back_levels=[(3.9, 42.0)],
                              total_matched=1020.0)])
        mgr_a.passive_book.on_tick(tick)

        # Fresh BetManager: internal accumulator is reset.
        mgr_b = BetManager(starting_budget=500.0)
        assert mgr_b.passive_book._last_total_matched == {}


# ── Test 7: No aggressive regression ─────────────────────────────────────────


class TestNoAggressiveRegression:
    """Passive orders coexist with aggressive bets without interference."""

    def test_aggressive_bets_still_go_through_bets_list(self):
        mgr = _mgr()
        snap = _runner(ltp=4.0,
                       back_levels=[(3.9, 100.0)],
                       lay_levels=[(4.1, 100.0)])

        # Aggressive back.
        agg_bet = mgr.place_back(snap, stake=15.0, market_id="1.23")
        assert agg_bet is not None
        assert len(mgr.bets) == 1

        # Passive lay.
        passive_order = mgr.passive_book.place(snap, stake=10.0,
                                               side=BetSide.LAY,
                                               market_id="1.23",
                                               tick_index=1)
        assert passive_order is not None

        # Aggressive bet is in bets; passive is only in passive_book.
        assert len(mgr.bets) == 1
        assert len(mgr.passive_book.orders) == 1
        # They are distinct objects.
        assert mgr.bets[0] is agg_bet
        assert mgr.passive_book.orders[0] is passive_order

    def test_passive_placement_does_not_alter_aggressive_fill_size(self):
        """Passive book has no self-depletion effect on the aggressive matcher."""
        mgr = _mgr()
        snap = _runner(ltp=4.0, lay_levels=[(4.1, 50.0)])

        # Place passive back first.
        mgr.passive_book.place(snap, stake=30.0, side=BetSide.BACK,
                               market_id="1.23", tick_index=0)
        # Aggressive back should still see full 50 available on the lay side.
        bet = mgr.place_back(snap, stake=30.0, market_id="1.23")
        assert bet is not None
        assert bet.matched_stake == pytest.approx(30.0)


# ── Test 8: Budget reserved at passive placement (session 26) ─────────────────


class TestBudgetReservedAtPassivePlacement:
    """Session 26 lands budget reservation: placement immediately reduces budget."""

    def test_available_budget_drops_by_stake_on_back_place(self):
        mgr = _mgr()
        budget_before = mgr.available_budget
        snap = _runner(ltp=4.0, back_levels=[(3.9, 42.0)])
        mgr.passive_book.place(snap, stake=50.0, side=BetSide.BACK,
                               market_id="1.23", tick_index=0)
        assert mgr.available_budget == pytest.approx(budget_before - 50.0)

    def test_budget_field_drops_by_stake_on_back_place(self):
        mgr = _mgr()
        snap = _runner(ltp=4.0, back_levels=[(3.9, 42.0)])
        mgr.passive_book.place(snap, stake=50.0, side=BetSide.BACK,
                               market_id="1.23", tick_index=0)
        assert mgr.budget == pytest.approx(mgr.starting_budget - 50.0)
