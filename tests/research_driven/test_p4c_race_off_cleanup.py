"""Tests for P4c: race-off cleanup for unfilled passive orders (session 27).

Verifies that at race-off:
1.  Unfilled passive orders are cancelled cleanly.
2.  Cancelled passives contribute zero P&L.
3.  Budget is fully restored after cleanup.
4.  Cleanup is idempotent.
5.  Filled passives are NOT touched by cleanup.
6.  Cleanup does not leak state into the next race.
7.  Efficiency penalty interaction: cancelled passives count toward bet_count.
8.  raw + shaped ≈ total_reward invariant holds.
9.  Mixed aggressive + passive race settles correctly end-to-end.

All CPU, all fast.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from data.episode_builder import Day, PriceSize, Race, RunnerMeta, RunnerSnap, Tick
from env.bet_manager import BetManager, BetSide, BetOutcome


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_runner_meta(selection_id: int, name: str = "Horse") -> RunnerMeta:
    """Create a minimal RunnerMeta with sensible defaults."""
    return RunnerMeta(
        selection_id=selection_id,
        runner_name=name,
        sort_priority="1",
        handicap="0",
        sire_name="Sire",
        dam_name="Dam",
        damsire_name="DamSire",
        bred="GB",
        official_rating="85",
        adjusted_rating="85",
        age="4",
        sex_type="GELDING",
        colour_type="BAY",
        weight_value="140",
        weight_units="LB",
        jockey_name="J Smith",
        jockey_claim="0",
        trainer_name="T Jones",
        owner_name="Owner",
        stall_draw="3",
        cloth_number="1",
        form="1234",
        days_since_last_run="14",
        wearing="",
        forecastprice_numerator="3",
        forecastprice_denominator="1",
    )


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


def _tick(
    runners: list[RunnerSnap],
    in_play: bool = False,
    market_id: str = "1.99999",
) -> Tick:
    return Tick(
        market_id=market_id,
        timestamp=_TS,
        sequence_number=1,
        venue="Test",
        market_start_time=_TS,
        number_of_active_runners=len(runners),
        traded_volume=0.0,
        in_play=in_play,
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


# ── Test 1: Unfilled passive is cancelled at race-off ────────────────────────


class TestUnfilledPassiveCancelledAtRaceOff:
    """An unfilled passive order is removed from the open list on cancel_all."""

    def test_cancelled_and_removed_from_open_list(self):
        mgr = _mgr()
        snap = _runner(ltp=4.0, back_levels=[(3.9, 200.0)], total_matched=1000.0)
        order = mgr.passive_book.place(
            snap, stake=10.0, side=BetSide.BACK, market_id="1.99", tick_index=0,
        )
        assert order is not None
        assert len(mgr.passive_book.orders) == 1

        # Simulate some traded volume but not enough to fill.
        t = _tick([_runner(ltp=4.0, back_levels=[(3.9, 200.0)], total_matched=1050.0)])
        mgr.passive_book.on_tick(t)
        assert len(mgr.passive_book.orders) == 1  # still open

        # Race-off cleanup.
        mgr.passive_book.cancel_all("race-off")

        assert len(mgr.passive_book.orders) == 0
        assert order.cancelled is True
        assert order.cancel_reason == "race-off"
        assert len(mgr.bets) == 0  # no Bet created

    def test_cancellation_event_emitted(self):
        mgr = _mgr()
        snap = _runner(ltp=4.0, back_levels=[(3.9, 200.0)], total_matched=1000.0)
        mgr.passive_book.place(
            snap, stake=10.0, side=BetSide.BACK, market_id="1.99", tick_index=0,
        )
        mgr.passive_book.cancel_all("race-off")

        cancels = mgr.passive_book.last_cancels
        assert len(cancels) == 1
        assert cancels[0]["selection_id"] == 1001
        assert cancels[0]["price"] == pytest.approx(3.9)
        assert cancels[0]["requested_stake"] == pytest.approx(10.0)
        assert cancels[0]["reason"] == "race-off"

    def test_cancelled_order_in_history(self):
        mgr = _mgr()
        snap = _runner(ltp=4.0, back_levels=[(3.9, 200.0)], total_matched=1000.0)
        mgr.passive_book.place(
            snap, stake=10.0, side=BetSide.BACK, market_id="1.99", tick_index=0,
        )
        mgr.passive_book.cancel_all("race-off")

        history = mgr.passive_book.cancelled_orders
        assert len(history) == 1
        assert history[0].cancelled is True


# ── Test 2: Cancelled passive contributes zero P&L ──────────────────────────


class TestCancelledPassiveZeroPnl:
    """P&L from the race equals P&L from non-passive bets only."""

    def test_cancelled_passive_does_not_affect_pnl(self):
        mgr = _mgr(budget=500.0)
        snap = _runner(
            selection_id=1001, ltp=4.0,
            back_levels=[(3.9, 200.0)], lay_levels=[(4.1, 200.0)],
            total_matched=1000.0,
        )

        # Place an aggressive back that will match.
        agg = mgr.place_back(snap, stake=20.0, market_id="1.99")
        assert agg is not None
        agg_price = agg.average_price

        # Place a passive that will NOT fill.
        mgr.passive_book.place(
            snap, stake=10.0, side=BetSide.BACK, market_id="1.99", tick_index=0,
        )

        # No traded volume → passive stays open.
        t = _tick([_runner(
            selection_id=1001, ltp=4.0,
            back_levels=[(3.9, 200.0)], lay_levels=[(4.1, 200.0)],
            total_matched=1000.0,
        )])
        mgr.passive_book.on_tick(t)

        # Cancel passives, then settle.
        mgr.passive_book.cancel_all("race-off")
        pnl = mgr.settle_race(winning_selection_ids=1001, market_id="1.99")

        # P&L should come only from the aggressive bet.
        expected_pnl = 20.0 * (agg_price - 1.0)  # back winner profit
        assert pnl == pytest.approx(expected_pnl, rel=1e-6)

    def test_cancelled_passive_zero_pnl_no_aggressive_bets(self):
        """If the only order was a passive that got cancelled, race P&L = 0."""
        mgr = _mgr(budget=500.0)
        snap = _runner(ltp=4.0, back_levels=[(3.9, 200.0)], total_matched=1000.0)
        mgr.passive_book.place(
            snap, stake=10.0, side=BetSide.BACK, market_id="1.99", tick_index=0,
        )
        mgr.passive_book.cancel_all("race-off")
        pnl = mgr.settle_race(winning_selection_ids=1001, market_id="1.99")
        assert pnl == pytest.approx(0.0)
        assert mgr.realised_pnl == pytest.approx(0.0)


# ── Test 3: Budget fully restored ───────────────────────────────────────────


class TestBudgetFullyRestored:
    """After cleanup, available_budget equals what it would be if the passive never existed."""

    def test_back_budget_restored(self):
        mgr = _mgr(budget=500.0)
        snap = _runner(ltp=4.0, back_levels=[(3.9, 200.0)], total_matched=1000.0)
        mgr.passive_book.place(
            snap, stake=50.0, side=BetSide.BACK, market_id="1.99", tick_index=0,
        )
        assert mgr.available_budget == pytest.approx(450.0)

        mgr.passive_book.cancel_all("race-off")
        assert mgr.available_budget == pytest.approx(500.0)

    def test_lay_budget_restored(self):
        mgr = _mgr(budget=500.0)
        snap = _runner(ltp=4.0, lay_levels=[(4.1, 200.0)], total_matched=1000.0)
        mgr.passive_book.place(
            snap, stake=50.0, side=BetSide.LAY, market_id="1.99", tick_index=0,
        )
        liability = 50.0 * (4.1 - 1.0)  # 155.0
        assert mgr.available_budget == pytest.approx(500.0 - liability)
        assert mgr.open_liability == pytest.approx(liability)

        mgr.passive_book.cancel_all("race-off")
        assert mgr.available_budget == pytest.approx(500.0)
        assert mgr.open_liability == pytest.approx(0.0)


# ── Test 4: Idempotent cleanup ──────────────────────────────────────────────


class TestIdempotentCleanup:
    """Calling cancel_all twice produces the same state as calling it once."""

    def test_double_cancel_is_idempotent(self):
        mgr = _mgr(budget=500.0)
        snap = _runner(ltp=4.0, back_levels=[(3.9, 200.0)], total_matched=1000.0)
        mgr.passive_book.place(
            snap, stake=50.0, side=BetSide.BACK, market_id="1.99", tick_index=0,
        )

        mgr.passive_book.cancel_all("race-off")
        # Capture state after first cancel.
        budget_after_first = mgr.available_budget
        orders_after_first = len(mgr.passive_book.orders)
        cancel_count_after_first = mgr.passive_book.cancel_count
        history_after_first = len(mgr.passive_book.cancelled_orders)

        mgr.passive_book.cancel_all("race-off")
        # State must be identical.
        assert mgr.available_budget == pytest.approx(budget_after_first)
        assert len(mgr.passive_book.orders) == orders_after_first
        assert mgr.passive_book.cancel_count == cancel_count_after_first
        assert len(mgr.passive_book.cancelled_orders) == history_after_first


# ── Test 5: Cleanup does not touch filled passives ──────────────────────────


class TestCleanupDoesNotTouchFilledPassives:
    """A filled passive stays in BetManager.bets; only the unfilled one is cancelled."""

    def test_filled_survives_cancel_unfilled_removed(self):
        mgr = _mgr(budget=500.0)
        snap = _runner(
            selection_id=1001, ltp=4.0,
            back_levels=[(3.9, 50.0)], lay_levels=[(4.1, 200.0)],
            total_matched=1000.0,
        )

        # Place two passives: one will fill, one won't.
        p1 = mgr.passive_book.place(
            snap, stake=10.0, side=BetSide.BACK, market_id="1.99", tick_index=0,
        )
        p2 = mgr.passive_book.place(
            snap, stake=10.0, side=BetSide.BACK, market_id="1.99", tick_index=1,
        )
        assert p1 is not None and p2 is not None

        # Fill p1 only: threshold for p1=50, for p2=50+10=60.
        # Traded volume delta = 55 → p1 fills (55>=50), p2 doesn't (55<60).
        t = _tick([_runner(
            selection_id=1001, ltp=4.0,
            back_levels=[(3.9, 50.0)], lay_levels=[(4.1, 200.0)],
            total_matched=1055.0,
        )])
        mgr.passive_book.on_tick(t)
        assert len(mgr.bets) == 1               # p1 filled
        assert len(mgr.passive_book.orders) == 1  # p2 still open

        # Race-off cleanup.
        mgr.passive_book.cancel_all("race-off")

        # p1 is in bets (filled), p2 is cancelled.
        assert len(mgr.bets) == 1
        assert mgr.bets[0].matched_stake == pytest.approx(10.0)
        assert len(mgr.passive_book.orders) == 0

        # Settle: runner 1001 wins.
        pnl = mgr.settle_race(winning_selection_ids=1001, market_id="1.99")
        expected = 10.0 * (3.9 - 1.0)  # only the filled passive contributes
        assert pnl == pytest.approx(expected, rel=1e-6)
        assert mgr.bets[0].outcome is BetOutcome.WON


# ── Test 6: Race reset isolation ────────────────────────────────────────────


class TestRaceResetIsolation:
    """After cleanup for race A, race B's state is unaffected."""

    def test_race_b_passive_fills_normally_after_race_a_cleanup(self):
        # --- Race A ---
        mgr_a = _mgr(budget=500.0)
        snap_a = _runner(
            selection_id=1001, ltp=4.0,
            back_levels=[(3.9, 200.0)], total_matched=1000.0,
        )
        mgr_a.passive_book.place(
            snap_a, stake=10.0, side=BetSide.BACK, market_id="race_a", tick_index=0,
        )
        mgr_a.passive_book.cancel_all("race-off")
        mgr_a.settle_race(winning_selection_ids=1001, market_id="race_a")

        # --- Race B (fresh BetManager, as the env does) ---
        mgr_b = _mgr(budget=500.0)
        snap_b = _runner(
            selection_id=2001, ltp=3.0,
            back_levels=[(2.8, 30.0)], lay_levels=[(3.2, 200.0)],
            total_matched=2000.0,
        )
        p_b = mgr_b.passive_book.place(
            snap_b, stake=15.0, side=BetSide.BACK, market_id="race_b", tick_index=0,
        )
        assert p_b is not None

        # Fill in race B.
        t_b = _tick([_runner(
            selection_id=2001, ltp=3.0,
            back_levels=[(2.8, 30.0)], lay_levels=[(3.2, 200.0)],
            total_matched=2031.0,
        )])
        mgr_b.passive_book.on_tick(t_b)
        assert len(mgr_b.bets) == 1
        assert len(mgr_b.passive_book.orders) == 0

        # Settle race B: runner 2001 wins.
        pnl_b = mgr_b.settle_race(winning_selection_ids=2001, market_id="race_b")
        expected = 15.0 * (2.8 - 1.0)
        assert pnl_b == pytest.approx(expected, rel=1e-6)


# ── Test 7: Efficiency penalty interaction ──────────────────────────────────


class TestEfficiencyPenaltyInteraction:
    """Cancelled-at-race-off passives count toward efficiency_penalty × bet_count.

    This is the correct behaviour because in live trading, placing the order
    cost an API call — the friction is real.  Ignoring it would let
    passive-heavy policies look artificially efficient.

    Tested via a minimal BetfairEnv episode to verify the shaped reward term.
    """

    def test_cancelled_passive_counts_in_efficiency_penalty(self):
        start = datetime(2026, 4, 10, 14, 30, tzinfo=timezone.utc)
        r1001 = _runner(
            selection_id=1001, ltp=4.0,
            back_levels=[(3.9, 200.0)], lay_levels=[(4.1, 200.0)],
            total_matched=1000.0,
        )
        r1002 = _runner(
            selection_id=1002, ltp=6.0,
            back_levels=[(5.8, 200.0)], lay_levels=[(6.2, 200.0)],
            total_matched=500.0,
        )
        ticks = [
            _tick([r1001, r1002], in_play=False),
            _tick([r1001, r1002], in_play=False),
            _tick([r1001, r1002], in_play=True),
        ]
        meta = {
            1001: _make_runner_meta(1001, "Horse A"),
            1002: _make_runner_meta(1002, "Horse B"),
        }
        race = Race(
            "1.99999", "Test", start, 1001, ticks, meta,
            winning_selection_ids={1001},
        )
        day = Day(date="2026-04-10", races=[race])

        from env.betfair_env import BetfairEnv

        config = {
            "training": {
                "max_runners": 14,
                "starting_budget": 500.0,
                "max_bets_per_race": 20,
            },
            "reward": {
                "efficiency_penalty": 0.5,
                "early_pick_bonus_min": 1.0,
                "early_pick_bonus_max": 1.0,
                "early_pick_min_seconds": 0,
                "precision_bonus": 0.0,
                "terminal_bonus_weight": 0.0,
                "drawdown_shaping_weight": 0.0,
                "spread_cost_weight": 0.0,
            },
        }
        env = BetfairEnv(day=day, config=config, emit_debug_features=False)
        obs, info = env.reset()

        # Tick 0: place a passive back that won't fill (queue-ahead=200, no volume).
        assert env.bet_manager is not None
        snap = env.day.races[0].ticks[0].runners[0]
        p = env.bet_manager.passive_book.place(
            snap, stake=10.0, side=BetSide.BACK,
            market_id="1.99999", tick_index=0,
        )
        assert p is not None

        # Step through all ticks with no-op action.
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        total_reward = 0.0
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        assert terminated
        # The passive was cancelled at race-off.  With 0 matched bets +
        # 1 cancelled passive, efficiency_cost = 1 × 0.5 = 0.5.
        # No other shaped terms are enabled, so shaped = -0.5.
        # race_pnl = 0 (no matched bets) → reward = 0 + (-0.5) = -0.5.
        assert total_reward == pytest.approx(-0.5, abs=1e-6)


# ── Test 8: raw + shaped ≈ total_reward invariant ───────────────────────────


class TestRewardInvariant:
    """raw + shaped ≈ total_reward when passive cancellation is involved."""

    def test_invariant_with_cancelled_passive(self):
        start = datetime(2026, 4, 10, 14, 30, tzinfo=timezone.utc)
        r1001 = _runner(
            selection_id=1001, ltp=4.0,
            back_levels=[(3.9, 200.0)], lay_levels=[(4.1, 200.0)],
            total_matched=1000.0,
        )
        r1002 = _runner(
            selection_id=1002, ltp=6.0,
            back_levels=[(5.8, 200.0)], lay_levels=[(6.2, 200.0)],
            total_matched=500.0,
        )
        ticks = [
            _tick([r1001, r1002], in_play=False),
            _tick([r1001, r1002], in_play=False),
            _tick([r1001, r1002], in_play=True),
        ]
        meta = {
            1001: _make_runner_meta(1001, "Horse A"),
            1002: _make_runner_meta(1002, "Horse B"),
        }
        race = Race(
            "1.99999", "Test", start, 1001, ticks, meta,
            winning_selection_ids={1001},
        )
        day = Day(date="2026-04-10", races=[race])

        from env.betfair_env import BetfairEnv

        config = {
            "training": {
                "max_runners": 14,
                "starting_budget": 500.0,
                "max_bets_per_race": 20,
            },
            "reward": {
                "efficiency_penalty": 0.01,
                "early_pick_bonus_min": 1.0,
                "early_pick_bonus_max": 1.5,
                "early_pick_min_seconds": 300,
                "precision_bonus": 0.5,
                "terminal_bonus_weight": 0.1,
                "drawdown_shaping_weight": 0.0,
                "spread_cost_weight": 0.1,
            },
        }
        env = BetfairEnv(day=day, config=config, emit_debug_features=False)
        obs, info = env.reset()

        # Place an aggressive back AND a passive that won't fill.
        assert env.bet_manager is not None
        snap = env.day.races[0].ticks[0].runners[0]
        env.bet_manager.place_back(snap, stake=20.0, market_id="1.99999")
        env.bet_manager.passive_book.place(
            snap, stake=10.0, side=BetSide.BACK,
            market_id="1.99999", tick_index=0,
        )

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        total_reward = 0.0
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        assert terminated
        raw = info["raw_pnl_reward"]
        shaped = info["shaped_bonus"]
        assert raw + shaped == pytest.approx(total_reward, abs=1e-6)


# ── Test 9: Aggressive + passive mixed run ──────────────────────────────────


class TestAggressivePassiveMixedRun:
    """A race with both aggressive and passive bets settles correctly end-to-end."""

    def test_mixed_bets_settle_correctly(self):
        mgr = _mgr(budget=500.0)
        snap = _runner(
            selection_id=1001, ltp=4.0,
            back_levels=[(3.9, 50.0)], lay_levels=[(4.1, 200.0)],
            total_matched=1000.0,
        )

        # Aggressive back £20 at 3.9 (best back price).
        agg = mgr.place_back(snap, stake=20.0, market_id="1.99")
        assert agg is not None

        # Passive back £10 at 3.9 — will fill.
        p_fill = mgr.passive_book.place(
            snap, stake=10.0, side=BetSide.BACK, market_id="1.99", tick_index=0,
        )
        assert p_fill is not None

        # Passive back £10 at 3.9 — will NOT fill (depletion: threshold=50+10=60).
        p_cancel = mgr.passive_book.place(
            snap, stake=10.0, side=BetSide.BACK, market_id="1.99", tick_index=1,
        )
        assert p_cancel is not None

        # Traded volume: 55 → p_fill fills (55>=50), p_cancel stays (55<60).
        t = _tick([_runner(
            selection_id=1001, ltp=4.0,
            back_levels=[(3.9, 50.0)], lay_levels=[(4.1, 200.0)],
            total_matched=1055.0,
        )])
        mgr.passive_book.on_tick(t)
        assert len(mgr.bets) == 2  # aggressive + filled passive
        assert len(mgr.passive_book.orders) == 1  # p_cancel still open

        # Race-off cleanup.
        mgr.passive_book.cancel_all("race-off")
        assert len(mgr.passive_book.orders) == 0
        assert p_cancel.cancelled is True

        # Settle: runner 1001 wins.
        pnl = mgr.settle_race(winning_selection_ids=1001, market_id="1.99")

        # Expected P&L: aggressive back 20 @ 3.9 + passive fill 10 @ 3.9.
        expected = 20.0 * (3.9 - 1.0) + 10.0 * (3.9 - 1.0)
        assert pnl == pytest.approx(expected, rel=1e-6)

        # Budget check: started at 500, all matched bets won.
        # Back bets: cost deducted at placement, on win: budget += stake + profit.
        # Cancelled passive: budget restored.
        # After settlement budget should be 500 + agg_profit + passive_profit.
        assert mgr.budget == pytest.approx(
            500.0 + 20.0 * (3.9 - 1.0) + 10.0 * (3.9 - 1.0), rel=1e-6,
        )
