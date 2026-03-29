"""Tests for env/bet_manager.py — bet tracking, liability, P&L, budget."""

from __future__ import annotations

import pytest

from data.episode_builder import PriceSize, RunnerSnap
from env.bet_manager import Bet, BetManager, BetOutcome, BetSide


# ── Helpers ──────────────────────────────────────────────────────────────────


def _runner(
    selection_id: int = 1001,
    back_levels: list[tuple[float, float]] | None = None,
    lay_levels: list[tuple[float, float]] | None = None,
) -> RunnerSnap:
    """Create a RunnerSnap with specified order book levels.

    Args:
        back_levels: (price, size) tuples for available_to_back.
        lay_levels: (price, size) tuples for available_to_lay.
    """
    atb = [PriceSize(p, s) for p, s in (back_levels or [])]
    atl = [PriceSize(p, s) for p, s in (lay_levels or [])]
    return RunnerSnap(
        selection_id=selection_id,
        status="ACTIVE",
        last_traded_price=3.0,
        total_matched=1000.0,
        starting_price_near=0.0,
        starting_price_far=0.0,
        adjustment_factor=None,
        bsp=None,
        sort_priority=1,
        removal_date=None,
        available_to_back=atb,
        available_to_lay=atl,
    )


# ── BetManager initialisation ───────────────────────────────────────────────


class TestBetManagerInit:
    def test_initial_state(self):
        mgr = BetManager(starting_budget=100.0)
        assert mgr.budget == 100.0
        assert mgr.available_budget == 100.0
        assert mgr.realised_pnl == 0.0
        assert mgr.bet_count == 0
        assert mgr.winning_bets == 0
        assert mgr.open_liability == 0.0
        assert mgr.bets == []

    def test_custom_budget(self):
        mgr = BetManager(starting_budget=500.0)
        assert mgr.budget == 500.0
        assert mgr.available_budget == 500.0


# ── Back bet placement ───────────────────────────────────────────────────────


class TestPlaceBack:
    def test_simple_back(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(lay_levels=[(3.0, 50.0)])

        bet = mgr.place_back(runner, stake=10.0)

        assert bet is not None
        assert bet.side is BetSide.BACK
        assert bet.matched_stake == 10.0
        assert bet.average_price == 3.0
        assert bet.outcome is BetOutcome.UNSETTLED
        assert mgr.budget == pytest.approx(90.0)
        assert mgr.bet_count == 1

    def test_back_capped_to_budget(self):
        mgr = BetManager(starting_budget=20.0)
        runner = _runner(lay_levels=[(3.0, 100.0)])

        bet = mgr.place_back(runner, stake=50.0)

        assert bet is not None
        assert bet.matched_stake == 20.0
        assert bet.requested_stake == 50.0
        assert mgr.budget == pytest.approx(0.0)

    def test_back_partial_fill(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(lay_levels=[(3.0, 5.0)])

        bet = mgr.place_back(runner, stake=20.0)

        assert bet is not None
        assert bet.matched_stake == 5.0
        assert mgr.budget == pytest.approx(95.0)

    def test_back_no_liquidity(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(lay_levels=[])

        bet = mgr.place_back(runner, stake=10.0)

        assert bet is None
        assert mgr.budget == 100.0
        assert mgr.bet_count == 0

    def test_back_zero_budget(self):
        mgr = BetManager(starting_budget=0.0)
        runner = _runner(lay_levels=[(3.0, 50.0)])

        bet = mgr.place_back(runner, stake=10.0)

        assert bet is None

    def test_back_multi_level_fill(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(lay_levels=[(3.0, 5.0), (3.5, 10.0), (4.0, 20.0)])

        bet = mgr.place_back(runner, stake=12.0)

        assert bet is not None
        assert bet.matched_stake == 12.0
        assert mgr.budget == pytest.approx(88.0)

    def test_back_with_market_id(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(lay_levels=[(3.0, 50.0)])

        bet = mgr.place_back(runner, stake=10.0, market_id="1.234")

        assert bet is not None
        assert bet.market_id == "1.234"

    def test_multiple_backs_deplete_budget(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(lay_levels=[(3.0, 100.0)])

        mgr.place_back(runner, stake=40.0)
        mgr.place_back(runner, stake=40.0)

        assert mgr.budget == pytest.approx(20.0)
        assert mgr.bet_count == 2

        bet3 = mgr.place_back(runner, stake=30.0)
        assert bet3 is not None
        assert bet3.matched_stake == 20.0
        assert mgr.budget == pytest.approx(0.0)

    def test_back_zero_stake(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(lay_levels=[(3.0, 50.0)])

        bet = mgr.place_back(runner, stake=0.0)
        assert bet is None


# ── Lay bet placement ────────────────────────────────────────────────────────


class TestPlaceLay:
    def test_simple_lay(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(selection_id=2001, back_levels=[(4.0, 50.0)])

        bet = mgr.place_lay(runner, stake=10.0)

        assert bet is not None
        assert bet.side is BetSide.LAY
        assert bet.matched_stake == 10.0
        assert bet.average_price == 4.0
        # Liability = 10 × (4 − 1) = 30
        assert bet.liability == pytest.approx(30.0)
        assert mgr.open_liability == pytest.approx(30.0)
        # Budget unchanged; liability reserved internally.
        assert mgr.budget == 100.0
        assert mgr.available_budget == pytest.approx(70.0)

    def test_lay_capped_by_liability(self):
        mgr = BetManager(starting_budget=50.0)
        # Price 6.0: liability per £1 stake = 5.0
        # Budget 50 → max stake = 50/5 = 10
        runner = _runner(back_levels=[(6.0, 100.0)])

        bet = mgr.place_lay(runner, stake=20.0)

        assert bet is not None
        assert bet.matched_stake == pytest.approx(10.0)
        assert bet.liability == pytest.approx(50.0)
        assert mgr.available_budget == pytest.approx(0.0)

    def test_lay_no_liquidity(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(back_levels=[])

        bet = mgr.place_lay(runner, stake=10.0)

        assert bet is None
        assert mgr.open_liability == 0.0

    def test_lay_zero_budget(self):
        mgr = BetManager(starting_budget=0.0)
        runner = _runner(back_levels=[(3.0, 50.0)])

        bet = mgr.place_lay(runner, stake=10.0)

        assert bet is None

    def test_lay_partial_fill(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(back_levels=[(4.0, 3.0)])

        bet = mgr.place_lay(runner, stake=10.0)

        assert bet is not None
        assert bet.matched_stake == 3.0
        assert bet.liability == pytest.approx(9.0)
        assert mgr.open_liability == pytest.approx(9.0)

    def test_lay_zero_stake(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(back_levels=[(4.0, 50.0)])

        bet = mgr.place_lay(runner, stake=0.0)
        assert bet is None

    def test_multiple_lays_accumulate_liability(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(back_levels=[(3.0, 100.0)])

        # Lay £10 at 3.0 → liability = 10×2 = 20
        mgr.place_lay(runner, stake=10.0)
        assert mgr.open_liability == pytest.approx(20.0)
        assert mgr.available_budget == pytest.approx(80.0)

        # Lay another £10 → liability += 20
        mgr.place_lay(runner, stake=10.0)
        assert mgr.open_liability == pytest.approx(40.0)
        assert mgr.available_budget == pytest.approx(60.0)


# ── Bet.liability property ───────────────────────────────────────────────────


class TestBetLiability:
    def test_back_bet_has_zero_liability(self):
        bet = Bet(
            selection_id=1,
            side=BetSide.BACK,
            requested_stake=10.0,
            matched_stake=10.0,
            average_price=3.0,
            market_id="",
        )
        assert bet.liability == 0.0

    def test_lay_bet_liability(self):
        bet = Bet(
            selection_id=1,
            side=BetSide.LAY,
            requested_stake=10.0,
            matched_stake=10.0,
            average_price=5.0,
            market_id="",
        )
        # 10 × (5 − 1) = 40
        assert bet.liability == pytest.approx(40.0)


# ── Settlement ───────────────────────────────────────────────────────────────


class TestSettlement:
    def test_back_winner(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])

        mgr.place_back(runner, stake=10.0, market_id="m1")
        # Budget: 100 − 10 = 90

        pnl = mgr.settle_race(winning_selection_ids=1001, market_id="m1")

        # Profit: 10 × (3 − 1) = 20
        assert pnl == pytest.approx(20.0)
        assert mgr.budget == pytest.approx(120.0)  # 90 + 10 (stake back) + 20
        assert mgr.realised_pnl == pytest.approx(20.0)
        assert mgr.bets[0].outcome is BetOutcome.WON

    def test_back_loser(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])

        mgr.place_back(runner, stake=10.0, market_id="m1")

        pnl = mgr.settle_race(winning_selection_ids=9999, market_id="m1")

        assert pnl == pytest.approx(-10.0)
        assert mgr.budget == pytest.approx(90.0)  # Stake was already deducted
        assert mgr.realised_pnl == pytest.approx(-10.0)
        assert mgr.bets[0].outcome is BetOutcome.LOST

    def test_lay_winner_loses_liability(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(selection_id=2001, back_levels=[(4.0, 50.0)])

        mgr.place_lay(runner, stake=10.0, market_id="m1")
        # Liability = 10 × (4 − 1) = 30
        assert mgr.open_liability == pytest.approx(30.0)

        pnl = mgr.settle_race(winning_selection_ids=2001, market_id="m1")

        # Runner won → layer loses liability
        assert pnl == pytest.approx(-30.0)
        assert mgr.budget == pytest.approx(70.0)  # 100 − 30
        assert mgr.open_liability == pytest.approx(0.0)
        assert mgr.bets[0].outcome is BetOutcome.LOST

    def test_lay_loser_keeps_stake(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(selection_id=2001, back_levels=[(4.0, 50.0)])

        mgr.place_lay(runner, stake=10.0, market_id="m1")

        pnl = mgr.settle_race(winning_selection_ids=9999, market_id="m1")

        # Runner lost → layer keeps the backer's stake
        assert pnl == pytest.approx(10.0)
        assert mgr.budget == pytest.approx(110.0)
        assert mgr.open_liability == pytest.approx(0.0)
        assert mgr.bets[0].outcome is BetOutcome.WON

    def test_settle_multiple_bets_same_race(self):
        mgr = BetManager(starting_budget=100.0)
        winner = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])
        loser = _runner(selection_id=1002, lay_levels=[(5.0, 50.0)])

        mgr.place_back(winner, stake=10.0, market_id="m1")
        mgr.place_back(loser, stake=10.0, market_id="m1")
        # Budget: 100 − 10 − 10 = 80

        pnl = mgr.settle_race(winning_selection_ids=1001, market_id="m1")

        # Winner: +20, Loser: −10 → net +10
        assert pnl == pytest.approx(10.0)
        assert mgr.budget == pytest.approx(110.0)

    def test_settle_only_affects_specified_market(self):
        mgr = BetManager(starting_budget=100.0)
        r1 = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])
        r2 = _runner(selection_id=2001, lay_levels=[(4.0, 50.0)])

        mgr.place_back(r1, stake=10.0, market_id="m1")
        mgr.place_back(r2, stake=10.0, market_id="m2")

        pnl = mgr.settle_race(winning_selection_ids=1001, market_id="m1")

        # Only m1 settled
        assert mgr.bets[0].outcome is BetOutcome.WON
        assert mgr.bets[1].outcome is BetOutcome.UNSETTLED
        assert pnl == pytest.approx(20.0)

    def test_settle_without_market_id_settles_all(self):
        mgr = BetManager(starting_budget=100.0)
        r1 = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])
        r2 = _runner(selection_id=1002, lay_levels=[(4.0, 50.0)])

        mgr.place_back(r1, stake=10.0, market_id="m1")
        mgr.place_back(r2, stake=10.0, market_id="m1")

        pnl = mgr.settle_race(winning_selection_ids=1001)

        assert mgr.bets[0].outcome is BetOutcome.WON
        assert mgr.bets[1].outcome is BetOutcome.LOST

    def test_double_settle_is_noop(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])

        mgr.place_back(runner, stake=10.0, market_id="m1")
        pnl1 = mgr.settle_race(winning_selection_ids=1001, market_id="m1")
        budget_after = mgr.budget

        pnl2 = mgr.settle_race(winning_selection_ids=1001, market_id="m1")

        assert pnl2 == 0.0
        assert mgr.budget == budget_after

    def test_mixed_back_and_lay_settlement(self):
        mgr = BetManager(starting_budget=100.0)
        r_back = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])
        r_lay = _runner(selection_id=1002, back_levels=[(4.0, 50.0)])

        mgr.place_back(r_back, stake=10.0, market_id="m1")
        mgr.place_lay(r_lay, stake=10.0, market_id="m1")
        # Budget: 100 − 10 = 90 (back cost). Lay liability = 30.
        assert mgr.budget == pytest.approx(90.0)
        assert mgr.open_liability == pytest.approx(30.0)
        assert mgr.available_budget == pytest.approx(60.0)

        pnl = mgr.settle_race(winning_selection_ids=1001, market_id="m1")

        # Back on 1001 wins: +20
        # Lay on 1002 — runner lost → layer keeps £10
        # Net: +30
        assert pnl == pytest.approx(30.0)
        assert mgr.open_liability == pytest.approx(0.0)


# ── Budget enforcement across races ──────────────────────────────────────────


class TestBudgetAcrossRaces:
    def test_budget_carries_across_races(self):
        mgr = BetManager(starting_budget=100.0)
        runner1 = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])
        runner2 = _runner(selection_id=2001, lay_levels=[(4.0, 50.0)])

        # Race 1: back £10, wins
        mgr.place_back(runner1, stake=10.0, market_id="race1")
        mgr.settle_race(winning_selection_ids=1001, market_id="race1")
        assert mgr.budget == pytest.approx(120.0)

        # Race 2: can now bet with the profit
        bet = mgr.place_back(runner2, stake=25.0, market_id="race2")
        assert bet is not None
        assert bet.matched_stake == 25.0
        assert mgr.budget == pytest.approx(95.0)

    def test_losing_race_reduces_budget_for_next(self):
        mgr = BetManager(starting_budget=100.0)
        runner1 = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])
        runner2 = _runner(selection_id=2001, lay_levels=[(4.0, 50.0)])

        # Race 1: back £50, loses
        mgr.place_back(runner1, stake=50.0, market_id="race1")
        mgr.settle_race(winning_selection_ids=9999, market_id="race1")
        assert mgr.budget == pytest.approx(50.0)

        # Race 2: budget is now only 50
        bet = mgr.place_back(runner2, stake=80.0, market_id="race2")
        assert bet is not None
        assert bet.matched_stake == 50.0  # Capped to budget


# ── unsettled_bets / race_bets helpers ───────────────────────────────────────


class TestHelpers:
    def test_unsettled_bets(self):
        mgr = BetManager(starting_budget=100.0)
        r1 = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])
        r2 = _runner(selection_id=2001, lay_levels=[(4.0, 50.0)])

        mgr.place_back(r1, stake=10.0, market_id="m1")
        mgr.place_back(r2, stake=10.0, market_id="m2")

        unsettled = mgr.unsettled_bets()
        assert len(unsettled) == 2

        mgr.settle_race(winning_selection_ids=1001, market_id="m1")

        unsettled = mgr.unsettled_bets()
        assert len(unsettled) == 1
        assert unsettled[0].market_id == "m2"

    def test_unsettled_bets_filtered_by_market(self):
        mgr = BetManager(starting_budget=100.0)
        r1 = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])

        mgr.place_back(r1, stake=10.0, market_id="m1")
        mgr.place_back(r1, stake=10.0, market_id="m2")

        assert len(mgr.unsettled_bets(market_id="m1")) == 1
        assert len(mgr.unsettled_bets(market_id="m2")) == 1
        assert len(mgr.unsettled_bets(market_id="m3")) == 0

    def test_race_bets(self):
        mgr = BetManager(starting_budget=100.0)
        r = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])

        mgr.place_back(r, stake=10.0, market_id="m1")
        mgr.place_back(r, stake=5.0, market_id="m1")
        mgr.place_back(r, stake=5.0, market_id="m2")

        assert len(mgr.race_bets("m1")) == 2
        assert len(mgr.race_bets("m2")) == 1

    def test_winning_bets_count(self):
        mgr = BetManager(starting_budget=100.0)
        winner = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])
        loser = _runner(selection_id=1002, lay_levels=[(4.0, 50.0)])

        mgr.place_back(winner, stake=10.0, market_id="m1")
        mgr.place_back(loser, stake=10.0, market_id="m1")

        assert mgr.winning_bets == 0

        mgr.settle_race(winning_selection_ids=1001, market_id="m1")

        assert mgr.winning_bets == 1


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_lay_at_price_1_has_zero_liability(self):
        """Edge: price=1.0 means liability = stake×0 = 0."""
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(back_levels=[(1.01, 50.0)])

        bet = mgr.place_lay(runner, stake=10.0)

        assert bet is not None
        assert bet.liability == pytest.approx(0.1)

    def test_back_and_lay_same_runner(self):
        """Agent backs and lays the same runner — hedging."""
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(
            selection_id=1001,
            back_levels=[(4.0, 50.0)],
            lay_levels=[(3.0, 50.0)],
        )

        mgr.place_back(runner, stake=10.0, market_id="m1")
        mgr.place_lay(runner, stake=10.0, market_id="m1")

        # Back cost: 10. Lay liability: 10×3=30.
        assert mgr.budget == pytest.approx(90.0)
        assert mgr.open_liability == pytest.approx(30.0)

        # Runner wins:
        # Back wins: +20. Lay loses: -30. Net: -10
        pnl = mgr.settle_race(winning_selection_ids=1001, market_id="m1")
        assert pnl == pytest.approx(-10.0)

    def test_back_and_lay_same_runner_loser(self):
        """Agent backs and lays the same runner, and it loses."""
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(
            selection_id=1001,
            back_levels=[(4.0, 50.0)],
            lay_levels=[(3.0, 50.0)],
        )

        mgr.place_back(runner, stake=10.0, market_id="m1")
        mgr.place_lay(runner, stake=10.0, market_id="m1")

        # Runner loses:
        # Back loses: -10. Lay wins: +10. Net: 0
        pnl = mgr.settle_race(winning_selection_ids=9999, market_id="m1")
        assert pnl == pytest.approx(0.0)

    def test_very_large_lay_capped(self):
        """Lay stake much larger than budget can support."""
        mgr = BetManager(starting_budget=10.0)
        runner = _runner(back_levels=[(11.0, 1000.0)])
        # Liability per £1 = 10.0. Budget = 10 → max stake = 1.0

        bet = mgr.place_lay(runner, stake=500.0)

        assert bet is not None
        assert bet.matched_stake == pytest.approx(1.0)
        assert bet.liability == pytest.approx(10.0)
        assert mgr.available_budget == pytest.approx(0.0)

    def test_full_episode_flow(self):
        """Simulate a mini day: 2 races, budget carries."""
        mgr = BetManager(starting_budget=100.0)

        # --- Race 1 ---
        r1_winner = _runner(selection_id=101, lay_levels=[(2.5, 50.0)])
        r1_loser = _runner(selection_id=102, lay_levels=[(6.0, 50.0)])

        mgr.place_back(r1_winner, stake=20.0, market_id="race1")
        mgr.place_back(r1_loser, stake=5.0, market_id="race1")
        # Budget: 100 − 20 − 5 = 75

        pnl1 = mgr.settle_race(winning_selection_ids=101, market_id="race1")
        # Winner: 20×1.5=30 profit. Loser: −5. Net: +25
        assert pnl1 == pytest.approx(25.0)
        assert mgr.budget == pytest.approx(125.0)

        # --- Race 2 ---
        r2_loser1 = _runner(selection_id=201, lay_levels=[(3.0, 50.0)])
        r2_lay = _runner(selection_id=202, back_levels=[(4.0, 50.0)])

        mgr.place_back(r2_loser1, stake=30.0, market_id="race2")
        mgr.place_lay(r2_lay, stake=10.0, market_id="race2")

        pnl2 = mgr.settle_race(winning_selection_ids=999, market_id="race2")
        # Back 201 loses: −30. Lay 202 wins (runner lost): +10. Net: −20
        assert pnl2 == pytest.approx(-20.0)
        assert mgr.budget == pytest.approx(105.0)

        # --- Overall ---
        assert mgr.realised_pnl == pytest.approx(5.0)
        assert mgr.bet_count == 4
        assert mgr.winning_bets == 2  # r1_winner back + r2_lay


# ── EACH_WAY / PLACED settlement ────────────────────────────────────────────


class TestEachWaySettlement:
    """Tests for EACH_WAY markets where PLACED runners also pay out."""

    def test_placed_runner_back_wins(self):
        """A back bet on a PLACED runner should pay out at full price."""
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])
        mgr.place_back(runner, stake=10.0, market_id="m1")
        # Runner placed (not the winner) — should still pay out
        pnl = mgr.settle_race({2001, 1001}, market_id="m1")
        assert pnl == pytest.approx(20.0)  # 10 × (3.0 - 1) = 20
        assert mgr.bets[0].outcome is BetOutcome.WON

    def test_placed_runner_lay_loses(self):
        """A lay bet on a PLACED runner should lose (runner placed)."""
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(selection_id=1001, back_levels=[(4.0, 50.0)])
        mgr.place_lay(runner, stake=10.0, market_id="m1")
        # Runner placed — layer loses
        pnl = mgr.settle_race({2001, 1001}, market_id="m1")
        liability = 10.0 * (4.0 - 1.0)  # = 30
        assert pnl == pytest.approx(-liability)
        assert mgr.bets[0].outcome is BetOutcome.LOST

    def test_non_placed_runner_loses(self):
        """Back bet on non-placed runner still loses in EACH_WAY."""
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(selection_id=3001, lay_levels=[(5.0, 50.0)])
        mgr.place_back(runner, stake=10.0, market_id="m1")
        # Winners are {1001, 2001} — 3001 is a loser
        pnl = mgr.settle_race({1001, 2001}, market_id="m1")
        assert pnl == pytest.approx(-10.0)
        assert mgr.bets[0].outcome is BetOutcome.LOST

    def test_multiple_placed_runners_all_pay(self):
        """Multiple bets on different placed runners should all win."""
        mgr = BetManager(starting_budget=100.0)
        r1 = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])
        r2 = _runner(selection_id=1002, lay_levels=[(5.0, 50.0)])
        mgr.place_back(r1, stake=10.0, market_id="m1")
        mgr.place_back(r2, stake=10.0, market_id="m1")
        # Both placed
        pnl = mgr.settle_race({1001, 1002, 1003}, market_id="m1")
        # r1: 10×2=20, r2: 10×4=40
        assert pnl == pytest.approx(60.0)
        assert mgr.winning_bets == 2

    def test_single_int_still_works(self):
        """Passing a single int (backward compat) should still work."""
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])
        mgr.place_back(runner, stake=10.0, market_id="m1")
        pnl = mgr.settle_race(1001, market_id="m1")
        assert pnl == pytest.approx(20.0)


# ── Commission tests ────────────────────────────────────────────────────────


class TestCommission:
    """Tests for Betfair commission on winning bets."""

    def test_back_winner_with_commission(self):
        """Commission should be deducted from profit on a winning back bet."""
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])
        mgr.place_back(runner, stake=10.0, market_id="m1")
        # Budget: 90
        pnl = mgr.settle_race(1001, market_id="m1", commission=0.05)
        # Gross profit: 10 × 2 = 20. Commission: 20 × 0.05 = 1. Net: 19
        assert pnl == pytest.approx(19.0)
        assert mgr.budget == pytest.approx(90.0 + 10.0 + 19.0)  # 119

    def test_back_loser_no_commission(self):
        """No commission on losing bets."""
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])
        mgr.place_back(runner, stake=10.0, market_id="m1")
        pnl = mgr.settle_race(9999, market_id="m1", commission=0.05)
        assert pnl == pytest.approx(-10.0)

    def test_lay_winner_no_commission_on_loss(self):
        """Lay bet losing (runner won) — no commission, just lose liability."""
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(selection_id=1001, back_levels=[(4.0, 50.0)])
        mgr.place_lay(runner, stake=10.0, market_id="m1")
        pnl = mgr.settle_race(1001, market_id="m1", commission=0.05)
        # Layer loses liability: 10 × 3 = 30 (no commission on losses)
        assert pnl == pytest.approx(-30.0)

    def test_lay_loser_with_commission(self):
        """Lay bet winning (runner lost) — commission on the profit."""
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(selection_id=1001, back_levels=[(4.0, 50.0)])
        mgr.place_lay(runner, stake=10.0, market_id="m1")
        pnl = mgr.settle_race(9999, market_id="m1", commission=0.05)
        # Gross profit: 10 (backer's stake). Commission: 10 × 0.05 = 0.5. Net: 9.5
        assert pnl == pytest.approx(9.5)

    def test_zero_commission(self):
        """With zero commission, results match the old behaviour."""
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])
        mgr.place_back(runner, stake=10.0, market_id="m1")
        pnl = mgr.settle_race(1001, market_id="m1", commission=0.0)
        assert pnl == pytest.approx(20.0)

    def test_commission_with_placed_runners(self):
        """Commission applies to placed runner payouts too."""
        mgr = BetManager(starting_budget=100.0)
        r1 = _runner(selection_id=1001, lay_levels=[(5.0, 50.0)])
        mgr.place_back(r1, stake=10.0, market_id="m1")
        # Runner 1001 placed
        pnl = mgr.settle_race({1001, 2001}, market_id="m1", commission=0.10)
        # Gross profit: 10 × 4 = 40. Commission: 40 × 0.10 = 4. Net: 36
        assert pnl == pytest.approx(36.0)


# ── Session 4.10 — Position tracking & race bet count ──────────────────────


class TestPositionTracking:
    """Tests for BetManager.get_positions() and race_bet_count()."""

    def test_get_positions_empty(self):
        mgr = BetManager(starting_budget=100.0)
        positions = mgr.get_positions("m1")
        assert positions == {}

    def test_get_positions_single_back(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])
        mgr.place_back(runner, stake=10.0, market_id="m1")

        positions = mgr.get_positions("m1")
        assert 1001 in positions
        assert positions[1001]["back_exposure"] == pytest.approx(10.0)
        assert positions[1001]["lay_exposure"] == pytest.approx(0.0)
        assert positions[1001]["bet_count"] == 1

    def test_get_positions_accumulated_back(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])
        mgr.place_back(runner, stake=10.0, market_id="m1")
        mgr.place_back(runner, stake=5.0, market_id="m1")

        positions = mgr.get_positions("m1")
        assert positions[1001]["back_exposure"] == pytest.approx(15.0)
        assert positions[1001]["bet_count"] == 2

    def test_get_positions_lay(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(selection_id=1001, back_levels=[(4.0, 50.0)])
        mgr.place_lay(runner, stake=10.0, market_id="m1")

        positions = mgr.get_positions("m1")
        assert 1001 in positions
        assert positions[1001]["back_exposure"] == pytest.approx(0.0)
        # Liability = 10 × (4 - 1) = 30
        assert positions[1001]["lay_exposure"] == pytest.approx(30.0)
        assert positions[1001]["bet_count"] == 1

    def test_get_positions_mixed(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(
            selection_id=1001,
            back_levels=[(4.0, 50.0)],
            lay_levels=[(3.0, 50.0)],
        )
        mgr.place_back(runner, stake=10.0, market_id="m1")
        mgr.place_lay(runner, stake=5.0, market_id="m1")

        positions = mgr.get_positions("m1")
        assert positions[1001]["back_exposure"] == pytest.approx(10.0)
        # Lay liability = 5 × (4 - 1) = 15
        assert positions[1001]["lay_exposure"] == pytest.approx(15.0)
        assert positions[1001]["bet_count"] == 2

    def test_get_positions_filtered_by_market(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])
        mgr.place_back(runner, stake=10.0, market_id="m1")
        mgr.place_back(runner, stake=5.0, market_id="m2")

        pos_m1 = mgr.get_positions("m1")
        pos_m2 = mgr.get_positions("m2")
        assert pos_m1[1001]["back_exposure"] == pytest.approx(10.0)
        assert pos_m2[1001]["back_exposure"] == pytest.approx(5.0)

    def test_race_bet_count(self):
        mgr = BetManager(starting_budget=100.0)
        runner = _runner(selection_id=1001, lay_levels=[(3.0, 50.0)])
        mgr.place_back(runner, stake=10.0, market_id="m1")
        mgr.place_back(runner, stake=5.0, market_id="m1")
        mgr.place_back(runner, stake=5.0, market_id="m2")

        assert mgr.race_bet_count("m1") == 2
        assert mgr.race_bet_count("m2") == 1
        assert mgr.race_bet_count("m3") == 0
