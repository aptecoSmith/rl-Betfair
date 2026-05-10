"""Each-way action surface — predictor-integration Session 04.

Per `plans/predictor-integration/session_prompts/04_each_way_action_surface.md`:

- `BetManager.place_back / place_lay` accept `each_way: bool = False`,
  `each_way_divisor: float | None`, `number_of_places: int | None`.
- When `each_way=True` AND the divisor + places are present, the placed
  `Bet` carries `is_each_way=True` plus the EW metadata fields.
- The existing `settle_race` path (plans/ew-settlement, complete) reads
  `bet.is_each_way` and applies doubled-stake + place-fraction
  settlement; this session DOES NOT modify settlement
  (hard_constraints.md §6).

Default `each_way=False` keeps existing call sites byte-identical
(arb mode, value_win, agent-initiated closes, env force-closes).
"""

from __future__ import annotations

import pytest

from data.episode_builder import PriceSize, RunnerSnap
from env.bet_manager import BetManager, BetSide


def _runner(
    selection_id: int = 1001,
    back_levels: list[tuple[float, float]] | None = None,
    lay_levels: list[tuple[float, float]] | None = None,
    ltp: float | None = None,
) -> RunnerSnap:
    """RunnerSnap helper mirroring tests/test_bet_manager.py::_runner."""
    atb = [PriceSize(p, s) for p, s in (back_levels or [])]
    atl = [PriceSize(p, s) for p, s in (lay_levels or [])]
    if ltp is None:
        sample = []
        if atb:
            sample.append(atb[0].price)
        if atl:
            sample.append(atl[0].price)
        ltp = sum(sample) / len(sample) if sample else 5.0
    return RunnerSnap(
        selection_id=selection_id,
        status="ACTIVE",
        last_traded_price=ltp,
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


# ─── Placement: BACK ─────────────────────────────────────────────────────────


def test_place_back_with_each_way_sets_bet_flag():
    mgr = BetManager(starting_budget=100.0)
    runner = _runner(back_levels=[(5.0, 50.0)])
    bet = mgr.place_back(
        runner,
        stake=10.0,
        each_way=True,
        each_way_divisor=4.0,
        number_of_places=3,
    )
    assert bet is not None
    assert bet.is_each_way is True
    assert bet.each_way_divisor == 4.0
    assert bet.number_of_places == 3
    # effective_place_odds = (price - 1) / divisor + 1 = (5 - 1) / 4 + 1 = 2.0
    assert bet.effective_place_odds == pytest.approx(2.0)


def test_place_back_default_each_way_false():
    """Hard_constraints §1: default-off means existing call sites are
    byte-identical."""
    mgr = BetManager(starting_budget=100.0)
    runner = _runner(back_levels=[(5.0, 50.0)])
    bet = mgr.place_back(runner, stake=10.0)
    assert bet is not None
    assert bet.is_each_way is False
    assert bet.each_way_divisor is None
    assert bet.number_of_places is None
    assert bet.effective_place_odds is None


def test_place_back_each_way_without_divisor_returns_none():
    """A caller that asks for EW without supplying the divisor / places
    is a config error; refuse silently rather than crashing settle_race."""
    mgr = BetManager(starting_budget=100.0)
    runner = _runner(back_levels=[(5.0, 50.0)])
    bet = mgr.place_back(
        runner,
        stake=10.0,
        each_way=True,
        each_way_divisor=None,
        number_of_places=3,
    )
    assert bet is None
    bet = mgr.place_back(
        runner,
        stake=10.0,
        each_way=True,
        each_way_divisor=4.0,
        number_of_places=None,
    )
    assert bet is None


def test_place_back_each_way_doubles_budget_consumption():
    """EW back bets reserve 2x the stake (half on win + half on place)."""
    mgr = BetManager(starting_budget=100.0)
    runner = _runner(back_levels=[(5.0, 50.0)])
    bet = mgr.place_back(
        runner,
        stake=10.0,
        each_way=True,
        each_way_divisor=4.0,
        number_of_places=3,
    )
    assert bet is not None
    # 100 - 2 * 10 = 80
    assert mgr.budget == pytest.approx(80.0)


# ─── Placement: LAY ──────────────────────────────────────────────────────────


def test_place_lay_with_each_way_sets_bet_flag():
    """Lay-side EW symmetry — completeness check; back-only is the
    smoke-cohort default but the lay path must round-trip cleanly."""
    mgr = BetManager(starting_budget=100.0)
    runner = _runner(lay_levels=[(5.0, 50.0)])
    bet = mgr.place_lay(
        runner,
        stake=10.0,
        each_way=True,
        each_way_divisor=4.0,
        number_of_places=3,
    )
    assert bet is not None
    assert bet.side is BetSide.LAY
    assert bet.is_each_way is True
    assert bet.each_way_divisor == 4.0
    assert bet.number_of_places == 3


def test_place_lay_default_each_way_false():
    mgr = BetManager(starting_budget=100.0)
    runner = _runner(lay_levels=[(5.0, 50.0)])
    bet = mgr.place_lay(runner, stake=10.0)
    assert bet is not None
    assert bet.is_each_way is False


# ─── Cross-check: settle_race still handles existing EW bets ────────────────


def test_existing_ew_settlement_unaffected():
    """Hard_constraints §6: settlement was untouched. The pre-Session-04
    EW settlement contract (plans/ew-settlement) still produces the
    documented payouts.

    Spot-check rather than full coverage — `tests/test_bet_manager.py::
    TestEachWaySettlementCorrected` is the load-bearing settlement
    suite.
    """
    mgr = BetManager(starting_budget=100.0)
    runner = _runner(back_levels=[(5.0, 50.0)])
    bet = mgr.place_back(
        runner,
        stake=10.0,
        each_way=True,
        each_way_divisor=4.0,
        number_of_places=3,
    )
    assert bet is not None
    # Settle as winner: EW splits stake across two legs (half each).
    # Win leg: 5 * (5-1) = +20. Place leg: 5 * ((5-1)/4) = +5.
    # Total +25. (Matches plans/ew-settlement formula: half-stake on
    # win leg + half-stake on place leg at fractional odds.)
    # The runner is both the winner and a placer (winners are placers
    # by definition in EW settlement). Pass selection_id in
    # winning_selection_ids; winner_selection_id distinguishes the
    # actual winner for the win-leg payout.
    mgr.settle_race(
        winning_selection_ids={runner.selection_id},
        market_id="",
        each_way_divisor=4.0,
        winner_selection_id=runner.selection_id,
        number_of_places=3,
    )
    settled = [b for b in mgr.bets if b.outcome.name in ("WON", "LOST", "VOID")]
    assert any(b.is_each_way for b in settled)
    pnl = sum(b.pnl for b in settled)
    assert pnl == pytest.approx(25.0)
