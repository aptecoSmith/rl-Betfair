"""
env/bet_manager.py — Track open bets, liability, budget and P&L.

The ``BetManager`` is the single source of truth for an agent's financial
state during an episode (one racing day).  It enforces budget limits, records
matched bets, settles them at race end, and exposes summary metrics.

Key rules
---------
- **Back bet cost** = matched stake (deducted from budget on placement).
- **Lay liability** = matched stake × (price − 1).  This is the maximum the
  layer can lose and is *reserved* from the budget on placement.
- **Budget enforcement**: a bet is rejected (or the stake capped) if it would
  exceed the remaining budget.
- **Settlement**: at race end the winner is known.  Back bets on the winner
  pay ``stake × (price − 1)``; on losers the stake is lost.  Lay bets are
  the inverse — lay on the winner loses the liability, lay on losers keeps
  the stake.

Usage::

    mgr = BetManager(starting_budget=100.0)
    mgr.place_back(runner_snap, stake=10.0)
    mgr.place_lay(runner_snap, stake=5.0)
    mgr.settle_race(winner_selection_id=12345)
    print(mgr.budget, mgr.realised_pnl)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from data.episode_builder import PriceSize, RunnerSnap
from env.order_book import MatchResult, match_back, match_lay


class BetSide(str, Enum):
    BACK = "back"
    LAY = "lay"


class BetOutcome(str, Enum):
    WON = "won"
    LOST = "lost"
    UNSETTLED = "unsettled"


@dataclass(slots=True)
class Bet:
    """A single matched (or partially matched) bet."""

    selection_id: int
    side: BetSide
    requested_stake: float
    matched_stake: float
    average_price: float
    market_id: str
    outcome: BetOutcome = BetOutcome.UNSETTLED
    pnl: float = 0.0

    @property
    def liability(self) -> float:
        """Lay liability — the amount reserved from budget for a lay bet."""
        if self.side is BetSide.LAY:
            return self.matched_stake * (self.average_price - 1.0)
        return 0.0


@dataclass(slots=True)
class BetManager:
    """Tracks bets, budget and P&L across an episode.

    Args:
        starting_budget: Initial budget in £ for the episode.
    """

    starting_budget: float
    budget: float = field(init=False)
    realised_pnl: float = 0.0
    bets: list[Bet] = field(default_factory=list)
    _open_liability: float = 0.0

    def __post_init__(self) -> None:
        self.budget = self.starting_budget

    # ── Public properties ────────────────────────────────────────────────

    @property
    def open_liability(self) -> float:
        """Total liability currently reserved for unsettled lay bets."""
        return self._open_liability

    @property
    def available_budget(self) -> float:
        """Budget available for new bets (after reserving open liability)."""
        return self.budget - self._open_liability

    @property
    def bet_count(self) -> int:
        return len(self.bets)

    @property
    def winning_bets(self) -> int:
        return sum(1 for b in self.bets if b.outcome is BetOutcome.WON)

    # ── Bet placement ────────────────────────────────────────────────────

    def place_back(
        self,
        runner: RunnerSnap,
        stake: float,
        market_id: str = "",
    ) -> Bet | None:
        """Place a back bet, matching against available-to-lay volume.

        The stake is capped to available budget.  Returns ``None`` if no
        volume can be matched or the budget is exhausted.
        """
        capped = min(stake, self.available_budget)
        if capped <= 0.0:
            return None

        result: MatchResult = match_back(runner.available_to_lay, capped)
        if result.matched_stake <= 0.0:
            return None

        bet = Bet(
            selection_id=runner.selection_id,
            side=BetSide.BACK,
            requested_stake=stake,
            matched_stake=result.matched_stake,
            average_price=result.average_price,
            market_id=market_id,
        )
        # Back bets: the stake is the cost (paid up-front).
        self.budget -= result.matched_stake
        self.bets.append(bet)
        return bet

    def place_lay(
        self,
        runner: RunnerSnap,
        stake: float,
        market_id: str = "",
    ) -> Bet | None:
        """Place a lay bet, matching against available-to-back volume.

        The effective cost is the *liability* = stake × (price − 1).  The
        stake is capped so that the liability doesn't exceed the available
        budget.  Returns ``None`` if nothing can be matched.
        """
        if stake <= 0.0:
            return None

        # First, try to match to find out the average price.
        result: MatchResult = match_lay(runner.available_to_back, stake)
        if result.matched_stake <= 0.0:
            return None

        liability = result.matched_stake * (result.average_price - 1.0)

        # If the liability exceeds available budget, scale down the stake.
        if liability > self.available_budget:
            if result.average_price <= 1.0:
                return None
            max_stake = self.available_budget / (result.average_price - 1.0)
            if max_stake <= 0.0:
                return None
            # Re-match with the reduced stake.
            result = match_lay(runner.available_to_back, max_stake)
            if result.matched_stake <= 0.0:
                return None
            liability = result.matched_stake * (result.average_price - 1.0)

        bet = Bet(
            selection_id=runner.selection_id,
            side=BetSide.LAY,
            requested_stake=stake,
            matched_stake=result.matched_stake,
            average_price=result.average_price,
            market_id=market_id,
        )
        # Reserve the liability from the budget.
        self._open_liability += liability
        self.bets.append(bet)
        return bet

    # ── Settlement ───────────────────────────────────────────────────────

    def settle_race(self, winner_selection_id: int, market_id: str = "") -> float:
        """Settle all unsettled bets for a race and return the race P&L.

        Args:
            winner_selection_id: The selection ID of the winning runner.
            market_id: If provided, only settle bets for this market.

        Returns:
            Net P&L for the settled bets.
        """
        race_pnl = 0.0

        for bet in self.bets:
            if bet.outcome is not BetOutcome.UNSETTLED:
                continue
            if market_id and bet.market_id != market_id:
                continue

            won_selection = bet.selection_id == winner_selection_id

            if bet.side is BetSide.BACK:
                if won_selection:
                    # Winner: profit = stake × (price − 1), get stake back.
                    profit = bet.matched_stake * (bet.average_price - 1.0)
                    self.budget += bet.matched_stake + profit
                    bet.pnl = profit
                    bet.outcome = BetOutcome.WON
                else:
                    # Loser: stake already deducted, nothing returned.
                    bet.pnl = -bet.matched_stake
                    bet.outcome = BetOutcome.LOST

            elif bet.side is BetSide.LAY:
                liability = bet.matched_stake * (bet.average_price - 1.0)
                if won_selection:
                    # Runner won → layer loses liability.
                    self.budget -= liability
                    self._open_liability -= liability
                    bet.pnl = -liability
                    bet.outcome = BetOutcome.LOST
                else:
                    # Runner lost → layer keeps the backer's stake.
                    self.budget += bet.matched_stake
                    self._open_liability -= liability
                    bet.pnl = bet.matched_stake
                    bet.outcome = BetOutcome.WON

            race_pnl += bet.pnl

        self.realised_pnl += race_pnl
        return race_pnl

    def unsettled_bets(self, market_id: str = "") -> list[Bet]:
        """Return all unsettled bets, optionally filtered by market."""
        return [
            b
            for b in self.bets
            if b.outcome is BetOutcome.UNSETTLED
            and (not market_id or b.market_id == market_id)
        ]

    def race_bets(self, market_id: str) -> list[Bet]:
        """Return all bets for a specific market/race."""
        return [b for b in self.bets if b.market_id == market_id]
