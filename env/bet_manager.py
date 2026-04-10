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
from typing import TYPE_CHECKING

from data.episode_builder import PriceSize, RunnerSnap
from env.exchange_matcher import DEFAULT_MATCHER, ExchangeMatcher, MatchResult

if TYPE_CHECKING:
    from data.episode_builder import Tick


class BetSide(str, Enum):
    BACK = "back"
    LAY = "lay"


class BetOutcome(str, Enum):
    WON = "won"
    LOST = "lost"
    VOID = "void"
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
    tick_index: int = -1  # index into Race.ticks where bet was placed (-1 = not recorded)
    ltp_at_placement: float = 0.0  # runner's last traded price when the bet was placed (Session 23 — used by spread_cost shaping)

    @property
    def liability(self) -> float:
        """Lay liability — the amount reserved from budget for a lay bet."""
        if self.side is BetSide.LAY:
            return self.matched_stake * (self.average_price - 1.0)
        return 0.0


@dataclass(slots=True)
class PassiveOrder:
    """A resting (unmatched) order waiting in the queue.

    Queue position is estimated by snapshotting the own-side top-of-book size
    at placement time (``queue_ahead_at_placement``) and accumulating traded
    volume on that runner since placement (``traded_volume_since_placement``).
    Fill logic is not implemented here — that lands in session 26.

    Fields marked "reserved for session 26/29" are populated as 0 / False
    now so the struct is stable across sessions.
    """

    selection_id: int
    side: BetSide
    price: float                          # price the order rests at
    requested_stake: float
    queue_ahead_at_placement: float       # own-side top-level size at placement
    placed_tick_index: int
    market_id: str
    traded_volume_since_placement: float = 0.0
    matched_stake: float = 0.0            # reserved for session 26
    cancelled: bool = False               # reserved for session 29

    def to_dict(self) -> dict:
        return {
            "selection_id": self.selection_id,
            "side": self.side.value,
            "price": self.price,
            "requested_stake": self.requested_stake,
            "queue_ahead_at_placement": self.queue_ahead_at_placement,
            "placed_tick_index": self.placed_tick_index,
            "market_id": self.market_id,
            "traded_volume_since_placement": self.traded_volume_since_placement,
            "matched_stake": self.matched_stake,
            "cancelled": self.cancelled,
        }


class PassiveOrderBook:
    """Bookkeeping container for resting (passive) orders.

    Owned by :class:`BetManager` as ``self.passive_book``. Responsible for:
    - Snapshotting queue-ahead at placement time.
    - Accumulating traded-volume deltas per runner across ticks.

    Fill logic, budget reservation, and cancellation all land in later
    sessions (26 and 29). This class is deliberately minimal.

    The matcher stays stateless and is only consulted to apply the same
    junk-filter as aggressive orders, so refused passive placements are
    consistent with refused aggressive placements.
    """

    def __init__(self, matcher: ExchangeMatcher = DEFAULT_MATCHER) -> None:
        self._matcher = matcher
        self._orders: list[PassiveOrder] = []
        # Per-runner last-seen total_matched value, for computing deltas.
        # Populated on first on_tick call; reset when the PassiveOrderBook
        # is replaced (fresh BetManager per race).
        self._last_total_matched: dict[int, float] = {}

    @property
    def orders(self) -> list[PassiveOrder]:
        """All passive orders in this race (including filled/cancelled in future sessions)."""
        return list(self._orders)

    def place(
        self,
        runner: RunnerSnap,
        stake: float,
        side: BetSide,
        market_id: str,
        tick_index: int,
    ) -> PassiveOrder | None:
        """Record a resting order at the own-side best price.

        For a passive back the order rests in the available_to_back queue
        (the price other backers are offering); for a passive lay it rests
        in the available_to_lay queue.

        The best post-filter price is found via the same junk filter the
        aggressive matcher uses. If no valid level exists (empty ladder or
        all levels outside ±``max_price_deviation_pct`` from LTP), returns
        ``None`` — passive placement into filtered-out levels silently
        succeeds in no simulator; it must fail explicitly here.

        Budget is **not** affected. That is session 26's job.
        """
        ltp = runner.last_traded_price
        if ltp is None or ltp <= 0.0:
            return None

        if side is BetSide.BACK:
            # Passive back: order rests on the back side (we are offering to back)
            levels = runner.available_to_back
            lower_is_better = False  # highest back price is best for a backer
        else:
            # Passive lay: order rests on the lay side
            levels = runner.available_to_lay
            lower_is_better = True   # lowest lay price is best for a layer

        top_price = self._matcher.pick_top_price(
            levels,
            reference_price=ltp,
            lower_is_better=lower_is_better,
        )
        if top_price is None:
            return None

        # Snapshot the size at that level for queue_ahead_at_placement.
        lo = ltp * (1.0 - self._matcher.max_price_deviation_pct)
        hi = ltp * (1.0 + self._matcher.max_price_deviation_pct)
        filtered = [
            lv for lv in levels
            if lv.price > 0.0 and lv.size > 0.0 and lo <= lv.price <= hi
        ]
        if not filtered:
            return None
        if lower_is_better:
            top_level = min(filtered, key=lambda lv: lv.price)
        else:
            top_level = max(filtered, key=lambda lv: lv.price)

        order = PassiveOrder(
            selection_id=runner.selection_id,
            side=side,
            price=top_level.price,
            requested_stake=stake,
            queue_ahead_at_placement=top_level.size,
            placed_tick_index=tick_index,
            market_id=market_id,
        )
        self._orders.append(order)
        # Seed the volume baseline so the first on_tick call computes the
        # delta from the moment of placement, not from first sight.
        # Per open_questions.md Q4: "compute at runtime by snapshotting at
        # placement and subtracting."
        sid = runner.selection_id
        if sid not in self._last_total_matched:
            self._last_total_matched[sid] = runner.total_matched
        return order

    def on_tick(self, tick: "Tick") -> None:
        """Accumulate traded-volume deltas for all open passive orders.

        For each runner in the tick, computes the delta of
        ``RunnerSnap.total_matched`` since the last time we saw this runner
        and adds it to ``traded_volume_since_placement`` for every passive
        order on that runner.

        Volume from runners that don't match any passive order's
        ``selection_id`` is ignored, so passive orders on runner A are
        unaffected by trading on runner B.
        """
        runner_by_sid = {r.selection_id: r for r in tick.runners}

        # Collect which selection_ids have open passive orders.
        active_sids: set[int] = {o.selection_id for o in self._orders}

        for sid in active_sids:
            snap = runner_by_sid.get(sid)
            if snap is None:
                continue
            prev = self._last_total_matched.get(sid)
            delta = 0.0 if prev is None else max(0.0, snap.total_matched - prev)
            self._last_total_matched[sid] = snap.total_matched

            if delta > 0.0:
                for order in self._orders:
                    if order.selection_id == sid:
                        order.traded_volume_since_placement += delta


@dataclass(slots=True)
class BetManager:
    """Tracks bets, budget and P&L across an episode.

    Args:
        starting_budget: Initial budget in £ for the episode.
        matcher: Optional :class:`ExchangeMatcher` instance. Defaults to
            the module-level ``DEFAULT_MATCHER`` which filters junk
            ladder levels more than 50 % from LTP and matches only at
            the best post-filter price (no ladder walking). Pass a
            custom instance to tighten/loosen the junk filter.
    """

    starting_budget: float
    budget: float = field(init=False)
    realised_pnl: float = 0.0
    bets: list[Bet] = field(default_factory=list)
    _open_liability: float = 0.0
    matcher: ExchangeMatcher = field(default=DEFAULT_MATCHER)
    _matched_at_level: dict[tuple[int, BetSide, float], float] = field(
        init=False, default_factory=dict, repr=False
    )
    passive_book: PassiveOrderBook = field(init=False)

    def __post_init__(self) -> None:
        self.budget = self.starting_budget
        self.passive_book = PassiveOrderBook(matcher=self.matcher)

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
        max_price: float | None = None,
    ) -> Bet | None:
        """Place a back bet via the :class:`ExchangeMatcher`.

        The matcher filters out stale parked ladder levels (anything
        more than ``max_price_deviation_pct`` from the runner's LTP),
        then matches at the best single remaining lay price — no ladder
        walking. If ``max_price`` is set and the best filtered price
        still exceeds it, the bet is refused entirely (implements
        ``betting_constraints.max_back_price``).

        Returns ``None`` if nothing could be matched, the budget is
        exhausted, the bet is refused by the price cap, or the runner
        has no LTP.
        """
        capped = min(stake, self.available_budget)
        if capped <= 0.0:
            return None

        # Peek at the top-of-book price so we can look up how much of that
        # level the agent has already consumed in this race.
        top_price = self.matcher.pick_top_price(
            runner.available_to_lay,
            reference_price=runner.last_traded_price,
            lower_is_better=True,
        )
        already_matched = (
            self._matched_at_level.get((runner.selection_id, BetSide.BACK, top_price), 0.0)
            if top_price is not None else 0.0
        )

        result: MatchResult = self.matcher.match_back(
            runner.available_to_lay,
            stake=capped,
            reference_price=runner.last_traded_price,
            max_price=max_price,
            already_matched_at_top=already_matched,
        )
        if result.matched_stake <= 0.0:
            return None

        bet = Bet(
            selection_id=runner.selection_id,
            side=BetSide.BACK,
            requested_stake=stake,
            matched_stake=result.matched_stake,
            average_price=result.average_price,
            market_id=market_id,
            ltp_at_placement=runner.last_traded_price or 0.0,
        )
        # Back bets: the stake is the cost (paid up-front).
        self.budget -= result.matched_stake
        self.bets.append(bet)
        key = (runner.selection_id, BetSide.BACK, result.average_price)
        self._matched_at_level[key] = self._matched_at_level.get(key, 0.0) + result.matched_stake
        return bet

    def place_lay(
        self,
        runner: RunnerSnap,
        stake: float,
        market_id: str = "",
        max_price: float | None = None,
    ) -> Bet | None:
        """Place a lay bet via the :class:`ExchangeMatcher`.

        The matcher filters stale parked ladder levels and matches at
        the best single remaining back price. If ``max_price`` is set
        and the best filtered back price exceeds it, the bet is refused
        — this prevents the layer from being matched into catastrophic
        liabilities (implements ``betting_constraints.max_lay_price``).

        Because lay bets reserve liability (``stake × (price − 1)``)
        rather than stake itself, this method may shrink the requested
        stake so the resulting liability fits inside ``available_budget``.
        Returns ``None`` if nothing could be matched, the liability
        cannot be reserved, the bet is refused by the price cap, or
        the runner has no LTP.
        """
        if stake <= 0.0:
            return None

        # Peek at the top-of-book price so we can look up how much of that
        # level the agent has already consumed in this race.
        top_price = self.matcher.pick_top_price(
            runner.available_to_back,
            reference_price=runner.last_traded_price,
            lower_is_better=False,
        )
        already_matched = (
            self._matched_at_level.get((runner.selection_id, BetSide.LAY, top_price), 0.0)
            if top_price is not None else 0.0
        )

        # First pass: probe the top-of-book price at the requested stake.
        result: MatchResult = self.matcher.match_lay(
            runner.available_to_back,
            stake=stake,
            reference_price=runner.last_traded_price,
            max_price=max_price,
            already_matched_at_top=already_matched,
        )
        if result.matched_stake <= 0.0:
            return None

        liability = result.matched_stake * (result.average_price - 1.0)

        # If the liability exceeds available budget, scale the requested
        # stake down so the liability fits and re-match.
        if liability > self.available_budget:
            if result.average_price <= 1.0:
                return None
            max_stake = self.available_budget / (result.average_price - 1.0)
            if max_stake <= 0.0:
                return None
            result = self.matcher.match_lay(
                runner.available_to_back,
                stake=max_stake,
                reference_price=runner.last_traded_price,
                max_price=max_price,
                already_matched_at_top=already_matched,
            )
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
            ltp_at_placement=runner.last_traded_price or 0.0,
        )
        # Reserve the liability from the budget.
        self._open_liability += liability
        self.bets.append(bet)
        key = (runner.selection_id, BetSide.LAY, result.average_price)
        self._matched_at_level[key] = self._matched_at_level.get(key, 0.0) + result.matched_stake
        return bet

    # ── Settlement ───────────────────────────────────────────────────────

    def settle_race(
        self,
        winning_selection_ids: int | set[int],
        market_id: str = "",
        commission: float = 0.0,
    ) -> float:
        """Settle all unsettled bets for a race and return the race P&L.

        Args:
            winning_selection_ids: Selection ID(s) that won/placed.
                For WIN markets pass the single winner.  For EACH_WAY
                (place) markets pass a set containing WINNER + PLACED IDs.
                Betfair EACH_WAY odds already include the place fraction
                so PLACED runners pay at the quoted price.
                Accepts a single int for backward compatibility.
            market_id: If provided, only settle bets for this market.
            commission: Betfair commission rate applied to net profit
                (e.g. 0.05 for 5%).  Only deducted from winning bets.

        Returns:
            Net P&L for the settled bets (after commission).
        """
        # Normalise to a set
        if isinstance(winning_selection_ids, int):
            winners = {winning_selection_ids}
        else:
            winners = winning_selection_ids

        race_pnl = 0.0

        for bet in self.bets:
            if bet.outcome is not BetOutcome.UNSETTLED:
                continue
            if market_id and bet.market_id != market_id:
                continue

            won_selection = bet.selection_id in winners

            if bet.side is BetSide.BACK:
                if won_selection:
                    # Winner/placed: profit = stake × (price − 1), get stake back.
                    gross_profit = bet.matched_stake * (bet.average_price - 1.0)
                    net_profit = gross_profit * (1.0 - commission)
                    self.budget += bet.matched_stake + net_profit
                    bet.pnl = net_profit
                    bet.outcome = BetOutcome.WON
                else:
                    # Loser: stake already deducted, nothing returned.
                    bet.pnl = -bet.matched_stake
                    bet.outcome = BetOutcome.LOST

            elif bet.side is BetSide.LAY:
                liability = bet.matched_stake * (bet.average_price - 1.0)
                if won_selection:
                    # Runner won/placed → layer loses liability.
                    self.budget -= liability
                    self._open_liability -= liability
                    bet.pnl = -liability
                    bet.outcome = BetOutcome.LOST
                else:
                    # Runner lost → layer keeps the backer's stake.
                    gross_profit = bet.matched_stake
                    net_profit = gross_profit * (1.0 - commission)
                    self.budget += net_profit
                    self._open_liability -= liability
                    bet.pnl = net_profit
                    bet.outcome = BetOutcome.WON

            race_pnl += bet.pnl

        self.realised_pnl += race_pnl
        return race_pnl

    def void_race(self, market_id: str = "") -> float:
        """Void all unsettled bets for a race — refund stakes and liability.

        Used when no winner is known (e.g. race result missing from data).
        Returns 0.0 (no P&L impact).
        """
        for bet in self.bets:
            if bet.outcome is not BetOutcome.UNSETTLED:
                continue
            if market_id and bet.market_id != market_id:
                continue

            if bet.side is BetSide.BACK:
                # Refund the stake that was deducted on placement
                self.budget += bet.matched_stake
            elif bet.side is BetSide.LAY:
                # Release the reserved liability
                liability = bet.matched_stake * (bet.average_price - 1.0)
                self._open_liability -= liability

            bet.pnl = 0.0
            bet.outcome = BetOutcome.VOID

        return 0.0

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

    def race_bet_count(self, market_id: str) -> int:
        """Return the number of bets placed in a specific race."""
        return sum(1 for b in self.bets if b.market_id == market_id)

    def get_positions(self, market_id: str) -> dict[int, dict]:
        """Get accumulated net position per runner for a race.

        Returns a dict keyed by ``selection_id`` with values::

            {
                "back_exposure": float,  # total matched back stake
                "lay_exposure": float,   # total matched lay stake (liability)
                "bet_count": int,
            }
        """
        positions: dict[int, dict] = {}
        for bet in self.bets:
            if bet.market_id != market_id:
                continue
            sid = bet.selection_id
            if sid not in positions:
                positions[sid] = {"back_exposure": 0.0, "lay_exposure": 0.0, "bet_count": 0}
            pos = positions[sid]
            pos["bet_count"] += 1
            if bet.side is BetSide.BACK:
                pos["back_exposure"] += bet.matched_stake
            elif bet.side is BetSide.LAY:
                pos["lay_exposure"] += bet.liability
        return positions
