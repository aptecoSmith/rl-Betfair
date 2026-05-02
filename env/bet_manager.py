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
from typing import TYPE_CHECKING, Literal

from data.episode_builder import PriceSize, RunnerSnap
from env.exchange_matcher import DEFAULT_MATCHER, ExchangeMatcher, MatchResult

if TYPE_CHECKING:
    from data.episode_builder import Tick


# Betfair Exchange minimum stake — bets below this are rejected.
# Real Betfair minimum is £2; this constant is used in BetManager to
# reject partial fills that fall below the threshold.
MIN_BET_STAKE = 2.00


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
    available_at_price: float = 0.0  # raw size at fill price before self-depletion (diagnostic — verify matched_stake ≤ this)
    # EW metadata — populated by settle_race() for each-way races
    is_each_way: bool = False
    each_way_divisor: float | None = None        # e.g. 4.0 for 1/4 odds
    number_of_places: int | None = None          # e.g. 3
    settlement_type: str = "standard"            # "standard" | "ew_winner" | "ew_placed" | "ew_unplaced"
    effective_place_odds: float | None = None    # (price-1)/divisor + 1, for display
    # Forced-arbitrage / scalping — links an aggressive fill to its
    # auto-generated passive counter-order (and vice-versa). None for
    # bets placed outside scalping_mode. See plans/issues-12-04-2026/
    # 05-forced-arbitrage/.
    pair_id: str | None = None
    # Actual liability reserved at placement (after any paired-arb offset).
    # ``None`` means "use the full ``self.liability`` value" — i.e. the bet
    # was placed un-paired, so the standard reservation rules apply.
    # Paired LAY bets where the partner BACK's stake fully covers the
    # worst-case loss have ``reserved_liability = 0.0``. Settlement &
    # cancellation paths must release this exact amount, NOT the full
    # liability, or `_open_liability` will go negative. See bet_manager
    # docstring on "freed budget" for the Betfair offset rationale.
    reserved_liability: float | None = None
    # Scalping-active-management session 02 — policy's fill-probability
    # prediction at the tick that placed the PAIR. The aggressive leg gets
    # this stamped on it by the trainer's rollout-time capture; the passive
    # leg inherits it from the aggressive partner (matched by ``pair_id``)
    # when the passive fills inside :meth:`PassiveOrderBook.on_tick`. Per
    # hard_constraints.md §10 the value is captured at decision time, never
    # recomputed later. ``None`` for bets not produced by the scalping head
    # (directional bets, pre-Session-02 data, stub tests constructing
    # ``Bet`` directly).
    fill_prob_at_placement: float | None = None
    # Scalping-active-management session 03 — policy's risk-head outputs
    # at the tick that placed the PAIR. ``predicted_locked_pnl`` is the
    # mean channel (expected locked P&L in £); ``predicted_locked_stddev``
    # is ``exp(0.5 * clamped_log_var)``, pre-computed at capture time so
    # parquet consumers / UI badges don't have to replicate the math.
    # Same capture→attach flow as Session 02 (aggressive stamped by the
    # PPO rollout; passive inherits via ``pair_id``). ``None`` for
    # directional bets, pre-Session-03 data, stub tests.
    predicted_locked_pnl_at_placement: float | None = None
    predicted_locked_stddev_at_placement: float | None = None
    # Scalping-close-signal session 01 — marks a Bet placed by
    # :meth:`BetfairEnv._attempt_close` as the aggressive close leg of
    # a pair the agent deliberately crossed out of. A pair with any
    # ``close_leg=True`` bet is classified as an ``arbs_closed`` event
    # at settlement (distinct from ``arbs_completed`` which reserves
    # pairs whose passive leg filled naturally). Default False for
    # every non-close bet — including the original aggressive leg of
    # the closed pair.
    close_leg: bool = False
    # Arb-signal-cleanup Session 01 (2026-04-21) — distinguishes env-
    # initiated force-closes at T−N seconds from agent-initiated
    # ``close_signal`` closes. ``force_close=True`` implies
    # ``close_leg=True`` (both are set together at placement). A pair
    # with any ``force_close=True`` leg is classified as
    # ``arbs_force_closed`` at settlement — excluded from the matured-
    # arb bonus and from the ``+£1 per close_signal`` shaped bonus
    # (the agent didn't choose these closes). See
    # plans/arb-signal-cleanup/hard_constraints.md §7, §12, §14.
    force_close: bool = False
    # Force-close-architecture Session 02 (2026-05-02) — distinguishes
    # env-initiated mid-race stop-closes (fired when per-pair MTM
    # crosses -stop_loss_pnl_threshold) from agent-initiated and T−N
    # force-closes. ``stop_close=True`` implies ``close_leg=True``;
    # ``force_close`` and ``stop_close`` are mutually exclusive
    # (stop-close goes through the strict matcher, not the relaxed
    # force-close path). A pair with any ``stop_close=True`` leg is
    # classified as ``arbs_stop_closed`` at settlement — excluded
    # from the matured-arb bonus and the ``+£1 per close_signal``
    # shaped bonus (the agent didn't choose these closes either).
    # See plans/rewrite/phase-3-followups/force-close-architecture/.
    stop_close: bool = False

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
    Fill logic lands in session 26; cancellation in session 29.
    """

    selection_id: int
    side: BetSide
    price: float                          # price the order rests at
    requested_stake: float
    queue_ahead_at_placement: float       # own-side top-level size at placement
    placed_tick_index: int
    market_id: str
    ltp_at_placement: float = 0.0         # LTP when order was placed (for Bet.ltp_at_placement on fill)
    traded_volume_since_placement: float = 0.0
    matched_stake: float = 0.0            # set to requested_stake on fill (session 26)
    cancelled: bool = False
    cancel_reason: str = ""                # e.g. "race-off" or "agent" (session 29)
    # Forced-arbitrage pairing — matches a passive counter-order to its
    # aggressive fill. None for ordinary passive orders placed directly
    # by the agent outside scalping_mode.
    pair_id: str | None = None
    # Actual liability reserved at placement (after any paired-arb offset).
    # See ``Bet.reserved_liability`` for full explanation. Carried over to
    # the resulting Bet on fill, and to cancel_* on cancellation, so the
    # release math always undoes exactly what was reserved.
    reserved_liability: float | None = None
    # Time-to-off (seconds) at placement. Used by the observation builder
    # to compute ``seconds_since_passive_placed`` — the elapsed real
    # seconds between this order's placement and the current tick, as a
    # per-runner obs feature in scalping mode. 0.0 for orders placed
    # outside scalping mode or before this field was introduced.
    placed_time_to_off: float = 0.0

    def to_dict(self) -> dict:
        return {
            "selection_id": self.selection_id,
            "side": self.side.value,
            "price": self.price,
            "requested_stake": self.requested_stake,
            "queue_ahead_at_placement": self.queue_ahead_at_placement,
            "placed_tick_index": self.placed_tick_index,
            "market_id": self.market_id,
            "ltp_at_placement": self.ltp_at_placement,
            "traded_volume_since_placement": self.traded_volume_since_placement,
            "matched_stake": self.matched_stake,
            "cancelled": self.cancelled,
            "cancel_reason": self.cancel_reason,
            "pair_id": self.pair_id,
        }


class PassiveOrderBook:
    """Bookkeeping container for resting (passive) orders.

    Owned by :class:`BetManager` as ``self.passive_book``. Responsible for:
    - Snapshotting queue-ahead at placement time.
    - Reserving budget at placement (back: deduct stake; lay: reserve liability).
    - Accumulating traded-volume deltas per runner across ticks.
    - Converting orders to ``Bet`` objects once the fill threshold is crossed.

    The ``_bet_manager`` back-reference is set by ``BetManager.__post_init__``
    so that this class can access and mutate budget fields and the bets list
    without circular import issues.

    Race-off cancellation via ``cancel_all`` (session 27); agent-driven cancel
    action lands in session 29.
    """

    def __init__(
        self,
        matcher: ExchangeMatcher = DEFAULT_MATCHER,
        fill_mode: Literal["volume", "pragmatic"] = "volume",
    ) -> None:
        self._matcher = matcher
        # Phase −1 env audit Session 03 (2026-04-26): which Phase-1
        # accumulator on_tick should run. ``"volume"`` is spec-faithful
        # per-runner ``total_matched`` deltas (requires F7-fixed data).
        # ``"pragmatic"`` prorates market-level traded volume across
        # runners by visible book size — fallback for historical days
        # captured before the StreamRecorder per-runner volume fix.
        # Set per-race from ``Day.fill_mode`` via ``BetManager``;
        # default ``"volume"`` keeps stub / synthetic tests on the
        # spec path. Phase 2 (fill check) is shared between modes.
        self._fill_mode: Literal["volume", "pragmatic"] = fill_mode
        self._orders: list[PassiveOrder] = []
        # Index: selection_id → list of orders for O(1) lookup in on_tick.
        self._orders_by_sid: dict[int, list[PassiveOrder]] = {}
        # Per-runner last-seen total_matched value, for computing deltas.
        # Populated on first on_tick call; reset when the PassiveOrderBook
        # is replaced (fresh BetManager per race). Used in volume mode.
        self._last_total_matched: dict[int, float] = {}
        # Pragmatic mode — last-seen market-level traded_volume scalar.
        # Sentinel ``None`` = first tick (delta is zero by definition).
        self._prev_market_tv: float | None = None
        # Back-reference to the owning BetManager, set by BetManager.__post_init__.
        # Used for budget reservation at placement and Bet creation at fill.
        self._bet_manager: "BetManager | None" = None
        # Passive self-depletion: tracks own-side stake already filled at each
        # (selection_id, side, price) level within this race.  This shifts the
        # fill threshold for subsequent passive orders resting at the same price,
        # simulating the fact that the first order consumed part of the queue.
        # Keyed on own-side (resting) price levels — DISTINCT from
        # BetManager._matched_at_level, which tracks aggressive (opposite-side)
        # fill consumption at the time of aggressive placement.
        self._passive_matched_at_level: dict[tuple[int, "BetSide", float], float] = {}
        # Fill events emitted by the most recent on_tick call.  Reset at the
        # start of each on_tick; read by BetfairEnv._get_info for
        # info["passive_fills"].  Each entry: (selection_id, price, filled_stake).
        self._last_fills: list[tuple[int, float, float]] = []
        # Cancellation events emitted by cancel_all().  Read by
        # BetfairEnv._get_info for info["passive_cancels"].
        self._last_cancels: list[dict] = []
        # History of cancelled orders — kept for the replay UI.
        self._cancelled_orders: list[PassiveOrder] = []
        # Paired-order diagnostics — counts silent rejections of explicit-price
        # passive placements (those carrying a ``pair_id``) broken down by
        # reason, plus per-tick fill-check skips for paired orders that
        # already rest in the book. Cumulative per BetManager (i.e. per race);
        # BetfairEnv accumulates across races. See plans/arb-improvements/
        # for context — until this counter existed, every paired-leg failure
        # was invisible and the registry showed 0/7750 completed/naked arbs.
        self._paired_place_rejects: dict[str, int] = {
            "no_ltp": 0,
            "price_invalid": 0,
            "budget_back": 0,
            "budget_lay": 0,
        }
        self._paired_fill_skips_ltp_filter: int = 0

    @property
    def orders(self) -> list[PassiveOrder]:
        """Open passive orders (excludes already-filled orders)."""
        return list(self._orders)

    @property
    def last_fills(self) -> list[tuple[int, float, float]]:
        """Fill events from the most recent ``on_tick`` call.

        Each entry is ``(selection_id, price, filled_stake)``.  Reset at the
        start of every ``on_tick``; an empty list means no fills this tick.
        """
        return list(self._last_fills)

    @property
    def last_cancels(self) -> list[dict]:
        """Cancellation events from the most recent ``cancel_all`` call."""
        return list(self._last_cancels)

    @property
    def cancelled_orders(self) -> list[PassiveOrder]:
        """All orders cancelled during this race (history for replay UI)."""
        return list(self._cancelled_orders)

    @property
    def cancel_count(self) -> int:
        """Number of orders cancelled in this race."""
        return len(self._cancelled_orders)

    def cancel_order(
        self,
        order: PassiveOrder,
        reason: str = "requote",
    ) -> PassiveOrder | None:
        """Cancel a specific open passive order by identity.

        Used by the Session 01 re-quote mechanic where the caller has
        already resolved which paired passive to cancel. Releases budget
        via the same freed-budget rule as ``cancel_oldest_for`` /
        ``cancel_all``. Returns the cancelled order, or ``None`` if the
        order is already cancelled or not tracked in the book (idempotent).
        """
        if order.cancelled:
            return None
        sid_orders = self._orders_by_sid.get(order.selection_id)
        if not sid_orders or order not in sid_orders:
            return None

        order.cancelled = True
        order.cancel_reason = reason

        bm = self._bet_manager
        if bm is not None:
            if order.side is BetSide.BACK:
                bm.budget += order.requested_stake
            else:  # LAY
                released = (
                    order.reserved_liability
                    if order.reserved_liability is not None
                    else order.requested_stake * (order.price - 1.0)
                )
                bm._open_liability -= released

        self._last_cancels.append({
            "selection_id": order.selection_id,
            "price": order.price,
            "requested_stake": order.requested_stake,
            "reason": reason,
        })

        self._cancelled_orders.append(order)

        sid_orders.remove(order)
        if not sid_orders:
            del self._orders_by_sid[order.selection_id]
        self._orders.remove(order)
        return order

    def cancel_oldest_for(
        self,
        selection_id: int,
        reason: str = "policy cancel",
    ) -> PassiveOrder | None:
        """Cancel the oldest open passive order on *selection_id*.

        Budget reservation is released via the same path as ``cancel_all``.
        Returns the cancelled order, or ``None`` if no open order existed
        (idempotent — spurious cancels are fine).
        """
        sid_orders = self._orders_by_sid.get(selection_id)
        if not sid_orders:
            return None

        # Oldest = first element (FIFO insert order).
        order = sid_orders[0]
        order.cancelled = True
        order.cancel_reason = reason

        # Release budget reservation. Honour the order's actual reservation
        # if recorded (paired-arb freed-budget path); otherwise fall back to
        # the textbook full-liability release.
        bm = self._bet_manager
        if bm is not None:
            if order.side is BetSide.BACK:
                bm.budget += order.requested_stake
            else:  # LAY
                released = (
                    order.reserved_liability
                    if order.reserved_liability is not None
                    else order.requested_stake * (order.price - 1.0)
                )
                bm._open_liability -= released

        # Emit cancellation event.
        self._last_cancels.append({
            "selection_id": order.selection_id,
            "price": order.price,
            "requested_stake": order.requested_stake,
            "reason": reason,
        })

        # Keep in history for replay UI.
        self._cancelled_orders.append(order)

        # Remove from tracking structures.
        sid_orders.pop(0)
        if not sid_orders:
            del self._orders_by_sid[selection_id]
        self._orders.remove(order)

        return order

    def cancel_all(self, reason: str = "race-off") -> None:
        """Cancel all open passive orders, releasing budget reservations.

        Called at race-off (session 27) to clean up unfilled passive orders
        before race settlement runs.  Cancelled orders contribute zero P&L.

        Idempotent: calling twice produces the same state as calling once.

        Args:
            reason: Human-readable reason string for the replay UI.
        """
        self._last_cancels = []
        bm = self._bet_manager

        for order in self._orders:
            order.cancelled = True
            order.cancel_reason = reason

            # Release budget reservation. Same freed-budget rule as
            # cancel_oldest_for above.
            if bm is not None:
                if order.side is BetSide.BACK:
                    bm.budget += order.requested_stake
                else:  # LAY
                    released = (
                        order.reserved_liability
                        if order.reserved_liability is not None
                        else order.requested_stake * (order.price - 1.0)
                    )
                    bm._open_liability -= released

            # Emit cancellation event for info["passive_cancels"].
            self._last_cancels.append({
                "selection_id": order.selection_id,
                "price": order.price,
                "requested_stake": order.requested_stake,
                "reason": reason,
            })

            # Keep in history for replay UI.
            self._cancelled_orders.append(order)

        # Clear all open orders and the sid index.
        self._orders.clear()
        self._orders_by_sid.clear()

    def place(
        self,
        runner: RunnerSnap,
        stake: float,
        side: BetSide,
        market_id: str,
        tick_index: int,
        *,
        price: float | None = None,
        pair_id: str | None = None,
        time_to_off: float = 0.0,
    ) -> PassiveOrder | None:
        """Record a resting order at the own-side best price and reserve budget.

        For a passive back the order rests in the available_to_back queue
        (the price other backers are offering); for a passive lay it rests
        in the available_to_lay queue.

        The best post-filter price is found via the same junk filter the
        aggressive matcher uses. If no valid level exists (empty ladder or
        all levels outside ±``max_price_deviation_pct`` from LTP), returns
        ``None``.

        Budget is reserved immediately on placement:

        - **Back**: ``stake`` is deducted from ``BetManager.budget``.  If
          ``stake > available_budget``, the order is refused (returns ``None``).
        - **Lay**: ``stake × (price − 1)`` is added to
          ``BetManager._open_liability``.  If the liability would exceed
          ``available_budget``, the order is refused.

        On fill the reservation is converted in-place — no second deduction.
        On cancel (session 29) the reservation is released.
        """
        ltp = runner.last_traded_price
        if ltp is None or ltp <= 0.0:
            if pair_id is not None:
                self._paired_place_rejects["no_ltp"] += 1
            return None

        if side is BetSide.BACK:
            # Passive back: order rests on the back side (we are offering to back)
            levels = runner.available_to_back
            lower_is_better = False  # highest back price is best for a backer
        else:
            # Passive lay: order rests on the lay side
            levels = runner.available_to_lay
            lower_is_better = True   # lowest lay price is best for a layer

        # Junk-filter bounds — reused below for both the top-of-book and the
        # explicit-price paths.
        lo = ltp * (1.0 - self._matcher.max_price_deviation_pct)
        hi = ltp * (1.0 + self._matcher.max_price_deviation_pct)

        if price is None:
            # ── Default path: rest at the best post-filter own-side price. ─
            top_price = self._matcher.pick_top_price(
                levels,
                reference_price=ltp,
                lower_is_better=lower_is_better,
            )
            if top_price is None:
                return None

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
            resting_price = top_level.price
            queue_ahead = top_level.size
        else:
            # ── Explicit-price path (forced arbitrage paired order). ──────
            # The caller supplies the exact resting price, derived from a
            # real fill via ``tick_offset`` (which already hard-clamps to
            # Betfair's [1.01, 1000] price band). The LTP-relative junk
            # filter that guards the default path does NOT apply here:
            #
            # - The risk it mitigates ("walking a stale £1000 parked
            #   order") is irrelevant for a passive resting order — the
            #   order rests at the agent's chosen price and only fills
            #   when traded volume actually crosses it.
            # - At MAX_ARB_TICKS=100 a paired leg can legitimately sit
            #   well outside ±max_price_deviation_pct of the *fill-time*
            #   LTP, especially on short-priced runners where each tick
            #   is a small absolute move but compounds quickly. With the
            #   junk filter on, every such pair was silently refused —
            #   producing the "always-naked arb" symptom seen in the
            #   registry (376 eval days, 7,750 naked, 0 completed).
            #
            # We still require a positive price (defensive — caller
            # should never pass <=0).
            if price <= 0.0:
                if pair_id is not None:
                    self._paired_place_rejects["price_invalid"] += 1
                return None
            resting_price = price
            # Queue ahead: size already sitting at that level, or 0 if the
            # level is currently empty (we'd be first in the queue).
            existing = next(
                (lv for lv in levels
                 if lv.price > 0.0 and abs(lv.price - price) < 1e-9),
                None,
            )
            queue_ahead = existing.size if existing is not None else 0.0

        # Budget reservation — must happen before the order is appended so
        # that a failed reservation leaves the book unchanged.
        #
        # Paired-arb "freed budget" rule (added 2026-04-15):
        # Real Betfair recognises that a back-and-lay on the same selection
        # can never both lose, and frees up the reservation accordingly.
        # Worst-case loss = max(back_stake, lay_liability), NOT
        # back_stake + lay_liability. For typical scalping prices (lay ≤ 2.0)
        # the lay liability is fully covered by the back stake — zero new
        # reservation. We model this for the paired LAY-after-BACK path
        # below; the paired BACK-after-LAY path is left additive for now
        # (TODO: symmetric handling once the LAY-aggressive path starts
        # bumping into rej_bback in the diagnostic).
        bm = self._bet_manager
        reserved_for_order: float | None = None
        if bm is not None:
            if side is BetSide.BACK:
                if stake > bm.available_budget:
                    if pair_id is not None:
                        self._paired_place_rejects["budget_back"] += 1
                    return None
                bm.budget -= stake
                # Symmetric paired BACK-after-LAY freeing: when the second
                # leg of a (LAY first → BACK second) pair lands, the
                # aggressive lay's liability already covers part of the
                # joint exposure. Release that overlap on the LAY side and
                # mark the LAY's reserved_liability so settlement releases
                # the same reduced amount (otherwise _open_liability would
                # go negative). Mirrors the LAY-after-BACK case below.
                if pair_id is not None:
                    for prior in reversed(bm.bets):
                        if (
                            prior.pair_id == pair_id
                            and prior.side is BetSide.LAY
                        ):
                            already_reserved = (
                                prior.reserved_liability
                                if prior.reserved_liability is not None
                                else prior.matched_stake
                                    * (prior.average_price - 1.0)
                            )
                            offset = min(stake, already_reserved)
                            if offset > 0.0:
                                bm._open_liability -= offset
                                prior.reserved_liability = (
                                    already_reserved - offset
                                )
                            break
            else:  # LAY
                liability = stake * (resting_price - 1.0)
                # Paired LAY-after-BACK offset: find the matched aggressive
                # back leg for this pair_id (most recent if multiple) and
                # net its stake against the lay's liability.
                offset = 0.0
                if pair_id is not None:
                    for prior in reversed(bm.bets):
                        if (
                            prior.pair_id == pair_id
                            and prior.side is BetSide.BACK
                        ):
                            offset = min(prior.matched_stake, liability)
                            break
                reserved = max(0.0, liability - offset)
                if reserved > bm.available_budget:
                    if pair_id is not None:
                        self._paired_place_rejects["budget_lay"] += 1
                    return None
                bm._open_liability += reserved
                reserved_for_order = reserved

        order = PassiveOrder(
            selection_id=runner.selection_id,
            side=side,
            price=resting_price,
            requested_stake=stake,
            queue_ahead_at_placement=queue_ahead,
            placed_tick_index=tick_index,
            market_id=market_id,
            ltp_at_placement=ltp,
            pair_id=pair_id,
            reserved_liability=reserved_for_order,
            placed_time_to_off=time_to_off,
        )
        self._orders.append(order)
        self._orders_by_sid.setdefault(order.selection_id, []).append(order)
        # Seed the volume baseline so the first on_tick call computes the
        # delta from the moment of placement, not from first sight.
        # Per open_questions.md Q4: "compute at runtime by snapshotting at
        # placement and subtracting."
        sid = runner.selection_id
        if sid not in self._last_total_matched:
            self._last_total_matched[sid] = runner.total_matched
        return order

    def on_tick(self, tick: "Tick", tick_index: int = -1) -> None:
        """Accumulate traded-volume deltas and convert filled orders to Bets.

        Two-phase per tick:

        1. **Volume accumulation** — Phase 1 dispatches by ``self._fill_mode``.
           ``"volume"`` uses the spec-faithful per-runner ``total_matched``
           delta; ``"pragmatic"`` prorates the market-level traded-volume
           delta across runners by visible book size for historical days
           where per-runner volume wasn't captured. Both feed the same
           ``order.traded_volume_since_placement`` accumulator.

        2. **Fill check** — for each open order, skip if the resting price has
           drifted outside the LTP ±``max_price_deviation_pct`` junk filter
           (the order stays open and will re-evaluate next tick).  Otherwise,
           compute the fill threshold:
           ``queue_ahead_at_placement + passive_self_depletion_at_this_price``.
           If ``traded_volume_since_placement ≥ threshold``, the order fills:
           a ``Bet`` is created (price = queue price, not opposite-side top),
           appended to ``BetManager.bets``, and the order is removed from the
           open list.  Fill events are recorded in ``_last_fills``.

        Budget is **not** changed on fill — the stake/liability was already
        reserved at placement.
        """
        self._last_fills = []
        runner_by_sid = {r.selection_id: r for r in tick.runners}

        # ── Phase 1: accumulate traded-volume deltas ──────────────────────
        if self._fill_mode == "volume":
            self._volume_phase_1(tick, runner_by_sid)
        else:
            self._pragmatic_phase_1(tick, runner_by_sid)

        # ── Phase 2: fill check ───────────────────────────────────────────
        if self._bet_manager is None:
            return

        # Collect filled orders, then remove in bulk to avoid O(n) per removal.
        # The self-depletion accumulator is updated eagerly inside the loop so
        # that the second order at the same price immediately sees the first
        # order's filled stake in its threshold calculation.
        filled: list[PassiveOrder] = []

        for order in self._orders:
            snap = runner_by_sid.get(order.selection_id)
            if snap is None:
                continue

            # Junk filter: if the resting price has drifted outside the LTP
            # tolerance, skip this tick.  The order stays open and re-evaluates
            # on the next tick (session 26 constraint 5 — no auto-cancel here).
            ltp = snap.last_traded_price
            if ltp is None or ltp <= 0.0:
                continue
            lo = ltp * (1.0 - self._matcher.max_price_deviation_pct)
            hi = ltp * (1.0 + self._matcher.max_price_deviation_pct)
            if not (lo <= order.price <= hi):
                # Diagnostic: for paired orders this is the second silent
                # gate that blocks fills (the first being placement-time).
                # Counted per (order × tick) so a long-lived paired leg
                # outside the LTP window will rack up many skips.
                if order.pair_id is not None:
                    self._paired_fill_skips_ltp_filter += 1
                continue

            # Passive self-depletion: shift the threshold by how much stake
            # has already been filled at this own-side price level this race.
            # Updated eagerly so subsequent orders in this same tick see the
            # correct cumulative value.
            key = (order.selection_id, order.side, order.price)
            already_filled = self._passive_matched_at_level.get(key, 0.0)
            fill_threshold = order.queue_ahead_at_placement + already_filled

            if order.traded_volume_since_placement < fill_threshold:
                continue

            # ── Fill ────────────────────────────────────────────────────
            order.matched_stake = order.requested_stake

            # Scalping-active-management §02/§03: the passive leg inherits
            # its aggressive partner's decision-time aux-head predictions
            # (fill-prob, risk mean + stddev) via ``pair_id`` lookup. Per
            # hard_constraints §10 the values are captured at decision
            # time, never recomputed post-hoc — this keeps UI calibration
            # plots honest. Aggressive legs are always appended to
            # ``bm.bets`` before the passive order is registered, so the
            # lookup succeeds whenever the aggressive carried predictions.
            inherited_fill_prob: float | None = None
            inherited_risk_pnl: float | None = None
            inherited_risk_stddev: float | None = None
            if order.pair_id is not None and self._bet_manager is not None:
                for existing in self._bet_manager.bets:
                    if existing.pair_id == order.pair_id:
                        inherited_fill_prob = existing.fill_prob_at_placement
                        inherited_risk_pnl = (
                            existing.predicted_locked_pnl_at_placement
                        )
                        inherited_risk_stddev = (
                            existing.predicted_locked_stddev_at_placement
                        )
                        break

            # Convert to a Bet.  Fill price is the queue price (order.price),
            # not the opposite-side top — this is the key invariant of passive
            # orders (cheaper than crossing the spread).  Budget unchanged:
            # the stake/liability was already reserved at placement.
            bet = Bet(
                selection_id=order.selection_id,
                side=order.side,
                requested_stake=order.requested_stake,
                matched_stake=order.requested_stake,
                average_price=order.price,
                market_id=order.market_id,
                ltp_at_placement=order.ltp_at_placement,
                pair_id=order.pair_id,
                tick_index=tick_index,
                reserved_liability=order.reserved_liability,
                fill_prob_at_placement=inherited_fill_prob,
                predicted_locked_pnl_at_placement=inherited_risk_pnl,
                predicted_locked_stddev_at_placement=inherited_risk_stddev,
            )
            self._bet_manager.bets.append(bet)

            # Eagerly update passive self-depletion accumulator.
            self._passive_matched_at_level[key] = already_filled + order.requested_stake

            # Emit fill event for info["passive_fills"] and the replay UI.
            self._last_fills.append(
                (order.selection_id, order.price, order.requested_stake)
            )

            filled.append(order)

        # Bulk removal: rebuild the order list and index excluding filled orders.
        if filled:
            filled_set = set(id(o) for o in filled)
            self._orders = [o for o in self._orders if id(o) not in filled_set]
            for sid, sid_orders in self._orders_by_sid.items():
                self._orders_by_sid[sid] = [
                    o for o in sid_orders if id(o) not in filled_set
                ]

    def _volume_phase_1(
        self,
        tick: "Tick",
        runner_by_sid: dict[int, RunnerSnap],
    ) -> None:
        """Spec-faithful Phase 1 — per-runner ``total_matched`` deltas.

        Use the sid index to avoid scanning all orders per runner.

        Per-order crossability gate (2026-04-22 fix). Without this guard
        the cumulative-volume threshold a resting order needs to fill
        was advanced by ANY trade on the runner, regardless of price.
        A resting LAY at 1.29 would be filled from trades at 1.52 that
        couldn't possibly cross it — a backer getting 1.52 has no
        reason to drop to 1.29. That produced fictional "locked"
        pairs in the bet log (e.g. operator-flagged Wolverhampton
        17:05 Rb Yas Sir A on 313cec8e, lay@1.29 and back@1.52 at
        the same tick_timestamp with back > lay — impossible in a
        real Betfair book).

        The crossability check: a LAY at price P fills only when a
        trade happens at or below P (a backer crosses the spread
        down to take the lay). A BACK at price P fills only when a
        trade happens at or above P. We proxy "trade price" with
        LTP, the last-traded-price on this tick — not perfect (LTP
        is a single value but many trades can happen between ticks
        at different prices) but strictly better than counting ALL
        volume at ALL prices.
        """
        for sid, sid_orders in self._orders_by_sid.items():
            if not sid_orders:
                continue
            snap = runner_by_sid.get(sid)
            if snap is None:
                continue
            prev = self._last_total_matched.get(sid)
            delta = 0.0 if prev is None else max(0.0, snap.total_matched - prev)
            self._last_total_matched[sid] = snap.total_matched

            if delta > 0.0:
                ltp = snap.last_traded_price
                for order in sid_orders:
                    if ltp is None or ltp <= 0.0:
                        # No LTP on this tick — we can't tell whether
                        # trades would have crossed. Skip accumulation
                        # rather than wrongly advancing the queue.
                        continue
                    if order.side is BetSide.LAY and ltp > order.price:
                        # Trades at a price strictly above this lay's
                        # price — a backer taking higher prices has no
                        # reason to cross down. Order stays in queue.
                        continue
                    if order.side is BetSide.BACK and ltp < order.price:
                        # Trades at a price strictly below this back's
                        # price — a layer taking lower prices has no
                        # reason to cross up.
                        continue
                    order.traded_volume_since_placement += delta

    def _pragmatic_phase_1(
        self,
        tick: "Tick",
        runner_by_sid: dict[int, RunnerSnap],
    ) -> None:
        """Pragmatic Phase 1 — prorate market-level traded-volume delta.

        Used on historical days where ``RunnerSnap.total_matched`` is
        identically zero (F7: the upstream poller never captured per-
        runner cumulative volume). Market-level ``tick.traded_volume``
        IS populated on those days (£100k–£5M per race), so we sum
        the per-tick delta and prorate it across runners by visible
        book size weight (back + lay sizes summed, normalised). The
        same per-order crossability gate as volume mode is applied
        — a resting LAY at price P only counts trades whose proxied
        trade-price (LTP) is at or below P. See the docstring on
        ``_volume_phase_1`` for the rationale on the LTP proxy.

        Phase 2 (fill check) is unchanged. Both modes feed the same
        ``order.traded_volume_since_placement`` accumulator; only the
        source of ``delta`` differs. Pragmatic mode is a documented
        approximation; the active mode is surfaced via
        ``info["fill_mode_active"]``, ``RaceRecord.fill_mode``, and
        episode JSONL ``fill_mode``. See plans/rewrite/
        phase-minus-1-env-audit/session_prompts/03_dual_mode_fill_env.md.
        """
        market_tv = tick.traded_volume
        if self._prev_market_tv is None:
            # First tick — seed the baseline so the first delta is zero
            # by definition (matches volume mode's first-tick semantics).
            self._prev_market_tv = market_tv
            return
        market_delta = max(0.0, market_tv - self._prev_market_tv)
        self._prev_market_tv = market_tv
        if market_delta <= 0.0:
            return

        # Build runner weights by total visible book size (back + lay).
        # Runners with thicker books trade proportionally more.
        runner_weights: dict[int, float] = {}
        total_visible = 0.0
        for r in tick.runners:
            if r.status != "ACTIVE":
                continue
            size = (
                sum(lv.size for lv in r.available_to_back)
                + sum(lv.size for lv in r.available_to_lay)
            )
            runner_weights[r.selection_id] = size
            total_visible += size
        if total_visible <= 0.0:
            return

        for sid, sid_orders in self._orders_by_sid.items():
            if not sid_orders:
                continue
            weight = runner_weights.get(sid, 0.0) / total_visible
            synth_delta = market_delta * weight
            if synth_delta <= 0.0:
                continue
            snap = runner_by_sid.get(sid)
            if snap is None:
                continue
            ltp = snap.last_traded_price
            for order in sid_orders:
                # Crossability gate — same logic as volume mode.
                if ltp is None or ltp <= 0.0:
                    continue
                if order.side is BetSide.LAY and ltp > order.price:
                    continue
                if order.side is BetSide.BACK and ltp < order.price:
                    continue
                order.traded_volume_since_placement += synth_delta


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
    # Phase −1 env audit Session 03 (2026-04-26): passive-fill mode for
    # this race's PassiveOrderBook. Set per-race by BetfairEnv from
    # ``Day.fill_mode``. Default ``"volume"`` keeps existing stub /
    # synthetic tests on the spec path.
    fill_mode: Literal["volume", "pragmatic"] = "volume"
    # Aggressive self-depletion: tracks opposite-side stake already consumed at
    # each (selection_id, side, price) level in this race by aggressive bets.
    # Distinct from PassiveOrderBook._passive_matched_at_level, which tracks
    # own-side levels consumed by passive fills.
    _matched_at_level: dict[tuple[int, BetSide, float], float] = field(
        init=False, default_factory=dict, repr=False
    )
    passive_book: PassiveOrderBook = field(init=False)

    def __post_init__(self) -> None:
        self.budget = self.starting_budget
        self.passive_book = PassiveOrderBook(
            matcher=self.matcher,
            fill_mode=self.fill_mode,
        )
        # Wire the back-reference so PassiveOrderBook can access budget fields
        # and the bets list for placement-time reservation and fill conversion.
        self.passive_book._bet_manager = self

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
        *,
        pair_id: str | None = None,
        force_close: bool = False,
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

        ``force_close=True`` switches to the relaxed match semantics
        (no LTP requirement, no junk filter; hard price cap still
        enforced). It ALSO bypasses the per-race budget clamp — the
        live trader has more than the per-race budget in the bank,
        so an overdraft to flatten an already-matched position is
        always available (and the cost flows through ``race_pnl`` at
        settle so the agent learns from it). ``MIN_BET_STAKE`` (£2)
        still applies — Betfair's real minimum. Only the env's
        force-close path sets this — regular opens and agent-
        initiated closes keep the strict matching. See CLAUDE.md
        "Force-close at T−N" and
        plans/arb-signal-cleanup/hard_constraints.md §11.
        """
        if force_close:
            # Overdraft allowed: skip the ``available_budget`` clamp.
            # ``bm.budget`` may go negative after this placement; that's
            # real cash cost the agent sees in raw P&L at settle.
            capped = stake
        else:
            capped = min(stake, self.available_budget)
        if capped <= 0.0:
            return None

        # Peek at the top-of-book price so we can look up how much of that
        # level the agent has already consumed in this race.
        top_price = self.matcher.pick_top_price(
            runner.available_to_back,
            reference_price=runner.last_traded_price,
            lower_is_better=False,
            force_close=force_close,
        )
        already_matched = (
            self._matched_at_level.get((runner.selection_id, BetSide.BACK, top_price), 0.0)
            if top_price is not None else 0.0
        )

        result: MatchResult = self.matcher.match_back(
            runner.available_to_back,
            stake=capped,
            reference_price=runner.last_traded_price,
            max_price=max_price,
            already_matched_at_top=already_matched,
            force_close=force_close,
        )
        if result.matched_stake < MIN_BET_STAKE:
            return None

        bet = Bet(
            selection_id=runner.selection_id,
            side=BetSide.BACK,
            requested_stake=stake,
            matched_stake=result.matched_stake,
            average_price=result.average_price,
            market_id=market_id,
            ltp_at_placement=runner.last_traded_price or 0.0,
            available_at_price=result.top_level_size,
            pair_id=pair_id,
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
        *,
        pair_id: str | None = None,
        force_close: bool = False,
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

        ``force_close=True`` switches to the relaxed match semantics
        (no LTP requirement, no junk filter; hard price cap still
        enforced). It ALSO bypasses the liability budget gate — see
        ``place_back`` for the overdraft rationale. ``MIN_BET_STAKE``
        still applies. Only the env's force-close path sets this.
        """
        if stake <= 0.0:
            return None

        # Peek at the top-of-book price so we can look up how much of that
        # level the agent has already consumed in this race.
        top_price = self.matcher.pick_top_price(
            runner.available_to_lay,
            reference_price=runner.last_traded_price,
            lower_is_better=True,
            force_close=force_close,
        )
        already_matched = (
            self._matched_at_level.get((runner.selection_id, BetSide.LAY, top_price), 0.0)
            if top_price is not None else 0.0
        )

        # First pass: probe the top-of-book price at the requested stake.
        result: MatchResult = self.matcher.match_lay(
            runner.available_to_lay,
            stake=stake,
            reference_price=runner.last_traded_price,
            max_price=max_price,
            already_matched_at_top=already_matched,
            force_close=force_close,
        )
        if result.matched_stake < MIN_BET_STAKE:
            return None

        liability = result.matched_stake * (result.average_price - 1.0)

        # If the liability exceeds available budget, scale the requested
        # stake down so the liability fits and re-match. Force-close
        # bypasses this scale-down (overdraft allowed per place_back
        # docstring) — the close leg lands at the requested 1:1 stake
        # even if liability > available_budget; ``bm.budget`` goes
        # negative and the cost surfaces in raw P&L at settle.
        if not force_close and liability > self.available_budget:
            if result.average_price <= 1.0:
                return None
            max_stake = self.available_budget / (result.average_price - 1.0)
            if max_stake < MIN_BET_STAKE:
                return None
            result = self.matcher.match_lay(
                runner.available_to_lay,
                stake=max_stake,
                reference_price=runner.last_traded_price,
                max_price=max_price,
                already_matched_at_top=already_matched,
                force_close=force_close,
            )
            if result.matched_stake < MIN_BET_STAKE:
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
            available_at_price=result.top_level_size,
            pair_id=pair_id,
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
        each_way_divisor: float | None = None,
        winner_selection_id: int | None = None,
        number_of_places: int | None = None,
    ) -> float:
        """Settle all unsettled bets for a race and return the race P&L.

        Args:
            winning_selection_ids: Selection ID(s) that won/placed.
                For WIN markets pass the single winner.  For EACH_WAY
                markets pass a set containing WINNER + PLACED IDs.
                Accepts a single int for backward compatibility.
            market_id: If provided, only settle bets for this market.
            commission: Betfair commission rate applied to net profit
                (e.g. 0.05 for 5%).  Only deducted from winning bets.
            each_way_divisor: When not None, activates EW settlement.
                Place odds = (price - 1) / divisor + 1.  The stake is
                split internally into two equal half-legs (win + place).
                Commission is applied per-leg on each leg's gross profit.
            winner_selection_id: The single race winner (required when
                each_way_divisor is set).  Distinguishes winner (both
                legs pay) from placed-only (place leg pays, win leg
                loses).
            number_of_places: Number of EW places paid (e.g. 3).
                Stored on each settled Bet for downstream display.

        Returns:
            Net P&L for the settled bets (after commission).
        """
        # Normalise to a set
        if isinstance(winning_selection_ids, int):
            winners = {winning_selection_ids}
        else:
            winners = winning_selection_ids

        ew = each_way_divisor is not None

        race_pnl = 0.0

        for bet in self.bets:
            if bet.outcome is not BetOutcome.UNSETTLED:
                continue
            if market_id and bet.market_id != market_id:
                continue

            in_winners = bet.selection_id in winners
            is_winner = ew and bet.selection_id == winner_selection_id
            is_placed = ew and in_winners and not is_winner

            if ew and in_winners:
                # ── Each-way settlement (winner or placed) ───────────
                half = bet.matched_stake / 2.0
                price = bet.average_price
                place_profit_per_unit = (price - 1.0) / each_way_divisor

                if bet.side is BetSide.BACK:
                    if is_winner:
                        # Both legs pay.
                        win_gross = half * (price - 1.0)
                        place_gross = half * place_profit_per_unit
                        win_net = win_gross * (1.0 - commission)
                        place_net = place_gross * (1.0 - commission)
                        bet.pnl = win_net + place_net
                        # Stake was deducted at placement; return both
                        # half-stakes plus net profits.
                        self.budget += bet.matched_stake + bet.pnl
                    else:
                        # Placed only — win leg loses, place leg pays.
                        place_gross = half * place_profit_per_unit
                        place_net = place_gross * (1.0 - commission)
                        bet.pnl = -half + place_net
                        # Win half-stake already deducted; return place
                        # half-stake + place net profit.
                        self.budget += half + place_net

                elif bet.side is BetSide.LAY:
                    liability = bet.matched_stake * (price - 1.0)
                    win_liability = half * (price - 1.0)
                    place_liability = half * place_profit_per_unit

                    if is_winner:
                        # Layer loses both legs.
                        bet.pnl = -(win_liability + place_liability)
                        self.budget -= (win_liability + place_liability)
                    else:
                        # Placed only — layer wins win leg, loses place leg.
                        win_gross = half
                        win_net = win_gross * (1.0 - commission)
                        bet.pnl = win_net - place_liability
                        self.budget += win_net - place_liability

                    # Release whatever was actually reserved at placement
                    # (paired-arb freed-budget rule), falling back to full
                    # liability for non-paired lays.
                    self._open_liability -= (
                        bet.reserved_liability
                        if bet.reserved_liability is not None
                        else liability
                    )

                bet.is_each_way = True
                bet.each_way_divisor = each_way_divisor
                bet.number_of_places = number_of_places
                bet.effective_place_odds = (bet.average_price - 1.0) / each_way_divisor + 1.0
                bet.settlement_type = "ew_winner" if is_winner else "ew_placed"
                bet.outcome = BetOutcome.WON if bet.pnl > 0 else BetOutcome.LOST

            else:
                # ── Non-EW path OR unplaced runner ───────────────────
                if ew:
                    bet.is_each_way = True
                    bet.each_way_divisor = each_way_divisor
                    bet.number_of_places = number_of_places
                    bet.settlement_type = "ew_unplaced"

                if bet.side is BetSide.BACK:
                    if in_winners:
                        gross_profit = bet.matched_stake * (bet.average_price - 1.0)
                        net_profit = gross_profit * (1.0 - commission)
                        self.budget += bet.matched_stake + net_profit
                        bet.pnl = net_profit
                        bet.outcome = BetOutcome.WON
                    else:
                        bet.pnl = -bet.matched_stake
                        bet.outcome = BetOutcome.LOST

                elif bet.side is BetSide.LAY:
                    liability = bet.matched_stake * (bet.average_price - 1.0)
                    # Same freed-budget release rule for non-EW lays.
                    released = (
                        bet.reserved_liability
                        if bet.reserved_liability is not None
                        else liability
                    )
                    if in_winners:
                        self.budget -= liability
                        self._open_liability -= released
                        bet.pnl = -liability
                        bet.outcome = BetOutcome.LOST
                    else:
                        gross_profit = bet.matched_stake
                        net_profit = gross_profit * (1.0 - commission)
                        self.budget += net_profit
                        self._open_liability -= released
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
                # Release the reserved liability — honour the actual
                # paired-arb freed amount if recorded, else full liability.
                liability = bet.matched_stake * (bet.average_price - 1.0)
                self._open_liability -= (
                    bet.reserved_liability
                    if bet.reserved_liability is not None
                    else liability
                )

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

    # ── Forced-arbitrage helpers ─────────────────────────────────────────

    def get_paired_positions(
        self, market_id: str = "", commission: float = 0.05,
    ) -> list[dict]:
        """Group matched bets by ``pair_id`` for scalping diagnostics.

        Each returned dict describes one pair::

            {
                "pair_id": str,
                "aggressive": Bet | None,
                "passive":    Bet | None,
                "complete":   bool,       # both legs matched
                "locked_pnl": float,      # net PnL (after commission) if complete
            }

        Only matched/settled ``Bet`` objects are considered — unfilled
        passive orders still sit in ``passive_book.orders`` and are not
        counted here. ``market_id`` filters to a single race.
        """
        by_pair: dict[str, list[Bet]] = {}
        for bet in self.bets:
            if bet.pair_id is None:
                continue
            if market_id and bet.market_id != market_id:
                continue
            by_pair.setdefault(bet.pair_id, []).append(bet)

        results: list[dict] = []
        for pid, legs in by_pair.items():
            # First leg is always the aggressive one (it triggered the
            # pair creation); any later leg arriving via passive fill is
            # the passive counter-order.
            aggressive: Bet | None = None
            passive: Bet | None = None
            for leg in legs:
                # Aggressive back pairs to a passive lay at a lower price;
                # aggressive lay pairs to a passive back at a higher price.
                # We distinguish by price ordering — the agent's higher-
                # priced back leg is always the "sell" side.
                pass
            # Simpler: the aggressive leg is whichever back leg has the
            # higher price (for back/lay pairs this is unambiguous).
            backs = [b for b in legs if b.side is BetSide.BACK]
            lays = [b for b in legs if b.side is BetSide.LAY]
            if backs and lays:
                # Back @ high price vs lay @ low price = scalping spread.
                aggressive = max(backs, key=lambda b: b.average_price) \
                    if max(backs, key=lambda b: b.average_price).average_price \
                    > min(lays, key=lambda b: b.average_price).average_price else None
                # Simpler assignment: just pick the first of each side —
                # callers usually care about completion + locked_pnl, not
                # which was aggressive/passive.
                aggressive = backs[0] if aggressive is None else aggressive
                passive = lays[0]
            elif backs:
                aggressive = backs[0]
            elif lays:
                aggressive = lays[0]

            complete = bool(backs and lays)
            locked = 0.0
            if complete:
                back = max(backs, key=lambda b: b.average_price)
                lay = min(lays, key=lambda b: b.average_price)
                # Locked P&L = guaranteed floor across outcomes (the profit
                # the pair nails down regardless of whether the runner
                # wins or loses). Previous formula used stake × spread,
                # which is the MAX outcome of an equal-stake pair — that
                # rewarded lucky directional outcomes as if they were
                # skilled scalps. The floor formula below correctly
                # reports 0 for equal-stake pairs and the true lock amount
                # for properly-sized asymmetric hedges
                # (S_lay = S_back × P_back / P_lay).
                #
                # Commission is applied only to the winning leg in each
                # outcome (losers don't pay commission on Betfair).
                win_pnl = (
                    back.matched_stake * (back.average_price - 1.0)
                    * (1.0 - commission)
                    - lay.matched_stake * (lay.average_price - 1.0)
                )
                lose_pnl = (
                    -back.matched_stake
                    + lay.matched_stake * (1.0 - commission)
                )
                locked = max(0.0, min(win_pnl, lose_pnl))

            results.append({
                "pair_id": pid,
                "aggressive": aggressive,
                "passive": passive,
                "complete": complete,
                "locked_pnl": locked,
            })
        return results

    def get_naked_per_pair_pnls(self, market_id: str = "") -> list[float]:
        """Per-pair realised P&L of every naked aggressive leg.

        A "naked" pair is one whose aggressive leg matched but whose
        paired passive never filled before race-off (unfilled passives
        are cancelled at race-off so they never appear as settled
        bets). The returned list contains one entry per such
        aggressive leg, in ``self.bets`` insertion order, holding that
        leg's settled ``pnl``.

        Read-only and deterministic. Used by
        ``env.betfair_env._settle_current_race`` to compute the
        asymmetric per-pair naked penalty introduced by the
        ``scalping-naked-asymmetry`` plan (2026-04-18): replacing the
        aggregate ``min(0, sum(naked_pnls))`` with
        ``sum(min(0, per_pair_pnl))`` so individual naked losses can
        no longer be cancelled by lucky unrelated naked wins in the
        same race.

        A leg whose ``pnl`` is None (not yet settled) is skipped — the
        caller invokes this AFTER per-bet settlement has populated
        ``Bet.pnl`` for every matched bet.
        """
        pairs = self.get_paired_positions(market_id=market_id)
        out: list[float] = []
        for p in pairs:
            if p["complete"]:
                continue
            agg = p["aggressive"]
            if agg is None or agg.pnl is None:
                continue
            out.append(float(agg.pnl))
        return out

    def get_naked_exposure(self, market_id: str = "") -> float:
        """Sum of worst-case loss on unpaired matched bets.

        A "naked" bet is one whose ``pair_id`` is not shared with an
        opposite-side matched bet — either it was placed without a pair
        (ordinary directional bet) or its passive counter-leg never
        filled. The returned figure is the £ amount the agent stands to
        lose if every naked bet loses.
        """
        # Build pair index once — naked = bets whose pair has no
        # opposite-side partner.
        by_pair_side: dict[str, set[BetSide]] = {}
        for bet in self.bets:
            if bet.pair_id is None:
                continue
            if market_id and bet.market_id != market_id:
                continue
            by_pair_side.setdefault(bet.pair_id, set()).add(bet.side)

        naked = 0.0
        for bet in self.bets:
            if market_id and bet.market_id != market_id:
                continue
            if bet.outcome is not BetOutcome.UNSETTLED:
                continue
            is_naked = True
            if bet.pair_id is not None:
                sides = by_pair_side.get(bet.pair_id, set())
                if BetSide.BACK in sides and BetSide.LAY in sides:
                    is_naked = False
            if is_naked:
                if bet.side is BetSide.BACK:
                    naked += bet.matched_stake
                else:
                    naked += bet.matched_stake * (bet.average_price - 1.0)
        return naked

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
