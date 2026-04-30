"""Label simulation for the Phase 0 supervised scorer.

Per opportunity = ``(date, market_id, runner_idx, tick_idx, side)``:

* ``label = 1.0`` iff a back/lay aggressive open at this tick on this
  side could be placed AND the equal-profit passive on the opposite
  side would naturally fill before the force-close window starts.
* ``label = 0.0`` iff the open could be placed but the passive never
  matures — the pair would either be force-closed at T-N or settle
  naked.
* ``label = NaN`` (caller masks out of training) iff the open is
  infeasible (no LTP, empty book, hard cap exceeded, refusal under the
  same matcher contract the env uses at runtime).

Implementation reuses ``env.bet_manager.BetManager`` end-to-end —
``place_back`` / ``place_lay`` for the aggressive, the explicit-price
``passive_book.place`` for the resting passive, ``passive_book.on_tick``
for queue-position simulation, and the same ``force_close=True`` matcher
path the env hits at T-N. Any divergence from this code path would
miscalibrate the scorer at Phase 1 wiring time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal

from data.episode_builder import Race, Tick
from env.bet_manager import MIN_BET_STAKE, BetManager, BetSide
from env.exchange_matcher import DEFAULT_MATCHER, ExchangeMatcher
from env.scalping_math import (
    equal_profit_back_stake,
    equal_profit_lay_stake,
    locked_pnl_per_unit_stake,
    min_arb_ticks_for_profit,
)
from env.tick_ladder import tick_offset


# Match env defaults: see config.yaml.
_DEFAULT_COMMISSION = 0.05
_DEFAULT_MAX_BACK_PRICE = 50.0
_DEFAULT_MAX_LAY_PRICE: float | None = None
_DEFAULT_FORCE_CLOSE_THRESHOLD_SEC = 30.0
_DEFAULT_BACK_STAKE = 10.0  # arbitrary positive — matters for MIN_BET_STAKE
                              # gate and for sizing the equal-profit passive
_DEFAULT_MAX_ARB_TICKS = 25  # MAX_ARB_TICKS in env/betfair_env.py


# ── arb_ticks default: 20 (NOT min_arb_ticks_for_profit) ────────────────
#
# This is a TEMPORARY label-balance accommodation documented in
# ``plans/rewrite/phase-0-supervised-scorer/session_01_findings.md``
# and the auto-memory ``book_depth_n3_widen_later`` note.
#
# Why arb_ticks = 20:
#
# The captured book depth in ``data/processed/<date>.parquet`` is 3
# levels per side (StreamRecorder1 polls top-3 only). With this
# constraint, two failure modes show up at the extremes of the
# ``arb_ticks`` spectrum, both producing pathologically imbalanced
# labels:
#
# * ``arb_ticks = 1`` (close to LTP): the passive sits on the very
#   next tick. Any small LTP fluctuation crosses it. Empirical
#   matured rate on the 19-day dataset: **99.81 %** — model has no
#   discriminative gradient.
#
# * ``arb_ticks = min_arb_ticks_for_profit`` (8-20 ticks for typical
#   horse-market prices): the passive lands OUTSIDE the captured
#   3-level book in ~100 % of cases (verified empirically on
#   2026-04-21 races 0-7: 3,142 / 3,142 placements at
#   ``queue_ahead = 0``). Under ``PassiveOrderBook.on_tick`` Phase 2
#   semantics (``traded_volume_since_placement >=
#   queue_ahead + already_filled``), an order with ``queue_ahead = 0``
#   fills on the first tick after placement REGARDLESS of the
#   LTP-crossability gate (which only filters Phase 1's volume
#   accumulation, not Phase 2's threshold check). Empirical matured
#   rate: **83.6 %** — also too imbalanced for clean training, and
#   the matures are nearly all "fast-fill artifact", not real queue
#   crossings.
#
# Sweep on 2026-04-21 (15 races):
#
#   arb_ticks=1   : 99.8% matured / 0.0% fc / 0.2% naked
#   arb_ticks=4   : 99.6% matured / 0.2% fc / 0.2% naked
#   arb_ticks=8   : 95.5% matured / 3.7% fc / 0.9% naked
#   arb_ticks=15  : 79.7% matured / 17.8% fc / 2.5% naked
#   arb_ticks=20  : 63.7% matured / 32.2% fc / 4.1% naked   ← chosen
#   arb_ticks=25  : 49.8% matured / 45.0% fc / 5.2% naked
#
# 20 ticks gives the most workable label balance for Session 02
# (~36% negative class — class weights handle this trivially) while
# staying within the env's ``MAX_ARB_TICKS = 25`` runtime cap.
#
# When can we revert to ``min_arb_ticks_for_profit`` (``arb_ticks=None``)?
#
# Once StreamRecorder1's book-depth widening lands and at least
# ~20 captured levels per side become the default for new days, OR
# a historical re-capture extends old days the same way. With deeper
# book capture, an 8-20-tick passive will typically have observable
# ``queue_ahead`` and the fast-fill artifact disappears. At that
# point ``arb_ticks=None`` is preferable because it picks the
# minimum spread that's profitable after commission — a principled
# choice rather than the empirically-tuned 20 here.
#
# The constructor parameter ``arb_ticks`` is exposed so the operator
# can override per-run without code edit (e.g. once book depth
# widens, pass ``arb_ticks=None`` to re-enable the dynamic
# ``min_arb_ticks_for_profit`` lookup).
_DEFAULT_ARB_TICKS: int | None = 20


class LabelOutcome(str, Enum):
    """Resolution class of one simulated open."""

    MATURED = "matured"
    FORCE_CLOSED = "force_closed"
    NAKED = "naked"
    # NaN-label feasibility refusals — kept distinct so sanity checks
    # can quantify drop reasons without re-running the simulator.
    INFEASIBLE_INACTIVE = "infeasible_inactive"
    INFEASIBLE_NO_LTP = "infeasible_no_ltp"
    INFEASIBLE_AGG_REFUSED = "infeasible_agg_refused"
    INFEASIBLE_NO_PROFITABLE_SPREAD = "infeasible_no_profitable_spread"
    INFEASIBLE_PASSIVE_REFUSED = "infeasible_passive_refused"
    INFEASIBLE_IN_PLAY = "infeasible_in_play"


@dataclass(frozen=True, slots=True)
class LabelResult:
    """Outcome of one simulated opportunity."""

    label: float | None  # None denotes NaN — caller writes np.nan into parquet
    outcome: LabelOutcome
    # Diagnostic — populated when the open succeeded; useful for
    # sanity-checking the dataset and for downstream feature-importance
    # debugging. ``None`` on infeasible cases.
    aggressive_price: float | None = None
    passive_price: float | None = None


_PAIR_ID = "scorer_v1_simulated"


class LabelGenerator:
    """Stateless label simulator.

    One ``LabelGenerator`` instance can serve a whole dataset build —
    each ``generate`` call constructs a fresh ``BetManager`` so there's
    no cross-opportunity contamination. The matcher is shared
    (it's stateless apart from its junk-filter constant).
    """

    def __init__(
        self,
        *,
        matcher: ExchangeMatcher = DEFAULT_MATCHER,
        commission: float = _DEFAULT_COMMISSION,
        max_back_price: float | None = _DEFAULT_MAX_BACK_PRICE,
        max_lay_price: float | None = _DEFAULT_MAX_LAY_PRICE,
        force_close_threshold_sec: float = _DEFAULT_FORCE_CLOSE_THRESHOLD_SEC,
        back_stake: float = _DEFAULT_BACK_STAKE,
        max_arb_ticks: int = _DEFAULT_MAX_ARB_TICKS,
        arb_ticks: int | None = _DEFAULT_ARB_TICKS,
        starting_budget: float = 1000.0,
        fill_mode: Literal["volume", "pragmatic"] = "volume",
    ) -> None:
        """``arb_ticks``: fixed tick offset between aggressive and
        passive legs. ``None`` falls back to
        ``min_arb_ticks_for_profit`` (the spread that clears
        commission). Default ``1`` is a temporary book-depth
        accommodation — see the module-level ``_DEFAULT_ARB_TICKS``
        comment for why and when to revert.
        """
        if back_stake < MIN_BET_STAKE:
            raise ValueError(
                f"back_stake ({back_stake}) must be ≥ MIN_BET_STAKE "
                f"({MIN_BET_STAKE})",
            )
        if arb_ticks is not None and arb_ticks < 1:
            raise ValueError(
                f"arb_ticks must be ≥ 1 or None, got {arb_ticks}",
            )
        self.matcher = matcher
        self.commission = float(commission)
        self.max_back_price = max_back_price
        self.max_lay_price = max_lay_price
        self.force_close_threshold_sec = float(force_close_threshold_sec)
        self.back_stake = float(back_stake)
        self.max_arb_ticks = int(max_arb_ticks)
        self.arb_ticks: int | None = arb_ticks
        self.starting_budget = float(starting_budget)
        self.fill_mode: Literal["volume", "pragmatic"] = fill_mode

    def generate(
        self,
        race: Race,
        tick_idx: int,
        runner_idx: int,
        side: Literal["back", "lay"],
    ) -> LabelResult:
        """Simulate one open at ``(tick_idx, runner_idx, side)``."""
        if side not in ("back", "lay"):
            raise ValueError(f"side must be 'back' or 'lay', got {side!r}")
        if not (0 <= tick_idx < len(race.ticks)):
            raise IndexError(
                f"tick_idx {tick_idx} out of range [0, {len(race.ticks)})",
            )
        start_tick = race.ticks[tick_idx]

        # Pre-race only — placing once the gate's open is a different
        # micro-structure regime and the env restricts placement to
        # pre-race ticks. The model would never need to score in-play
        # opportunities so flag and drop.
        if start_tick.in_play:
            return LabelResult(None, LabelOutcome.INFEASIBLE_IN_PLAY)

        if not (0 <= runner_idx < len(start_tick.runners)):
            raise IndexError(
                f"runner_idx {runner_idx} out of range "
                f"[0, {len(start_tick.runners)})",
            )
        runner_snap = start_tick.runners[runner_idx]
        target_sid = runner_snap.selection_id

        if runner_snap.status != "ACTIVE":
            return LabelResult(None, LabelOutcome.INFEASIBLE_INACTIVE)
        ltp = runner_snap.last_traded_price
        if ltp is None or ltp <= 1.0:
            return LabelResult(None, LabelOutcome.INFEASIBLE_NO_LTP)

        # Fresh BetManager — no cross-opportunity contamination. Use a
        # generous budget so the equal-profit passive (which can be
        # several × the aggressive stake at extreme prices) reserves
        # cleanly inside the budget.
        bm = BetManager(
            starting_budget=self.starting_budget,
            matcher=self.matcher,
            fill_mode=self.fill_mode,
        )

        # ── Aggressive open ────────────────────────────────────────────
        if side == "back":
            agg = bm.place_back(
                runner_snap,
                self.back_stake,
                market_id=race.market_id,
                max_price=self.max_back_price,
                pair_id=_PAIR_ID,
            )
        else:
            agg = bm.place_lay(
                runner_snap,
                self.back_stake,
                market_id=race.market_id,
                max_price=self.max_lay_price,
                pair_id=_PAIR_ID,
            )
        if agg is None:
            return LabelResult(None, LabelOutcome.INFEASIBLE_AGG_REFUSED)

        # ── Choose passive offset + price + stake ──────────────────────
        # ``arb_ticks=None`` re-enables the profitable-spread lookup;
        # the default ``arb_ticks=1`` is the book-depth-aware override
        # documented at module top.
        agg_side_str: Literal["back", "lay"] = "back" if side == "back" else "lay"
        if self.arb_ticks is None:
            n_ticks = min_arb_ticks_for_profit(
                agg.average_price,
                agg_side_str,
                self.commission,
                max_ticks=self.max_arb_ticks,
            )
            if n_ticks is None:
                return LabelResult(
                    None, LabelOutcome.INFEASIBLE_NO_PROFITABLE_SPREAD,
                    aggressive_price=agg.average_price,
                )
        else:
            n_ticks = self.arb_ticks
        # Direction: aggressive back wants a passive lay BELOW
        # (P_back > P_lay scalping spread); aggressive lay wants a
        # passive back ABOVE.
        direction = -1 if side == "back" else +1
        passive_price = tick_offset(agg.average_price, n_ticks, direction)
        if passive_price <= 1.0:
            return LabelResult(
                None, LabelOutcome.INFEASIBLE_NO_PROFITABLE_SPREAD,
                aggressive_price=agg.average_price,
                passive_price=passive_price,
            )

        # Equal-profit sizing — back-first uses lay sizing helper, etc.
        try:
            if side == "back":
                passive_stake = equal_profit_lay_stake(
                    back_stake=agg.matched_stake,
                    back_price=agg.average_price,
                    lay_price=passive_price,
                    commission=self.commission,
                )
                passive_side = BetSide.LAY
            else:
                passive_stake = equal_profit_back_stake(
                    lay_stake=agg.matched_stake,
                    lay_price=agg.average_price,
                    back_price=passive_price,
                    commission=self.commission,
                )
                passive_side = BetSide.BACK
        except ValueError:
            return LabelResult(
                None, LabelOutcome.INFEASIBLE_NO_PROFITABLE_SPREAD,
                aggressive_price=agg.average_price,
                passive_price=passive_price,
            )
        if passive_stake < MIN_BET_STAKE:
            # Under MIN_BET_STAKE the passive can never be placed — so
            # there's no realistic open here. Treat as infeasible.
            return LabelResult(
                None, LabelOutcome.INFEASIBLE_PASSIVE_REFUSED,
                aggressive_price=agg.average_price,
                passive_price=passive_price,
            )

        # ── Place passive at explicit price ────────────────────────────
        race_off_ts = race.market_start_time.timestamp()
        time_to_off_at_open = race_off_ts - start_tick.timestamp.timestamp()
        passive_order = bm.passive_book.place(
            runner_snap,
            passive_stake,
            passive_side,
            market_id=race.market_id,
            tick_index=tick_idx,
            price=passive_price,
            pair_id=_PAIR_ID,
            time_to_off=time_to_off_at_open,
        )
        if passive_order is None:
            return LabelResult(
                None, LabelOutcome.INFEASIBLE_PASSIVE_REFUSED,
                aggressive_price=agg.average_price,
                passive_price=passive_price,
            )

        # ── Walk forward — fill / force-close / naked ──────────────────
        outcome = self._walk_forward(
            bm, race, tick_idx, target_sid, race_off_ts,
            agg.side, agg.matched_stake, agg.average_price,
        )
        return LabelResult(
            label=1.0 if outcome is LabelOutcome.MATURED else 0.0,
            outcome=outcome,
            aggressive_price=agg.average_price,
            passive_price=passive_price,
        )

    def _walk_forward(
        self,
        bm: BetManager,
        race: Race,
        start_tick_idx: int,
        target_sid: int,
        race_off_ts: float,
        agg_side: BetSide,
        agg_matched_stake: float,
        agg_price: float,
    ) -> LabelOutcome:
        """Iterate ticks from ``start_tick_idx + 1`` onward.

        Returns the first resolution observed:
        - ``MATURED`` once ``passive_book.on_tick`` produces a fill on
          our pair_id (the resting passive crossed by traded volume).
        - ``FORCE_CLOSED`` if ``time_to_off`` falls inside the threshold
          AND the same env force-close path the runtime hits at T-N
          successfully places a close leg.
        - ``NAKED`` if the force-close window is hit but the matcher
          refuses, or if the race-off arrives without a fill.
        """
        force_threshold = self.force_close_threshold_sec
        for j in range(start_tick_idx + 1, len(race.ticks)):
            tick: Tick = race.ticks[j]
            ts = tick.timestamp.timestamp()
            time_to_off = race_off_ts - ts

            # Step 1: advance the passive's queue accumulator.
            bm.passive_book.on_tick(tick, tick_index=j)

            # Did our passive fill this tick? Look for a Bet on
            # the resting side with our pair_id.
            for bet in bm.bets:
                if bet.pair_id != _PAIR_ID:
                    continue
                # The aggressive leg landed in bm.bets at placement and
                # is on agg_side. Any leg on the OPPOSITE side is the
                # passive — we matured.
                if bet.side != agg_side:
                    return LabelOutcome.MATURED

            # Step 2: have we hit the force-close window?
            #
            # Important: in-play ticks are excluded — the env only fires
            # force-close on pre-race ticks. The passive accumulator
            # already advanced above; if a fill was going to land it
            # already did.
            if (
                not tick.in_play
                and 0.0 <= time_to_off <= force_threshold
            ):
                # Find the runner snap on this tick (selection_id must
                # match — runner_idx ordering may shift across ticks).
                runner_now = next(
                    (r for r in tick.runners if r.selection_id == target_sid),
                    None,
                )
                if runner_now is None or runner_now.status != "ACTIVE":
                    return LabelOutcome.NAKED

                # Cancel the unfilled passive first (matches
                # _attempt_close ordering — frees the budget reservation
                # before the close-leg places). If cancel returns None
                # the order was already filled (race condition with the
                # check above) — re-loop in case we missed it; but this
                # branch is unreachable once we've reached force-close.
                target_passive = next(
                    (
                        o for o in bm.passive_book.orders
                        if o.pair_id == _PAIR_ID
                        and o.selection_id == target_sid
                    ),
                    None,
                )
                if target_passive is not None:
                    bm.passive_book.cancel_order(target_passive, reason="close")

                # Place the force-close leg via the same place_back /
                # place_lay path the env uses, with force_close=True so
                # the relaxed matcher applies (LTP not required, junk
                # filter dropped, hard cap still enforced).
                if agg_side is BetSide.BACK:
                    # Close a back-aggressive by aggressive lay.
                    close_price = self.matcher.pick_top_price(
                        runner_now.available_to_lay,
                        reference_price=runner_now.last_traded_price,
                        lower_is_better=True,
                        force_close=True,
                    )
                    if close_price is None:
                        return LabelOutcome.NAKED
                    try:
                        close_stake = equal_profit_lay_stake(
                            back_stake=agg_matched_stake,
                            back_price=agg_price,
                            lay_price=close_price,
                            commission=self.commission,
                        )
                    except ValueError:
                        return LabelOutcome.NAKED
                    close_bet = bm.place_lay(
                        runner_now, close_stake,
                        market_id=race.market_id,
                        max_price=self.max_lay_price,
                        pair_id=_PAIR_ID,
                        force_close=True,
                    )
                else:
                    close_price = self.matcher.pick_top_price(
                        runner_now.available_to_back,
                        reference_price=runner_now.last_traded_price,
                        lower_is_better=False,
                        force_close=True,
                    )
                    if close_price is None:
                        return LabelOutcome.NAKED
                    try:
                        close_stake = equal_profit_back_stake(
                            lay_stake=agg_matched_stake,
                            lay_price=agg_price,
                            back_price=close_price,
                            commission=self.commission,
                        )
                    except ValueError:
                        return LabelOutcome.NAKED
                    close_bet = bm.place_back(
                        runner_now, close_stake,
                        market_id=race.market_id,
                        max_price=self.max_back_price,
                        pair_id=_PAIR_ID,
                        force_close=True,
                    )
                if close_bet is None:
                    return LabelOutcome.NAKED
                return LabelOutcome.FORCE_CLOSED

        # Race ran out of ticks — the env's race-off path cancels the
        # passive. Naked.
        return LabelOutcome.NAKED


__all__ = ["LabelGenerator", "LabelOutcome", "LabelResult"]


# locked_pnl_per_unit_stake / datetime kept on the import surface for
# downstream tests that want to assert the simulator matches the env's
# spread-profitability logic; ruff-noqa would also work but explicit
# re-export documents intent.
_ = locked_pnl_per_unit_stake
_ = datetime
