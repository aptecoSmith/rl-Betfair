"""
env/exchange_matcher.py — Realistic Betfair Exchange order matching.

The Betfair Exchange matching engine does **not** walk the order book when
executing a bet.  A bet targets a single price and either matches against
counter-side liquidity available at that price *now* or rests on the book
until such liquidity appears.  The previous simulator implementation
walked the ladder level-by-level consuming everything it could, which
produced absurd average prices when the historical order book contained
stale parked orders at extreme prices (e.g. a lone £1000 level on a runner
whose true market price was 4.3).  A £100 back bet would match entirely
at the parked price and, if the runner won, settle for £94 905 of phantom
profit.

This module replaces that behaviour with a realistic single-price match:

    1. **Reference price.** The runner's last traded price (LTP) is used
       as the "truth" of where the market really is. A runner with no
       LTP (never traded) is treated as unpriceable and skipped.

    2. **Junk filter.** Ladder levels whose price deviates more than
       ``max_price_deviation_pct`` from the LTP are dropped before
       matching. This cuts stale parked orders at Betfair's extreme
       ends (£1–£1000) without destroying genuine fast-market levels.

    3. **Single-price match.** After filtering, the best remaining level
       (highest back price for a back bet, lowest lay price for a lay
       bet) is used as the sole fill price. The bet matches up to that
       level's available size; anything over is left unmatched (the
       caller may treat that as cancelled).  **No ladder walking.**

    4. **Hard cap.** An optional ``max_price`` refuses the bet entirely
       if the best filtered price still exceeds it — this is the
       plumbing behind ``betting_constraints.max_back_price`` and
       ``max_lay_price`` in ``config.yaml``.

The class is deliberately dependency-free (only the standard library and
a small structural protocol for price-level inputs) so the same file can
be vendored into the ``ai-betfair`` live-inference project without
modification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable


@runtime_checkable
class PriceLevel(Protocol):
    """Anything with ``price`` and ``size`` attributes.

    Both :class:`data.episode_builder.PriceSize` (rl-betfair) and the
    equivalent dataclass in the ai-betfair live feed satisfy this, so
    the matcher accepts either without an adapter layer.
    """

    price: float
    size: float


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Result of attempting to match a bet.

    Attributes
    ----------
    matched_stake:
        Amount of the requested stake that actually matched.  Always
        ``<= requested_stake`` and ``<= top_level.size``.
    unmatched_stake:
        Leftover stake that could not be matched at the target price.
        Cancelled by convention — no resting orders.
    average_price:
        The single fill price.  ``0.0`` when ``matched_stake == 0``.
    skipped_reason:
        Human-readable reason the bet was refused or partially filled,
        or ``None`` on a clean full fill.  Useful for logging / debug.
    top_level_size:
        Raw size on the best ladder level *before* self-depletion.
        ``0.0`` when no valid level was found.  Diagnostic field —
        compare against ``matched_stake`` to verify the fill didn't
        exceed available liquidity.
    """

    matched_stake: float
    unmatched_stake: float
    average_price: float
    skipped_reason: str | None = None
    top_level_size: float = 0.0

    @property
    def fully_matched(self) -> bool:
        return self.unmatched_stake <= 0.0 and self.matched_stake > 0.0


def passes_junk_filter(
    price: float,
    reference_price: float,
    max_dev_pct: float,
) -> bool:
    """True iff *price* is within ±*max_dev_pct* of *reference_price*.

    Pure-function mirror of the filter applied inside
    :meth:`ExchangeMatcher._match`. Exported so the oracle and other
    offline tools can apply the same filter without instantiating the
    class.
    """
    if reference_price is None or reference_price <= 0.0 or price <= 0.0:
        return False
    lo = reference_price * (1.0 - max_dev_pct)
    hi = reference_price * (1.0 + max_dev_pct)
    return lo <= price <= hi


def passes_price_cap(price: float, max_price: float | None) -> bool:
    """True iff *price* does not exceed the optional hard cap.

    Pure-function mirror of the cap check inside
    :meth:`ExchangeMatcher._match`.
    """
    return max_price is None or price <= max_price


class ExchangeMatcher:
    """Single-price order matcher with junk-level filtering.

    Parameters
    ----------
    max_price_deviation_pct:
        Fractional deviation from the runner's LTP allowed before a
        ladder level is treated as stale parked liquidity and dropped.
        For ``0.5`` (the default), any level outside ``[LTP × 0.5,
        LTP × 1.5]`` is discarded.  This default is generous enough
        to preserve fast-market moves (prices rarely double or halve
        in one tick on a pre-race horse market) while reliably cutting
        the £1–£1000 parked orders routinely present in real Betfair
        ladders.

    Notes
    -----
    The matcher is stateless apart from its configuration, so one
    instance can be shared across a whole training run or inference
    session.
    """

    def __init__(
        self,
        max_price_deviation_pct: float = 0.5,
        force_close_max_deviation_pct: float | None = None,
    ) -> None:
        if max_price_deviation_pct <= 0.0:
            raise ValueError(
                f"max_price_deviation_pct must be > 0 "
                f"(got {max_price_deviation_pct})"
            )
        self.max_price_deviation_pct = float(max_price_deviation_pct)
        # Force-close safety barrier (2026-05-31). The force-close path
        # historically SKIPPED the junk filter entirely (``force_close=True``
        # → "any priceable level is a valid close target") on the rationale
        # that crossing a thin book beats leaving a pair naked. A
        # fill-price forensic (plans/bc-getting-it-right) showed that holds
        # in the median (~+3% past LTP) but the "any price" tail has NO
        # guardrail — ``max_lay_price`` was ``null`` in config, so a thin
        # near-off book could fill a close at a junk level (e.g. 2× LTP+).
        # When this is set, the force-close path applies a WIDER-than-open
        # deviation bound: it may cross up to ±this fraction of the LTP, but
        # refuses beyond it (the close fails → the pair settles naked, whose
        # downside is bounded by the original aggressive stake — strictly
        # safer than crossing into junk). ``None`` (default) preserves the
        # legacy full-skip behaviour, so existing callers / tests that don't
        # set it stay byte-identical. Must be > 0 when set.
        if (
            force_close_max_deviation_pct is not None
            and force_close_max_deviation_pct <= 0.0
        ):
            raise ValueError(
                f"force_close_max_deviation_pct must be > 0 when set "
                f"(got {force_close_max_deviation_pct})"
            )
        self.force_close_max_deviation_pct: float | None = (
            float(force_close_max_deviation_pct)
            if force_close_max_deviation_pct is not None
            else None
        )

    # ── Public API ──────────────────────────────────────────────────

    def _force_close_filter(
        self,
        valid_levels: list["PriceLevel"],
        reference_price: float | None,
    ) -> list["PriceLevel"]:
        """Apply the force-close deviation barrier to already-validated levels.

        ``valid_levels`` must already be filtered to ``price > 0, size > 0``.
        When ``force_close_max_deviation_pct`` is set AND a reference price
        is available, drop levels more than that fraction from the LTP (a
        WIDER bound than the open-side ``max_price_deviation_pct``). When the
        barrier is unset, or no reference price exists to bound against,
        return ``valid_levels`` unchanged — the legacy full-skip behaviour.
        Keeping this in ONE place mirrors the open-side junk-filter contract.
        """
        dev = self.force_close_max_deviation_pct
        if dev is None:
            # Barrier OFF: legacy full-skip — any priceable level is valid.
            return valid_levels
        if reference_price is None or reference_price <= 0.0:
            # Barrier ON but NO reference to judge against (runner has no
            # LTP). We can't tell a fair level from junk, so refuse rather
            # than cross blind — the pair settles naked, downside bounded by
            # the original aggressive stake (operator decision 2026-05-31,
            # plans/bc-getting-it-right; 0 such cases in the holdout
            # forensic, so no economic impact — pure safety completeness).
            return []
        lo = reference_price * (1.0 - dev)
        hi = reference_price * (1.0 + dev)
        return [lv for lv in valid_levels if lo <= lv.price <= hi]

    def pick_top_level(
        self,
        levels: Iterable[PriceLevel],
        reference_price: float,
        lower_is_better: bool,
        *,
        force_close: bool = False,
    ) -> "PriceLevel | None":
        """Return the best post-filter ``PriceLevel`` (price + size).

        Mirrors :meth:`pick_top_price` but returns the full level so
        callers can inspect both price AND size. Used by R4
        (robust-phenotype, 2026-05-19) — the liquidity-floor open
        gate refuses opens when the post-junk-filter top level's
        ``size`` is below threshold. Returning the level rather than
        re-filtering in the caller keeps the junk-filter contract in
        ONE place (hard_constraints §5 of robust-phenotype).
        """
        lst = list(levels)
        if not lst:
            return None
        valid = [lv for lv in lst if lv.price > 0.0 and lv.size > 0.0]
        if not valid:
            return None
        if force_close:
            filtered = self._force_close_filter(valid, reference_price)
        else:
            if reference_price is None or reference_price <= 0.0:
                return None
            lo = reference_price * (1.0 - self.max_price_deviation_pct)
            hi = reference_price * (1.0 + self.max_price_deviation_pct)
            filtered = [lv for lv in valid if lo <= lv.price <= hi]
        if not filtered:
            return None
        return (
            min(filtered, key=lambda lv: lv.price)
            if lower_is_better
            else max(filtered, key=lambda lv: lv.price)
        )

    def pick_top_price(
        self,
        levels: Iterable[PriceLevel],
        reference_price: float,
        lower_is_better: bool,
        *,
        force_close: bool = False,
    ) -> float | None:
        """Return the best post-filter top-of-book price without executing a fill.

        Returns ``None`` if there are no valid levels (no LTP, empty ladder,
        or all levels outside the junk filter).  Used by ``BetManager`` to
        peek at the fill price before consulting the ``_matched_at_level``
        accumulator, keeping filter logic in one place.

        ``force_close=True`` flips this to the force-close semantics
        (see :meth:`_match`): the LTP requirement is dropped and the
        ±``max_price_deviation_pct`` junk filter is skipped. Useful for
        env-initiated close-out attempts at T−N where leaving a pair
        naked is worse than crossing into a thin / unpriced book.
        """
        lst = list(levels)
        if not lst:
            return None
        valid = [lv for lv in lst if lv.price > 0.0 and lv.size > 0.0]
        if not valid:
            return None
        if force_close:
            filtered = self._force_close_filter(valid, reference_price)
        else:
            if reference_price is None or reference_price <= 0.0:
                return None
            lo = reference_price * (1.0 - self.max_price_deviation_pct)
            hi = reference_price * (1.0 + self.max_price_deviation_pct)
            filtered = [lv for lv in valid if lo <= lv.price <= hi]
        if not filtered:
            return None
        top = min(filtered, key=lambda lv: lv.price) if lower_is_better \
            else max(filtered, key=lambda lv: lv.price)
        return top.price

    def match_back(
        self,
        available_to_back: Iterable[PriceLevel],
        stake: float,
        reference_price: float,
        max_price: float | None = None,
        already_matched_at_top: float = 0.0,
        *,
        force_close: bool = False,
        walk_to_price: float | None = None,
    ) -> MatchResult:
        """Attempt to fill a back bet against the back side of the book.

        ``available_to_back`` contains the resting lay orders that a
        backer can match against (the blue column on the Betfair UI).
        The *best* price is the **highest** back offer, because a
        higher price means greater profit if the selection wins.

        ``walk_to_price`` (close-walk, 2026-05-30): when set, the match
        walks DOWN the ladder (worse back prices) consuming successive
        levels until the stake is filled or a level's price drops below
        ``walk_to_price``. See :meth:`_match`.
        """
        return self._match(
            list(available_to_back),
            stake=stake,
            reference_price=reference_price,
            max_price=max_price,
            lower_is_better=False,
            already_matched_at_top=already_matched_at_top,
            force_close=force_close,
            walk_to_price=walk_to_price,
        )

    def match_lay(
        self,
        available_to_lay: Iterable[PriceLevel],
        stake: float,
        reference_price: float,
        max_price: float | None = None,
        already_matched_at_top: float = 0.0,
        *,
        force_close: bool = False,
        walk_to_price: float | None = None,
    ) -> MatchResult:
        """Attempt to fill a lay bet against the lay side of the book.

        ``available_to_lay`` contains the resting back orders that a
        layer can match against (the pink column on the Betfair UI).
        The *best* price is the **lowest** lay offer, because a lower
        price means less liability for the layer.

        ``walk_to_price`` (close-walk, 2026-05-30): when set, the match
        walks UP the ladder (worse lay prices) consuming successive
        levels until the stake is filled or a level's price exceeds
        ``walk_to_price``. See :meth:`_match`.
        """
        return self._match(
            list(available_to_lay),
            stake=stake,
            reference_price=reference_price,
            max_price=max_price,
            lower_is_better=True,
            already_matched_at_top=already_matched_at_top,
            force_close=force_close,
            walk_to_price=walk_to_price,
        )

    # ── Internals ───────────────────────────────────────────────────

    def _match(
        self,
        levels: list[PriceLevel],
        *,
        stake: float,
        reference_price: float,
        max_price: float | None,
        lower_is_better: bool,
        already_matched_at_top: float = 0.0,
        force_close: bool = False,
        walk_to_price: float | None = None,
    ) -> MatchResult:
        if stake <= 0.0:
            return MatchResult(0.0, 0.0, 0.0, "non-positive stake")
        if not levels:
            return MatchResult(0.0, stake, 0.0, "empty ladder")

        # Force-close path (arb-signal-cleanup, 2026-04-21): crossing to
        # close an already-matched aggressive leg is usually better than
        # leaving the pair naked through the off (±£100s of directional
        # variance). So for close-out attempts at T−N we drop the LTP
        # REQUIREMENT (a thin / unpriced book is still a valid close
        # target). Historically this ALSO skipped the deviation filter
        # entirely — but with ``max_lay_price: null`` that left the
        # force-close LAY with NO upper-price guardrail, so a thin
        # near-off book could fill a close at a junk level (2× LTP+).
        # Safety barrier (2026-05-31, plans/bc-getting-it-right): when
        # ``force_close_max_deviation_pct`` is set, apply a WIDER-than-open
        # deviation bound here. The close may cross up to ±that fraction of
        # the LTP but is refused beyond it — the pair then settles naked
        # (downside bounded by the original aggressive stake), strictly
        # safer than crossing into junk. The hard ``max_price`` cap still
        # applies on top. See CLAUDE.md "Force-close at T−N".
        if force_close:
            valid = [
                lv for lv in levels if lv.price > 0.0 and lv.size > 0.0
            ]
            filtered = self._force_close_filter(valid, reference_price)
            if not filtered:
                return MatchResult(
                    0.0, stake, 0.0,
                    "no force-close level within deviation barrier",
                )
        else:
            if reference_price is None or reference_price <= 0.0:
                return MatchResult(0.0, stake, 0.0, "no LTP for runner")
            # Filter junk: drop any level whose price is more than
            # ``max_price_deviation_pct`` away from the LTP.
            lo = reference_price * (1.0 - self.max_price_deviation_pct)
            hi = reference_price * (1.0 + self.max_price_deviation_pct)
            filtered = [
                lv for lv in levels
                if lv.price > 0.0 and lv.size > 0.0 and lo <= lv.price <= hi
            ]
            if not filtered:
                return MatchResult(
                    0.0, stake, 0.0,
                    f"all ladder levels outside ±{self.max_price_deviation_pct:.0%} "
                    f"of LTP {reference_price:.2f}",
                )

        # The input ladder is ordered best-first by the upstream parser,
        # but we don't trust that after filtering — pick the best level
        # explicitly. ``lower_is_better`` flips which end of the sorted
        # list counts as "best" (back bets want the lowest lay price,
        # lay bets want the highest back price).
        top = min(filtered, key=lambda lv: lv.price) if lower_is_better \
            else max(filtered, key=lambda lv: lv.price)

        # Hard cap from betting_constraints.* — applies to both sides
        # with the same semantic: "refuse if the only prices I could
        # fill at are beyond my pain threshold".
        if max_price is not None and top.price > max_price:
            return MatchResult(
                0.0, stake, 0.0,
                f"best price {top.price:.2f} exceeds max_price cap {max_price:.2f}",
                top_level_size=top.size,
            )

        # ── Close-walk path (2026-05-30) ─────────────────────────────────
        # When ``walk_to_price`` is supplied (close / force-close only),
        # fill across SUCCESSIVE levels from the best toward — and
        # including — ``walk_to_price``, until the stake is exhausted.
        # This is a bounded LIMIT order, not the unbounded market sweep
        # the no-walk rule bans: the OPEN path never passes
        # ``walk_to_price`` so it keeps the strict single-level contract.
        # See CLAUDE.md "Order matching" + docs/betfair_market_model.md
        # §2 ("matches at the named price OR BETTER, leaving the
        # remainder resting") + findings.md KEY FINDING #2. Walking lets
        # a close complete the equal-profit hedge instead of leaving the
        # aggressive leg directionally exposed against a thin top level.
        #
        # ``lower_is_better`` (lay closing a back): walk to HIGHER prices,
        # taking every level whose price <= walk_to_price.
        # ``not lower_is_better`` (back closing a lay): walk to LOWER
        # prices, taking every level whose price >= walk_to_price.
        # The hard ``max_price`` cap is enforced on every walked level.
        if walk_to_price is not None:
            ordered = sorted(
                filtered,
                key=lambda lv: lv.price,
                reverse=not lower_is_better,
            )
            remaining = stake
            filled_size = 0.0
            filled_value = 0.0
            for i, lv in enumerate(ordered):
                if remaining <= 0.0:
                    break
                # Stop once we'd cross past the walk limit.
                if lower_is_better and lv.price > walk_to_price:
                    break
                if (not lower_is_better) and lv.price < walk_to_price:
                    break
                # Never walk past the hard price cap.
                if max_price is not None and lv.price > max_price:
                    break
                # Self-depletion only ever recorded against the best
                # level (the OPEN path's single-level fill); subtract it
                # from the first level here, deeper levels are untouched.
                avail = lv.size - (already_matched_at_top if i == 0 else 0.0)
                if avail <= 0.0:
                    continue
                take = min(remaining, avail)
                filled_size += take
                filled_value += take * lv.price
                remaining -= take
            if filled_size <= 0.0:
                return MatchResult(
                    0.0, stake, 0.0,
                    "self-depletion exhausted level",
                    top_level_size=top.size,
                )
            return MatchResult(
                matched_stake=filled_size,
                unmatched_stake=stake - filled_size,
                average_price=filled_value / filled_size,
                skipped_reason=None,
                top_level_size=top.size,
            )

        # Single-price fill: take up to the available size after subtracting
        # any stake the agent already matched at this level in this race.
        # No walking to worse levels.
        adjusted_size = max(0.0, top.size - already_matched_at_top)
        if adjusted_size == 0.0:
            return MatchResult(
                0.0, stake, 0.0,
                "self-depletion exhausted level",
                top_level_size=top.size,
            )
        matched = min(stake, adjusted_size)
        unmatched = stake - matched
        return MatchResult(
            matched_stake=matched,
            unmatched_stake=unmatched,
            average_price=top.price,
            skipped_reason=None,
            top_level_size=top.size,
        )


# ── Module-level default instance ────────────────────────────────────
#
# Most callers want the default 50 % junk-filter tolerance and don't
# need to vary it per-bet, so they can just import this singleton
# instead of instantiating their own.

DEFAULT_MATCHER = ExchangeMatcher(max_price_deviation_pct=0.5)
