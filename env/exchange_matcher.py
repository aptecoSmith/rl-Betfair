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
       (lowest lay price for a back bet, highest back price for a lay
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
    """

    matched_stake: float
    unmatched_stake: float
    average_price: float
    skipped_reason: str | None = None

    @property
    def fully_matched(self) -> bool:
        return self.unmatched_stake <= 0.0 and self.matched_stake > 0.0


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

    def __init__(self, max_price_deviation_pct: float = 0.5) -> None:
        if max_price_deviation_pct <= 0.0:
            raise ValueError(
                f"max_price_deviation_pct must be > 0 "
                f"(got {max_price_deviation_pct})"
            )
        self.max_price_deviation_pct = float(max_price_deviation_pct)

    # ── Public API ──────────────────────────────────────────────────

    def match_back(
        self,
        available_to_lay: Iterable[PriceLevel],
        stake: float,
        reference_price: float,
        max_price: float | None = None,
    ) -> MatchResult:
        """Attempt to fill a back bet against the lay side of the book.

        For a back bet the *best* price is the **lowest** lay offer,
        because a lower lay price means the backer pays less for the
        same potential payout.
        """
        return self._match(
            list(available_to_lay),
            stake=stake,
            reference_price=reference_price,
            max_price=max_price,
            lower_is_better=True,
        )

    def match_lay(
        self,
        available_to_back: Iterable[PriceLevel],
        stake: float,
        reference_price: float,
        max_price: float | None = None,
    ) -> MatchResult:
        """Attempt to fill a lay bet against the back side of the book.

        For a lay bet the *best* price is the **highest** back offer,
        because a higher back price means the layer gets more stake
        in return for the same liability.
        """
        return self._match(
            list(available_to_back),
            stake=stake,
            reference_price=reference_price,
            max_price=max_price,
            lower_is_better=False,
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
    ) -> MatchResult:
        if stake <= 0.0:
            return MatchResult(0.0, 0.0, 0.0, "non-positive stake")
        if reference_price is None or reference_price <= 0.0:
            return MatchResult(0.0, stake, 0.0, "no LTP for runner")
        if not levels:
            return MatchResult(0.0, stake, 0.0, "empty ladder")

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
            )

        # Single-price fill: take up to the available size, leave the
        # remainder unmatched. No walking to worse levels.
        matched = min(stake, top.size)
        unmatched = stake - matched
        return MatchResult(
            matched_stake=matched,
            unmatched_stake=unmatched,
            average_price=top.price,
            skipped_reason=None,
        )


# ── Module-level default instance ────────────────────────────────────
#
# Most callers want the default 50 % junk-filter tolerance and don't
# need to vary it per-bet, so they can just import this singleton
# instead of instantiating their own.

DEFAULT_MATCHER = ExchangeMatcher(max_price_deviation_pct=0.5)
