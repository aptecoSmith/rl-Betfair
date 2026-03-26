"""
env/order_book.py — Realistic bet matching against historical order books.

Back bets consume ``AvailableToLay`` volume (the lay side is the counterparty).
Lay bets consume ``AvailableToBack`` volume.

Matching walks the price ladder level by level (index 0 = best price).
Any unmatched remainder after all levels are exhausted is cancelled — no
resting orders in v1.

Usage::

    from env.order_book import match_back, match_lay, MatchResult

    result = match_back(runner_snap.available_to_lay, stake=10.0)
    result = match_lay(runner_snap.available_to_back, stake=10.0)
"""

from __future__ import annotations

from dataclasses import dataclass

from data.episode_builder import PriceSize


@dataclass(frozen=True, slots=True)
class Fill:
    """A single fill at one price level."""

    price: float
    size: float  # Amount matched at this price


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Result of attempting to match a bet against the order book.

    Attributes:
        fills: Individual fills at each price level consumed.
        matched_stake: Total stake that was successfully matched.
        unmatched_stake: Remainder that could not be matched (cancelled).
        average_price: Weighted-average price across all fills, or 0.0 if
            nothing matched.
    """

    fills: tuple[Fill, ...]
    matched_stake: float
    unmatched_stake: float
    average_price: float


def match_back(
    available_to_lay: list[PriceSize],
    stake: float,
) -> MatchResult:
    """Match a back bet against the available-to-lay ladder.

    A back bet is filled by counterparties willing to lay.  The best (lowest)
    lay price is the price the backer pays.  We walk the ladder from index 0
    (best) upward, consuming volume at each level.

    Args:
        available_to_lay: Order book lay side, index 0 = best (lowest) price.
        stake: Desired stake in £.

    Returns:
        A :class:`MatchResult` describing what was filled.
    """
    return _match(available_to_lay, stake)


def match_lay(
    available_to_back: list[PriceSize],
    stake: float,
) -> MatchResult:
    """Match a lay bet against the available-to-back ladder.

    A lay bet is filled by counterparties willing to back.  The best (highest)
    back price is the price the layer offers.  We walk the ladder from index 0
    (best) upward, consuming volume at each level.

    Args:
        available_to_back: Order book back side, index 0 = best (highest) price.
        stake: Desired stake in £.

    Returns:
        A :class:`MatchResult` describing what was filled.
    """
    return _match(available_to_back, stake)


def _match(levels: list[PriceSize], stake: float) -> MatchResult:
    """Walk price levels and fill as much of *stake* as possible."""
    if stake <= 0.0 or not levels:
        return MatchResult(
            fills=(),
            matched_stake=0.0,
            unmatched_stake=max(stake, 0.0),
            average_price=0.0,
        )

    fills: list[Fill] = []
    remaining = stake

    for level in levels:
        if remaining <= 0.0:
            break
        if level.size <= 0.0 or level.price <= 0.0:
            continue

        consumed = min(remaining, level.size)
        fills.append(Fill(price=level.price, size=consumed))
        remaining -= consumed

    matched = stake - max(remaining, 0.0)
    avg_price = (
        sum(f.price * f.size for f in fills) / matched if matched > 0.0 else 0.0
    )

    return MatchResult(
        fills=tuple(fills),
        matched_stake=matched,
        unmatched_stake=max(remaining, 0.0),
        average_price=avg_price,
    )
