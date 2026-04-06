"""
env/order_book.py — **DEPRECATED**.

The original ``match_back`` / ``match_lay`` helpers walked the price
ladder level-by-level consuming every fillable size, which is NOT how
the Betfair Exchange works: a real bet targets a single price and only
matches against counter-side liquidity at that price (or better). The
walking behaviour also admitted stale parked orders at extreme prices
into the average fill, producing hundreds-of-times-market-value fills
and phantom P&L of tens of thousands of pounds per bet.

All call sites have been migrated to :class:`env.exchange_matcher.ExchangeMatcher`,
which implements realistic single-price matching with a junk-level
filter.  This module is kept only so that ``MatchResult`` can still be
imported from its original location by any straggler code.  The old
``match_back`` / ``match_lay`` / ``Fill`` / ``_match`` symbols have
been removed — import from :mod:`env.exchange_matcher` instead.
"""

from __future__ import annotations

from env.exchange_matcher import MatchResult  # re-export for back-compat

__all__ = ["MatchResult"]
