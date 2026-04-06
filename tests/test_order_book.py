"""Backwards-compatibility shim tests for env/order_book.py.

The original walking-ladder ``match_back`` / ``match_lay`` functions
have been removed — see :mod:`tests.test_exchange_matcher` for the
full suite of tests covering the replacement.

This file only verifies that the remaining public re-export still
resolves, so any straggler code importing ``MatchResult`` from the
legacy location keeps working.
"""

from __future__ import annotations


def test_match_result_reexport():
    """``MatchResult`` should still be importable from ``env.order_book``."""
    from env.order_book import MatchResult as LegacyMatchResult
    from env.exchange_matcher import MatchResult as NewMatchResult
    assert LegacyMatchResult is NewMatchResult


def test_walking_ladder_helpers_removed():
    """The old ``match_back`` / ``match_lay`` / ``Fill`` exports must be gone.

    Keeping them around (even as shims) risks re-introducing the
    phantom-profit bug — we want an ``ImportError`` at call sites so
    any stragglers are caught loudly.
    """
    import env.order_book as ob
    for removed in ("match_back", "match_lay", "Fill", "_match"):
        assert not hasattr(ob, removed), (
            f"env.order_book.{removed} should be removed but is still present"
        )
