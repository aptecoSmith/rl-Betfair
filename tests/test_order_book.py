"""Tests for env/order_book.py — realistic bet matching simulation."""

from __future__ import annotations

import pytest

from data.episode_builder import PriceSize
from env.order_book import Fill, MatchResult, match_back, match_lay


# ── Helpers ──────────────────────────────────────────────────────────────────


def _levels(*pairs: tuple[float, float]) -> list[PriceSize]:
    """Create a list of PriceSize from (price, size) tuples."""
    return [PriceSize(price=p, size=s) for p, s in pairs]


# ── match_back tests ─────────────────────────────────────────────────────────


class TestMatchBack:
    """Back bets consume AvailableToLay volume."""

    def test_full_fill_single_level(self):
        levels = _levels((3.0, 50.0))
        result = match_back(levels, stake=10.0)

        assert result.matched_stake == 10.0
        assert result.unmatched_stake == 0.0
        assert result.average_price == 3.0
        assert len(result.fills) == 1
        assert result.fills[0] == Fill(price=3.0, size=10.0)

    def test_full_fill_exact_size(self):
        levels = _levels((4.0, 25.0))
        result = match_back(levels, stake=25.0)

        assert result.matched_stake == 25.0
        assert result.unmatched_stake == 0.0
        assert result.average_price == 4.0

    def test_partial_fill_single_level(self):
        levels = _levels((3.5, 8.0))
        result = match_back(levels, stake=20.0)

        assert result.matched_stake == 8.0
        assert result.unmatched_stake == pytest.approx(12.0)
        assert result.average_price == 3.5

    def test_multi_level_fill(self):
        levels = _levels((3.0, 5.0), (3.5, 10.0), (4.0, 20.0))
        result = match_back(levels, stake=12.0)

        assert result.matched_stake == 12.0
        assert result.unmatched_stake == 0.0
        assert len(result.fills) == 2
        assert result.fills[0] == Fill(price=3.0, size=5.0)
        assert result.fills[1] == Fill(price=3.5, size=7.0)
        # Weighted average: (3.0*5 + 3.5*7) / 12 = 39.5/12 ≈ 3.2917
        assert result.average_price == pytest.approx(39.5 / 12.0)

    def test_multi_level_partial_fill(self):
        """Exhaust all three levels, still have unmatched remainder."""
        levels = _levels((2.5, 3.0), (3.0, 4.0), (3.5, 3.0))
        result = match_back(levels, stake=20.0)

        assert result.matched_stake == 10.0
        assert result.unmatched_stake == 10.0
        assert len(result.fills) == 3

    def test_all_three_levels_consumed(self):
        levels = _levels((2.0, 10.0), (2.5, 10.0), (3.0, 10.0))
        result = match_back(levels, stake=30.0)

        assert result.matched_stake == 30.0
        assert result.unmatched_stake == 0.0
        assert len(result.fills) == 3

    def test_empty_order_book(self):
        result = match_back([], stake=10.0)

        assert result.matched_stake == 0.0
        assert result.unmatched_stake == 10.0
        assert result.average_price == 0.0
        assert result.fills == ()

    def test_zero_stake(self):
        levels = _levels((3.0, 50.0))
        result = match_back(levels, stake=0.0)

        assert result.matched_stake == 0.0
        assert result.unmatched_stake == 0.0
        assert result.fills == ()

    def test_negative_stake(self):
        levels = _levels((3.0, 50.0))
        result = match_back(levels, stake=-5.0)

        assert result.matched_stake == 0.0
        assert result.fills == ()

    def test_zero_size_level_skipped(self):
        levels = _levels((2.0, 0.0), (3.0, 10.0))
        result = match_back(levels, stake=5.0)

        assert result.matched_stake == 5.0
        assert result.average_price == 3.0
        assert len(result.fills) == 1

    def test_zero_price_level_skipped(self):
        levels = _levels((0.0, 10.0), (3.0, 10.0))
        result = match_back(levels, stake=5.0)

        assert result.matched_stake == 5.0
        assert result.average_price == 3.0

    def test_fills_are_immutable(self):
        levels = _levels((3.0, 50.0))
        result = match_back(levels, stake=10.0)
        assert isinstance(result.fills, tuple)

    def test_very_small_stake(self):
        levels = _levels((3.0, 50.0))
        result = match_back(levels, stake=0.01)

        assert result.matched_stake == pytest.approx(0.01)
        assert result.unmatched_stake == pytest.approx(0.0)


# ── match_lay tests ──────────────────────────────────────────────────────────


class TestMatchLay:
    """Lay bets consume AvailableToBack volume."""

    def test_full_fill_single_level(self):
        levels = _levels((5.0, 30.0))
        result = match_lay(levels, stake=10.0)

        assert result.matched_stake == 10.0
        assert result.unmatched_stake == 0.0
        assert result.average_price == 5.0

    def test_partial_fill(self):
        levels = _levels((4.0, 6.0))
        result = match_lay(levels, stake=20.0)

        assert result.matched_stake == 6.0
        assert result.unmatched_stake == 14.0

    def test_multi_level_fill(self):
        levels = _levels((5.0, 10.0), (4.5, 10.0), (4.0, 10.0))
        result = match_lay(levels, stake=15.0)

        assert result.matched_stake == 15.0
        assert len(result.fills) == 2
        assert result.fills[0] == Fill(price=5.0, size=10.0)
        assert result.fills[1] == Fill(price=4.5, size=5.0)

    def test_empty_order_book(self):
        result = match_lay([], stake=10.0)

        assert result.matched_stake == 0.0
        assert result.unmatched_stake == 10.0

    def test_zero_stake(self):
        levels = _levels((5.0, 30.0))
        result = match_lay(levels, stake=0.0)

        assert result.matched_stake == 0.0


# ── MatchResult properties ───────────────────────────────────────────────────


class TestMatchResult:
    def test_weighted_average_price_two_levels(self):
        levels = _levels((2.0, 4.0), (3.0, 6.0))
        result = match_back(levels, stake=10.0)

        # (2.0*4 + 3.0*6) / 10 = 26/10 = 2.6
        assert result.average_price == pytest.approx(2.6)

    def test_no_match_gives_zero_average_price(self):
        result = match_back([], stake=10.0)
        assert result.average_price == 0.0

    def test_single_fill_average_equals_price(self):
        levels = _levels((7.5, 100.0))
        result = match_back(levels, stake=20.0)
        assert result.average_price == pytest.approx(7.5)
