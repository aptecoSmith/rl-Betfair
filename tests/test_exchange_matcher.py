"""Tests for env/exchange_matcher.py.

The matcher replaces the old walking-ladder ``order_book.match_*``
functions with a realistic single-price match plus a junk-level filter,
and is the fix for the phantom-profit bug where historical Betfair
order books contained stale parked orders at £1000 that the walking
matcher would gleefully consume.
"""

from __future__ import annotations

import pytest

from data.episode_builder import PriceSize
from env.exchange_matcher import (
    DEFAULT_MATCHER,
    ExchangeMatcher,
    MatchResult,
    PriceLevel,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _levels(*pairs: tuple[float, float]) -> list[PriceSize]:
    """Build a list of PriceSize from (price, size) tuples."""
    return [PriceSize(price=p, size=s) for p, s in pairs]


@pytest.fixture
def matcher() -> ExchangeMatcher:
    """Default matcher — 50 % deviation tolerance."""
    return ExchangeMatcher(max_price_deviation_pct=0.5)


# ── Constructor validation ───────────────────────────────────────────────────


class TestConstruction:
    def test_default_deviation(self):
        assert ExchangeMatcher().max_price_deviation_pct == 0.5

    def test_custom_deviation(self):
        assert ExchangeMatcher(0.1).max_price_deviation_pct == pytest.approx(0.1)

    def test_zero_deviation_rejected(self):
        with pytest.raises(ValueError):
            ExchangeMatcher(0.0)

    def test_negative_deviation_rejected(self):
        with pytest.raises(ValueError):
            ExchangeMatcher(-0.2)

    def test_default_instance_exists(self):
        assert isinstance(DEFAULT_MATCHER, ExchangeMatcher)
        assert DEFAULT_MATCHER.max_price_deviation_pct == 0.5


# ── Price-level protocol ─────────────────────────────────────────────────────


class TestPriceLevelProtocol:
    def test_pricesize_satisfies_protocol(self):
        ps = PriceSize(price=3.0, size=5.0)
        assert isinstance(ps, PriceLevel)

    def test_arbitrary_dataclass_satisfies_protocol(self):
        """Any object with .price / .size attributes should work — this is
        the whole point of the Protocol, and is what makes the matcher
        portable to the ai-betfair live inference project without an
        adapter layer."""
        class Level:
            def __init__(self, price: float, size: float):
                self.price = price
                self.size = size
        lv = Level(4.0, 10.0)
        assert isinstance(lv, PriceLevel)


# ── Single-price match semantics (back) ──────────────────────────────────────


class TestMatchBackSinglePrice:
    def test_full_fill_at_best_price(self, matcher):
        """Back bet matches at the single best lay price — no walking."""
        levels = _levels((3.0, 50.0))
        r = matcher.match_back(levels, stake=10.0, reference_price=3.0)
        assert r.matched_stake == 10.0
        assert r.unmatched_stake == 0.0
        assert r.average_price == 3.0

    def test_partial_fill_when_stake_exceeds_top_size(self, matcher):
        """If requested stake > top-of-book size, the remainder is UNMATCHED
        — we do NOT walk to the next level. This is the core behavioural
        difference from the old walking implementation."""
        levels = _levels((3.0, 100.0), (3.5, 100.0), (4.0, 5.0))
        r = matcher.match_back(levels, stake=20.0, reference_price=3.5)
        assert r.matched_stake == 5.0          # only top level's size
        assert r.unmatched_stake == 15.0
        assert r.average_price == 4.0          # top-of-book only, not a weighted avg

    def test_highest_back_price_is_best(self, matcher):
        """For a back bet, the *highest* back price wins — a backer wants
        the best odds. The matcher must pick it even if the input
        list isn't sorted that way (defensive sort after filtering)."""
        levels = _levels((4.0, 10.0), (3.5, 8.0), (4.5, 12.0))  # deliberately unsorted
        r = matcher.match_back(levels, stake=5.0, reference_price=4.0)
        assert r.matched_stake == 5.0
        assert r.average_price == 4.5  # the highest price in the list

    def test_does_not_walk_into_junk_even_if_top_exhausted(self, matcher):
        """Regression test for the phantom-profit bug.
        Top-of-book has tiny size at a realistic price, the next level is
        the Betfair £1000 parked junk. Even with stake >> top size, the
        matcher must NOT walk down to the junk level — it must leave the
        remainder unmatched and keep average_price anchored at the real
        market price.
        """
        levels = _levels((4.3, 14.45), (14.5, 11.53), (980.0, 1.73))
        r = matcher.match_back(levels, stake=100.0, reference_price=4.3)
        assert r.matched_stake == pytest.approx(14.45)
        assert r.unmatched_stake == pytest.approx(100.0 - 14.45)
        assert r.average_price == 4.3


# ── Single-price match semantics (lay) ───────────────────────────────────────


class TestMatchLaySinglePrice:
    def test_full_fill_at_best_price(self, matcher):
        levels = _levels((5.0, 30.0))
        r = matcher.match_lay(levels, stake=10.0, reference_price=5.0)
        assert r.matched_stake == 10.0
        assert r.average_price == 5.0

    def test_partial_fill_when_stake_exceeds_top_size(self, matcher):
        """Same single-price rule on the lay side."""
        levels = _levels((4.5, 6.0), (5.0, 50.0))
        r = matcher.match_lay(levels, stake=20.0, reference_price=5.0)
        assert r.matched_stake == 6.0
        assert r.unmatched_stake == 14.0
        assert r.average_price == 4.5  # best lay = lowest; no walk to 5.0

    def test_lowest_lay_price_is_best(self, matcher):
        """For a lay bet the *lowest* lay price wins — layer gets least
        liability. Matcher must pick it regardless of input ordering."""
        levels = _levels((4.0, 10.0), (4.5, 8.0), (3.5, 12.0))
        r = matcher.match_lay(levels, stake=5.0, reference_price=4.0)
        assert r.matched_stake == 5.0
        assert r.average_price == 3.5


# ── Junk-level filter ────────────────────────────────────────────────────────


class TestJunkFilter:
    def test_level_outside_deviation_dropped(self, matcher):
        """Any level more than ±50 % from LTP is treated as junk."""
        levels = _levels((4.3, 10.0), (1000.0, 500.0))
        r = matcher.match_back(levels, stake=5.0, reference_price=4.3)
        assert r.matched_stake == 5.0
        assert r.average_price == 4.3

    def test_only_junk_returns_no_match(self, matcher):
        """If EVERY level is junk, the bet is refused — not filled at
        the junk price. This is the catastrophic case: a runner whose
        only lay offer is £1000 must not produce a £94 905 phantom
        profit on a £100 stake."""
        levels = _levels((1000.0, 500.0))
        r = matcher.match_back(levels, stake=100.0, reference_price=4.3)
        assert r.matched_stake == 0.0
        assert r.unmatched_stake == 100.0
        assert r.skipped_reason is not None
        assert "junk" in r.skipped_reason.lower() or "outside" in r.skipped_reason.lower()

    def test_tighter_deviation_drops_more_levels(self):
        """A stricter matcher should reject levels a generous one accepts."""
        tight = ExchangeMatcher(max_price_deviation_pct=0.05)  # ±5 %
        levels = _levels((4.3, 10.0), (5.0, 20.0))
        # 5.0 is ~16 % above 4.3 LTP → outside ±5 % → dropped.
        r = tight.match_back(levels, stake=5.0, reference_price=4.3)
        assert r.matched_stake == 5.0
        assert r.average_price == 4.3  # 5.0 was filtered, only 4.3 remains

    def test_deviation_symmetric_around_ltp(self, matcher):
        """Levels 50 % below LTP are also junk."""
        levels = _levels((1.0, 10.0))  # 50 % below LTP 2.0 — right on the edge? Use 0.99 to be below.
        r = matcher.match_lay(
            _levels((0.99, 10.0)),
            stake=5.0,
            reference_price=2.0,
        )
        assert r.matched_stake == 0.0
        assert r.skipped_reason is not None

    def test_level_exactly_at_deviation_boundary_kept(self, matcher):
        """Price == LTP × (1 + deviation) should be INSIDE the window.
        Default deviation is 0.5 so LTP 4.0 × 1.5 = 6.0 is still valid.
        """
        levels = _levels((6.0, 10.0))
        r = matcher.match_back(levels, stake=5.0, reference_price=4.0)
        assert r.matched_stake == 5.0
        assert r.average_price == 6.0


# ── max_price cap ────────────────────────────────────────────────────────────


class TestMaxPriceCap:
    def test_best_price_within_cap_allowed(self, matcher):
        levels = _levels((4.3, 10.0))
        r = matcher.match_back(
            levels, stake=5.0, reference_price=4.3, max_price=100.0,
        )
        assert r.matched_stake == 5.0

    def test_best_price_over_cap_refused(self, matcher):
        """Best POST-FILTER price above the cap → whole bet refused."""
        levels = _levels((150.0, 10.0))
        r = matcher.match_back(
            levels, stake=5.0, reference_price=150.0, max_price=100.0,
        )
        assert r.matched_stake == 0.0
        assert r.skipped_reason is not None
        assert "exceeds" in r.skipped_reason.lower() or "cap" in r.skipped_reason.lower()

    def test_cap_applied_after_junk_filter(self, matcher):
        """The cap must be evaluated against the POST-FILTER top of book.
        If junk-filtering removes the £1000 level and leaves a £4.3 level,
        a max_price of 100 must NOT refuse the bet.
        """
        levels = _levels((4.3, 10.0), (1000.0, 500.0))
        r = matcher.match_back(
            levels, stake=5.0, reference_price=4.3, max_price=100.0,
        )
        assert r.matched_stake == 5.0
        assert r.average_price == 4.3

    def test_no_cap_means_no_limit(self, matcher):
        levels = _levels((90.0, 10.0))
        r = matcher.match_back(
            levels, stake=5.0, reference_price=90.0, max_price=None,
        )
        assert r.matched_stake == 5.0


# ── Input validation & edge cases ────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_ladder(self, matcher):
        r = matcher.match_back([], stake=10.0, reference_price=3.0)
        assert r.matched_stake == 0.0
        assert r.unmatched_stake == 10.0
        assert r.skipped_reason is not None

    def test_zero_stake(self, matcher):
        levels = _levels((3.0, 50.0))
        r = matcher.match_back(levels, stake=0.0, reference_price=3.0)
        assert r.matched_stake == 0.0
        assert r.skipped_reason is not None

    def test_negative_stake(self, matcher):
        levels = _levels((3.0, 50.0))
        r = matcher.match_back(levels, stake=-5.0, reference_price=3.0)
        assert r.matched_stake == 0.0
        assert r.skipped_reason is not None

    def test_no_ltp_refuses_bet(self, matcher):
        """A runner with LTP = 0 (never traded) must not be bet on —
        we can't price-reference the ladder without an anchor."""
        levels = _levels((3.0, 50.0))
        r = matcher.match_back(levels, stake=10.0, reference_price=0.0)
        assert r.matched_stake == 0.0
        assert r.skipped_reason is not None
        assert "ltp" in r.skipped_reason.lower()

    def test_negative_ltp_refused(self, matcher):
        levels = _levels((3.0, 50.0))
        r = matcher.match_back(levels, stake=10.0, reference_price=-1.0)
        assert r.matched_stake == 0.0

    def test_zero_size_level_skipped(self, matcher):
        """Zero-size levels are dropped even if their price is sensible."""
        levels = _levels((3.0, 0.0), (3.1, 10.0))
        r = matcher.match_back(levels, stake=5.0, reference_price=3.0)
        assert r.matched_stake == 5.0
        assert r.average_price == 3.1

    def test_zero_price_level_skipped(self, matcher):
        levels = _levels((0.0, 10.0), (3.0, 10.0))
        r = matcher.match_back(levels, stake=5.0, reference_price=3.0)
        assert r.matched_stake == 5.0
        assert r.average_price == 3.0


# ── MatchResult helpers ──────────────────────────────────────────────────────


class TestMatchResult:
    def test_fully_matched_true_on_full_fill(self):
        r = MatchResult(
            matched_stake=10.0, unmatched_stake=0.0,
            average_price=3.0, skipped_reason=None,
        )
        assert r.fully_matched is True

    def test_fully_matched_false_on_partial(self):
        r = MatchResult(
            matched_stake=5.0, unmatched_stake=5.0,
            average_price=3.0, skipped_reason=None,
        )
        assert r.fully_matched is False

    def test_fully_matched_false_on_zero_match(self):
        r = MatchResult(
            matched_stake=0.0, unmatched_stake=10.0,
            average_price=0.0, skipped_reason="empty",
        )
        assert r.fully_matched is False
