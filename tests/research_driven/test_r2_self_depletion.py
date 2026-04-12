"""Tests for R-2 fix: matcher self-depletion of previously-matched volume.

When the agent places multiple bets at the same price on the same selection
within one race, the second bet must see a reduced available size equal to
the original size minus what the agent already matched.

All tests are CPU-only and run in milliseconds.
"""

from __future__ import annotations

import pytest

from data.episode_builder import PriceSize, RunnerSnap
from env.bet_manager import BetManager, BetSide
from env.exchange_matcher import ExchangeMatcher


# ── Helpers ──────────────────────────────────────────────────────────────────


def _runner(
    selection_id: int = 1001,
    back_levels: list[tuple[float, float]] | None = None,
    lay_levels: list[tuple[float, float]] | None = None,
    ltp: float | None = None,
) -> RunnerSnap:
    """Build a RunnerSnap from (price, size) tuples.

    ``ltp`` defaults to the midpoint of provided prices so all levels
    fall inside the matcher's ±50 % junk filter.
    """
    atb = [PriceSize(p, s) for p, s in (back_levels or [])]
    atl = [PriceSize(p, s) for p, s in (lay_levels or [])]
    if ltp is None:
        prices = [ps.price for ps in atb + atl]
        ltp = sum(prices) / len(prices) if prices else 3.0
    return RunnerSnap(
        selection_id=selection_id,
        status="ACTIVE",
        last_traded_price=ltp,
        total_matched=1000.0,
        starting_price_near=0.0,
        starting_price_far=0.0,
        adjustment_factor=None,
        bsp=None,
        sort_priority=1,
        removal_date=None,
        available_to_back=atb,
        available_to_lay=atl,
    )


# ── Test 1: two back bets at the same price, same runner ────────────────────


class TestTwoBackBetsSamePriceSameRunner:
    """Top-of-book £21 at price P.  First bet £12.10, second bet £17."""

    PRICE = 4.0
    TOB_SIZE = 21.0

    def _mgr(self) -> BetManager:
        return BetManager(starting_budget=500.0)

    def _snap(self) -> RunnerSnap:
        # Both bets see the same historical ladder snapshot.
        return _runner(back_levels=[(self.PRICE, self.TOB_SIZE)], ltp=self.PRICE)

    def test_first_bet_fills_fully(self):
        mgr = self._mgr()
        snap = self._snap()
        bet1 = mgr.place_back(snap, stake=12.10)
        assert bet1 is not None
        assert bet1.matched_stake == pytest.approx(12.10)
        assert bet1.average_price == pytest.approx(self.PRICE)

    def test_second_bet_capped_at_remainder(self):
        mgr = self._mgr()
        snap = self._snap()
        mgr.place_back(snap, stake=12.10)
        bet2 = mgr.place_back(snap, stake=17.0)
        assert bet2 is not None
        assert bet2.matched_stake == pytest.approx(21.0 - 12.10)

    def test_total_matched_equals_tob_size(self):
        mgr = self._mgr()
        snap = self._snap()
        bet1 = mgr.place_back(snap, stake=12.10)
        bet2 = mgr.place_back(snap, stake=17.0)
        assert bet1 is not None and bet2 is not None
        total = bet1.matched_stake + bet2.matched_stake
        assert total == pytest.approx(self.TOB_SIZE)


# ── Test 2: two back bets at different prices, same runner ──────────────────


class TestTwoBackBetsDifferentPricesSameRunner:
    """Accumulator is keyed per-price; different prices don't interfere."""

    P1 = 3.0
    P2 = 5.0

    def test_different_price_bets_both_fill_independently(self):
        mgr = BetManager(starting_budget=500.0)
        snap_p1 = _runner(back_levels=[(self.P1, 50.0)], ltp=self.P1)
        snap_p2 = _runner(back_levels=[(self.P2, 50.0)], ltp=self.P2)

        bet1 = mgr.place_back(snap_p1, stake=20.0)
        bet2 = mgr.place_back(snap_p2, stake=20.0)

        assert bet1 is not None
        assert bet1.matched_stake == pytest.approx(20.0)
        assert bet1.average_price == pytest.approx(self.P1)

        assert bet2 is not None
        assert bet2.matched_stake == pytest.approx(20.0)
        assert bet2.average_price == pytest.approx(self.P2)


# ── Test 3: two back bets at the same price, different runners ──────────────


class TestTwoBackBetsSamePriceDifferentRunners:
    """Accumulator is keyed per-selection; different runners don't interfere."""

    PRICE = 4.0

    def test_different_runners_both_fill_fully(self):
        mgr = BetManager(starting_budget=500.0)
        snap_a = _runner(selection_id=101, back_levels=[(self.PRICE, 21.0)], ltp=self.PRICE)
        snap_b = _runner(selection_id=202, back_levels=[(self.PRICE, 21.0)], ltp=self.PRICE)

        bet_a = mgr.place_back(snap_a, stake=15.0)
        bet_b = mgr.place_back(snap_b, stake=15.0)

        assert bet_a is not None and bet_a.matched_stake == pytest.approx(15.0)
        assert bet_b is not None and bet_b.matched_stake == pytest.approx(15.0)


# ── Test 4: back then lay at same price, same runner ────────────────────────


class TestBackThenLaySamePriceSameRunner:
    """Accumulator is keyed per-side; back fills don't deplete lay budget."""

    PRICE = 4.0

    def test_back_accumulator_does_not_affect_lay(self):
        mgr = BetManager(starting_budget=500.0)
        # Runner has both sides at the same price, each with £21.
        snap = _runner(
            selection_id=101,
            lay_levels=[(self.PRICE, 21.0)],
            back_levels=[(self.PRICE, 21.0)],
            ltp=self.PRICE,
        )

        back_bet = mgr.place_back(snap, stake=15.0)
        lay_bet = mgr.place_lay(snap, stake=15.0)

        assert back_bet is not None and back_bet.matched_stake == pytest.approx(15.0)
        assert lay_bet is not None and lay_bet.matched_stake == pytest.approx(15.0)


# ── Test 5: same-price back bets across two races ───────────────────────────


class TestSamePriceBackBetsAcrossRaces:
    """A fresh BetManager for race B ignores race A's accumulator state."""

    PRICE = 4.0
    TOB_SIZE = 21.0

    def test_second_race_fills_fully_again(self):
        snap = _runner(back_levels=[(self.PRICE, self.TOB_SIZE)], ltp=self.PRICE)

        # Race A: fill the whole level.
        mgr_a = BetManager(starting_budget=500.0)
        bet_a1 = mgr_a.place_back(snap, stake=12.10)
        bet_a2 = mgr_a.place_back(snap, stake=17.0)
        assert bet_a1 is not None and bet_a2 is not None
        assert bet_a1.matched_stake + bet_a2.matched_stake == pytest.approx(self.TOB_SIZE)

        # Race B: fresh BetManager — should fill £21 from scratch.
        mgr_b = BetManager(starting_budget=500.0)
        bet_b1 = mgr_b.place_back(snap, stake=21.0)
        assert bet_b1 is not None
        assert bet_b1.matched_stake == pytest.approx(self.TOB_SIZE)


# ── Test 6: skipped reason on full self-exhaustion ──────────────────────────


class TestSkippedReasonOnSelfExhaustion:
    """After exhausting a level the next bet at that price returns matched=0
    with a truthy skipped_reason that mentions self-depletion."""

    PRICE = 4.0
    TOB_SIZE = 10.0

    def test_exhausted_level_returns_none_from_place_back(self):
        """BetManager returns None when the level is self-exhausted."""
        mgr = BetManager(starting_budget=500.0)
        snap = _runner(back_levels=[(self.PRICE, self.TOB_SIZE)], ltp=self.PRICE)

        # Drain the level completely.
        bet1 = mgr.place_back(snap, stake=10.0)
        assert bet1 is not None and bet1.matched_stake == pytest.approx(10.0)

        # Second bet on the same exhausted level.
        bet2 = mgr.place_back(snap, stake=5.0)
        assert bet2 is None

    def test_matcher_skipped_reason_mentions_self_depletion(self):
        """The raw matcher returns a truthy skipped_reason string."""
        matcher = ExchangeMatcher()
        from data.episode_builder import PriceSize

        levels = [PriceSize(price=self.PRICE, size=self.TOB_SIZE)]
        # First match drains the level.
        r1 = matcher.match_back(levels, stake=10.0, reference_price=self.PRICE)
        assert r1.matched_stake == pytest.approx(10.0)

        # Second match passes the already-matched amount.
        r2 = matcher.match_back(
            levels,
            stake=5.0,
            reference_price=self.PRICE,
            already_matched_at_top=r1.matched_stake,
        )
        assert r2.matched_stake == pytest.approx(0.0)
        assert r2.skipped_reason
        assert "self-depletion" in r2.skipped_reason
