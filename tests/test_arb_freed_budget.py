"""Tests for the 2026-04-15 forced-arb changes:

- Relaxed LTP-relative junk filter on the explicit-price (paired-arb)
  placement path (kept on the default-price path).
- Paired-arb diagnostic counters (paired_place_rejects per reason +
  paired_fill_skips for on-tick LTP-filter skips).
- Joint-affordability pre-flight: aggressive stake gets sized down so
  the paired counter-leg fits.
- Freed-budget reservation: paired LAY reserves only
  ``max(0, liability - back_stake)``; symmetric path for paired BACK
  retroactively reduces the partner LAY's reserved_liability. Settle /
  cancel / void release the actual reserved amount, never the textbook
  full liability.
- Asymmetric naked-loss term in raw reward (loss counted at 0.5x,
  windfalls still excluded).

Each test is intentionally self-contained — the synthetic helpers in
``tests.test_betfair_env`` give us full control over price, LTP and
ladder size so the assertions are about behaviour, not about bouncing
through PPO sampling.
"""

from __future__ import annotations

import numpy as np
import pytest

from env.bet_manager import BetManager, BetSide, PassiveOrder
from env.betfair_env import (
    MAX_ARB_TICKS,
    SCALPING_ACTIONS_PER_RUNNER,
    BetfairEnv,
)

from tests.test_betfair_env import _make_day, _make_runner_snap
from tests.test_forced_arbitrage import scalping_config  # noqa: F401  (fixture)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _runner(sid: int = 101, ltp: float = 4.0, back: float = 4.0,
            lay: float = 4.2, size: float = 100.0):
    return _make_runner_snap(sid, ltp=ltp, back_price=back, lay_price=lay,
                             size=size)


# ── Freed-budget reservation ─────────────────────────────────────────────────


class TestFreedBudgetReservation:
    """The Betfair "freed budget" rule: a paired back+lay can never both
    lose, so worst-case reservation = max(back_stake, lay_liability),
    not the additive sum. For typical scalping prices (lay <= 2.0) the
    back stake fully covers the lay liability — zero new reservation.
    """

    def test_paired_lay_reserves_zero_when_back_covers_liability(self):
        """back_stake = 10, lay at 1.50 → liability = 10 * 0.5 = 5.0 ≤
        back_stake, so the paired lay reserves 0 new liability."""
        bm = BetManager(starting_budget=100.0)
        bm.place_back(_runner(101, ltp=2.0, back=2.0, lay=2.1, size=200.0),
                      stake=10.0, market_id="m1", pair_id="pp")
        # Sanity: back stake debited, no liability yet.
        assert bm.budget == pytest.approx(90.0)
        assert bm._open_liability == pytest.approx(0.0)

        # Place the paired lay at 1.50 (well below back price 2.0).
        runner_lay = _runner(101, ltp=2.0, back=1.50, lay=1.51, size=200.0)
        order = bm.passive_book.place(
            runner_lay, stake=10.0, side=BetSide.LAY,
            market_id="m1", tick_index=0,
            price=1.50, pair_id="pp",
        )
        assert order is not None
        assert order.reserved_liability == pytest.approx(0.0), \
            "lay liability (5.0) ≤ back stake (10.0); no new reservation needed"
        # Open liability should NOT have moved.
        assert bm._open_liability == pytest.approx(0.0)

    def test_paired_lay_tops_up_when_liability_exceeds_back(self):
        """back_stake = 10, lay at 3.0 → liability = 10 * 2.0 = 20 >
        back_stake — reserve only the top-up of 10."""
        bm = BetManager(starting_budget=100.0)
        bm.place_back(_runner(101, ltp=4.0, back=4.0, lay=4.1, size=200.0),
                      stake=10.0, market_id="m1", pair_id="pp")
        # Pair the back at 4.0 with a lay at 3.0 → liability 20, top-up 10.
        runner_lay = _runner(101, ltp=4.0, back=3.0, lay=3.05, size=200.0)
        order = bm.passive_book.place(
            runner_lay, stake=10.0, side=BetSide.LAY,
            market_id="m1", tick_index=0,
            price=3.0, pair_id="pp",
        )
        assert order is not None
        assert order.reserved_liability == pytest.approx(10.0)
        assert bm._open_liability == pytest.approx(10.0)

    def test_unpaired_lay_reserves_full_liability(self):
        """Without a partner back, the paired-arb offset doesn't apply —
        full liability reservation as before. The
        ``reserved_liability`` field is still populated (= full
        liability) so the release math stays consistent across paired
        and unpaired paths."""
        bm = BetManager(starting_budget=100.0)
        runner = _runner(101, ltp=4.0, back=4.0, lay=4.1, size=200.0)
        order = bm.passive_book.place(
            runner, stake=10.0, side=BetSide.LAY,
            market_id="m1", tick_index=0,
            price=3.0,
            # NB: no pair_id → no offset, reserves the full liability.
        )
        assert order is not None
        # Liability = 10 * (3 - 1) = 20.
        assert order.reserved_liability == pytest.approx(20.0)
        assert bm._open_liability == pytest.approx(20.0)

    def test_paired_back_after_lay_releases_partner_offset(self):
        """Symmetric path: agg LAY first, then a paired passive BACK
        landing — the partner LAY's reserved_liability shrinks by the
        offset and _open_liability is released by the same amount."""
        bm = BetManager(starting_budget=100.0)
        # Aggressive lay at 4.0, stake 10 → liability 30, reserved fully.
        agg_lay = bm.place_lay(
            _runner(101, ltp=4.0, back=3.95, lay=4.0, size=200.0),
            stake=10.0, market_id="m1", pair_id="pp",
        )
        assert agg_lay is not None
        assert bm._open_liability == pytest.approx(30.0)
        assert agg_lay.reserved_liability is None  # not yet adjusted

        # Paired passive BACK at 4.5 (above lay price). Offset =
        # min(stake=10, lay_liability=30) = 10 → release 10 from
        # _open_liability and store the residual 20 on the LAY.
        runner_back = _runner(101, ltp=4.0, back=4.5, lay=4.55, size=200.0)
        order = bm.passive_book.place(
            runner_back, stake=10.0, side=BetSide.BACK,
            market_id="m1", tick_index=0,
            price=4.5, pair_id="pp",
        )
        assert order is not None
        # Back stake debited as usual.
        # Lay's reserved liability now reduced to 20.
        assert agg_lay.reserved_liability == pytest.approx(20.0)
        assert bm._open_liability == pytest.approx(20.0)


# ── Settle / cancel / void release the actual reserved amount ────────────────


class TestFreedBudgetReleaseInvariants:
    """_open_liability must not go negative through settlement,
    cancellation, or void when the paired-arb offset is in play.
    This is the invariant we'd most likely break."""

    def _make_paired_book_with_filled_lay(self):
        """Helper: agg back + filled passive lay, both with pair_id."""
        bm = BetManager(starting_budget=100.0)
        bm.place_back(_runner(101, ltp=2.0, back=2.0, lay=2.05, size=200.0),
                      stake=10.0, market_id="m1", pair_id="pp")
        # Place the paired lay (zero new reservation because lay ≤ back).
        order = bm.passive_book.place(
            _runner(101, ltp=2.0, back=1.50, lay=1.51, size=200.0),
            stake=10.0, side=BetSide.LAY,
            market_id="m1", tick_index=0,
            price=1.50, pair_id="pp",
        )
        assert order is not None
        # Force the order to "fill" by promoting it directly into bets.
        order.matched_stake = order.requested_stake
        from env.bet_manager import Bet
        bet = Bet(
            selection_id=order.selection_id, side=order.side,
            requested_stake=order.requested_stake,
            matched_stake=order.requested_stake,
            average_price=order.price, market_id=order.market_id,
            pair_id=order.pair_id, tick_index=0,
            reserved_liability=order.reserved_liability,
        )
        bm.bets.append(bet)
        bm.passive_book._orders.clear()
        bm.passive_book._orders_by_sid.clear()
        return bm, bet

    def test_settle_releases_only_reserved_not_full_liability(self):
        bm, lay_bet = self._make_paired_book_with_filled_lay()
        liability_pre = bm._open_liability
        assert liability_pre == pytest.approx(0.0)
        # Settle with selection 101 LOST → both legs settle as a pair.
        # Lay wins (pays stake, releases its reserved_liability=0).
        bm.settle_race({999}, market_id="m1")  # 101 not in winners
        assert bm._open_liability == pytest.approx(0.0), \
            "settlement must not over-release; lay was reserved=0"

    def test_cancel_releases_only_reserved(self):
        """Race-off cancellation of an unfilled paired lay must release
        order.reserved_liability, not the full textbook liability."""
        bm = BetManager(starting_budget=100.0)
        bm.place_back(_runner(101, ltp=2.0, back=2.0, lay=2.05, size=200.0),
                      stake=10.0, market_id="m1", pair_id="pp")
        # Paired lay at 1.50 — reserved 0 because covered by back.
        bm.passive_book.place(
            _runner(101, ltp=2.0, back=1.50, lay=1.51, size=200.0),
            stake=10.0, side=BetSide.LAY,
            market_id="m1", tick_index=0,
            price=1.50, pair_id="pp",
        )
        liability_pre = bm._open_liability
        bm.passive_book.cancel_all("race-off")
        # _open_liability should be unchanged (we released the same 0
        # we'd reserved).
        assert bm._open_liability == pytest.approx(liability_pre)

    def test_void_releases_only_reserved(self):
        bm, _ = self._make_paired_book_with_filled_lay()
        bm.void_race(market_id="m1")
        assert bm._open_liability == pytest.approx(0.0)


# ── Paired-arb diagnostic counters ───────────────────────────────────────────


class TestPairedArbDiagnostics:

    def test_budget_lay_rejection_increments_counter(self):
        """A paired lay that exceeds available_budget bumps the
        ``budget_lay`` counter, not silently dropped."""
        bm = BetManager(starting_budget=10.0)
        # Use up most of the budget on the aggressive back.
        bm.place_back(_runner(101, ltp=10.0, back=10.0, lay=10.5, size=200.0),
                      stake=9.0, market_id="m1", pair_id="pp")
        # Paired lay at 5.0 → liability 9 * 4 = 36; offset = min(9, 36) = 9;
        # reserved = 27 > available_budget (1) → refused.
        order = bm.passive_book.place(
            _runner(101, ltp=10.0, back=5.0, lay=5.1, size=200.0),
            stake=9.0, side=BetSide.LAY,
            market_id="m1", tick_index=0,
            price=5.0, pair_id="pp",
        )
        assert order is None
        assert bm.passive_book._paired_place_rejects["budget_lay"] == 1

    def test_no_ltp_rejection_increments_counter(self):
        """A paired placement on a runner with no LTP increments
        ``no_ltp``."""
        bm = BetManager(starting_budget=100.0)
        runner = _make_runner_snap(101, ltp=None, size=200.0)  # type: ignore[arg-type]
        order = bm.passive_book.place(
            runner, stake=5.0, side=BetSide.LAY,
            market_id="m1", tick_index=0,
            price=2.0, pair_id="pp",
        )
        assert order is None
        assert bm.passive_book._paired_place_rejects["no_ltp"] == 1

    def test_unpaired_rejection_does_not_increment_counters(self):
        """Same rejection without a pair_id MUST NOT increment the
        paired counters — they're forced-arb-specific."""
        bm = BetManager(starting_budget=100.0)
        runner = _make_runner_snap(101, ltp=None, size=200.0)  # type: ignore[arg-type]
        order = bm.passive_book.place(
            runner, stake=5.0, side=BetSide.LAY,
            market_id="m1", tick_index=0,
            price=2.0,  # no pair_id
        )
        assert order is None
        assert bm.passive_book._paired_place_rejects["no_ltp"] == 0


# ── Joint-affordability pre-flight ───────────────────────────────────────────


class TestJointAffordabilityPreflight:
    """The pre-flight in _process_action sizes the aggressive stake by
    ``available_budget / max(1, lay_price - 1)`` so the paired
    reservation always fits. Hard to test through PPO sampling, so we
    drive the env directly with a deterministic action."""

    def test_aggressive_back_stake_capped_by_paired_lay(self, scalping_config):
        """Request 100% of budget on the back leg → after pre-flight,
        actually-matched stake reflects the joint cap, not the request.
        """
        env = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), scalping_config)
        env.reset()
        a = np.zeros(14 * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32)
        a[0] = 1.0    # back signal
        a[14] = 1.0   # stake = 100% of budget (would normally over-commit)
        a[28] = 1.0   # aggressive
        a[56] = -1.0  # min ticks
        env.step(a)
        bm = env.bet_manager
        backs = [b for b in bm.bets if b.side is BetSide.BACK and b.pair_id]
        # The pre-flight should have shrunk the stake; placement must
        # still succeed (so a back bet exists) AND the budget should
        # not be fully consumed (joint reservation respected).
        assert backs, "aggressive back should still place after pre-flight cap"
        assert bm.budget > 0.0, \
            "budget must not be fully consumed; joint reservation needs headroom"


# ── Explicit-price junk filter relaxed (paired path only) ────────────────────


class TestExplicitPriceJunkFilterRelaxed:

    def test_paired_lay_accepted_far_below_ltp(self):
        """A paired lay at half of LTP used to be silently refused by
        the LTP-relative junk filter; now it's accepted because the
        price came from tick_offset, not the noisy ladder."""
        bm = BetManager(starting_budget=100.0)
        # Far-below-LTP price would be outside the ±50% junk window.
        runner = _runner(101, ltp=10.0, back=2.0, lay=2.1, size=200.0)
        order = bm.passive_book.place(
            runner, stake=5.0, side=BetSide.LAY,
            market_id="m1", tick_index=0,
            price=2.0,           # 80% below LTP=10.0
            pair_id="pp",        # paired path → filter relaxed
        )
        assert order is not None
        assert order.price == pytest.approx(2.0)

    def test_default_path_still_rejects_when_top_outside_filter(self):
        """Regression guard: the default-price path (no explicit price)
        must still respect the junk filter on the top of book."""
        bm = BetManager(starting_budget=100.0)
        # Build a runner whose only resting back levels are outside ±50%
        # of LTP — pick_top_price returns None → place returns None.
        from data.episode_builder import PriceSize, RunnerSnap
        runner = RunnerSnap(
            selection_id=101, status="ACTIVE", last_traded_price=4.0,
            total_matched=500.0, starting_price_near=0.0,
            starting_price_far=0.0, adjustment_factor=None, bsp=None,
            sort_priority=1, removal_date=None,
            available_to_back=[PriceSize(price=999.0, size=10.0)],  # junk
            available_to_lay=[PriceSize(price=999.0, size=10.0)],   # junk
        )
        order = bm.passive_book.place(
            runner, stake=5.0, side=BetSide.LAY,
            market_id="m1", tick_index=0,
            # default path: no `price` supplied
        )
        assert order is None


# ── Asymmetric naked-loss reward ─────────────────────────────────────────────


class TestAsymmetricNakedLossReward:
    """The raw reward in scalping mode is
    ``locked_pnl + 0.5 * sum(min(0, per_pair_naked_pnl))``
    (per-pair aggregation, 2026-04-18 ``scalping-naked-asymmetry``).
    Individual naked losses count (at 0.5×); windfalls remain
    excluded; wins no longer cancel unrelated losses."""

    def test_naked_loss_subtracted_from_raw_at_half_factor(self,
                                                            scalping_config):
        """Drive a single naked-back episode where the runner LOSES so
        the incomplete pair's aggressive leg settles negative, and
        verify ``raw_pnl_reward == 0.5 * per_pair_naked_loss``."""
        cfg = dict(scalping_config)
        cfg["reward"] = dict(cfg["reward"])
        # Strip shaping so raw_pnl_reward is the only thing we measure.
        cfg["reward"]["naked_penalty_weight"] = 0.0
        cfg["reward"]["early_lock_bonus_weight"] = 0.0
        cfg["reward"]["terminal_bonus_weight"] = 0.0
        cfg["reward"]["efficiency_penalty"] = 0.0

        # Force the synthetic race winner to be a runner OTHER than 101
        # so the agent's back on 101 loses → the incomplete pair's
        # aggressive leg settles at a loss.
        day = _make_day(n_races=1, n_pre_ticks=3)
        for race in day.races:
            race.winner_selection_id = 102
            race.winning_selection_ids = {102}
        env = BetfairEnv(day, cfg)

        a = np.zeros(14 * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32)
        a[0] = 1.0      # back
        a[14] = -0.8    # small stake
        a[28] = 1.0     # aggressive
        a[56] = 1.0     # max ticks (passive lay placed far → won't fill)
        env.reset()
        env.step(a)

        # Keep pair_ids intact — scalping-mode bets are always paired,
        # and the per-pair accessor needs the aggressive leg's pair_id
        # to classify it as a naked pair. Clear the passive_book so the
        # resting lay can't match on subsequent ticks (would turn the
        # pair complete and change the arithmetic). Previous iteration
        # of this test stripped pair_ids to force the old
        # ``race_pnl - locked - closed`` aggregate path; that path is
        # now gone (2026-04-18 ``scalping-naked-asymmetry``).
        bm = env.bet_manager
        bm.passive_book._orders.clear()
        bm.passive_book._orders_by_sid.clear()

        hold = np.zeros_like(a)
        terminated = False
        info: dict = {}
        while not terminated:
            _, _, terminated, _, info = env.step(hold)

        # The aggressive back on a losing runner produced a loss.
        assert info["naked_pnl"] < 0.0
        assert info["arbs_completed"] == 0
        assert info["locked_pnl"] == pytest.approx(0.0)
        # With exactly one naked pair, sum(min(0, per_pair)) ==
        # per_pair_loss == aggregate naked_pnl, so raw = 0.5 × naked_pnl.
        expected_raw = 0.5 * info["naked_pnl"]
        assert info["raw_pnl_reward"] == pytest.approx(expected_raw, abs=1e-6)


# ── Realistic MAX_ARB_TICKS bound ────────────────────────────────────────────


class TestArbTicksBounds:

    def test_max_arb_ticks_realistic_bound(self):
        """MAX_ARB_TICKS was 100 (fantasy zone); 25 reflects what
        actually fills in real markets. Guard against accidental
        re-expansion."""
        assert MAX_ARB_TICKS == 25
