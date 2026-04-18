"""Tests for forced-arbitrage (scalping) mechanics — Issue 05, session 1."""

from __future__ import annotations

import numpy as np
import pytest

from data.episode_builder import PriceSize
from env.bet_manager import BetManager, BetSide, PassiveOrder, PassiveOrderBook
from env.betfair_env import (
    ACTIONS_PER_RUNNER,
    MAX_ARB_TICKS,
    MIN_ARB_TICKS,
    POSITION_DIM,
    SCALPING_ACTIONS_PER_RUNNER,
    SCALPING_POSITION_DIM,
    BetfairEnv,
    _compute_scalping_reward_terms,
)
from env.exchange_matcher import ExchangeMatcher
from env.tick_ladder import snap_to_tick, tick_offset, ticks_between

from tests.test_betfair_env import _make_day, _make_runner_snap


# ── tick_ladder utilities ───────────────────────────────────────────────────


class TestTickLadder:
    def test_snap_inside_band(self):
        assert snap_to_tick(1.235) == 1.24 or snap_to_tick(1.235) == 1.23

    def test_snap_clamps_to_min(self):
        assert snap_to_tick(1.0) == 1.01

    def test_snap_clamps_to_max(self):
        assert snap_to_tick(1500.0) == 1000.0

    def test_offset_up_within_band(self):
        # 1.50 → +1 tick = 1.51 (0.01 band)
        assert tick_offset(1.50, 1, +1) == 1.51

    def test_offset_down_within_band(self):
        assert tick_offset(1.50, 1, -1) == 1.49

    def test_offset_crosses_band_boundary(self):
        # 2.00 → +1 tick = 2.02 (entering 0.02 band)
        assert tick_offset(2.00, 1, +1) == 2.02
        # 3.00 → +1 tick = 3.05 (entering 0.05 band)
        assert tick_offset(3.00, 1, +1) == 3.05

    def test_offset_down_across_band(self):
        # 2.00 is on the boundary — one tick down goes into the 0.01 band
        assert tick_offset(2.00, 1, -1) == 1.99

    def test_offset_zero_returns_snapped(self):
        assert tick_offset(3.14, 0, +1) == snap_to_tick(3.14)

    def test_offset_clamps_at_max(self):
        # Huge tick count should clamp to 1000.
        assert tick_offset(999.0, 10_000, +1) == 1000.0

    def test_offset_clamps_at_min(self):
        assert tick_offset(1.02, 10_000, -1) == 1.01

    def test_ticks_between_within_band(self):
        assert ticks_between(1.50, 1.55) == 5

    def test_ticks_between_across_bands(self):
        # 1.99 → 2.00 (1 tick), 2.00 → 2.02 (1 tick) = 2 total
        assert ticks_between(1.99, 2.02) == 2

    def test_offset_rejects_bad_direction(self):
        with pytest.raises(ValueError):
            tick_offset(3.0, 1, 0)


# ── BetManager pair_id helpers ──────────────────────────────────────────────


class TestPairedPositions:
    def _runner(self, sid=101):
        return _make_runner_snap(sid, ltp=4.0, back_price=4.0, lay_price=4.1,
                                 size=100.0)

    def test_get_paired_positions_empty(self):
        bm = BetManager(starting_budget=100.0)
        assert bm.get_paired_positions() == []

    def test_get_paired_positions_incomplete(self):
        bm = BetManager(starting_budget=100.0)
        bm.place_back(self._runner(), stake=10.0, market_id="m1",
                      pair_id="abc123")
        pairs = bm.get_paired_positions(market_id="m1")
        assert len(pairs) == 1
        assert pairs[0]["pair_id"] == "abc123"
        assert not pairs[0]["complete"]
        assert pairs[0]["locked_pnl"] == 0.0

    def test_equal_stake_pair_locks_nothing(self):
        """Equal stakes on a back+lay pair guarantee £0 in the worst case.

        The pair's win outcome might be a large directional profit, but
        the lose outcome is break-even. The LOCKED floor — what the
        agent actually nailed down by sizing — is zero. Previously the
        formula was ``stake × spread × (1-comm)`` which reported the
        MAX outcome and masked the fact that the agent hadn't hedged
        properly.
        """
        bm = BetManager(starting_budget=100.0)
        bm.place_back(
            _make_runner_snap(101, ltp=5.0, back_price=5.0, lay_price=5.2, size=200.0),
            stake=10.0, market_id="m1", pair_id="pp",
        )
        bm.place_lay(
            _make_runner_snap(101, ltp=4.5, back_price=4.5, lay_price=4.6, size=200.0),
            stake=10.0, market_id="m1", pair_id="pp",
        )
        pairs = bm.get_paired_positions(market_id="m1", commission=0.05)
        assert len(pairs) == 1
        p = pairs[0]
        assert p["complete"]
        # back £10 @ 5.0, lay £10 @ 4.6 (equal stakes).
        #   win_pnl  = 10×4×0.95 − 10×3.6 = £2.00
        #   lose_pnl = −10 + 10×0.95     = −£0.50
        # Floor = max(0, min) = 0. Equal-stake pairs never lock.
        assert p["locked_pnl"] == pytest.approx(0.0, abs=1e-6)

    def test_properly_sized_pair_locks_positive_floor(self):
        """Sizing the lay stake by S_back × P_back / P_lay locks profit."""
        bm = BetManager(starting_budget=1000.0)
        # Back £20 @ 12.5. To lock, lay stake = 20 × 12.5 / 6.0 ≈ £41.67.
        bm.place_back(
            _make_runner_snap(
                101, ltp=12.5, back_price=12.5, lay_price=12.6, size=500.0,
            ),
            stake=20.0, market_id="m1", pair_id="pp",
        )
        bm.place_lay(
            _make_runner_snap(
                101, ltp=6.0, back_price=6.0, lay_price=6.0, size=500.0,
            ),
            stake=41.67, market_id="m1", pair_id="pp",
        )
        pairs = bm.get_paired_positions(market_id="m1", commission=0.05)
        assert len(pairs) == 1
        p = pairs[0]
        assert p["complete"]
        # win_pnl  = 20×11.5×0.95 − 41.67×5  = 218.50 − 208.35 = £10.15
        # lose_pnl = −20 + 41.67×0.95        = −20 + 39.5865   = £19.59
        # Floor ≈ £10.15 (commission on the back leg makes win the
        # tighter side with these prices).
        assert p["locked_pnl"] > 5.0
        assert p["locked_pnl"] < 25.0

    def test_backwards_pair_locks_zero(self):
        """A pair where prices moved the wrong way after entry locks £0.

        Back at 4.0, lay at 4.2 (price drifted out after backing). The
        pair has a negative "spread" and will lose on every outcome;
        locked_pnl must clamp to zero (the agent didn't earn anything,
        even if realised P&L happens to be positive on a lucky outcome).
        """
        bm = BetManager(starting_budget=100.0)
        bm.place_back(
            _make_runner_snap(101, ltp=4.0, back_price=4.0, lay_price=4.1, size=200.0),
            stake=10.0, market_id="m1", pair_id="pp",
        )
        bm.place_lay(
            _make_runner_snap(101, ltp=4.2, back_price=4.2, lay_price=4.3, size=200.0),
            stake=10.0, market_id="m1", pair_id="pp",
        )
        pairs = bm.get_paired_positions(market_id="m1", commission=0.05)
        assert pairs[0]["locked_pnl"] == pytest.approx(0.0, abs=1e-6)

    def test_get_naked_exposure_back(self):
        bm = BetManager(starting_budget=100.0)
        bm.place_back(self._runner(), stake=10.0, market_id="m1")
        # Naked back exposure = matched stake
        assert bm.get_naked_exposure(market_id="m1") == pytest.approx(10.0)

    def test_get_naked_exposure_excludes_completed_pair(self):
        bm = BetManager(starting_budget=100.0)
        bm.place_back(
            _make_runner_snap(101, ltp=5.0, back_price=5.0, lay_price=5.2, size=200.0),
            stake=10.0, market_id="m1", pair_id="pp",
        )
        bm.place_lay(
            _make_runner_snap(101, ltp=4.5, back_price=4.5, lay_price=4.6, size=200.0),
            stake=10.0, market_id="m1", pair_id="pp",
        )
        # Both legs matched → zero naked exposure.
        assert bm.get_naked_exposure(market_id="m1") == pytest.approx(0.0)


# ── BetfairEnv scalping action / obs spaces ──────────────────────────────────


@pytest.fixture
def scalping_config() -> dict:
    return {
        "training": {
            "max_runners": 14,
            "starting_budget": 100.0,
            "max_bets_per_race": 20,
            "scalping_mode": True,
        },
        "actions": {"force_aggressive": True},
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "efficiency_penalty": 0.01,
        },
    }


@pytest.fixture
def legacy_config() -> dict:
    return {
        "training": {
            "max_runners": 14,
            "starting_budget": 100.0,
            "max_bets_per_race": 20,
        },
        "actions": {"force_aggressive": True},
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "efficiency_penalty": 0.01,
        },
    }


class TestScalpingEnvSpaces:
    def test_action_space_expanded_when_scalping(self, scalping_config):
        env = BetfairEnv(_make_day(n_races=1), scalping_config)
        assert env.action_space.shape == (14 * SCALPING_ACTIONS_PER_RUNNER,)

    def test_action_space_unchanged_when_scalping_off(self, legacy_config):
        env = BetfairEnv(_make_day(n_races=1), legacy_config)
        assert env.action_space.shape == (14 * ACTIONS_PER_RUNNER,)

    def test_obs_space_grows_when_scalping(self, scalping_config, legacy_config):
        env_on = BetfairEnv(_make_day(n_races=1), scalping_config)
        env_off = BetfairEnv(_make_day(n_races=1), legacy_config)
        # Session 01 of scalping-active-management grew
        # SCALPING_POSITION_DIM from 2 → 4 (added
        # seconds_since_passive_placed + passive_price_vs_current_ltp_ticks).
        # 4 extra per runner + 2 global = 4*14 + 2 = 58 extra.
        assert env_on.observation_space.shape[0] == env_off.observation_space.shape[0] + 58

    def test_legacy_step_matches_before_scalping(self, legacy_config):
        """Round-trip a random-seeded rollout to verify shape/behaviour invariance."""
        env = BetfairEnv(_make_day(n_races=1), legacy_config)
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs2, _, _, _, _ = env.step(action)
        assert obs2.shape == env.observation_space.shape


class TestScalpingPairedPlacement:
    def test_aggressive_back_generates_paired_lay(self, scalping_config):
        """An aggressive back fill auto-creates a passive lay at price - arb ticks."""
        env = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), scalping_config)
        env.reset()

        # Build a 5-dim action: back signal for slot 0, full stake, aggressive,
        # no cancel, min arb ticks (-1 → 1 tick away).
        a = np.zeros(14 * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32)
        a[0] = 1.0            # signal = BACK
        a[14] = -0.8          # stake = ~10% of budget (leaves room for lay liability)
        a[28] = 1.0           # aggression = aggressive
        a[42] = 0.0           # cancel off
        a[56] = -1.0          # arb_spread = min ticks
        env.step(a)

        bm = env.bet_manager
        # One matched aggressive back + one resting passive lay with the
        # same pair_id.
        pair_ids = {b.pair_id for b in bm.bets if b.pair_id is not None}
        assert len(pair_ids) == 1, bm.bets
        resting = [o for o in bm.passive_book.orders if o.pair_id is not None]
        assert len(resting) == 1
        assert resting[0].pair_id in pair_ids
        assert resting[0].side is BetSide.LAY

    def test_paired_passive_stake_sized_asymmetrically(self, scalping_config):
        """Passive lay stake = S_back × [P_back × (1 − c) + c] / (P_lay − c).

        This commission-aware closed-form is the only stake ratio that
        equalises P&L across both race outcomes (see
        ``plans/scalping-equal-profit-sizing/purpose.md``). The earlier
        ``S_back × P_back / P_lay`` form only equalises *exposure* when
        commission is non-zero. A BACK→LAY pair still places MORE lay
        stake than the aggressive back stake (because passive_price <
        agg_price) — the magnitude is just slightly smaller than the
        commission-free formula yielded.
        """
        from env.scalping_math import equal_profit_lay_stake

        env = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), scalping_config)
        env.reset()
        a = np.zeros(14 * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32)
        a[0] = 1.0
        a[14] = -0.8
        a[28] = 1.0
        a[56] = -1.0       # min ticks so fill probability stays high
        env.step(a)

        bm = env.bet_manager
        agg = [b for b in bm.bets if b.pair_id is not None][0]
        resting = [
            o for o in bm.passive_book.orders if o.pair_id == agg.pair_id
        ][0]
        # Agg is BACK, passive is LAY at a lower price.
        assert agg.side is BetSide.BACK
        assert resting.side is BetSide.LAY
        assert resting.price < agg.average_price
        # Correct equal-profit asymmetric stake.
        expected = equal_profit_lay_stake(
            back_stake=agg.matched_stake,
            back_price=agg.average_price,
            lay_price=resting.price,
            commission=scalping_config["reward"].get("commission", 0.05),
        )
        assert resting.requested_stake == pytest.approx(expected, rel=1e-6)
        # And the passive stake must be STRICTLY larger than the agg
        # stake for BACK→LAY on a lower-priced passive (proves the
        # regression away from the old equal-stake behaviour).
        assert resting.requested_stake > agg.matched_stake

    def test_scalping_off_no_paired_order(self, legacy_config):
        env = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), legacy_config)
        env.reset()
        a = np.zeros(14 * ACTIONS_PER_RUNNER, dtype=np.float32)
        a[0] = 1.0
        a[14] = -0.8
        a[28] = 1.0
        env.step(a)
        bm = env.bet_manager
        # No paired orders should be created when scalping is off.
        assert not any(b.pair_id for b in bm.bets)
        assert not any(o.pair_id for o in bm.passive_book.orders)

    def test_arb_ticks_mapping_respects_range(self, scalping_config):
        """arb_spread=+1 → MAX ticks, -1 → MIN ticks (direction-aware)."""
        env = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), scalping_config)
        env.reset()
        a = np.zeros(14 * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32)
        a[0] = 1.0
        a[14] = -0.8
        a[28] = 1.0
        a[56] = 1.0   # arb_spread = max
        env.step(a)
        bm = env.bet_manager
        # Aggressive back priced around 4.2 in synthetic data; 15 ticks down
        # on the 1.01–2.00 and 2.00–3.00 and 3.00–4.00 ladder. Verify the
        # resting passive lay rests strictly below the fill price.
        backs = [b for b in bm.bets if b.side is BetSide.BACK and b.pair_id]
        assert backs
        passive_lays = [
            o for o in bm.passive_book.orders
            if o.pair_id == backs[0].pair_id and o.side is BetSide.LAY
        ]
        assert passive_lays
        assert passive_lays[0].price < backs[0].average_price


# ── Session 2: Reward + settlement ──────────────────────────────────────────


class TestScalpingReward:
    """Reward structure for forced-arbitrage mode (Issue 05, session 2)."""

    def _run_episode(self, env, action):
        """Drive *env* through one full episode with *action* each step.

        Returns the final ``info`` dict after termination.
        """
        obs, _ = env.reset()
        terminated = False
        info = {}
        while not terminated:
            obs, _, terminated, _, info = env.step(action)
        return info

    def test_completed_arb_locks_real_pnl_via_race_pnl(self, scalping_config):
        """A completed arb's locked PnL shows up in raw race P&L, not shaping."""
        env = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), scalping_config)
        env.reset()
        a = np.zeros(14 * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32)
        a[0] = 1.0
        a[14] = -0.8
        a[28] = 1.0
        a[56] = 1.0        # MAX ticks — large enough spread to beat commission
        env.step(a)

        bm = env.bet_manager
        # Force the paired passive to "fill" by creating a matching Bet
        # with the same pair_id. This is a test harness shortcut — it
        # skips the on_tick volume accumulation but is equivalent to a
        # real fill for settlement purposes.
        from env.bet_manager import Bet, BetSide
        agg = [b for b in bm.bets if b.pair_id is not None][0]
        resting = [
            o for o in bm.passive_book.orders if o.pair_id == agg.pair_id
        ][0]
        bm.bets.append(Bet(
            selection_id=resting.selection_id,
            side=resting.side,
            requested_stake=resting.requested_stake,
            matched_stake=resting.requested_stake,
            average_price=resting.price,
            market_id=resting.market_id,
            ltp_at_placement=resting.ltp_at_placement,
            pair_id=resting.pair_id,
            tick_index=2,
        ))

        pairs = bm.get_paired_positions(
            market_id=agg.market_id, commission=0.05,
        )
        assert pairs and pairs[0]["complete"]
        # Completed pair: passive leg auto-sized by ``_maybe_place_paired``
        # (S_lay = S_back × P_back / P_lay). At MAX_ARB_TICKS the spread
        # is large enough that the asymmetric sizing produces a positive
        # floor even after commission — a real locked P&L rather than a
        # lucky directional outcome.
        assert pairs[0]["locked_pnl"] > 0.0

    def test_precision_and_early_pick_zeroed_in_scalping_mode(
        self, scalping_config,
    ):
        """Directional shaping is switched off when scalping_mode is on.

        Post-2026-04-18 (``naked-clip-and-stability``) shaped also
        carries the naked-winner clip and close-bonus, so an absolute
        threshold on ``shaped_bonus`` no longer isolates the
        precision/early_pick leak. Differential form: two runs with
        identical bets but radically different precision/early_pick
        weights must produce identical shaped_bonus — the clip and
        bonus depend only on naked outcomes, not these gates.
        """
        cfg_hi = dict(scalping_config)
        cfg_hi["reward"] = dict(cfg_hi["reward"])
        cfg_hi["reward"]["precision_bonus"] = 10.0
        cfg_hi["reward"]["early_pick_bonus_min"] = 5.0
        cfg_hi["reward"]["early_pick_bonus_max"] = 5.0

        cfg_lo = dict(scalping_config)
        cfg_lo["reward"] = dict(cfg_lo["reward"])
        cfg_lo["reward"]["precision_bonus"] = 0.0
        cfg_lo["reward"]["early_pick_bonus_min"] = 1.0
        cfg_lo["reward"]["early_pick_bonus_max"] = 1.0

        a = np.zeros(14 * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32)
        a[0] = 1.0
        a[14] = -0.8
        a[28] = 1.0

        env_hi = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), cfg_hi)
        info_hi = self._run_episode(env_hi, a)
        env_lo = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), cfg_lo)
        info_lo = self._run_episode(env_lo, a)

        assert info_hi["shaped_bonus"] == pytest.approx(
            info_lo["shaped_bonus"], abs=1e-6,
        )

    def test_naked_penalty_scales_with_exposure_weight(self, scalping_config):
        """Doubling the penalty weight roughly doubles the shaped penalty."""
        cfg_lo = dict(scalping_config)
        cfg_lo["reward"] = dict(cfg_lo["reward"])
        cfg_lo["reward"]["naked_penalty_weight"] = 1.0

        cfg_hi = dict(scalping_config)
        cfg_hi["reward"] = dict(cfg_hi["reward"])
        cfg_hi["reward"]["naked_penalty_weight"] = 4.0

        a = np.zeros(14 * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32)
        a[0] = 1.0
        a[14] = -0.8
        a[28] = 1.0
        a[56] = 1.0  # passive 15 ticks away → likely naked

        env_lo = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), cfg_lo)
        info_lo = self._run_episode(env_lo, a)

        env_hi = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), cfg_hi)
        info_hi = self._run_episode(env_hi, a)

        # Higher weight → more negative shaped contribution.
        assert info_hi["shaped_bonus"] < info_lo["shaped_bonus"] - 0.01

    def test_early_lock_bonus_is_time_proportional(self, scalping_config):
        """A pair that locks on an earlier tick yields a larger bonus."""
        # Drive the reward weight up so the bonus dominates shaping.
        cfg = dict(scalping_config)
        cfg["reward"] = dict(cfg["reward"])
        cfg["reward"]["early_lock_bonus_weight"] = 10.0
        # Disable penalty to isolate the bonus signal.
        cfg["reward"]["naked_penalty_weight"] = 0.0

        env = BetfairEnv(_make_day(n_races=1, n_pre_ticks=10), cfg)
        env.reset()
        a = np.zeros(14 * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32)
        a[0] = 1.0
        a[14] = -0.8
        a[28] = 1.0
        a[56] = -1.0
        env.step(a)

        from env.bet_manager import Bet
        bm = env.bet_manager
        agg = [b for b in bm.bets if b.pair_id is not None][0]
        resting = [
            o for o in bm.passive_book.orders if o.pair_id == agg.pair_id
        ][0]

        # Simulate the passive filling on tick 1 (very early — lots of
        # runway left) and compare against a fill on the penultimate
        # tick (almost no runway).
        def synth_pair_fill(fill_tick: int) -> float:
            env2 = BetfairEnv(_make_day(n_races=1, n_pre_ticks=10), cfg)
            env2.reset()
            env2.step(a)
            bm2 = env2.bet_manager
            agg2 = [b for b in bm2.bets if b.pair_id is not None][0]
            resting2 = [
                o for o in bm2.passive_book.orders if o.pair_id == agg2.pair_id
            ][0]
            # Override the aggressive leg + synthesise a passive leg at a
            # price that guarantees ``locked_pnl > 0``. Without this the
            # pair locks £0 after 5 % commission and the post-fix
            # early_lock_bonus gate silences the bonus on both early and
            # late — breaking time-proportionality asymmetry the test is
            # meant to exercise.
            #
            # Aggressive: BACK £10 @ 4.00. Passive: LAY @ 3.40, stake
            # sized by the equal-P&L formula S_passive = 10 × 4.00/3.40 =
            # 11.765. Lose-side = −10 + 11.765 × 0.95 ≈ +£1.18, win-side
            # ≈ +£0.39, so locked = +£0.39 > 0.
            agg2.matched_stake = 10.0
            agg2.average_price = 4.0
            lock_stake = 10.0 * 4.0 / 3.4
            bm2.bets.append(Bet(
                selection_id=resting2.selection_id,
                side=resting2.side,
                requested_stake=lock_stake,
                matched_stake=lock_stake,
                average_price=3.4,
                market_id=resting2.market_id,
                ltp_at_placement=resting2.ltp_at_placement,
                pair_id=resting2.pair_id,
                tick_index=fill_tick,
            ))
            # Manually flush passive order so settlement doesn't double up.
            bm2.passive_book._orders = [
                o for o in bm2.passive_book._orders if id(o) != id(resting2)
            ]
            for sid_orders in bm2.passive_book._orders_by_sid.values():
                for o in sid_orders[:]:
                    if id(o) == id(resting2):
                        sid_orders.remove(o)
            # Step through remaining ticks to drive settlement.
            terminated = False
            info = {}
            while not terminated:
                _, _, terminated, _, info = env2.step(a)
            return info["shaped_bonus"]

        early = synth_pair_fill(1)
        late = synth_pair_fill(9)
        assert early > late

    def test_early_lock_bonus_gated_on_locked_pnl_positive(self, scalping_config):
        """Zero-locked completed pairs contribute zero early_lock_bonus.

        Without this gate the agent can farm the shaped bonus by doing
        1-tick pairs that fill instantly (``remaining_frac ≈ 1``) but
        round-trip for £0 after commission — observed in the
        activation-A-baseline gen-0 population as many
        ``Arb completed: Back @ X / Lay @ X−1tick → locked £+0.00`` lines.
        """
        from env.bet_manager import Bet

        def run_with_passive_price(passive_price: float) -> tuple[float, float]:
            """Replay an identical setup with a specified passive-leg
            price. Returns (shaped_bonus, locked_pnl)."""
            cfg = dict(scalping_config)
            cfg["reward"] = dict(cfg["reward"])
            cfg["reward"]["early_lock_bonus_weight"] = 10.0
            cfg["reward"]["naked_penalty_weight"] = 0.0
            # Zero out early_pick_bonus so shaped_bonus is dominated by
            # the early_lock_bonus term we're probing.
            cfg["reward"]["early_pick_bonus_min"] = 1.0
            cfg["reward"]["early_pick_bonus_max"] = 1.0
            # Efficiency penalty lands on bet_count; keep deterministic.
            cfg["reward"]["efficiency_penalty"] = 0.0
            cfg["reward"]["precision_bonus"] = 0.0

            env = BetfairEnv(_make_day(n_races=1, n_pre_ticks=10), cfg)
            env.reset()
            a = np.zeros(14 * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32)
            a[0] = 1.0
            a[14] = -0.8
            a[28] = 1.0
            a[56] = -1.0
            env.step(a)
            bm = env.bet_manager
            agg = [b for b in bm.bets if b.pair_id is not None][0]
            resting = [
                o for o in bm.passive_book.orders if o.pair_id == agg.pair_id
            ][0]

            # Override aggressive BACK £10 @ 4.00, passive LAY @ given
            # price. Equal-P&L sizing S_lay = 10 × 4.00 / passive_price.
            agg.matched_stake = 10.0
            agg.average_price = 4.0
            lock_stake = 10.0 * 4.0 / passive_price
            bm.bets.append(Bet(
                selection_id=resting.selection_id,
                side=resting.side,
                requested_stake=lock_stake,
                matched_stake=lock_stake,
                average_price=passive_price,
                market_id=resting.market_id,
                ltp_at_placement=resting.ltp_at_placement,
                pair_id=resting.pair_id,
                tick_index=1,  # very early fill — max remaining_frac
            ))
            bm.passive_book._orders = [
                o for o in bm.passive_book._orders if id(o) != id(resting)
            ]
            for sid_orders in bm.passive_book._orders_by_sid.values():
                for o in sid_orders[:]:
                    if id(o) == id(resting):
                        sid_orders.remove(o)
            pairs = bm.get_paired_positions()
            locked_pnls = [p["locked_pnl"] for p in pairs if p["complete"]]

            terminated = False
            info: dict = {}
            while not terminated:
                _, _, terminated, _, info = env.step(a)
            return info["shaped_bonus"], max(locked_pnls) if locked_pnls else 0.0

        # Zero-locked pair: 1-tick below (4.00 → 3.95) round-trips to
        # min(−1.38, −0.38) = −1.38 after 5% commission, floored to 0.
        zero_bonus, zero_locked = run_with_passive_price(3.95)
        # Positive-locked pair: 3.40 is wide enough to lock +£0.39.
        positive_bonus, positive_locked = run_with_passive_price(3.40)

        # Test setup sanity — must exercise both regimes.
        assert zero_locked == 0.0, (
            f"test setup failed: expected zero-locked, got {zero_locked}"
        )
        assert positive_locked > 0.0, (
            f"test setup failed: expected positive-locked, got {positive_locked}"
        )
        # With the gate, the zero-lock scenario earns NO early_lock_bonus
        # and the positive-lock scenario DOES earn the full bonus. So
        # positive_bonus should exceed zero_bonus by roughly
        # weight × remaining_frac ≈ 10 × 1 = 10.
        assert positive_bonus - zero_bonus > 5.0, (
            "positive-locked pair should earn substantially more shaped "
            f"bonus than zero-locked. zero={zero_bonus:.3f} "
            f"positive={positive_bonus:.3f}"
        )

    def test_invariant_raw_plus_shaped_equals_total_reward(
        self, scalping_config,
    ):
        """The raw+shaped accounting invariant survives scalping mode."""
        cfg = dict(scalping_config)
        cfg["reward"] = dict(cfg["reward"])
        cfg["reward"]["naked_penalty_weight"] = 2.5
        cfg["reward"]["early_lock_bonus_weight"] = 1.5

        env = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), cfg)
        obs, _ = env.reset()
        total_reward = 0.0
        terminated = False
        info = {}
        a = np.zeros(14 * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32)
        a[0] = 1.0
        a[14] = -0.8
        a[28] = 1.0
        while not terminated:
            obs, r, terminated, _, info = env.step(a)
            total_reward += r
        # Summing raw and shaped must recover the episode reward.
        assert total_reward == pytest.approx(
            info["raw_pnl_reward"] + info["shaped_bonus"], abs=1e-6,
        )

    def test_info_exposes_scalping_rollups(self, scalping_config):
        """Episode info carries arbs_completed / arbs_naked / locked_pnl."""
        env = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), scalping_config)
        a = np.zeros(14 * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32)
        a[0] = 1.0
        a[14] = -0.8
        a[28] = 1.0
        info = self._run_episode(env, a)
        # Keys must exist even when no pairs completed.
        for key in ("arbs_completed", "arbs_naked", "locked_pnl", "naked_pnl"):
            assert key in info
        # arb_events list is always present (Issue 05 — session 3
        # activity-log plumbing) and its length matches arbs_completed.
        assert "arb_events" in info
        assert isinstance(info["arb_events"], list)
        assert len(info["arb_events"]) == info["arbs_completed"]

    def test_info_arb_events_populated_on_completed_pair(self, scalping_config):
        """A completed pair produces one arb_events entry with back/lay prices."""
        env = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), scalping_config)
        env.reset()
        a = np.zeros(14 * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32)
        a[0] = 1.0
        a[14] = -0.8
        a[28] = 1.0
        a[56] = 1.0      # MAX ticks
        env.step(a)

        # Force the paired passive to "fill" by creating a matching Bet
        # (same harness shortcut used by the session-2 tests).
        from env.bet_manager import Bet
        bm = env.bet_manager
        agg = [b for b in bm.bets if b.pair_id is not None][0]
        resting = [
            o for o in bm.passive_book.orders if o.pair_id == agg.pair_id
        ][0]
        bm.bets.append(Bet(
            selection_id=resting.selection_id,
            side=resting.side,
            requested_stake=resting.requested_stake,
            matched_stake=resting.requested_stake,
            average_price=resting.price,
            market_id=resting.market_id,
            ltp_at_placement=resting.ltp_at_placement,
            pair_id=resting.pair_id,
            tick_index=2,
        ))
        # Remove the resting order so settlement doesn't try to double
        # up; use the same flush pattern as the existing tests.
        bm.passive_book._orders = [
            o for o in bm.passive_book._orders if id(o) != id(resting)
        ]
        for sid_orders in bm.passive_book._orders_by_sid.values():
            sid_orders[:] = [
                o for o in sid_orders if id(o) != id(resting)
            ]

        hold = np.zeros(14 * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32)
        terminated = False
        info = {}
        while not terminated:
            _, _, terminated, _, info = env.step(hold)

        assert info["arbs_completed"] == 1
        events = info["arb_events"]
        assert len(events) == 1
        ev = events[0]
        for key in ("selection_id", "back_price", "lay_price", "locked_pnl"):
            assert key in ev
        assert ev["back_price"] > 0.0
        assert ev["lay_price"] > 0.0

    def test_unfilled_passives_cancelled_at_race_off(self, scalping_config):
        """Race-off cancels resting passive legs; their budget is released."""
        env = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), scalping_config)
        env.reset()
        a = np.zeros(14 * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32)
        a[0] = 1.0
        a[14] = -0.8
        a[28] = 1.0
        a[56] = 1.0  # max ticks → passive likely won't fill
        env.step(a)
        # Drive to settlement.
        terminated = False
        info = {}
        while not terminated:
            _, _, terminated, _, info = env.step(a)
        # After race-off cleanup there are no resting passive orders.
        assert info["passive_orders"] == []

    def test_scalping_mode_off_unchanged_shaping(self, legacy_config):
        """Legacy directional shaping still runs when scalping is off."""
        cfg = dict(legacy_config)
        cfg["reward"] = dict(cfg["reward"])
        cfg["reward"]["precision_bonus"] = 2.0
        # The legacy path should yield non-scalping info keys at zero.
        env = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), cfg)
        a = np.zeros(14 * ACTIONS_PER_RUNNER, dtype=np.float32)
        a[0] = 1.0
        a[14] = -0.8
        a[28] = 1.0
        obs, _ = env.reset()
        terminated = False
        info = {}
        while not terminated:
            obs, _, terminated, _, info = env.step(a)
        assert info["arbs_completed"] == 0
        assert info["arbs_naked"] == 0
        assert info["locked_pnl"] == 0.0


# ── Per-pair naked P&L asymmetry (scalping-naked-asymmetry, 2026-04-18) ─────


class TestPerPairNakedAsymmetry:
    """Covers the 2026-04-18 switch from race-aggregate to per-pair
    naked-loss accounting in the scalping raw reward.

    Pre-fix:  raw += 0.5 × min(0, sum(naked_pnls))          (aggregate)
    Post-fix: raw += 0.5 × sum(min(0, per_pair_naked_pnl))  (per-pair)

    The per-pair form stops lucky winning nakeds from masking unrelated
    losing nakeds in the same race. All five cases from
    ``plans/scalping-naked-asymmetry/hard_constraints.md §12``.
    """

    def _place_naked_back(
        self,
        bm: BetManager,
        selection_id: int,
        pair_id: str,
        *,
        price: float = 4.0,
        stake: float = 10.0,
        market_id: str = "m1",
    ) -> None:
        """Place an aggressive back with ``pair_id`` and no paired lay.

        That alone satisfies ``get_paired_positions`` → incomplete pair
        (``complete=False``, aggressive=the back leg) which is exactly
        the "naked" condition the accessor looks for.
        """
        bm.place_back(
            _make_runner_snap(
                selection_id, ltp=price, back_price=price,
                lay_price=price + 0.1, size=1000.0,
            ),
            stake=stake, market_id=market_id, pair_id=pair_id,
        )

    def test_two_naked_pairs_one_win_one_loss_no_cancellation(self):
        """Win+loss pair set: naked term = sum(min(0, …)) = −loss,
        NOT min(0, win+loss) = 0. This is the whole point of the
        plan."""
        bm = BetManager(starting_budget=1000.0)
        # Pair A: naked back on sid=101 @ 4.0, £10 stake → wins +£30
        # if 101 wins.
        self._place_naked_back(bm, 101, "pp_win", price=4.0, stake=10.0)
        # Pair B: naked back on sid=202 @ 4.0, £10 stake → loses
        # −£10 if 202 loses.
        self._place_naked_back(bm, 202, "pp_lose", price=4.0, stake=10.0)

        bm.settle_race({101}, market_id="m1", commission=0.0)

        pnls = bm.get_naked_per_pair_pnls(market_id="m1")
        assert len(pnls) == 2
        # Two distinct per-pair P&Ls: one win (+30), one loss (−10).
        assert sorted(pnls) == [pytest.approx(-10.0), pytest.approx(30.0)]

        # Pre-fix aggregate: min(0, 30 − 10) = 0 — no penalty.
        aggregate_term = min(0.0, sum(pnls))
        assert aggregate_term == pytest.approx(0.0)
        # Post-fix per-pair: min(0, 30) + min(0, −10) = −10 — the
        # loss is not masked.
        per_pair_term = sum(min(0.0, p) for p in pnls)
        assert per_pair_term == pytest.approx(-10.0)

    def test_single_losing_naked_unchanged(self):
        """Single losing naked: pre and post fix give the same −loss."""
        bm = BetManager(starting_budget=1000.0)
        self._place_naked_back(bm, 202, "pp_lose", price=4.0, stake=10.0)

        bm.settle_race({101}, market_id="m1", commission=0.0)

        pnls = bm.get_naked_per_pair_pnls(market_id="m1")
        assert pnls == [pytest.approx(-10.0)]
        assert min(0.0, sum(pnls)) == pytest.approx(-10.0)
        assert sum(min(0.0, p) for p in pnls) == pytest.approx(-10.0)

    def test_single_winning_naked_unchanged(self):
        """Single winning naked: pre and post fix give the same zero."""
        bm = BetManager(starting_budget=1000.0)
        self._place_naked_back(bm, 101, "pp_win", price=4.0, stake=10.0)

        bm.settle_race({101}, market_id="m1", commission=0.0)

        pnls = bm.get_naked_per_pair_pnls(market_id="m1")
        assert pnls == [pytest.approx(30.0)]
        assert min(0.0, sum(pnls)) == pytest.approx(0.0)
        assert sum(min(0.0, p) for p in pnls) == pytest.approx(0.0)

    def test_all_completed_no_naked_contribution(self):
        """A race whose every pair completed contributes zero."""
        bm = BetManager(starting_budget=1000.0)
        # Complete pair: back £10 @ 5.0, lay £10 @ 4.6 on sid=101.
        bm.place_back(
            _make_runner_snap(
                101, ltp=5.0, back_price=5.0, lay_price=5.2, size=200.0,
            ),
            stake=10.0, market_id="m1", pair_id="pp_done",
        )
        bm.place_lay(
            _make_runner_snap(
                101, ltp=4.5, back_price=4.5, lay_price=4.6, size=200.0,
            ),
            stake=10.0, market_id="m1", pair_id="pp_done",
        )
        bm.settle_race({101}, market_id="m1", commission=0.05)

        pnls = bm.get_naked_per_pair_pnls(market_id="m1")
        assert pnls == []
        assert sum(min(0.0, p) for p in pnls) == 0.0

    def test_random_zero_ev_nakeds_term_is_non_positive(self):
        """Sample naked P&Ls from a zero-EV symmetric distribution.

        The per-pair penalty is ≤ 0 by construction (each individual
        loss contributes a negative amount; wins contribute 0). Sanity
        check that the aggregation preserves the asymmetric design
        intent from ``scalping-asymmetric-hedging``'s purpose.md —
        the accessor consumes realised P&Ls; the plan's ≤0 guarantee
        is what this assertion verifies.

        Also asserts the weaker statistical property: the mean of the
        term over many independent samples is strictly negative (the
        punishment lands on losers), confirming that the per-pair
        form is BY DESIGN more punitive than the pre-fix aggregate on
        heterogeneous naked books.
        """
        rng = np.random.default_rng(seed=20260418)
        per_pair_terms = []
        for _ in range(200):
            samples = rng.normal(0.0, 10.0, size=5).tolist()
            term = sum(min(0.0, p) for p in samples)
            per_pair_terms.append(term)
            assert term <= 0.0, f"per-pair term must be ≤ 0, got {term}"

        # Strictly negative in expectation (average-of-loss-only on a
        # zero-mean draw is ≤ 0 with equality iff all samples ≥ 0 —
        # vanishingly unlikely over 200 × 5 draws).
        mean_term = sum(per_pair_terms) / len(per_pair_terms)
        assert mean_term < -1.0, (
            f"expected strictly negative mean naked term under "
            f"zero-EV sampling, got {mean_term:.3f}"
        )


# ── Naked-winner clip + close bonus (naked-clip-and-stability, 2026-04-18) ──


class TestNakedWinnerClipAndCloseBonus:
    """Reward-shape worked examples from the
    ``plans/naked-clip-and-stability/purpose.md`` outcome table.

    Raw channel reports actual race cashflow: locked-pair floor plus
    full per-pair naked P&L (winners AND losers). Shaped channel
    carries the training-signal adjustments: a 95 % clip on naked
    winners that neutralises directional luck, plus a +£1 bonus per
    ``close_signal`` success that gives the close mechanic a positive
    gradient beyond its realised locked P&L.

    See ``plans/naked-clip-and-stability/hard_constraints.md`` §4–§6.
    """

    def test_single_naked_winner_raw_full_shaped_clipped(self):
        """Naked winner +£100 → raw=+100, shaped=−95, net=+5."""
        raw, shaped = _compute_scalping_reward_terms(
            race_pnl=100.0,
            naked_per_pair=[100.0],
            n_close_signal_successes=0,
        )
        assert raw == pytest.approx(100.0)
        assert shaped == pytest.approx(-95.0)
        assert raw + shaped == pytest.approx(5.0)

    def test_single_naked_loser_raw_full_shaped_zero(self):
        """Naked loser −£80 → raw=−80, shaped=0, net=−80."""
        raw, shaped = _compute_scalping_reward_terms(
            race_pnl=-80.0,
            naked_per_pair=[-80.0],
            n_close_signal_successes=0,
        )
        assert raw == pytest.approx(-80.0)
        assert shaped == pytest.approx(0.0)
        assert raw + shaped == pytest.approx(-80.0)

    def test_mixed_win_and_loss_per_pair_aggregation(self):
        """Winner +£100 and loser −£80 in the same race.

        Raw is the whole-race cashflow (+£20). Shaped clips only the
        winner (−£95); loser contributes 0 to the clip because
        ``max(0, −80) = 0``. Net = −£75 — the aggregate ‘lucky wins
        cancel losses’ masking that per-pair aggregation was
        introduced to kill still cannot rescue this race.
        """
        raw, shaped = _compute_scalping_reward_terms(
            race_pnl=20.0,
            naked_per_pair=[100.0, -80.0],
            n_close_signal_successes=0,
        )
        assert raw == pytest.approx(20.0)
        assert shaped == pytest.approx(-95.0)
        assert raw + shaped == pytest.approx(-75.0)

    def test_scalp_using_close_signal_earns_bonus(self):
        """Closed pair at +£2 cash → raw=+2, shaped=+1, net=+3.

        A profitable ``close_signal`` success contributes the pair's
        cash via ``race_pnl`` (the helper's raw channel) plus the
        shaped +£1 per-close bonus.
        """
        raw, shaped = _compute_scalping_reward_terms(
            race_pnl=2.0,
            naked_per_pair=[],
            n_close_signal_successes=1,
        )
        assert raw == pytest.approx(2.0)
        assert shaped == pytest.approx(1.0)
        assert raw + shaped == pytest.approx(3.0)

    def test_loss_closed_scalp_reports_full_loss_in_raw(self):
        """Close_signal closes a pair at −£5 cash → raw=−5, shaped=+1, net=−4.

        Under Session 01's draft (raw = ``scalping_locked_pnl +
        sum(naked_per_pair)``) the pair's locked floor was 0, no naked
        contribution, so raw=0 and net=+£1 from the close bonus —
        rewarding a trade that actually lost real cash. Under Session
        01b (raw = ``race_pnl``) the close-leg loss flows through
        ``scalping_closed_pnl`` into ``race_pnl`` and net is correctly
        negative. The close bonus still gives closing a positive
        gradient over holding-to-settle (a naked −£80 would net −£80
        vs this close's −£4), so the learning signal favours closing
        without letting close be an unconditional reward.
        """
        raw, shaped = _compute_scalping_reward_terms(
            race_pnl=-5.0,
            naked_per_pair=[],
            n_close_signal_successes=1,
        )
        assert raw == pytest.approx(-5.0)
        assert shaped == pytest.approx(1.0)
        assert raw + shaped == pytest.approx(-4.0)

    def test_multiple_close_signal_successes_accumulate(self):
        """N closes in one race → shaped += N × £1."""
        for n in (2, 3, 5):
            raw, shaped = _compute_scalping_reward_terms(
                race_pnl=0.0,
                naked_per_pair=[],
                n_close_signal_successes=n,
            )
            assert raw == pytest.approx(0.0)
            assert shaped == pytest.approx(float(n))

    def test_raw_plus_shaped_invariant_with_new_terms(self):
        """Mixed race exercises every new term simultaneously:
        locked scalp +£5, naked winner +£50, naked loser −£30, and
        two ``close_signal`` successes. ``race_pnl`` is the sum of
        all cash contributions (+£25). Confirms raw and shaped
        add up to the full contribution (no leakage).
        """
        raw, shaped = _compute_scalping_reward_terms(
            race_pnl=25.0,
            naked_per_pair=[50.0, -30.0],
            n_close_signal_successes=2,
        )
        # Raw: race_pnl = 5 + 50 + (−30) = +£25.
        assert raw == pytest.approx(25.0)
        # Shaped: −0.95 × 50 + 2 × 1.0 = −47.5 + 2.0 = −£45.5.
        assert shaped == pytest.approx(-45.5)
        # Net: +25 + (−45.5) = −£20.5.
        assert raw + shaped == pytest.approx(-20.5)


# ── Session 3: gene / hyperparameter integration ───────────────────────────


class TestScalpingGenes:
    """Scalping-related hyperparameters wire through the search space, the
    env, and the population manager cleanly.
    """

    def test_scalping_genes_in_schema(self):
        """arb_spread_scale, naked_penalty_weight and early_lock_bonus_weight
        are sampleable from config.yaml — so the schema inspector shows them
        and the GA can evolve them."""
        import yaml
        from agents.population_manager import parse_search_ranges

        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        specs = {
            s.name: s
            for s in parse_search_ranges(cfg["hyperparameters"]["search_ranges"])
        }
        for name in (
            "arb_spread_scale",
            "naked_penalty_weight",
            "early_lock_bonus_weight",
        ):
            assert name in specs, f"gene {name} missing from search_ranges"
            assert specs[name].type == "float"

    def test_population_manager_pins_scalping_mode_on_agents(self):
        """Initialised agents carry the run-level scalping_mode flag in hp."""
        import yaml
        from registry.model_store import ModelStore
        from agents.population_manager import PopulationManager

        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        cfg["training"]["scalping_mode"] = True
        cfg["population"]["size"] = 2
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            store = ModelStore(
                db_path=f"{td}/models.db", weights_dir=f"{td}/weights",
            )
            pm = PopulationManager(cfg, model_store=store)
            agents = pm.initialise_population(generation=0, seed=42)
            assert len(agents) == 2
            for a in agents:
                assert a.hyperparameters["scalping_mode"] is True

    def test_env_respects_arb_spread_scale_override(self, scalping_config):
        """arb_spread_scale expands / compresses the mapped tick range
        within the hard [MIN_ARB_TICKS, MAX_ARB_TICKS] clamp."""
        # A scale of 0.5 with arb_frac=1.0 halves the mapped range, so the
        # resulting tick count is below MAX_ARB_TICKS — proving the gene took
        # effect.  Default 1.0 would give MAX_ARB_TICKS exactly.
        day = _make_day(n_races=1, n_pre_ticks=3)
        env_fast = BetfairEnv(day, scalping_config,
                              scalping_overrides={"arb_spread_scale": 0.5})
        env_default = BetfairEnv(day, scalping_config)

        assert env_fast._arb_spread_scale == 0.5
        assert env_default._arb_spread_scale == 1.0

    def test_env_reward_overrides_naked_penalty_weight(self, scalping_config):
        """naked_penalty_weight flows through the existing reward-override
        mechanism so the GA can evolve it per-agent."""
        env = BetfairEnv(
            _make_day(n_races=1),
            scalping_config,
            reward_overrides={"naked_penalty_weight": 3.5,
                              "early_lock_bonus_weight": 0.7},
        )
        assert env._naked_penalty_weight == pytest.approx(3.5)
        assert env._early_lock_bonus_weight == pytest.approx(0.7)

    def test_naked_windfall_in_raw_with_shaped_winner_clip(self, scalping_config):
        """Naked windfall lands in raw at full cash; shaped clips
        winners by 95 % to neutralise the training incentive for
        directional luck.

        Pre-``naked-clip-and-stability`` (2026-04-18) raw masked naked
        winners entirely via ``min(0, ...)`` so the agent saw zero
        reward for a naked win regardless of outcome. The refactor
        moves the asymmetry out of raw — raw now reports actual
        cashflow (``race_pnl``) so winners land at full cash — and
        into shaped, which subtracts 0.95 × per-pair-naked-winner.
        Net training signal on a winning naked is +0.05 × win — small
        enough to avoid teaching "leave bets naked and hope", while
        raw reports honest day P&L. See
        ``plans/naked-clip-and-stability/`` and CLAUDE.md.
        """
        cfg = dict(scalping_config)
        cfg["reward"] = dict(cfg["reward"])
        # Turn non-clip shaping way down so the raw reward + shaped
        # winner-clip are the only signals in the reward stream.
        cfg["reward"]["naked_penalty_weight"] = 0.0
        cfg["reward"]["early_lock_bonus_weight"] = 0.0
        cfg["reward"]["terminal_bonus_weight"] = 0.0
        cfg["reward"]["efficiency_penalty"] = 0.0

        env = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), cfg)
        a = np.zeros(14 * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32)
        a[0] = 1.0       # back signal
        a[14] = -0.8     # small stake
        a[28] = 1.0      # aggressive
        a[56] = 1.0      # 15-tick spread
        env.reset()
        env.step(a)
        # Subsequent steps: hold (no new bets) so we can isolate the
        # behaviour of the first naked leg without more pairs stacking.
        hold = np.zeros(14 * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32)
        # Keep pair_ids intact — scalping-mode bets are always paired,
        # and the per-pair naked accessor needs the aggressive leg's
        # pair_id to classify it. Clear the passive_book so the
        # resting lay never matches on subsequent ticks (would turn
        # the pair complete and change the arithmetic).
        bm = env.bet_manager
        bm.passive_book._orders = []
        bm.passive_book._orders_by_sid.clear()
        terminated = False
        info = {}
        while not terminated:
            _, _, terminated, _, info = env.step(hold)
        # No pair was completed, locked is zero, and day P&L equals
        # the naked aggregate — the naked aggressive leg is the only
        # thing that settled.
        assert info["arbs_completed"] == 0
        assert info["locked_pnl"] == 0.0
        assert info["naked_pnl"] == pytest.approx(info["day_pnl"], abs=1e-6)
        # Raw now reports full race cashflow (including naked P&L).
        assert info["raw_pnl_reward"] == pytest.approx(info["day_pnl"], abs=1e-6)
        # Shaped applies the 95 % winner clip if the naked won, and
        # is zero if it lost. The synthetic fixture produces a
        # winning back here (naked_pnl > 0).
        assert info["naked_pnl"] > 0.0
        assert info["shaped_bonus"] == pytest.approx(
            -0.95 * info["naked_pnl"], abs=1e-6,
        )

    def test_evaluator_collects_scalping_metrics(self, scalping_config):
        """EvaluationDayRecord carries arbs_completed / arbs_naked /
        locked_pnl / naked_pnl sourced from the env's info dict."""
        from registry.model_store import EvaluationDayRecord

        rec = EvaluationDayRecord(
            run_id="r", date="2026-04-01", day_pnl=1.0, bet_count=4,
            winning_bets=2, bet_precision=0.5, pnl_per_bet=0.25,
            early_picks=0, profitable=True,
            arbs_completed=3, arbs_naked=1,
            locked_pnl=1.14, naked_pnl=-0.14,
        )
        assert rec.arbs_completed == 3
        assert rec.arbs_naked == 1
        assert rec.locked_pnl == pytest.approx(1.14)
        assert rec.naked_pnl == pytest.approx(-0.14)


# ── Scalping-active-management session 01: active re-quote ─────────────────


def _scalping_action(
    max_runners: int = 14,
    *,
    slot: int = 0,
    signal: float = 0.0,
    stake: float = -1.0,
    aggression: float = -1.0,
    cancel: float = -1.0,
    arb_spread: float = -1.0,
    requote: float = -1.0,
) -> np.ndarray:
    """Build a scalping-layout action vector with explicit per-dim values.

    Defaults put every slot in a "do nothing" configuration; override the
    specific values the test cares about. ``slot`` controls which runner
    index receives the overrides.
    """
    action = np.zeros(
        max_runners * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32,
    )
    action[slot] = signal
    action[max_runners + slot] = stake
    action[2 * max_runners + slot] = aggression
    action[3 * max_runners + slot] = cancel
    action[4 * max_runners + slot] = arb_spread
    action[5 * max_runners + slot] = requote
    return action


class TestScalpingRequote:
    """Active re-quote mechanic added in scalping-active-management §01."""

    def _make_env(self, scalping_config, **kwargs) -> BetfairEnv:
        return BetfairEnv(
            _make_day(n_races=1, n_pre_ticks=5, n_inplay_ticks=2),
            scalping_config,
            **kwargs,
        )

    def _place_initial_pair(self, env: BetfairEnv) -> tuple:
        """Drive env through one tick to produce an aggressive back + paired lay.

        Returns ``(agg_bet, passive_order)`` for the placed pair. The
        resting passive's ``queue_ahead_at_placement`` is set very high
        so that ``on_tick`` does not auto-fill it on the next step —
        without this the synthetic market (constant ``total_matched``
        and ladder levels that don't match the paired price → queue
        ahead of 0) instantly fills any paired passive, defeating the
        cancel-and-replace tests.
        """
        a = _scalping_action(
            signal=1.0, stake=-0.8, aggression=1.0, arb_spread=-1.0,
        )
        env.step(a)
        bm = env.bet_manager
        paired_bets = [b for b in bm.bets if b.pair_id is not None]
        assert paired_bets, "expected an aggressive fill with a pair_id"
        agg = paired_bets[0]
        pairing = [
            o for o in bm.passive_book.orders if o.pair_id == agg.pair_id
        ]
        assert pairing, "expected a paired passive to rest after the fill"
        # Fence against on_tick auto-fill; see docstring.
        for o in bm.passive_book.orders:
            if o.pair_id is not None:
                o.queue_ahead_at_placement = 1e12
        return agg, pairing[0]

    # ── 1. Re-quote on a runner with no open passive is a no-op. ───────

    def test_requote_noop_without_open_passive(self, scalping_config):
        env = self._make_env(scalping_config)
        env.reset()

        # Fire requote_signal with no prior aggressive placement on slot 0.
        a = _scalping_action(requote=1.0)
        env.step(a)

        bm = env.bet_manager
        # No bets placed; no paired passives resting.
        assert bm.bets == []
        assert bm.passive_book.orders == []
        # Diagnostic tag recorded.
        sid = env._slot_maps[env._race_idx].get(0)
        debug = env._last_action_debug.get(sid, {})
        assert debug.get("requote_attempted") is True
        assert debug.get("requote_failed") is True
        assert debug.get("requote_reason") == "no_open_passive"

    # ── 2. Re-quote cancels old passive and re-places at new offset. ───

    def test_requote_cancels_and_replaces(self, scalping_config):
        env = self._make_env(scalping_config)
        env.reset()

        agg, old_passive = self._place_initial_pair(env)
        bm = env.bet_manager

        # Capture the old price so we can assert it moved.
        old_price = old_passive.price
        old_id = id(old_passive)

        # Fire requote at a wider offset. The placement path's tick math
        # snaps price to tick grid, so the new price almost always differs
        # unless arb_spread=-1 was used both times — here we flip to +1
        # (MAX_ARB_TICKS) to force a large move.
        a = _scalping_action(requote=1.0, arb_spread=1.0)
        env.step(a)

        # Old order no longer in the book (identity check — the book
        # cancelled it, not silently duplicated).
        assert not any(id(o) == old_id for o in bm.passive_book.orders)

        # New passive present with the SAME pair_id.
        new_passives = [
            o for o in bm.passive_book.orders if o.pair_id == agg.pair_id
        ]
        assert len(new_passives) == 1
        new_order = new_passives[0]
        assert new_order.price != old_price
        # Bet history unchanged (no new Bet created — the passive hasn't filled).
        assert len([b for b in bm.bets if b.pair_id is not None]) == 1

    # ── 3. Pair-id preserved across the re-quote. ──────────────────────

    def test_requote_preserves_pair_id(self, scalping_config):
        env = self._make_env(scalping_config)
        env.reset()

        agg, _ = self._place_initial_pair(env)
        bm = env.bet_manager
        a = _scalping_action(requote=1.0, arb_spread=1.0)
        env.step(a)

        new_passives = [
            o for o in bm.passive_book.orders if o.pair_id == agg.pair_id
        ]
        assert new_passives, "re-quoted passive must keep the aggressive bet's pair_id"

        # Simulate the re-quoted passive filling by manually constructing
        # a matching Bet (same harness shortcut used elsewhere in this file).
        resting = new_passives[0]
        from env.bet_manager import Bet
        bm.bets.append(Bet(
            selection_id=resting.selection_id,
            side=resting.side,
            requested_stake=resting.requested_stake,
            matched_stake=resting.requested_stake,
            average_price=resting.price,
            market_id=resting.market_id,
            ltp_at_placement=resting.ltp_at_placement,
            pair_id=resting.pair_id,
            tick_index=env._tick_idx,
        ))
        pairs = bm.get_paired_positions(
            market_id=agg.market_id, commission=0.05,
        )
        assert len(pairs) == 1
        assert pairs[0]["complete"]
        assert pairs[0]["pair_id"] == agg.pair_id

    # ── 4. Budget accounting: net change is just the liability delta. ──

    def test_requote_budget_accounting(self, scalping_config):
        env = self._make_env(scalping_config)
        env.reset()

        agg, old_passive = self._place_initial_pair(env)
        bm = env.bet_manager

        old_reserved = (
            old_passive.reserved_liability
            if old_passive.reserved_liability is not None
            else old_passive.requested_stake * (old_passive.price - 1.0)
        )
        avail_before = bm.available_budget

        a = _scalping_action(requote=1.0, arb_spread=1.0)
        env.step(a)

        # Find the new paired passive.
        new_passives = [
            o for o in bm.passive_book.orders if o.pair_id == agg.pair_id
        ]
        assert new_passives
        new_order = new_passives[0]
        new_reserved = (
            new_order.reserved_liability
            if new_order.reserved_liability is not None
            else new_order.requested_stake * (new_order.price - 1.0)
        )

        # Net change in available_budget = old_reserved - new_reserved
        # (old reservation returned, new reservation taken).
        expected_delta = old_reserved - new_reserved
        assert bm.available_budget == pytest.approx(
            avail_before + expected_delta, abs=0.01,
        )

    # ── 5. Junk-band re-quote: old cancelled, new refused, tag set. ────

    def test_requote_into_junk_band_silent_failure(self, scalping_config):
        env = self._make_env(scalping_config)
        env.reset()

        agg, old_passive = self._place_initial_pair(env)
        bm = env.bet_manager
        sid = old_passive.selection_id

        # Tighten the junk-filter window so the re-quoted tick distance
        # falls outside even a small arb offset.
        bm.passive_book._matcher = ExchangeMatcher(
            max_price_deviation_pct=0.001,
        )

        a = _scalping_action(requote=1.0, arb_spread=1.0)
        env.step(a)

        # No paired passive should remain — the old was cancelled and the
        # new was refused by the junk-band guard.
        assert [
            o for o in bm.passive_book.orders if o.pair_id == agg.pair_id
        ] == []
        # Aggressive leg still exists (we never touch matched bets).
        assert any(b.pair_id == agg.pair_id for b in bm.bets)
        # Diagnostic tag set.
        debug = env._last_action_debug.get(sid, {})
        assert debug.get("requote_failed") is True
        assert debug.get("requote_reason") == "junk_band"

    # ── 6. The re-quote uses the same single-price path: no walking. ───

    def test_requote_ladder_walk_prevented(self, scalping_config):
        env = self._make_env(scalping_config)
        env.reset()
        agg, _ = self._place_initial_pair(env)
        bm = env.bet_manager

        a = _scalping_action(requote=1.0, arb_spread=1.0)
        env.step(a)

        new_passives = [
            o for o in bm.passive_book.orders if o.pair_id == agg.pair_id
        ]
        assert len(new_passives) == 1
        new_order = new_passives[0]
        # The new order rests at a single snapped-to-tick price — it has
        # not walked into a deeper level of the ladder. Asserting the
        # price equals tick_offset(current_ltp, arb_ticks, direction)
        # proves the placement did not spill across levels.
        tick = env.day.races[0].ticks[env._tick_idx - 1]
        runner = next(r for r in tick.runners if r.selection_id == agg.selection_id)
        ltp = runner.last_traded_price
        # arb=1.0 → MAX_ARB_TICKS ticks at default spread_scale=1.0.
        expected_price = tick_offset(ltp, MAX_ARB_TICKS, -1)
        assert new_order.price == expected_price
        # Single resting order — we never doubled up on fills.
        assert new_order.matched_stake == 0.0

    # ── 7. New obs features are present and monotonic with elapsed time. ─

    def test_obs_features_present(self, scalping_config):
        env = self._make_env(scalping_config)
        env.reset()
        agg, _ = self._place_initial_pair(env)

        # Sample the feature block at two successive ticks: elapsed time
        # must be non-decreasing, and the features must live at the
        # expected per-runner offsets.
        obs1 = env._get_position_vector()
        # Hold (no new bet / no requote); let the env advance one tick.
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        obs2 = env._get_position_vector()

        slot = env._runner_maps[env._race_idx][agg.selection_id]
        per_runner = POSITION_DIM + SCALPING_POSITION_DIM
        base = slot * per_runner + POSITION_DIM
        # Feature layout: [has_arb, proximity, seconds_since, price_delta]
        has_arb_1 = obs1[base]
        seconds_since_1 = obs1[base + 2]
        price_delta_1 = obs1[base + 3]
        has_arb_2 = obs2[base]
        seconds_since_2 = obs2[base + 2]

        # Paired passive is resting on both ticks.
        assert has_arb_1 == pytest.approx(1.0)
        assert has_arb_2 == pytest.approx(1.0)
        # Seconds-since is non-negative in [0, 1] and strictly
        # non-decreasing as time passes.
        assert 0.0 <= seconds_since_1 <= 1.0
        assert 0.0 <= seconds_since_2 <= 1.0
        assert seconds_since_2 >= seconds_since_1
        # Price-delta is bounded to [-1, 1].
        assert -1.0 <= price_delta_1 <= 1.0

    # ── 8. With no open passive, the new obs features are exactly 0. ────

    def test_obs_features_zero_without_passive(self, scalping_config):
        env = self._make_env(scalping_config)
        env.reset()

        # No bets placed → no paired passive.
        obs = env._get_position_vector()
        per_runner = POSITION_DIM + SCALPING_POSITION_DIM
        for slot in range(env.max_runners):
            base = slot * per_runner + POSITION_DIM
            # has_open_arb
            assert obs[base] == pytest.approx(0.0)
            # proximity
            assert obs[base + 1] == pytest.approx(0.0)
            # seconds_since_passive_placed
            assert obs[base + 2] == pytest.approx(0.0)
            # passive_price_vs_current_ltp_ticks
            assert obs[base + 3] == pytest.approx(0.0)

    # ── 9. Action-space grows in scalping mode, unchanged otherwise. ────

    def test_action_space_size_grows(self, scalping_config, legacy_config):
        env_on = BetfairEnv(_make_day(n_races=1), scalping_config)
        env_off = BetfairEnv(_make_day(n_races=1), legacy_config)
        # scalping-close-signal session 01 bumped per-runner dim 6 → 7.
        assert env_on.action_space.shape == (14 * 7,)
        assert env_off.action_space.shape == (14 * 4,)

    # ── 10. Pre-Session-01 checkpoints migrate cleanly. ─────────────────

    def test_legacy_checkpoint_loads(self):
        import torch
        from agents.policy_network import (
            PPOLSTMPolicy,
            migrate_scalping_action_head,
        )

        max_runners = 14
        obs_dim = 32  # size doesn't matter — we only inspect actor/log_std.
        old_per_runner = 5
        new_per_runner = 6
        hp = {"lstm_hidden_size": 16, "mlp_hidden_size": 16, "mlp_layers": 1}

        old_net = PPOLSTMPolicy(
            obs_dim=obs_dim,
            action_dim=max_runners * old_per_runner,
            max_runners=max_runners,
            hyperparams=hp,
        )
        # Checkpoint the pre-Session-01 state dict.
        old_state = {k: v.clone() for k, v in old_net.state_dict().items()}

        # Migrate: widen actor-head and log-std to the new per-runner dim.
        migrated = migrate_scalping_action_head(
            old_state,
            max_runners=max_runners,
            old_per_runner=old_per_runner,
            new_per_runner=new_per_runner,
        )

        new_net = PPOLSTMPolicy(
            obs_dim=obs_dim,
            action_dim=max_runners * new_per_runner,
            max_runners=max_runners,
            hyperparams=hp,
        )
        # strict=True must succeed after migration.
        missing, unexpected = new_net.load_state_dict(migrated, strict=True)
        assert missing == [] and unexpected == []

        # Original rows preserved bit-for-bit.
        old_final_w = old_state["actor_head.2.weight"]
        new_final_w = new_net.actor_head[2].weight.detach()
        assert torch.allclose(
            new_final_w[:old_per_runner], old_final_w,
        )
        # New row is present and not equal to a legacy row (freshly inited).
        assert new_final_w.shape[0] == new_per_runner
        assert not torch.allclose(
            new_final_w[new_per_runner - 1], old_final_w[old_per_runner - 1],
        )
        # action_log_std: old entries preserved, new entries are zero init.
        old_log_std = old_state["action_log_std"]
        new_log_std = new_net.action_log_std.detach()
        assert torch.allclose(
            new_log_std[: max_runners * old_per_runner], old_log_std,
        )
        assert torch.allclose(
            new_log_std[max_runners * old_per_runner :],
            torch.zeros(max_runners * (new_per_runner - old_per_runner)),
        )

    # ── 11. raw + shaped ≈ total reward with re-quotes firing. ──────────

    def test_raw_plus_shaped_invariant_holds(self, scalping_config):
        cfg = dict(scalping_config)
        cfg["reward"] = dict(cfg["reward"])
        cfg["reward"]["naked_penalty_weight"] = 1.0
        cfg["reward"]["early_lock_bonus_weight"] = 0.5

        env = BetfairEnv(
            _make_day(n_races=1, n_pre_ticks=5, n_inplay_ticks=2), cfg,
        )
        obs, _ = env.reset()

        # Every step: place + requote. The first tick's re-quote is a
        # no-op (no passive yet); from the second tick onwards the
        # outstanding paired passive gets cancelled and re-placed.
        a = _scalping_action(
            signal=1.0, stake=-0.8, aggression=1.0,
            arb_spread=0.2, requote=1.0,
        )
        total_reward = 0.0
        terminated = False
        info = {}
        while not terminated:
            obs, r, terminated, _, info = env.step(a)
            total_reward += r

        assert total_reward == pytest.approx(
            info["raw_pnl_reward"] + info["shaped_bonus"], abs=1e-6,
        )


# ── Scalping-active-management session 02: fill-probability head ───────────


def _run_ppo_and_measure(
    scalping_config: dict, *, weight: float, assert_eq: bool,
) -> float:
    """Run one PPO update on a tiny synthetic rollout and return the
    policy gradient norm.

    Shared by ``TestFillProbHead`` tests 6 (``weight=0`` noop) and 7
    (``weight>0`` grad-norm lift). Builds a tiny rollout with hand-
    crafted transitions (including a mix of resolved and unresolved
    ``fill_prob_labels``) so the BCE term has something to train on
    when ``weight > 0``.

    When ``assert_eq`` is True, also asserts that the reported loss
    components sum to the session-01 total (weight=0 must contribute
    exactly 0) within 1e-7 — the test-6 invariant.
    """
    import numpy as np
    import torch
    from agents.architecture_registry import create_policy
    from agents.ppo_trainer import PPOTrainer, Rollout, Transition

    cfg = dict(scalping_config)
    hp = {
        "lstm_hidden_size": 16, "mlp_hidden_size": 8, "mlp_layers": 1,
        "learning_rate": 1e-4, "gamma": 0.99, "gae_lambda": 0.95,
        "ppo_clip_epsilon": 0.2, "entropy_coefficient": 0.01,
        "value_loss_coeff": 0.5, "max_grad_norm": 1e9,
        "ppo_epochs": 1, "mini_batch_size": 4,
        "fill_prob_loss_weight": weight,
    }
    env_probe = BetfairEnv(_make_day(n_races=1), cfg)
    obs_dim = int(env_probe.observation_space.shape[0])
    action_dim = int(env_probe.action_space.shape[0])
    max_runners = env_probe.max_runners

    # Seed so weight=0 vs weight=1 runs are directly comparable.
    torch.manual_seed(0)
    np.random.seed(0)
    policy = create_policy(
        name="ppo_lstm_v1",
        obs_dim=obs_dim, action_dim=action_dim,
        max_runners=max_runners, hyperparams=hp,
    )
    trainer = PPOTrainer(policy=policy, config=cfg, hyperparams=hp)

    # Hand-built rollout: 8 transitions, every other one carries a
    # resolved fill-prob label in slot 0 so the BCE term has gradient.
    rng = np.random.default_rng(0)
    rollout = Rollout()
    for i in range(8):
        obs_i = rng.standard_normal(obs_dim).astype(np.float32)
        action_i = rng.standard_normal(action_dim).astype(np.float32)
        labels = np.full(max_runners, np.nan, dtype=np.float32)
        if i % 2 == 0:
            labels[0] = 1.0 if i % 4 == 0 else 0.0
        rollout.append(Transition(
            obs=obs_i, action=action_i,
            log_prob=-1.0, value=0.0,
            reward=float(rng.standard_normal()),
            done=(i == 7),
            training_reward=0.0,
            fill_prob_labels=labels,
        ))

    loss_info = trainer._ppo_update(rollout)

    # Gradient norm across all policy parameters after the last mini-
    # batch's backward. weight=0 and weight=1 runs use the same seed +
    # data, so the only difference is the aux loss contribution.
    grad_sq_sum = 0.0
    for p in policy.parameters():
        if p.grad is not None:
            grad_sq_sum += float(p.grad.detach().pow(2).sum().item())
    grad_norm = grad_sq_sum ** 0.5

    if assert_eq:
        # Test-6 invariant: with weight=0 the aux term must contribute
        # exactly 0 to the optimised loss, so the reported component
        # breakdown sums to the session-01 total.
        pl = loss_info["policy_loss"]
        vl = loss_info["value_loss"]
        ent = loss_info["entropy"]
        aux = loss_info["fill_prob_loss"]
        expected = pl + trainer.value_loss_coeff * vl - trainer.entropy_coeff * ent
        total_with_aux = expected + trainer.fill_prob_loss_weight * aux
        assert total_with_aux == pytest.approx(expected, abs=1e-7)

    return grad_norm


class TestFillProbHead:
    """Fill-probability auxiliary head added in scalping-active-management §02.

    Plumbing-off by default (``fill_prob_loss_weight=0.0``). Each test
    below pins one invariant of the head: shape/range, decision-time
    capture, passive-leg inheritance, BCE correctness, gradient direction,
    weight=0 noop, weight>0 grad-norm lift, parquet back-compat + roundtrip,
    checkpoint back-compat, raw+shaped reward invariant, and unresolved-
    sample exclusion.
    """

    @staticmethod
    def _small_hp() -> dict:
        """Small architecture so construction is cheap across all arches."""
        return {
            "lstm_hidden_size": 32,
            "mlp_hidden_size": 16,
            "mlp_layers": 1,
            "transformer_heads": 2,
            "transformer_depth": 1,
            "transformer_ctx_ticks": 8,
        }

    # ── 1. Shape + range of the new head across all 3 architectures. ────

    def test_fill_prob_output_shape_and_range(self, scalping_config):
        import torch
        from agents.policy_network import (
            PPOLSTMPolicy,
            PPOTimeLSTMPolicy,
            PPOTransformerPolicy,
        )

        env = BetfairEnv(_make_day(n_races=1), scalping_config)
        obs_dim = int(env.observation_space.shape[0])
        max_runners = env.max_runners
        action_dim = int(env.action_space.shape[0])
        hp = self._small_hp()

        for cls in (PPOLSTMPolicy, PPOTimeLSTMPolicy, PPOTransformerPolicy):
            net = cls(
                obs_dim=obs_dim, action_dim=action_dim,
                max_runners=max_runners, hyperparams=hp,
            )
            obs = torch.zeros(3, obs_dim, dtype=torch.float32)
            out = net(obs)
            assert out.fill_prob_per_runner.shape == (3, max_runners), (
                f"{cls.__name__}: expected (3, {max_runners}), got "
                f"{tuple(out.fill_prob_per_runner.shape)}"
            )
            fp = out.fill_prob_per_runner
            assert torch.all(fp >= 0.0), f"{cls.__name__}: fill_prob < 0"
            assert torch.all(fp <= 1.0), f"{cls.__name__}: fill_prob > 1"

    # ── 2. Decision-time capture stamps the prediction on the Bet. ──────

    def test_fill_prob_recorded_on_bet_at_placement(self, scalping_config):
        """Drive one env step with an aggressive back and manually walk
        the capture path the PPO trainer runs per tick — asserts the
        Bet ends up carrying a non-``None`` prediction in [0, 1].
        """
        import torch
        from agents.architecture_registry import create_policy

        cfg = dict(scalping_config)
        hp = self._small_hp()
        day = _make_day(n_races=1, n_pre_ticks=3, n_inplay_ticks=1)
        env = BetfairEnv(day, cfg)
        obs, _ = env.reset()
        policy = create_policy(
            name="ppo_lstm_v1",
            obs_dim=int(env.observation_space.shape[0]),
            action_dim=int(env.action_space.shape[0]),
            max_runners=env.max_runners,
            hyperparams=hp,
        )

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = policy(obs_tensor)
        fp_per_runner = (
            out.fill_prob_per_runner.detach().cpu().numpy().reshape(-1)
        )

        # Drive a definite aggressive back on slot 0 — the force_aggressive
        # flag in scalping_config makes this a single-step placement.
        a = _scalping_action(signal=1.0, stake=-0.8, aggression=1.0)
        _, _, _, _, info = env.step(a)

        # Replay the trainer's stamp-on-bet logic manually.
        sid_to_slot = env.current_runner_to_slot()
        bm = env.bet_manager
        for sid, entry in info.get("action_debug", {}).items():
            if not entry.get("aggressive_placed", False):
                continue
            slot_idx = sid_to_slot.get(sid)
            if slot_idx is None:
                continue
            fp_val = float(fp_per_runner[slot_idx])
            for bet in reversed(bm.bets):
                if (
                    bet.selection_id == sid
                    and bet.fill_prob_at_placement is None
                ):
                    bet.fill_prob_at_placement = fp_val
                    break

        stamped = [
            b for b in bm.bets if b.fill_prob_at_placement is not None
        ]
        assert stamped, (
            "expected at least one Bet to carry fill_prob_at_placement"
        )
        for b in stamped:
            assert 0.0 <= b.fill_prob_at_placement <= 1.0

    # ── 3. Paired passive inherits the aggressive's fill_prob. ──────────

    def test_fill_prob_inherited_by_paired_passive(self):
        """When a paired passive fills, its Bet carries the SAME
        ``fill_prob_at_placement`` as the aggressive leg — captured, not
        recomputed (hard_constraints §10).
        """
        from env.bet_manager import BetManager, BetSide, PassiveOrder

        bm = BetManager(starting_budget=1000.0)
        # Simulate an aggressive back Bet carrying a prediction.
        agg = bm.place_back(
            _make_runner_snap(101, ltp=4.0, back_price=4.0,
                              lay_price=4.1, size=200.0),
            stake=10.0, market_id="m1", pair_id="pair-xyz",
        )
        assert agg is not None
        # Stamp the prediction that the trainer would have written.
        agg.fill_prob_at_placement = 0.73

        # Register a paired passive lay directly so on_tick will match it.
        # queue_ahead=0 + traded_volume>0 → fill threshold crossed.
        order = PassiveOrder(
            selection_id=101,
            side=BetSide.LAY,
            price=4.1,
            requested_stake=10.0,
            queue_ahead_at_placement=0.0,
            placed_tick_index=0,
            market_id="m1",
            ltp_at_placement=4.0,
            pair_id="pair-xyz",
            reserved_liability=0.0,
        )
        book = bm.passive_book
        book._orders.append(order)
        book._orders_by_sid.setdefault(101, []).append(order)

        class _Runner:
            selection_id = 101
            total_matched = 1000.0  # > 0 → traded_volume_since... > 0
            available_to_back: list = []
            available_to_lay: list = []
            last_traded_price = 4.0
            status = "ACTIVE"

        class _Tick:
            runners = [_Runner()]

        book.on_tick(_Tick(), tick_index=1)  # type: ignore[arg-type]

        passive_bets = [
            b for b in bm.bets
            if b.side == BetSide.LAY and b.pair_id == "pair-xyz"
        ]
        assert len(passive_bets) == 1
        assert passive_bets[0].fill_prob_at_placement == pytest.approx(0.73)

    # ── 4. BCE ≈ 0 when predictions exactly match outcomes. ─────────────

    def test_fill_prob_bce_zero_on_perfect_predictions(self):
        import torch
        from agents.ppo_trainer import _compute_fill_prob_bce

        preds = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        loss = _compute_fill_prob_bce(preds, labels)
        assert loss.item() < 1e-5

    # ── 5. BCE gradient has correct sign in both directions. ────────────

    def test_fill_prob_bce_gradient_direction(self):
        import torch
        from agents.ppo_trainer import _compute_fill_prob_bce

        # Case A: pred=0.1, label=1 → gradient of the BCE wrt the
        # pre-sigmoid logit is negative (optimiser would INCREASE the
        # logit to push the prediction up toward 1).
        logit_a = torch.tensor([[0.0]], requires_grad=True)
        pred_a = torch.sigmoid(logit_a - 2.1972245773362196)  # ≈ 0.1
        loss_a = _compute_fill_prob_bce(
            pred_a, torch.tensor([[1.0]], dtype=torch.float32),
        )
        loss_a.backward()
        assert logit_a.grad is not None
        assert logit_a.grad.item() < 0.0

        # Case B: pred=0.9, label=0 → gradient positive (push logit
        # down, move prediction toward 0).
        logit_b = torch.tensor([[0.0]], requires_grad=True)
        pred_b = torch.sigmoid(logit_b + 2.1972245773362196)  # ≈ 0.9
        loss_b = _compute_fill_prob_bce(
            pred_b, torch.tensor([[0.0]], dtype=torch.float32),
        )
        loss_b.backward()
        assert logit_b.grad is not None
        assert logit_b.grad.item() > 0.0

    # ── 6. weight=0 makes total loss byte-identical to session 01. ──────

    def test_fill_prob_weight_zero_is_noop_on_total_loss(
        self, scalping_config,
    ):
        _run_ppo_and_measure(scalping_config, weight=0.0, assert_eq=True)

    # ── 7. weight=1 lifts the total-param gradient norm. ────────────────

    def test_fill_prob_weight_positive_changes_gradient_norm(
        self, scalping_config,
    ):
        grad_norm_zero = _run_ppo_and_measure(
            scalping_config, weight=0.0, assert_eq=False,
        )
        grad_norm_one = _run_ppo_and_measure(
            scalping_config, weight=1.0, assert_eq=False,
        )
        assert grad_norm_one > grad_norm_zero, (
            f"aux loss weight=1 did not lift grad norm "
            f"(zero={grad_norm_zero:.6f}, one={grad_norm_one:.6f})"
        )

    # ── 8. Parquet reader tolerates absence of the new column. ──────────

    def test_parquet_backcompat_missing_column(self, tmp_path):
        import pandas as pd
        from registry.model_store import EvaluationBetRecord, ModelStore

        store = ModelStore(
            db_path=tmp_path / "m.db",
            weights_dir=tmp_path / "w",
            bet_logs_dir=tmp_path / "b",
        )
        rec = EvaluationBetRecord(
            run_id="r1", date="2026-04-17", market_id="m1",
            tick_timestamp="t", seconds_to_off=10.0, runner_id=101,
            runner_name="A", action="back", price=4.0, stake=10.0,
            matched_size=10.0, outcome="won", pnl=3.0,
            fill_prob_at_placement=0.5,
        )
        path = store.write_bet_logs_parquet("r1", "2026-04-17", [rec])
        assert path is not None and path.exists()

        # Rewrite without the new column → simulates a pre-Session-02 file.
        df = pd.read_parquet(path)
        df = df.drop(columns=["fill_prob_at_placement"])
        df.to_parquet(path, index=False)

        loaded = store.get_evaluation_bets("r1")
        assert len(loaded) == 1
        assert loaded[0].fill_prob_at_placement is None

    # ── 9. Parquet roundtrip preserves the new column. ──────────────────

    def test_parquet_roundtrip_with_fill_prob(self, tmp_path):
        from registry.model_store import EvaluationBetRecord, ModelStore

        store = ModelStore(
            db_path=tmp_path / "m.db",
            weights_dir=tmp_path / "w",
            bet_logs_dir=tmp_path / "b",
        )
        rec = EvaluationBetRecord(
            run_id="r2", date="2026-04-17", market_id="m1",
            tick_timestamp="t", seconds_to_off=10.0, runner_id=101,
            runner_name="A", action="back", price=4.0, stake=10.0,
            matched_size=10.0, outcome="won", pnl=3.0,
            fill_prob_at_placement=0.73,
        )
        store.write_bet_logs_parquet("r2", "2026-04-17", [rec])
        loaded = store.get_evaluation_bets("r2")
        assert len(loaded) == 1
        assert loaded[0].fill_prob_at_placement == pytest.approx(0.73)

    # ── 10. Legacy state-dict loads via migrate_fill_prob_head. ─────────

    def test_legacy_checkpoint_loads_with_fill_prob_head(self):
        import torch
        from agents.policy_network import (
            PPOLSTMPolicy,
            migrate_fill_prob_head,
        )

        hp = self._small_hp()
        new_net = PPOLSTMPolicy(
            obs_dim=32, action_dim=14 * 6, max_runners=14, hyperparams=hp,
        )

        # Simulate a pre-Session-02 state-dict by dropping the new head's
        # keys — everything else matches the current architecture.
        legacy_state = {
            k: v.clone() for k, v in new_net.state_dict().items()
            if not k.startswith("fill_prob_head.")
        }
        assert "fill_prob_head.weight" not in legacy_state
        assert "fill_prob_head.bias" not in legacy_state

        # Strict load should fail on a legacy dict…
        target = PPOLSTMPolicy(
            obs_dim=32, action_dim=14 * 6, max_runners=14, hyperparams=hp,
        )
        with pytest.raises(RuntimeError):
            target.load_state_dict(legacy_state, strict=True)

        # …but succeeds after migration.
        migrated = migrate_fill_prob_head(legacy_state, target)
        missing, unexpected = target.load_state_dict(migrated, strict=True)
        assert missing == []
        assert unexpected == []

        # Fresh-init contract: weight non-zero (orthogonal gain=0.01),
        # bias zero.
        assert target.fill_prob_head.weight.abs().sum().item() > 0.0
        assert torch.allclose(
            target.fill_prob_head.bias,
            torch.zeros_like(target.fill_prob_head.bias),
        )

    # ── 11. raw + shaped invariant holds with aux weight > 0 on env. ────

    def test_raw_plus_shaped_invariant_still_holds_with_aux_loss(
        self, scalping_config,
    ):
        """``fill_prob_loss_weight`` is a TRAINER knob, not an env reward
        knob. Whitelisting it in ``_REWARD_OVERRIDE_KEYS`` must not change
        the env's reward accumulators, so the raw+shaped invariant holds
        unchanged when the key flows through ``reward_overrides``.
        """
        cfg = dict(scalping_config)
        cfg["reward"] = dict(cfg["reward"])
        env = BetfairEnv(
            _make_day(n_races=1, n_pre_ticks=5, n_inplay_ticks=2), cfg,
            reward_overrides={"fill_prob_loss_weight": 0.5},
        )
        env.reset()

        a = _scalping_action(
            signal=1.0, stake=-0.8, aggression=1.0,
            arb_spread=0.2, requote=1.0,
        )
        total_reward = 0.0
        terminated = False
        info: dict = {}
        while not terminated:
            _, r, terminated, _, info = env.step(a)
            total_reward += r

        assert total_reward == pytest.approx(
            info["raw_pnl_reward"] + info["shaped_bonus"], abs=1e-6,
        )

    # ── 12. Unresolved (NaN) samples don't contribute to the BCE loss. ──

    def test_fill_prob_excluded_from_loss_when_outcome_unresolved(self):
        import torch
        from agents.ppo_trainer import _compute_fill_prob_bce

        # Two slots: slot 0 has a resolved label; slot 1 is NaN
        # (unresolved). The loss must depend only on slot 0 — mutating
        # slot 1's prediction must leave the value unchanged.
        preds_a = torch.tensor([[0.3, 0.4]], dtype=torch.float32)
        preds_b = torch.tensor([[0.3, 0.9]], dtype=torch.float32)  # slot 1 moved
        labels = torch.tensor(
            [[1.0, float("nan")]], dtype=torch.float32,
        )

        loss_a = _compute_fill_prob_bce(preds_a, labels)
        loss_b = _compute_fill_prob_bce(preds_b, labels)
        assert loss_a.item() == pytest.approx(loss_b.item(), abs=1e-7)


# ── Scalping-active-management session 03: risk / predicted-variance head ──


def _run_ppo_with_risk_and_measure(
    scalping_config: dict,
    *,
    fill_prob_weight: float,
    risk_weight: float,
    assert_eq: bool,
) -> float:
    """Run one PPO update with both aux weights set; return grad norm.

    Shared by ``TestRiskHead`` tests 6 (``risk_weight=0`` noop) and 7
    (``risk_weight=1`` grad-norm lift). Mirrors ``_run_ppo_and_measure``
    but populates a non-NaN ``risk_labels`` on some transitions so the
    NLL term has gradient when ``risk_weight > 0``.
    """
    import numpy as np
    import torch
    from agents.architecture_registry import create_policy
    from agents.ppo_trainer import PPOTrainer, Rollout, Transition

    cfg = dict(scalping_config)
    hp = {
        "lstm_hidden_size": 16, "mlp_hidden_size": 8, "mlp_layers": 1,
        "learning_rate": 1e-4, "gamma": 0.99, "gae_lambda": 0.95,
        "ppo_clip_epsilon": 0.2, "entropy_coefficient": 0.01,
        "value_loss_coeff": 0.5, "max_grad_norm": 1e9,
        "ppo_epochs": 1, "mini_batch_size": 4,
        "fill_prob_loss_weight": fill_prob_weight,
        "risk_loss_weight": risk_weight,
    }
    env_probe = BetfairEnv(_make_day(n_races=1), cfg)
    obs_dim = int(env_probe.observation_space.shape[0])
    action_dim = int(env_probe.action_space.shape[0])
    max_runners = env_probe.max_runners

    torch.manual_seed(0)
    np.random.seed(0)
    policy = create_policy(
        name="ppo_lstm_v1",
        obs_dim=obs_dim, action_dim=action_dim,
        max_runners=max_runners, hyperparams=hp,
    )
    trainer = PPOTrainer(policy=policy, config=cfg, hyperparams=hp)

    rng = np.random.default_rng(0)
    rollout = Rollout()
    for i in range(8):
        obs_i = rng.standard_normal(obs_dim).astype(np.float32)
        action_i = rng.standard_normal(action_dim).astype(np.float32)
        fp_labels = np.full(max_runners, np.nan, dtype=np.float32)
        risk_labels = np.full(max_runners, np.nan, dtype=np.float32)
        if i % 2 == 0:
            # Resolved-pair slot: fill-prob completed + realised locked_pnl.
            fp_labels[0] = 1.0 if i % 4 == 0 else 0.0
            risk_labels[0] = float(rng.uniform(-5.0, 5.0))
        rollout.append(Transition(
            obs=obs_i, action=action_i,
            log_prob=-1.0, value=0.0,
            reward=float(rng.standard_normal()),
            done=(i == 7),
            training_reward=0.0,
            fill_prob_labels=fp_labels,
            risk_labels=risk_labels,
        ))

    loss_info = trainer._ppo_update(rollout)

    grad_sq_sum = 0.0
    for p in policy.parameters():
        if p.grad is not None:
            grad_sq_sum += float(p.grad.detach().pow(2).sum().item())
    grad_norm = grad_sq_sum ** 0.5

    if assert_eq:
        pl = loss_info["policy_loss"]
        vl = loss_info["value_loss"]
        ent = loss_info["entropy"]
        fp = loss_info["fill_prob_loss"]
        risk = loss_info["risk_loss"]
        expected = (
            pl + trainer.value_loss_coeff * vl - trainer.entropy_coeff * ent
            + trainer.fill_prob_loss_weight * fp
        )
        total_with_risk = expected + trainer.risk_loss_weight * risk
        assert total_with_risk == pytest.approx(expected, abs=1e-7)

    return grad_norm


class TestRiskHead:
    """Risk / predicted-variance auxiliary head (scalping-active-management §03).

    Plumbing-off by default (``risk_loss_weight=0.0``). Each test pins one
    invariant: shape / clamp band, decision-time capture, passive-leg
    inheritance, NLL on perfect predictions, NLL gradient direction,
    weight=0 noop, weight>0 grad-norm lift, parquet back-compat + roundtrip,
    checkpoint back-compat, raw+shaped reward invariant, unresolved-sample
    exclusion, and log-var clamp enforcement inside ``forward``.
    """

    @staticmethod
    def _small_hp() -> dict:
        return {
            "lstm_hidden_size": 32,
            "mlp_hidden_size": 16,
            "mlp_layers": 1,
            "transformer_heads": 2,
            "transformer_depth": 1,
            "transformer_ctx_ticks": 8,
        }

    # ── 1. Shape + clamp-band range across all 3 architectures. ─────────

    def test_risk_output_shape_and_range(self, scalping_config):
        import torch
        from agents.policy_network import (
            PPOLSTMPolicy,
            PPOTimeLSTMPolicy,
            PPOTransformerPolicy,
            RISK_LOG_VAR_MAX,
            RISK_LOG_VAR_MIN,
        )

        env = BetfairEnv(_make_day(n_races=1), scalping_config)
        obs_dim = int(env.observation_space.shape[0])
        max_runners = env.max_runners
        action_dim = int(env.action_space.shape[0])
        hp = self._small_hp()

        for cls in (PPOLSTMPolicy, PPOTimeLSTMPolicy, PPOTransformerPolicy):
            net = cls(
                obs_dim=obs_dim, action_dim=action_dim,
                max_runners=max_runners, hyperparams=hp,
            )
            obs = torch.zeros(3, obs_dim, dtype=torch.float32)
            out = net(obs)
            assert out.predicted_locked_pnl_per_runner.shape == (
                3, max_runners,
            ), f"{cls.__name__}: bad mean shape"
            assert out.predicted_locked_log_var_per_runner.shape == (
                3, max_runners,
            ), f"{cls.__name__}: bad log-var shape"
            lv = out.predicted_locked_log_var_per_runner
            assert torch.all(lv >= RISK_LOG_VAR_MIN), (
                f"{cls.__name__}: log_var below clamp min"
            )
            assert torch.all(lv <= RISK_LOG_VAR_MAX), (
                f"{cls.__name__}: log_var above clamp max"
            )

    # ── 2. Decision-time capture stamps risk fields on the Bet. ─────────

    def test_risk_recorded_on_bet_at_placement(self, scalping_config):
        import numpy as np
        import torch
        from agents.architecture_registry import create_policy

        cfg = dict(scalping_config)
        hp = self._small_hp()
        day = _make_day(n_races=1, n_pre_ticks=3, n_inplay_ticks=1)
        env = BetfairEnv(day, cfg)
        obs, _ = env.reset()
        policy = create_policy(
            name="ppo_lstm_v1",
            obs_dim=int(env.observation_space.shape[0]),
            action_dim=int(env.action_space.shape[0]),
            max_runners=env.max_runners,
            hyperparams=hp,
        )

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = policy(obs_tensor)
        risk_mean = (
            out.predicted_locked_pnl_per_runner.detach()
            .cpu().numpy().reshape(-1)
        )
        risk_stddev = np.exp(
            0.5 * out.predicted_locked_log_var_per_runner.detach()
            .cpu().numpy().reshape(-1)
        )
        fp = out.fill_prob_per_runner.detach().cpu().numpy().reshape(-1)

        a = _scalping_action(signal=1.0, stake=-0.8, aggression=1.0)
        _, _, _, _, info = env.step(a)

        sid_to_slot = env.current_runner_to_slot()
        bm = env.bet_manager
        for sid, entry in info.get("action_debug", {}).items():
            if not entry.get("aggressive_placed", False):
                continue
            slot_idx = sid_to_slot.get(sid)
            if slot_idx is None:
                continue
            fp_val = float(fp[slot_idx])
            risk_mean_val = float(risk_mean[slot_idx])
            risk_stddev_val = float(risk_stddev[slot_idx])
            for bet in reversed(bm.bets):
                if (
                    bet.selection_id == sid
                    and bet.fill_prob_at_placement is None
                ):
                    bet.fill_prob_at_placement = fp_val
                    bet.predicted_locked_pnl_at_placement = risk_mean_val
                    bet.predicted_locked_stddev_at_placement = risk_stddev_val
                    break

        stamped = [
            b for b in bm.bets
            if b.predicted_locked_pnl_at_placement is not None
        ]
        assert stamped, "expected at least one Bet with risk fields stamped"
        for b in stamped:
            # Mean is any float; stddev must be strictly positive.
            assert isinstance(b.predicted_locked_pnl_at_placement, float)
            assert b.predicted_locked_stddev_at_placement is not None
            assert b.predicted_locked_stddev_at_placement > 0.0

    # ── 3. Paired passive inherits risk fields from the aggressive. ─────

    def test_risk_inherited_by_paired_passive(self):
        from env.bet_manager import BetManager, BetSide, PassiveOrder

        bm = BetManager(starting_budget=1000.0)
        agg = bm.place_back(
            _make_runner_snap(101, ltp=4.0, back_price=4.0,
                              lay_price=4.1, size=200.0),
            stake=10.0, market_id="m1", pair_id="pair-risk",
        )
        assert agg is not None
        agg.fill_prob_at_placement = 0.62
        agg.predicted_locked_pnl_at_placement = 1.25
        agg.predicted_locked_stddev_at_placement = 0.8

        order = PassiveOrder(
            selection_id=101,
            side=BetSide.LAY,
            price=4.1,
            requested_stake=10.0,
            queue_ahead_at_placement=0.0,
            placed_tick_index=0,
            market_id="m1",
            ltp_at_placement=4.0,
            pair_id="pair-risk",
            reserved_liability=0.0,
        )
        book = bm.passive_book
        book._orders.append(order)
        book._orders_by_sid.setdefault(101, []).append(order)

        class _Runner:
            selection_id = 101
            total_matched = 1000.0
            available_to_back: list = []
            available_to_lay: list = []
            last_traded_price = 4.0
            status = "ACTIVE"

        class _Tick:
            runners = [_Runner()]

        book.on_tick(_Tick(), tick_index=1)  # type: ignore[arg-type]

        passive_bets = [
            b for b in bm.bets
            if b.side == BetSide.LAY and b.pair_id == "pair-risk"
        ]
        assert len(passive_bets) == 1
        p = passive_bets[0]
        assert p.predicted_locked_pnl_at_placement == pytest.approx(1.25)
        assert p.predicted_locked_stddev_at_placement == pytest.approx(0.8)

    # ── 4. NLL analytic value on perfect predictions + clamp-min log_var.

    def test_risk_nll_zero_on_perfect_predictions(self):
        import torch
        from agents.policy_network import RISK_LOG_VAR_MIN
        from agents.ppo_trainer import _compute_risk_nll

        means = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
        log_vars = torch.full(
            (1, 2), float(RISK_LOG_VAR_MIN), dtype=torch.float32,
        )
        labels = torch.tensor([[1.0, -2.0]], dtype=torch.float32)

        loss = _compute_risk_nll(means, log_vars, labels)
        # NLL per element = 0.5 * (log_var + 0 / exp(log_var))
        #                 = 0.5 * RISK_LOG_VAR_MIN
        expected = 0.5 * RISK_LOG_VAR_MIN
        assert loss.item() == pytest.approx(expected, abs=1e-5)

    # ── 5. NLL gradient direction for a contrived bad prediction. ───────

    def test_risk_nll_gradient_direction(self):
        import torch
        from agents.ppo_trainer import _compute_risk_nll

        # mean=5.0, target=0.0, log_var=0.0. NLL wrt mean should be
        # positive (push mean DOWN); wrt log_var should be negative
        # (widen variance to explain the 5.0 residual).
        mean = torch.tensor([[5.0]], requires_grad=True)
        log_var = torch.tensor([[0.0]], requires_grad=True)
        label = torch.tensor([[0.0]], dtype=torch.float32)

        loss = _compute_risk_nll(mean, log_var, label)
        loss.backward()
        assert mean.grad is not None and mean.grad.item() > 0.0
        assert log_var.grad is not None and log_var.grad.item() < 0.0

    # ── 6. weight=0 makes total loss byte-identical to session 02. ──────

    def test_risk_weight_zero_is_noop_on_total_loss(self, scalping_config):
        _run_ppo_with_risk_and_measure(
            scalping_config,
            fill_prob_weight=0.0, risk_weight=0.0, assert_eq=True,
        )

    # ── 7. weight=1 lifts the total-param gradient norm. ────────────────

    def test_risk_weight_positive_changes_gradient_norm(self, scalping_config):
        gn_off = _run_ppo_with_risk_and_measure(
            scalping_config,
            fill_prob_weight=0.0, risk_weight=0.0, assert_eq=False,
        )
        gn_on = _run_ppo_with_risk_and_measure(
            scalping_config,
            fill_prob_weight=0.0, risk_weight=1.0, assert_eq=False,
        )
        assert gn_on > gn_off, (
            f"risk weight=1 did not lift grad norm "
            f"(off={gn_off:.6f}, on={gn_on:.6f})"
        )

    # ── 8. Parquet reader tolerates absence of the new columns. ─────────

    def test_risk_parquet_backcompat_missing_columns(self, tmp_path):
        import pandas as pd
        from registry.model_store import EvaluationBetRecord, ModelStore

        store = ModelStore(
            db_path=tmp_path / "m.db",
            weights_dir=tmp_path / "w",
            bet_logs_dir=tmp_path / "b",
        )
        rec = EvaluationBetRecord(
            run_id="r1", date="2026-04-17", market_id="m1",
            tick_timestamp="t", seconds_to_off=10.0, runner_id=101,
            runner_name="A", action="back", price=4.0, stake=10.0,
            matched_size=10.0, outcome="won", pnl=3.0,
            predicted_locked_pnl_at_placement=2.5,
            predicted_locked_stddev_at_placement=1.5,
        )
        path = store.write_bet_logs_parquet("r1", "2026-04-17", [rec])
        assert path is not None and path.exists()

        # Strip both risk columns → simulates a pre-Session-03 file.
        df = pd.read_parquet(path)
        df = df.drop(columns=[
            "predicted_locked_pnl_at_placement",
            "predicted_locked_stddev_at_placement",
        ])
        df.to_parquet(path, index=False)

        loaded = store.get_evaluation_bets("r1")
        assert len(loaded) == 1
        assert loaded[0].predicted_locked_pnl_at_placement is None
        assert loaded[0].predicted_locked_stddev_at_placement is None

    # ── 9. Parquet roundtrip preserves the new columns. ─────────────────

    def test_risk_parquet_roundtrip(self, tmp_path):
        from registry.model_store import EvaluationBetRecord, ModelStore

        store = ModelStore(
            db_path=tmp_path / "m.db",
            weights_dir=tmp_path / "w",
            bet_logs_dir=tmp_path / "b",
        )
        rec = EvaluationBetRecord(
            run_id="r2", date="2026-04-17", market_id="m1",
            tick_timestamp="t", seconds_to_off=10.0, runner_id=101,
            runner_name="A", action="back", price=4.0, stake=10.0,
            matched_size=10.0, outcome="won", pnl=3.0,
            predicted_locked_pnl_at_placement=1.23,
            predicted_locked_stddev_at_placement=2.5,
        )
        store.write_bet_logs_parquet("r2", "2026-04-17", [rec])
        loaded = store.get_evaluation_bets("r2")
        assert len(loaded) == 1
        assert loaded[0].predicted_locked_pnl_at_placement == pytest.approx(
            1.23,
        )
        assert loaded[0].predicted_locked_stddev_at_placement == pytest.approx(
            2.5,
        )

    # ── 10. Legacy state-dict loads via migrate_risk_head. ──────────────

    def test_legacy_checkpoint_loads_with_risk_head(self):
        import torch
        from agents.policy_network import (
            PPOLSTMPolicy,
            migrate_risk_head,
        )

        hp = self._small_hp()
        new_net = PPOLSTMPolicy(
            obs_dim=32, action_dim=14 * 6, max_runners=14, hyperparams=hp,
        )

        # Pre-Session-03 state-dict → drop only the risk_head.* keys.
        # The fill_prob_head.* keys are present (Session 02 landed them).
        legacy_state = {
            k: v.clone() for k, v in new_net.state_dict().items()
            if not k.startswith("risk_head.")
        }
        assert "risk_head.weight" not in legacy_state
        assert "risk_head.bias" not in legacy_state

        target = PPOLSTMPolicy(
            obs_dim=32, action_dim=14 * 6, max_runners=14, hyperparams=hp,
        )
        with pytest.raises(RuntimeError):
            target.load_state_dict(legacy_state, strict=True)

        migrated = migrate_risk_head(legacy_state, target)
        missing, unexpected = target.load_state_dict(migrated, strict=True)
        assert missing == []
        assert unexpected == []

        # Fresh-init contract: weight non-zero (orthogonal gain=0.01),
        # bias zero.
        assert target.risk_head.weight.abs().sum().item() > 0.0
        assert torch.allclose(
            target.risk_head.bias,
            torch.zeros_like(target.risk_head.bias),
        )

    # ── 11. raw + shaped invariant holds with risk weight > 0 on env. ───

    def test_raw_plus_shaped_invariant_still_holds_with_risk_loss(
        self, scalping_config,
    ):
        """``risk_loss_weight`` is a TRAINER knob; env whitelisting must
        leave the reward accumulators alone. Same invariant as the
        Session-02 equivalent, with both aux weights threaded through
        ``reward_overrides``.
        """
        cfg = dict(scalping_config)
        cfg["reward"] = dict(cfg["reward"])
        env = BetfairEnv(
            _make_day(n_races=1, n_pre_ticks=5, n_inplay_ticks=2), cfg,
            reward_overrides={
                "fill_prob_loss_weight": 0.5,
                "risk_loss_weight": 0.5,
            },
        )
        env.reset()

        a = _scalping_action(
            signal=1.0, stake=-0.8, aggression=1.0,
            arb_spread=0.2, requote=1.0,
        )
        total_reward = 0.0
        terminated = False
        info: dict = {}
        while not terminated:
            _, r, terminated, _, info = env.step(a)
            total_reward += r

        assert total_reward == pytest.approx(
            info["raw_pnl_reward"] + info["shaped_bonus"], abs=1e-6,
        )

    # ── 12. Unresolved (NaN) samples don't contribute to the NLL. ───────

    def test_risk_excluded_from_loss_when_outcome_unresolved(self):
        import torch
        from agents.ppo_trainer import _compute_risk_nll

        means_a = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        means_b = torch.tensor([[1.0, 9.0]], dtype=torch.float32)  # slot 1 moved
        log_vars = torch.zeros((1, 2), dtype=torch.float32)
        labels = torch.tensor(
            [[0.5, float("nan")]], dtype=torch.float32,
        )

        loss_a = _compute_risk_nll(means_a, log_vars, labels)
        loss_b = _compute_risk_nll(means_b, log_vars, labels)
        assert loss_a.item() == pytest.approx(loss_b.item(), abs=1e-7)

    # ── 13. Log-var clamping happens inside forward. ────────────────────

    def test_log_var_clamped_in_forward(self, scalping_config):
        """Drive the risk head to an extreme raw output and assert the
        tensor reaching ``PolicyOutput`` is strictly within the clamp
        band regardless. Without this, ``exp(log_var)`` overflows in
        the NLL and poisons the optimiser.
        """
        import torch
        from agents.policy_network import (
            PPOLSTMPolicy,
            RISK_LOG_VAR_MAX,
            RISK_LOG_VAR_MIN,
        )

        env = BetfairEnv(_make_day(n_races=1), scalping_config)
        obs_dim = int(env.observation_space.shape[0])
        max_runners = env.max_runners
        action_dim = int(env.action_space.shape[0])
        hp = self._small_hp()

        net = PPOLSTMPolicy(
            obs_dim=obs_dim, action_dim=action_dim,
            max_runners=max_runners, hyperparams=hp,
        )
        # Overwrite the risk head so its raw output is far outside the
        # clamp band on both ends, regardless of input.
        with torch.no_grad():
            net.risk_head.weight.zero_()
            # Alternate very-large-positive and very-large-negative biases
            # so some slots would otherwise exceed MAX, others fall below
            # MIN — one test covers both clamp edges.
            big_bias = torch.tensor(
                [100.0 if i % 2 == 0 else -100.0
                 for i in range(max_runners * 2)],
                dtype=net.risk_head.bias.dtype,
            )
            net.risk_head.bias.copy_(big_bias)

        obs = torch.zeros(2, obs_dim, dtype=torch.float32)
        out = net(obs)
        lv = out.predicted_locked_log_var_per_runner
        assert torch.all(lv >= RISK_LOG_VAR_MIN)
        assert torch.all(lv <= RISK_LOG_VAR_MAX)


# ── Equal-profit sizing end-to-end (plans/scalping-equal-profit-sizing) ─────


class TestEqualProfitSizingEndToEnd:
    """Sizing helper is correctly wired into the three placement paths
    (``_maybe_place_paired``, ``_attempt_close``, ``_attempt_requote``).

    Hard_constraints §16 items 6–8 + the canonical worked example from
    ``purpose.md`` (Back £16 @ 8.20 / Lay @ 6.00 / c=5% → locked ≈ £4.03).
    """

    def _make_env(self, scalping_config, **kwargs) -> BetfairEnv:
        return BetfairEnv(
            _make_day(n_races=1, n_pre_ticks=5, n_inplay_ticks=2),
            scalping_config,
            **kwargs,
        )

    def _place_initial_pair(self, env: BetfairEnv) -> tuple:
        """Drive env through one tick producing an aggressive back + paired lay."""
        a = _scalping_action(
            signal=1.0, stake=-0.8, aggression=1.0, arb_spread=-1.0,
        )
        env.step(a)
        bm = env.bet_manager
        paired_bets = [b for b in bm.bets if b.pair_id is not None]
        assert paired_bets, "expected an aggressive fill with a pair_id"
        agg = paired_bets[0]
        pairing = [
            o for o in bm.passive_book.orders if o.pair_id == agg.pair_id
        ]
        assert pairing, "expected a paired passive to rest after the fill"
        # Fence against on_tick auto-fill so subsequent steps don't drain
        # the resting passive before close/requote tests exercise it.
        for o in bm.passive_book.orders:
            if o.pair_id is not None:
                o.queue_ahead_at_placement = 1e12
        return agg, pairing[0]

    def test_paired_passive_stake_uses_equal_profit_formula(
        self, scalping_config,
    ):
        """Placement of the auto-paired passive uses ``equal_profit_lay_stake``
        (aggressive BACK) for back-first scalps. Formula:

            S_lay = S_back × [P_back × (1 − c) + c] / (P_lay − c)
        """
        from env.scalping_math import equal_profit_lay_stake

        env = self._make_env(scalping_config)
        env.reset()
        agg, passive = self._place_initial_pair(env)
        assert agg.side is BetSide.BACK
        assert passive.side is BetSide.LAY

        c = scalping_config["reward"].get("commission", 0.05)
        expected = equal_profit_lay_stake(
            back_stake=agg.matched_stake,
            back_price=agg.average_price,
            lay_price=passive.price,
            commission=c,
        )
        assert passive.requested_stake == pytest.approx(expected, rel=1e-6)

    def test_close_leg_stake_uses_equal_profit_formula(
        self, scalping_config,
    ):
        """The close mechanic sizes its closing leg via the same helper.

        ``_attempt_close`` calls ``bm.place_back`` / ``bm.place_lay`` with
        the helper's output. The returned ``close_bet.requested_stake``
        must equal the helper's value for the aggressive leg's price / the
        close-side best price.
        """
        from env.scalping_math import equal_profit_lay_stake

        env = self._make_env(scalping_config)
        env.reset()
        agg, _ = self._place_initial_pair(env)
        bm = env.bet_manager

        # Fire the close signal on slot 0 (close index = 6 per scalping v4).
        a = _scalping_action(signal=0.0, stake=-1.0)
        a[6 * 14 + 0] = 1.0
        env.step(a)

        pair_bets = [b for b in bm.bets if b.pair_id == agg.pair_id]
        close_bet = next(b for b in pair_bets if getattr(b, "close_leg", False))
        # Close is LAY for an aggressive BACK; equal_profit_lay_stake is
        # the right helper.
        c = scalping_config["reward"].get("commission", 0.05)
        expected = equal_profit_lay_stake(
            back_stake=agg.matched_stake,
            back_price=agg.average_price,
            lay_price=close_bet.average_price,
            commission=c,
        )
        assert close_bet.requested_stake == pytest.approx(expected, rel=1e-6)

    def test_requote_resizes_at_new_lay_price(self, scalping_config):
        """Re-quote re-sizes the passive at the new lay price — it does
        NOT carry the old stake forward (hard_constraints §8 third bullet).
        """
        from env.scalping_math import equal_profit_lay_stake

        env = self._make_env(scalping_config)
        env.reset()
        agg, old_passive = self._place_initial_pair(env)
        bm = env.bet_manager
        old_price = old_passive.price
        old_stake = old_passive.requested_stake

        # Re-quote at MAX arb offset so the new price differs meaningfully.
        a = _scalping_action(requote=1.0, arb_spread=1.0)
        env.step(a)

        new_passive = next(
            o for o in bm.passive_book.orders if o.pair_id == agg.pair_id
        )
        assert new_passive.price != old_price, (
            "test setup failed: re-quote must move price to exercise re-sizing"
        )
        # Old behaviour would have kept the stake identical across prices.
        assert new_passive.requested_stake != pytest.approx(old_stake, rel=1e-9)
        # New stake must match the helper's output at the NEW price.
        c = scalping_config["reward"].get("commission", 0.05)
        expected = equal_profit_lay_stake(
            back_stake=agg.matched_stake,
            back_price=agg.average_price,
            lay_price=new_passive.price,
            commission=c,
        )
        assert new_passive.requested_stake == pytest.approx(expected, rel=1e-6)

    def test_canonical_worked_example_locks_4_03(self, scalping_config):
        """Worked example from ``purpose.md``: Back £16 @ 8.20, Lay @ 6.00,
        c=5% locks ≈ £4.03 (not the £0.08 the old sizing produced).

        Synthesises a completed pair by overriding the aggressive leg's
        stake/price and appending a matching passive Bet at the canonical
        lay price with the helper-sized stake. ``get_paired_positions``
        then computes the real ``locked_pnl`` via the existing
        ``min(win, lose)`` formula on the matched legs — we assert the
        result matches the purpose.md number.
        """
        from env.bet_manager import Bet
        from env.scalping_math import equal_profit_lay_stake

        env = self._make_env(scalping_config)
        env.reset()
        agg, resting = self._place_initial_pair(env)
        bm = env.bet_manager

        # Override aggressive leg to the canonical BACK £16 @ 8.20.
        agg.matched_stake = 16.0
        agg.average_price = 8.20

        # Size the passive lay via the helper at the canonical 6.00 / 5%.
        c = 0.05
        lay_stake = equal_profit_lay_stake(
            back_stake=16.0, back_price=8.20, lay_price=6.00, commission=c,
        )
        assert lay_stake == pytest.approx(21.0823529, rel=1e-6)

        # Synthesise the passive fill at 6.00 and flush the originally-
        # rested order so settlement doesn't double-count.
        bm.bets.append(Bet(
            selection_id=resting.selection_id,
            side=resting.side,
            requested_stake=lay_stake,
            matched_stake=lay_stake,
            average_price=6.00,
            market_id=resting.market_id,
            ltp_at_placement=resting.ltp_at_placement,
            pair_id=resting.pair_id,
            tick_index=env._tick_idx,
        ))
        bm.passive_book._orders = [
            o for o in bm.passive_book._orders if id(o) != id(resting)
        ]
        for sid_orders in bm.passive_book._orders_by_sid.values():
            sid_orders[:] = [o for o in sid_orders if id(o) != id(resting)]

        pairs = bm.get_paired_positions(
            market_id=agg.market_id, commission=c,
        )
        assert pairs and pairs[0]["complete"]
        # purpose.md: £4.03 (old sizing gave £0.08).
        assert pairs[0]["locked_pnl"] == pytest.approx(4.03, abs=0.01)

        # Cross-check by computing min(win, lose) directly from the
        # matched stakes / prices.
        S_b, P_b, S_l, P_l = 16.0, 8.20, lay_stake, 6.00
        win_pnl = S_b * (P_b - 1.0) * (1.0 - c) - S_l * (P_l - 1.0)
        lose_pnl = -S_b + S_l * (1.0 - c)
        assert abs(win_pnl - lose_pnl) < 0.01  # equal-profit invariant
        assert min(win_pnl, lose_pnl) == pytest.approx(4.03, abs=0.01)
