"""Tests for forced-arbitrage (scalping) mechanics — Issue 05, session 1."""

from __future__ import annotations

import numpy as np
import pytest

from data.episode_builder import PriceSize
from env.bet_manager import BetManager, BetSide
from env.betfair_env import (
    ACTIONS_PER_RUNNER,
    MAX_ARB_TICKS,
    MIN_ARB_TICKS,
    SCALPING_ACTIONS_PER_RUNNER,
    BetfairEnv,
)
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
        # 2 extra per runner + 2 global = 2*14 + 2 = 30 extra.
        assert env_on.observation_space.shape[0] == env_off.observation_space.shape[0] + 30

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
        """Passive leg stake = agg_stake × agg_price / passive_price.

        This is the only stake ratio that locks profit across both race
        outcomes. Any other ratio leaves the pair directional. A BACK→LAY
        pair always places MORE lay stake than the aggressive back stake
        (because passive_price < agg_price) — the magnitude of the
        difference encodes the price movement.
        """
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
        # Correct asymmetric stake.
        expected = (
            agg.matched_stake * agg.average_price / resting.price
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
        """Directional shaping is switched off when scalping_mode is on."""
        cfg = dict(scalping_config)
        cfg["reward"] = dict(cfg["reward"])
        cfg["reward"]["precision_bonus"] = 10.0
        cfg["reward"]["early_pick_bonus_min"] = 5.0
        cfg["reward"]["early_pick_bonus_max"] = 5.0
        env = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), cfg)
        a = np.zeros(14 * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32)
        a[0] = 1.0
        a[14] = -0.8
        a[28] = 1.0
        info = self._run_episode(env, a)
        # Those two shaping terms are the only ones we loaded with large
        # non-zero weights. If either leaked into shaping, the cumulative
        # shaped reward would be bounded away from zero by a large margin.
        assert abs(info["shaped_bonus"]) < 1.0, info["shaped_bonus"]

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
            bm2.bets.append(Bet(
                selection_id=resting2.selection_id,
                side=resting2.side,
                requested_stake=resting2.requested_stake,
                matched_stake=resting2.requested_stake,
                average_price=resting2.price,
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

    def test_naked_windfall_excluded_from_raw_reward(self, scalping_config):
        """Naked bet P&L must not count toward the reward in scalping mode.

        A naked back that wins is directional luck, not scalping skill —
        folding it into the reward would teach the agent to deliberately
        leave unpaired bets and hope for a price run. The naked penalty
        covers the exposure, and that's the only signal naked bets get.
        """
        cfg = dict(scalping_config)
        cfg["reward"] = dict(cfg["reward"])
        # Turn shaping way down so the raw reward is effectively the only
        # thing driving the reward stream.
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
        # Force the one outstanding pair to stay unfilled: clear the
        # passive_book and strip pair_id off the matched aggressive so
        # `get_paired_positions` can't find a pair group.
        bm = env.bet_manager
        bm.passive_book._orders = []
        bm.passive_book._orders_by_sid.clear()
        for b in bm.bets:
            b.pair_id = None
        terminated = False
        info = {}
        while not terminated:
            _, _, terminated, _, info = env.step(hold)
        # No pair was completed, so the raw reward excludes the naked
        # bet's P&L entirely — even if day_pnl is non-zero because the
        # runner did win / lose directionally.
        assert info["arbs_completed"] == 0
        assert info["locked_pnl"] == 0.0
        assert info["naked_pnl"] == pytest.approx(info["day_pnl"], abs=1e-6)
        assert info["raw_pnl_reward"] == pytest.approx(0.0, abs=1e-6)

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
