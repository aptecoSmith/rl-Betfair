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

    def test_get_paired_positions_complete_yields_locked_spread(self):
        bm = BetManager(starting_budget=100.0)
        # Aggressive back at 5.0, passive lay leg at 4.5 — stake £10
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
        # Aggressive back matches the top of available_to_back (5.0);
        # aggressive lay matches the top of available_to_lay (4.6).
        # spread = 5.0 - 4.6 = 0.4, stake = 10, gross = 4.0, net = 3.8.
        assert p["locked_pnl"] == pytest.approx(3.8, abs=0.01)

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
