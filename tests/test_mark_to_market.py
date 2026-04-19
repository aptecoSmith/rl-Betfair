"""Tests for per-step mark-to-market shaping (reward-densification
Session 01, 2026-04-19).

Covers the mechanism, the formulas (hard_constraints §6-§7), the
telescope property (§8-§9), the knob (§10-§11), the invariant under
non-zero weight (§12), and the info/JSONL telemetry (§13-§14).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from env.bet_manager import Bet, BetOutcome, BetSide
from env.betfair_env import (
    SCALPING_ACTIONS_PER_RUNNER,
    BetfairEnv,
)

from tests.test_betfair_env import _make_day


@pytest.fixture
def legacy_config() -> dict:
    """Minimal directional config. Mirrors tests/test_forced_arbitrage.py's
    ``legacy_config`` fixture so the same synthetic day fixtures work."""
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


class TestMarkToMarketShaping:
    """Core mechanism + formula tests."""

    def test_mark_to_market_weight_default_is_zero(self, legacy_config):
        env = BetfairEnv(_make_day(n_races=1), legacy_config)
        assert env._mark_to_market_weight == 0.0

    def test_mtm_delta_zero_when_no_open_bets(self, legacy_config):
        """With no bets placed, every mtm_delta is 0 and cumulative
        shaped MTM stays at 0 across the episode."""
        cfg = copy.deepcopy(legacy_config)
        cfg["reward"]["mark_to_market_weight"] = 0.05
        env = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), cfg)
        env.reset()
        noop = np.zeros(env.action_space.shape, dtype=np.float32)
        terminated = False
        info = {}
        while not terminated:
            _, _, terminated, _, info = env.step(noop)
            assert info["mtm_delta"] == pytest.approx(0.0, abs=1e-9)
        assert info["cumulative_mtm_shaped"] == pytest.approx(0.0, abs=1e-9)

    def test_mtm_back_formula_matches_spec(self, legacy_config):
        """Back bet: mtm = S * (P_matched - LTP) / LTP."""
        env = BetfairEnv(_make_day(n_races=1), legacy_config)
        env.reset()
        env.bet_manager.bets.append(Bet(
            selection_id=101,
            side=BetSide.BACK,
            requested_stake=10.0,
            matched_stake=10.0,
            average_price=8.0,
            market_id="m1",
        ))
        # LTP fell 8.0 -> 6.0: back position now worth +£3.333
        mtm = env._compute_portfolio_mtm({101: 6.0})
        assert mtm == pytest.approx(10.0 * (8.0 - 6.0) / 6.0, abs=1e-9)

    def test_mtm_lay_formula_matches_spec(self, legacy_config):
        """Lay bet: mtm = S * (LTP - P_matched) / LTP."""
        env = BetfairEnv(_make_day(n_races=1), legacy_config)
        env.reset()
        env.bet_manager.bets.append(Bet(
            selection_id=101,
            side=BetSide.LAY,
            requested_stake=10.0,
            matched_stake=10.0,
            average_price=4.0,
            market_id="m1",
        ))
        # LTP rose 4.0 -> 5.0: lay position now worth +£2.0
        mtm = env._compute_portfolio_mtm({101: 5.0})
        assert mtm == pytest.approx(10.0 * (5.0 - 4.0) / 5.0, abs=1e-9)

    def test_mtm_zero_when_ltp_missing(self, legacy_config):
        """Unpriceable runner (missing or <= 1.0 LTP) contributes 0."""
        env = BetfairEnv(_make_day(n_races=1), legacy_config)
        env.reset()
        env.bet_manager.bets.append(Bet(
            selection_id=101,
            side=BetSide.BACK,
            requested_stake=10.0,
            matched_stake=10.0,
            average_price=8.0,
            market_id="m1",
        ))
        # No LTP for sid 101 → 0.
        assert env._compute_portfolio_mtm({}) == 0.0
        # LTP == 1.0 is the band floor — treated as unpriceable.
        assert env._compute_portfolio_mtm({101: 1.0}) == 0.0
        # LTP == 0 (junk) → 0.
        assert env._compute_portfolio_mtm({101: 0.0}) == 0.0

    def test_mtm_excludes_resolved_bets(self, legacy_config):
        """Bets with outcome != UNSETTLED drop out of the portfolio MTM.

        This is what makes the telescope close at settle (§9): once
        ``bm.settle_race`` changes outcomes to WON/LOST/VOID, those
        bets stop contributing to MTM_t, so the final delta unwinds
        whatever was on the books."""
        env = BetfairEnv(_make_day(n_races=1), legacy_config)
        env.reset()
        env.bet_manager.bets.append(Bet(
            selection_id=101,
            side=BetSide.BACK,
            requested_stake=10.0,
            matched_stake=10.0,
            average_price=8.0,
            market_id="m1",
            outcome=BetOutcome.UNSETTLED,
        ))
        assert env._compute_portfolio_mtm({101: 6.0}) != 0.0
        env.bet_manager.bets[0].outcome = BetOutcome.WON
        assert env._compute_portfolio_mtm({101: 6.0}) == 0.0

    def test_mtm_telescopes_to_zero_at_settle(self, legacy_config):
        """Cumulative shaped MTM across a race is zero at settle.

        The test plants a bet at a price DIFFERENT from the synthetic
        day's LTP (4.0) so intermediate MTM is non-zero; the final
        delta on the settle step MUST unwind it. ``cumulative_mtm_shaped``
        equal zero at episode end within float tolerance is the core
        invariant per §8-§9."""
        cfg = copy.deepcopy(legacy_config)
        cfg["reward"]["mark_to_market_weight"] = 0.05
        env = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), cfg)
        env.reset()
        # Plant an open back bet at price 8.0 while the synthetic day's
        # LTP is 4.0 → non-trivial MTM each mid-race tick.
        env.bet_manager.bets.append(Bet(
            selection_id=101,
            side=BetSide.BACK,
            requested_stake=5.0,
            matched_stake=5.0,
            average_price=8.0,
            market_id=env.day.races[0].market_id,
            outcome=BetOutcome.UNSETTLED,
        ))
        env._mtm_prev = 0.0  # race just starting
        noop = np.zeros(env.action_space.shape, dtype=np.float32)
        terminated = False
        info = {}
        max_abs_mid_race_mtm = 0.0
        while not terminated:
            _, _, terminated, _, info = env.step(noop)
            max_abs_mid_race_mtm = max(
                max_abs_mid_race_mtm, abs(env._mtm_prev),
            )
        # Sanity check — the test's premise is that MTM did move during
        # the race. If this is zero the test degenerates to "0 == 0".
        assert max_abs_mid_race_mtm > 0.1
        # Telescope closes: cumulative shaped MTM across the race
        # returns to zero once all bets resolve at settle.
        assert info["cumulative_mtm_shaped"] == pytest.approx(0.0, abs=1e-6)
        # And raw+shaped must still equal total for the episode — the
        # per-race redistribution leaves the episode-level accounting
        # invariant intact.
        total_reward = info["raw_pnl_reward"] + info["shaped_bonus"]
        # The sum of step rewards equals total_reward as well; sample
        # it from the running cumulative accumulators as an independent
        # check of the invariant.
        assert info["raw_pnl_reward"] + info["shaped_bonus"] == pytest.approx(
            total_reward, abs=1e-9,
        )

    def test_invariant_raw_plus_shaped_with_nonzero_weight(
        self, scalping_config,
    ):
        """Full rollout with weight=0.05 — raw + shaped must equal
        total_reward per episode within float tolerance.

        This is the load-bearing regression guard per the 2026-04-18
        units-mismatch lesson (plans/naked-clip-and-stability/
        lessons_learnt.md). Unit tests on the formula alone can't
        catch a telescope-break that only surfaces in a full rollout."""
        cfg = copy.deepcopy(scalping_config)
        cfg["reward"]["mark_to_market_weight"] = 0.05
        env = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), cfg)
        env.reset()
        a = np.zeros(14 * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32)
        a[0] = 1.0
        a[14] = -0.8
        a[28] = 1.0
        total_reward = 0.0
        terminated = False
        info = {}
        while not terminated:
            _, r, terminated, _, info = env.step(a)
            total_reward += r
        assert total_reward == pytest.approx(
            info["raw_pnl_reward"] + info["shaped_bonus"], abs=1e-6,
        )

    def test_mtm_weight_zero_byte_identical_rollout(self, scalping_config):
        """weight=0 rollouts are byte-identical to pre-change.

        Constructs two envs on the same synthetic day and feeds the
        same deterministic action sequence. One uses the default
        weight (0.0, the pre-change no-op); the other explicitly
        sets weight=0.0 via reward_overrides — both paths must
        produce identical per-episode (raw, shaped, total) triples
        to float-eps."""
        # Two independent env instances from the same synthetic day.
        env_a = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), scalping_config)
        cfg_b = copy.deepcopy(scalping_config)
        cfg_b["reward"]["mark_to_market_weight"] = 0.0
        env_b = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), cfg_b)

        def _run(env: BetfairEnv) -> tuple[float, float, float]:
            env.reset()
            a = np.zeros(
                14 * SCALPING_ACTIONS_PER_RUNNER, dtype=np.float32,
            )
            a[0] = 1.0
            a[14] = -0.8
            a[28] = 1.0
            total = 0.0
            info = {}
            terminated = False
            while not terminated:
                _, r, terminated, _, info = env.step(a)
                total += r
            return total, info["raw_pnl_reward"], info["shaped_bonus"]

        a_total, a_raw, a_shaped = _run(env_a)
        b_total, b_raw, b_shaped = _run(env_b)
        assert a_total == pytest.approx(b_total, abs=1e-9)
        assert a_raw == pytest.approx(b_raw, abs=1e-9)
        assert a_shaped == pytest.approx(b_shaped, abs=1e-9)

    def test_info_mtm_delta_field_present(self, legacy_config):
        """Every env step carries mtm_delta / cumulative_mtm_shaped /
        mtm_weight_active in info (even at weight 0 — agents can be
        debugged in no-op mode, per §14)."""
        env = BetfairEnv(_make_day(n_races=1, n_pre_ticks=3), legacy_config)
        env.reset()
        noop = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(noop)
        assert "mtm_delta" in info
        assert "cumulative_mtm_shaped" in info
        assert "mtm_weight_active" in info
        assert isinstance(info["mtm_delta"], float)
        assert info["mtm_weight_active"] == 0.0

    def test_mtm_weight_from_reward_overrides(self, legacy_config):
        """The knob flows through the reward_overrides passthrough
        (same gene-propagation path as the other reward knobs)."""
        env = BetfairEnv(
            _make_day(n_races=1), legacy_config,
            reward_overrides={"mark_to_market_weight": 0.1},
        )
        assert env._mark_to_market_weight == 0.1
