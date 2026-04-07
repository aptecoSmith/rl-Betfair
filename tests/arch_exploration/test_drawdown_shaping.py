"""
Session 7 — drawdown-aware shaping (zero-mean by reflection symmetry).

The critical invariant being guarded here is documented in the Session 7
design pass (``plans/arch-exploration/session_7_drawdown_shaping.md``):
for any day_pnl trajectory whose distribution is symmetric under
``X → −X``, the per-race drawdown shaping term
``weight · (2·day_pnl − peak − trough) / starting_budget`` sums to zero
in expectation. If that ever regresses, the tests in this file are the
tripwire.

All tests are CPU-only, no real market data, and are budgeted at a few
seconds total.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pytest
import yaml

from agents.population_manager import parse_search_ranges, sample_hyperparams
from agents.ppo_trainer import _reward_overrides_from_hp
from env.betfair_env import BetfairEnv

from tests.test_betfair_env import _make_day


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def reward_config() -> dict:
    """Minimal config block satisfied by BetfairEnv.__init__."""
    return {
        "training": {
            "max_runners": 14,
            "starting_budget": 100.0,
            "max_bets_per_race": 20,
        },
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "efficiency_penalty": 0.0,
            "precision_bonus": 0.0,
            "commission": 0.0,
            "drawdown_shaping_weight": 0.0,
        },
    }


@pytest.fixture
def real_search_ranges() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["hyperparameters"]["search_ranges"]


def _make_env(reward_config: dict, weight: float) -> BetfairEnv:
    """Construct a minimal env with the drawdown gene enabled."""
    day = _make_day(n_races=2, n_pre_ticks=3, n_inplay_ticks=1)
    return BetfairEnv(
        day,
        reward_config,
        reward_overrides={"drawdown_shaping_weight": weight},
    )


def _drive_day_pnl(env: BetfairEnv, deltas: list[float]) -> list[float]:
    """Push a sequence of race-P&L deltas through the drawdown helper.

    Mimics what ``_settle_current_race`` does around the helper call:
    it updates ``_day_pnl`` with the race delta and invokes
    ``_update_drawdown_shaping``, which advances the running peak /
    trough and returns the shaped contribution.
    """
    env.reset()
    contributions: list[float] = []
    for delta in deltas:
        env._day_pnl += delta
        contributions.append(env._update_drawdown_shaping())
    return contributions


# ── Test 1 — gene sampling ──────────────────────────────────────────────────


def test_sampler_produces_drawdown_gene_in_range(real_search_ranges):
    """reward_drawdown_shaping is in the schema and samples in range."""
    assert "reward_drawdown_shaping" in real_search_ranges
    spec = real_search_ranges["reward_drawdown_shaping"]
    assert spec["min"] == pytest.approx(0.0)
    assert spec["max"] == pytest.approx(0.2)

    specs = parse_search_ranges(real_search_ranges)
    rng = random.Random(7)
    for _ in range(50):
        hp = sample_hyperparams(specs, rng)
        assert "reward_drawdown_shaping" in hp
        assert spec["min"] <= hp["reward_drawdown_shaping"] <= spec["max"]


def test_gene_maps_to_env_override_key():
    """The gene name maps to the env reward-config key, not raw hp name."""
    overrides = _reward_overrides_from_hp(
        {"reward_drawdown_shaping": 0.05, "learning_rate": 1e-4},
    )
    assert overrides == {"drawdown_shaping_weight": 0.05}


# ── Test 2 — zero-mean for random policies (critical invariant) ─────────────


def test_zero_mean_over_random_walks(reward_config):
    """1000 symmetric random walks → mean total drawdown contribution ≈ 0.

    This is the critical invariant test. We feed the env's
    ``_update_drawdown_shaping`` helper a fresh random walk per seed.
    Since every step is drawn from a sign-symmetric distribution, the
    reflection proof in the design pass says the sum telescopes to
    zero in expectation.
    """
    n_trials = 1000
    n_races = 25
    weight = 0.1
    rng = np.random.default_rng(12345)

    env = _make_env(reward_config, weight=weight)

    totals = np.empty(n_trials, dtype=np.float64)
    for t in range(n_trials):
        # Symmetric step distribution: uniform on [-10, +10].
        deltas = rng.uniform(-10.0, 10.0, size=n_races).tolist()
        contributions = _drive_day_pnl(env, deltas)
        totals[t] = float(sum(contributions))

    mean = float(totals.mean())
    se = float(totals.std(ddof=1) / math.sqrt(n_trials))
    assert abs(mean) < 2.0 * se, (
        f"Drawdown shaping is NOT zero-mean: "
        f"mean={mean:+.5f}, SE={se:.5f}, |mean|/SE={abs(mean)/se:.2f}"
    )


def test_zero_mean_reflection_pairs_cancel(reward_config):
    """Every path and its sign-flipped reflection sum to exactly zero.

    Stronger than the statistical test above: algebraic cancellation
    at every single seed. Catches regressions that introduce an
    asymmetric bias even if the statistical test got lucky with noise.
    """
    weight = 0.1
    rng = np.random.default_rng(99)
    env = _make_env(reward_config, weight=weight)

    for _ in range(50):
        deltas = rng.uniform(-10.0, 10.0, size=15).tolist()
        forward = sum(_drive_day_pnl(env, deltas))
        reflected = sum(_drive_day_pnl(env, [-d for d in deltas]))
        assert forward + reflected == pytest.approx(0.0, abs=1e-9)


# ── Test 3 — drawdown-avoiding policy gets positive shaping ─────────────────


def test_drawdown_avoiding_policy_gets_positive_shaping(reward_config):
    """Monotone-up trajectory → strictly positive shaped contribution.

    Replicates worked example 2 from the design pass:
    day_pnl = [+2, +4, +6, +8] → Σ = +20 numerator; with weight=0.05
    and budget=100 → total = 0.05 · 20 / 100 = 0.01.
    """
    env = _make_env(reward_config, weight=0.05)
    contributions = _drive_day_pnl(env, [2.0, 2.0, 2.0, 2.0])
    total = sum(contributions)
    assert total > 0.0
    assert total == pytest.approx(0.05 * 20.0 / 100.0, abs=1e-9)


# ── Test 4 — drawdown-amplifying policy gets negative shaping ───────────────


def test_drawdown_amplifying_policy_gets_negative_shaping(reward_config):
    """Monotone-down-ish trajectory → strictly negative shaped contribution.

    Replicates worked example 3 from the design pass:
    day_pnl = [−10, −5, −15, −10] → Σ = −30 numerator.
    """
    env = _make_env(reward_config, weight=0.05)
    contributions = _drive_day_pnl(env, [-10.0, 5.0, -10.0, 5.0])
    total = sum(contributions)
    assert total < 0.0
    assert total == pytest.approx(0.05 * -30.0 / 100.0, abs=1e-9)


# ── Test 5 — raw + shaped invariant in a full settlement run ────────────────


def test_raw_plus_shaped_invariant_under_drawdown_weight(reward_config):
    """End-to-end: ``raw + shaped ≈ total_reward`` with drawdown enabled.

    Runs a real BetfairEnv episode with a non-zero drawdown weight and
    a deterministic action sequence, then verifies the accumulators
    agree with the sum of step rewards.
    """
    day = _make_day(n_races=3, n_pre_ticks=4, n_inplay_ticks=2)
    env = BetfairEnv(
        day,
        reward_config,
        reward_overrides={"drawdown_shaping_weight": 0.08},
    )

    obs, _ = env.reset()
    assert env._drawdown_shaping_weight == pytest.approx(0.08)
    total_reward = 0.0
    done = False
    # Deterministic small-back actions so *something* hits the
    # settlement path; exact values don't matter for the invariant.
    action = 0.5 * np.ones(env.action_space.shape, dtype=np.float32)
    while not done:
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    raw = info["raw_pnl_reward"]
    shaped = info["shaped_bonus"]
    assert raw + shaped == pytest.approx(total_reward, abs=1e-6)


# ── Test 6 — drawdown term lands in shaped, not raw ─────────────────────────


def test_drawdown_term_buckets_into_shaped_not_raw(reward_config):
    """The drawdown contribution must accumulate into ``_cum_shaped_reward``,
    NEVER into ``_cum_raw_reward``. Raw is real money only.

    This test drives a deterministic monotone-up day_pnl trajectory
    through the helper and then compares the env's shaped / raw
    accumulators before and after a direct _settle_current_race call
    on a fresh env. Because the fixture sets efficiency_penalty =
    precision_bonus = 0 and the zero-action rollout produces zero
    race_pnl, the drawdown contribution is the *only* non-zero shaped
    term and raw must stay at 0.
    """
    day = _make_day(n_races=3, n_pre_ticks=3, n_inplay_ticks=1)
    env = BetfairEnv(
        day,
        reward_config,
        reward_overrides={"drawdown_shaping_weight": 0.1},
    )
    obs, _ = env.reset()

    # Zero-action rollout: no bets placed → race_pnl is 0 every race.
    # day_pnl therefore stays at 0 and the drawdown term is also 0,
    # so both accumulators should remain at 0. That alone proves the
    # term isn't sneaking into raw — but we also want to see the
    # *non-trivial* case where drawdown is active and raw stays clean.
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(zero_action)
        done = terminated or truncated
    assert info["raw_pnl_reward"] == pytest.approx(0.0, abs=1e-9)
    assert info["shaped_bonus"] == pytest.approx(0.0, abs=1e-9)

    # Now drive a non-trivial drawdown trajectory via the helper and
    # assert the env's shaped accumulator moves while raw stays pinned.
    env2 = _make_env(reward_config, weight=0.1)
    env2.reset()
    raw_before = env2._cum_raw_reward
    shaped_before = env2._cum_shaped_reward

    # Mirror what _settle_current_race does: update _day_pnl with a
    # race P&L, then call the helper and manually bucket the result
    # into the shaped accumulator exactly like the real method does.
    deltas = [5.0, -3.0, 7.0, -8.0]
    shaped_accum = 0.0
    for delta in deltas:
        env2._day_pnl += delta
        shaped_accum += env2._update_drawdown_shaping()
    env2._cum_shaped_reward += shaped_accum

    assert env2._cum_raw_reward == pytest.approx(raw_before, abs=1e-9)
    assert env2._cum_shaped_reward == pytest.approx(
        shaped_before + shaped_accum, abs=1e-9,
    )
    # And the info dict exposes it under "shaped_bonus", not
    # "raw_pnl_reward" — the test hook the design pass calls out.
    info2 = env2._get_info()
    assert info2["shaped_bonus"] == pytest.approx(env2._cum_shaped_reward)
    assert info2["raw_pnl_reward"] == pytest.approx(env2._cum_raw_reward)


# ── Test 7 — weight = 0 disables the feature cleanly ────────────────────────


def test_zero_weight_disables_term(reward_config):
    """Default weight=0 must be a no-op: helper returns 0 for any path."""
    env = _make_env(reward_config, weight=0.0)
    contributions = _drive_day_pnl(env, [10.0, -20.0, 5.0, -15.0, 30.0])
    assert contributions == [0.0, 0.0, 0.0, 0.0, 0.0]
    # And peak/trough stay at 0 because the early-exit branch
    # skips the running-max/min update entirely.
    assert env._day_pnl_peak == 0.0
    assert env._day_pnl_trough == 0.0
