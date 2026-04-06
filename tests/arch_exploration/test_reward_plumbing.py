"""
Session 1 — verify reward-shaping genes actually reach BetfairEnv.

These tests are the regression harness for the "sampled ≠ used" bug
(see plans/arch-exploration/lessons_learnt.md). Every gene that the
genetic sampler produces must have a corresponding plumbing test that
asserts the downstream consumer observes the value — not merely that
the value was sampled.

All tests in this file are CPU-only and must run in well under a
second each.  No GPU, no full training loops, no real market data.
"""

from __future__ import annotations

import random

import pytest
import torch
import torch.nn as nn
import yaml
from pathlib import Path

from agents.population_manager import (
    parse_search_ranges,
    sample_hyperparams,
)
from agents.ppo_trainer import PPOTrainer, _reward_overrides_from_hp
from env.betfair_env import BetfairEnv

# Re-use the synthetic Day/Race/Tick builders from the existing env tests.
# They produce the smallest in-memory fixture the env can step through,
# which is exactly what the session plan asks for.
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
            "efficiency_penalty": 0.01,
            "precision_bonus": 1.0,
            "commission": 0.05,
        },
    }


@pytest.fixture
def real_search_ranges() -> dict:
    """The hyperparameter search_ranges block from the live config.yaml."""
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["hyperparameters"]["search_ranges"]


# ── Test 1 — sampling produces reward genes in range ───────────────────────


def test_sampler_produces_reward_genes_in_range(real_search_ranges):
    """Sanity: the three reward genes are still sampled and within range.

    This is the "does the sampler know about them at all" half of the
    plumbing. Test 4 covers the "does the trainer act on them" half.
    """
    specs = parse_search_ranges(real_search_ranges)
    rng = random.Random(0)

    for _ in range(20):
        hp = sample_hyperparams(specs, rng)
        assert "reward_early_pick_bonus" in hp
        assert "reward_efficiency_penalty" in hp
        assert "reward_precision_bonus" in hp

        epb_spec = real_search_ranges["reward_early_pick_bonus"]
        eff_spec = real_search_ranges["reward_efficiency_penalty"]
        prc_spec = real_search_ranges["reward_precision_bonus"]
        assert epb_spec["min"] <= hp["reward_early_pick_bonus"] <= epb_spec["max"]
        assert eff_spec["min"] <= hp["reward_efficiency_penalty"] <= eff_spec["max"]
        assert prc_spec["min"] <= hp["reward_precision_bonus"] <= prc_spec["max"]


# ── Test 2 — env honours reward_overrides ──────────────────────────────────


def test_env_honours_reward_overrides(reward_config):
    """BetfairEnv must use the per-agent overrides, not config defaults."""
    day = _make_day(n_races=1, n_pre_ticks=3, n_inplay_ticks=1)
    env = BetfairEnv(
        day,
        reward_config,
        reward_overrides={
            "efficiency_penalty": 0.99,
            "precision_bonus": 0.0,
        },
    )

    assert env._efficiency_penalty == pytest.approx(0.99)
    assert env._precision_bonus == pytest.approx(0.0)
    # Un-overridden keys keep the config defaults.
    assert env._early_pick_min == pytest.approx(1.2)
    assert env._early_pick_max == pytest.approx(1.5)


def test_env_overrides_do_not_mutate_shared_config(reward_config):
    """Overrides must NOT mutate the shared config dict — that would
    mean every agent clobbers its neighbours' reward block."""
    original_efficiency = reward_config["reward"]["efficiency_penalty"]
    day = _make_day(n_races=1, n_pre_ticks=3, n_inplay_ticks=1)
    BetfairEnv(
        day,
        reward_config,
        reward_overrides={"efficiency_penalty": 0.99},
    )
    assert reward_config["reward"]["efficiency_penalty"] == original_efficiency


# ── Test 3 — unknown override keys are ignored ─────────────────────────────


def test_env_ignores_unknown_override_keys(reward_config, caplog):
    """A typoed / unknown override key must not crash construction.

    The env silently drops it (a debug log is emitted, per the
    ``_REWARD_OVERRIDE_KEYS`` contract documented on BetfairEnv).
    """
    day = _make_day(n_races=1, n_pre_ticks=3, n_inplay_ticks=1)
    env = BetfairEnv(
        day,
        reward_config,
        reward_overrides={
            "efficiency_penalty": 0.5,
            "this_key_does_not_exist": 999.0,
            "another_typo": "banana",
        },
    )
    # Known override applied.
    assert env._efficiency_penalty == pytest.approx(0.5)
    # Unknown keys did not land on the env as attributes.
    assert not hasattr(env, "_this_key_does_not_exist")
    assert not hasattr(env, "_another_typo")


# ── Test 4 — trainer passes overrides into env ─────────────────────────────


def test_reward_overrides_from_hp_maps_gene_names():
    """Direct unit test of the gene-name → reward-config-key mapping."""
    hp = {
        "learning_rate": 1e-4,          # not a reward gene — must be ignored
        "reward_early_pick_bonus": 1.42,
        "reward_efficiency_penalty": 0.037,
        "reward_precision_bonus": 2.1,
        "architecture_name": "ppo_lstm_v1",  # must be ignored
    }
    overrides = _reward_overrides_from_hp(hp)

    assert overrides == {
        "early_pick_bonus_min": 1.42,
        "early_pick_bonus_max": 1.42,
        "efficiency_penalty": 0.037,
        "precision_bonus": 2.1,
    }


def test_trainer_passes_reward_overrides_to_env(monkeypatch, reward_config):
    """PPOTrainer must construct BetfairEnv with a reward_overrides kwarg
    derived from self.hyperparams. This is the test that proves the
    plumbing is live end-to-end."""
    captured: dict = {}

    class _CapturingEnvError(RuntimeError):
        """Raised by the mock env to stop rollout collection early."""

    def _fake_env(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise _CapturingEnvError("captured")

    monkeypatch.setattr(
        "agents.ppo_trainer.BetfairEnv",
        _fake_env,
    )

    # A minimal torch module stands in for BasePolicy — PPOTrainer only
    # needs .to(device) and .parameters() at construction time, and we
    # never let rollout collection get past BetfairEnv(...).
    policy = nn.Linear(1, 1)
    hp = {
        "learning_rate": 3e-4,
        "reward_early_pick_bonus": 1.33,
        "reward_efficiency_penalty": 0.02,
        "reward_precision_bonus": 0.5,
    }
    trainer = PPOTrainer(
        policy=policy,
        config=reward_config,
        hyperparams=hp,
        device="cpu",
    )

    # Construction-time check: the trainer cached the derived overrides.
    assert trainer.reward_overrides == {
        "early_pick_bonus_min": 1.33,
        "early_pick_bonus_max": 1.33,
        "efficiency_penalty": 0.02,
        "precision_bonus": 0.5,
    }

    # Run-time check: those overrides reach the BetfairEnv constructor.
    # _collect_rollout is the single call-site so triggering it once is
    # enough to capture the kwargs before the mock raises.
    sentinel_day = object()
    with pytest.raises(_CapturingEnvError):
        trainer._collect_rollout(sentinel_day)

    assert captured["kwargs"].get("reward_overrides") == trainer.reward_overrides
    assert captured["args"][0] is sentinel_day


# ── Test 5 — raw + shaped invariant under non-default overrides ────────────


def test_raw_plus_shaped_equals_total_under_overrides(reward_config):
    """The CLAUDE.md invariant ``raw + shaped ≈ total_reward`` must still
    hold after we let the genetic algorithm mutate the shaping knobs."""
    day = _make_day(n_races=2, n_pre_ticks=4, n_inplay_ticks=2)
    env = BetfairEnv(
        day,
        reward_config,
        reward_overrides={
            "efficiency_penalty": 0.25,     # large, so shaping is not a rounding error
            "precision_bonus": 2.5,
            "early_pick_bonus_min": 1.4,
            "early_pick_bonus_max": 1.4,
        },
    )

    _, _ = env.reset()
    total_reward = 0.0
    done = False
    # Deterministic zero actions: no bets placed, so race_pnl == 0 but
    # the invariant must still hold term-by-term.
    zero_action = env.action_space.sample() * 0.0
    while not done:
        _, reward, terminated, truncated, info = env.step(zero_action)
        total_reward += reward
        done = terminated or truncated

    raw = info["raw_pnl_reward"]
    shaped = info["shaped_bonus"]
    assert raw + shaped == pytest.approx(total_reward, abs=1e-6)


# ── Test 6 — observation_window_ticks is gone ──────────────────────────────


def test_observation_window_ticks_retired(real_search_ranges):
    """The dead gene must be removed from the schema sampled by Gen 0."""
    assert "observation_window_ticks" not in real_search_ranges

    specs = parse_search_ranges(real_search_ranges)
    rng = random.Random(0)
    hp = sample_hyperparams(specs, rng)
    assert "observation_window_ticks" not in hp
