"""Integration tests for policy network with real extracted data.

These tests load real Parquet data, build episodes, engineer features,
construct observation vectors, and feed them through the policy network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from agents.architecture_registry import create_policy
from agents.policy_network import PPOLSTMPolicy
from data.episode_builder import load_day
from data.feature_engineer import engineer_day
from env.betfair_env import (
    ACTIONS_PER_RUNNER,
    AGENT_STATE_DIM,
    MARKET_DIM,
    MARKET_KEYS,
    MARKET_VELOCITY_KEYS,
    RUNNER_DIM,
    RUNNER_KEYS,
    VELOCITY_DIM,
)

pytestmark = pytest.mark.integration

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def _find_real_date() -> str | None:
    """Find a date that has both ticks and runners Parquet files."""
    if not PROCESSED_DIR.exists():
        return None
    for f in sorted(PROCESSED_DIR.glob("*_runners.parquet")):
        date_str = f.stem.replace("_runners", "")
        ticks_file = PROCESSED_DIR / f"{date_str}.parquet"
        if ticks_file.exists():
            return date_str
    return None


@pytest.fixture(scope="module")
def real_date() -> str:
    date = _find_real_date()
    if date is None:
        pytest.skip("No real extracted data available")
    return date


@pytest.fixture(scope="module")
def real_day(real_date: str):
    return load_day(real_date, data_dir=str(PROCESSED_DIR))


@pytest.fixture(scope="module")
def engineered_day(real_day):
    return engineer_day(real_day)


@pytest.fixture(scope="module")
def policy(config: dict) -> PPOLSTMPolicy:
    max_runners = config["training"]["max_runners"]
    obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * max_runners) + AGENT_STATE_DIM
    action_dim = max_runners * ACTIONS_PER_RUNNER
    return create_policy(
        config["training"]["architecture"],
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_runners=max_runners,
    )


def _safe_float(v: object) -> float:
    """Convert a value to float, replacing None/NaN/inf with 0.0."""
    import math

    if v is None:
        return 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return f


def _build_obs_vector(
    tick_features: dict,
    max_runners: int,
) -> torch.Tensor:
    """Build a flat observation vector from engineered tick features.

    Mirrors the logic in BetfairEnv._build_obs but standalone for testing.
    NaN/inf values are replaced with 0.0 (same as the real env).
    """
    market_vals = [_safe_float(tick_features["market"].get(k, 0.0)) for k in MARKET_KEYS]
    velocity_vals = [
        _safe_float(tick_features["market_velocity"].get(k, 0.0))
        for k in MARKET_VELOCITY_KEYS
    ]

    runner_vals: list[float] = []
    runner_ids = sorted(tick_features["runners"].keys())
    for i in range(max_runners):
        if i < len(runner_ids):
            rid = runner_ids[i]
            rdata = tick_features["runners"][rid]
            for k in RUNNER_KEYS:
                runner_vals.append(_safe_float(rdata.get(k, 0.0)))
        else:
            runner_vals.extend([0.0] * RUNNER_DIM)

    # Agent state (placeholder values for testing)
    agent_state = [0.0, 1.0, 0.0, 0.0, 0.5]  # in_play, budget_frac, liability, bets, races

    flat = market_vals + velocity_vals + runner_vals + agent_state
    return torch.tensor(flat, dtype=torch.float32).unsqueeze(0)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestRealDataForwardPass:
    def test_obs_vector_from_real_data(self, engineered_day, config: dict):
        """Build obs vector from real features — verify correct dimension."""
        max_runners = config["training"]["max_runners"]
        expected_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * max_runners) + AGENT_STATE_DIM

        race = engineered_day[0]
        tick_features = race[0]
        obs = _build_obs_vector(tick_features, max_runners)
        assert obs.shape == (1, expected_dim)

    def test_forward_pass_on_real_obs(self, engineered_day, policy: PPOLSTMPolicy, config: dict):
        """Feed real observation through policy network — verify output shapes."""
        max_runners = config["training"]["max_runners"]
        race = engineered_day[0]
        tick_features = race[0]
        obs = _build_obs_vector(tick_features, max_runners)

        out = policy(obs)
        assert out.action_mean.shape == (1, max_runners * ACTIONS_PER_RUNNER)
        assert out.value.shape == (1, 1)
        assert not torch.isnan(out.action_mean).any()
        assert not torch.isnan(out.value).any()

    def test_hidden_state_carries_across_real_ticks(
        self, engineered_day, policy: PPOLSTMPolicy, config: dict
    ):
        """Process multiple real ticks, verifying hidden state propagation."""
        max_runners = config["training"]["max_runners"]
        race = engineered_day[0]

        hidden = None
        prev_h = None
        for i, tick_features in enumerate(race[:5]):  # first 5 ticks
            obs = _build_obs_vector(tick_features, max_runners)
            out = policy(obs, hidden_state=hidden)
            hidden = out.hidden_state

            h, _ = hidden
            if prev_h is not None:
                # Hidden state should change between ticks
                assert not torch.allclose(h, prev_h), f"Hidden unchanged at tick {i}"
            prev_h = h.clone()

    def test_hidden_state_carries_across_races(
        self, engineered_day, policy: PPOLSTMPolicy, config: dict
    ):
        """Hidden state persists from last tick of race N to first tick of race N+1."""
        max_runners = config["training"]["max_runners"]

        if len(engineered_day) < 2:
            pytest.skip("Need at least 2 races for cross-race test")

        # Process last tick of first race
        race1 = engineered_day[0]
        obs1 = _build_obs_vector(race1[-1], max_runners)
        out1 = policy(obs1)

        # Process first tick of second race using hidden from first race
        race2 = engineered_day[1]
        obs2 = _build_obs_vector(race2[0], max_runners)
        out_with = policy(obs2, hidden_state=out1.hidden_state)
        out_without = policy(obs2)

        # Should be different
        assert not torch.allclose(out_with.action_mean, out_without.action_mean)

    def test_output_per_runner_count(
        self, engineered_day, policy: PPOLSTMPolicy, config: dict
    ):
        """Action output has entries for all runner slots (padded to max_runners)."""
        max_runners = config["training"]["max_runners"]
        race = engineered_day[0]
        obs = _build_obs_vector(race[0], max_runners)
        out = policy(obs)

        # First half: action signals, second half: stake fractions
        signals = out.action_mean[0, :max_runners]
        stakes = out.action_mean[0, max_runners:]
        assert signals.shape == (max_runners,)
        assert stakes.shape == (max_runners,)

    def test_no_nan_across_full_race(
        self, engineered_day, policy: PPOLSTMPolicy, config: dict
    ):
        """Process every tick in a full race — no NaN in any output."""
        max_runners = config["training"]["max_runners"]
        race = engineered_day[0]

        hidden = None
        for i, tick_features in enumerate(race):
            obs = _build_obs_vector(tick_features, max_runners)
            out = policy(obs, hidden_state=hidden)
            hidden = out.hidden_state
            assert not torch.isnan(out.action_mean).any(), f"NaN at tick {i}"
            assert not torch.isnan(out.value).any(), f"NaN value at tick {i}"

    def test_action_distribution_on_real_data(
        self, engineered_day, policy: PPOLSTMPolicy, config: dict
    ):
        """Get action distribution from real data — sample and compute log prob."""
        max_runners = config["training"]["max_runners"]
        race = engineered_day[0]
        obs = _build_obs_vector(race[0], max_runners)

        dist, value, hidden = policy.get_action_distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        assert action.shape == (1, max_runners * ACTIONS_PER_RUNNER)
        assert log_prob.shape == (1, max_runners * ACTIONS_PER_RUNNER)
        assert not torch.isnan(log_prob).any()


class TestRegistryWithConfig:
    def test_config_architecture_in_registry(self, config: dict):
        """The architecture named in config.yaml is registered."""
        from agents.architecture_registry import REGISTRY
        arch_name = config["training"]["architecture"]
        assert arch_name in REGISTRY

    def test_create_from_config(self, config: dict):
        """Create policy using config values end-to-end."""
        max_runners = config["training"]["max_runners"]
        obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * max_runners) + AGENT_STATE_DIM
        action_dim = max_runners * ACTIONS_PER_RUNNER

        policy = create_policy(
            config["training"]["architecture"],
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_runners=max_runners,
        )
        assert isinstance(policy, PPOLSTMPolicy)
        obs = torch.randn(1, obs_dim)
        out = policy(obs)
        assert out.action_mean.shape == (1, action_dim)
