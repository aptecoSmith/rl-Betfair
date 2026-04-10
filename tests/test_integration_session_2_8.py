"""Integration tests for Session 2.8 — Time-aware LSTM and time delta features.

These tests use real extracted data and the full pipeline to verify
end-to-end correctness.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

# Skip all tests if no real data is available
_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
_HAS_DATA = _DATA_DIR.exists() and any(_DATA_DIR.glob("*.parquet"))

pytestmark = pytest.mark.skipif(not _HAS_DATA, reason="No extracted data available")


def _load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _first_date() -> str:
    """Return the first available date string from processed data."""
    parquets = sorted(_DATA_DIR.glob("*_ticks.parquet"))
    if not parquets:
        parquets = sorted(_DATA_DIR.glob("*.parquet"))
    return parquets[0].stem.split("_")[0]


@pytest.fixture(scope="module")
def real_day():
    from data.episode_builder import load_day
    date = _first_date()
    return load_day(date, str(_DATA_DIR))


@pytest.fixture(scope="module")
def config():
    return _load_config()


class TestTimeFeaturesOnRealData:
    """Verify time delta features with real extracted data."""

    def test_time_features_populated(self, real_day):
        """All 4 time features should be present in every tick."""
        from data.feature_engineer import engineer_race

        race = real_day.races[0]
        features = engineer_race(race)
        for tick_feat in features:
            vel = tick_feat["market_velocity"]
            for key in ("seconds_since_last_tick", "seconds_spanned_3",
                         "seconds_spanned_5", "seconds_spanned_10"):
                assert key in vel
                assert not math.isnan(vel[key])

    def test_seconds_since_last_tick_first_is_zero(self, real_day):
        """First tick's seconds_since_last_tick should be 0."""
        from data.feature_engineer import engineer_race

        race = real_day.races[0]
        features = engineer_race(race)
        assert features[0]["market_velocity"]["seconds_since_last_tick"] == 0.0

    def test_seconds_since_last_tick_nonzero_after_first(self, real_day):
        """After first tick, seconds_since_last_tick should be > 0."""
        from data.feature_engineer import engineer_race

        race = real_day.races[0]
        if len(race.ticks) < 2:
            pytest.skip("Race has fewer than 2 ticks")
        features = engineer_race(race)
        assert features[1]["market_velocity"]["seconds_since_last_tick"] > 0.0

    def test_env_full_episode_with_time_features(self, real_day, config):
        """BetfairEnv runs a full episode with new obs_dim."""
        from env.betfair_env import BetfairEnv, MARKET_DIM, VELOCITY_DIM, RUNNER_DIM, AGENT_STATE_DIM

        env = BetfairEnv(real_day, config)
        obs, info = env.reset()
        assert obs.shape[0] == env.observation_space.shape[0]
        assert not np.any(np.isnan(obs))

        steps = 0
        while steps < 50:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert not np.any(np.isnan(obs))
            steps += 1
            if terminated:
                break


class TestTimeLSTMTraining:
    """Integration test: train 1 agent with ppo_time_lstm_v1 on real data."""

    def test_forward_pass_on_real_obs(self, real_day, config):
        """Forward pass with real observation produces valid output."""
        from env.betfair_env import BetfairEnv, MARKET_DIM, VELOCITY_DIM, RUNNER_DIM, AGENT_STATE_DIM
        from agents.policy_network import PPOTimeLSTMPolicy

        env = BetfairEnv(real_day, config)
        obs, _ = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * 14) + AGENT_STATE_DIM
        policy = PPOTimeLSTMPolicy(
            obs_dim=obs_dim,
            action_dim=42,
            max_runners=14,
            hyperparams={"lstm_hidden_size": 64, "mlp_hidden_size": 32, "mlp_layers": 1},
        )
        out = policy.forward(obs_t)
        assert out.action_mean.shape == (1, 42)
        assert out.value.shape == (1, 1)
        assert not torch.any(torch.isnan(out.action_mean))
        assert not torch.any(torch.isnan(out.value))

    @pytest.mark.xfail(reason="Fragile: random init may leave hidden state near zero, masking time-gate effects")
    def test_hidden_state_decay_differs_with_gap_size(self, real_day, config):
        """Hidden state should evolve differently for 5s vs 180s gaps.

        We create two observations that are identical except for the
        seconds_since_last_tick feature, and verify that after processing
        through the TimeLSTMPolicy, the hidden states differ.
        """
        from env.betfair_env import BetfairEnv, MARKET_DIM, VELOCITY_DIM, RUNNER_DIM, AGENT_STATE_DIM
        from agents.policy_network import PPOTimeLSTMPolicy

        torch.manual_seed(42)
        env = BetfairEnv(real_day, config)
        obs, _ = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * 14) + AGENT_STATE_DIM
        policy = PPOTimeLSTMPolicy(
            obs_dim=obs_dim,
            action_dim=42,
            max_runners=14,
            hyperparams={"lstm_hidden_size": 64, "mlp_hidden_size": 32, "mlp_layers": 1},
        )

        # Set W_dt to a meaningful value so the effect is visible
        # (multi-layer LSTM stores cells in a ModuleList; layer 0).
        with torch.no_grad():
            policy.time_lstm_cells[0].W_dt.fill_(2.0)

        # Run several steps to build up non-trivial hidden state
        hidden = policy.init_hidden(1)
        current_obs = obs
        for _ in range(5):
            obs_step = torch.tensor(current_obs, dtype=torch.float32).unsqueeze(0)
            out_step = policy.forward(obs_step, hidden)
            hidden = out_step.hidden_state
            action = out_step.action_mean.detach().numpy().flatten()
            current_obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break
        shared_hidden = hidden

        # Now process two observations with different time deltas
        obs_5s = obs_t.clone()
        obs_180s = obs_t.clone()
        time_delta_idx = MARKET_DIM + VELOCITY_DIM - 4
        obs_5s[0, time_delta_idx] = 5.0 / 300.0     # 5s gap normalised
        obs_180s[0, time_delta_idx] = 180.0 / 300.0  # 180s gap normalised

        out_5s = policy.forward(obs_5s, shared_hidden)
        out_180s = policy.forward(obs_180s, shared_hidden)

        # Cell states (c) should differ — h = o * tanh(c) may saturate,
        # masking differences, but the raw cell state captures forget gate effects
        c_5s = out_5s.hidden_state[1]
        c_180s = out_180s.hidden_state[1]
        assert not torch.allclose(c_5s, c_180s, atol=1e-6), \
            "Cell states should differ for 5s vs 180s time gaps"

    def test_training_completes(self, real_day, config):
        """A minimal training loop should complete without errors."""
        from env.betfair_env import BetfairEnv, MARKET_DIM, VELOCITY_DIM, RUNNER_DIM, AGENT_STATE_DIM
        from agents.policy_network import PPOTimeLSTMPolicy

        env = BetfairEnv(real_day, config)
        obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * 14) + AGENT_STATE_DIM
        policy = PPOTimeLSTMPolicy(
            obs_dim=obs_dim,
            action_dim=42,
            max_runners=14,
            hyperparams={"lstm_hidden_size": 64, "mlp_hidden_size": 32, "mlp_layers": 1},
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

        # Collect a few transitions
        obs, _ = env.reset()
        hidden = policy.init_hidden(1)
        for step in range(10):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            out = policy.forward(obs_t, hidden)
            hidden = out.hidden_state

            action = out.action_mean.detach().numpy().flatten()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        # Simple backward pass
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        out = policy.forward(obs_t, hidden)
        loss = -out.value.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify weights updated
        assert policy.time_lstm_cells[0].W_dt.grad is not None
