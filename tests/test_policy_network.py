"""Unit tests for agents/policy_network.py and agents/architecture_registry.py."""

from __future__ import annotations

import pytest
import torch

from agents.architecture_registry import REGISTRY, create_policy
from agents.policy_network import (
    MARKET_TOTAL_DIM,
    BasePolicy,
    PolicyOutput,
    PPOLSTMPolicy,
)
from env.betfair_env import (
    AGENT_STATE_DIM,
    MARKET_DIM,
    RUNNER_DIM,
    VELOCITY_DIM,
)


# ── Constants for tests ─────────────────────────────────────────────────────

MAX_RUNNERS = 14
OBS_DIM = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * MAX_RUNNERS) + AGENT_STATE_DIM
ACTION_DIM = MAX_RUNNERS * 2  # action_signal + stake_fraction per runner


@pytest.fixture
def default_hyperparams() -> dict:
    return {
        "lstm_hidden_size": 64,
        "mlp_hidden_size": 32,
        "mlp_layers": 2,
    }


@pytest.fixture
def policy(default_hyperparams: dict) -> PPOLSTMPolicy:
    return PPOLSTMPolicy(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        max_runners=MAX_RUNNERS,
        hyperparams=default_hyperparams,
    )


# ── Architecture registry ───────────────────────────────────────────────────


class TestArchitectureRegistry:
    def test_ppo_lstm_v1_registered(self):
        assert "ppo_lstm_v1" in REGISTRY

    def test_registry_returns_correct_class(self):
        assert REGISTRY["ppo_lstm_v1"] is PPOLSTMPolicy

    def test_create_policy_by_name(self, default_hyperparams: dict):
        policy = create_policy(
            "ppo_lstm_v1",
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            max_runners=MAX_RUNNERS,
            hyperparams=default_hyperparams,
        )
        assert isinstance(policy, PPOLSTMPolicy)

    def test_create_policy_unknown_name_raises(self):
        with pytest.raises(KeyError, match="Unknown architecture"):
            create_policy("nonexistent", obs_dim=100, action_dim=28, max_runners=14)

    def test_create_policy_default_hyperparams(self):
        policy = create_policy(
            "ppo_lstm_v1",
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            max_runners=MAX_RUNNERS,
        )
        assert isinstance(policy, PPOLSTMPolicy)


# ── Observation dimension constants ─────────────────────────────────────────


class TestObsDimConstants:
    def test_market_dim(self):
        assert MARKET_DIM == 31  # 25 base + 6 race status one-hot

    def test_velocity_dim(self):
        assert VELOCITY_DIM == 7  # 6 base + 1 time_since_status_change

    def test_runner_dim(self):
        assert RUNNER_DIM == 93

    def test_agent_state_dim(self):
        assert AGENT_STATE_DIM == 5

    def test_market_total_dim(self):
        assert MARKET_TOTAL_DIM == 43  # 31 + 7 + 5

    def test_obs_dim_matches_env(self):
        expected = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * MAX_RUNNERS) + AGENT_STATE_DIM
        assert expected == 1345
        assert OBS_DIM == 1345


# ── PPOLSTMPolicy architecture ──────────────────────────────────────────────


class TestPPOLSTMPolicyInit:
    def test_architecture_name(self, policy: PPOLSTMPolicy):
        assert policy.architecture_name == "ppo_lstm_v1"

    def test_description_not_empty(self, policy: PPOLSTMPolicy):
        assert len(policy.description) > 10

    def test_obs_dim_stored(self, policy: PPOLSTMPolicy):
        assert policy.obs_dim == OBS_DIM

    def test_action_dim_stored(self, policy: PPOLSTMPolicy):
        assert policy.action_dim == ACTION_DIM

    def test_max_runners_stored(self, policy: PPOLSTMPolicy):
        assert policy.max_runners == MAX_RUNNERS

    def test_lstm_hidden_size_from_hyperparams(self):
        p = PPOLSTMPolicy(OBS_DIM, ACTION_DIM, MAX_RUNNERS, {"lstm_hidden_size": 512})
        assert p.lstm_hidden_size == 512

    def test_default_lstm_hidden_size(self):
        p = PPOLSTMPolicy(OBS_DIM, ACTION_DIM, MAX_RUNNERS, {})
        assert p.lstm_hidden_size == 256

    def test_is_nn_module(self, policy: PPOLSTMPolicy):
        assert isinstance(policy, torch.nn.Module)

    def test_is_base_policy(self, policy: PPOLSTMPolicy):
        assert isinstance(policy, BasePolicy)


# ── Forward pass shapes ─────────────────────────────────────────────────────


class TestForwardPassShapes:
    def test_single_timestep_output_shapes(self, policy: PPOLSTMPolicy):
        obs = torch.randn(1, OBS_DIM)
        out = policy(obs)
        assert isinstance(out, PolicyOutput)
        assert out.action_mean.shape == (1, ACTION_DIM)
        assert out.action_log_std.shape == (1, ACTION_DIM)
        assert out.value.shape == (1, 1)

    def test_batch_output_shapes(self, policy: PPOLSTMPolicy):
        batch = 4
        obs = torch.randn(batch, OBS_DIM)
        out = policy(obs)
        assert out.action_mean.shape == (batch, ACTION_DIM)
        assert out.action_log_std.shape == (batch, ACTION_DIM)
        assert out.value.shape == (batch, 1)

    def test_sequence_output_shapes(self, policy: PPOLSTMPolicy):
        batch, seq_len = 2, 10
        obs = torch.randn(batch, seq_len, OBS_DIM)
        out = policy(obs)
        assert out.action_mean.shape == (batch, ACTION_DIM)
        assert out.value.shape == (batch, 1)

    def test_hidden_state_shape(self, policy: PPOLSTMPolicy):
        obs = torch.randn(1, OBS_DIM)
        out = policy(obs)
        h, c = out.hidden_state
        assert h.shape == (1, 1, 64)  # (num_layers, batch, hidden)
        assert c.shape == (1, 1, 64)

    def test_hidden_state_batch(self, policy: PPOLSTMPolicy):
        batch = 3
        obs = torch.randn(batch, OBS_DIM)
        out = policy(obs)
        h, c = out.hidden_state
        assert h.shape == (1, batch, 64)
        assert c.shape == (1, batch, 64)


# ── Hidden state persistence ────────────────────────────────────────────────


class TestHiddenStatePersistence:
    def test_hidden_state_carries_forward(self, policy: PPOLSTMPolicy):
        """Hidden state from tick N feeds into tick N+1."""
        obs1 = torch.randn(1, OBS_DIM)
        obs2 = torch.randn(1, OBS_DIM)

        out1 = policy(obs1)
        out2_with_state = policy(obs2, hidden_state=out1.hidden_state)
        out2_without_state = policy(obs2)

        # Outputs should differ when hidden state is passed vs fresh
        assert not torch.allclose(
            out2_with_state.action_mean, out2_without_state.action_mean
        )

    def test_init_hidden_zeros(self, policy: PPOLSTMPolicy):
        h, c = policy.init_hidden(batch_size=2)
        assert torch.all(h == 0)
        assert torch.all(c == 0)
        assert h.shape == (1, 2, 64)

    def test_explicit_none_hidden_uses_zeros(self, policy: PPOLSTMPolicy):
        """Passing None should produce same result as init_hidden."""
        obs = torch.randn(1, OBS_DIM)
        torch.manual_seed(42)
        out_none = policy(obs, hidden_state=None)
        torch.manual_seed(42)
        out_zeros = policy(obs, hidden_state=policy.init_hidden(1))
        assert torch.allclose(out_none.action_mean, out_zeros.action_mean, atol=1e-6)

    def test_hidden_state_changes_across_ticks(self, policy: PPOLSTMPolicy):
        """Hidden state should not be identical after processing different inputs."""
        obs1 = torch.randn(1, OBS_DIM)
        obs2 = torch.randn(1, OBS_DIM)

        out1 = policy(obs1)
        out2 = policy(obs2, hidden_state=out1.hidden_state)

        h1, _ = out1.hidden_state
        h2, _ = out2.hidden_state
        assert not torch.allclose(h1, h2)


# ── Gradient flow ────────────────────────────────────────────────────────────


class TestGradientFlow:
    def test_gradients_flow_through_actor(self, policy: PPOLSTMPolicy):
        obs = torch.randn(2, OBS_DIM)
        out = policy(obs)
        loss = out.action_mean.sum()
        loss.backward()

        # Check gradients exist on key actor components (not critic, not log_std)
        # action_log_std only gets gradient when used in the distribution
        for name, param in policy.named_parameters():
            if param.requires_grad and "critic" not in name and name != "action_log_std":
                assert param.grad is not None, f"No grad for {name}"

    def test_gradients_flow_through_critic(self, policy: PPOLSTMPolicy):
        obs = torch.randn(2, OBS_DIM)
        out = policy(obs)
        loss = out.value.sum()
        loss.backward()

        for name, param in policy.named_parameters():
            if param.requires_grad and "critic" in name:
                assert param.grad is not None, f"No grad for critic param {name}"

    def test_gradients_flow_through_lstm(self, policy: PPOLSTMPolicy):
        obs = torch.randn(2, OBS_DIM)
        out = policy(obs)
        loss = out.action_mean.sum() + out.value.sum()
        loss.backward()

        for name, param in policy.lstm.named_parameters():
            assert param.grad is not None, f"No grad for LSTM param {name}"

    def test_gradients_flow_through_runner_encoder(self, policy: PPOLSTMPolicy):
        obs = torch.randn(2, OBS_DIM)
        out = policy(obs)
        loss = out.action_mean.sum()
        loss.backward()

        for name, param in policy.runner_encoder.named_parameters():
            assert param.grad is not None, f"No grad for runner_encoder {name}"

    def test_gradients_flow_through_market_encoder(self, policy: PPOLSTMPolicy):
        obs = torch.randn(2, OBS_DIM)
        out = policy(obs)
        loss = out.action_mean.sum()
        loss.backward()

        for name, param in policy.market_encoder.named_parameters():
            assert param.grad is not None, f"No grad for market_encoder {name}"

    def test_log_std_has_gradient(self, policy: PPOLSTMPolicy):
        obs = torch.randn(2, OBS_DIM)
        out = policy(obs)
        # Use log_std in loss to ensure grad flows
        std = out.action_log_std.exp()
        dist = torch.distributions.Normal(out.action_mean, std)
        action = dist.rsample()
        loss = dist.log_prob(action).sum()
        loss.backward()
        assert policy.action_log_std.grad is not None

    def test_gradients_through_sequence(self, policy: PPOLSTMPolicy):
        """Gradients flow through multi-timestep sequence."""
        obs = torch.randn(1, 5, OBS_DIM)
        out = policy(obs)
        loss = out.action_mean.sum() + out.value.sum()
        loss.backward()

        for name, param in policy.lstm.named_parameters():
            assert param.grad is not None, f"No grad for LSTM param {name} (sequence)"


# ── Action distribution ─────────────────────────────────────────────────────


class TestActionDistribution:
    def test_get_action_distribution(self, policy: PPOLSTMPolicy):
        obs = torch.randn(1, OBS_DIM)
        dist, value, hidden = policy.get_action_distribution(obs)
        assert isinstance(dist, torch.distributions.Normal)
        assert value.shape == (1, 1)

    def test_sample_action_shape(self, policy: PPOLSTMPolicy):
        obs = torch.randn(1, OBS_DIM)
        dist, _, _ = policy.get_action_distribution(obs)
        action = dist.sample()
        assert action.shape == (1, ACTION_DIM)

    def test_log_prob_shape(self, policy: PPOLSTMPolicy):
        obs = torch.randn(1, OBS_DIM)
        dist, _, _ = policy.get_action_distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        assert log_prob.shape == (1, ACTION_DIM)

    def test_action_distribution_with_hidden_state(self, policy: PPOLSTMPolicy):
        obs = torch.randn(1, OBS_DIM)
        h = policy.init_hidden(1)
        dist, value, new_h = policy.get_action_distribution(obs, h)
        assert isinstance(dist, torch.distributions.Normal)
        assert new_h[0].shape == (1, 1, 64)


# ── Observation splitting ────────────────────────────────────────────────────


class TestObsSplitting:
    def test_split_obs_shapes(self, policy: PPOLSTMPolicy):
        obs = torch.randn(2, OBS_DIM)
        market_feats, runner_feats = policy._split_obs(obs)
        assert market_feats.shape == (2, MARKET_TOTAL_DIM)
        assert runner_feats.shape == (2, MAX_RUNNERS, RUNNER_DIM)

    def test_split_obs_market_values_correct(self, policy: PPOLSTMPolicy):
        """Verify the split correctly extracts market features."""
        obs = torch.zeros(1, OBS_DIM)
        # Set first market feature to known value
        obs[0, 0] = 42.0
        # Set first velocity feature
        obs[0, MARKET_DIM] = 7.0
        # Set first agent state feature
        agent_start = MARKET_DIM + VELOCITY_DIM + MAX_RUNNERS * RUNNER_DIM
        obs[0, agent_start] = 99.0

        market_feats, _ = policy._split_obs(obs)
        assert market_feats[0, 0].item() == 42.0      # market[0]
        assert market_feats[0, MARKET_DIM].item() == 7.0  # velocity[0]
        assert market_feats[0, MARKET_DIM + VELOCITY_DIM].item() == 99.0  # agent[0]

    def test_split_obs_runner_values_correct(self, policy: PPOLSTMPolicy):
        """Verify runner features land in the right slots."""
        obs = torch.zeros(1, OBS_DIM)
        runner_start = MARKET_DIM + VELOCITY_DIM
        # Set first feature of runner 0
        obs[0, runner_start] = 1.0
        # Set first feature of runner 1
        obs[0, runner_start + RUNNER_DIM] = 2.0

        _, runner_feats = policy._split_obs(obs)
        assert runner_feats[0, 0, 0].item() == 1.0
        assert runner_feats[0, 1, 0].item() == 2.0


# ── Different hyperparameter configurations ──────────────────────────────────


class TestHyperparamVariations:
    @pytest.mark.parametrize("lstm_hidden", [64, 128, 256, 512])
    def test_different_lstm_sizes(self, lstm_hidden: int):
        p = PPOLSTMPolicy(
            OBS_DIM, ACTION_DIM, MAX_RUNNERS,
            {"lstm_hidden_size": lstm_hidden, "mlp_hidden_size": 32, "mlp_layers": 1},
        )
        obs = torch.randn(1, OBS_DIM)
        out = p(obs)
        assert out.action_mean.shape == (1, ACTION_DIM)
        h, c = out.hidden_state
        assert h.shape == (1, 1, lstm_hidden)

    @pytest.mark.parametrize("mlp_layers", [1, 2, 3])
    def test_different_mlp_depths(self, mlp_layers: int):
        p = PPOLSTMPolicy(
            OBS_DIM, ACTION_DIM, MAX_RUNNERS,
            {"lstm_hidden_size": 64, "mlp_hidden_size": 32, "mlp_layers": mlp_layers},
        )
        obs = torch.randn(1, OBS_DIM)
        out = p(obs)
        assert out.action_mean.shape == (1, ACTION_DIM)

    @pytest.mark.parametrize("mlp_hidden", [64, 128, 256])
    def test_different_mlp_widths(self, mlp_hidden: int):
        p = PPOLSTMPolicy(
            OBS_DIM, ACTION_DIM, MAX_RUNNERS,
            {"lstm_hidden_size": 64, "mlp_hidden_size": mlp_hidden, "mlp_layers": 1},
        )
        obs = torch.randn(1, OBS_DIM)
        out = p(obs)
        assert out.action_mean.shape == (1, ACTION_DIM)


# ── Runner encoder sharing ──────────────────────────────────────────────────


class TestRunnerEncoderSharing:
    def test_shared_weights_produce_same_output(self, policy: PPOLSTMPolicy):
        """Identical runner inputs should produce identical embeddings."""
        runner_input = torch.randn(1, RUNNER_DIM)
        emb1 = policy.runner_encoder(runner_input)
        emb2 = policy.runner_encoder(runner_input)
        assert torch.allclose(emb1, emb2)

    def test_different_runners_produce_different_output(self, policy: PPOLSTMPolicy):
        r1 = torch.randn(1, RUNNER_DIM)
        r2 = torch.randn(1, RUNNER_DIM)
        emb1 = policy.runner_encoder(r1)
        emb2 = policy.runner_encoder(r2)
        assert not torch.allclose(emb1, emb2)


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_zero_observation(self, policy: PPOLSTMPolicy):
        """All-zero observation (padded runners) should not crash."""
        obs = torch.zeros(1, OBS_DIM)
        out = policy(obs)
        assert out.action_mean.shape == (1, ACTION_DIM)
        assert not torch.isnan(out.action_mean).any()
        assert not torch.isnan(out.value).any()

    def test_large_observation_values(self, policy: PPOLSTMPolicy):
        """Large inputs should not produce NaN (tests numerical stability)."""
        obs = torch.ones(1, OBS_DIM) * 100.0
        out = policy(obs)
        assert not torch.isnan(out.action_mean).any()
        assert not torch.isnan(out.value).any()

    def test_negative_observation_values(self, policy: PPOLSTMPolicy):
        obs = torch.ones(1, OBS_DIM) * -10.0
        out = policy(obs)
        assert not torch.isnan(out.action_mean).any()

    def test_sequence_length_one(self, policy: PPOLSTMPolicy):
        """Explicit 3-D input with seq_len=1."""
        obs = torch.randn(1, 1, OBS_DIM)
        out = policy(obs)
        assert out.action_mean.shape == (1, ACTION_DIM)

    def test_deterministic_with_same_seed(self, policy: PPOLSTMPolicy):
        """Same seed + same input → same output (determinism check)."""
        obs = torch.randn(1, OBS_DIM)
        torch.manual_seed(0)
        out1 = policy(obs)
        torch.manual_seed(0)
        out2 = policy(obs)
        assert torch.allclose(out1.action_mean, out2.action_mean)


# ── Parameter count sanity ───────────────────────────────────────────────────


class TestParameterCount:
    def test_has_parameters(self, policy: PPOLSTMPolicy):
        total = sum(p.numel() for p in policy.parameters())
        assert total > 0

    def test_all_parameters_require_grad(self, policy: PPOLSTMPolicy):
        for name, p in policy.named_parameters():
            assert p.requires_grad, f"{name} does not require grad"

    def test_parameter_count_reasonable(self, policy: PPOLSTMPolicy):
        """With small hidden sizes, param count should be moderate."""
        total = sum(p.numel() for p in policy.parameters())
        # Small config (lstm=64, mlp=32, layers=2): should be < 1M params
        assert total < 1_000_000, f"Too many params: {total}"
        assert total > 1_000, f"Too few params: {total}"
