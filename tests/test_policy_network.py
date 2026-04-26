"""Unit tests for agents/policy_network.py and agents/architecture_registry.py."""

from __future__ import annotations

import pytest
import torch

from agents.architecture_registry import REGISTRY, create_policy
from agents.policy_network import (
    MARKET_TOTAL_DIM,
    RUNNER_INPUT_DIM,
    BasePolicy,
    PolicyOutput,
    PPOLSTMPolicy,
    PPOTimeLSTMPolicy,
    PPOTransformerPolicy,
)
from env.betfair_env import (
    ACTIONS_PER_RUNNER,
    AGENT_STATE_DIM,
    MARKET_DIM,
    POSITION_DIM,
    RUNNER_DIM,
    RUNNER_KEYS,
    VELOCITY_DIM,
)


# ── Constants for tests ─────────────────────────────────────────────────────

MAX_RUNNERS = 14
OBS_DIM = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * MAX_RUNNERS) + AGENT_STATE_DIM + (POSITION_DIM * MAX_RUNNERS)
ACTION_DIM = MAX_RUNNERS * ACTIONS_PER_RUNNER  # action_signal + stake_fraction per runner


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
            create_policy("nonexistent", obs_dim=100, action_dim=42, max_runners=14)

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
        assert MARKET_DIM == 37  # 25 base + 6 race status + 6 market type/each-way

    def test_velocity_dim(self):
        assert VELOCITY_DIM == 11  # 6 base + 1 time_since_status_change + 4 market velocity (Session 2.8)

    def test_runner_dim(self):
        assert RUNNER_DIM == len(RUNNER_KEYS)

    def test_agent_state_dim(self):
        assert AGENT_STATE_DIM == 6  # +1 day_pnl_norm (Session 4.10)

    def test_position_dim(self):
        assert POSITION_DIM == 3  # back_exposure, lay_exposure, bet_count per runner

    def test_market_total_dim(self):
        assert MARKET_TOTAL_DIM == 54  # 37 + 11 + 6

    def test_obs_dim_matches_env(self):
        expected = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * MAX_RUNNERS) + AGENT_STATE_DIM + (POSITION_DIM * MAX_RUNNERS)
        assert OBS_DIM == expected


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

        # Check gradients exist on key actor components (not critic, not
        # log_std, not sibling aux heads).
        # - ``action_log_std`` only gets gradient when used in the
        #   distribution (not this synthetic scalar loss).
        # - ``fill_prob_head`` (scalping-active-management §02) and
        #   ``risk_head`` (scalping-active-management §03) share the
        #   backbone with the actor but are SEPARATE heads; they
        #   legitimately don't receive gradient from an actor-mean-only
        #   loss — see hard_constraints §8.
        for name, param in policy.named_parameters():
            if not param.requires_grad:
                continue
            if "critic" in name:
                continue
            if name == "action_log_std":
                continue
            if "fill_prob_head" in name:
                continue
            if "risk_head" in name:
                continue
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
        assert runner_feats.shape == (2, MAX_RUNNERS, RUNNER_INPUT_DIM)

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
        runner_input = torch.randn(1, RUNNER_INPUT_DIM)
        emb1 = policy.runner_encoder(runner_input)
        emb2 = policy.runner_encoder(runner_input)
        assert torch.allclose(emb1, emb2)

    def test_different_runners_produce_different_output(self, policy: PPOLSTMPolicy):
        r1 = torch.randn(1, RUNNER_INPUT_DIM)
        r2 = torch.randn(1, RUNNER_INPUT_DIM)
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


# ── fill-prob-in-actor (2026-04-26) ─────────────────────────────────────────
#
# Regression guards for the architectural change that feeds
# ``fill_prob_per_runner`` into ``actor_head`` so the per-runner action
# distribution can condition on the policy's own per-runner fill
# forecast. See plans/fill-prob-in-actor/hard_constraints.md §10 — the
# gradient-through check is load-bearing; without it, an accidental
# detach would silently re-introduce the cohort-O / cohort-O2 dead-end.


def _build_lstm_policy() -> PPOLSTMPolicy:
    return PPOLSTMPolicy(
        OBS_DIM, ACTION_DIM, MAX_RUNNERS,
        {"lstm_hidden_size": 64, "mlp_hidden_size": 32, "mlp_layers": 2},
    )


def _build_time_lstm_policy() -> PPOTimeLSTMPolicy:
    return PPOTimeLSTMPolicy(
        OBS_DIM, ACTION_DIM, MAX_RUNNERS,
        {"lstm_hidden_size": 64, "mlp_hidden_size": 32, "mlp_layers": 2},
    )


def _build_transformer_policy() -> PPOTransformerPolicy:
    return PPOTransformerPolicy(
        OBS_DIM, ACTION_DIM, MAX_RUNNERS,
        {
            "lstm_hidden_size": 64,
            "mlp_hidden_size": 32,
            "mlp_layers": 2,
            "transformer_ctx_ticks": 32,
            "transformer_n_layers": 1,
            "transformer_n_heads": 4,
        },
    )


class TestFillProbInActor:
    """``fill_prob_per_runner`` is concatenated into actor_input.

    Hard_constraints §3, §5, §10 in plans/fill-prob-in-actor.
    """

    # -- input-dim guards --------------------------------------------------

    def test_lstm_actor_input_includes_fill_prob(self):
        p = _build_lstm_policy()
        # actor_head is nn.Sequential — first Linear is index 0.
        # Width is runner_embed + backbone + 2 (one column each for
        # fill_prob and mature_prob — see mature-prob-head 2026-04-26).
        expected = p.runner_embed_dim + p.lstm_hidden_size + 2
        assert p.actor_head[0].weight.shape[1] == expected
        obs = torch.randn(2, OBS_DIM)
        out = p(obs)
        assert out.action_mean.shape == (2, ACTION_DIM)
        assert out.fill_prob_per_runner.shape == (2, MAX_RUNNERS)

    def test_time_lstm_actor_input_includes_fill_prob(self):
        p = _build_time_lstm_policy()
        expected = p.runner_embed_dim + p.lstm_hidden_size + 2
        assert p.actor_head[0].weight.shape[1] == expected
        obs = torch.randn(2, OBS_DIM)
        out = p(obs)
        assert out.action_mean.shape == (2, ACTION_DIM)
        assert out.fill_prob_per_runner.shape == (2, MAX_RUNNERS)

    def test_transformer_actor_input_includes_fill_prob(self):
        p = _build_transformer_policy()
        expected = p.runner_embed_dim + p.d_model + 2
        assert p.actor_head[0].weight.shape[1] == expected
        obs = torch.randn(2, OBS_DIM)
        out = p(obs)
        assert out.action_mean.shape == (2, ACTION_DIM)
        assert out.fill_prob_per_runner.shape == (2, MAX_RUNNERS)

    # -- gradient-through check (load-bearing) ----------------------------

    @staticmethod
    def _action_mean_depends_on_fill_prob_head(policy):
        """Return True iff perturbing fill_prob_head.weight changes
        ``action_mean`` for a fixed obs / hidden_state.

        This is the stricter version of the §10 gradient-through check:
        we don't just confirm the gradient is non-None (which a detach
        bug could mask if ``fill_prob_head`` ALSO gets a BCE-side
        gradient through ``fill_prob_per_runner`` directly) — we
        confirm the actor's forward output literally depends on the
        head's weights.
        """
        torch.manual_seed(0)
        obs = torch.randn(2, OBS_DIM)
        baseline = policy(obs).action_mean.detach().clone()

        # Perturb the head's weight in-place; no autograd needed for this
        # forward-only check.
        with torch.no_grad():
            policy.fill_prob_head.weight.add_(
                torch.randn_like(policy.fill_prob_head.weight) * 0.5,
            )

        perturbed = policy(obs).action_mean.detach().clone()
        return not torch.allclose(baseline, perturbed, atol=1e-7)

    def test_lstm_action_mean_depends_on_fill_prob_head_weights(self):
        p = _build_lstm_policy()
        assert self._action_mean_depends_on_fill_prob_head(p), (
            "Surrogate-loss path is detached: action_mean did not change "
            "when fill_prob_head.weight was perturbed."
        )

    def test_time_lstm_action_mean_depends_on_fill_prob_head_weights(self):
        p = _build_time_lstm_policy()
        assert self._action_mean_depends_on_fill_prob_head(p), (
            "Surrogate-loss path is detached: action_mean did not change "
            "when fill_prob_head.weight was perturbed."
        )

    def test_transformer_action_mean_depends_on_fill_prob_head_weights(self):
        p = _build_transformer_policy()
        assert self._action_mean_depends_on_fill_prob_head(p), (
            "Surrogate-loss path is detached: action_mean did not change "
            "when fill_prob_head.weight was perturbed."
        )

    # Backward-side: an actor-only loss MUST send gradient through
    # ``fill_prob_head`` (Hard_constraints §5).

    def _assert_actor_loss_grads_fill_prob_head(self, policy):
        obs = torch.randn(2, OBS_DIM)
        out = policy(obs)
        loss = out.action_mean.sum()
        loss.backward()
        assert policy.fill_prob_head.weight.grad is not None, (
            "fill_prob_head.weight has no gradient — surrogate path "
            "appears detached."
        )
        # Non-trivial gradient (the head's init is small but the
        # actor-loss path should produce a non-zero update).
        assert policy.fill_prob_head.weight.grad.abs().max() > 0.0

    def test_lstm_actor_loss_routes_grad_through_fill_prob_head(self):
        self._assert_actor_loss_grads_fill_prob_head(_build_lstm_policy())

    def test_time_lstm_actor_loss_routes_grad_through_fill_prob_head(self):
        self._assert_actor_loss_grads_fill_prob_head(_build_time_lstm_policy())

    def test_transformer_actor_loss_routes_grad_through_fill_prob_head(self):
        self._assert_actor_loss_grads_fill_prob_head(_build_transformer_policy())

    # -- cross-load failure (Hard_constraints §4) -------------------------
    #
    # Pre-plan checkpoints had ``actor_head[0].weight`` shape
    # ``(hidden, runner_embed + lstm_hidden)``. The new architecture
    # expects ``(hidden, runner_embed + lstm_hidden + 1)``. Loading an
    # old state_dict with strict=True must FAIL with a shape-mismatch
    # mentioning ``actor_head``.
    #
    # The variant identity is carried by the changed weight shape; we
    # don't add a new explicit version field, we confirm the existing
    # PyTorch state_dict checking observes it.

    @staticmethod
    def _make_pre_plan_state_dict(policy, old_extra_dim: int) -> dict:
        """Build a state_dict matching ``policy``'s shapes EXCEPT for
        ``actor_head.0.weight`` / ``.bias``, which carry the pre-plan
        input width.

        ``old_extra_dim`` is how many post-plan columns to drop. After
        mature-prob-head (2026-04-26) the new actor_head[0] is two
        columns wider than pre-fill-prob (one for fill_prob, one for
        mature_prob), so this test fixes ``old_extra_dim=2`` to mimic
        the actual pre-fill-prob shape rather than the post-fill /
        pre-mature shape (which is exercised separately by
        ``TestMatureProbInActor``).
        """
        sd = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        old_w = sd["actor_head.0.weight"]
        new_in = old_w.shape[1]  # == runner_embed + backbone + 2
        old_in = new_in - old_extra_dim  # pre-fill-prob width
        # Replace with a tensor matching the pre-plan shape.
        sd["actor_head.0.weight"] = torch.zeros(old_w.shape[0], old_in)
        # Bias shape is unchanged (output dim == hidden_dim) — kept as-is.
        return sd

    def _assert_pre_plan_load_fails(self, policy):
        sd = self._make_pre_plan_state_dict(policy, old_extra_dim=2)
        with pytest.raises((RuntimeError, ValueError)) as excinfo:
            policy.load_state_dict(sd, strict=True)
        msg = str(excinfo.value).lower()
        assert "actor_head" in msg or "size mismatch" in msg or "shape" in msg, (
            f"Expected a shape-mismatch on actor_head, got: {excinfo.value}"
        )

    def test_lstm_pre_plan_weights_fail_to_load(self):
        self._assert_pre_plan_load_fails(_build_lstm_policy())

    def test_time_lstm_pre_plan_weights_fail_to_load(self):
        self._assert_pre_plan_load_fails(_build_time_lstm_policy())

    def test_transformer_pre_plan_weights_fail_to_load(self):
        self._assert_pre_plan_load_fails(_build_transformer_policy())


class TestMatureProbInActor:
    """``mature_prob_per_runner`` is concatenated into actor_input.

    Mirrors ``TestFillProbInActor`` for the new strict-label head added
    by mature-prob-head (2026-04-26). The mature-prob head is wired
    alongside fill-prob; ``actor_head[0].weight`` therefore widens by
    one further column (runner_embed + backbone + 2). See
    plans/per-runner-credit/findings.md for why this head exists.
    """

    # -- input-dim guards -------------------------------------------------

    def test_lstm_actor_input_includes_mature_prob(self):
        p = _build_lstm_policy()
        expected = p.runner_embed_dim + p.lstm_hidden_size + 2
        assert p.actor_head[0].weight.shape[1] == expected
        obs = torch.randn(2, OBS_DIM)
        out = p(obs)
        assert out.action_mean.shape == (2, ACTION_DIM)
        assert out.mature_prob_per_runner.shape == (2, MAX_RUNNERS)
        # Sigmoid output bounded at construction time (init produces
        # ≈0 logits → ≈0.5 probs); we only check the shape + range
        # invariants here so the orthogonal-init randomness doesn't
        # make this seed-fragile.
        assert torch.all(out.mature_prob_per_runner >= 0.0)
        assert torch.all(out.mature_prob_per_runner <= 1.0)

    def test_time_lstm_actor_input_includes_mature_prob(self):
        p = _build_time_lstm_policy()
        expected = p.runner_embed_dim + p.lstm_hidden_size + 2
        assert p.actor_head[0].weight.shape[1] == expected
        obs = torch.randn(2, OBS_DIM)
        out = p(obs)
        assert out.action_mean.shape == (2, ACTION_DIM)
        assert out.mature_prob_per_runner.shape == (2, MAX_RUNNERS)
        assert torch.all(out.mature_prob_per_runner >= 0.0)
        assert torch.all(out.mature_prob_per_runner <= 1.0)

    def test_transformer_actor_input_includes_mature_prob(self):
        p = _build_transformer_policy()
        expected = p.runner_embed_dim + p.d_model + 2
        assert p.actor_head[0].weight.shape[1] == expected
        obs = torch.randn(2, OBS_DIM)
        out = p(obs)
        assert out.action_mean.shape == (2, ACTION_DIM)
        assert out.mature_prob_per_runner.shape == (2, MAX_RUNNERS)
        assert torch.all(out.mature_prob_per_runner >= 0.0)
        assert torch.all(out.mature_prob_per_runner <= 1.0)

    # -- gradient-through check (load-bearing) ---------------------------

    @staticmethod
    def _action_mean_depends_on_mature_prob_head(policy):
        """Return True iff perturbing mature_prob_head.weight changes
        ``action_mean`` for a fixed obs / hidden_state.

        Mirror of ``TestFillProbInActor._action_mean_depends_on_fill_prob_head``
        — the strict forward-only check. A detach in the actor-input
        concat would let the head still receive gradient via its own
        BCE auxiliary while silently severing the surrogate-loss
        pathway; this check catches that.
        """
        torch.manual_seed(0)
        obs = torch.randn(2, OBS_DIM)
        baseline = policy(obs).action_mean.detach().clone()

        with torch.no_grad():
            policy.mature_prob_head.weight.add_(
                torch.randn_like(policy.mature_prob_head.weight) * 0.5,
            )

        perturbed = policy(obs).action_mean.detach().clone()
        return not torch.allclose(baseline, perturbed, atol=1e-7)

    def test_lstm_action_mean_depends_on_mature_prob_head_weights(self):
        p = _build_lstm_policy()
        assert self._action_mean_depends_on_mature_prob_head(p), (
            "Surrogate-loss path is detached: action_mean did not change "
            "when mature_prob_head.weight was perturbed."
        )

    def test_time_lstm_action_mean_depends_on_mature_prob_head_weights(self):
        p = _build_time_lstm_policy()
        assert self._action_mean_depends_on_mature_prob_head(p), (
            "Surrogate-loss path is detached: action_mean did not change "
            "when mature_prob_head.weight was perturbed."
        )

    def test_transformer_action_mean_depends_on_mature_prob_head_weights(self):
        p = _build_transformer_policy()
        assert self._action_mean_depends_on_mature_prob_head(p), (
            "Surrogate-loss path is detached: action_mean did not change "
            "when mature_prob_head.weight was perturbed."
        )

    # Backward-side: an actor-only loss MUST send gradient through
    # ``mature_prob_head``.

    def _assert_actor_loss_grads_mature_prob_head(self, policy):
        obs = torch.randn(2, OBS_DIM)
        out = policy(obs)
        loss = out.action_mean.sum()
        loss.backward()
        assert policy.mature_prob_head.weight.grad is not None, (
            "mature_prob_head.weight has no gradient — surrogate path "
            "appears detached."
        )
        assert policy.mature_prob_head.weight.grad.abs().max() > 0.0

    def test_lstm_actor_loss_routes_grad_through_mature_prob_head(self):
        self._assert_actor_loss_grads_mature_prob_head(_build_lstm_policy())

    def test_time_lstm_actor_loss_routes_grad_through_mature_prob_head(self):
        self._assert_actor_loss_grads_mature_prob_head(
            _build_time_lstm_policy(),
        )

    def test_transformer_actor_loss_routes_grad_through_mature_prob_head(self):
        self._assert_actor_loss_grads_mature_prob_head(
            _build_transformer_policy(),
        )

    # -- cross-load failure (architecture-hash break) --------------------
    #
    # Pre-mature-prob-head checkpoints had ``actor_head[0].weight``
    # shape ``(hidden, runner_embed + backbone + 1)`` (post-fill-prob
    # but pre-mature). The new architecture expects
    # ``(hidden, runner_embed + backbone + 2)``. Loading an old
    # state_dict with strict=True must FAIL with a shape-mismatch.
    #
    # The variant identity is carried by the changed weight shape; we
    # don't add a new explicit version field, we confirm PyTorch's
    # state_dict checking observes it (same pattern as
    # ``TestFillProbInActor._assert_pre_plan_load_fails``).

    @staticmethod
    def _make_pre_mature_state_dict(policy) -> dict:
        """Build a state_dict matching ``policy``'s shapes EXCEPT for
        ``actor_head.0.weight``, which carries the pre-mature-plan
        input width (one column narrower; the post-fill-prob shape).
        """
        sd = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        old_w = sd["actor_head.0.weight"]
        new_in = old_w.shape[1]  # == runner_embed + backbone + 2
        old_in = new_in - 1  # pre-mature width (still has fill_prob)
        sd["actor_head.0.weight"] = torch.zeros(old_w.shape[0], old_in)
        return sd

    def _assert_pre_mature_load_fails(self, policy):
        sd = self._make_pre_mature_state_dict(policy)
        with pytest.raises((RuntimeError, ValueError)) as excinfo:
            policy.load_state_dict(sd, strict=True)
        msg = str(excinfo.value).lower()
        assert "actor_head" in msg or "size mismatch" in msg or "shape" in msg, (
            f"Expected a shape-mismatch on actor_head, got: {excinfo.value}"
        )

    def test_lstm_pre_mature_weights_fail_to_load(self):
        self._assert_pre_mature_load_fails(_build_lstm_policy())

    def test_time_lstm_pre_mature_weights_fail_to_load(self):
        self._assert_pre_mature_load_fails(_build_time_lstm_policy())

    def test_transformer_pre_mature_weights_fail_to_load(self):
        self._assert_pre_mature_load_fails(_build_transformer_policy())
