"""Regression guard: direction_prob_head respects active runner-dim.

2026-05-24. Bug background:

The Phase-15 cohort 1779613306 trained 16 days at
direction_prob_loss_weight ≥ 0.515 but the head's BCE never moved
off the pos-weighted random-uniform-0.5 floor. The root cause —
diagnosed via tools/backbone_signal_probe.py and direct inspection
of the trained agent's weights — was that the policy hard-coded
``RUNNER_DIM = 143`` (full obs) for both the head's LayerNorm/Linear
input dim AND the runner-block slicing in forward(). Under lean-obs
(``--predictor-lean-obs``, per-runner block = 23 dims) the slicing
fell into a zero-pad test-mode fallback, producing structurally
garbage head input.

These tests enforce:

  §1 Lean-obs construction: head's first layer dims match
     runner_dim=23 when caller passes runner_dim=23.
  §2 Full-obs default: no runner_dim kwarg → 143 (back-compat).
  §3 Runner-block size matches the env's actual layout under lean
     obs (no test-mode fallback in production-sized obs).
  §4 Gradient flow: BCE loss produces non-None gradients on every
     head weight under lean-obs.
  §5 load_state_dict refuses pre-fix weights (with LayerNorm(143))
     into a lean-obs policy (LayerNorm(23)) — strict shape check.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from agents_v2.action_space import DiscreteActionSpace
from agents_v2.discrete_policy import DiscreteLSTMPolicy
from env.betfair_env import LEAN_RUNNER_DIM, RUNNER_DIM


MAX_RUNNERS = 14
LEAN_OBS_DIM = 574    # = 56 + 37 * MAX_RUNNERS (env.observation_space)
FULL_OBS_DIM = 2254   # = 56 + 157 * MAX_RUNNERS (approx; not exact for tests)


@pytest.fixture
def action_space() -> DiscreteActionSpace:
    return DiscreteActionSpace(max_runners=MAX_RUNNERS)


class TestDirectionHeadRunnerDim:

    def test_lean_obs_runner_dim_sizes_head_correctly(self, action_space):
        """§1: head's LayerNorm + first Linear sized to 23 when
        runner_dim=23 is passed."""
        p = DiscreteLSTMPolicy(
            obs_dim=LEAN_OBS_DIM,
            action_space=action_space,
            hidden_size=128,
            runner_dim=LEAN_RUNNER_DIM,  # 23
        )
        # direction_prob_head is nn.Sequential(LayerNorm, Linear, ReLU, Linear)
        layer_norm = p.direction_prob_head[0]
        first_linear = p.direction_prob_head[1]
        assert layer_norm.normalized_shape == (LEAN_RUNNER_DIM,)
        assert first_linear.in_features == LEAN_RUNNER_DIM

    def test_full_obs_default_keeps_back_compat(self, action_space):
        """§2: no runner_dim kwarg → head sized for RUNNER_DIM=143."""
        p = DiscreteLSTMPolicy(
            obs_dim=FULL_OBS_DIM,
            action_space=action_space,
            hidden_size=128,
        )
        assert p._runner_dim == RUNNER_DIM == 143
        layer_norm = p.direction_prob_head[0]
        assert layer_norm.normalized_shape == (RUNNER_DIM,)
        assert p.direction_prob_head[1].in_features == RUNNER_DIM

    def test_lean_obs_runner_block_matches_env_layout(self, action_space):
        """§3: under lean obs, the runner-block slice in forward()
        uses the production layout (no zero-pad fallback)."""
        p = DiscreteLSTMPolicy(
            obs_dim=LEAN_OBS_DIM,
            action_space=action_space,
            hidden_size=128,
            runner_dim=LEAN_RUNNER_DIM,
        )
        # MARKET_DIM (37) + VELOCITY_DIM (11) = 48 offset
        assert p._runner_block_offset == 48
        assert p._runner_block_size == MAX_RUNNERS * LEAN_RUNNER_DIM == 322
        assert p._runner_block_full_size == 322
        # Test-mode fallback NOT triggered (size == full_size).
        assert p._runner_block_size == p._runner_block_full_size

    def test_lean_obs_gradient_flow_through_direction_head(
        self, action_space,
    ):
        """§4: BCE on direction logits produces non-None gradients
        on ALL direction_prob_head parameters under lean obs."""
        p = DiscreteLSTMPolicy(
            obs_dim=LEAN_OBS_DIM,
            action_space=action_space,
            hidden_size=128,
            runner_dim=LEAN_RUNNER_DIM,
        )
        obs = torch.randn(2, LEAN_OBS_DIM)
        out = p(obs)
        label_back = torch.zeros(2, MAX_RUNNERS)
        label_back[0, 3] = 1.0
        loss = F.binary_cross_entropy_with_logits(
            out.direction_back_logits_per_runner, label_back,
        )
        loss.backward()
        for name, param in p.direction_prob_head.named_parameters():
            assert param.grad is not None, (
                f"direction_prob_head.{name}.grad is None — gradient "
                "didn't flow back through the head"
            )
            assert torch.isfinite(param.grad).all(), (
                f"direction_prob_head.{name}.grad has non-finite values"
            )

    def test_pre_fix_weights_fail_to_load_into_lean_policy(
        self, action_space,
    ):
        """§5: a state_dict from a pre-fix policy (LayerNorm(143))
        must fail strict load into a lean-obs policy (LayerNorm(23))
        — architecture-hash break is the correct behaviour."""
        # Build a "pre-fix" policy on full-obs sizing (143).
        p_old = DiscreteLSTMPolicy(
            obs_dim=FULL_OBS_DIM,
            action_space=action_space,
            hidden_size=128,
            # No runner_dim → default RUNNER_DIM=143
        )
        old_state = p_old.state_dict()
        # Build a "new" policy on lean-obs sizing (23).
        p_new = DiscreteLSTMPolicy(
            obs_dim=LEAN_OBS_DIM,
            action_space=action_space,
            hidden_size=128,
            runner_dim=LEAN_RUNNER_DIM,
        )
        # Strict load must raise — at minimum on
        # direction_prob_head.0.weight (LayerNorm 143 vs 23) and
        # direction_prob_head.1.weight (Linear (64, 143) vs (64, 23)).
        with pytest.raises(RuntimeError):
            p_new.load_state_dict(old_state, strict=True)

    def test_lean_obs_forward_returns_correct_shapes(self, action_space):
        """Sanity: forward returns the documented (batch, max_runners)
        shapes for direction outputs under lean obs."""
        p = DiscreteLSTMPolicy(
            obs_dim=LEAN_OBS_DIM,
            action_space=action_space,
            hidden_size=128,
            runner_dim=LEAN_RUNNER_DIM,
        )
        obs = torch.randn(3, LEAN_OBS_DIM)
        out = p(obs)
        assert out.direction_back_logits_per_runner.shape == (
            3, MAX_RUNNERS,
        )
        assert out.direction_lay_logits_per_runner.shape == (
            3, MAX_RUNNERS,
        )
        assert out.direction_back_prob_per_runner.shape == (
            3, MAX_RUNNERS,
        )
        assert out.direction_lay_prob_per_runner.shape == (
            3, MAX_RUNNERS,
        )

    def test_invalid_runner_dim_raises(self, action_space):
        """Defensive: runner_dim ≤ 0 raises at construction time."""
        with pytest.raises(ValueError, match="runner_dim must be positive"):
            DiscreteLSTMPolicy(
                obs_dim=LEAN_OBS_DIM,
                action_space=action_space,
                hidden_size=128,
                runner_dim=0,
            )
        with pytest.raises(ValueError, match="runner_dim must be positive"):
            DiscreteLSTMPolicy(
                obs_dim=LEAN_OBS_DIM,
                action_space=action_space,
                hidden_size=128,
                runner_dim=-5,
            )
