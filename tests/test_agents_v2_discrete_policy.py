"""Tests for ``agents_v2.discrete_policy``.

Phase 1, Session 02 deliverable. The policy class produces:

    - masked categorical logits (sampling never lands on a masked idx)
    - per-runner value head (shape ``(batch, max_runners)``)
    - Beta stake parameters with α, β > 1 (unimodal)

Hidden-state pack/slice helpers mirror the v1 contract — load-bearing
for Phase 2's PPO update path.
"""

from __future__ import annotations

import pytest
import torch

from agents_v2.action_space import DiscreteActionSpace
from agents_v2.discrete_policy import (
    DiscreteLSTMPolicy,
    DiscretePolicyOutput,
)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def space() -> DiscreteActionSpace:
    return DiscreteActionSpace(max_runners=14)


@pytest.fixture
def policy(space):
    """Default-shape policy for batch / shape tests."""
    return DiscreteLSTMPolicy(
        obs_dim=64, action_space=space, hidden_size=32,
    )


# ── Forward shape contract ────────────────────────────────────────────────


class TestForwardShapes:
    def test_forward_shapes_batch_only(self, policy, space):
        batch = 4
        obs = torch.randn(batch, policy.obs_dim)
        out = policy(obs)
        assert isinstance(out, DiscretePolicyOutput)
        assert out.logits.shape == (batch, space.n)
        assert out.masked_logits.shape == (batch, space.n)
        assert out.value_per_runner.shape == (batch, space.max_runners)
        assert out.stake_alpha.shape == (batch,)
        assert out.stake_beta.shape == (batch,)
        # Hidden state shape: ``(num_layers, batch, hidden)``.
        h, c = out.new_hidden_state
        assert h.shape == (1, batch, policy.hidden_size)
        assert c.shape == (1, batch, policy.hidden_size)

    def test_forward_accepts_sequence_input(self, policy, space):
        batch, ctx = 4, 8
        obs = torch.randn(batch, ctx, policy.obs_dim)
        out = policy(obs)
        assert out.logits.shape == (batch, space.n)
        assert out.value_per_runner.shape == (batch, space.max_runners)

    def test_value_head_outputs_per_runner_not_scalar(self, policy):
        """Explicit shape guard — Phase 2's per-runner GAE depends on it.

        A regression that returned ``(batch, 1)`` would silently
        collapse Phase 2's credit-assignment story; this test fails
        loudly.
        """
        out = policy(torch.randn(3, policy.obs_dim))
        assert out.value_per_runner.shape == (3, policy.max_runners)
        assert out.value_per_runner.shape[1] >= 2

    def test_stake_parameters_strictly_greater_than_one(self, policy):
        out = policy(torch.randn(8, policy.obs_dim))
        # ``softplus(x) + 1`` is strictly > 1 for any finite ``x``.
        assert torch.all(out.stake_alpha > 1.0)
        assert torch.all(out.stake_beta > 1.0)

    def test_stake_distribution_samples_in_unit_interval(self, policy):
        """Sanity: Beta with α, β > 1 lives entirely in ``(0, 1)``."""
        out = policy(torch.randn(8, policy.obs_dim))
        dist = torch.distributions.Beta(out.stake_alpha, out.stake_beta)
        sample = dist.sample()
        assert torch.all(sample > 0.0)
        assert torch.all(sample < 1.0)


# ── Mask correctness ──────────────────────────────────────────────────────


class TestMaskedCategorical:
    def test_masked_logits_are_neg_inf_at_masked_indices(self, policy, space):
        obs = torch.randn(2, policy.obs_dim)
        mask = torch.ones(2, space.n, dtype=torch.bool)
        # Mask out every other action (keep NOOP=0 legal).
        for i in range(2):
            for k in range(2, space.n, 2):
                mask[i, k] = False
        out = policy(obs, mask=mask)
        assert out.masked_logits[~mask].isneginf().all()
        # Legal entries unchanged.
        assert torch.equal(
            out.masked_logits[mask],
            out.logits[mask],
        )

    def test_masked_categorical_assigns_zero_probability_to_masked_actions(
        self, policy, space,
    ):
        """Sample 1000× from a half-masked distribution; nothing illegal."""
        torch.manual_seed(0)
        mask = torch.ones(1, space.n, dtype=torch.bool)
        # Mask off every other action — NOOP stays legal.
        legal_set = set()
        for k in range(space.n):
            if k % 2 == 1 and k > 0:
                mask[0, k] = False
            else:
                legal_set.add(k)
        obs = torch.randn(1, policy.obs_dim)
        out = policy(obs, mask=mask)
        samples = out.action_dist.sample(torch.Size([1000])).flatten().tolist()
        for s in samples:
            assert s in legal_set, (
                f"sampled illegal action {s}; mask shouldn't allow it"
            )

    def test_mask_can_be_one_dim_and_broadcasts(self, policy, space):
        """Common rollout shape: a single (n,) mask for the whole batch."""
        obs = torch.randn(3, policy.obs_dim)
        mask = torch.ones(space.n, dtype=torch.bool)
        mask[5] = False
        out = policy(obs, mask=mask)
        assert out.masked_logits[:, 5].isneginf().all()

    def test_no_mask_means_all_logits_pass_through(self, policy):
        obs = torch.randn(3, policy.obs_dim)
        out = policy(obs, mask=None)
        assert torch.equal(out.masked_logits, out.logits)


# ── Hidden state pack / slice ─────────────────────────────────────────────


class TestHiddenState:
    def test_init_hidden_zero(self, policy):
        h, c = policy.init_hidden(batch=5)
        assert h.shape == (1, 5, policy.hidden_size)
        assert c.shape == (1, 5, policy.hidden_size)
        assert torch.all(h == 0.0)
        assert torch.all(c == 0.0)

    def test_pack_slice_round_trip_lstm(self, policy):
        """Pack 4 single-batch states, slice [0, 2], assert equality."""
        torch.manual_seed(7)
        states = [
            (
                torch.randn(1, 1, policy.hidden_size),
                torch.randn(1, 1, policy.hidden_size),
            )
            for _ in range(4)
        ]
        packed = DiscreteLSTMPolicy.pack_hidden_states(states)
        # Packed shape: (num_layers=1, batch=4, hidden).
        assert packed[0].shape == (1, 4, policy.hidden_size)
        assert packed[1].shape == (1, 4, policy.hidden_size)

        idx = torch.tensor([0, 2])
        sliced = DiscreteLSTMPolicy.slice_hidden_states(packed, idx)
        assert sliced[0].shape == (1, 2, policy.hidden_size)
        assert sliced[1].shape == (1, 2, policy.hidden_size)

        # Slot 0
        assert torch.allclose(sliced[0][:, 0:1, :], states[0][0])
        assert torch.allclose(sliced[1][:, 0:1, :], states[0][1])
        # Slot 1 of the slice == slot 2 of the original
        assert torch.allclose(sliced[0][:, 1:2, :], states[2][0])
        assert torch.allclose(sliced[1][:, 1:2, :], states[2][1])

    def test_forward_returns_non_zero_hidden_state(self, policy):
        """Smoke: a real forward pass produces a non-zero new hidden."""
        torch.manual_seed(0)
        out = policy(torch.randn(2, policy.obs_dim))
        h, c = out.new_hidden_state
        assert (h.abs().sum() + c.abs().sum()).item() > 0.0


# ── Backward / gradient flow ──────────────────────────────────────────────


class TestBackwardGradients:
    def test_backward_produces_gradients_on_all_params(self, policy):
        """Sum all heads, backward, every requires_grad param has .grad.

        Catches accidental ``.detach()`` on a head — the failure mode
        the v1 fill-prob plan tripped over (CLAUDE.md "fill_prob feeds
        actor_head" §"Do not detach").
        """
        out = policy(torch.randn(3, policy.obs_dim))
        loss = (
            out.logits.sum()
            + out.value_per_runner.sum()
            + out.stake_alpha.sum()
            + out.stake_beta.sum()
        )
        loss.backward()
        for name, p in policy.named_parameters():
            if not p.requires_grad:
                continue
            assert p.grad is not None, (
                f"param {name!r} got no gradient — head accidentally "
                f"detached?"
            )

    def test_value_head_gradient_independent_from_logits(self, policy):
        """Backward on value alone reaches value_head; logits_head untouched."""
        out = policy(torch.randn(3, policy.obs_dim))
        out.value_per_runner.sum().backward()
        assert policy.value_head.weight.grad is not None
        # ``logits_head`` doesn't feed into the value path, so its
        # gradient stays None.
        assert policy.logits_head.weight.grad is None
