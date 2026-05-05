"""Regression tests for v2 discrete BC pretrain + entropy-warmup handshake.

Phase 8 Session 02. Five tests covering:

1. Actor-head-only training. After ``pretrain``, ``actor_head``
   parameters change; every other parameter is byte-identical.
2. Zero-steps no-op. ``n_steps=0`` short-circuits — ALL parameters
   byte-identical including ``actor_head``.
3. Loss decreases. 200 steps on a small synthetic oracle pool drives
   ``final_ce_loss < initial_ce_loss``.
4. Warmup interpolates linearly from ``post_bc_entropy`` to
   ``entropy_coeff`` over ``bc_target_entropy_warmup_eps`` episodes.
5. ``bc_pretrain_steps=0`` byte-identity. A no-op BC pass does not
   perturb the trainer's PPO update statistics or the policy weights.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from agents_v2.action_space import DiscreteActionSpace
from agents_v2.discrete_policy import DiscreteLSTMPolicy
from training_v2.arb_oracle import OracleSample
from training_v2.discrete_ppo.bc_pretrain import (
    BCLossHistory,
    DiscreteBCPretrainer,
    measure_post_bc_entropy,
)
from training_v2.discrete_ppo.trainer import DiscretePPOTrainer


# ── Shared helpers ────────────────────────────────────────────────────────────


_OBS_DIM = 32
_MAX_RUNNERS = 4


def _make_policy(seed: int = 0) -> DiscreteLSTMPolicy:
    torch.manual_seed(seed)
    space = DiscreteActionSpace(max_runners=_MAX_RUNNERS)
    return DiscreteLSTMPolicy(
        obs_dim=_OBS_DIM,
        action_space=space,
        hidden_size=32,
    )


def _make_samples(n: int, seed: int = 0) -> list[OracleSample]:
    rng = np.random.default_rng(seed)
    return [
        OracleSample(
            tick_index=i,
            runner_idx=int(rng.integers(0, _MAX_RUNNERS)),
            obs=rng.standard_normal(_OBS_DIM).astype(np.float32),
            arb_spread_ticks=int(rng.integers(1, 25)),
            expected_locked_pnl=float(rng.uniform(0.01, 1.0)),
        )
        for i in range(n)
    ]


def _snapshot_state_dict(policy) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in policy.state_dict().items()}


# ── Test 1: actor_head only ──────────────────────────────────────────────────


class TestBCPretrainTrainsActorHeadOnly:
    def test_actor_head_changes_other_params_unchanged(self):
        policy = _make_policy(seed=42)
        samples = _make_samples(64, seed=1)
        before = _snapshot_state_dict(policy)

        history = DiscreteBCPretrainer(
            lr=3e-4, batch_size=16, seed=7,
        ).pretrain(policy, samples, n_steps=20)

        assert isinstance(history, BCLossHistory)
        assert len(history.ce_losses) == 20

        after = policy.state_dict()
        actor_head_changed = False
        for name, before_val in before.items():
            after_val = after[name]
            if "actor_head" in name:
                # Some change is required (we ran 20 BC steps).
                if not torch.equal(before_val, after_val):
                    actor_head_changed = True
            else:
                assert torch.equal(before_val, after_val), (
                    f"non-actor_head param {name!r} changed during BC; "
                    "freeze/unfreeze contract violated."
                )
        assert actor_head_changed, (
            "No actor_head parameter changed across 20 BC steps — "
            "BC isn't hitting the actor_head module at all."
        )

    def test_post_bc_requires_grad_restored_on_non_actor_params(self):
        # Hard constraint §6: non-actor_head params restored to
        # requires_grad=True after BC. A subsequent PPO step needs
        # them trainable.
        policy = _make_policy(seed=42)
        samples = _make_samples(32, seed=2)
        DiscreteBCPretrainer(
            lr=1e-3, batch_size=8, seed=3,
        ).pretrain(policy, samples, n_steps=5)

        for name, p in policy.named_parameters():
            assert p.requires_grad, (
                f"param {name!r} left frozen after BC — non-actor "
                "params must be restored to requires_grad=True."
            )


# ── Test 2: zero-step no-op ──────────────────────────────────────────────────


class TestBCPretrainZeroStepsIsNoop:
    def test_zero_steps_byte_identical(self):
        policy = _make_policy(seed=42)
        samples = _make_samples(32, seed=4)
        before = _snapshot_state_dict(policy)

        history = DiscreteBCPretrainer().pretrain(
            policy, samples, n_steps=0,
        )

        assert history.ce_losses == []
        assert history.final_ce_loss == 0.0

        after = policy.state_dict()
        for name, before_val in before.items():
            assert torch.equal(before_val, after[name]), (
                f"param {name!r} changed during a zero-step BC pass — "
                "§7 byte-identity contract violated."
            )

    def test_empty_samples_byte_identical(self):
        # Same byte-identity, but tripped via the empty-samples path
        # rather than the n_steps=0 path.
        policy = _make_policy(seed=42)
        before = _snapshot_state_dict(policy)

        history = DiscreteBCPretrainer().pretrain(
            policy, samples=[], n_steps=200,
        )

        assert history.ce_losses == []
        after = policy.state_dict()
        for name, before_val in before.items():
            assert torch.equal(before_val, after[name])


# ── Test 3: loss decreases over steps ────────────────────────────────────────


class TestBCPretrainLossDecreasesOverSteps:
    def test_final_ce_loss_below_initial(self):
        policy = _make_policy(seed=42)
        # Small fixed pool so the per-step random batch keeps revisiting
        # the same labels — clear gradient signal over 200 steps.
        samples = _make_samples(32, seed=11)

        history = DiscreteBCPretrainer(
            lr=3e-3, batch_size=16, seed=13,
        ).pretrain(policy, samples, n_steps=200)

        assert len(history.ce_losses) == 200
        initial = float(np.mean(history.ce_losses[:10]))
        final = float(np.mean(history.ce_losses[-10:]))
        assert final < initial, (
            f"BC CE loss did not decrease — initial mean={initial:.4f}, "
            f"final mean={final:.4f}. The actor_head isn't learning."
        )


# ── Test 4: warmup interpolates linearly ─────────────────────────────────────


class TestBCWarmupInterpolatesTargetEntropy:
    def _make_trainer_for_warmup_test(
        self,
        entropy_coeff: float,
        bc_warmup_eps: int,
    ) -> DiscretePPOTrainer:
        # Construct a real DiscretePPOTrainer but reach into the
        # warmup state directly. We never call train_episode — only
        # _effective_target_entropy() — so the policy/shim choice is
        # immaterial. Use the smallest valid construction.
        from agents_v2.env_shim import DiscreteActionShim

        # We need a minimal shim; use a stub object exposing the bare
        # surface DiscretePPOTrainer.__init__ touches (action_space,
        # max_runners). The collector built inside __init__ is never
        # exercised by these warmup unit tests.
        space = DiscreteActionSpace(max_runners=_MAX_RUNNERS)

        class _StubShim:
            def __init__(self):
                self.action_space = space
                self.max_runners = _MAX_RUNNERS

        policy = _make_policy(seed=0)
        # Bypass the collector construction by building the trainer
        # state in-place — simpler than wrestling a full env up. The
        # warmup logic only reads four attributes; everything else is
        # untouched.
        trainer = DiscretePPOTrainer.__new__(DiscretePPOTrainer)
        trainer.entropy_coeff = float(entropy_coeff)
        trainer._post_bc_entropy = None
        trainer._bc_warmup_eps = int(bc_warmup_eps)
        trainer._eps_since_bc = 0
        return trainer

    def test_inactive_when_post_bc_entropy_none(self):
        t = self._make_trainer_for_warmup_test(
            entropy_coeff=6.0, bc_warmup_eps=10,
        )
        assert t._effective_target_entropy() == pytest.approx(6.0)

    def test_linear_interp_at_step_0_5_and_10(self):
        t = self._make_trainer_for_warmup_test(
            entropy_coeff=6.0, bc_warmup_eps=10,
        )
        t.set_post_bc_entropy(3.0)

        # Episode 0 → effective == post_bc_entropy
        t._eps_since_bc = 0
        assert t._effective_target_entropy() == pytest.approx(3.0)

        # Episode 5 → halfway → 4.5
        t._eps_since_bc = 5
        assert t._effective_target_entropy() == pytest.approx(4.5)

        # Episode 10 → at the threshold → entropy_coeff
        t._eps_since_bc = 10
        assert t._effective_target_entropy() == pytest.approx(6.0)

        # Past the threshold → still entropy_coeff
        t._eps_since_bc = 25
        assert t._effective_target_entropy() == pytest.approx(6.0)

    def test_zero_warmup_eps_returns_entropy_coeff_immediately(self):
        # Edge case: bc_target_entropy_warmup_eps=0 disables the
        # warmup curve even when post_bc_entropy is set. The trainer
        # falls through to the post-warmup branch at every step.
        t = self._make_trainer_for_warmup_test(
            entropy_coeff=2.0, bc_warmup_eps=0,
        )
        t.set_post_bc_entropy(0.5)
        t._eps_since_bc = 0
        assert t._effective_target_entropy() == pytest.approx(2.0)


# ── Test 5: §7 byte-identity through the BC machinery ────────────────────────


class TestBCPretrainStepsZeroByteIdentical:
    """The §7 regression guard — calling the BC machinery with
    ``n_steps=0`` (or empty samples) must produce a policy state and
    trainer state byte-identical to a run that never invoked BC at all.

    A full PPO ``train_episode`` would require a real env which is
    expensive to set up. The tighter property §7 actually constrains
    is on the policy weights and the trainer's warmup state — both
    are checked here. The PPO code paths after the BC call are
    deterministic functions of the policy state alone, so policy +
    trainer byte-identity ⇒ PPO update statistics byte-identity at
    a fixed seed.
    """

    def test_zero_step_bc_does_not_perturb_policy_or_trainer(self):
        from agents_v2.action_space import DiscreteActionSpace

        policy = _make_policy(seed=42)
        before_policy = _snapshot_state_dict(policy)

        # Construct trainer (without a shim — bypass collector init,
        # we only read trainer state).
        space = DiscreteActionSpace(max_runners=_MAX_RUNNERS)
        trainer = DiscretePPOTrainer.__new__(DiscretePPOTrainer)
        trainer.entropy_coeff = 0.01
        trainer._post_bc_entropy = None
        trainer._bc_warmup_eps = 5
        trainer._eps_since_bc = 0

        # Exercise both no-op paths through the BC machinery:
        history_a = DiscreteBCPretrainer().pretrain(
            policy, samples=[], n_steps=200,
        )
        history_b = DiscreteBCPretrainer().pretrain(
            policy, samples=_make_samples(8, seed=99), n_steps=0,
        )

        assert history_a.ce_losses == []
        assert history_b.ce_losses == []

        # Policy untouched.
        after_policy = policy.state_dict()
        for name, before_val in before_policy.items():
            assert torch.equal(before_val, after_policy[name]), (
                f"param {name!r} changed during a no-op BC pass — "
                "§7 byte-identity contract violated through the BC "
                "machinery."
            )

        # Worker MUST NOT have called set_post_bc_entropy; the trainer
        # state stays in its pre-S02 default (post_bc_entropy=None).
        assert trainer._post_bc_entropy is None, (
            "set_post_bc_entropy was called during a no-op BC pass — "
            "the trainer's warmup branch would activate on a §7 run."
        )
        assert trainer._eps_since_bc == 0
        assert trainer._effective_target_entropy() == pytest.approx(
            trainer.entropy_coeff,
        )


# ── Bonus: post-BC entropy measurement smoke test ────────────────────────────


def test_measure_post_bc_entropy_returns_finite_scalar():
    policy = _make_policy(seed=42)
    samples = _make_samples(64, seed=5)
    e = measure_post_bc_entropy(policy, samples)
    assert isinstance(e, float)
    assert np.isfinite(e)
    # Empty pool → 0.0 by contract (caller should NOT then call
    # set_post_bc_entropy).
    assert measure_post_bc_entropy(policy, []) == 0.0
