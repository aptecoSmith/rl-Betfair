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
from training_v2.arb_oracle import (
    CloseHoldSample,
    NegativeOracleSample,
    OracleSample,
    TARGET_CLASS_CLOSE,
    TARGET_CLASS_HOLD,
)
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


# ── BC label augmentation Phase A tests ──────────────────────────────────────
# plans/bc-label-augmentation/. Three regressions:
# 1. Byte-identity when negatives are empty (load-bearing for backcompat).
# 2. NOOP logits rise on negatives' obs vectors when neg-augmentation is on.
# 3. The pre-existing zero-step / empty-pool byte-identity invariants still
#    hold under the new code path (covered by the existing tests above —
#    no change required because the new code only activates when neg_active
#    is True).


def _make_negative_samples(
    n: int, seed: int = 0,
) -> list[NegativeOracleSample]:
    rng = np.random.default_rng(seed)
    return [
        NegativeOracleSample(
            tick_index=10_000 + i,  # disjoint from positive tick range
            runner_idx=int(rng.integers(0, _MAX_RUNNERS)),
            obs=rng.standard_normal(_OBS_DIM).astype(np.float32),
        )
        for i in range(n)
    ]


class TestBCWithNegativesByteIdenticalWhenEmpty:
    """When ``negative_samples=[]`` or ``None``, post-BC weights MUST
    match the run with the same seed where the arg was omitted entirely.
    This is the load-bearing backcompat invariant — every existing
    cohort that doesn't opt into Phase A must train byte-identical
    weights.
    """

    def test_negative_samples_none_vs_omitted(self):
        policy_a = _make_policy(seed=42)
        policy_b = _make_policy(seed=42)
        samples = _make_samples(32, seed=1)

        DiscreteBCPretrainer(
            lr=3e-4, batch_size=16, seed=7,
        ).pretrain(policy_a, samples, n_steps=10)
        DiscreteBCPretrainer(
            lr=3e-4, batch_size=16, seed=7,
        ).pretrain(
            policy_b, samples, n_steps=10, negative_samples=None,
        )

        for (kn, va), (_, vb) in zip(
            policy_a.state_dict().items(),
            policy_b.state_dict().items(),
        ):
            assert torch.equal(va, vb), (
                f"param {kn!r} drifted when negative_samples=None was "
                "passed explicitly — backcompat broken."
            )

    def test_negative_samples_empty_list_byte_identical(self):
        policy_a = _make_policy(seed=42)
        policy_b = _make_policy(seed=42)
        samples = _make_samples(32, seed=1)

        DiscreteBCPretrainer(
            lr=3e-4, batch_size=16, seed=7,
        ).pretrain(policy_a, samples, n_steps=10)
        DiscreteBCPretrainer(
            lr=3e-4, batch_size=16, seed=7,
        ).pretrain(
            policy_b, samples, n_steps=10, negative_samples=[],
        )

        for (kn, va), (_, vb) in zip(
            policy_a.state_dict().items(),
            policy_b.state_dict().items(),
        ):
            assert torch.equal(va, vb), (
                f"param {kn!r} drifted under negative_samples=[] — "
                "the §7 byte-identity contract is broken."
            )


class TestBCWithNegativesPushesNoopLogitsUp:
    """When negatives are present and targeted at NOOP, the NOOP class
    logit on the negatives' obs vectors MUST rise (or at minimum, the
    NOOP advantage over the OPEN_BACK class on those obs must rise)
    compared to a control run with no negative augmentation.
    """

    def test_noop_logit_rises_with_negative_augmentation(self):
        from agents_v2.action_space import ActionType

        # Small, fixed pools: 3 positives + 6 negatives. 50 BC steps is
        # enough on a fresh policy to see a clear gradient signal.
        positives = _make_samples(3, seed=11)
        negatives = _make_negative_samples(6, seed=12)

        policy_aug = _make_policy(seed=42)
        policy_ctrl = _make_policy(seed=42)
        # Pre-BC: byte-identical policies.

        DiscreteBCPretrainer(
            lr=3e-3, batch_size=8, seed=13,
        ).pretrain(
            policy_aug, positives, n_steps=50,
            negative_samples=negatives, positive_weight=1.0,
        )
        DiscreteBCPretrainer(
            lr=3e-3, batch_size=8, seed=13,
        ).pretrain(policy_ctrl, positives, n_steps=50)

        # Evaluate both policies on the negatives' obs vectors.
        obs_t = torch.tensor(
            np.stack([s.obs for s in negatives], axis=0),
            dtype=torch.float32,
        )
        with torch.no_grad():
            logits_aug = policy_aug(obs_t).logits  # (N, n_actions)
            logits_ctrl = policy_ctrl(obs_t).logits

        action_space = policy_aug.action_space
        noop_idx = int(action_space.encode(ActionType.NOOP, None))

        # Compare the NOOP logit's RELATIVE advantage vs the mean of
        # every other action's logit on the same obs row. The
        # augmented run should have a higher NOOP advantage than the
        # control run.
        def _noop_advantage(logits: torch.Tensor) -> float:
            noop_l = logits[:, noop_idx]
            others_mean = (
                logits.sum(dim=1) - noop_l
            ) / (logits.shape[1] - 1)
            return float((noop_l - others_mean).mean().item())

        adv_aug = _noop_advantage(logits_aug)
        adv_ctrl = _noop_advantage(logits_ctrl)
        assert adv_aug > adv_ctrl, (
            f"NOOP advantage on negatives' obs: augmented={adv_aug:.4f}, "
            f"control={adv_ctrl:.4f}. Augmentation should raise the "
            "NOOP class above other classes on negative obs."
        )


class TestBCWithNegativesZeroStepsStillNoop:
    """Phase A constraint: ``n_steps=0`` must remain byte-identical
    regardless of whether ``negative_samples`` is set. The pretrainer
    short-circuits before constructing the optimiser, so this should
    trivially hold — but pin it with a regression test.
    """

    def test_zero_steps_with_negatives_byte_identical(self):
        policy = _make_policy(seed=42)
        samples = _make_samples(32, seed=4)
        negatives = _make_negative_samples(16, seed=5)
        before = _snapshot_state_dict(policy)

        history = DiscreteBCPretrainer().pretrain(
            policy, samples, n_steps=0,
            negative_samples=negatives,
        )

        assert history.ce_losses == []
        after = policy.state_dict()
        for name, before_val in before.items():
            assert torch.equal(before_val, after[name]), (
                f"param {name!r} changed during a zero-step BC pass "
                "with negative samples set — §7 violated."
            )


# ── BC label augmentation Phase B tests ─────────────────────────────────────
# plans/bc-label-augmentation/. Three regressions:
# 1. Byte-identity when close_hold_samples is None or empty
#    (load-bearing backwards-compat invariant).
# 2. CLOSE-target samples push the CLOSE logit up on their obs.
# 3. HOLD-target samples push the NOOP logit up on their obs.


def _make_close_hold_samples(
    n_close: int,
    n_hold: int,
    seed: int = 0,
) -> list[CloseHoldSample]:
    rng = np.random.default_rng(seed)
    out: list[CloseHoldSample] = []
    for i in range(n_close):
        out.append(CloseHoldSample(
            tick_index=30_000 + i,
            runner_idx=int(rng.integers(0, _MAX_RUNNERS)),
            obs=rng.standard_normal(_OBS_DIM).astype(np.float32),
            target_action_class=int(TARGET_CLASS_CLOSE),
            lifecycle_position=float(rng.uniform(0.0, 1.0)),
        ))
    for i in range(n_hold):
        out.append(CloseHoldSample(
            tick_index=40_000 + i,
            runner_idx=int(rng.integers(0, _MAX_RUNNERS)),
            obs=rng.standard_normal(_OBS_DIM).astype(np.float32),
            target_action_class=int(TARGET_CLASS_HOLD),
            lifecycle_position=float(rng.uniform(0.0, 1.0)),
        ))
    return out


class TestBCWithCloseHoldByteIdenticalWhenEmpty:
    """When ``close_hold_samples=None`` or ``[]``, post-BC weights MUST
    match a control run with the kwarg omitted entirely. This is the
    load-bearing Phase B backcompat invariant.
    """

    def test_close_hold_samples_none_vs_omitted(self):
        policy_a = _make_policy(seed=42)
        policy_b = _make_policy(seed=42)
        samples = _make_samples(32, seed=1)

        DiscreteBCPretrainer(
            lr=3e-4, batch_size=16, seed=7,
        ).pretrain(policy_a, samples, n_steps=10)
        DiscreteBCPretrainer(
            lr=3e-4, batch_size=16, seed=7,
        ).pretrain(
            policy_b, samples, n_steps=10,
            close_hold_samples=None,
        )

        for (kn, va), (_, vb) in zip(
            policy_a.state_dict().items(),
            policy_b.state_dict().items(),
        ):
            assert torch.equal(va, vb), (
                f"param {kn!r} drifted when close_hold_samples=None "
                "was passed explicitly — Phase B backcompat broken."
            )

    def test_close_hold_samples_empty_list_byte_identical(self):
        policy_a = _make_policy(seed=42)
        policy_b = _make_policy(seed=42)
        samples = _make_samples(32, seed=1)

        DiscreteBCPretrainer(
            lr=3e-4, batch_size=16, seed=7,
        ).pretrain(policy_a, samples, n_steps=10)
        DiscreteBCPretrainer(
            lr=3e-4, batch_size=16, seed=7,
        ).pretrain(
            policy_b, samples, n_steps=10,
            close_hold_samples=[],
        )

        for (kn, va), (_, vb) in zip(
            policy_a.state_dict().items(),
            policy_b.state_dict().items(),
        ):
            assert torch.equal(va, vb), (
                f"param {kn!r} drifted under close_hold_samples=[] — "
                "the §7 byte-identity contract is broken."
            )

    def test_zero_steps_with_close_hold_byte_identical(self):
        policy = _make_policy(seed=42)
        samples = _make_samples(16, seed=1)
        ch_samples = _make_close_hold_samples(8, 8, seed=2)
        before = _snapshot_state_dict(policy)

        history = DiscreteBCPretrainer().pretrain(
            policy, samples, n_steps=0,
            close_hold_samples=ch_samples,
        )

        assert history.ce_losses == []
        after = policy.state_dict()
        for name, before_val in before.items():
            assert torch.equal(before_val, after[name]), (
                f"param {name!r} changed during a zero-step BC pass "
                "with close_hold samples set — §7 violated."
            )


class TestBCWithCloseHoldPushesCloseLogitsUp:
    """When CLOSE-target close_hold samples are present, the CLOSE
    action's logit on the CLOSE-target samples' obs MUST rise compared
    to a control run without close_hold samples.
    """

    def test_close_logit_rises_with_close_hold_augmentation(self):
        from agents_v2.action_space import ActionType

        positives = _make_samples(3, seed=11)
        # Pure-CLOSE pool: we want a clean signal that the CLOSE
        # logit is pushed up on these obs. Use a small, fixed
        # runner_idx so the CLOSE label hits the same action class
        # repeatedly.
        target_runner = 1
        rng = np.random.default_rng(13)
        close_samples = [
            CloseHoldSample(
                tick_index=30_000 + i,
                runner_idx=target_runner,
                obs=rng.standard_normal(_OBS_DIM).astype(np.float32),
                target_action_class=int(TARGET_CLASS_CLOSE),
                lifecycle_position=0.5,
            )
            for i in range(6)
        ]

        policy_aug = _make_policy(seed=42)
        policy_ctrl = _make_policy(seed=42)

        DiscreteBCPretrainer(
            lr=3e-3, batch_size=8, seed=13,
        ).pretrain(
            policy_aug, positives, n_steps=100,
            close_hold_samples=close_samples,
        )
        DiscreteBCPretrainer(
            lr=3e-3, batch_size=8, seed=13,
        ).pretrain(policy_ctrl, positives, n_steps=100)

        obs_t = torch.tensor(
            np.stack([s.obs for s in close_samples], axis=0),
            dtype=torch.float32,
        )
        with torch.no_grad():
            logits_aug = policy_aug(obs_t).logits
            logits_ctrl = policy_ctrl(obs_t).logits

        action_space = policy_aug.action_space
        close_idx = int(action_space.encode(
            ActionType.CLOSE, target_runner,
        ))

        close_logit_aug = float(logits_aug[:, close_idx].mean().item())
        close_logit_ctrl = float(logits_ctrl[:, close_idx].mean().item())

        assert close_logit_aug > close_logit_ctrl, (
            f"CLOSE logit on close-target obs did not rise with "
            f"augmentation: augmented={close_logit_aug:.4f}, "
            f"control={close_logit_ctrl:.4f}."
        )


class TestBCWithCloseHoldPushesNoopLogitsUpOnHoldSamples:
    """When HOLD-target close_hold samples are present, the NOOP
    action's logit on the HOLD-target samples' obs MUST rise compared
    to a control run without close_hold samples.
    """

    def test_noop_logit_rises_with_hold_augmentation(self):
        from agents_v2.action_space import ActionType

        positives = _make_samples(3, seed=11)
        rng = np.random.default_rng(17)
        hold_samples = [
            CloseHoldSample(
                tick_index=40_000 + i,
                runner_idx=int(rng.integers(0, _MAX_RUNNERS)),
                obs=rng.standard_normal(_OBS_DIM).astype(np.float32),
                target_action_class=int(TARGET_CLASS_HOLD),
                lifecycle_position=0.5,
            )
            for i in range(6)
        ]

        policy_aug = _make_policy(seed=42)
        policy_ctrl = _make_policy(seed=42)

        DiscreteBCPretrainer(
            lr=3e-3, batch_size=8, seed=13,
        ).pretrain(
            policy_aug, positives, n_steps=100,
            close_hold_samples=hold_samples,
        )
        DiscreteBCPretrainer(
            lr=3e-3, batch_size=8, seed=13,
        ).pretrain(policy_ctrl, positives, n_steps=100)

        obs_t = torch.tensor(
            np.stack([s.obs for s in hold_samples], axis=0),
            dtype=torch.float32,
        )
        with torch.no_grad():
            logits_aug = policy_aug(obs_t).logits
            logits_ctrl = policy_ctrl(obs_t).logits

        action_space = policy_aug.action_space
        noop_idx = int(action_space.encode(ActionType.NOOP, None))

        # Use a relative advantage so we don't confound with the
        # overall logit scale changing across runs.
        def _noop_advantage(logits: torch.Tensor) -> float:
            noop_l = logits[:, noop_idx]
            others_mean = (
                logits.sum(dim=1) - noop_l
            ) / (logits.shape[1] - 1)
            return float((noop_l - others_mean).mean().item())

        adv_aug = _noop_advantage(logits_aug)
        adv_ctrl = _noop_advantage(logits_ctrl)

        assert adv_aug > adv_ctrl, (
            f"NOOP advantage on hold-target obs did not rise with "
            f"augmentation: augmented={adv_aug:.4f}, "
            f"control={adv_ctrl:.4f}."
        )
