"""Regression tests for the phase-13 S05 direction-targeted BC layer.

The new BC plumbing layers a per-side direction target on top of
phase-8's oracle target. Tests in this file exercise:

1. Default weight 0 → byte-identical to phase-8 BC (no behaviour
   change on unaffected agents).
2. ``build_direction_target_map`` correctly classifies the four
   label tuples ``(b, l)`` ∈ {(0,0), (1,0), (0,1), (1,1)} per the
   D2 spec.
3. With weight > 0 and an unambiguous lay-side label, the loss
   pushes the policy's probability mass toward OPEN_LAY at the
   target runner (a directional sanity check).
4. With weight > 0 but no unambiguous direction entries in the
   batch, the layered loss collapses to oracle-only (the direction
   CE term contributes a zero gradient).
"""

from __future__ import annotations

import numpy as np
import torch

from agents_v2.action_space import ActionType, DiscreteActionSpace
from agents_v2.discrete_policy import DiscreteLSTMPolicy
from training_v2.arb_oracle import OracleSample
from training_v2.direction_label_scan import DirectionLabel
from training_v2.discrete_ppo.bc_pretrain import (
    DiscreteBCPretrainer,
    build_direction_bce_label_map,
    build_direction_target_map,
)


_OBS_DIM = 32
_MAX_RUNNERS = 4
_HIDDEN = 16


def _make_policy(seed: int = 0) -> DiscreteLSTMPolicy:
    torch.manual_seed(seed)
    space = DiscreteActionSpace(max_runners=_MAX_RUNNERS)
    return DiscreteLSTMPolicy(
        obs_dim=_OBS_DIM,
        action_space=space,
        hidden_size=_HIDDEN,
    )


def _make_oracle_samples(
    n: int,
    runner_idx: int = 0,
    tick_index_start: int = 0,
) -> list[OracleSample]:
    rng = np.random.default_rng(0)
    samples: list[OracleSample] = []
    for i in range(n):
        samples.append(OracleSample(
            tick_index=tick_index_start + i,
            runner_idx=runner_idx,
            obs=rng.standard_normal(_OBS_DIM).astype(np.float32),
            arb_spread_ticks=5,
            expected_locked_pnl=0.5,
        ))
    return samples


# ── 1. byte-identity at weight 0 ────────────────────────────────────────────


class TestDefaultWeightByteIdentical:
    def test_weight_zero_matches_oracle_only(self):
        policy_a = _make_policy(seed=42)
        policy_b = _make_policy(seed=42)
        samples = _make_oracle_samples(50)
        # Build a non-trivial direction map; with weight 0 it should
        # be ignored entirely.
        space = policy_a.action_space
        direction_map = {
            (0, 0): space.encode(ActionType.OPEN_LAY, 0),
        }

        DiscreteBCPretrainer(lr=1e-3, batch_size=8, seed=0).pretrain(
            policy=policy_a, samples=samples, n_steps=20,
        )
        DiscreteBCPretrainer(lr=1e-3, batch_size=8, seed=0).pretrain(
            policy=policy_b, samples=samples, n_steps=20,
            direction_target_map=direction_map,
            direction_target_weight=0.0,  # disabled
        )
        # Final actor_head weights must match exactly — same seed,
        # same samples, same effective loss.
        for k_a, p_a in policy_a.named_parameters():
            p_b = dict(policy_b.named_parameters())[k_a]
            assert torch.allclose(p_a, p_b, atol=1e-7), (
                f"Param {k_a} diverged at weight=0 — direction term "
                "leaked into the loss path."
            )


# ── 2. direction map construction ──────────────────────────────────────────


class TestBuildDirectionTargetMap:
    def _make(self, label_back: float, label_lay: float) -> DirectionLabel:
        return DirectionLabel(
            tick_index=7, runner_idx=2,
            label_back=label_back, label_lay=label_lay,
            ltp_at_open=5.0, threshold_back=4.5, threshold_lay=5.5,
            first_back_fav_tick=-1, first_lay_fav_tick=-1,
        )

    def test_back_only_emits_open_back(self):
        space = DiscreteActionSpace(max_runners=_MAX_RUNNERS)
        m = build_direction_target_map(
            [self._make(1.0, 0.0)], space,
        )
        expected = space.encode(ActionType.OPEN_BACK, 2)
        assert m == {(7, 2): expected}

    def test_lay_only_emits_open_lay(self):
        space = DiscreteActionSpace(max_runners=_MAX_RUNNERS)
        m = build_direction_target_map(
            [self._make(0.0, 1.0)], space,
        )
        expected = space.encode(ActionType.OPEN_LAY, 2)
        assert m == {(7, 2): expected}

    def test_both_zero_omitted(self):
        space = DiscreteActionSpace(max_runners=_MAX_RUNNERS)
        m = build_direction_target_map(
            [self._make(0.0, 0.0)], space,
        )
        assert m == {}

    def test_both_one_omitted(self):
        space = DiscreteActionSpace(max_runners=_MAX_RUNNERS)
        m = build_direction_target_map(
            [self._make(1.0, 1.0)], space,
        )
        assert m == {}


# ── 3. lay-side label pushes probability toward OPEN_LAY ───────────────────


class TestLayLabelPushesPolicyTowardOpenLay:
    def test_after_bc_open_lay_logit_higher(self):
        # Use a SINGLE sample (deterministic batch). The oracle would
        # push toward OPEN_BACK_0; the direction layer with weight=1.0
        # pushes toward OPEN_LAY_0. With weight=1.0 the oracle term
        # contributes zero, so the policy should converge on
        # OPEN_LAY_0 after enough steps.
        policy = _make_policy(seed=7)
        space = policy.action_space
        samples = _make_oracle_samples(1)
        sample = samples[0]
        direction_map = {
            (sample.tick_index, sample.runner_idx):
                space.encode(ActionType.OPEN_LAY, sample.runner_idx),
        }

        DiscreteBCPretrainer(lr=5e-3, batch_size=1, seed=0).pretrain(
            policy=policy, samples=samples, n_steps=200,
            direction_target_map=direction_map,
            direction_target_weight=1.0,  # direction-only
        )

        with torch.no_grad():
            obs_t = torch.tensor(
                sample.obs, dtype=torch.float32,
            ).unsqueeze(0)
            out = policy(obs_t)
            logits = out.logits.squeeze(0)
            ob_idx = space.encode(
                ActionType.OPEN_BACK, sample.runner_idx,
            )
            ol_idx = space.encode(
                ActionType.OPEN_LAY, sample.runner_idx,
            )
        assert logits[ol_idx].item() > logits[ob_idx].item(), (
            "After direction-only BC with lay label, OPEN_LAY logit "
            "should exceed OPEN_BACK logit at the target runner."
        )


# ── 4. empty direction map is still safe at weight > 0 ─────────────────────


class TestEmptyDirectionMapStable:
    def test_no_direction_entries_does_not_break_training(self):
        policy = _make_policy(seed=11)
        samples = _make_oracle_samples(40, runner_idx=0)
        # Map has NO entries matching any sample's (tick, runner) →
        # the direction CE branch's mask is all-False every batch,
        # the per-step ``direction_ce`` collapses to a zero scalar,
        # and oracle CE drives training (scaled by ``1 - w``). The
        # important thing is no NaN / no exception.
        empty_map: dict[tuple[int, int], int] = {(999, 9): 0}

        history = DiscreteBCPretrainer(
            lr=1e-3, batch_size=8, seed=0,
        ).pretrain(
            policy=policy, samples=samples, n_steps=10,
            direction_target_map=empty_map,
            direction_target_weight=0.5,
        )
        assert len(history.ce_losses) == 10
        assert all(
            np.isfinite(loss) for loss in history.ce_losses
        ), "BC loss became NaN with empty direction map"


# ── 5. Phase-15 S02: direction-head BCE training ───────────────────────────


class TestPhase15DirectionBceLabelMap:
    """``build_direction_bce_label_map`` preserves binary (b, l) tuples
    (vs ``build_direction_target_map`` which collapses ambiguous rows
    to no-entry).
    """

    def _label(self, b: float, l: float) -> DirectionLabel:
        return DirectionLabel(
            tick_index=3, runner_idx=1,
            label_back=b, label_lay=l,
            ltp_at_open=5.0, threshold_back=4.5, threshold_lay=5.5,
            first_back_fav_tick=-1, first_lay_fav_tick=-1,
        )

    def test_all_four_label_combinations_preserved(self):
        space = DiscreteActionSpace(max_runners=_MAX_RUNNERS)
        labels = [
            DirectionLabel(
                tick_index=t, runner_idx=1,
                label_back=b, label_lay=l,
                ltp_at_open=5.0, threshold_back=4.5, threshold_lay=5.5,
                first_back_fav_tick=-1, first_lay_fav_tick=-1,
            )
            for t, (b, l) in enumerate([
                (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),
            ])
        ]
        m = build_direction_bce_label_map(labels, space)
        # All 4 entries kept (unlike build_direction_target_map which
        # drops (0,0) and (1,1)).
        assert len(m) == 4
        assert m[(0, 1)] == (0.0, 0.0)
        assert m[(1, 1)] == (1.0, 0.0)
        assert m[(2, 1)] == (0.0, 1.0)
        assert m[(3, 1)] == (1.0, 1.0)


class TestPhase15DirectionBceTrainsHead:
    """With ``direction_bce_weight > 0``, BC trains
    ``direction_prob_head`` directly (not just actor_head).
    """

    def test_direction_prob_head_weights_change(self):
        policy = _make_policy(seed=11)
        # Snapshot the head's weights BEFORE BC.
        before = {
            k: v.detach().clone()
            for k, v in policy.named_parameters()
            if "direction_prob_head" in k
        }

        samples = _make_oracle_samples(50)
        # Build a label map that has entries matching samples' keys.
        bce_map = {
            (s.tick_index, s.runner_idx): (1.0, 0.0)
            for s in samples
        }

        DiscreteBCPretrainer(lr=1e-2, batch_size=8, seed=0).pretrain(
            policy=policy, samples=samples, n_steps=50,
            direction_bce_label_map=bce_map,
            direction_bce_weight=1.0,
        )

        # Some direction_prob_head param must have changed.
        any_changed = False
        for k, p_after in policy.named_parameters():
            if "direction_prob_head" in k:
                if not torch.allclose(before[k], p_after, atol=1e-6):
                    any_changed = True
                    break
        assert any_changed, (
            "direction_prob_head weights unchanged after BC with "
            "direction_bce_weight=1.0 — supervised pathway not wired."
        )

    def test_direction_prob_head_unchanged_when_weight_zero(self):
        policy = _make_policy(seed=11)
        before = {
            k: v.detach().clone()
            for k, v in policy.named_parameters()
            if "direction_prob_head" in k
        }
        samples = _make_oracle_samples(50)
        bce_map = {
            (s.tick_index, s.runner_idx): (1.0, 0.0)
            for s in samples
        }

        DiscreteBCPretrainer(lr=1e-2, batch_size=8, seed=0).pretrain(
            policy=policy, samples=samples, n_steps=50,
            direction_bce_label_map=bce_map,
            direction_bce_weight=0.0,  # disabled
        )

        # Phase-15 S02 amendment: when bce weight is 0, the head
        # might still get a tiny gradient via the BC oracle CE
        # path through actor_head reading the head's columns.
        # However, with the .detach() in discrete_policy.py
        # (phase-15 S01 amendment), the actor pathway is severed,
        # so the head MUST be unchanged.
        any_changed = False
        for k, p_after in policy.named_parameters():
            if "direction_prob_head" in k:
                if not torch.allclose(before[k], p_after, atol=1e-6):
                    any_changed = True
        assert not any_changed, (
            "direction_prob_head weights changed at "
            "direction_bce_weight=0 — leak in the supervised path."
        )


class TestPhase15PosWeightKnob:
    """``direction_bce_use_pos_weight=False`` produces different
    training trajectory than the default True, on imbalanced data.
    """

    def test_pos_weight_knob_changes_trajectory(self):
        torch.manual_seed(0)
        policy_a = _make_policy(seed=42)
        policy_b = _make_policy(seed=42)
        samples = _make_oracle_samples(50)
        # Imbalanced labels: only first 10 samples positive on back.
        bce_map = {}
        for i, s in enumerate(samples):
            lb = 1.0 if i < 10 else 0.0
            bce_map[(s.tick_index, s.runner_idx)] = (lb, 0.0)

        DiscreteBCPretrainer(lr=1e-2, batch_size=8, seed=0).pretrain(
            policy=policy_a, samples=samples, n_steps=50,
            direction_bce_label_map=bce_map,
            direction_bce_weight=1.0,
            direction_bce_use_pos_weight=True,
        )
        DiscreteBCPretrainer(lr=1e-2, batch_size=8, seed=0).pretrain(
            policy=policy_b, samples=samples, n_steps=50,
            direction_bce_label_map=bce_map,
            direction_bce_weight=1.0,
            direction_bce_use_pos_weight=False,
        )

        # The two trained heads should differ noticeably.
        diffs = []
        for k, p_a in policy_a.named_parameters():
            if "direction_prob_head" in k:
                p_b = dict(policy_b.named_parameters())[k]
                diffs.append(float((p_a - p_b).abs().max().item()))
        assert any(d > 1e-4 for d in diffs), (
            "pos_weight True vs False produced byte-identical heads — "
            "pos_weight knob is not wired."
        )
