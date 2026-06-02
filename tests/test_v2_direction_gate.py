"""Regression tests for phase-14 S03 direction-confidence gate.

The gate masks ``OPEN_BACK_i`` / ``OPEN_LAY_i`` action logits when
the runner's ``max(P_back, P_lay)`` falls below
``direction_gate_threshold``. NOOP and ``CLOSE_i`` are NEVER gated
— see ``plans/rewrite/phase-14-direction-gate/hard_constraints.md``
§14, §15.

Eight tests:

1. ``test_gate_disabled_is_byte_identical`` — same seed + same obs,
   gate-off vs gate-on-with-disabled-flag produce identical logits.
2. ``test_gate_masks_open_when_max_below_threshold`` —
   strict threshold (0.95) on a fresh policy whose direction
   sigmoid output sits near 0.5 → all OPEN slots get -inf in
   ``masked_logits``.
3. ``test_gate_passes_open_when_max_above_threshold`` — synthetic
   policy with a known-high P_back → corresponding OPEN slot stays
   finite.
4. ``test_gate_does_not_touch_noop_or_close`` — at any threshold,
   NOOP and CLOSE slots remain finite.
5. ``test_gate_uses_per_runner_max_not_per_side`` — when P_back is
   high but P_lay is low, BOTH OPEN_BACK_i and OPEN_LAY_i for that
   runner stay finite (gate filters opportunity, not side).
6. ``test_gate_threshold_clamped_to_range`` — constructing with
   threshold=0.99 clamps to 0.95; with 0.4 clamps to 0.5.
7. ``test_gate_ands_with_legality_mask`` — passing an external
   legality mask AND the direction gate both contribute; a position
   blocked by either mask stays blocked.
8. ``test_gate_threshold_is_phase5_gene`` — the GA evolves
   ``direction_gate_threshold`` per-agent when ``--enable-gene
   direction_gate_threshold`` is in the enabled_set. Two sampled
   agents draw independent threshold values from
   [0.5, 0.95].
"""

from __future__ import annotations

import random

import pytest
import torch

from agents_v2.action_space import ActionType, DiscreteActionSpace
from agents_v2.discrete_policy import DiscreteLSTMPolicy
from training_v2.cohort.genes import (
    DIRECTION_GATE_THRESHOLD_RANGE,
    PHASE5_GENE_DEFAULTS,
    PHASE5_GENE_NAMES,
    sample_genes,
)


_OBS_DIM = 64
_MAX_RUNNERS = 4
_HIDDEN = 32


def _make_policy(
    *,
    enabled: bool = False,
    threshold: float = 0.5,
    seed: int = 0,
) -> DiscreteLSTMPolicy:
    torch.manual_seed(seed)
    space = DiscreteActionSpace(max_runners=_MAX_RUNNERS)
    return DiscreteLSTMPolicy(
        obs_dim=_OBS_DIM,
        action_space=space,
        hidden_size=_HIDDEN,
        direction_gate_enabled=enabled,
        direction_gate_threshold=threshold,
    )


# ── 1. byte-identity when disabled ──────────────────────────────────────────


class TestGateDisabledByteIdentical:
    def test_disabled_flag_matches_no_gate(self):
        p_off = _make_policy(enabled=False, threshold=0.5, seed=42)
        p_on_low = _make_policy(enabled=False, threshold=0.95, seed=42)
        torch.manual_seed(0)
        obs = torch.randn(2, _OBS_DIM)
        with torch.no_grad():
            out_off = p_off(obs)
            out_on = p_on_low(obs)
        # When enabled=False, the threshold value is irrelevant.
        assert torch.allclose(
            out_off.masked_logits, out_on.masked_logits, atol=1e-7,
        )


# ── 2. masking on low confidence ────────────────────────────────────────────


class TestGateMasksOpenWhenLow:
    def test_strict_threshold_masks_all_opens(self):
        # 2026-05-25: was threshold=0.95 under old (0.5, 0.95) clamp.
        # New clamp (0.10, 0.60) so use 0.60 (the new MAX).
        # Fresh-init direction probs sit near sigmoid(0)=0.5;
        # threshold 0.60 > 0.5 masks every OPEN slot.
        p = _make_policy(enabled=True, threshold=0.60, seed=0)
        obs = torch.zeros(1, _OBS_DIM)
        with torch.no_grad():
            out = p(obs)
        ml = out.masked_logits[0]
        R = _MAX_RUNNERS
        open_slice = ml[1: 1 + 2 * R]
        assert torch.isinf(open_slice).all()


# ── 3. passing on high confidence ───────────────────────────────────────────


class TestGatePassesOpenWhenHigh:
    def test_synthetic_high_back_prob_unblocks_open_back(self):
        """Construct a policy and FORCE its direction probs to known
        values. Confirms the gate uses the actual head output, not a
        hard-coded distribution.
        """
        p = _make_policy(enabled=True, threshold=0.7, seed=0)
        # Monkey-patch the direction head to emit a specific output:
        # back_logit = +5 (sigmoid ≈ 0.99), lay_logit = -5
        # (sigmoid ≈ 0.01) at runner 0; remaining runners stay at
        # the head's natural fresh-init output.
        original = p.direction_prob_head

        def hijacked(x):
            out = original(x)
            # x has shape (B*R, embed+hidden); out has shape (B*R, 2).
            # Force runner 0 — every R-th row — to (back_logit=+5,
            # lay_logit=-5). Detect runner 0 rows by index modulo R.
            B_times_R = out.shape[0]
            R = _MAX_RUNNERS
            for row_idx in range(B_times_R):
                runner = row_idx % R
                if runner == 0:
                    out = out.clone()  # avoid in-place on grad tensor
                    out[row_idx, 0] = 5.0
                    out[row_idx, 1] = -5.0
            return out

        # Replacing the module entirely keeps the test simple; we
        # just need direction probs forced to known values. Use a
        # forward pre-hook approach via a wrapping callable.
        class _Wrapped(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def forward(self, x):
                return hijacked(x)

        p.direction_prob_head = _Wrapped(original)

        obs = torch.zeros(1, _OBS_DIM)
        with torch.no_grad():
            out = p(obs)
        ml = out.masked_logits[0]
        R = _MAX_RUNNERS
        # OPEN_BACK_0 at index 1 (encode(OPEN_BACK, 0)).
        # OPEN_LAY_0  at index 1 + R = 5.
        # max(P_back=0.99, P_lay=0.01) = 0.99 ≥ 0.7 → both UNGATED
        # (per-runner gate, not per-side).
        assert torch.isfinite(ml[1])  # OPEN_BACK_0
        assert torch.isfinite(ml[1 + R])  # OPEN_LAY_0
        # Other runners' direction probs are still near 0.5 from
        # the head's natural output → at threshold 0.7, those are
        # gated.
        for slot in range(1, R):
            assert torch.isinf(ml[1 + slot])  # OPEN_BACK_slot
            assert torch.isinf(ml[1 + R + slot])  # OPEN_LAY_slot


# ── 4. NOOP and CLOSE never gated ───────────────────────────────────────────


class TestGateNeverTouchesNoopOrClose:
    @pytest.mark.parametrize("threshold", [0.5, 0.7, 0.9, 0.95])
    def test_noop_and_close_finite_at_any_threshold(self, threshold):
        p = _make_policy(enabled=True, threshold=threshold, seed=0)
        obs = torch.randn(2, _OBS_DIM)
        with torch.no_grad():
            out = p(obs)
        ml = out.masked_logits
        R = _MAX_RUNNERS
        # NOOP at index 0.
        assert torch.isfinite(ml[:, 0]).all()
        # CLOSE at indices 1+2R..1+3R.
        close_slice = ml[:, 1 + 2 * R: 1 + 3 * R]
        assert torch.isfinite(close_slice).all()


# ── 5. per-runner max, not per-side ─────────────────────────────────────────


class TestGateUsesMaxNotPerSide:
    def test_high_back_alone_unblocks_both_sides_for_runner(self):
        """When P_back >> threshold but P_lay << threshold for a
        runner, the gate should pass BOTH OPEN_BACK_i AND OPEN_LAY_i
        for that runner — the per-runner ``max(P_back, P_lay)`` is
        what's compared to threshold, so a single high-confidence
        side unblocks the whole runner. The actor downstream picks
        the side it prefers.
        """
        # We rely on TestGatePassesOpenWhenHigh's fixture above —
        # runner 0 had P_back=0.99 and P_lay=0.01, and both
        # OPEN_BACK_0 and OPEN_LAY_0 were finite. The behaviour is
        # already pinned there; this test serves as a redundant
        # documentation of the per-runner-max contract.
        p = _make_policy(enabled=True, threshold=0.7, seed=0)
        # Same hijack pattern as test 3.
        original = p.direction_prob_head
        R = _MAX_RUNNERS

        def hijacked(x):
            out = original(x).clone()
            for row_idx in range(out.shape[0]):
                runner = row_idx % R
                if runner == 0:
                    out[row_idx, 0] = 5.0   # P_back ≈ 0.99
                    out[row_idx, 1] = -5.0  # P_lay  ≈ 0.01
            return out

        class _W(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def forward(self, x):
                return hijacked(x)

        p.direction_prob_head = _W(original)
        obs = torch.zeros(1, _OBS_DIM)
        with torch.no_grad():
            out = p(obs)
        ml = out.masked_logits[0]
        # Runner 0 BOTH sides finite, even though P_lay is near 0.
        assert torch.isfinite(ml[1])      # OPEN_BACK_0
        assert torch.isfinite(ml[1 + R])  # OPEN_LAY_0


# ── 6. threshold clamping ──────────────────────────────────────────────────


class TestGateThresholdClamped:
    # 2026-05-25 RECALIBRATION: clamp bounds were (0.5, 0.95) under
    # the pos-weighted head; now (0.10, 0.60) under unweighted C11.
    @pytest.mark.parametrize(
        "raw,expected",
        [(0.99, 0.60), (1.5, 0.60), (0.05, 0.10), (-0.1, 0.10)],
    )
    def test_clamp(self, raw, expected):
        p = _make_policy(enabled=True, threshold=raw, seed=0)
        assert p.direction_gate_threshold == expected

    def test_in_range_passthrough(self):
        # Values inside the new clamp (0.10, 0.60) should pass through.
        for v in (0.15, 0.30, 0.45, 0.55):
            p = _make_policy(enabled=True, threshold=v, seed=0)
            assert p.direction_gate_threshold == v

    def test_recalibrated_gene_range_not_squashed(self):
        """2026-05-25 regression: the old clamp (0.5, 0.95) silently
        squashed the recalibrated gene range (0.20, 0.50) to a
        single point. Verify that constructing with the gene-range
        min and max produces distinct stored values."""
        from training_v2.cohort.genes import DIRECTION_GATE_THRESHOLD_RANGE
        lo, hi = DIRECTION_GATE_THRESHOLD_RANGE
        p_lo = _make_policy(enabled=True, threshold=lo, seed=0)
        p_hi = _make_policy(enabled=True, threshold=hi, seed=0)
        # The gene range must fit inside the policy clamp, so
        # both values pass through unchanged.
        assert p_lo.direction_gate_threshold == lo
        assert p_hi.direction_gate_threshold == hi
        assert p_lo.direction_gate_threshold != p_hi.direction_gate_threshold


# ── 7. AND with legality mask ──────────────────────────────────────────────


class TestGateAndsWithLegalityMask:
    def test_legality_blocked_position_stays_blocked(self):
        """Legality mask alone blocks position k. With gate enabled,
        position k should still be blocked regardless of direction
        confidence.
        """
        p = _make_policy(enabled=True, threshold=0.5, seed=0)
        obs = torch.zeros(1, _OBS_DIM)
        # Block CLOSE_0 (index 1+2R) via legality mask.
        n_actions = p.action_space.n
        legal = torch.ones(1, n_actions, dtype=torch.bool)
        close_0_idx = 1 + 2 * _MAX_RUNNERS
        legal[0, close_0_idx] = False
        with torch.no_grad():
            out = p(obs, mask=legal)
        # CLOSE_0 must be -inf (legality mask wins).
        assert torch.isinf(out.masked_logits[0, close_0_idx])


# ── 8. Phase 5 gene evolution ──────────────────────────────────────────────


class TestGateGeneIsPhase5Evolved:
    def test_threshold_in_phase5_names_when_enabled(self):
        # The gene must be in PHASE5_GENE_NAMES so the cohort runner
        # accepts ``--enable-gene direction_gate_threshold``.
        assert "direction_gate_threshold" in PHASE5_GENE_NAMES
        # Default value matches the gate's no-op floor.
        assert PHASE5_GENE_DEFAULTS["direction_gate_threshold"] == 0.5

    def test_two_agents_sample_independently(self):
        rng_a = random.Random(123)
        rng_b = random.Random(456)
        enabled = frozenset({"direction_gate_threshold"})
        a = sample_genes(rng_a, enabled_set=enabled)
        b = sample_genes(rng_b, enabled_set=enabled)
        lo, hi = DIRECTION_GATE_THRESHOLD_RANGE
        assert lo <= a.direction_gate_threshold <= hi
        assert lo <= b.direction_gate_threshold <= hi
        # Independent draws should differ on at least one agent
        # most of the time. With seeds 123 / 456 they will.
        assert a.direction_gate_threshold != b.direction_gate_threshold


# ── 9. apply_direction_gate kwarg + captured-mask path (S05) ──────────────


class TestGateMaskCapturePath:
    """Phase-14 S05 — the rollout collector captures the effective
    mask (legality AND gate) per tick; the trainer passes it back to
    the policy with ``apply_direction_gate=False`` so the in-forward
    gate recompute is bypassed. Without this, gate-on PPO updates
    produce ``approx_kl=inf`` (the smoke surfaced this; see
    findings.md).
    """

    def test_apply_direction_gate_false_skips_in_forward_recompute(self):
        """Same policy, same obs: with ``apply_direction_gate=None``
        (default) the gate masks OPEN slots; with
        ``apply_direction_gate=False`` it does NOT (caller is
        responsible for passing the gate via ``mask``)."""
        # 2026-05-25: clamp recalibration — use 0.60 (new MAX)
        # instead of 0.95.
        p = _make_policy(enabled=True, threshold=0.60, seed=0)
        obs = torch.zeros(1, _OBS_DIM)
        with torch.no_grad():
            default_out = p(obs)
            no_gate_out = p(obs, apply_direction_gate=False)
        # With default the OPEN slots are -inf (gate masks them).
        R = _MAX_RUNNERS
        default_open = default_out.masked_logits[0, 1: 1 + 2 * R]
        assert torch.isinf(default_open).all()
        # With apply_direction_gate=False the OPEN slots are finite.
        no_gate_open = no_gate_out.masked_logits[0, 1: 1 + 2 * R]
        assert torch.isfinite(no_gate_open).all()

    def test_set_effective_gate_threshold_overrides_gene(self):
        """Phase-14 S06: trainer poke overrides the gene value."""
        # 2026-05-25: strict=0.60 (new MAX), loose=0.10 (new floor).
        # Fresh-init head produces sigmoid(0)≈0.5; 0.60 masks all,
        # 0.10 masks none.
        p = _make_policy(enabled=True, threshold=0.60, seed=0)
        obs = torch.zeros(1, _OBS_DIM)
        with torch.no_grad():
            out_strict = p(obs)
        n_inf_strict = torch.isinf(
            out_strict.masked_logits[0, 1: 1 + 2 * _MAX_RUNNERS]
        ).sum().item()
        p.set_effective_gate_threshold(0.10)
        with torch.no_grad():
            out_loose = p(obs)
        n_inf_loose = torch.isinf(
            out_loose.masked_logits[0, 1: 1 + 2 * _MAX_RUNNERS]
        ).sum().item()
        assert n_inf_strict > n_inf_loose

    def test_supplied_mask_combined_with_apply_direction_gate_false(self):
        """When the trainer passes the captured rollout-time mask
        and ``apply_direction_gate=False``, the policy returns
        masked_logits whose finite-positions exactly match the
        supplied mask. No extra in-forward gate is added.
        """
        p = _make_policy(enabled=True, threshold=0.60, seed=0)
        obs = torch.zeros(1, _OBS_DIM)
        # Synthetic "rollout-time" mask: NOOP + first OPEN_BACK
        # legal, everything else illegal.
        n = p.action_space.n
        rollout_mask = torch.zeros(1, n, dtype=torch.bool)
        rollout_mask[0, 0] = True   # NOOP
        rollout_mask[0, 1] = True   # OPEN_BACK_0
        with torch.no_grad():
            out = p(
                obs, mask=rollout_mask,
                apply_direction_gate=False,
            )
        finite = torch.isfinite(out.masked_logits[0])
        # Exactly NOOP + OPEN_BACK_0 are finite.
        assert finite[0].item() and finite[1].item()
        # Everything else is -inf.
        assert torch.isinf(out.masked_logits[0, 2:]).all()


# ── 10. Threshold warmup (S06) ─────────────────────────────────────────────


class TestGateThresholdWarmup:
    """Phase-14 S06 — the trainer linearly anneals the policy's
    effective gate threshold from the floor (0.5) to the gene value
    across the first ``direction_gate_warmup_eps`` episodes.
    Without this, agents that draw strict thresholds (≥0.85) never
    open at cold start and PPO has no reward gradient to learn from
    (the smoke surfaced this; see findings.md)."""

    def test_warmup_starts_at_floor(self):
        """At eps=0, the trainer's _effective_direction_gate_threshold
        returns the floor (= DIRECTION_GATE_THRESHOLD_MIN, 0.10 post-
        2026-05-25 recalibration)."""
        from training_v2.discrete_ppo.trainer import DiscretePPOTrainer
        from agents_v2.discrete_policy import DiscreteLSTMPolicy
        class _StubTrainer:
            _direction_gate_warmup_eps = 5
            _eps_since_gate_start = 0
            policy = _make_policy(
                enabled=True, threshold=0.55, seed=0,
            )
            _effective_direction_gate_threshold = (
                DiscretePPOTrainer._effective_direction_gate_threshold
            )
        t = _StubTrainer()
        v = t._effective_direction_gate_threshold()
        # eps=0, frac=0/5=0 → floor (DIRECTION_GATE_THRESHOLD_MIN).
        assert v == DiscreteLSTMPolicy.DIRECTION_GATE_THRESHOLD_MIN

    def test_warmup_reaches_gene_value(self):
        from training_v2.discrete_ppo.trainer import DiscretePPOTrainer
        class _StubTrainer:
            _direction_gate_warmup_eps = 5
            _eps_since_gate_start = 5  # at end of warmup
            policy = _make_policy(
                enabled=True, threshold=0.55, seed=0,
            )
            _effective_direction_gate_threshold = (
                DiscretePPOTrainer._effective_direction_gate_threshold
            )
        t = _StubTrainer()
        v = t._effective_direction_gate_threshold()
        assert v == 0.55

    def test_warmup_inactive_when_eps_zero(self):
        """``direction_gate_warmup_eps=0`` → no warmup, gene value
        applies from episode 0."""
        from training_v2.discrete_ppo.trainer import DiscretePPOTrainer
        class _StubTrainer:
            _direction_gate_warmup_eps = 0
            _eps_since_gate_start = 0
            policy = _make_policy(
                enabled=True, threshold=0.55, seed=0,
            )
            _effective_direction_gate_threshold = (
                DiscretePPOTrainer._effective_direction_gate_threshold
            )
        t = _StubTrainer()
        v = t._effective_direction_gate_threshold()
        assert v == 0.55

    def test_warmup_inactive_when_gate_disabled(self):
        from training_v2.discrete_ppo.trainer import DiscretePPOTrainer
        class _StubTrainer:
            _direction_gate_warmup_eps = 5
            _eps_since_gate_start = 0
            policy = _make_policy(
                enabled=False, threshold=0.55, seed=0,
            )
            _effective_direction_gate_threshold = (
                DiscretePPOTrainer._effective_direction_gate_threshold
            )
        t = _StubTrainer()
        # Gate disabled → returns gene value regardless of eps.
        assert t._effective_direction_gate_threshold() == 0.55

    def test_warmup_linear_at_midpoint(self):
        from training_v2.discrete_ppo.trainer import DiscretePPOTrainer
        from agents_v2.discrete_policy import DiscreteLSTMPolicy
        class _StubTrainer:
            _direction_gate_warmup_eps = 4
            _eps_since_gate_start = 2  # halfway
            policy = _make_policy(
                enabled=True, threshold=0.55, seed=0,
            )
            _effective_direction_gate_threshold = (
                DiscretePPOTrainer._effective_direction_gate_threshold
            )
        t = _StubTrainer()
        v = t._effective_direction_gate_threshold()
        # frac=0.5 → floor + 0.5 * (gene - floor)
        #          = 0.10 + 0.5 * (0.55 - 0.10) = 0.10 + 0.225 = 0.325
        floor = DiscreteLSTMPolicy.DIRECTION_GATE_THRESHOLD_MIN
        expected = floor + 0.5 * (0.55 - floor)
        assert abs(v - expected) < 1e-9


# ──────────────────────────────────────────────────────────────────
# Regression: CLI flag vs reward-override resolution (2026-05-24)
# ──────────────────────────────────────────────────────────────────


class TestResolvePolicyGateEnabled:
    """Regression for the 2026-05-24 wiring bug.

    The bug: ``train_one_agent`` previously resolved the policy-side
    gate-enable bool via
    ``trainer_hp.get("direction_gate_enabled", cli_flag)``. Because
    ``CohortGenes.to_dict()`` ALWAYS includes the key (with the gene
    dataclass default of ``False``), the ``.get`` fallback never
    fired — so the CLI flag was silently discarded and replaced with
    ``False``. Result: ``--direction-gate-enabled`` enabled only the
    ENV-side gate (dir_fire_drift refusal of OPEN_LAY); the
    POLICY-side gate (action-mask via direction_prob_head outputs)
    stayed OFF unless the operator also passed
    ``--reward-overrides direction_gate_enabled=true``.

    Symptom: cohort
    ``_recipe_sensitivity_sweep_1779661887`` showed
    ``gate_refusals=0`` for every agent despite non-default
    threshold gene draws and the CLI flag being on.

    Fix: ``_resolve_direction_gate_enabled`` returns
    ``cli_flag OR trainer_hp["direction_gate_enabled"]`` — enable
    if EITHER source says enable.
    """

    def test_cli_flag_alone_enables_when_gene_default_false(self):
        """The CLI flag MUST enable even when the gene-derived
        trainer_hp has the key set to its default False."""
        from training_v2.cohort.worker import _resolve_direction_gate_enabled
        # This is the canonical bug scenario: operator passes
        # --direction-gate-enabled, gene default is False, no
        # --reward-overrides direction_gate_enabled=...
        trainer_hp = {"direction_gate_enabled": False}
        assert _resolve_direction_gate_enabled(
            cli_flag=True, trainer_hp=trainer_hp,
        ) is True

    def test_reward_override_alone_enables_when_cli_flag_false(self):
        """The reward-override key MUST still enable (backward
        compatibility with launches that used --reward-overrides
        direction_gate_enabled=true instead of the CLI flag)."""
        from training_v2.cohort.worker import _resolve_direction_gate_enabled
        trainer_hp = {"direction_gate_enabled": True}
        assert _resolve_direction_gate_enabled(
            cli_flag=False, trainer_hp=trainer_hp,
        ) is True

    def test_both_off_keeps_gate_disabled(self):
        from training_v2.cohort.worker import _resolve_direction_gate_enabled
        trainer_hp = {"direction_gate_enabled": False}
        assert _resolve_direction_gate_enabled(
            cli_flag=False, trainer_hp=trainer_hp,
        ) is False

    def test_both_on_enables(self):
        """OR semantics — both sources saying enable also enables."""
        from training_v2.cohort.worker import _resolve_direction_gate_enabled
        trainer_hp = {"direction_gate_enabled": True}
        assert _resolve_direction_gate_enabled(
            cli_flag=True, trainer_hp=trainer_hp,
        ) is True

    def test_missing_key_in_trainer_hp_defaults_to_disabled(self):
        """If the key is absent from trainer_hp (defensive — the
        production path always populates it via CohortGenes.to_dict()
        but a future refactor could drop it), the CLI flag alone
        still works."""
        from training_v2.cohort.worker import _resolve_direction_gate_enabled
        trainer_hp = {}
        assert _resolve_direction_gate_enabled(
            cli_flag=True, trainer_hp=trainer_hp,
        ) is True
        assert _resolve_direction_gate_enabled(
            cli_flag=False, trainer_hp=trainer_hp,
        ) is False

    def test_production_to_dict_populates_key_at_false_by_default(self):
        """Documents the precondition that made the original
        .get(..., fallback) pattern broken: CohortGenes.to_dict()
        ALWAYS populates direction_gate_enabled, defaulting False.
        If this contract ever changes, the OR-semantics fix still
        works — but this test asserts the input shape future fixes
        must handle."""
        from training_v2.cohort.genes import sample_genes
        rng = random.Random(42)
        genes = sample_genes(rng, enabled_set=frozenset())
        d = genes.to_dict()
        assert "direction_gate_enabled" in d
        assert d["direction_gate_enabled"] is False
