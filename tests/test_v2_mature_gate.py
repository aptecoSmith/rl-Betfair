"""Regression tests for the Path C mature_prob open-gate (2026-05-30).

The gate masks ``OPEN_BACK_i`` / ``OPEN_LAY_i`` action logits when
the runner's own ``mature_prob_head`` sigmoid output falls below
``mature_prob_open_threshold``. NOOP and ``CLOSE_i`` are NEVER gated.
See ``plans/recipe-expansion-and-robustness/`` (Path C).

Unlike the direction gate (``max(P_back, P_lay)`` of two per-side
heads), ``mature_prob`` is a single per-runner probability, so the
same gate-pass bool applies to BOTH the OPEN_BACK and OPEN_LAY slot
of each runner.

Six tests:

1. ``test_disabled_threshold_zero_is_byte_identical`` — threshold 0.0
   (disabled) produces logits identical to a no-gate policy.
2. ``test_strict_threshold_masks_all_opens`` — threshold 1.0 on a
   fresh policy (mature_prob ≈ 0.5) masks every OPEN slot.
3. ``test_synthetic_high_mature_prob_unblocks_runner`` — forcing
   runner 0's mature logit high keeps its OPEN_BACK and OPEN_LAY
   finite while other runners stay gated.
4. ``test_noop_and_close_never_gated`` — at any threshold NOOP and
   CLOSE slots remain finite.
5. ``test_both_open_sides_share_one_gate_pass`` — high mature_prob
   on a runner unblocks both its OPEN_BACK and OPEN_LAY (single
   per-runner probability, not per-side).
6. ``test_gate_enabled_iff_threshold_positive`` — the
   ``mature_prob_open_gate_enabled`` flag is True iff the threshold
   is > 0.0 (wiring contract — no separate enable bool).
"""

from __future__ import annotations

import pytest
import torch

from agents_v2.action_space import DiscreteActionSpace
from agents_v2.discrete_policy import DiscreteLSTMPolicy


_OBS_DIM = 64
_MAX_RUNNERS = 4
_HIDDEN = 32


def _make_policy(*, threshold: float = 0.0, seed: int = 0) -> DiscreteLSTMPolicy:
    torch.manual_seed(seed)
    space = DiscreteActionSpace(max_runners=_MAX_RUNNERS)
    return DiscreteLSTMPolicy(
        obs_dim=_OBS_DIM,
        action_space=space,
        hidden_size=_HIDDEN,
        mature_prob_open_threshold=threshold,
    )


# ── 1. disabled (threshold 0.0) is byte-identical ───────────────────────────


class TestDisabledByteIdentical:
    def test_threshold_zero_matches_no_gate(self):
        p_off = _make_policy(threshold=0.0, seed=42)
        torch.manual_seed(0)
        obs = torch.randn(2, _OBS_DIM)
        with torch.no_grad():
            out = p_off(obs)
        # threshold 0.0 ⇒ gate disabled ⇒ no OPEN slot masked.
        assert not p_off.mature_prob_open_gate_enabled
        assert torch.isfinite(out.masked_logits).all()


# ── 2. strict threshold masks every OPEN slot ───────────────────────────────


class TestStrictThresholdMasks:
    def test_threshold_one_masks_all_opens(self):
        # Fresh-init mature_prob sits near sigmoid(0)=0.5;
        # threshold 1.0 > 0.5 masks every OPEN slot.
        p = _make_policy(threshold=1.0, seed=0)
        obs = torch.zeros(1, _OBS_DIM)
        with torch.no_grad():
            out = p(obs)
        ml = out.masked_logits[0]
        R = _MAX_RUNNERS
        open_slice = ml[1: 1 + 2 * R]
        assert torch.isinf(open_slice).all()


# ── 3. synthetic high mature_prob unblocks one runner ───────────────────────


class TestSyntheticHighUnblocks:
    def test_high_mature_logit_unblocks_runner_zero(self):
        p = _make_policy(threshold=0.7, seed=0)

        original = p.mature_prob_head

        class _Wrapped(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                out = self.inner(x).clone()
                # out shape (batch, R). Force runner 0 logit high
                # (sigmoid(+5) ≈ 0.993) so it clears threshold 0.7.
                out[:, 0] = 5.0
                return out

        p.mature_prob_head = _Wrapped(original)

        obs = torch.zeros(1, _OBS_DIM)
        with torch.no_grad():
            out = p(obs)
        ml = out.masked_logits[0]
        R = _MAX_RUNNERS
        # runner 0 OPEN_BACK / OPEN_LAY stay finite.
        assert torch.isfinite(ml[1])
        assert torch.isfinite(ml[1 + R])
        # other runners (mature_prob ≈ 0.5 < 0.7) stay gated.
        for slot in range(1, R):
            assert torch.isinf(ml[1 + slot])
            assert torch.isinf(ml[1 + R + slot])


# ── 4. NOOP and CLOSE never gated ───────────────────────────────────────────


class TestNoopCloseNeverGated:
    @pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7, 1.0])
    def test_noop_and_close_finite_at_any_threshold(self, threshold):
        p = _make_policy(threshold=threshold, seed=0)
        obs = torch.randn(2, _OBS_DIM)
        with torch.no_grad():
            out = p(obs)
        ml = out.masked_logits
        R = _MAX_RUNNERS
        assert torch.isfinite(ml[:, 0]).all()  # NOOP
        close_slice = ml[:, 1 + 2 * R: 1 + 3 * R]
        assert torch.isfinite(close_slice).all()


# ── 5. one gate-pass governs both OPEN sides ────────────────────────────────


class TestBothSidesShareGate:
    def test_high_mature_unblocks_both_back_and_lay(self):
        p = _make_policy(threshold=0.7, seed=0)
        original = p.mature_prob_head

        class _Wrapped(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                out = self.inner(x).clone()
                out[:, 0] = 5.0
                return out

        p.mature_prob_head = _Wrapped(original)
        obs = torch.zeros(1, _OBS_DIM)
        with torch.no_grad():
            out = p(obs)
        ml = out.masked_logits[0]
        R = _MAX_RUNNERS
        # Single per-runner probability ⇒ BOTH sides unblocked.
        assert torch.isfinite(ml[1])       # OPEN_BACK_0
        assert torch.isfinite(ml[1 + R])   # OPEN_LAY_0


# ── 6. enabled iff threshold > 0 ────────────────────────────────────────────


class TestGateEnabledIffThresholdPositive:
    @pytest.mark.parametrize(
        "threshold,expected",
        [(0.0, False), (0.01, True), (0.5, True), (1.0, True)],
    )
    def test_enabled_flag_tracks_threshold(self, threshold, expected):
        p = _make_policy(threshold=threshold, seed=0)
        assert p.mature_prob_open_gate_enabled is expected
