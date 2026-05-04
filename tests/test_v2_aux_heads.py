"""Tests for the Phase 7 S01 auxiliary-head port.

Phase 7 Session 01 deliverable. Three new heads on
:class:`agents_v2.discrete_policy.DiscreteLSTMPolicy`:

- ``fill_prob_head`` — per-runner BCE-trained fill-probability forecast.
  Sigmoid output is column-concat'd into the per-runner ``actor_head``
  input.
- ``mature_prob_head`` — per-runner BCE-trained "mature naturally OR
  closed by agent signal" forecast (force-closed pairs are the negative
  class). Sigmoid output is column-concat'd into ``actor_head``
  alongside ``fill_prob``.
- ``risk_head`` — per-runner Gaussian forecast ``(mean, log_var)`` of
  the locked-P&L outcome. Does NOT feed ``actor_head`` — surfaces on
  ``DiscretePolicyOutput`` for the trainer's NLL term (Phase 7 S02)
  and downstream consumers.

The plan ports the v1 contract verbatim (CLAUDE.md §"fill_prob feeds
actor_head", §"mature_prob_head feeds actor_head") into the v2 single
``DiscreteLSTMPolicy`` class.

These tests are forward-path only — the trainer-side BCE / NLL losses
land in Phase 7 S02.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from agents.policy_network import (
    RISK_LOG_VAR_MAX as V1_RISK_LOG_VAR_MAX,
    RISK_LOG_VAR_MIN as V1_RISK_LOG_VAR_MIN,
)
from agents_v2.action_space import DiscreteActionSpace
from agents_v2.discrete_policy import (
    RISK_LOG_VAR_MAX,
    RISK_LOG_VAR_MIN,
    DiscreteLSTMPolicy,
)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def space() -> DiscreteActionSpace:
    return DiscreteActionSpace(max_runners=14)


@pytest.fixture
def policy(space):
    """Default-shape policy. Hidden=32 to keep the tests cheap."""
    torch.manual_seed(0)
    return DiscreteLSTMPolicy(
        obs_dim=64, action_space=space, hidden_size=32,
    )


# ── Constants port ─────────────────────────────────────────────────────────


def test_risk_log_var_constants_match_v1():
    """The clamp bounds must port verbatim from v1; do NOT invent new ones.

    Plan: ``Locate RISK_LOG_VAR_MIN and RISK_LOG_VAR_MAX constants in
    agents/policy_network.py — port the existing values verbatim. Do
    not invent new bounds.``
    """
    assert RISK_LOG_VAR_MIN == V1_RISK_LOG_VAR_MIN
    assert RISK_LOG_VAR_MAX == V1_RISK_LOG_VAR_MAX


# ── Architecture shape ─────────────────────────────────────────────────────


def test_actor_input_dim_includes_bce_aux_columns(policy):
    """``actor_head[0]`` accepts ``runner_embed + hidden + 2`` features.

    The +2 columns are ``fill_prob`` and ``mature_prob`` per runner.
    ``risk_head`` does NOT add a third column — risk is a side-channel
    that surfaces on ``DiscretePolicyOutput`` for the NLL term, not an
    actor input. (Future readers: the third aux head is NOT a "third
    column" — see purpose.md §"What's locked".)
    """
    expected_in = policy.runner_embed_dim + policy.hidden_size + 2
    first_linear = policy.actor_head[0]
    assert isinstance(first_linear, nn.Linear)
    assert first_linear.weight.shape[1] == expected_in


def test_aux_heads_have_v1_shapes(policy, space):
    """``fill_prob_head`` / ``mature_prob_head`` / ``risk_head`` shapes
    must port v1 verbatim — ``Linear(hidden, R)`` for the BCE pair and
    ``Linear(hidden, R * 2)`` for risk.
    """
    R = space.max_runners
    H = policy.hidden_size
    assert policy.fill_prob_head.weight.shape == (R, H)
    assert policy.mature_prob_head.weight.shape == (R, H)
    assert policy.risk_head.weight.shape == (R * 2, H)


# ── Gradient flow into actor (forward-side perturb test) ──────────────────


def _action_logits_with_perturbed_param(
    policy, param_to_perturb, obs, mask, *, scale=1e-2,
):
    """Snapshot logits, perturb a parameter in-place, snapshot again."""
    with torch.no_grad():
        out_before = policy(obs, mask=mask)
        before = out_before.logits.detach().clone()

        delta = torch.randn_like(param_to_perturb) * scale
        param_to_perturb.add_(delta)
        try:
            out_after = policy(obs, mask=mask)
            after = out_after.logits.detach().clone()
        finally:
            # Restore so the policy fixture is unmodified for other tests.
            param_to_perturb.sub_(delta)
    return before, after


def test_action_logits_depend_on_fill_prob_head_weights(policy, space):
    """Perturbing ``fill_prob_head.weight`` must change the action logits.

    Forward-side gradient-through guard. If the head is accidentally
    detached from ``actor_head``'s input, this test trips.
    """
    torch.manual_seed(1)
    obs = torch.randn(2, policy.obs_dim)
    before, after = _action_logits_with_perturbed_param(
        policy, policy.fill_prob_head.weight, obs, mask=None,
    )
    # The per-runner OB / OL / CL slices should change. NOOP need not.
    per_runner_slice = slice(1, 1 + 3 * space.max_runners)
    assert not torch.allclose(
        before[:, per_runner_slice], after[:, per_runner_slice],
    )


def test_action_logits_depend_on_mature_prob_head_weights(policy, space):
    """Perturbing ``mature_prob_head.weight`` must change the action logits.

    Same intent as the fill_prob version — the strict-label head must
    also feed actor_head (CLAUDE.md §"mature_prob_head feeds actor_head").
    """
    torch.manual_seed(2)
    obs = torch.randn(2, policy.obs_dim)
    before, after = _action_logits_with_perturbed_param(
        policy, policy.mature_prob_head.weight, obs, mask=None,
    )
    per_runner_slice = slice(1, 1 + 3 * space.max_runners)
    assert not torch.allclose(
        before[:, per_runner_slice], after[:, per_runner_slice],
    )


def test_action_logits_do_NOT_depend_on_risk_head_weights(policy):
    """Perturbing ``risk_head.weight`` must NOT change the action logits.

    Symmetric guard: ``risk_head`` is a side-channel by design (purpose.md
    §"What's locked"). If it ever starts feeding actor_input this test
    trips and the operator chooses whether the change is intentional.
    """
    torch.manual_seed(3)
    obs = torch.randn(2, policy.obs_dim)
    before, after = _action_logits_with_perturbed_param(
        policy, policy.risk_head.weight, obs, mask=None,
    )
    assert torch.equal(before, after)


def test_actor_loss_routes_grad_through_fill_and_mature_heads(policy):
    """Backward-side complement: ``logits.sum().backward()`` produces
    non-None grads on both BCE heads. Pairs with the forward-side
    perturb tests to catch a partial-detach where the forward pass
    consumes the value but ``.detach()`` blocks gradient.
    """
    torch.manual_seed(4)
    obs = torch.randn(2, policy.obs_dim)
    # Make sure no grads are carried over from earlier tests.
    policy.zero_grad(set_to_none=True)
    out = policy(obs)
    out.logits.sum().backward()
    assert policy.fill_prob_head.weight.grad is not None
    assert policy.mature_prob_head.weight.grad is not None
    # And risk_head must NOT receive gradient from the actor path.
    assert policy.risk_head.weight.grad is None


# ── Risk-head output contract ─────────────────────────────────────────────


def test_risk_head_outputs_present_on_forward(policy, space):
    """``risk_mean`` and ``risk_log_var`` are present on ``DiscretePolicyOutput``
    with shape ``(batch, max_runners)`` each.
    """
    batch = 5
    obs = torch.randn(batch, policy.obs_dim)
    out = policy(obs)
    assert out.predicted_locked_pnl_per_runner.shape == (
        batch, space.max_runners,
    )
    assert out.predicted_locked_log_var_per_runner.shape == (
        batch, space.max_runners,
    )


def test_risk_log_var_is_clamped(policy, space):
    """Driving ``risk_head`` weights / bias to extreme values must not
    leak past the clamp bounds.

    Tests both directions — extreme positive should clamp to
    ``RISK_LOG_VAR_MAX``, extreme negative to ``RISK_LOG_VAR_MIN``. The
    clamp lives at the forward boundary so downstream consumers (UI,
    parquet, NLL) never see an unsafe value (purpose.md §"What's locked").
    """
    R = space.max_runners
    obs = torch.randn(3, policy.obs_dim)

    # ── Drive log_var channel positive ────────────────────────────
    # The risk_head output is laid out as (batch, R, 2); the log_var
    # channel is index 1. The flat Linear output's row-major layout puts
    # log_var rows at odd indices (0=mean_0, 1=log_var_0, 2=mean_1, ...).
    with torch.no_grad():
        for r in range(R):
            policy.risk_head.weight[2 * r + 1, :] = 0.0
            policy.risk_head.bias[2 * r + 1] = 1e6
        out = policy(obs)
    assert torch.all(
        out.predicted_locked_log_var_per_runner == RISK_LOG_VAR_MAX,
    )

    # ── Drive log_var channel negative ────────────────────────────
    with torch.no_grad():
        for r in range(R):
            policy.risk_head.bias[2 * r + 1] = -1e6
        out = policy(obs)
    assert torch.all(
        out.predicted_locked_log_var_per_runner == RISK_LOG_VAR_MIN,
    )


# ── State-dict break ──────────────────────────────────────────────────────


def _construct_pre_plan_state_dict(policy):
    """Synthesise a state_dict from a hypothetical pre-plan policy.

    Pre-Phase-7 ``DiscreteLSTMPolicy`` had a single
    ``logits_head: Linear(hidden, action_space.n)`` and none of the
    aux heads / per-runner actor / runner-slot embedding / noop_head.
    This helper builds a state_dict that mirrors that shape so the
    cross-load test can assert it raises.
    """
    # Carry the keys that did not change shape verbatim from the
    # current policy so the failure isolates to the new keys / the
    # vanished ``logits_head``.
    sd = OrderedDict()
    for name, tensor in policy.state_dict().items():
        if name.startswith(
            (
                "fill_prob_head.",
                "mature_prob_head.",
                "risk_head.",
                "actor_head.",
                "noop_head.",
                "runner_slot_embedding.",
            )
        ):
            continue
        sd[name] = tensor.detach().clone()
    # Inject the pre-plan flat ``logits_head`` (Linear(hidden, n)).
    n_actions = policy.action_space.n
    H = policy.hidden_size
    sd["logits_head.weight"] = torch.zeros(n_actions, H)
    sd["logits_head.bias"] = torch.zeros(n_actions)
    return sd


def test_pre_plan_weights_fail_to_load(policy):
    """Pre-plan v2 weights must raise on ``load_state_dict(strict=True)``.

    The architecture-hash break is intentional (purpose.md §"What's
    locked" — Architecture-hash break). The error message must mention
    BOTH the missing aux-head keys AND the vanished ``logits_head`` so
    operators carrying weights forward see the full picture.
    """
    pre_plan_sd = _construct_pre_plan_state_dict(policy)
    with pytest.raises(RuntimeError) as exc_info:
        policy.load_state_dict(pre_plan_sd, strict=True)
    msg = str(exc_info.value)
    # Missing keys — the new heads.
    assert "fill_prob_head" in msg
    assert "mature_prob_head" in msg
    assert "risk_head" in msg
    # The pre-plan ``logits_head`` key is unexpected in the new shape.
    assert "logits_head" in msg


# ── v1 ↔ v2 parity at fixed head weights ──────────────────────────────────


def test_v1_v2_head_parity_at_fixed_weights(policy, space):
    """Per-head parity: copy v1 head weights into v2, feed identical
    ``lstm_last``, assert head outputs match within fp32 epsilon.

    Subset rationale (per the session prompt's "compatible weights"
    instruction):

    The v1 ``PPOLSTMPolicy`` and v2 ``DiscreteLSTMPolicy`` have
    structurally-incompatible obs encoders (v1 splits obs into
    per-runner + per-market sub-tensors and runs separate MLPs; v2 uses
    a single ``Linear(obs_dim, hidden) → ReLU`` projection) AND
    structurally-incompatible action heads (v1 emits continuous Normal
    means over the 70-dim multi-action space; v2 emits a categorical
    over 1 + 3*R discrete actions). Action-logit parity is therefore
    impossible by construction.

    The three aux heads (``fill_prob_head``, ``mature_prob_head``,
    ``risk_head``) ARE structurally identical between v1 and v2 — each
    is a single ``nn.Linear`` from the LSTM backbone output. So given
    the same head weights and the same backbone-output tensor, the
    head outputs must match to fp32 epsilon. That's the load-bearing
    spec the parity test exists to guard.
    """
    R = space.max_runners
    H = policy.hidden_size

    # Build v1-shape head weights (no need to construct a real
    # PPOLSTMPolicy — its head shapes are part of the spec, ported
    # verbatim into v2).
    torch.manual_seed(42)
    v1_fill_w = torch.randn(R, H)
    v1_fill_b = torch.randn(R)
    v1_mature_w = torch.randn(R, H)
    v1_mature_b = torch.randn(R)
    v1_risk_w = torch.randn(R * 2, H)
    v1_risk_b = torch.randn(R * 2)

    # Copy into v2.
    with torch.no_grad():
        policy.fill_prob_head.weight.copy_(v1_fill_w)
        policy.fill_prob_head.bias.copy_(v1_fill_b)
        policy.mature_prob_head.weight.copy_(v1_mature_w)
        policy.mature_prob_head.bias.copy_(v1_mature_b)
        policy.risk_head.weight.copy_(v1_risk_w)
        policy.risk_head.bias.copy_(v1_risk_b)

    # Synthesise an ``lstm_last`` tensor and run both v1's and v2's
    # head arithmetic against it.
    batch = 4
    lstm_last = torch.randn(batch, H)

    # ── v1 reference arithmetic (ported from agents/policy_network.py
    # lines 800-810 + 851-856) ──
    v1_fill_logit = lstm_last @ v1_fill_w.T + v1_fill_b
    v1_fill_prob = torch.sigmoid(v1_fill_logit)
    v1_mature_logit = lstm_last @ v1_mature_w.T + v1_mature_b
    v1_mature_prob = torch.sigmoid(v1_mature_logit)
    v1_risk_out = (lstm_last @ v1_risk_w.T + v1_risk_b).view(batch, R, 2)
    v1_risk_mean = v1_risk_out[..., 0]
    v1_risk_log_var = v1_risk_out[..., 1].clamp(
        V1_RISK_LOG_VAR_MIN, V1_RISK_LOG_VAR_MAX,
    )

    # ── v2 head arithmetic (run modules directly on lstm_last) ──
    with torch.no_grad():
        v2_fill_prob = torch.sigmoid(policy.fill_prob_head(lstm_last))
        v2_mature_prob = torch.sigmoid(policy.mature_prob_head(lstm_last))
        v2_risk_out = policy.risk_head(lstm_last).view(batch, R, 2)
        v2_risk_mean = v2_risk_out[..., 0]
        v2_risk_log_var = v2_risk_out[..., 1].clamp(
            RISK_LOG_VAR_MIN, RISK_LOG_VAR_MAX,
        )

    eps = 1e-6
    assert torch.allclose(v1_fill_prob, v2_fill_prob, atol=eps)
    assert torch.allclose(v1_mature_prob, v2_mature_prob, atol=eps)
    assert torch.allclose(v1_risk_mean, v2_risk_mean, atol=eps)
    assert torch.allclose(v1_risk_log_var, v2_risk_log_var, atol=eps)
