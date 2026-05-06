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

import numpy as np
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
    """``actor_head[0]`` accepts ``runner_embed + hidden + 4`` features.

    The +4 columns are ``fill_prob``, ``mature_prob`` (Phase 7 S01) +
    ``direction_back_prob``, ``direction_lay_prob`` (phase-13 S03)
    per runner. ``risk_head`` does NOT add columns — risk is a
    side-channel that surfaces on ``DiscretePolicyOutput`` for the
    NLL term, not an actor input.
    """
    expected_in = policy.runner_embed_dim + policy.hidden_size + 4
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


# ── Phase 7 Session 02 ─────────────────────────────────────────────────────
#
# S02 wires the three aux losses into ``DiscretePPOTrainer`` and adds the
# load-bearing fix for the v1↔v2 hp-dict-precedence trap (see
# ``plans/rewrite/phase-7-port-aux-heads/lessons_learnt.md``).


from env.bet_manager import Bet, BetSide  # noqa: E402
from training_v2.discrete_ppo.aux_labels import (  # noqa: E402
    AUX_LABEL_COMMISSION,
    PerRunnerAuxLabels,
    compute_pair_locked_pnl,
    compute_per_runner_aux_labels,
)


def _make_bet(
    *,
    selection_id: int,
    side: BetSide,
    matched_stake: float,
    average_price: float,
    market_id: str = "M",
    pair_id: str = "P",
    force_close: bool = False,
    close_leg: bool = False,
) -> Bet:
    return Bet(
        selection_id=int(selection_id),
        side=side,
        requested_stake=float(matched_stake),
        matched_stake=float(matched_stake),
        average_price=float(average_price),
        market_id=str(market_id),
        pair_id=str(pair_id),
        force_close=bool(force_close),
        close_leg=bool(close_leg),
    )


# ── Pure label-helper tests ───────────────────────────────────────────────


def test_risk_label_arithmetic_matches_v1():
    """Locked-P&L on a hand-computed fixture.

    Fixture from session prompt §"Tests": BACK £10 @ 5.0, LAY £8 @ 4.5,
    commission 0.05. Hand computation:

        win_pnl  = 10 * (5.0 - 1.0) * (1 - 0.05) - 8 * (4.5 - 1)
                 = 10 * 4 * 0.95 - 8 * 3.5
                 = 38.0 - 28.0
                 = 10.0
        lose_pnl = -10 + 8 * (1 - 0.05)
                 = -10 + 7.6
                 = -2.4
        locked   = max(0, min(10.0, -2.4)) = 0.0

    Anchors the v2 port to v1 ``agents/ppo_trainer.py:1696-1712``.
    """
    legs = [
        _make_bet(
            selection_id=1, side=BetSide.BACK,
            matched_stake=10.0, average_price=5.0,
        ),
        _make_bet(
            selection_id=1, side=BetSide.LAY,
            matched_stake=8.0, average_price=4.5,
        ),
    ]
    expected = 0.0  # see docstring derivation.
    assert compute_pair_locked_pnl(legs) == pytest.approx(expected)
    assert AUX_LABEL_COMMISSION == 0.05


def test_risk_label_arithmetic_positive_lock():
    """Fixture chosen so locked > 0 — sanity check on the algebra.

    BACK £10 @ 5.0, LAY £11 @ 4.0:
        win_pnl  = 10 * 4 * 0.95 - 11 * 3 = 38 - 33 = 5.0
        lose_pnl = -10 + 11 * 0.95        = -10 + 10.45 = 0.45
        locked   = max(0, min(5.0, 0.45)) = 0.45
    """
    legs = [
        _make_bet(
            selection_id=1, side=BetSide.BACK,
            matched_stake=10.0, average_price=5.0,
        ),
        _make_bet(
            selection_id=1, side=BetSide.LAY,
            matched_stake=11.0, average_price=4.0,
        ),
    ]
    assert compute_pair_locked_pnl(legs) == pytest.approx(0.45, abs=1e-6)


def test_risk_label_naked_pair_returns_nan_in_per_runner_aggregation():
    """A pair with only one matched leg → naked → risk label is NaN.

    The aggregation function emits NaN at slots with no completed pair;
    the trainer's NLL term masks NaN out via ``risk_mask``.
    """
    legs = [
        _make_bet(
            selection_id=1, side=BetSide.BACK,
            matched_stake=10.0, average_price=5.0,
            pair_id="naked",
        ),
    ]
    market_to_runner_map = {"M": {1: 0}}
    out = compute_per_runner_aux_labels(legs, market_to_runner_map, max_runners=4)
    assert np.isnan(out.risk_label[0])
    assert out.risk_mask[0] is np.bool_(False) or out.risk_mask[0] == False  # noqa: E712
    # Naked pair counts toward runner_mask (slot had a matched leg) but
    # NOT toward risk_mask (no completed pair on that slot).
    assert out.runner_mask[0] == True  # noqa: E712
    assert out.fill_label[0] == 0.0
    assert out.mature_label[0] == 0.0


def test_strict_mature_label_excludes_force_closes():
    """Three pairs (matured, agent-closed, force-closed) → labels [1, 1, 0].

    Load-bearing semantic test for the strict mature_prob label
    contract — force-closed pairs land in the negative class even
    though both legs matched (CLAUDE.md §"mature_prob_head feeds
    actor_head").
    """
    bets: list[Bet] = []
    # Pair 0 — matured naturally on slot 0.
    bets.append(_make_bet(
        selection_id=10, side=BetSide.BACK, matched_stake=5.0,
        average_price=3.0, pair_id="p0",
    ))
    bets.append(_make_bet(
        selection_id=10, side=BetSide.LAY, matched_stake=5.0,
        average_price=2.9, pair_id="p0",
    ))
    # Pair 1 — agent-closed on slot 1 (close_leg=True, force_close=False).
    bets.append(_make_bet(
        selection_id=20, side=BetSide.BACK, matched_stake=5.0,
        average_price=3.0, pair_id="p1",
    ))
    bets.append(_make_bet(
        selection_id=20, side=BetSide.LAY, matched_stake=5.0,
        average_price=2.9, pair_id="p1",
        close_leg=True,  # agent-closed, NOT force_close.
    ))
    # Pair 2 — env force-closed on slot 2.
    bets.append(_make_bet(
        selection_id=30, side=BetSide.BACK, matched_stake=5.0,
        average_price=3.0, pair_id="p2",
    ))
    bets.append(_make_bet(
        selection_id=30, side=BetSide.LAY, matched_stake=5.0,
        average_price=2.9, pair_id="p2",
        force_close=True, close_leg=True,
    ))

    market_to_runner_map = {"M": {10: 0, 20: 1, 30: 2}}
    out = compute_per_runner_aux_labels(
        bets, market_to_runner_map, max_runners=4,
    )
    assert out.mature_label[0] == 1.0  # matured naturally
    assert out.mature_label[1] == 1.0  # agent-closed → positive class
    assert out.mature_label[2] == 0.0  # force-closed → negative class
    assert out.mature_label[3] == 0.0  # no pair on slot 3
    # All three pairs have matched_legs >= 2 so fill_label = 1.0 on
    # slots 0/1/2.
    assert out.fill_label[0] == 1.0
    assert out.fill_label[1] == 1.0
    assert out.fill_label[2] == 1.0
    assert out.fill_label[3] == 0.0


# ── Trainer hp-dict precedence (load-bearing regression guard) ─────────────


def _build_minimal_trainer(hp: dict | None = None):
    """Build a DiscretePPOTrainer + tiny dummy shim without exercising the env.

    The trainer's hp-dict reads happen in ``__init__`` BEFORE any
    rollout, so we don't need a real env / shim — only enough surface
    for the constructor to succeed (action_space, max_runners,
    .env, .obs_dim).
    """
    from agents_v2.action_space import DiscreteActionSpace
    from agents_v2.discrete_policy import DiscreteLSTMPolicy
    from training_v2.discrete_ppo.trainer import DiscretePPOTrainer

    space = DiscreteActionSpace(max_runners=4)

    class _StubShim:
        def __init__(self):
            self.action_space = space
            self.max_runners = space.max_runners
            self.obs_dim = 8
            self.env = None  # collector tries to read this but only on rollout

    shim = _StubShim()
    policy = DiscreteLSTMPolicy(
        obs_dim=shim.obs_dim, action_space=space, hidden_size=16,
    )
    trainer = DiscretePPOTrainer(
        policy=policy, shim=shim, hp=hp, device="cpu",
    )
    return trainer


def test_trainer_reads_loss_weights_from_hp_dict():
    """``hp[name]`` is the source of truth for the three weights."""
    hp = {
        "fill_prob_loss_weight": 0.1,
        "mature_prob_loss_weight": 0.2,
        "risk_loss_weight": 0.3,
    }
    trainer = _build_minimal_trainer(hp=hp)
    assert trainer.fill_prob_loss_weight == 0.1
    assert trainer.mature_prob_loss_weight == 0.2
    assert trainer.risk_loss_weight == 0.3


def test_trainer_defaults_loss_weights_to_zero_when_hp_absent():
    """No hp dict → all three weights default to 0.0 (byte-identical)."""
    trainer = _build_minimal_trainer(hp=None)
    assert trainer.fill_prob_loss_weight == 0.0
    assert trainer.mature_prob_loss_weight == 0.0
    assert trainer.risk_loss_weight == 0.0


def test_trainer_does_NOT_consult_config_reward_fallback():
    """Forward-looking guard against re-introducing the v1 precedence trap.

    Read the trainer's source and assert the loss-weight reads do NOT
    cascade into a ``config["reward"][...]`` lookup. v1's pattern
    silently swallows ``--reward-overrides`` under v2's always-populated
    hp dict; copying it would un-ship the entire phase-7 work.
    """
    import inspect
    from training_v2.discrete_ppo import trainer as trainer_mod
    src = inspect.getsource(trainer_mod.DiscretePPOTrainer.__init__)
    # Heuristic: a v1-style fallback would read ``config.get("reward",``
    # somewhere near the loss-weight reads. The trainer constructor
    # never reads `config` at all in v2 (it takes individual kwargs).
    assert 'config["reward"]' not in src
    assert 'config.get("reward"' not in src


# ── Worker pre-merge (Path A) — load-bearing integration test ─────────────


@pytest.mark.parametrize("key", [
    "fill_prob_loss_weight",
    "mature_prob_loss_weight",
    "risk_loss_weight",
])
def test_reward_overrides_reaches_trainer_via_real_genes_flow(key):
    """``--reward-overrides <key>=0.5`` reaches ``trainer.<key>`` even
    when the gene's default is 0.0 in ``CohortGenes.to_dict()``.

    This is the **load-bearing regression guard** for the v1↔v2
    precedence trap (purpose.md §"Trainer reads weights..."). The
    test must use the real ``CohortGenes`` → ``hp`` flow because the
    failure mode is specifically v2's always-populated hp dict — a
    hand-constructed hp dict would not exercise it.
    """
    from training_v2.cohort.genes import CohortGenes
    from training_v2.cohort.worker import _build_trainer_hp

    genes = CohortGenes(
        learning_rate=3e-4,
        entropy_coeff=0.01,
        clip_range=0.2,
        gae_lambda=0.95,
        value_coeff=0.5,
        mini_batch_size=64,
        hidden_size=64,
    )
    # Sanity — the gene's default is 0.0, and to_dict() carries it.
    assert genes.to_dict()[key] == 0.0

    hp = _build_trainer_hp(
        cohort_overrides={key: 0.5},
        genes=genes,
        enabled_set=frozenset(),  # not enabled — gene default is 0.0
    )
    # The override survived the merge — NOT swallowed by the gene
    # default.
    assert hp[key] == 0.5

    # And the trainer reads it from hp.
    trainer = _build_minimal_trainer(hp=hp)
    assert getattr(trainer, key) == 0.5


def test_build_trainer_hp_gene_value_used_when_no_override():
    """No cohort override + gene enabled → hp carries the gene draw."""
    from training_v2.cohort.genes import CohortGenes
    from training_v2.cohort.worker import _build_trainer_hp

    genes = CohortGenes(
        learning_rate=3e-4,
        entropy_coeff=0.01,
        clip_range=0.2,
        gae_lambda=0.95,
        value_coeff=0.5,
        mini_batch_size=64,
        hidden_size=64,
        mature_prob_loss_weight=0.25,
    )
    hp = _build_trainer_hp(
        cohort_overrides=None,
        genes=genes,
        enabled_set=frozenset({"mature_prob_loss_weight"}),
    )
    assert hp["mature_prob_loss_weight"] == 0.25


# ── _compute_aux_losses behavioural tests ──────────────────────────────────


def _aux_label_pack(
    *,
    fill_label: list[float],
    mature_label: list[float],
    risk_label: list[float],
    runner_mask: list[bool],
    risk_mask: list[bool],
) -> PerRunnerAuxLabels:
    return PerRunnerAuxLabels(
        fill_label=np.asarray(fill_label, dtype=np.float32),
        mature_label=np.asarray(mature_label, dtype=np.float32),
        risk_label=np.asarray(risk_label, dtype=np.float32),
        runner_mask=np.asarray(runner_mask, dtype=bool),
        risk_mask=np.asarray(risk_mask, dtype=bool),
    )


def _aux_loss_inputs(trainer, labels: PerRunnerAuxLabels):
    """Run a forward pass on synthetic obs and return (out, label tensors)."""
    torch.manual_seed(7)
    obs = torch.randn(3, trainer.policy.obs_dim)  # batch=3
    out = trainer.policy(obs)
    device = obs.device
    fill_label_t = torch.from_numpy(labels.fill_label).to(device)
    mature_label_t = torch.from_numpy(labels.mature_label).to(device)
    risk_label_t = torch.from_numpy(
        np.nan_to_num(labels.risk_label, nan=0.0)
    ).to(device)
    runner_mask_t = torch.from_numpy(
        labels.runner_mask.astype(np.float32)
    ).to(device)
    risk_mask_t = torch.from_numpy(
        labels.risk_mask.astype(np.float32)
    ).to(device)
    return out, (
        fill_label_t, mature_label_t, risk_label_t,
        runner_mask_t, risk_mask_t,
    )


def test_compute_aux_losses_zero_when_all_runners_unmasked():
    """Empty rollout (no matched pairs anywhere) → BCE denom guarded.

    The BCE term's masked-mean denominator clamps to ``1.0`` when the
    runner mask is all-False, so the loss is ``(0 / 1) = 0`` rather
    than ``0 / 0 = NaN``. Risk loss takes the explicit "no completed
    pair" early-out and returns scalar 0.
    """
    trainer = _build_minimal_trainer(hp={
        "fill_prob_loss_weight": 0.5,
        "mature_prob_loss_weight": 0.5,
        "risk_loss_weight": 0.5,
    })
    R = trainer.max_runners
    labels = _aux_label_pack(
        fill_label=[0.0] * R,
        mature_label=[0.0] * R,
        risk_label=[float("nan")] * R,
        runner_mask=[False] * R,
        risk_mask=[False] * R,
    )
    out, lts = _aux_loss_inputs(trainer, labels)
    fill_loss, mature_loss, risk_loss = trainer._compute_aux_losses(
        policy_out=out,
        fill_label=lts[0], mature_label=lts[1], risk_label=lts[2],
        runner_mask=lts[3], risk_mask=lts[4],
    )
    assert torch.isfinite(fill_loss)
    assert torch.isfinite(mature_loss)
    assert torch.isfinite(risk_loss)
    assert float(fill_loss.item()) == 0.0
    assert float(mature_loss.item()) == 0.0
    assert float(risk_loss.item()) == 0.0


def test_compute_aux_losses_bce_nonzero_when_label_disagrees_with_pred():
    """Labels = 1 + sigmoid output near 0.5 → BCE ≈ -log(0.5) ≈ 0.693.

    Establishes that the BCE term moves with the labels. The fresh-init
    sigmoid output is near 0.5 (Linear weights ~ 0 init → logit ~ 0 →
    prob ~ 0.5), so a target of 1.0 produces a non-trivial BCE.
    """
    trainer = _build_minimal_trainer(hp={
        "fill_prob_loss_weight": 0.5,
        "mature_prob_loss_weight": 0.5,
        "risk_loss_weight": 0.0,
    })
    R = trainer.max_runners
    labels = _aux_label_pack(
        fill_label=[1.0] * R,
        mature_label=[1.0] * R,
        risk_label=[float("nan")] * R,
        runner_mask=[True] * R,
        risk_mask=[False] * R,
    )
    out, lts = _aux_loss_inputs(trainer, labels)
    fill_loss, mature_loss, _ = trainer._compute_aux_losses(
        policy_out=out,
        fill_label=lts[0], mature_label=lts[1], risk_label=lts[2],
        runner_mask=lts[3], risk_mask=lts[4],
    )
    assert float(fill_loss.item()) > 0.1
    assert float(mature_loss.item()) > 0.1


def test_compute_aux_losses_risk_nll_nonzero_when_completed_pair_present():
    """Risk NLL on a fixture with one completed pair → non-zero.

    Lock in that the NLL term contributes when at least one slot has
    a real label. The NLL has the form
    ``0.5 * (diff^2/exp(log_var) + log_var)``; with fresh-init
    log_var ~ 0 and diff non-trivial, the term is positive.
    """
    trainer = _build_minimal_trainer(hp={
        "fill_prob_loss_weight": 0.0,
        "mature_prob_loss_weight": 0.0,
        "risk_loss_weight": 0.5,
    })
    R = trainer.max_runners
    risk_label_arr = [float("nan")] * R
    risk_mask_arr = [False] * R
    risk_label_arr[1] = 5.0          # one slot has a real label
    risk_mask_arr[1] = True
    labels = _aux_label_pack(
        fill_label=[0.0] * R,
        mature_label=[0.0] * R,
        risk_label=risk_label_arr,
        runner_mask=[False] * R,
        risk_mask=risk_mask_arr,
    )
    out, lts = _aux_loss_inputs(trainer, labels)
    _, _, risk_loss = trainer._compute_aux_losses(
        policy_out=out,
        fill_label=lts[0], mature_label=lts[1], risk_label=lts[2],
        runner_mask=lts[3], risk_mask=lts[4],
    )
    assert float(risk_loss.item()) != 0.0
    assert torch.isfinite(risk_loss)


def test_compute_aux_losses_risk_nll_skipped_when_all_naked():
    """All-naked rollout → risk NLL returns scalar 0 without NaN.

    The trainer's ``_compute_aux_losses`` short-circuits when
    ``risk_mask.sum() == 0`` so no division-by-zero / NaN can leak
    into total_loss. Regression guard for the
    "test_risk_nll_zero_when_no_completed_pairs_in_minibatch" case.
    """
    trainer = _build_minimal_trainer(hp={
        "fill_prob_loss_weight": 0.0,
        "mature_prob_loss_weight": 0.0,
        "risk_loss_weight": 0.5,
    })
    R = trainer.max_runners
    labels = _aux_label_pack(
        fill_label=[1.0] * R,         # naked pair counts toward fill
        mature_label=[0.0] * R,
        risk_label=[float("nan")] * R,
        runner_mask=[True] * R,
        risk_mask=[False] * R,        # no completed pairs anywhere
    )
    out, lts = _aux_loss_inputs(trainer, labels)
    _, _, risk_loss = trainer._compute_aux_losses(
        policy_out=out,
        fill_label=lts[0], mature_label=lts[1], risk_label=lts[2],
        runner_mask=lts[3], risk_mask=lts[4],
    )
    assert float(risk_loss.item()) == 0.0
    assert torch.isfinite(risk_loss)
