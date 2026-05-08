"""Regression tests for the phase-13 S03 direction-prob head.

Mirrors the four-test pattern from v1's ``TestMatureProbInActor`` for
the v2 :class:`agents_v2.discrete_policy.DiscreteLSTMPolicy`:

1. ``test_actor_input_includes_direction_prob`` —
   ``actor_head[0].weight.shape[1] == runner_embed + hidden + 4``.
2. ``test_action_logits_depend_on_direction_prob_head_weights`` —
   forward-side gradient guard. Perturbing the direction head's
   weight changes the actor logits.
3. ``test_actor_loss_routes_grad_through_direction_prob_head`` —
   backward-side guard. Loss on the actor logits produces non-None
   gradient on ``direction_prob_head.weight``.
4. ``test_pre_direction_weights_fail_to_load`` — old state_dict
   (post-mature, two-narrower ``actor_head[0].weight``) raises on
   strict load. Architecture-hash break is the variant identity.
"""

from __future__ import annotations

import pytest
import torch

from agents_v2.action_space import DiscreteActionSpace
from agents_v2.discrete_policy import DiscreteLSTMPolicy
from env.betfair_env import MARKET_DIM, RUNNER_DIM, VELOCITY_DIM


# Phase-15 S01: obs_dim must accommodate the env's runner block
# (``MARKET_DIM + VELOCITY_DIM + max_runners * RUNNER_DIM``)
# because ``direction_prob_head`` now slices the per-runner
# feature block out of obs directly. Pre-phase-15 tests used
# ``_OBS_DIM = 64`` which was sufficient when the head read only
# ``lstm_last`` + slot embedding.
_MAX_RUNNERS = 5
_HIDDEN = 32
_RUNNER_BLOCK = MARKET_DIM + VELOCITY_DIM + _MAX_RUNNERS * RUNNER_DIM
_OBS_DIM = _RUNNER_BLOCK + 32  # +32 for agent_state + position tail


def _build_policy() -> DiscreteLSTMPolicy:
    action_space = DiscreteActionSpace(max_runners=_MAX_RUNNERS)
    return DiscreteLSTMPolicy(
        obs_dim=_OBS_DIM,
        action_space=action_space,
        hidden_size=_HIDDEN,
    )


class TestDirectionProbInActor:

    # 1. shape ---------------------------------------------------------------

    def test_actor_input_includes_direction_prob(self):
        p = _build_policy()
        expected = p.runner_embed_dim + p.hidden_size + 4
        actual = p.actor_head[0].weight.shape[1]
        assert actual == expected, (
            f"actor_head[0].weight.shape[1] = {actual}, expected "
            f"runner_embed ({p.runner_embed_dim}) + hidden "
            f"({p.hidden_size}) + 4 = {expected}"
        )

    def test_direction_prob_head_is_per_runner_mlp(self):
        """Phase-15 S01: head is a 2-layer MLP per slot fed the
        runner's RAW feature slice.

        First Linear: ``(actor_mlp_hidden, RUNNER_DIM)``.
        Second Linear: ``(2, actor_mlp_hidden)`` — emits
        ``[direction_back_logit, direction_lay_logit]`` per slot.
        Replaces phase-14's
        ``Linear(actor_mlp_hidden, runner_embed + hidden)`` first-
        layer shape — the bottleneck-bypass fix.
        """
        p = _build_policy()
        # First layer: input dim is RUNNER_DIM (the runner's raw
        # feature slice), NOT runner_embed + hidden.
        assert p.direction_prob_head[0].weight.shape == (
            p.actor_mlp_hidden, RUNNER_DIM,
        )
        # ReLU at index 1 (no weights).
        # Second layer: 2 outputs per slot.
        assert p.direction_prob_head[2].weight.shape == (
            2, p.actor_mlp_hidden,
        )

    # 2. forward-side gradient guard ----------------------------------------

    def test_action_logits_depend_on_direction_prob_head_weights(self):
        p = _build_policy()
        torch.manual_seed(0)
        obs = torch.randn(2, _OBS_DIM)
        with torch.no_grad():
            out_a = p(obs)
            logits_a = out_a.logits.detach().clone()
            # Perturb the direction head's first-layer weight and
            # re-run. Phase-14 S01: the head is a 2-layer MLP, so
            # we perturb the first Linear.
            p.direction_prob_head[0].weight.add_(0.5)
            out_b = p(obs)
            logits_b = out_b.logits.detach().clone()
        assert not torch.allclose(logits_a, logits_b), (
            "Perturbing direction_prob_head[0].weight did not change "
            "actor logits — the head's output is not feeding "
            "actor_head."
        )

    # 3. backward-side guard ------------------------------------------------

    def test_actor_loss_routes_grad_through_direction_prob_head(self):
        p = _build_policy()
        torch.manual_seed(0)
        obs = torch.randn(2, _OBS_DIM)
        out = p(obs)
        loss = out.logits.sum()
        loss.backward()
        # Phase-14 S01: direction_prob_head is a Sequential MLP.
        # Both layers' weights must receive gradient (path is not
        # detached).
        for i in (0, 2):
            grad = p.direction_prob_head[i].weight.grad
            assert grad is not None, (
                f"direction_prob_head[{i}].weight has no gradient — "
                "surrogate path appears detached."
            )
            assert grad.abs().max() > 0.0

    # 4. cross-load failure: pre-phase-13 (single-Linear-on-actor) ----------

    def test_pre_phase13_actor_weights_fail_to_load(self):
        """A pre-phase-13 checkpoint had ``actor_head[0].weight``
        shape ``(hidden, runner_embed + hidden + 2)`` (only fill_prob
        + mature_prob columns). Phase-13 widened it to +4.

        This test pins that the +4 actor input shape is enforced —
        if a pre-phase-13 (or older) checkpoint sneaks in, the
        load fails.
        """
        p = _build_policy()
        sd = {k: v.detach().clone() for k, v in p.state_dict().items()}
        old_w = sd["actor_head.0.weight"]
        new_in = old_w.shape[1]  # runner_embed + hidden + 4
        old_in = new_in - 2  # pre-direction-prob width (post-mature)
        sd["actor_head.0.weight"] = torch.zeros(old_w.shape[0], old_in)
        with pytest.raises((RuntimeError, ValueError)) as excinfo:
            p.load_state_dict(sd, strict=True)
        msg = str(excinfo.value).lower()
        assert (
            "actor_head" in msg
            or "size mismatch" in msg
            or "shape" in msg
        ), (
            f"Expected a shape-mismatch on actor_head, got: {excinfo.value}"
        )

    # 4b. cross-load failure: phase-13 (single-Linear direction head) -------

    def test_pre_phase14_direction_head_fails_to_load(self):
        """Phase-14 S01 changed ``direction_prob_head`` from
        ``Linear(hidden, max_runners*2)`` to a 2-layer per-runner MLP.
        Pre-S01 (i.e. phase-13 S03) checkpoints have
        ``direction_prob_head.weight`` shape ``(R*2, hidden)``;
        post-S01 has ``direction_prob_head.0.weight`` at a
        per-runner-MLP shape. Strict load rejects the cross.

        The architecture-hash break is the variant identity (no
        explicit version field). Same protocol as fill-prob-in-actor /
        mature-prob-in-actor / phase-13 S03.
        """
        p = _build_policy()
        sd = {k: v.detach().clone() for k, v in p.state_dict().items()}
        # Phase-13 had direction_prob_head as a single Linear:
        # ``direction_prob_head.weight`` of shape ``(R*2, hidden)``.
        # Phase-14 / phase-15 have direction_prob_head as
        # nn.Sequential, so the state_dict keys are
        # ``direction_prob_head.0.weight`` etc. Pre-S01 state_dicts
        # will have the OLD ``.weight`` key (no index).
        del sd["direction_prob_head.0.weight"]
        del sd["direction_prob_head.0.bias"]
        del sd["direction_prob_head.2.weight"]
        del sd["direction_prob_head.2.bias"]
        sd["direction_prob_head.weight"] = torch.zeros(
            p.max_runners * 2, p.hidden_size,
        )
        sd["direction_prob_head.bias"] = torch.zeros(p.max_runners * 2)
        with pytest.raises((RuntimeError, ValueError)) as excinfo:
            p.load_state_dict(sd, strict=True)
        msg = str(excinfo.value).lower()
        # The error message will mention the missing or unexpected
        # keys; both indicate the cross-load is correctly refused.
        assert (
            "direction_prob_head" in msg
            or "missing" in msg
            or "unexpected" in msg
        ), (
            f"Expected a key/shape mismatch on direction_prob_head, "
            f"got: {excinfo.value}"
        )

    # 4c. cross-load failure: pre-phase-15 (LSTM-bottleneck input) -----------

    def test_pre_phase15_direction_head_fails_to_load(self):
        """Phase-15 S01 changed ``direction_prob_head[0]``'s input
        dim from ``runner_embed + hidden`` (phase-14) to
        ``RUNNER_DIM`` (phase-15). A pre-phase-15 state_dict has
        the wider first-layer weight; strict load must reject it.

        This is the load-bearing arch-hash guard for phase 15 —
        the variant identity is carried by the weight shape, not
        an explicit version field. Same protocol as phase-7
        (fill-prob), phase-13 (direction-prob), phase-14
        (per-runner direction MLP).
        """
        p = _build_policy()
        sd = {k: v.detach().clone() for k, v in p.state_dict().items()}
        # Pre-phase-15 first-layer weight had input dim
        # ``runner_embed_dim + hidden_size``. Substitute in a
        # tensor of that older width.
        old_in = p.runner_embed_dim + p.hidden_size
        sd["direction_prob_head.0.weight"] = torch.zeros(
            p.actor_mlp_hidden, old_in,
        )
        with pytest.raises((RuntimeError, ValueError)) as excinfo:
            p.load_state_dict(sd, strict=True)
        msg = str(excinfo.value).lower()
        assert (
            "direction_prob_head" in msg or "size mismatch" in msg
            or "shape" in msg
        ), (
            f"Expected a shape-mismatch on "
            f"direction_prob_head.0.weight, got: {excinfo.value}"
        )

    # 5. default-weight byte-identity to ~0.5 (sanity) ---------------------

    def test_direction_outputs_near_05_on_fresh_init(self):
        """Untrained sigmoid output sits near 0.5; the actor_input
        column from a fresh head is therefore ~constant 0.5 noise.
        """
        p = _build_policy()
        torch.manual_seed(0)
        obs = torch.zeros(1, _OBS_DIM)
        with torch.no_grad():
            out = p(obs)
        # On a zero-init weight matrix sigmoid output is exactly 0.5
        # plus the bias contribution; we just assert the output is
        # bounded, finite, and shape-correct.
        prob_back = out.direction_back_prob_per_runner
        prob_lay = out.direction_lay_prob_per_runner
        assert prob_back.shape == (1, _MAX_RUNNERS)
        assert prob_lay.shape == (1, _MAX_RUNNERS)
        assert torch.all(torch.isfinite(prob_back))
        assert torch.all((prob_back > 0.0) & (prob_back < 1.0))
        assert torch.all((prob_lay > 0.0) & (prob_lay < 1.0))


class TestPhase15FeatureSliceInput:
    """Phase-15 S01 regression guards.

    The direction head now reads each runner's RAW per-runner
    feature slice (``RUNNER_DIM`` floats from obs) instead of
    ``concat([slot_emb_i, lstm_last])``. These tests pin the new
    contract: the head consumes the per-runner slice, does NOT
    depend on ``lstm_last``, and does not mix features across
    runners. See ``plans/rewrite/phase-15-direction-head-feature-
    slice/hard_constraints.md §6``.
    """

    # Per-runner consumption: perturbing runner i's feature block
    # changes runner i's logit but leaves runner j's logit alone.
    def test_direction_head_consumes_runner_feature_slice(self):
        p = _build_policy()
        torch.manual_seed(0)
        obs_a = torch.randn(1, _OBS_DIM)

        with torch.no_grad():
            out_a = p(obs_a)
            back_a = out_a.direction_back_logits_per_runner.clone()

            # Perturb runner 0's feature slice only.
            i = 0
            j = 2
            obs_b = obs_a.clone()
            run_off = p._runner_block_offset
            i_start = run_off + i * RUNNER_DIM
            i_end = i_start + RUNNER_DIM
            obs_b[:, i_start:i_end] += 1.5  # large perturbation

            out_b = p(obs_b)
            back_b = out_b.direction_back_logits_per_runner

            # Runner i's logit MUST change.
            assert not torch.allclose(back_a[:, i], back_b[:, i]), (
                f"Perturbing runner {i}'s feature slice did not "
                f"change runner {i}'s direction logit — head is not "
                "consuming the per-runner slice."
            )
            # Runner j's logit MUST NOT change (no cross-runner mix).
            assert torch.allclose(back_a[:, j], back_b[:, j]), (
                f"Perturbing runner {i}'s feature slice changed "
                f"runner {j}'s direction logit — feature slice is "
                "leaking across runners."
            )

    # Bypass guard: perturbing the LSTM weights changes lstm_last,
    # but direction logits stay fixed because the head no longer
    # reads lstm_last (hard_constraints §2).
    def test_direction_head_does_not_depend_on_lstm_last(self):
        p = _build_policy()
        torch.manual_seed(0)
        obs = torch.randn(1, _OBS_DIM)

        with torch.no_grad():
            out_a = p(obs)
            back_a = out_a.direction_back_logits_per_runner.clone()
            lay_a = out_a.direction_lay_logits_per_runner.clone()

            # Perturb the LSTM's recurrent weight. Any non-zero
            # change to lstm_last should leave direction logits
            # untouched if the bypass is honoured.
            for name, param in p.lstm.named_parameters():
                if "weight" in name:
                    param.add_(0.5)

            out_b = p(obs)
            back_b = out_b.direction_back_logits_per_runner
            lay_b = out_b.direction_lay_logits_per_runner

            # Sanity: lstm_last DID change (otherwise the test is
            # vacuous). Use the actor logits as a proxy — they
            # depend on lstm_last via actor_head's input.
            assert not torch.allclose(out_a.logits, out_b.logits), (
                "LSTM perturbation did not propagate through "
                "actor_head — test is vacuous."
            )
            # The head must NOT have moved.
            assert torch.allclose(back_a, back_b), (
                "Perturbing the LSTM changed direction_back_logits "
                "— head is still reading lstm_last (hard_"
                "constraints §2 violation)."
            )
            assert torch.allclose(lay_a, lay_b), (
                "Perturbing the LSTM changed direction_lay_logits "
                "— head is still reading lstm_last (hard_"
                "constraints §2 violation)."
            )

    # First-layer weight shape pinned to RUNNER_DIM. Belt-and-
    # braces complement to test_direction_prob_head_is_per_runner_mlp.
    def test_direction_head_first_layer_input_dim_is_runner_dim(self):
        p = _build_policy()
        assert p.direction_prob_head[0].weight.shape[1] == RUNNER_DIM, (
            f"direction_prob_head[0] input dim must be RUNNER_DIM "
            f"({RUNNER_DIM}), got {p.direction_prob_head[0].weight.shape[1]}"
        )

    # obs_dim layout: production-sized obs picks the natural
    # offset; test-sized obs falls back to offset 0 with the
    # forward zero-padding the slice up to the head's full input.
    def test_obs_dim_layout_natural_vs_fallback(self):
        space = DiscreteActionSpace(max_runners=_MAX_RUNNERS)

        # Production-sized: natural offset.
        p_big = DiscreteLSTMPolicy(
            obs_dim=_OBS_DIM, action_space=space, hidden_size=_HIDDEN,
        )
        assert p_big._runner_block_offset == MARKET_DIM + VELOCITY_DIM
        assert p_big._runner_block_size == _MAX_RUNNERS * RUNNER_DIM
        assert p_big._runner_block_full_size == _MAX_RUNNERS * RUNNER_DIM

        # Test-sized: offset clamps to 0; size clamps to obs_dim.
        small = 64
        p_small = DiscreteLSTMPolicy(
            obs_dim=small, action_space=space, hidden_size=_HIDDEN,
        )
        assert p_small._runner_block_offset == 0
        assert p_small._runner_block_size == min(
            small, _MAX_RUNNERS * RUNNER_DIM,
        )
        assert p_small._runner_block_full_size == _MAX_RUNNERS * RUNNER_DIM
        # Forward still runs without crashing on the small obs.
        with torch.no_grad():
            out = p_small(torch.randn(1, small))
        assert out.direction_back_logits_per_runner.shape == (1, _MAX_RUNNERS)
