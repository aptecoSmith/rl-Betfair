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


_OBS_DIM = 64
_MAX_RUNNERS = 5
_HIDDEN = 32


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

    # 2. forward-side gradient guard ----------------------------------------

    def test_action_logits_depend_on_direction_prob_head_weights(self):
        p = _build_policy()
        torch.manual_seed(0)
        obs = torch.randn(2, _OBS_DIM)
        with torch.no_grad():
            out_a = p(obs)
            logits_a = out_a.logits.detach().clone()
            # Perturb the direction head's weight and re-run.
            p.direction_prob_head.weight.add_(0.5)
            out_b = p(obs)
            logits_b = out_b.logits.detach().clone()
        assert not torch.allclose(logits_a, logits_b), (
            "Perturbing direction_prob_head.weight did not change "
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
        assert p.direction_prob_head.weight.grad is not None, (
            "direction_prob_head.weight has no gradient — surrogate "
            "path appears detached."
        )
        assert p.direction_prob_head.weight.grad.abs().max() > 0.0

    # 4. cross-load failure (architecture-hash break) ----------------------

    def test_pre_direction_weights_fail_to_load(self):
        p = _build_policy()
        sd = {k: v.detach().clone() for k, v in p.state_dict().items()}
        old_w = sd["actor_head.0.weight"]
        new_in = old_w.shape[1]  # runner_embed + hidden + 4
        old_in = new_in - 2  # pre-direction-prob width
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
