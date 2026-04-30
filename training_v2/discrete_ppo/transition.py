"""``Transition`` dataclass for the v2 discrete-PPO rollout buffer.

Phase 2, Session 01 deliverable. The transition carries everything
the (Session 02) PPO update will need to recompute log-probs, run
GAE, and compute the surrogate / value losses without touching the
env again.

Phase 3, Session 01b: the hidden state is stored as a tuple of
**device-resident torch tensors** (not CPU numpy arrays). Capturing
as numpy forced a per-tick CUDA→CPU sync inside the rollout —
~24 k sync barriers / episode, the dominant cost on the CUDA path
(see ``plans/rewrite/phase-3-cohort/findings.md`` "Session 01"). The
PPO update consumes the tensors directly: no ``torch.from_numpy``
round-trip. The tensors are detached and cloned at capture time so
they don't alias the LSTM's rolling hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from agents_v2.action_space import ActionType, DiscreteActionSpace


__all__ = ["Transition", "action_uses_stake"]


@dataclass(frozen=True)
class Transition:
    """One env step's worth of rollout data.

    Attributes
    ----------
    obs:
        ``(obs_dim,)`` float32. The observation the policy saw at
        this step. Stored CPU-side so the rollout doesn't pin GPU.
    hidden_state_in:
        Tuple of **device-resident torch tensors** representing the
        hidden state that was passed INTO ``policy.forward`` at this
        step (NOT the state the forward returned). ppo-kl-fix gotcha:
        capturing the post-forward state silently corrupts the PPO
        update. For :class:`agents_v2.discrete_policy.DiscreteLSTMPolicy`
        the 2-tuple is ``(h, c)`` each ``(num_layers, 1, hidden_size)``.
        Tensors are ``.detach().clone()`` of the rollout's running
        hidden state so subsequent LSTM forwards don't mutate them.
    mask:
        ``(action_space.n,)`` bool. The legality mask the rollout-
        time categorical was conditioned on. Carried so the PPO
        update applies the SAME mask when re-evaluating log-probs —
        otherwise an env-state shift between rollout and update could
        flip a sampled action's mask bit and produce ``-inf``
        new-log-probs (Phase 1 findings.md §2).
    action_idx:
        Sampled discrete action ``∈ [0, action_space.n)``. ``int``,
        not numpy scalar — keeps mini-batch indexing simple.
    stake_unit:
        Sampled stake from the policy's Beta head, ``∈ (0, 1)``.
        Re-scaled to £ outside the policy by the shim. Stored as the
        un-rescaled unit-interval value because that's the support
        of the Beta whose log-prob this transition carries.
    log_prob_action:
        ``Categorical(logits=masked_logits).log_prob(action_idx)``
        captured at rollout time. Float scalar.
    log_prob_stake:
        ``Beta(alpha, beta).log_prob(stake_unit)`` captured at
        rollout time. **Zero** when the chosen action does not use
        stake (NOOP, CLOSE_*) — the surrogate loss masks this slot
        out via :func:`action_uses_stake` so the placeholder zero
        never contributes to the gradient.
    value_per_runner:
        ``(max_runners,)`` float32. The per-runner critic head's
        output at this step — Phase 2's per-runner GAE reads this.
    per_runner_reward:
        ``(max_runners,)`` float32. The reward at this step,
        attributed across runners. ``per_runner_reward.sum()`` must
        equal the env's scalar ``reward`` to floating-point tolerance
        (the collector enforces this at runtime).
    done:
        ``True`` on the final transition of an episode. Intermediate
        transitions' ``done`` is ``False``.
    """

    obs: np.ndarray
    hidden_state_in: tuple[torch.Tensor, ...]
    mask: np.ndarray
    action_idx: int
    stake_unit: float
    log_prob_action: float
    log_prob_stake: float
    value_per_runner: np.ndarray
    per_runner_reward: np.ndarray
    done: bool


def action_uses_stake(
    action_space: DiscreteActionSpace,
    action_idx: int,
) -> bool:
    """Return ``True`` iff ``action_idx`` carries a meaningful stake.

    Per the locked action layout (``agents_v2/action_space.py``),
    only ``OPEN_BACK_*`` and ``OPEN_LAY_*`` actions use the Beta
    stake sample. ``NOOP`` doesn't place a bet; ``CLOSE_*`` sizes
    its close leg via the env's equal-profit helper, not the policy
    stake. The PPO update masks out ``log_prob_stake`` for the
    actions where this returns ``False``.
    """
    kind, _runner = action_space.decode(int(action_idx))
    return kind in (ActionType.OPEN_BACK, ActionType.OPEN_LAY)
