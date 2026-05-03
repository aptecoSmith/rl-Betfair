"""``Transition`` dataclass + ``RolloutBatch`` namedtuple for the v2
discrete-PPO rollout buffer.

Phase 2, Session 01 deliverable. The transition / batch carries
everything the (Session 02) PPO update will need to recompute log-probs,
run GAE, and compute the surrogate / value losses without touching the
env again.

Phase 3, Session 01b: the hidden state is stored as a tuple of
**device-resident torch tensors** (not CPU numpy arrays). Capturing
as numpy forced a per-tick CUDA→CPU sync inside the rollout —
~24 k sync barriers / episode, the dominant cost on the CUDA path
(see ``plans/rewrite/phase-3-cohort/findings.md`` "Session 01"). The
PPO update consumes the tensors directly: no ``torch.from_numpy``
round-trip. The tensors are detached and cloned at capture time so
they don't alias the LSTM's rolling hidden state.

Phase 4 Session 06 (2026-05-02): ``RolloutBatch`` namedtuple replaces
the per-tick ``list[Transition]`` round-trip on the sequential rollout
hot path. Sessions 02 / 04's pre-allocated obs / mask / hidden-state
buffers slot directly into ``RolloutBatch`` fields without the end-of-
episode ``Transition(...)`` list comprehension + ``float()``
conversions. The PPO update consumer reads the batch's pre-stacked
arrays directly — no ``np.stack([tr.field ...])`` pass.

``Transition`` is preserved for tests that explicitly construct
synthetic per-tick state and for the ``BatchedRolloutCollector``
(which still returns ``list[list[Transition]]`` per Session 06 hard
constraint #5). ``transitions_to_rollout_batch`` adapts the legacy
list form to the new batch form for the trainer's consumer side, and
``rollout_batch_to_transitions`` reverses the conversion for tests /
callers that still want per-tick views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, TYPE_CHECKING

import numpy as np
import torch

from agents_v2.action_space import ActionType, DiscreteActionSpace

if TYPE_CHECKING:  # pragma: no cover - import-cycle guard
    from agents_v2.discrete_policy import BaseDiscretePolicy


__all__ = [
    "RolloutBatch",
    "Transition",
    "action_uses_stake",
    "rollout_batch_to_transitions",
    "transitions_to_rollout_batch",
]


class RolloutBatch(NamedTuple):
    """Aligned per-tick rollout outputs, ready for PPO update.

    All numpy arrays have a leading time axis of length ``n_steps``.
    The fields' shapes / dtypes match what the pre-Session-06
    ``np.stack([tr.field for tr in transitions])`` pass produced in
    ``DiscretePPOTrainer._ppo_update`` — bit-identical inputs feed
    the surrogate loss.

    Attributes
    ----------
    obs:
        ``(n_steps, obs_dim)`` float32. View into the rollout's per-
        episode obs buffer (Session 02). Safe to alias because every
        downstream consumer either copies (``torch.from_numpy``) or
        treats the array as read-only.
    hidden_state_in:
        Tuple of pre-stacked torch tensors, one per element of the
        policy's hidden-state tuple. For
        :class:`agents_v2.discrete_policy.DiscreteLSTMPolicy` the
        2-tuple is ``(H, C)`` each ``(n_steps, num_layers, 1,
        hidden_size)``. The leading axis is the per-tick time axis;
        the remaining axes match the policy's per-tick element shape
        (i.e. the shape of one entry of ``init_hidden(batch=1)``).
        Slicing ``H[t]`` yields the per-tick state that was passed
        INTO ``policy.forward`` at tick ``t`` — same semantics as the
        pre-Session-06 ``Transition.hidden_state_in``.
    mask:
        ``(n_steps, action_space.n)`` bool.
    action_idx:
        ``(n_steps,)`` int64.
    stake_unit:
        ``(n_steps,)`` float32. Beta sample on ``(0, 1)`` per tick.
    log_prob_action:
        ``(n_steps,)`` float32.
    log_prob_stake:
        ``(n_steps,)`` float32. Zero for NOOP / CLOSE ticks; the
        surrogate loss masks those out via
        :func:`build_uses_stake_mask`.
    value_per_runner:
        ``(n_steps, max_runners)`` float32.
    per_runner_reward:
        ``(n_steps, max_runners)`` float32. Per-tick reward
        attributed across runner slots; sum over runners equals the
        env's scalar reward (within the attribution tolerance).
    done:
        ``(n_steps,)`` bool. Last entry is ``True`` for naturally
        terminating episodes.
    n_steps:
        Convenience int — same as ``obs.shape[0]``.
    """

    obs: np.ndarray
    hidden_state_in: tuple[torch.Tensor, ...]
    mask: np.ndarray
    action_idx: np.ndarray
    stake_unit: np.ndarray
    log_prob_action: np.ndarray
    log_prob_stake: np.ndarray
    value_per_runner: np.ndarray
    per_runner_reward: np.ndarray
    done: np.ndarray
    n_steps: int


@dataclass(frozen=True)
class Transition:
    """One env step's worth of rollout data.

    .. deprecated:: phase-4 Session 06
        The sequential rollout hot path no longer constructs
        ``Transition`` instances — see :class:`RolloutBatch`. The
        dataclass is kept for tests that build synthetic per-tick
        state and for the ``BatchedRolloutCollector`` which still
        returns ``list[list[Transition]]`` (Session 06 hard
        constraint #5). Removal is a separate plan.

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


def transitions_to_rollout_batch(
    transitions: list[Transition],
) -> RolloutBatch:
    """Adapt a ``list[Transition]`` (legacy form) to a :class:`RolloutBatch`.

    Used by :meth:`DiscretePPOTrainer.update_from_rollout` so the
    batched collector path
    (``training_v2.discrete_ppo.batched_rollout.BatchedRolloutCollector``,
    untouched per Session 06 hard constraint #5) and any test that
    constructs a synthetic transition list can still feed the trainer.
    The sequential rollout collector returns a :class:`RolloutBatch`
    directly — no adapter on the hot path.

    Parameters
    ----------
    transitions:
        Non-empty list of :class:`Transition`.

    Returns
    -------
    RolloutBatch
        With every field stacked along axis 0. ``hidden_state_in``
        is built by concatenating each per-tick tuple element along
        a new leading time axis (i.e. ``torch.stack`` per slot).
    """
    if not transitions:
        raise ValueError(
            "transitions_to_rollout_batch: list must be non-empty",
        )
    n = len(transitions)
    obs = np.stack([tr.obs for tr in transitions], axis=0).astype(
        np.float32, copy=False,
    )
    mask = np.stack([tr.mask for tr in transitions], axis=0).astype(
        bool, copy=False,
    )
    action_idx = np.array(
        [int(tr.action_idx) for tr in transitions], dtype=np.int64,
    )
    stake_unit = np.array(
        [float(tr.stake_unit) for tr in transitions], dtype=np.float32,
    )
    log_prob_action = np.array(
        [float(tr.log_prob_action) for tr in transitions],
        dtype=np.float32,
    )
    log_prob_stake = np.array(
        [float(tr.log_prob_stake) for tr in transitions],
        dtype=np.float32,
    )
    value_per_runner = np.stack(
        [tr.value_per_runner for tr in transitions], axis=0,
    ).astype(np.float32, copy=False)
    per_runner_reward = np.stack(
        [tr.per_runner_reward for tr in transitions], axis=0,
    ).astype(np.float32, copy=False)
    done = np.array([bool(tr.done) for tr in transitions], dtype=bool)

    n_slots = len(transitions[0].hidden_state_in)
    hidden_state_in: tuple[torch.Tensor, ...] = tuple(
        torch.stack(
            [tr.hidden_state_in[k] for tr in transitions], dim=0,
        )
        for k in range(n_slots)
    )

    return RolloutBatch(
        obs=obs,
        hidden_state_in=hidden_state_in,
        mask=mask,
        action_idx=action_idx,
        stake_unit=stake_unit,
        log_prob_action=log_prob_action,
        log_prob_stake=log_prob_stake,
        value_per_runner=value_per_runner,
        per_runner_reward=per_runner_reward,
        done=done,
        n_steps=n,
    )


def rollout_batch_to_transitions(
    batch: RolloutBatch,
) -> list[Transition]:
    """Reverse :func:`transitions_to_rollout_batch` for tests / legacy callers.

    Splits each batched field along the leading time axis and emits
    one :class:`Transition` per tick. ``hidden_state_in`` per tick is
    a tuple of slice views into the batch's pre-stacked hidden buffers
    (no copy) — same view semantics as the rollout collector's per-
    tick capture. Tests that mutate the underlying buffer would see
    the change reflected on every transition's ``hidden_state_in``,
    same as before Session 06.

    Used by tests that need to iterate transitions and by the eval
    helper in ``training_v2/cohort/worker.py``. NOT called on any
    hot path.
    """
    n = int(batch.n_steps)
    transitions: list[Transition] = []
    for i in range(n):
        h_tuple = tuple(buf[i] for buf in batch.hidden_state_in)
        transitions.append(Transition(
            obs=batch.obs[i],
            hidden_state_in=h_tuple,
            mask=batch.mask[i],
            action_idx=int(batch.action_idx[i]),
            stake_unit=float(batch.stake_unit[i]),
            log_prob_action=float(batch.log_prob_action[i]),
            log_prob_stake=float(batch.log_prob_stake[i]),
            value_per_runner=batch.value_per_runner[i],
            per_runner_reward=batch.per_runner_reward[i],
            done=bool(batch.done[i]),
        ))
    return transitions
