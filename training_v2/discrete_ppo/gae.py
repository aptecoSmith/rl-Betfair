"""Per-runner Generalised Advantage Estimation (pure NumPy).

Phase 2, Session 01 deliverable. Standard GAE applied per-runner —
with a per-runner value head and per-runner reward attribution
(:class:`Transition.per_runner_reward`), the GAE math is the textbook
recurrence applied in parallel across the runner axis.

Math (per runner ``i``)::

    δ_t^{(i)}  = r_t^{(i)} + γ · (1 - done_t) · V_{t+1}^{(i)} - V_t^{(i)}
    A_t^{(i)}  = δ_t^{(i)} + γλ · (1 - done_t) · A_{t+1}^{(i)}
    return_t^{(i)} = A_t^{(i)} + V_t^{(i)}

``done_t`` zeroes the bootstrap when the episode ends. For the final
step ``V_{t+1}`` is ``bootstrap_value`` — a separate forward pass on
the terminal observation, or zero when the episode terminated
naturally (``done_T = True``).
"""

from __future__ import annotations

import numpy as np


__all__ = ["compute_per_runner_gae"]


def compute_per_runner_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    bootstrap_value: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-runner GAE advantages + returns.

    Parameters
    ----------
    rewards:
        ``(n_steps, max_runners)`` per-runner reward at each step.
        Comes from :attr:`Transition.per_runner_reward`.
    values:
        ``(n_steps, max_runners)`` per-runner value-head output at
        each step. Comes from :attr:`Transition.value_per_runner`.
    bootstrap_value:
        ``(max_runners,)`` value of the post-episode state. Zero when
        the episode terminated naturally (``dones[-1] == True``);
        otherwise the result of a separate forward pass on the
        terminal observation.
    dones:
        ``(n_steps,)`` bool. ``True`` at the step where the episode
        ended. Note that done is shared across runners — a runner
        doesn't end mid-episode.
    gamma, gae_lambda:
        Standard PPO hyperparameters.

    Returns
    -------
    (advantages, returns):
        Both ``(n_steps, max_runners)`` float32. ``returns`` is
        defined as ``advantages + values`` exactly (no separate
        recurrence), which both keeps the implementation simple and
        guarantees the value-loss target satisfies the PPO contract.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    bootstrap_value = np.asarray(bootstrap_value, dtype=np.float32)
    dones = np.asarray(dones, dtype=bool)

    if rewards.ndim != 2:
        raise ValueError(
            f"rewards must be (n_steps, max_runners), got shape "
            f"{rewards.shape}",
        )
    if values.shape != rewards.shape:
        raise ValueError(
            f"values shape {values.shape} != rewards shape "
            f"{rewards.shape}",
        )
    n_steps, max_runners = rewards.shape
    if bootstrap_value.shape != (max_runners,):
        raise ValueError(
            f"bootstrap_value shape {bootstrap_value.shape} != "
            f"(max_runners={max_runners},)",
        )
    if dones.shape != (n_steps,):
        raise ValueError(
            f"dones shape {dones.shape} != (n_steps={n_steps},)",
        )

    advantages = np.zeros_like(rewards)
    next_advantage = np.zeros(max_runners, dtype=np.float32)
    next_value = bootstrap_value.astype(np.float32, copy=False)

    for t in reversed(range(n_steps)):
        not_done = 0.0 if dones[t] else 1.0
        delta = (
            rewards[t]
            + gamma * not_done * next_value
            - values[t]
        )
        next_advantage = delta + gamma * gae_lambda * not_done * next_advantage
        advantages[t] = next_advantage
        next_value = values[t]

    returns = advantages + values
    return advantages, returns
