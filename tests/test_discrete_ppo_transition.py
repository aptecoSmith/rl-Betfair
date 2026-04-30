"""Tests for ``training_v2.discrete_ppo.transition``.

Phase 2, Session 01 — pure-Python tests, no env, no torch (besides
the action-space class which is itself dependency-free).
"""

from __future__ import annotations

import numpy as np
import torch

from agents_v2.action_space import ActionType, DiscreteActionSpace
from training_v2.discrete_ppo.transition import (
    Transition,
    action_uses_stake,
)


def test_transition_round_trip():
    """Construct a :class:`Transition` and read every field back."""
    obs = np.zeros(10, dtype=np.float32)
    h = torch.zeros((1, 1, 8), dtype=torch.float32)
    c = torch.zeros((1, 1, 8), dtype=torch.float32)
    mask = np.array([True, False, True, True], dtype=bool)
    value_per_runner = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    per_runner_reward = np.array([0.0, 1.0, -0.5], dtype=np.float32)

    tr = Transition(
        obs=obs,
        hidden_state_in=(h, c),
        mask=mask,
        action_idx=2,
        stake_unit=0.42,
        log_prob_action=-0.7,
        log_prob_stake=-1.3,
        value_per_runner=value_per_runner,
        per_runner_reward=per_runner_reward,
        done=False,
    )

    assert tr.obs is obs
    assert tr.hidden_state_in[0] is h
    assert tr.hidden_state_in[1] is c
    assert (tr.mask == mask).all()
    assert tr.action_idx == 2
    assert tr.stake_unit == 0.42
    assert tr.log_prob_action == -0.7
    assert tr.log_prob_stake == -1.3
    assert (tr.value_per_runner == value_per_runner).all()
    assert (tr.per_runner_reward == per_runner_reward).all()
    assert tr.done is False


def test_uses_stake_only_for_open_actions():
    """``action_uses_stake`` is True for OPEN_*, False otherwise."""
    space = DiscreteActionSpace(max_runners=4)

    # NOOP: index 0
    assert action_uses_stake(space, 0) is False

    # All OPEN_BACK_i / OPEN_LAY_i — uses stake
    for slot in range(4):
        ob_idx = space.encode(ActionType.OPEN_BACK, slot)
        ol_idx = space.encode(ActionType.OPEN_LAY, slot)
        assert action_uses_stake(space, ob_idx) is True
        assert action_uses_stake(space, ol_idx) is True

    # All CLOSE_i — does NOT use stake
    for slot in range(4):
        cl_idx = space.encode(ActionType.CLOSE, slot)
        assert action_uses_stake(space, cl_idx) is False
