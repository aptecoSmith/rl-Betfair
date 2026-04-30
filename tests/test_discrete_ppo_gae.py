"""Tests for ``training_v2.discrete_ppo.gae.compute_per_runner_gae``.

Phase 2, Session 01 — pure NumPy, no torch, no env.
"""

from __future__ import annotations

import numpy as np
import pytest

from training_v2.discrete_ppo.gae import compute_per_runner_gae


def test_gae_per_runner_shape():
    rewards = np.zeros((5, 3), dtype=np.float32)
    values = np.zeros((5, 3), dtype=np.float32)
    bootstrap = np.zeros(3, dtype=np.float32)
    dones = np.zeros(5, dtype=bool)
    dones[-1] = True

    advantages, returns = compute_per_runner_gae(
        rewards, values, bootstrap, dones, gamma=0.99, gae_lambda=0.95,
    )

    assert advantages.shape == (5, 3)
    assert returns.shape == (5, 3)
    assert advantages.dtype == np.float32
    assert returns.dtype == np.float32


def test_gae_matches_hand_reference():
    """Hand-computed GAE on a 3-step / 2-runner toy.

    Setup
    -----
    γ = 0.9, λ = 0.5, n_steps = 3, max_runners = 2
    rewards = [[1.0, 2.0], [0.0, 1.0], [3.0, 0.0]]
    values  = [[0.5, 1.0], [0.0, 0.5], [1.0, 0.0]]
    bootstrap = [0.0, 0.0], dones = [F, F, T]

    Runner 0 — backward recurrence:
      t=2: δ = 3.0 - 1.0          = 2.0,    A_2 = 2.0
      t=1: δ = 0.0 + 0.9·1 - 0.0  = 0.9,    A_1 = 0.9 + 0.45·2.0 = 1.8
      t=0: δ = 1.0 + 0.9·0 - 0.5  = 0.5,    A_0 = 0.5 + 0.45·1.8 = 1.31

    Runner 1:
      t=2: δ = 0.0 - 0.0          = 0.0,    A_2 = 0.0
      t=1: δ = 1.0 + 0.9·0 - 0.5  = 0.5,    A_1 = 0.5 + 0.45·0.0 = 0.5
      t=0: δ = 2.0 + 0.9·0.5 - 1.0 = 1.45,  A_0 = 1.45 + 0.45·0.5 = 1.675
    """
    rewards = np.array([
        [1.0, 2.0],
        [0.0, 1.0],
        [3.0, 0.0],
    ], dtype=np.float32)
    values = np.array([
        [0.5, 1.0],
        [0.0, 0.5],
        [1.0, 0.0],
    ], dtype=np.float32)
    bootstrap = np.zeros(2, dtype=np.float32)
    dones = np.array([False, False, True])

    advantages, returns = compute_per_runner_gae(
        rewards, values, bootstrap, dones, gamma=0.9, gae_lambda=0.5,
    )

    expected_adv = np.array([
        [1.31,  1.675],
        [1.80,  0.500],
        [2.00,  0.000],
    ], dtype=np.float32)
    expected_returns = expected_adv + values

    np.testing.assert_allclose(advantages, expected_adv, atol=1e-6)
    np.testing.assert_allclose(returns, expected_returns, atol=1e-6)


def test_gae_returns_equal_advantages_plus_values():
    """``returns == advantages + values`` is exact, not approximate."""
    rng = np.random.default_rng(42)
    rewards = rng.normal(size=(8, 4)).astype(np.float32)
    values = rng.normal(size=(8, 4)).astype(np.float32)
    bootstrap = rng.normal(size=(4,)).astype(np.float32)
    dones = np.zeros(8, dtype=bool)
    dones[-1] = True

    advantages, returns = compute_per_runner_gae(
        rewards, values, bootstrap, dones,
    )
    np.testing.assert_array_equal(returns, advantages + values)


def test_gae_zero_rewards_zero_advantages_iff_values_constant():
    """Rewards = 0 + values constant across time → advantages = 0.

    With ``r = 0`` and ``V_t = V`` for all t, the bootstrap also at
    ``V`` (so the implicit terminal "next value" matches), the
    Bellman residual is::

        δ_t = 0 + γ·V - V = (γ - 1)·V

    which is non-zero unless γ=1. So we set the bootstrap such that
    the recurrence collapses cleanly: with values constant at 0 and
    bootstrap 0, every δ is 0 and every advantage is 0. This is the
    sanity guard the session prompt asks for.
    """
    n_steps, n_runners = 6, 3
    rewards = np.zeros((n_steps, n_runners), dtype=np.float32)
    values = np.zeros((n_steps, n_runners), dtype=np.float32)
    bootstrap = np.zeros(n_runners, dtype=np.float32)
    dones = np.zeros(n_steps, dtype=bool)

    advantages, returns = compute_per_runner_gae(
        rewards, values, bootstrap, dones,
    )
    np.testing.assert_array_equal(advantages, np.zeros_like(rewards))
    np.testing.assert_array_equal(returns, np.zeros_like(rewards))


def test_gae_done_zeroes_bootstrap_at_terminal():
    """Setting ``dones[-1] = True`` zeroes the bootstrap contribution."""
    rewards = np.array([[0.0]], dtype=np.float32)
    values = np.array([[0.0]], dtype=np.float32)
    # Bootstrap value of 100 — would dominate δ if done weren't honoured.
    bootstrap = np.array([100.0], dtype=np.float32)

    # done=True ⇒ δ = r + 0·V_next − V = 0 − 0 = 0
    adv_done, _ = compute_per_runner_gae(
        rewards, values, bootstrap, np.array([True]),
    )
    np.testing.assert_array_equal(adv_done, np.zeros((1, 1), dtype=np.float32))

    # done=False ⇒ δ = r + γ·V_next − V = γ·100 − 0 = 99 (γ=0.99)
    adv_not_done, _ = compute_per_runner_gae(
        rewards, values, bootstrap, np.array([False]),
    )
    np.testing.assert_allclose(
        adv_not_done, np.array([[99.0]], dtype=np.float32), atol=1e-5,
    )


def test_gae_shape_validation():
    rewards = np.zeros((3, 2), dtype=np.float32)
    values_wrong = np.zeros((3, 3), dtype=np.float32)
    bootstrap = np.zeros(2, dtype=np.float32)
    dones = np.zeros(3, dtype=bool)

    with pytest.raises(ValueError, match="values shape"):
        compute_per_runner_gae(rewards, values_wrong, bootstrap, dones)

    with pytest.raises(ValueError, match="bootstrap_value shape"):
        compute_per_runner_gae(
            rewards,
            np.zeros((3, 2), dtype=np.float32),
            np.zeros(3, dtype=np.float32),  # wrong size
            dones,
        )

    with pytest.raises(ValueError, match="dones shape"):
        compute_per_runner_gae(
            rewards,
            np.zeros((3, 2), dtype=np.float32),
            bootstrap,
            np.zeros(5, dtype=bool),  # wrong size
        )
