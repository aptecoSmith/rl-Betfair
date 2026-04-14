"""Session 2 (arb-improvements) — entropy floor + per-head entropy logging.

All tests are CPU-only and fast. Cross-reference:
``plans/arb-improvements/session_2_entropy_floor.md``.

The controller scales the *entropy coefficient* (PPO loss term), never
the action distribution itself (see hard_constraints.md §Stabilisation).
Every test therefore pokes state through the public ``self.entropy_coeff``
attribute and the ``_update_entropy_controller`` helper, rather than
second-guessing the policy's sampled actions.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from agents import ppo_trainer as ppo_trainer_module
from agents.policy_network import PolicyOutput
from agents.ppo_trainer import (
    EpisodeStats,
    PPOTrainer,
    Rollout,
    Transition,
    _HEAD_NAMES,
)


# ── Helpers ────────────────────────────────────────────────────────────────


class _StubPolicy(nn.Module):
    """Minimal multi-head policy whose flat action space maps onto the
    ``[signal × N | stake × N | aggression × N | cancel × N | arb_spread × N]``
    layout used by the trainer's per-head slicer.
    """

    def __init__(
        self,
        obs_dim: int = 4,
        max_runners: int = 2,
        per_runner_action_dim: int = 5,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.max_runners = max_runners
        self._per_runner_action_dim = per_runner_action_dim
        self.action_dim = max_runners * per_runner_action_dim
        # One trainable parameter so Adam has something to optimise.
        self.linear = nn.Linear(obs_dim, self.action_dim * 2 + 1)

    def init_hidden(self, batch_size: int = 1):
        return (
            torch.zeros(1, batch_size, 1),
            torch.zeros(1, batch_size, 1),
        )

    def forward(self, obs: torch.Tensor, hidden_state=None) -> PolicyOutput:
        batch = obs.shape[0]
        out = self.linear(obs)
        action_mean = out[:, : self.action_dim]
        action_log_std = out[:, self.action_dim : self.action_dim * 2]
        value = out[:, -1:]
        if hidden_state is None:
            hidden_state = (
                torch.zeros(1, batch, 1),
                torch.zeros(1, batch, 1),
            )
        return PolicyOutput(
            action_mean=action_mean,
            action_log_std=action_log_std,
            value=value,
            hidden_state=hidden_state,
        )


class _ScriptedEnv:
    """Scripted-reward env replacement. Re-used from test_reward_clipping
    rather than imported to keep each test file self-contained."""

    def __init__(self, rewards, obs_dim: int = 4) -> None:
        self._rewards = list(rewards)
        self._i = 0
        self._obs_dim = obs_dim

    def __call__(self, *args, **kwargs):  # pragma: no cover - patch hook
        return self

    def reset(self):
        self._i = 0
        return np.zeros(self._obs_dim, dtype=np.float32), {}

    def step(self, action):
        r = self._rewards[self._i]
        self._i += 1
        done = self._i >= len(self._rewards)
        info = {
            "day_pnl": float(sum(self._rewards[: self._i])),
            "raw_pnl_reward": float(sum(self._rewards[: self._i])),
            "shaped_bonus": 0.0,
            "bet_count": 0,
            "winning_bets": 0,
            "races_completed": 1 if done else 0,
            "budget": 100.0,
        }
        return (
            np.zeros(self._obs_dim, dtype=np.float32),
            float(r),
            done,
            False,
            info,
        )


def _minimal_config(tmp_path) -> dict:
    return {
        "paths": {"logs": str(tmp_path / "logs")},
        "reward": {},
        "training": {},
    }


def _make_trainer(
    tmp_path,
    *,
    entropy_coefficient: float = 0.01,
    entropy_floor: float = 0.0,
    entropy_floor_window: int = 10,
    entropy_boost_max: float = 10.0,
    entropy_collapse_patience: int = 5,
    obs_dim: int = 4,
    max_runners: int = 2,
    per_runner_action_dim: int = 5,
) -> PPOTrainer:
    hp = {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ppo_clip_epsilon": 0.2,
        "entropy_coefficient": entropy_coefficient,
        "value_loss_coeff": 0.5,
        "max_grad_norm": 0.5,
        "ppo_epochs": 1,
        "mini_batch_size": 8,
        "entropy_floor": entropy_floor,
        "entropy_floor_window": entropy_floor_window,
        "entropy_boost_max": entropy_boost_max,
        "entropy_collapse_patience": entropy_collapse_patience,
    }
    policy = _StubPolicy(
        obs_dim=obs_dim,
        max_runners=max_runners,
        per_runner_action_dim=per_runner_action_dim,
    )
    return PPOTrainer(
        policy=policy,
        config=_minimal_config(tmp_path),
        hyperparams=hp,
        device="cpu",
    )


def _feed_batches(trainer: PPOTrainer, batches):
    """Feed a sequence of per-head entropy dicts (one per PPO update)
    through the controller. Returns the last ``action_stats`` dict."""
    stats = None
    for per_head in batches:
        stats = trainer._update_entropy_controller(dict(per_head))
    return stats


def _uniform(value: float) -> dict[str, float]:
    """All five heads at the same entropy value."""
    return {h: value for h in _HEAD_NAMES}


# ── 1. Floor triggers scaling ──────────────────────────────────────────────


def test_floor_triggers_coefficient_scaling(tmp_path):
    """Rolling mean below the floor → coefficient scales by floor/mean,
    bounded by entropy_boost_max. With base=0.01, floor=0.5, mean=0.25,
    the active coefficient must be 2× base = 0.02."""
    trainer = _make_trainer(
        tmp_path,
        entropy_coefficient=0.01,
        entropy_floor=0.5,
        entropy_boost_max=10.0,
        entropy_floor_window=3,
    )
    # Feed three batches with mean entropy 0.25 < floor 0.5.
    stats = _feed_batches(
        trainer,
        [_uniform(0.25), _uniform(0.25), _uniform(0.25)],
    )
    # ratio = 0.5 / 0.25 = 2.0; multiplier capped at 10 → 2.0.
    assert trainer._entropy_coeff_active == pytest.approx(0.02, rel=1e-6)
    assert trainer.entropy_coeff == pytest.approx(0.02, rel=1e-6)
    assert stats["entropy_coeff_active"] == pytest.approx(0.02, rel=1e-6)


# ── 2. Recovery restores baseline ──────────────────────────────────────────


def test_recovery_restores_baseline(tmp_path):
    """Sustained-high entropy after a collapse must snap the coefficient
    back to the baseline once the rolling mean clears the floor."""
    trainer = _make_trainer(
        tmp_path,
        entropy_coefficient=0.01,
        entropy_floor=0.5,
        entropy_boost_max=10.0,
        entropy_floor_window=3,
    )
    # Collapse.
    _feed_batches(trainer, [_uniform(0.1)] * 3)
    assert trainer._entropy_coeff_active > 0.01 + 1e-9
    # Recover — three high-entropy batches push the rolling mean above
    # the floor (window size 3, so the old values are fully evicted).
    stats = _feed_batches(trainer, [_uniform(1.0)] * 3)
    assert trainer._entropy_coeff_active == pytest.approx(0.01, rel=1e-9)
    assert stats["entropy_coeff_active"] == pytest.approx(0.01, rel=1e-9)


# ── 3. Floor off by default ────────────────────────────────────────────────


def test_floor_off_by_default_coefficient_never_changes(tmp_path):
    """With entropy_floor=0 (default), the coefficient stays at the
    baseline regardless of what entropies the controller sees. This is
    the 'byte-identical training' guarantee for pre-session configs."""
    trainer = _make_trainer(
        tmp_path,
        entropy_coefficient=0.01,
        entropy_floor=0.0,
    )
    assert trainer.entropy_floor == 0.0
    # Feed a bunch of adversarial values — far above, far below, zero.
    _feed_batches(
        trainer,
        [
            _uniform(0.0),
            _uniform(0.001),
            _uniform(10.0),
            _uniform(1e-6),
        ],
    )
    assert trainer._entropy_coeff_active == pytest.approx(0.01, rel=1e-9)
    assert trainer.entropy_coeff == pytest.approx(0.01, rel=1e-9)


# ── 4. Per-head entropy in progress event ──────────────────────────────────


def test_per_head_entropy_in_progress_event(tmp_path):
    """The progress event construction carries an ``action_stats`` dict
    containing all five head entropies as floats plus the collapse flag
    and active coefficient."""
    trainer = _make_trainer(tmp_path)
    # Seed the controller with one batch so the per-head windows are
    # non-empty and a known-different value lets us verify the plumbing.
    per_head = {
        "signal": 0.9,
        "stake": 0.8,
        "aggression": 0.7,
        "cancel": 0.6,
        "arb_spread": 0.5,
    }
    action_stats = trainer._update_entropy_controller(per_head)
    loss_info = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "action_stats": action_stats,
    }

    ep = EpisodeStats(
        day_date="2026-04-14",
        total_reward=0.0,
        total_pnl=0.0,
        bet_count=0,
        winning_bets=0,
        races_completed=0,
        final_budget=100.0,
        n_steps=0,
    )

    captured: list[dict] = []

    class _FakeQueue:
        def put_nowait(self, item):
            captured.append(item)

    class _Tracker:
        completed = 1
        total = 1

        def to_dict(self):
            return {"completed": 1, "total": 1}

    trainer.progress_queue = _FakeQueue()
    trainer._publish_progress(ep, loss_info, _Tracker())

    assert captured, "progress event was not emitted"
    event = captured[0]
    assert "action_stats" in event
    stats = event["action_stats"]
    for head, expected in per_head.items():
        key = f"mean_entropy_{head}"
        assert key in stats
        assert isinstance(stats[key], float)
        assert stats[key] == pytest.approx(expected, rel=1e-6)
    assert isinstance(stats["entropy_collapse"], bool)
    assert isinstance(stats["entropy_coeff_active"], float)


# ── 5. entropy_collapse flag ───────────────────────────────────────────────


def test_entropy_collapse_flag_sets_and_clears(tmp_path):
    """A head's entropy must stay below the floor for >
    entropy_collapse_patience consecutive batches for the flag to fire.
    When the head recovers, the flag flips back to False.
    """
    trainer = _make_trainer(
        tmp_path,
        entropy_floor=0.5,
        entropy_collapse_patience=5,
        entropy_floor_window=10,
    )

    # All other heads healthy, signal is collapsing.
    def _signal_at(v: float) -> dict[str, float]:
        return {"signal": v, "stake": 1.0, "aggression": 1.0,
                "cancel": 1.0, "arb_spread": 1.0}

    # First 5 collapsed batches — streak == 5, NOT strictly > 5 → flag still off.
    for _ in range(5):
        stats = trainer._update_entropy_controller(_signal_at(0.1))
        assert stats["entropy_collapse"] is False
    # 6th consecutive collapse — streak > 5 → flag fires.
    stats = trainer._update_entropy_controller(_signal_at(0.1))
    assert stats["entropy_collapse"] is True

    # Recovery in one batch clears the streak and the flag.
    stats = trainer._update_entropy_controller(_signal_at(1.0))
    assert stats["entropy_collapse"] is False


# ── 6. entropy_boost_max caps the multiplier ───────────────────────────────


def test_entropy_boost_max_caps_multiplier(tmp_path):
    """When the rolling mean is tiny, floor/mean would go to infinity.
    The cap prevents the coefficient from blowing up."""
    trainer = _make_trainer(
        tmp_path,
        entropy_coefficient=0.01,
        entropy_floor=1.0,
        entropy_boost_max=3.0,
        entropy_floor_window=2,
    )
    # Mean entropy 0.01 → raw ratio = 100, should cap at 3.
    _feed_batches(trainer, [_uniform(0.01)] * 2)
    assert trainer._entropy_coeff_active == pytest.approx(0.03, rel=1e-9)

    # Even tinier entropy — cap still holds, no NaN, no overflow.
    _feed_batches(trainer, [_uniform(1e-9)] * 2)
    assert trainer._entropy_coeff_active == pytest.approx(0.03, rel=1e-9)


# ── 7. raw + shaped ≈ total_reward invariant still holds with floor on ─────


def test_raw_plus_shaped_invariant_holds_with_entropy_floor(tmp_path):
    """Entropy floor touches ``self.entropy_coeff`` (the PPO *loss* term),
    never the environment reward or its accumulators. Running a full
    rollout with the floor armed must leave the telemetry raw/shaped
    invariant untouched.
    """
    trainer = _make_trainer(
        tmp_path,
        entropy_coefficient=0.01,
        entropy_floor=0.5,      # armed
        entropy_boost_max=10.0,
    )

    rewards = [0.5, 1.5, -0.25]

    class _SplitEnv(_ScriptedEnv):
        def step(self, action):
            obs, r, done, trunc, info = super().step(action)
            info["raw_pnl_reward"] = 0.4 * sum(self._rewards[: self._i])
            info["shaped_bonus"] = 0.6 * sum(self._rewards[: self._i])
            return obs, r, done, trunc, info

    with patch.object(
        ppo_trainer_module, "BetfairEnv",
        lambda *a, **kw: _SplitEnv(rewards),
    ):
        day = MagicMock(date="2026-04-14")
        _rollout, ep_stats = trainer._collect_rollout(day)

    assert ep_stats.raw_pnl_reward + ep_stats.shaped_bonus == pytest.approx(
        ep_stats.total_reward, abs=1e-6,
    )
    # Floor active and base captured — these are the hooks the controller
    # operates on; confirm they exist and the accumulators weren't
    # monkey-patched by the floor code path.
    assert trainer._entropy_coeff_base == pytest.approx(0.01)
    assert ep_stats.total_reward == pytest.approx(sum(rewards), abs=1e-6)
