"""Session 3 (arb-improvements) — signal-bias warmup + bet-rate diagnostics.

All tests CPU-only and fast. Cross-reference:
``plans/arb-improvements/session_3_signal_bias_warmup.md``.

The signal-bias warmup is a *soft prior* on the per-runner ``signal``
head mean that decays linearly to zero by the configured warmup epoch
and never touches any other head (``hard_constraints.md §Stabilisation``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from agents.architecture_registry import create_policy
from agents.policy_network import PolicyOutput
from agents.ppo_trainer import (
    EpisodeStats,
    PPOTrainer,
    _HEAD_NAMES,
    _signal_bias_for_epoch,
)
from env.betfair_env import (
    AGENT_STATE_DIM,
    MARKET_DIM,
    POSITION_DIM,
    RUNNER_DIM,
    VELOCITY_DIM,
)


# ── Helpers ────────────────────────────────────────────────────────────────


_MAX_RUNNERS = 2
_PER_RUNNER_ACTION_DIM = 5  # signal, stake, aggression, cancel, arb_spread
_ACTION_DIM = _MAX_RUNNERS * _PER_RUNNER_ACTION_DIM

_OBS_DIM = (
    MARKET_DIM
    + VELOCITY_DIM
    + _MAX_RUNNERS * RUNNER_DIM
    + AGENT_STATE_DIM
    + _MAX_RUNNERS * POSITION_DIM
)


def _make_policy(arch: str):
    """Instantiate the given architecture on CPU with a small max_runners."""
    hp = {
        "lstm_hidden_size": 32,
        "mlp_hidden_size": 16,
        "mlp_layers": 1,
        "transformer_heads": 2,
        "transformer_depth": 1,
        "transformer_ctx_ticks": 8,
    }
    return create_policy(
        name=arch,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        max_runners=_MAX_RUNNERS,
        hyperparams=hp,
    )


def _obs(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, _OBS_DIM, generator=g)


_ALL_ARCHS = ("ppo_lstm_v1", "ppo_time_lstm_v1", "ppo_transformer_v1")


class _StubMultiHeadPolicy(nn.Module):
    """Minimal multi-head policy for the trainer-integration tests."""

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
        self.linear = nn.Linear(obs_dim, self.action_dim * 2 + 1)

    def init_hidden(self, batch_size: int = 1):
        return (
            torch.zeros(1, batch_size, 1),
            torch.zeros(1, batch_size, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state=None,
        signal_bias: float = 0.0,
    ) -> PolicyOutput:
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
        # Mirror the real policy's layout: signal is the first
        # ``max_runners`` entries of action_mean.
        if signal_bias != 0.0:
            bias = torch.zeros_like(action_mean)
            bias[:, : self.max_runners] = signal_bias
            action_mean = action_mean + bias
        return PolicyOutput(
            action_mean=action_mean,
            action_log_std=action_log_std,
            value=value,
            hidden_state=hidden_state,
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
    signal_bias_warmup: int = 0,
    signal_bias_magnitude: float = 0.0,
) -> PPOTrainer:
    hp = {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ppo_clip_epsilon": 0.2,
        "entropy_coefficient": 0.01,
        "value_loss_coeff": 0.5,
        "max_grad_norm": 0.5,
        "ppo_epochs": 1,
        "mini_batch_size": 8,
        "signal_bias_warmup": signal_bias_warmup,
        "signal_bias_magnitude": signal_bias_magnitude,
    }
    policy = _StubMultiHeadPolicy()
    return PPOTrainer(
        policy=policy,
        config=_minimal_config(tmp_path),
        hyperparams=hp,
        device="cpu",
    )


# ── 1. Bias applied at epoch 0 ─────────────────────────────────────────────


@pytest.mark.parametrize("arch", _ALL_ARCHS)
def test_bias_shifts_signal_mean_at_epoch_zero(arch):
    """Calling forward with signal_bias=0.5 must shift the signal-head
    means by exactly 0.5 relative to signal_bias=0.0, and leave every
    other head's means untouched."""
    policy = _make_policy(arch)
    policy.eval()
    obs = _obs(seed=42)

    with torch.no_grad():
        out_zero = policy(obs, None, signal_bias=0.0)
        out_bias = policy(obs, None, signal_bias=0.5)

    # Signal head occupies the first max_runners dims of action_mean.
    signal_zero = out_zero.action_mean[:, :_MAX_RUNNERS]
    signal_bias = out_bias.action_mean[:, :_MAX_RUNNERS]
    diff = (signal_bias - signal_zero).cpu().numpy()
    assert np.allclose(diff, 0.5, atol=1e-5), (
        f"{arch}: signal mean shift was {diff}, expected +0.5"
    )


# ── 2. Bias linearly decays ────────────────────────────────────────────────


def test_bias_linearly_decays():
    """With magnitude=1.0 and warmup=10, the per-epoch bias follows a
    linear decay from 1.0 at epoch 0 to 0.0 at epoch 10."""
    assert _signal_bias_for_epoch(0, 10, 1.0) == pytest.approx(1.0)
    assert _signal_bias_for_epoch(5, 10, 1.0) == pytest.approx(0.5)
    assert _signal_bias_for_epoch(10, 10, 1.0) == pytest.approx(0.0)
    # Past warmup clips to zero, never goes negative.
    assert _signal_bias_for_epoch(11, 10, 1.0) == pytest.approx(0.0)
    # Negative magnitude (lay bias) scales identically.
    assert _signal_bias_for_epoch(5, 10, -0.4) == pytest.approx(-0.2)


# ── 3. No effect at / after warmup ─────────────────────────────────────────


@pytest.mark.parametrize("arch", _ALL_ARCHS)
def test_forward_bit_identical_past_warmup(arch):
    """Once the trainer's epoch counter reaches the warmup window, the
    bias computed is 0.0 and the forward pass is byte-identical to the
    unbiased path."""
    policy = _make_policy(arch)
    policy.eval()
    obs = _obs(seed=7)

    bias_post = _signal_bias_for_epoch(
        epoch=10, warmup=10, magnitude=1.0,
    )
    assert bias_post == 0.0

    with torch.no_grad():
        out_zero = policy(obs, None, signal_bias=0.0)
        out_post = policy(obs, None, signal_bias=bias_post)

    assert torch.equal(out_zero.action_mean, out_post.action_mean)
    assert torch.equal(out_zero.value, out_post.value)


# ── 4. Off by default ──────────────────────────────────────────────────────


def test_bias_off_by_default():
    """magnitude=0 or warmup=0 must yield 0.0 at every epoch."""
    # magnitude=0
    for e in range(0, 20):
        assert _signal_bias_for_epoch(e, 10, 0.0) == 0.0
    # warmup=0
    for e in range(0, 20):
        assert _signal_bias_for_epoch(e, 0, 1.0) == 0.0


@pytest.mark.parametrize("arch", _ALL_ARCHS)
def test_default_forward_unchanged(arch):
    """Calling forward without the keyword (default 0.0) is bit-identical
    to calling with signal_bias=0.0."""
    policy = _make_policy(arch)
    policy.eval()
    obs = _obs(seed=101)

    with torch.no_grad():
        out_default = policy(obs, None)
        out_explicit_zero = policy(obs, None, signal_bias=0.0)

    assert torch.equal(
        out_default.action_mean, out_explicit_zero.action_mean,
    )
    assert torch.equal(out_default.value, out_explicit_zero.value)


# ── 5. Bias only affects signal head ───────────────────────────────────────


@pytest.mark.parametrize("arch", _ALL_ARCHS)
def test_bias_only_affects_signal_head(arch):
    """Stake, aggression, cancel, and arb_spread means must be bit-
    identical with vs without the bias; only signal shifts."""
    policy = _make_policy(arch)
    policy.eval()
    obs = _obs(seed=13)

    with torch.no_grad():
        out_zero = policy(obs, None, signal_bias=0.0)
        out_bias = policy(obs, None, signal_bias=0.3)

    # Other heads live at indices [max_runners : per_runner*max_runners].
    other_zero = out_zero.action_mean[:, _MAX_RUNNERS:]
    other_bias = out_bias.action_mean[:, _MAX_RUNNERS:]
    assert torch.equal(other_zero, other_bias), (
        f"{arch}: non-signal heads shifted — bias leaked beyond head 0"
    )
    # Value head (critic) must also be untouched.
    assert torch.equal(out_zero.value, out_bias.value)


# ── 7. bet_rate / arb_rate in progress event ───────────────────────────────


def test_bet_rate_arb_rate_and_bias_active_in_progress_event(tmp_path):
    """``_publish_progress`` must carry bet_rate, arb_rate, and bias_active
    inside the action_stats dict. Both fractions in [0, 1] and
    ``bias_active`` matches the epoch/warmup/magnitude state."""
    # Warmup armed and current_epoch < warmup → bias_active should be True.
    trainer = _make_trainer(
        tmp_path,
        signal_bias_warmup=5,
        signal_bias_magnitude=0.3,
    )
    trainer._current_epoch = 2  # within warmup window

    ep = EpisodeStats(
        day_date="2026-04-14",
        total_reward=0.0,
        total_pnl=0.0,
        bet_count=3,
        winning_bets=0,
        races_completed=1,
        final_budget=100.0,
        n_steps=100,
        bet_rate=0.47,
        arb_rate=0.75,
        arbs_completed=3,
        arbs_naked=1,
    )

    loss_info = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        # Session 2 action_stats already populated — Session 3 augments it.
        "action_stats": {
            f"mean_entropy_{h}": 0.0 for h in _HEAD_NAMES
        } | {"entropy_collapse": False, "entropy_coeff_active": 0.01},
    }

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
    stats = captured[0]["action_stats"]

    assert "bet_rate" in stats and "arb_rate" in stats and "bias_active" in stats
    assert 0.0 <= stats["bet_rate"] <= 1.0
    assert 0.0 <= stats["arb_rate"] <= 1.0
    assert stats["bet_rate"] == pytest.approx(0.47)
    assert stats["arb_rate"] == pytest.approx(0.75)
    assert stats["bias_active"] is True

    # Flip: past-warmup OR magnitude=0 flips bias_active False.
    trainer._current_epoch = 5  # == warmup, no longer active
    captured.clear()
    trainer._publish_progress(ep, loss_info, _Tracker())
    assert captured[0]["action_stats"]["bias_active"] is False

    # magnitude=0 trainer → bias_active False regardless of epoch.
    trainer_off = _make_trainer(
        tmp_path,
        signal_bias_warmup=5,
        signal_bias_magnitude=0.0,
    )
    trainer_off._current_epoch = 2
    captured.clear()
    trainer_off.progress_queue = _FakeQueue()

    class _Tr:
        completed = 1
        total = 1

        def to_dict(self):
            return {"completed": 1, "total": 1}

    # Re-wire the queue on the new trainer.
    trainer_off.progress_queue = _FakeQueue()
    # Capture via the new queue.
    new_captured: list[dict] = []

    class _Q2:
        def put_nowait(self, item):
            new_captured.append(item)

    trainer_off.progress_queue = _Q2()
    trainer_off._publish_progress(ep, loss_info, _Tr())
    assert new_captured[0]["action_stats"]["bias_active"] is False
