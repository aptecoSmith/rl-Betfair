"""Session 9 — GPU smoke test (the one GPU-allowed session).

Instantiates each registered architecture directly on CUDA, runs one
forward pass + one backward pass, and asserts the gradients are finite.
This is a sanity check — it does **not** run a training loop.

Marked ``@pytest.mark.gpu`` so it's deselected by default (see
``pyproject.toml`` / ``plans/arch-exploration/testing.md``). Run with::

    pytest tests/arch_exploration/test_gpu_smoke.py -m gpu
"""

from __future__ import annotations

import pytest
import torch

from agents.architecture_registry import REGISTRY, create_policy
from agents.policy_network import (
    AGENT_STATE_DIM,
    MARKET_DIM,
    POSITION_DIM,
    RUNNER_DIM,
    VELOCITY_DIM,
)


MAX_RUNNERS = 14  # matches config.yaml training.max_runners


def _obs_dim(max_runners: int = MAX_RUNNERS) -> int:
    return (
        MARKET_DIM
        + VELOCITY_DIM
        + RUNNER_DIM * max_runners
        + AGENT_STATE_DIM
        + POSITION_DIM * max_runners
    )


def _action_dim(max_runners: int = MAX_RUNNERS) -> int:
    return max_runners * 2


#: Minimal hyperparameter dict big enough for every arch but small
#: enough to fit comfortably on any modern GPU.
def _smoke_hp() -> dict:
    return {
        "lstm_hidden_size": 128,
        "mlp_hidden_size": 64,
        "mlp_layers": 2,
        "lstm_num_layers": 2,
        "lstm_dropout": 0.1,
        "lstm_layer_norm": 1,
        "transformer_heads": 4,
        "transformer_depth": 2,
        "transformer_ctx_ticks": 32,
    }


@pytest.mark.gpu
def test_cuda_is_available() -> None:
    """Bail out loudly if the GPU session is run without CUDA."""
    assert torch.cuda.is_available(), (
        "Session 9 GPU smoke test requires CUDA. "
        "Install CUDA-enabled PyTorch or run on a GPU host."
    )


@pytest.mark.gpu
@pytest.mark.parametrize("arch_name", sorted(REGISTRY.keys()))
def test_arch_forward_and_backward_on_cuda(arch_name: str) -> None:
    """Each registered architecture: forward + backward on CUDA, no NaNs."""
    device = torch.device("cuda")
    obs_dim = _obs_dim()
    action_dim = _action_dim()

    policy = create_policy(
        name=arch_name,
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_runners=MAX_RUNNERS,
        hyperparams=_smoke_hp(),
    ).to(device)

    # A 2-step rollout: (batch=1, seq_len=2, obs_dim)
    # The existing policies accept 2-D single-step obs OR 3-D sequences;
    # use 2-D to match the training hot path.
    obs = torch.randn(1, obs_dim, device=device)
    hidden = policy.init_hidden(batch_size=1)
    hidden = (hidden[0].to(device), hidden[1].to(device))

    out = policy(obs, hidden)

    # All outputs must be finite.
    assert torch.isfinite(out.action_mean).all(), f"{arch_name}: NaN in action_mean"
    assert torch.isfinite(out.action_log_std).all(), f"{arch_name}: NaN in log_std"
    assert torch.isfinite(out.value).all(), f"{arch_name}: NaN in value"

    # Fake a loss and backprop. Use a combined loss so the backward
    # exercises every head.
    loss = out.action_mean.sum() + out.value.sum() + out.action_log_std.sum()
    loss.backward()

    # Verify at least one parameter actually received a gradient and
    # none of the gradients contain NaN or inf.
    any_grad = False
    for name, p in policy.named_parameters():
        if p.grad is None:
            continue
        any_grad = True
        assert torch.isfinite(p.grad).all(), (
            f"{arch_name}: non-finite grad in {name}"
        )
    assert any_grad, f"{arch_name}: no parameter received a gradient"
