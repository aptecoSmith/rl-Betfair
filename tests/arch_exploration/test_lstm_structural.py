"""
Session 5 — LSTM structural knobs as live genes.

Verifies ``lstm_num_layers`` ∈ {1, 2}, ``lstm_dropout`` ∈ [0, 0.3] and
``lstm_layer_norm`` ∈ {false, true} are plumbed all the way through
both ``PPOLSTMPolicy`` (stock ``nn.LSTM``) and ``PPOTimeLSTMPolicy``
(stacked ``TimeLSTMCell``). CPU-only, no training loops.
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from agents.architecture_registry import create_policy
from agents.policy_network import (
    AGENT_STATE_DIM,
    MARKET_DIM,
    POSITION_DIM,
    RUNNER_DIM,
    VELOCITY_DIM,
    PPOLSTMPolicy,
    PPOTimeLSTMPolicy,
    TimeLSTMCell,
)
from agents.population_manager import parse_search_ranges, sample_hyperparams


# ── Fixtures ────────────────────────────────────────────────────────────────


MAX_RUNNERS = 12


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


@pytest.fixture
def real_search_ranges() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)["hyperparameters"]["search_ranges"]


# ── 1. Gene sampling ────────────────────────────────────────────────────────


def test_sampler_emits_all_lstm_structural_genes(real_search_ranges):
    specs = parse_search_ranges(real_search_ranges)
    rng = random.Random(1234)
    seen_num_layers: set[int] = set()
    seen_layer_norm: set[int] = set()

    for _ in range(200):
        hp = sample_hyperparams(specs, rng)
        assert "lstm_num_layers" in hp
        assert "lstm_dropout" in hp
        assert "lstm_layer_norm" in hp
        assert hp["lstm_num_layers"] in (1, 2)
        assert 0.0 <= hp["lstm_dropout"] <= 0.3
        assert hp["lstm_layer_norm"] in (0, 1)
        seen_num_layers.add(hp["lstm_num_layers"])
        seen_layer_norm.add(hp["lstm_layer_norm"])

    # Seeded 200-sample sweep should touch both ends of each choice gene.
    assert seen_num_layers == {1, 2}
    assert seen_layer_norm == {0, 1}


# ── 2. Policy instantiation grid + forward pass ────────────────────────────


_ARCHS = ("ppo_lstm_v1", "ppo_time_lstm_v1")
_NUM_LAYERS = (1, 2)
_DROPOUT = (0.0, 0.2)
_LAYER_NORM = (False, True)


@pytest.mark.parametrize(
    "arch,num_layers,dropout,layer_norm",
    list(itertools.product(_ARCHS, _NUM_LAYERS, _DROPOUT, _LAYER_NORM)),
)
def test_policy_instantiates_and_forwards(arch, num_layers, dropout, layer_norm):
    obs_dim = _obs_dim()
    action_dim = _action_dim()
    hp = {
        "lstm_hidden_size": 32,
        "mlp_hidden_size": 16,
        "mlp_layers": 1,
        "lstm_num_layers": num_layers,
        "lstm_dropout": dropout,
        "lstm_layer_norm": layer_norm,
    }
    policy = create_policy(
        name=arch,
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_runners=MAX_RUNNERS,
        hyperparams=hp,
    )
    policy.eval()

    # Structural attribute round-trip — proves the gene reached the module.
    assert policy.lstm_num_layers == num_layers
    assert policy.lstm_dropout == dropout
    assert policy.lstm_layer_norm_enabled == layer_norm
    if layer_norm:
        assert isinstance(policy.lstm_output_norm, nn.LayerNorm)
    else:
        assert isinstance(policy.lstm_output_norm, nn.Identity)

    obs = torch.zeros(2, obs_dim)
    out = policy(obs)
    assert out.action_mean.shape == (2, action_dim)
    assert out.action_log_std.shape == (2, action_dim)
    assert out.value.shape == (2, 1)
    # Hidden state must carry the stacked layer dim.
    h, c = out.hidden_state
    assert h.shape == (num_layers, 2, 32)
    assert c.shape == (num_layers, 2, 32)


# ── 3. Hidden-state init across num_layers ──────────────────────────────────


@pytest.mark.parametrize("arch", _ARCHS)
def test_init_hidden_shape_matches_num_layers(arch):
    obs_dim = _obs_dim()
    action_dim = _action_dim()
    policy = create_policy(
        name=arch,
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_runners=MAX_RUNNERS,
        hyperparams={
            "lstm_hidden_size": 24,
            "mlp_hidden_size": 16,
            "mlp_layers": 1,
            "lstm_num_layers": 2,
        },
    )
    h, c = policy.init_hidden(batch_size=4)
    assert h.shape == (2, 4, 24)
    assert c.shape == (2, 4, 24)
    assert torch.equal(h, torch.zeros_like(h))
    assert torch.equal(c, torch.zeros_like(c))


# ── 4. Stacked TimeLSTMCell behaviour ───────────────────────────────────────


def test_stacked_time_lstm_cell_forwards_two_timesteps_and_respects_train_eval():
    obs_dim = _obs_dim()
    action_dim = _action_dim()
    hp = {
        "lstm_hidden_size": 32,
        "mlp_hidden_size": 16,
        "mlp_layers": 1,
        "lstm_num_layers": 2,
        "lstm_dropout": 0.5,  # large, so if dropout is active zeroing will hit
    }
    policy = create_policy(
        name="ppo_time_lstm_v1",
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_runners=MAX_RUNNERS,
        hyperparams=hp,
    )

    # Cell stack is a ModuleList of length num_layers.
    assert isinstance(policy.time_lstm_cells, nn.ModuleList)
    assert len(policy.time_lstm_cells) == 2
    for cell in policy.time_lstm_cells:
        assert isinstance(cell, TimeLSTMCell)

    # Two-timestep sequence — explicit seq dim. Use a meaningful
    # magnitude so the actor head's 0.01-gain init doesn't collapse
    # the signal into floating-point noise.
    obs = torch.randn(3, 2, obs_dim)
    # Eval mode: dropout disabled, two forward passes on the same input
    # must be bit-for-bit identical regardless of RNG state.
    policy.eval()
    torch.manual_seed(0)
    out_a = policy(obs)
    torch.manual_seed(1)  # different seed — should not matter in eval
    out_b = policy(obs)
    assert torch.allclose(out_a.value, out_b.value, atol=1e-6)
    assert out_a.action_mean.shape == (3, action_dim)
    assert out_a.value.shape == (3, 1)
    h, c = out_a.hidden_state
    assert h.shape == (2, 3, 32)
    assert c.shape == (2, 3, 32)

    # Train mode with heavy dropout — two passes should diverge because
    # inter-layer dropout uses a fresh mask each time. Check the value
    # head: the critic head uses gain=1.0 init so differences show up
    # at normal tolerances (the actor head's 0.01-gain init would
    # attenuate the signal into float-noise).
    policy.train()
    torch.manual_seed(0)
    out_train_a = policy(obs)
    torch.manual_seed(1)
    out_train_b = policy(obs)
    assert not torch.allclose(
        out_train_a.value, out_train_b.value, atol=1e-4
    )


# ── 5. Backward compatibility ───────────────────────────────────────────────


@pytest.mark.parametrize("arch", _ARCHS)
def test_policy_defaults_without_new_keys(arch):
    """Checkpoints / call sites that don't know about the Session 5
    genes must still produce a single-layer, no-dropout, no-layer-norm
    policy — identical behaviour to pre-Session-5."""
    obs_dim = _obs_dim()
    action_dim = _action_dim()
    hp = {
        "lstm_hidden_size": 32,
        "mlp_hidden_size": 16,
        "mlp_layers": 1,
    }
    policy = create_policy(
        name=arch,
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_runners=MAX_RUNNERS,
        hyperparams=hp,
    )
    assert policy.lstm_num_layers == 1
    assert policy.lstm_dropout == 0.0
    assert policy.lstm_layer_norm_enabled is False
    assert isinstance(policy.lstm_output_norm, nn.Identity)

    # Forward pass with the default hidden state must work and return
    # the (1, batch, hidden) hidden shape that pre-Session-5 code used.
    policy.eval()
    out = policy(torch.zeros(1, obs_dim))
    h, c = out.hidden_state
    assert h.shape == (1, 1, 32)
    assert c.shape == (1, 1, 32)
