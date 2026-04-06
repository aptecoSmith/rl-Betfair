"""
Session 2 — verify gamma / gae_lambda / value_loss_coeff are live genes.

These three PPO hyperparameters have always been *read* by PPOTrainer
via ``hp.get(..., default)``, but until now they were hardcoded defaults
because the search schema never exposed them. Session 2 promotes them
to mutable genes in ``config.yaml`` — this file is the regression
harness that proves:

1. The sampler produces all three in the expected ranges.
2. PPOTrainer picks them up (the "gene is actually read" lessons_learnt
   rule — every new gene must have this test).
3. Mutation keeps them inside their declared bounds.

CPU-only, no training loops, no market data. Should run in well under
one second total.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch.nn as nn
import yaml

from agents.population_manager import (
    PopulationManager,
    parse_search_ranges,
    sample_hyperparams,
)
from agents.ppo_trainer import PPOTrainer


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def real_search_ranges() -> dict:
    """The ``hyperparameters.search_ranges`` block from the live config."""
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["hyperparameters"]["search_ranges"]


@pytest.fixture
def trainer_config() -> dict:
    """Minimal config block satisfied by ``PPOTrainer.__init__``."""
    return {
        "training": {
            "max_runners": 14,
            "starting_budget": 100.0,
            "max_bets_per_race": 20,
        },
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "efficiency_penalty": 0.01,
            "precision_bonus": 1.0,
            "commission": 0.05,
        },
    }


# ── Test 1 — sampler emits the three genes in range ────────────────────────


def test_sampler_produces_ppo_genes_in_range(real_search_ranges):
    """All three PPO genes must be sampled, each within its declared range."""
    assert "gamma" in real_search_ranges
    assert "gae_lambda" in real_search_ranges
    assert "value_loss_coeff" in real_search_ranges

    specs = parse_search_ranges(real_search_ranges)
    rng = random.Random(0)

    for _ in range(50):
        hp = sample_hyperparams(specs, rng)
        assert "gamma" in hp
        assert "gae_lambda" in hp
        assert "value_loss_coeff" in hp

        g_spec = real_search_ranges["gamma"]
        l_spec = real_search_ranges["gae_lambda"]
        v_spec = real_search_ranges["value_loss_coeff"]
        assert g_spec["min"] <= hp["gamma"] <= g_spec["max"]
        assert l_spec["min"] <= hp["gae_lambda"] <= l_spec["max"]
        assert v_spec["min"] <= hp["value_loss_coeff"] <= v_spec["max"]


# ── Test 2 — extreme-value round-trip through PPOTrainer ───────────────────


def test_trainer_reads_ppo_genes_from_hyperparams(trainer_config):
    """Construct a PPOTrainer with values pinned near the range extremes
    and assert the trainer attributes match. This is the "gene is
    actually read" test mandated by lessons_learnt.md."""
    policy = nn.Linear(1, 1)
    hp = {
        "learning_rate": 3e-4,
        "gamma": 0.951,
        "gae_lambda": 0.901,
        "value_loss_coeff": 0.26,
    }
    trainer = PPOTrainer(
        policy=policy,
        config=trainer_config,
        hyperparams=hp,
        device="cpu",
    )

    assert trainer.gamma == pytest.approx(0.951)
    assert trainer.gae_lambda == pytest.approx(0.901)
    assert trainer.value_loss_coeff == pytest.approx(0.26)


def test_trainer_defaults_survive_when_genes_missing(trainer_config):
    """The ``hp.get(..., default)`` fallbacks in ppo_trainer.py exist so
    that agents loaded from old checkpoints (written before these genes
    were added) still train. Do NOT remove them — this test fails loud
    if a future refactor drops the default."""
    policy = nn.Linear(1, 1)
    trainer = PPOTrainer(
        policy=policy,
        config=trainer_config,
        hyperparams={"learning_rate": 3e-4},  # no gamma/gae/vlc
        device="cpu",
    )

    assert trainer.gamma == pytest.approx(0.99)
    assert trainer.gae_lambda == pytest.approx(0.95)
    assert trainer.value_loss_coeff == pytest.approx(0.5)


# ── Test 3 — mutation stays inside the declared range ─────────────────────


def test_mutation_keeps_ppo_genes_in_range(real_search_ranges):
    """Sample a genome, mutate it many times, and assert every mutated
    value remains inside the declared bounds. Catches off-by-one
    clamping bugs in PopulationManager.mutate."""
    # Build a throwaway PopulationManager just so we can call .mutate.
    # We avoid its full __init__ because that imports the env and wants
    # a real config tree; instead use __new__ and set the two fields
    # mutate() actually touches.
    pm = PopulationManager.__new__(PopulationManager)
    pm.hp_specs = parse_search_ranges(real_search_ranges)

    rng = random.Random(42)
    hp = sample_hyperparams(pm.hp_specs, rng)

    g_spec = real_search_ranges["gamma"]
    l_spec = real_search_ranges["gae_lambda"]
    v_spec = real_search_ranges["value_loss_coeff"]

    for _ in range(200):
        # mutation_rate=1.0 forces every gene to mutate every iteration,
        # so 200 iterations stresses each of the three genes ~200 times.
        hp, _ = pm.mutate(hp, mutation_rate=1.0, rng=rng)
        assert g_spec["min"] <= hp["gamma"] <= g_spec["max"]
        assert l_spec["min"] <= hp["gae_lambda"] <= l_spec["max"]
        assert v_spec["min"] <= hp["value_loss_coeff"] <= v_spec["max"]
