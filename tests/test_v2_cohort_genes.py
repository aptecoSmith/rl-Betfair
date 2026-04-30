"""Tests for the Phase 3 cohort gene schema (Session 03).

Covers:
- Sampling produces in-range genes (every gene, every type).
- Crossover preserves the parent-only invariant: each child gene is
  one of the two parents'.
- Mutation: rate=0 is identity; rate=1 always re-samples (categoricals
  may collide so we test "every gene was redrawn from the schema",
  not "every gene differs from parent").
"""

from __future__ import annotations

import random

import pytest

from training_v2.cohort.genes import (
    CLIP_RANGE_RANGE,
    ENTROPY_COEFF_RANGE,
    GAE_LAMBDA_RANGE,
    HIDDEN_SIZE_CHOICES,
    LEARNING_RATE_RANGE,
    MINI_BATCH_SIZE_CHOICES,
    VALUE_COEFF_RANGE,
    CohortGenes,
    assert_in_range,
    crossover,
    mutate,
    sample_genes,
)


def test_sample_genes_in_range_for_many_seeds():
    """Every sampled gene set lands inside the locked schema's range."""
    for seed in range(100):
        rng = random.Random(seed)
        g = sample_genes(rng)
        assert_in_range(g)
        # Hand-verified field coverage
        assert LEARNING_RATE_RANGE[0] <= g.learning_rate <= LEARNING_RATE_RANGE[1]
        assert ENTROPY_COEFF_RANGE[0] <= g.entropy_coeff <= ENTROPY_COEFF_RANGE[1]
        assert CLIP_RANGE_RANGE[0] <= g.clip_range <= CLIP_RANGE_RANGE[1]
        assert GAE_LAMBDA_RANGE[0] <= g.gae_lambda <= GAE_LAMBDA_RANGE[1]
        assert VALUE_COEFF_RANGE[0] <= g.value_coeff <= VALUE_COEFF_RANGE[1]
        assert g.mini_batch_size in MINI_BATCH_SIZE_CHOICES
        assert g.hidden_size in HIDDEN_SIZE_CHOICES


def test_sample_genes_is_deterministic():
    """Same seed → same genes (locked-schema reproducibility)."""
    rng_a = random.Random(123)
    rng_b = random.Random(123)
    assert sample_genes(rng_a) == sample_genes(rng_b)


def test_crossover_child_genes_come_from_parents():
    """Every child gene is one of the two parents' values."""
    rng_p = random.Random(7)
    parent_a = sample_genes(rng_p)
    parent_b = sample_genes(rng_p)
    rng_c = random.Random(99)
    for _ in range(50):
        child = crossover(parent_a, parent_b, rng_c)
        for fname in (
            "learning_rate", "entropy_coeff", "clip_range", "gae_lambda",
            "value_coeff", "mini_batch_size", "hidden_size",
        ):
            cv = getattr(child, fname)
            assert cv in (
                getattr(parent_a, fname), getattr(parent_b, fname),
            ), f"{fname}={cv} not from parent"


def test_mutate_rate_zero_is_identity():
    rng = random.Random(11)
    parent = sample_genes(rng)
    out = mutate(parent, rng, mutation_rate=0.0)
    assert out == parent


def test_mutate_rate_one_resamples_every_gene():
    """At rate=1 every gene is redrawn from the schema range.

    For categoricals the redraw may collide with the parent value, so
    we test "every gene is in-range" rather than "every gene differs".
    For floats the collision probability is zero (continuous), so we
    additionally test that floats actually changed across most seeds.
    """
    rng_p = random.Random(2)
    parent = CohortGenes(
        learning_rate=1e-4,
        entropy_coeff=1e-3,
        clip_range=0.2,
        gae_lambda=0.95,
        value_coeff=0.5,
        mini_batch_size=64,
        hidden_size=128,
    )
    n_floats_changed = 0
    n_trials = 50
    for seed in range(n_trials):
        rng = random.Random(seed)
        out = mutate(parent, rng, mutation_rate=1.0)
        assert_in_range(out)
        if (
            out.learning_rate != parent.learning_rate
            and out.entropy_coeff != parent.entropy_coeff
            and out.clip_range != parent.clip_range
            and out.gae_lambda != parent.gae_lambda
            and out.value_coeff != parent.value_coeff
        ):
            n_floats_changed += 1
    # All five floats colliding by chance is impossibly rare with
    # log-uniform / uniform draws.
    assert n_floats_changed >= n_trials - 1


def test_mutate_rate_invalid():
    rng = random.Random(0)
    g = sample_genes(rng)
    with pytest.raises(ValueError):
        mutate(g, rng, mutation_rate=-0.1)
    with pytest.raises(ValueError):
        mutate(g, rng, mutation_rate=1.5)


def test_assert_in_range_rejects_out_of_range():
    bad = CohortGenes(
        learning_rate=10.0,  # way out of range
        entropy_coeff=1e-3,
        clip_range=0.2,
        gae_lambda=0.95,
        value_coeff=0.5,
        mini_batch_size=64,
        hidden_size=128,
    )
    with pytest.raises(ValueError):
        assert_in_range(bad)
