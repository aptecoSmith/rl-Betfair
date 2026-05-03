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
    ALPHA_LR_RANGE,
    ARB_SPREAD_SCALE_RANGE,
    CLIP_RANGE_RANGE,
    ENTROPY_COEFF_RANGE,
    FILL_PROB_LOSS_WEIGHT_RANGE,
    GAE_LAMBDA_RANGE,
    HIDDEN_SIZE_CHOICES,
    LEARNING_RATE_RANGE,
    MARK_TO_MARKET_WEIGHT_RANGE,
    MATURE_PROB_LOSS_WEIGHT_RANGE,
    MATURED_ARB_BONUS_WEIGHT_RANGE,
    MINI_BATCH_SIZE_CHOICES,
    NAKED_LOSS_SCALE_RANGE,
    OPEN_COST_RANGE,
    PHASE5_GENE_DEFAULTS,
    PHASE5_GENE_NAMES,
    REWARD_CLIP_RANGE,
    RISK_LOSS_WEIGHT_RANGE,
    STOP_LOSS_PNL_THRESHOLD_RANGE,
    VALUE_COEFF_RANGE,
    CohortGenes,
    assert_in_range,
    crossover,
    mutate,
    sample_genes,
)


_PHASE5_RANGE_TABLE: dict[str, tuple[float, float]] = {
    "open_cost": OPEN_COST_RANGE,
    "matured_arb_bonus_weight": MATURED_ARB_BONUS_WEIGHT_RANGE,
    "mark_to_market_weight": MARK_TO_MARKET_WEIGHT_RANGE,
    "naked_loss_scale": NAKED_LOSS_SCALE_RANGE,
    "stop_loss_pnl_threshold": STOP_LOSS_PNL_THRESHOLD_RANGE,
    "arb_spread_scale": ARB_SPREAD_SCALE_RANGE,
    "fill_prob_loss_weight": FILL_PROB_LOSS_WEIGHT_RANGE,
    "mature_prob_loss_weight": MATURE_PROB_LOSS_WEIGHT_RANGE,
    "risk_loss_weight": RISK_LOSS_WEIGHT_RANGE,
    "alpha_lr": ALPHA_LR_RANGE,
    "reward_clip": REWARD_CLIP_RANGE,
}


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


# ── Phase 5 (restore-genes, 2026-05-03) ───────────────────────────────────


class TestPhase5Genes:
    """Per-agent gene promotion with operator-controlled enable/disable
    per cohort. Disabled genes stay frozen at their pre-Phase-5
    cohort-wide default; enabled genes sample / mutate / crossover
    normally."""

    def test_legacy_sample_genes_signature_still_works(self):
        """``sample_genes(rng)`` (no ``enabled_set``) returns a valid
        ``CohortGenes`` whose Phase 5 fields are all at default."""
        rng = random.Random(42)
        genes = sample_genes(rng)
        assert_in_range(genes)
        for name, default in PHASE5_GENE_DEFAULTS.items():
            assert getattr(genes, name) == default

    def test_legacy_sampling_byte_identical(self):
        """Two ``sample_genes(rng)`` calls with same seed and empty
        ``enabled_set`` produce identical output."""
        rng_a = random.Random(42)
        rng_b = random.Random(42)
        assert sample_genes(rng_a) == sample_genes(rng_b)

    def test_legacy_default_matches_pre_plan_cohort_wide_values(self):
        """Each Phase 5 default must match the pre-plan cohort-wide
        default the env / trainer was using before the gene was
        promoted. This is the load-bearing byte-identity invariant."""
        expected = {
            "open_cost": 0.0,
            "matured_arb_bonus_weight": 0.0,
            "mark_to_market_weight": 0.05,
            "naked_loss_scale": 1.0,
            "stop_loss_pnl_threshold": 0.0,
            "arb_spread_scale": 1.0,
            "fill_prob_loss_weight": 0.0,
            "mature_prob_loss_weight": 0.0,
            "risk_loss_weight": 0.0,
            "alpha_lr": 1e-2,
            "reward_clip": 10.0,
        }
        assert PHASE5_GENE_DEFAULTS == expected

    def test_enabled_gene_is_sampled(self):
        """A gene in ``enabled_set`` produces varied values across
        seeds (not all at the disabled-default)."""
        values = set()
        for seed in range(20):
            rng = random.Random(seed)
            genes = sample_genes(
                rng, enabled_set=frozenset({"open_cost"}),
            )
            values.add(genes.open_cost)
        # Uniform [0, 2] over 20 seeds — overwhelmingly distinct.
        assert len(values) > 10
        # Other Phase 5 genes still at default.
        for name, default in PHASE5_GENE_DEFAULTS.items():
            if name == "open_cost":
                continue
            rng = random.Random(0)
            g = sample_genes(rng, enabled_set=frozenset({"open_cost"}))
            assert getattr(g, name) == default

    def test_disabled_gene_stays_at_default_during_mutation(self):
        """``mutate`` with ``mutation_rate=1.0`` and empty
        ``enabled_set`` leaves all Phase 5 genes at default even
        though the legacy 7 are re-sampled."""
        rng = random.Random(42)
        genes = sample_genes(rng)
        rng2 = random.Random(99)
        mutated = mutate(genes, rng2, mutation_rate=1.0)
        for name, default in PHASE5_GENE_DEFAULTS.items():
            assert getattr(mutated, name) == default

    def test_enabled_gene_mutates(self):
        """``mutate`` with ``mutation_rate=1.0`` re-samples enabled
        Phase 5 genes."""
        enabled = frozenset({"open_cost", "alpha_lr"})
        rng = random.Random(42)
        genes = sample_genes(rng, enabled_set=enabled)
        rng2 = random.Random(99)
        mutated = mutate(
            genes, rng2, mutation_rate=1.0, enabled_set=enabled,
        )
        # Continuous draws — collision probability is zero.
        assert mutated.open_cost != genes.open_cost
        assert mutated.alpha_lr != genes.alpha_lr
        # Disabled genes still pinned to default.
        assert mutated.matured_arb_bonus_weight == \
            PHASE5_GENE_DEFAULTS["matured_arb_bonus_weight"]

    def test_disabled_gene_in_crossover_takes_default(self):
        """``crossover`` with empty ``enabled_set`` puts the default
        on every Phase 5 gene of the child, regardless of what the
        parents carried (a manually-constructed CohortGenes can
        carry non-default Phase 5 values)."""
        parent_a = CohortGenes(
            learning_rate=1e-3, entropy_coeff=1e-3, clip_range=0.2,
            gae_lambda=0.95, value_coeff=0.5, mini_batch_size=64,
            hidden_size=128,
            open_cost=1.5, mark_to_market_weight=0.08,
        )
        parent_b = CohortGenes(
            learning_rate=2e-4, entropy_coeff=1e-2, clip_range=0.1,
            gae_lambda=0.98, value_coeff=0.7, mini_batch_size=128,
            hidden_size=256,
            open_cost=0.5, mark_to_market_weight=0.02,
        )
        rng = random.Random(42)
        child = crossover(parent_a, parent_b, rng)
        assert child.open_cost == 0.0
        assert child.mark_to_market_weight == 0.05

    def test_enabled_gene_in_crossover_inherits_from_parent(self):
        """An enabled gene in the child must come from one of the
        two parents, never the default (unless the parent's value
        happens to equal the default)."""
        parent_a = CohortGenes(
            learning_rate=1e-3, entropy_coeff=1e-3, clip_range=0.2,
            gae_lambda=0.95, value_coeff=0.5, mini_batch_size=64,
            hidden_size=128,
            open_cost=1.5,
        )
        parent_b = CohortGenes(
            learning_rate=2e-4, entropy_coeff=1e-2, clip_range=0.1,
            gae_lambda=0.98, value_coeff=0.7, mini_batch_size=128,
            hidden_size=256,
            open_cost=0.7,
        )
        enabled = frozenset({"open_cost"})
        for seed in range(20):
            rng = random.Random(seed)
            child = crossover(parent_a, parent_b, rng, enabled_set=enabled)
            assert child.open_cost in (1.5, 0.7)

    def test_each_phase5_gene_range_respected(self):
        """Sampled value of every Phase 5 gene lies inside its
        documented range."""
        for name, (lo, hi) in _PHASE5_RANGE_TABLE.items():
            for seed in range(20):
                rng = random.Random(seed)
                genes = sample_genes(
                    rng, enabled_set=frozenset({name}),
                )
                value = getattr(genes, name)
                assert lo <= value <= hi, (
                    f"{name}={value} outside [{lo}, {hi}] at seed {seed}"
                )

    def test_assert_in_range_validates_phase5_genes(self):
        """``assert_in_range`` raises when a Phase 5 gene is out
        of bounds."""
        bad = CohortGenes(
            learning_rate=1e-3, entropy_coeff=1e-3, clip_range=0.2,
            gae_lambda=0.95, value_coeff=0.5, mini_batch_size=64,
            hidden_size=128,
            open_cost=3.0,  # > OPEN_COST_RANGE upper of 2.0
        )
        with pytest.raises(ValueError):
            assert_in_range(bad)

    def test_to_dict_serialises_all_18_fields(self):
        rng = random.Random(0)
        genes = sample_genes(rng, enabled_set=PHASE5_GENE_NAMES)
        d = genes.to_dict()
        # 7 legacy + 11 Phase 5
        assert len(d) == 18
        for name in PHASE5_GENE_NAMES:
            assert name in d
            assert isinstance(d[name], float)

    def test_phase5_gene_names_set_size(self):
        assert len(PHASE5_GENE_NAMES) == 11
        assert PHASE5_GENE_NAMES == frozenset(PHASE5_GENE_DEFAULTS)
