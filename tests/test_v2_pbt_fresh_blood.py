"""Fresh-blood gene sampling + structural-gene freeze (pbt-breeding Step 1b).

HC#9: fresh blood samples the FULL gene space INCLUDING architecture.
HC#10: structural genes are frozen within a lineage — crossover/mutate
never touch them. HC#1: adding the architecture genes must NOT shift the
RNG stream of the base sampler / crossover / mutate (else --breeding pbt
off would stop being byte-identical to the gene-only GA).
"""

from __future__ import annotations

import random
from dataclasses import fields

from training_v2.cohort.genes import (
    ARCHITECTURE_CHOICES,
    ARCHITECTURE_GENE_NAMES,
    TRANSFORMER_CTX_TICKS_CHOICES,
    TRANSFORMER_DEPTH_CHOICES,
    TRANSFORMER_HEADS_CHOICES,
    CohortGenes,
    assert_in_range,
    crossover,
    mutate,
    sample_fresh_blood_genes,
    sample_genes,
)


class TestFreshBloodSamplesArchitecture:
    def test_draws_both_architectures_over_many_samples(self):
        rng = random.Random(1)
        seen = {
            sample_fresh_blood_genes(rng).architecture for _ in range(200)
        }
        assert seen == set(ARCHITECTURE_CHOICES), seen

    def test_every_draw_is_in_range(self):
        rng = random.Random(2)
        for _ in range(200):
            g = sample_fresh_blood_genes(rng)
            assert_in_range(g)  # must not raise
            assert g.transformer_depth in TRANSFORMER_DEPTH_CHOICES
            assert g.transformer_heads in TRANSFORMER_HEADS_CHOICES
            assert g.transformer_ctx_ticks in TRANSFORMER_CTX_TICKS_CHOICES

    def test_transformer_d_model_divisible_by_heads(self):
        # Every (hidden_size, n_heads) combo the sampler can draw must
        # satisfy the transformer's d_model % n_heads == 0 constraint, so
        # a fresh-blood transformer always builds.
        rng = random.Random(3)
        for _ in range(300):
            g = sample_fresh_blood_genes(rng)
            if g.architecture == "transformer":
                assert g.hidden_size % g.transformer_heads == 0, (
                    g.hidden_size, g.transformer_heads,
                )


class TestBaseSamplerByteIdentity:
    def test_base_sample_genes_always_lstm(self):
        rng = random.Random(0)
        for _ in range(50):
            g = sample_genes(rng)
            assert g.architecture == "lstm"
            assert g.transformer_depth == 2
            assert g.transformer_heads == 4
            assert g.transformer_ctx_ticks == 32


class TestStructuralGenesFreezeUnderBreeding:
    """crossover/mutate never alter the architecture genes AND never
    consume an RNG draw for them (HC#1 + HC#10)."""

    def _two_parents(self):
        rng = random.Random(11)
        a = sample_fresh_blood_genes(rng)
        b = sample_fresh_blood_genes(rng)
        return a, b

    def test_crossover_inherits_architecture_from_parent_a(self):
        a, b = self._two_parents()
        child = crossover(a, b, random.Random(5))
        for name in ARCHITECTURE_GENE_NAMES:
            assert getattr(child, name) == getattr(a, name)

    def test_mutate_never_changes_architecture(self):
        a, _ = self._two_parents()
        m = mutate(a, random.Random(5), mutation_rate=1.0)
        for name in ARCHITECTURE_GENE_NAMES:
            assert getattr(m, name) == getattr(a, name)

    def test_architecture_genes_do_not_shift_mutate_rng_stream(self):
        """Two genomes identical except in their (last-in-field-order,
        skipped-without-rng) architecture genes must mutate to the SAME
        non-architecture genes under the same seed — proving the
        architecture genes consume no RNG (HC#1 byte-identity)."""
        rng = random.Random(11)
        base = sample_fresh_blood_genes(rng)
        kw = {f.name: getattr(base, f.name) for f in fields(CohortGenes)}
        g_lstm = CohortGenes(**{
            **kw, "architecture": "lstm", "transformer_depth": 2,
            "transformer_heads": 4, "transformer_ctx_ticks": 32,
        })
        g_tr = CohortGenes(**{
            **kw, "architecture": "transformer", "transformer_depth": 3,
            "transformer_heads": 8, "transformer_ctx_ticks": 256,
        })
        m_lstm = mutate(g_lstm, random.Random(7), mutation_rate=0.5,
                        enabled_set=frozenset())
        m_tr = mutate(g_tr, random.Random(7), mutation_rate=0.5,
                      enabled_set=frozenset())
        for f in fields(CohortGenes):
            if f.name in ARCHITECTURE_GENE_NAMES:
                continue
            assert getattr(m_lstm, f.name) == getattr(m_tr, f.name), f.name
        # And the architecture genes themselves passed through untouched.
        assert m_lstm.architecture == "lstm"
        assert m_tr.architecture == "transformer"
        assert m_tr.transformer_ctx_ticks == 256

    def test_crossover_architecture_genes_do_not_shift_rng_stream(self):
        rng = random.Random(11)
        base = sample_fresh_blood_genes(rng)
        kw = {f.name: getattr(base, f.name) for f in fields(CohortGenes)}
        other = sample_fresh_blood_genes(rng)
        kw_other = {f.name: getattr(other, f.name) for f in fields(CohortGenes)}
        # parent_b differs in non-arch genes (so crossover picks vary), but
        # parent_a's arch differs only — verify non-arch crossover output is
        # invariant to parent_a's architecture-gene values.
        a_lstm = CohortGenes(**{**kw, "architecture": "lstm"})
        a_tr = CohortGenes(**{**kw, "architecture": "transformer",
                              "transformer_ctx_ticks": 256})
        b = CohortGenes(**kw_other)
        c_lstm = crossover(a_lstm, b, random.Random(9))
        c_tr = crossover(a_tr, b, random.Random(9))
        for f in fields(CohortGenes):
            if f.name in ARCHITECTURE_GENE_NAMES:
                continue
            assert getattr(c_lstm, f.name) == getattr(c_tr, f.name), f.name
