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

    def test_fresh_blood_samples_full_transformer_range_for_gpu_lane(self):
        """Un-capped 2026-06-04 for the GPU lane: fresh blood may again draw
        big-ctx transformers (the lane routes their forward+update to CUDA, so
        they no longer gate CPU generations). Big-ctx (>=128) are GPU-eligible;
        small transformers + every LSTM stay on the pure-CPU path."""
        from types import SimpleNamespace

        from training_v2.cohort.genes import (
            TRANSFORMER_CTX_TICKS_SAMPLE,
            is_gpu_lane_eligible,
        )
        rng = random.Random(7)
        ctx_seen = set()
        for _ in range(400):
            g = sample_fresh_blood_genes(rng)
            if g.architecture == "transformer":
                ctx_seen.add(g.transformer_ctx_ticks)
        assert ctx_seen <= set(TRANSFORMER_CTX_TICKS_SAMPLE)
        assert ctx_seen & {128, 256}, f"big-ctx never sampled: {ctx_seen}"

        def tf(ctx):
            return SimpleNamespace(architecture="transformer",
                                   transformer_ctx_ticks=ctx)
        assert is_gpu_lane_eligible(tf(256))
        assert is_gpu_lane_eligible(tf(128))
        assert not is_gpu_lane_eligible(tf(64))    # launch-bound on GPU
        assert not is_gpu_lane_eligible(tf(32))
        assert not is_gpu_lane_eligible(
            SimpleNamespace(architecture="lstm", transformer_ctx_ticks=0))

    def test_draws_both_lean_and_full_obs(self):
        """predictor_lean_obs is a fresh-blood OPTION — some lineages explore
        lean predictor obs, some full (operator 2026-06-04)."""
        rng = random.Random(4)
        seen = {
            sample_fresh_blood_genes(rng).predictor_lean_obs
            for _ in range(200)
        }
        assert seen == {True, False}, seen

    def test_predictor_lean_obs_is_structural_and_frozen(self):
        # It must be in the structural set so make_offspring freezes it
        # (lean<->full changes obs_dim -> weight shapes, breaks warm-start).
        assert "predictor_lean_obs" in ARCHITECTURE_GENE_NAMES

    def test_lstm_fresh_blood_large_transformer_to_512_on_gpu_lane(self):
        """Fresh-blood LSTMs may draw large hidden sizes (512/1024); a
        transformer's d_model (hidden_size) now reaches 512 (pbt-gpu-forward:
        big-ctx transformers train on the GPU lane, so the prior CPU-bound
        256 cap is lifted to 512). 1024 stays LSTM-only (d_model=1024
        attention is heavy even on GPU). The gene-only GA is unchanged
        (64/128/256 only) -> byte-identity."""
        from training_v2.cohort.genes import sample_genes
        rng = random.Random(13)
        lstm_h, tf_h = set(), set()
        for _ in range(400):
            g = sample_fresh_blood_genes(rng)
            (lstm_h if g.architecture == "lstm" else tf_h).add(g.hidden_size)
            assert_in_range(g)  # the widened _VALID set must accept it
        assert lstm_h & {512, 1024}, f"LSTM never went large: {lstm_h}"
        assert 512 in tf_h, f"transformer never reached 512: {tf_h}"
        assert tf_h <= {64, 128, 256, 512}, f"transformer d_model > 512: {tf_h}"
        # gene-only GA stays at the original sizes (byte-identity).
        ga = {sample_genes(random.Random(s)).hidden_size for s in range(80)}
        assert ga <= {64, 128, 256}, ga

    def test_transformer_config_genes_drawn_and_structural(self):
        """pbt-gpu-forward: fresh-blood transformers draw transformer_ffn_mult
        {2,4} and (sampling-gated) transformer_pos_encoding; both are
        STRUCTURAL (frozen per lineage — they change weight shapes / the
        encoder module set, so warm-start must not cross them). depth now
        reaches 4/6 for the GPU lane."""
        rng = random.Random(21)
        ffn, pos, depth = set(), set(), set()
        for _ in range(400):
            g = sample_fresh_blood_genes(rng)
            assert_in_range(g)
            if g.architecture == "transformer":
                ffn.add(g.transformer_ffn_mult)
                pos.add(g.transformer_pos_encoding)
                depth.add(g.transformer_depth)
        assert ffn == {2, 4}, f"ffn_mult not fully sampled: {ffn}"
        # "rope" is sampling-gated until the policy implements it (task #8),
        # so fresh blood currently only draws "learned".
        assert pos == {"learned"}, f"pos_encoding sampled rope too early: {pos}"
        assert {4, 6} & depth, f"transformer depth never went deep: {depth}"
        assert "transformer_ffn_mult" in ARCHITECTURE_GENE_NAMES
        assert "transformer_pos_encoding" in ARCHITECTURE_GENE_NAMES

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
