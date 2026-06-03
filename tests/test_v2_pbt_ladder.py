"""PBT promotion ladder + day rotation (pbt-breeding Steps 2-3).

Unit-gates the science core in ``training_v2/cohort/pbt.py``: the
rotation day-split (disjoint, deterministic, no sealed leak), offspring
structural-freeze, and the promotion-ladder partition (counts, tiers,
lineage propagation, warm-start pointers, hall-of-fame freeze) across the
transient → steady-state pipeline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, fields

import pytest

from training_v2.cohort.genes import (
    ARCHITECTURE_GENE_NAMES,
    PHASE5_GENE_NAMES,
    CohortGenes,
    sample_fresh_blood_genes,
)
from training_v2.cohort.pbt import (
    STRUCTURAL_GENE_NAMES,
    PbtConfig,
    breed_pbt,
    init_pbt_population,
    make_offspring,
    make_rotations,
)


# A duck-typed stand-in for AgentResult (breed_pbt reads .weights_path /
# .model_id; score_fn reads .score).
@dataclass
class _Res:
    model_id: str
    weights_path: str
    score: float


def _score(r: _Res) -> float:
    return r.score


def _cfg(**kw) -> PbtConfig:
    base = dict(
        n_agents=30, r2_size=10, r3_size=6,
        promote_from_r1=5, promote_from_r2=3, freeze_top_r3=3,
    )
    base.update(kw)
    return PbtConfig(**base)


# ── Day rotation ──────────────────────────────────────────────────────────


class TestMakeRotations:
    def _pool(self, n=30):
        return [f"2026-05-{d:02d}" for d in range(1, n + 1)]

    def test_three_disjoint_folds_cover_exactly_their_days(self):
        rots = make_rotations(self._pool(30), cohort_seed=7)
        assert len(rots) == 3
        all_days = []
        for r in rots:
            assert len(r.train_days) == 6 and len(r.eval_days) == 4
            assert not (set(r.train_days) & set(r.eval_days))
            all_days += list(r.train_days) + list(r.eval_days)
        # 3×10 distinct days, all from the pool.
        assert len(all_days) == 30 and len(set(all_days)) == 30
        assert set(all_days) <= set(self._pool(30))

    def test_deterministic_in_cohort_seed(self):
        a = make_rotations(self._pool(30), cohort_seed=11)
        b = make_rotations(self._pool(30), cohort_seed=11)
        c = make_rotations(self._pool(30), cohort_seed=12)
        assert a == b
        assert a != c  # different seed → different (paired-but-distinct) split

    def test_no_sealed_leak(self):
        """Rotation days come ONLY from the non-sealed pool the caller
        passes — a sealed day that is not in the pool can never appear."""
        sealed = {f"2026-05-{d:02d}" for d in range(20, 30)}  # May 20-29
        pool = [f"2026-04-{d:02d}" for d in range(1, 31)]     # all April
        rots = make_rotations(pool, cohort_seed=3)
        used = {d for r in rots for d in (r.train_days + r.eval_days)}
        assert not (used & sealed)

    def test_raises_on_too_small_pool(self):
        with pytest.raises(ValueError):
            make_rotations(self._pool(20), cohort_seed=1)  # need 30


# ── Offspring ─────────────────────────────────────────────────────────────


class TestMakeOffspring:
    def test_structural_genes_frozen(self):
        rng = random.Random(1)
        # Force a transformer parent so we exercise the structural set fully.
        base = sample_fresh_blood_genes(rng, enabled_set=PHASE5_GENE_NAMES)
        parent = CohortGenes(**{**base.to_dict(),
                                "architecture": "transformer",
                                "transformer_depth": 3, "transformer_heads": 8,
                                "transformer_ctx_ticks": 256,
                                "hidden_size": 256})
        child = make_offspring(parent, random.Random(2),
                               enabled_set=PHASE5_GENE_NAMES)
        for name in STRUCTURAL_GENE_NAMES:
            assert getattr(child, name) == getattr(parent, name), name

    def test_continuous_recipe_perturbed_within_band(self):
        rng = random.Random(5)
        parent = sample_fresh_blood_genes(rng, enabled_set=frozenset())
        # Make a parent with a clearly mid-range LR so ±20% stays in range.
        parent = CohortGenes(**{**parent.to_dict(), "learning_rate": 3e-4,
                                "architecture": "lstm"})
        kids = [make_offspring(parent, random.Random(s)) for s in range(50)]
        lrs = [k.learning_rate for k in kids]
        # All within ±20% of the parent, and at least some actually moved.
        assert all(3e-4 * 0.8 <= lr <= 3e-4 * 1.2 + 1e-12 for lr in lrs)
        assert any(abs(lr - 3e-4) > 1e-9 for lr in lrs)

    def test_disabled_phase5_genes_stay_at_default(self):
        rng = random.Random(9)
        parent = sample_fresh_blood_genes(rng, enabled_set=frozenset())
        child = make_offspring(parent, random.Random(3), enabled_set=frozenset())
        from training_v2.cohort.genes import PHASE5_GENE_DEFAULTS
        for name in PHASE5_GENE_NAMES:
            assert getattr(child, name) == PHASE5_GENE_DEFAULTS[name], name

    def test_offspring_passes_assert_in_range_always(self):
        for s in range(100):
            rng = random.Random(s)
            parent = sample_fresh_blood_genes(rng, enabled_set=PHASE5_GENE_NAMES)
            make_offspring(parent, random.Random(s + 1),
                           enabled_set=PHASE5_GENE_NAMES)  # must not raise


# ── Population init ───────────────────────────────────────────────────────


class TestInitPopulation:
    def test_all_fresh_tier1_distinct_lineages(self):
        pop = init_pbt_population(random.Random(0), _cfg(), enabled_set=frozenset())
        assert len(pop) == 30
        assert all(s.tier == 1 for s in pop)
        assert all(s.role == "fresh" for s in pop)
        assert all(s.init_weights_path is None for s in pop)
        assert all(s.rotations_seen == frozenset({1}) for s in pop)
        assert len({s.lineage_id for s in pop}) == 30


# ── Promotion ladder ──────────────────────────────────────────────────────


def _pair(specs, base=0.0):
    """Attach fake results; score = -position so the FIRST specs win."""
    return [
        (s, _Res(model_id=f"m{base}_{i}", weights_path=f"w{base}_{i}.pt",
                 score=-(i)))
        for i, s in enumerate(specs)
    ]


class TestBreedLadder:
    def test_gen0_to_gen1_partition(self):
        cfg = _cfg()
        pop = init_pbt_population(random.Random(0), cfg, enabled_set=frozenset())
        nxt, frozen = breed_pbt(_pair(pop), random.Random(1), cfg,
                                score_fn=_score)
        assert len(nxt) == cfg.n_agents
        tiers = {1: [], 2: [], 3: []}
        for s in nxt:
            tiers[s.tier].append(s)
        # No R3 yet (no R2 trained in gen 0). R2 = 10, R1 = 20.
        assert len(tiers[3]) == 0
        assert len(tiers[2]) == cfg.r2_size
        assert len(tiers[1]) == cfg.n_agents - cfg.r2_size
        assert not frozen  # no R3 existed to freeze
        # R2 = promote_from_r1 elites + offspring.
        elites = [s for s in tiers[2] if s.role == "elite"]
        offspring = [s for s in tiers[2] if s.role == "offspring"]
        assert len(elites) == cfg.promote_from_r1
        assert len(offspring) == cfg.r2_size - cfg.promote_from_r1
        # Elites carry their own weights + climbed to rotation 2.
        for s in elites:
            assert s.init_weights_path is not None
            assert s.rotations_seen == frozenset({1, 2})
        # Offspring inherit a parent's weights (warm-start) + lineage.
        for s in offspring:
            assert s.init_weights_path is not None
            assert s.parent_model_id is not None
        # Fresh R1 are brand-new lineages.
        assert all(s.role == "fresh" and s.init_weights_path is None
                   for s in tiers[1])

    def test_winners_promote_not_losers(self):
        cfg = _cfg()
        pop = init_pbt_population(random.Random(0), cfg, enabled_set=frozenset())
        paired = _pair(pop)  # score = -i, so pop[0..4] are the top 5
        nxt, _ = breed_pbt(paired, random.Random(1), cfg, score_fn=_score)
        promoted_lineages = {s.lineage_id for s in nxt if s.role == "elite"}
        winner_lineages = {pop[i].lineage_id for i in range(cfg.promote_from_r1)}
        assert promoted_lineages == winner_lineages

    def test_reaches_steady_state_and_freezes_r3(self):
        cfg = _cfg()
        rng = random.Random(0)
        pop = init_pbt_population(rng, cfg, enabled_set=frozenset())
        froze_any = False
        specs = pop
        for gen in range(4):
            nxt, frozen = breed_pbt(_pair(specs, base=gen), random.Random(gen + 1),
                                    cfg, score_fn=_score)
            counts = {t: sum(1 for s in nxt if s.tier == t) for t in (1, 2, 3)}
            assert sum(counts.values()) == cfg.n_agents
            if gen >= 2:
                # Steady state: R1=14, R2=10, R3=6.
                assert counts == {1: 14, 2: cfg.r2_size, 3: cfg.r3_size}
                # By now an R3 existed to freeze.
                assert len(frozen) == cfg.freeze_top_r3
                froze_any = True
            specs = nxt
        assert froze_any

    def test_frozen_agents_do_not_compete(self):
        """A frozen champion in the input is ignored by the ladder."""
        cfg = _cfg()
        pop = init_pbt_population(random.Random(0), cfg, enabled_set=frozenset())
        from dataclasses import replace
        pop = [replace(pop[0], frozen=True)] + pop[1:]
        nxt, _ = breed_pbt(_pair(pop), random.Random(1), cfg, score_fn=_score)
        # The frozen agent's lineage is NOT promoted as an elite (it didn't
        # compete); total still n_agents (R1 absorbs).
        assert len(nxt) == cfg.n_agents
        assert pop[0].lineage_id not in {
            s.lineage_id for s in nxt if s.role == "elite"
        }
