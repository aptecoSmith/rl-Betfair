"""Gauntlet breeder — separate selection + mutation (Phase 4).

`plans/gauntlet-pipeline/`. The ONLY place selection lives (the executor has
none — `hard_constraints.md` §"Execution decoupling"). The breeder reads the
ledger's **frontier pool** (lineages at the current deepest depth — a same-depth,
fully-comparable set), truncation-selects on the fc=0 validation composite,
keeps the top fraction (they stay active and advance when the next tranche
appends), culls the rest, and emits replacement recipes — **mutants**
(`make_offspring` of survivors) + **fresh blood** (`sample_fresh_blood_genes`,
later register-driven) — into `needs-T1`, where they climb the gauntlet from
scratch under recipe purity.

**Full fair shot (`hard_constraints.md`):** only the frontier is eligible for
culling. A lineage still climbing at a shallower depth is NEVER touched — it is
judged only once it has completed the same gauntlet as the incumbents, in a
same-depth pool. This is what `frontier()` (same-depth-only) guarantees.

The hard ceiling σ_naked_leg ≤ £30 (`holdout-selection.md`) is applied as a
pluggable pre-filter (`sigma_leg_fn`) — the per-leg σ comes from bet logs, wired
by the orchestrator in Phase 5; absent it, ranking is on the composite alone.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

from training_v2.cohort.executor import config_hash
from training_v2.cohort.genes import CohortGenes, sample_fresh_blood_genes
from training_v2.cohort.ledger import GauntletLedger, LedgerEntry
from training_v2.cohort.pbt import make_offspring

logger = logging.getLogger(__name__)

__all__ = ["BreedConfig", "BreedResult", "breed_frontier", "genes_from_dict"]


def genes_from_dict(d: dict) -> CohortGenes:
    """Reconstruct ``CohortGenes`` from a stored gene dict, dropping any keys
    that are not current dataclass fields (legacy genes from old eras)."""
    fields = set(CohortGenes.__dataclass_fields__)
    return CohortGenes(**{k: v for k, v in d.items() if k in fields})


@dataclass
class BreedConfig:
    keep_fraction: float = 0.5       # truncation survivor fraction
    perturb_frac: float = 0.20       # make_offspring perturbation strength
    mutant_fraction: float = 1.0     # P(replacement is a mutant); rest = fresh
    enabled_set: frozenset = frozenset()
    seed_bands: dict | None = None   # band-seeded fresh blood (Tock era)
    min_quorum: int = 2              # don't breed until frontier >= this many
    sigma_leg_ceiling: float = 30.0  # hard ceiling (when sigma_leg_fn supplied)


@dataclass
class BreedResult:
    depth: int
    survivors: list[str] = field(default_factory=list)      # kept lineage_ids
    culled: list[str] = field(default_factory=list)         # culled lineage_ids
    new_lineage_ids: list[str] = field(default_factory=list)  # added to needs-T1
    bred: bool = True

    @classmethod
    def skipped(cls, depth: int) -> "BreedResult":
        return cls(depth=depth, bred=False)


def _default_score(depth: int):
    def score(e: LedgerEntry) -> float:
        return float(e.validation_score.get(str(depth), float("-inf")))
    return score


def breed_frontier(
    ledger: GauntletLedger,
    rng: random.Random,
    *,
    cfg: BreedConfig,
    score_fn=None,
    sigma_leg_fn=None,
) -> BreedResult:
    """Run one breeding pass on the current frontier. Returns a BreedResult.

    No-op (``bred=False``) when the frontier is below quorum or depth 0. The
    ledger is mutated in place: culled lineages → status "culled"; replacements
    appended at depth 0 (needs-T1). Deterministic given ``rng``.
    """
    depth = ledger.frontier_depth()
    if depth <= 0:
        return BreedResult.skipped(depth)
    frontier = ledger.frontier(depth)
    if len(frontier) < max(2, int(cfg.min_quorum)):
        logger.info("breeder: frontier@%d has %d < quorum %d — skipping",
                    depth, len(frontier), max(2, int(cfg.min_quorum)))
        return BreedResult.skipped(depth)

    score = score_fn or _default_score(depth)

    # Hard ceiling on per-leg naked σ (deployment-critical). Filter BEFORE
    # ranking; if it would empty the pool, fall back to unfiltered (never
    # deadlock the gauntlet on a too-tight ceiling for one shallow round).
    eligible = frontier
    if sigma_leg_fn is not None and cfg.sigma_leg_ceiling:
        kept = []
        for e in frontier:
            s = sigma_leg_fn(e)
            if s is None or s <= float(cfg.sigma_leg_ceiling):
                kept.append(e)
        if kept:
            eligible = kept
        else:
            logger.warning("breeder: σ_leg ceiling %.0f culled the whole "
                           "frontier@%d — ranking unfiltered this round.",
                           cfg.sigma_leg_ceiling, depth)

    ranked = sorted(eligible, key=score, reverse=True)
    n_keep = max(1, round(len(frontier) * float(cfg.keep_fraction)))
    n_keep = min(n_keep, len(ranked))
    survivors = ranked[:n_keep]
    survivor_ids = {e.lineage_id for e in survivors}
    culled = [e for e in frontier if e.lineage_id not in survivor_ids]

    for e in culled:
        ledger.set_status(e.lineage_id, "culled")

    # Emit one replacement per culled slot → needs-T1 (climbs from scratch).
    new_ids: list[str] = []
    for _ in range(len(culled)):
        if survivors and rng.random() < float(cfg.mutant_fraction):
            parent = rng.choice(survivors)
            child = make_offspring(
                genes_from_dict(parent.genes), rng,
                enabled_set=cfg.enabled_set, frac=float(cfg.perturb_frac))
            entry = ledger.add_recipe(
                child, origin="mutant", config_hash=config_hash(child),
                parent_model_id=parent.last_agent_id or None,
                parent_lineage_id=parent.lineage_id)
        else:
            child = sample_fresh_blood_genes(
                rng, enabled_set=cfg.enabled_set, seed_bands=cfg.seed_bands)
            entry = ledger.add_recipe(
                child, origin="fresh", config_hash=config_hash(child))
        new_ids.append(entry.lineage_id)

    logger.info("breeder: frontier@%d — kept %d, culled %d, emitted %d to "
                "needs-T1", depth, len(survivors), len(culled), len(new_ids))
    return BreedResult(
        depth=depth, survivors=[e.lineage_id for e in survivors],
        culled=[e.lineage_id for e in culled], new_lineage_ids=new_ids)
