"""Gauntlet orchestrator (Phase 5) — drive executor + ledger + breeder.

`plans/gauntlet-pipeline/`. Ties the three Phase-2/3/4 components into the
decoupled pipeline:

    seed needs-T1  ──>  climb_to_frontier (uniform run_tranche batches)  ──>
    breed_frontier (cull + emit new T1 recipes)  ──>  repeat

Every executor call is uniform-cost (`batch × one tranche`). A recipe climbs
T1→T2→…→TN on its OWN weights (recipe purity); the breeder removes at the
frontier and adds fresh recipes at T1 that then climb. The **ledger is the
state** — `run_gauntlet` is resumable (re-load the ledger and continue) and a
second machine could read the same ledger to grab a batch.

The orchestrator owns NO selection (breeder) and NO training (executor); it only
schedules: pick the shallowest non-empty `needs-T(K)` queue, run it, record
results, and breed when the frontier is quorum-full. This keeps the
catch-up-in-one-generation coupling permanently out of the loop.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from training_v2.cohort.breeder import BreedConfig, breed_frontier, genes_from_dict
from training_v2.cohort.executor import (
    RecipeAgent,
    TrancheExecConfig,
    config_hash,
    run_tranche,
)
from training_v2.cohort.genes import sample_fresh_blood_genes
from training_v2.cohort.ledger import DaySplit, GauntletLedger

logger = logging.getLogger(__name__)

__all__ = ["GauntletConfig", "run_gauntlet", "climb_to_frontier", "seed_population"]


@dataclass
class GauntletConfig:
    n_recipes: int = 16            # initial fresh-blood population
    max_breed_rounds: int = 0      # breeding cycles after the first full climb
    seed_base: int = 0             # deterministic per-agent seeds
    enabled_set: frozenset = frozenset()
    seed_bands: dict | None = None  # band-seeded fresh blood (Tock era)
    breed: BreedConfig = None       # type: ignore[assignment]

    def __post_init__(self):
        if self.breed is None:
            self.breed = BreedConfig(enabled_set=self.enabled_set,
                                     seed_bands=self.seed_bands)


def _derive_seed(seed_base: int, lineage_id: str, tranche_K: int) -> int:
    """Deterministic per-(lineage, tranche) seed.

    Uses a stable hash (NOT Python's per-process-salted ``hash``) so a resumed
    run reproduces the same per-agent seeds.
    """
    import hashlib

    digest = hashlib.md5(
        f"{lineage_id}:{int(tranche_K)}".encode()).hexdigest()
    h = int(digest[:8], 16) & 0x7FFFFFFF
    return (int(seed_base) * 1_000_003 + h) & 0x7FFFFFFF


def seed_population(ledger: GauntletLedger, cfg: GauntletConfig, rng) -> int:
    """Seed ``n_recipes`` fresh-blood lineages into needs-T1 (once, if empty)."""
    if ledger.all_entries():
        return 0
    n = 0
    for _ in range(int(cfg.n_recipes)):
        g = sample_fresh_blood_genes(
            rng, enabled_set=cfg.enabled_set, seed_bands=cfg.seed_bands)
        ledger.add_recipe(g, origin="fresh", config_hash=config_hash(g))
        n += 1
    logger.info("gauntlet: seeded %d fresh-blood recipes into needs-T1", n)
    return n


def _entry_to_agent(entry, *, tranche_K: int, seed_base: int) -> RecipeAgent:
    """Build the executor input for a lineage's next tranche.

    A NEW ``agent_id`` (registry model_id) per tranche run; warm-start weights =
    the lineage's OWN K-1 checkpoint (None at K==1 — recipe purity).
    """
    return RecipeAgent(
        agent_id=uuid.uuid4().hex,
        genes=genes_from_dict(entry.genes),
        lineage_id=entry.lineage_id,
        origin=entry.origin,
        init_weights_path=(entry.weights_path or None) if tranche_K > 1 else None,
        parent_model_id=entry.parent_model_id,
        seed=_derive_seed(seed_base, entry.lineage_id, tranche_K),
    )


def climb_to_frontier(
    ledger: GauntletLedger,
    split: DaySplit,
    exec_cfg: TrancheExecConfig,
    *,
    seed_base: int = 0,
    executor=None,
    run_tranche_fn=run_tranche,
    score_result_fn=None,
) -> int:
    """Advance every active lineage to depth ``n_tranches`` via uniform runs.

    Each iteration picks the SHALLOWEST non-empty `needs-T(K)` queue and runs
    that whole same-depth cohort through tranche K (one `run_tranche` call =
    one uniform batch). Returns the number of tranche-runs executed.

    ``score_result_fn(AgentResult) -> float`` is the breeder's selection score
    recorded per tranche. Passed by the runner as
    ``_composite_score(eval, maturation_bonus_weight, composite_score_mode)`` so
    in-loop selection matches lockstep's discipline EXACTLY — the composite folds
    in the ``naked_std`` σ-penalty (locked_weighted) AND the force-close-rate
    penalty (global weight). ``None`` falls back to raw held-out locked.
    """
    n_tranches = split.n_tranches
    runs = 0
    while True:
        pending = [e for e in ledger.all_entries()
                   if e.status == "active" and e.tranches_completed < n_tranches]
        if not pending:
            break
        K = min(e.needs_tranche() for e in pending)
        batch = [e for e in pending if e.needs_tranche() == K]
        agents = [_entry_to_agent(e, tranche_K=K, seed_base=seed_base)
                  for e in batch]
        logger.info("gauntlet: climbing %d lineages through tranche %d/%d",
                    len(agents), K, n_tranches)
        results = run_tranche_fn(
            agents, tranche_K=K,
            train_days_for_K=split.train_days_for(K),
            validation_days=split.validation_days,
            cfg=exec_cfg, executor=executor)
        for e, r in zip(batch, results):
            if r is None or r.error or not r.weights_path:
                logger.warning("gauntlet: lineage %s failed tranche %d (%s) — "
                               "culling", e.lineage_id, K,
                               getattr(r, "error", "no result"))
                ledger.set_status(e.lineage_id, "culled")
                continue
            # Selection score = the lockstep composite (fc-penalty + naked_std
            # σ-penalty) when the runner supplies score_result_fn; else raw
            # held-out locked. ``locked``/``naked`` stay the raw structural
            # values regardless (for diagnostics / the post-run holdout board).
            if score_result_fn is not None and r.result is not None:
                comp = float(score_result_fn(r.result))
            else:
                comp = r.validation_locked
            ledger.record_tranche(
                e.lineage_id, K, weights_path=r.weights_path,
                composite=comp, locked=r.validation_locked,
                naked=r.validation_naked, agent_id=r.agent_id)
        runs += 1
    return runs


def run_gauntlet(
    *,
    split: DaySplit,
    exec_cfg: TrancheExecConfig,
    cfg: GauntletConfig,
    ledger_path,
    rng,
    executor=None,
    run_tranche_fn=run_tranche,
    sigma_leg_fn=None,
    score_result_fn=None,
) -> GauntletLedger:
    """Run the full gauntlet pipeline; return the (resumable) ledger.

    Resumes from ``ledger_path`` if it exists; otherwise seeds a fresh
    population. After the first full climb to depth N, runs up to
    ``cfg.max_breed_rounds`` breed→climb cycles (each emits new T1 recipes that
    climb the whole gauntlet — full fair shot).
    """
    from training_v2.cohort.ledger import assert_day_split_disjoint
    assert_day_split_disjoint(split)

    ledger = GauntletLedger.load(ledger_path)
    if ledger.split is None:
        ledger.set_split(split)
    seed_population(ledger, cfg, rng)

    # First full climb.
    climb_to_frontier(ledger, split, exec_cfg, seed_base=cfg.seed_base,
                      executor=executor, run_tranche_fn=run_tranche_fn,
                      score_result_fn=score_result_fn)

    # Breed→climb cycles.
    for round_i in range(int(cfg.max_breed_rounds)):
        res = breed_frontier(ledger, rng, cfg=cfg.breed, sigma_leg_fn=sigma_leg_fn)
        if not res.bred:
            logger.info("gauntlet: breed round %d skipped (frontier below "
                        "quorum) — stopping.", round_i + 1)
            break
        logger.info("gauntlet: breed round %d — kept %d, culled %d, +%d fresh",
                    round_i + 1, len(res.survivors), len(res.culled),
                    len(res.new_lineage_ids))
        climb_to_frontier(ledger, split, exec_cfg, seed_base=cfg.seed_base,
                          executor=executor, run_tranche_fn=run_tranche_fn,
                          score_result_fn=score_result_fn)

    ledger.compact()
    return ledger
