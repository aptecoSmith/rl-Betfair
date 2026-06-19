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

__all__ = ["GauntletConfig", "run_gauntlet", "climb_to_frontier",
           "climb_cull_per_tranche", "seed_population"]


@dataclass
class GauntletConfig:
    n_recipes: int = 16            # initial fresh-blood population
    max_breed_rounds: int = 0      # breeding cycles after the first full climb
    seed_base: int = 0             # deterministic per-agent seeds
    enabled_set: frozenset = frozenset()
    seed_bands: dict | None = None  # band-seeded fresh blood (Tock era)
    breed: BreedConfig = None       # type: ignore[assignment]
    # "frontier" = full fair shot (climb all to N, breed only at the frontier).
    # "per_tranche" = cull-early successive-halving (cull after EACH tranche,
    # mutants catch up). Operator 2026-06-19 — the tick uses per_tranche.
    cull_mode: str = "frontier"

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


def _run_tranche_and_record(
    ledger: GauntletLedger,
    batch: list,
    K: int,
    *,
    split: DaySplit,
    exec_cfg: TrancheExecConfig,
    seed_base: int,
    executor,
    run_tranche_fn,
    score_result_fn,
) -> None:
    """Run one same-depth batch through tranche K; record results to the ledger.

    Shared by both climb strategies. Failed lineages (crash / no weights) are
    culled. Successes record their fc=0 composite (the selection score), locked,
    and naked at depth K. ``score_result_fn(AgentResult)->float`` is the
    lockstep composite (σ-penalty + fc-penalty); ``None`` ⇒ raw held-out locked.
    """
    agents = [_entry_to_agent(e, tranche_K=K, seed_base=seed_base) for e in batch]
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
        if score_result_fn is not None and r.result is not None:
            comp = float(score_result_fn(r.result))
        else:
            comp = r.validation_locked
        ledger.record_tranche(
            e.lineage_id, K, weights_path=r.weights_path,
            composite=comp, locked=r.validation_locked,
            naked=r.validation_naked, agent_id=r.agent_id)


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

    FULL-FAIR-SHOT strategy: each iteration picks the SHALLOWEST non-empty
    `needs-T(K)` queue and runs that whole same-depth cohort through tranche K
    (one `run_tranche` call = one uniform batch) — NO mid-climb culling. Returns
    the number of tranche-runs executed. Selection happens only at the frontier
    (the caller's breed rounds). Contrast :func:`climb_cull_per_tranche`.
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
        logger.info("gauntlet: climbing %d lineages through tranche %d/%d",
                    len(batch), K, n_tranches)
        _run_tranche_and_record(
            ledger, batch, K, split=split, exec_cfg=exec_cfg,
            seed_base=seed_base, executor=executor,
            run_tranche_fn=run_tranche_fn, score_result_fn=score_result_fn)
        runs += 1
    return runs


def climb_cull_per_tranche(
    ledger: GauntletLedger,
    split: DaySplit,
    exec_cfg: TrancheExecConfig,
    rng,
    *,
    seed_base: int = 0,
    executor=None,
    run_tranche_fn=run_tranche,
    score_result_fn=None,
    breed_cfg: BreedConfig = None,  # type: ignore[assignment]
    sigma_leg_fn=None,
) -> int:
    """Successive-halving gauntlet: cull AFTER EACH tranche; mutants catch up.

    The cull-early "tick" (operator decision 2026-06-19, supersedes full-fair-
    shot for the tick): a resident pool runs tranche K, the bottom
    ``keep_fraction`` is eliminated IMMEDIATELY (``breed_frontier`` at depth K),
    survivors are mutated to refill, and the new mutants re-climb T1..TK (recipe-
    pure catch-up, no culling) to rejoin the pool at depth K — then tranche K+1.

    So selection pressure applies at EVERY depth and only survivors+caught-up
    mutants ever pay for deeper training (duds die after T1). The catch-up cost
    grows with depth (a mutant born after TK must climb K tranches) — the
    accepted price of cull-early vs :func:`climb_to_frontier`'s full fair shot.

    Returns the number of tranche-runs executed (incl. catch-up runs). Driven by
    the ledger queues, so a resumed run continues from wherever the pool sits.
    """
    if breed_cfg is None:
        breed_cfg = BreedConfig()
    n_tranches = split.n_tranches
    runs = 0
    for K in range(1, n_tranches + 1):
        batch = ledger.needs(K)  # the resident pool at depth K-1
        if not batch:
            break
        logger.info("gauntlet[cull]: tranche %d/%d on %d resident lineages",
                    K, n_tranches, len(batch))
        _run_tranche_and_record(
            ledger, batch, K, split=split, exec_cfg=exec_cfg,
            seed_base=seed_base, executor=executor,
            run_tranche_fn=run_tranche_fn, score_result_fn=score_result_fn)
        runs += 1
        # Eliminate the bottom keep_fraction at depth K + emit mutant refills
        # at needs-T1 (breed_frontier culls at the current frontier == K).
        res = breed_frontier(ledger, rng, cfg=breed_cfg, sigma_leg_fn=sigma_leg_fn)
        if not res.bred:
            continue
        logger.info("gauntlet[cull]: tranche %d — kept %d, culled %d, +%d "
                    "mutants to catch up", K, len(res.survivors),
                    len(res.culled), len(res.new_lineage_ids))
        # CATCH-UP: re-climb the new mutants T1..TK (no culling) so they rejoin
        # the pool at depth K. needs(j) at this point holds ONLY the climbing
        # mutants (survivors sit at depth K, never at depth j-1<K).
        for j in range(1, K + 1):
            cu = ledger.needs(j)
            if not cu:
                continue
            logger.info("gauntlet[cull]: catch-up tranche %d/%d on %d mutants",
                        j, K, len(cu))
            _run_tranche_and_record(
                ledger, cu, j, split=split, exec_cfg=exec_cfg,
                seed_base=seed_base, executor=executor,
                run_tranche_fn=run_tranche_fn, score_result_fn=score_result_fn)
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

    # Cull-early "tick": cull after EACH tranche, mutants catch up. Selection
    # at every depth; no separate breed-round loop (breeding is interleaved).
    if cfg.cull_mode == "per_tranche":
        climb_cull_per_tranche(
            ledger, split, exec_cfg, rng, seed_base=cfg.seed_base,
            executor=executor, run_tranche_fn=run_tranche_fn,
            score_result_fn=score_result_fn, breed_cfg=cfg.breed,
            sigma_leg_fn=sigma_leg_fn)
        ledger.compact()
        return ledger

    # Full fair shot: first full climb, then breed→climb cycles.
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
