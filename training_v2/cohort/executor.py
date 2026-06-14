"""Tranche executor — the gauntlet-pipeline's pure execution primitive.

`plans/gauntlet-pipeline/` Phase 2. ``run_tranche`` trains a BATCH of agents
through ONE tranche and evals them on the fixed validation set at fc=0. It
contains **NO selection** — selection lives only in the breeder (Phase 4). This
is the load-bearing decoupling (`hard_constraints.md` §"Execution decoupling"):
a selection step hidden in the executor would re-couple the loop and recreate
the catch-up-in-one-generation problem the re-architecture exists to kill.

The gauntlet execution model (vs the old `--breeding lockstep`):

* Every agent in a batch is at the SAME depth ``tranche_K`` and trains on
  ``train_days_for_K`` — ONE tranche's days, the same set for all of them.
* A climber at K>1 warm-starts ITS OWN K-1 weights (``init_weights_path``); a
  fresh-blood / mutant recipe enters at K==1 with no weights and trains from
  scratch. There is **no catch-up replay** — a recipe climbs T1→T2→…
  one tranche at a time on its own weights, so every run is uniform-cost
  (``batch × one tranche``) regardless of gauntlet depth.
* **Recipe purity (the non-negotiable):** an agent NEVER warm-starts from a
  different recipe's weights. ``run_tranche`` asserts this two ways — a
  structural invariant (K==1 ⟺ no weights) and a ``.genehash`` sidecar written
  next to each saved checkpoint and re-checked on warm-start.

This module reuses the existing worker verbatim (predictors, shared static_obs
day cache, BC, input-norm all carry over): it builds the SAME ``train_one_agent``
spec dicts the runner's multiprocess path builds and dispatches them through
``train_cluster_multiproc``. The old lockstep path is untouched.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from training_v2.cohort.genes import CohortGenes
from training_v2.cohort.multiproc_worker import (
    assert_day_cache_fits,
    make_pool,
    model_store_paths,
    prebuild_static_obs_cache,
    train_cluster_multiproc,
)
from training_v2.cohort.worker import (
    DEFAULT_SCORER_DIR,
    AgentResult,
    train_one_agent,
)

logger = logging.getLogger(__name__)

__all__ = [
    "RecipeAgent",
    "TrancheExecConfig",
    "TrancheResult",
    "config_hash",
    "run_tranche",
    "RecipePurityError",
]

GENEHASH_SUFFIX = ".genehash"


class RecipePurityError(AssertionError):
    """Raised when an agent would warm-start from a DIFFERENT recipe's weights.

    The load-bearing invariant (`hard_constraints.md` §"Recipe purity"): a model
    is cooked under ONE gene config from T1, warm-starting only its OWN weights.
    A breach here means the ledger / orchestrator threaded the wrong
    ``init_weights_path`` into a batch — loud failure, never silent chimera.
    """


def config_hash(genes: CohortGenes) -> str:
    """Canonical recipe identity — a hash of the full gene config.

    Two agents with the same recipe collapse to one hash; any gene difference
    (including a structural gene) yields a different hash. Floats are rounded to
    9 s.f. so round-trip noise doesn't split a recipe. Same scheme as
    ``tools/gene_register._config_hash`` so the register and the executor agree
    on what "one recipe" means.
    """
    d = genes.to_dict()
    items = []
    for k in sorted(d):
        v = d[k]
        if isinstance(v, float):
            v = round(v, 9)
        items.append((k, v))
    return json.dumps(items, sort_keys=True, default=str)


@dataclass(frozen=True)
class RecipeAgent:
    """One agent's identity carried INTO ``run_tranche``.

    ``init_weights_path`` is the agent's OWN K-1 checkpoint (None at K==1). The
    ledger/orchestrator owns lineage bookkeeping; the executor only verifies
    recipe purity at the boundary.
    """

    agent_id: str
    genes: CohortGenes
    lineage_id: str
    origin: str  # "fresh" | "mutant" | "survivor"/"climber"
    init_weights_path: str | None = None
    parent_model_id: str | None = None
    seed: int = 0


@dataclass
class TrancheExecConfig:
    """Cohort-wide context shared by every agent in a tranche batch.

    These are the knobs the runner threads cohort-wide (NOT per-agent). Per-agent
    variation lives on ``RecipeAgent`` (genes, weights, seed) + the tranche days.
    Defaults reproduce the predictors-ON fast path
    (``--parallel-agents 16 --device cpu``, never ``--batched``).
    """

    data_dir: Path
    output_dir: Path
    model_store: object  # registry.model_store.ModelStore
    # Predictor support (ALWAYS ON per hard_constraints). The worker rebuilds
    # the bundle from manifests, so multiprocess needs the manifest paths.
    predictor_bundle: object | None = None
    predictor_manifests: tuple | None = None
    use_race_outcome_predictor: bool = True
    use_direction_predictor: bool = False
    # Concurrency / device.
    parallel_agents: int = 16
    device: str = "cpu"
    big_model_threads: int = 1
    gpu_policy_lane: bool = False
    gpu_lane_max_concurrent: int = 2
    # Reward / gene wiring (carried verbatim from the cohort launch).
    enabled_set: frozenset = frozenset()
    reward_overrides: dict | None = None
    strategy_mode: str | None = None
    composite_score_mode: str = "locked_weighted"
    argmax_eval: bool = False
    per_transition_credit: bool = False
    # BC overrides (cohort-wide).
    bc_pretrain_steps_override: int | None = None
    bc_learning_rate_override: float | None = None
    bc_target_entropy_warmup_eps_override: int | None = None
    bc_include_negative_samples: bool = False
    bc_positive_weight: float = 1.0
    bc_include_close_hold_samples: bool = False
    arb_spread_target_lock_pct_override: float | None = None
    # Cohort-wide gate flags (per-agent gene resolution happens in the worker).
    predictor_p_win_back_threshold: float = 0.0
    predictor_p_win_back_max_threshold: float = 1.0
    predictor_p_win_lay_threshold: float = 1.0
    direction_gate_enabled: bool = False
    mature_prob_open_threshold: float = 0.0
    race_confidence_threshold: float = 0.0
    lay_price_max: float = 0.0
    frozen_direction_head_path: Path | None = None
    scorer_dir: Path = DEFAULT_SCORER_DIR
    n_agents_hint: int = 0  # for progress fields only

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)


@dataclass
class TrancheResult:
    """One agent's outcome from a tranche run. NO selection / ranking here."""

    agent_id: str
    lineage_id: str
    tranche_K: int
    weights_path: str
    result: AgentResult
    validation_locked: float
    validation_naked: float
    validation_day_pnl: float
    composite_score: float
    error: str | None = None
    bet_logs_dir: str | None = None


# ── Recipe-purity guard (sidecar + structural) ────────────────────────────


def _genehash_path(weights_path: str | Path) -> Path:
    return Path(str(weights_path) + GENEHASH_SUFFIX)


def _write_genehash_sidecar(weights_path: str | Path, genes: CohortGenes,
                            *, lineage_id: str, tranche_K: int) -> None:
    """Stamp the recipe identity next to a saved checkpoint.

    Read back on warm-start so an agent can never inherit a DIFFERENT recipe's
    weights without a loud failure (recipe purity). Best-effort: a write failure
    logs but does not kill the run (the structural K==1 invariant still holds).
    """
    p = _genehash_path(weights_path)
    try:
        p.write_text(json.dumps({
            "config_hash": config_hash(genes),
            "lineage_id": lineage_id,
            "tranche_K": int(tranche_K),
        }), encoding="utf-8")
    except Exception:
        logger.warning("could not write genehash sidecar %s", p, exc_info=True)


def _assert_recipe_purity(agent: RecipeAgent, tranche_K: int) -> None:
    """Verify ``agent`` may train at ``tranche_K`` under recipe purity.

    1. Structural invariant: K==1 ⟺ no warm-start weights; K>1 ⟺ has weights.
    2. Sidecar check (when the K-1 checkpoint carries a ``.genehash``): the
       inherited weights' config hash + lineage MUST match this agent's. A
       mismatch means a cross-recipe warm-start was threaded in — the exact
       chimera `purpose.md` rejects on principle.
    """
    has_weights = bool(agent.init_weights_path)
    if tranche_K <= 1 and has_weights:
        raise RecipePurityError(
            f"agent {agent.agent_id} at tranche 1 must train from scratch, but "
            f"init_weights_path={agent.init_weights_path!r} was supplied "
            f"(a T1 recipe never inherits weights).")
    if tranche_K > 1 and not has_weights:
        raise RecipePurityError(
            f"agent {agent.agent_id} at tranche {tranche_K} must warm-start its "
            f"OWN K-1 weights, but init_weights_path is None.")
    if not has_weights:
        return
    side = _genehash_path(agent.init_weights_path)
    if not side.exists():
        # No sidecar (e.g. weights produced before the executor existed) — the
        # ledger's lineage bookkeeping is the only guarantee; trust it but warn.
        logger.warning(
            "recipe-purity: no genehash sidecar for warm-start weights %s "
            "(agent %s) — relying on ledger lineage bookkeeping.",
            agent.init_weights_path, agent.agent_id)
        return
    try:
        meta = json.loads(side.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RecipePurityError(
            f"agent {agent.agent_id}: unreadable genehash sidecar {side}: {exc}")
    want = config_hash(agent.genes)
    if meta.get("config_hash") != want:
        raise RecipePurityError(
            f"agent {agent.agent_id}: warm-start weights {agent.init_weights_path} "
            f"were cooked under a DIFFERENT recipe (config_hash mismatch) — this "
            f"is a cross-recipe chimera, rejected by recipe purity.")
    if meta.get("lineage_id") != agent.lineage_id:
        raise RecipePurityError(
            f"agent {agent.agent_id}: warm-start weights belong to lineage "
            f"{meta.get('lineage_id')!r}, not {agent.lineage_id!r}.")


# ── Spec construction (mirrors runner.py's multiprocess path) ──────────────


def _threads_for_hidden(hidden_size: int, big_model_threads: int) -> int:
    """Big LSTMs get N threads (their wide matmuls use cores small agents free);
    the rest stay 1. Mirror of the runner's ``_threads_for_hidden`` (default
    N==1 ⇒ single-thread ⇒ bit-identical)."""
    if int(big_model_threads) <= 1:
        return 1
    return int(big_model_threads) if int(hidden_size) >= 512 else 1


def _build_spec(agent: RecipeAgent, *, tranche_K: int,
                train_days: list[str], validation_days: list[str],
                cfg: TrancheExecConfig, static_obs_paths_by_lean: dict | None,
                store_paths: dict, mp_predictor_manifests: tuple | None,
                idx: int, n_agents: int) -> dict:
    """Build ONE ``train_one_agent`` spec dict — same shape as runner.py.

    Kept field-for-field aligned with the runner's multiprocess spec so the
    executor reuses the identical worker plumbing (drift-guarded by
    ``tests/test_v2_executor.py::test_spec_keys_match_train_one_agent``).
    """
    genes = agent.genes
    lean = bool(getattr(genes, "predictor_lean_obs", False))
    _sobs_agent = None
    if static_obs_paths_by_lean is not None:
        variant = static_obs_paths_by_lean.get(lean, {})
        agent_days = set(train_days) | set(validation_days)
        _sobs_agent = {d: variant[d] for d in agent_days if d in variant}
    return dict(
        agent_id=agent.agent_id,
        genes=genes,
        days_to_train=list(train_days),
        eval_days=list(validation_days),
        init_weights_path=agent.init_weights_path,
        data_dir=cfg.data_dir,
        device=cfg.device,
        seed=int(agent.seed),
        model_store=None,  # worker rebuilds from paths
        generation=int(tranche_K),
        parent_a_id=agent.parent_model_id,
        parent_b_id=None,
        event_emitter=None,
        agent_idx=int(idx),
        n_agents=int(n_agents),
        reward_overrides=cfg.reward_overrides,
        enabled_set=cfg.enabled_set,
        argmax_eval=cfg.argmax_eval,
        per_transition_credit=cfg.per_transition_credit,
        bc_pretrain_steps_override=cfg.bc_pretrain_steps_override,
        bc_learning_rate_override=cfg.bc_learning_rate_override,
        bc_target_entropy_warmup_eps_override=(
            cfg.bc_target_entropy_warmup_eps_override),
        bc_include_negative_samples=cfg.bc_include_negative_samples,
        bc_positive_weight=cfg.bc_positive_weight,
        bc_include_close_hold_samples=cfg.bc_include_close_hold_samples,
        arb_spread_target_lock_pct_override=cfg.arb_spread_target_lock_pct_override,
        predictor_bundle=None,  # worker reloads from manifests
        strategy_mode=cfg.strategy_mode,
        use_race_outcome_predictor=cfg.use_race_outcome_predictor,
        predictor_lean_obs=lean,
        use_direction_predictor=bool(
            getattr(genes, "use_direction_predictor", False)
        ) or bool(cfg.use_direction_predictor),
        predictor_p_win_back_threshold=cfg.predictor_p_win_back_threshold,
        predictor_p_win_back_max_threshold=cfg.predictor_p_win_back_max_threshold,
        predictor_p_win_lay_threshold=cfg.predictor_p_win_lay_threshold,
        direction_gate_enabled=bool(genes.direction_gate_enabled)
        or bool(cfg.direction_gate_enabled),
        mature_prob_open_threshold=cfg.mature_prob_open_threshold,
        race_confidence_threshold=cfg.race_confidence_threshold,
        lay_price_max=cfg.lay_price_max,
        composite_score_mode=cfg.composite_score_mode,
        feature_cache=None,
        frozen_direction_head_path=cfg.frozen_direction_head_path,
        gpu_policy_lane=bool(cfg.gpu_policy_lane),
        gpu_lane_max_concurrent=int(cfg.gpu_lane_max_concurrent),
        _feature_cache_day_paths=None,
        _static_obs_day_paths=_sobs_agent,
        _model_store_paths=store_paths,
        _predictor_manifests=mp_predictor_manifests,
        _num_threads=_threads_for_hidden(genes.hidden_size, cfg.big_model_threads),
    )


# ── The primitive ──────────────────────────────────────────────────────────


def run_tranche(
    agents: list[RecipeAgent],
    *,
    tranche_K: int,
    train_days_for_K: list[str],
    validation_days: list[str],
    cfg: TrancheExecConfig,
    executor=None,
    train_one_agent_fn=train_one_agent,
    train_cluster_fn=train_cluster_multiproc,
) -> list[TrancheResult]:
    """Train ``agents`` through tranche ``tranche_K`` and eval at fc=0. No culling.

    Parameters
    ----------
    agents:
        The batch. Each is at depth ``tranche_K``; survivors/climbers carry
        their OWN K-1 ``init_weights_path``, fresh/mutants (K==1) carry None.
    train_days_for_K, validation_days:
        ONE tranche's train days (same for every agent — fixed tranche) and the
        FIXED validation set (held-out selection regime, fc=0). The caller
        guarantees ``validation ∩ train == ∅``.
    cfg:
        Cohort-wide context (predictors, device, reward wiring, …).
    executor:
        Optional warm ``ProcessPoolExecutor`` (reused across tranches). When
        None and ``parallel_agents>0`` a fresh pool is made for this call.

    Returns one :class:`TrancheResult` per input agent, in order. Recipe-purity
    is asserted BEFORE any training; a violation raises (never trains a chimera).
    """
    if not agents:
        return []
    if int(tranche_K) < 1:
        raise ValueError(f"tranche_K must be >= 1, got {tranche_K}")
    if not train_days_for_K:
        raise ValueError("train_days_for_K must be non-empty")
    if not validation_days:
        raise ValueError("validation_days must be non-empty (fc=0 select set)")
    leak = set(train_days_for_K) & set(validation_days)
    if leak:
        raise ValueError(
            f"validation ∩ train != empty (leakage): {sorted(leak)}")

    # Recipe-purity gate — BEFORE training (HC: never train a chimera).
    for a in agents:
        _assert_recipe_purity(a, int(tranche_K))

    n_agents = int(cfg.n_agents_hint or len(agents))

    # Shared static_obs day cache (predictors-ON path). Bake one variant per
    # obs representation the batch uses; each is a shared memmap (one physical
    # copy across workers). Mirrors runner.py.
    use_static_obs_cache = (
        cfg.predictor_bundle is not None
        and bool(cfg.use_race_outcome_predictor)
    )
    gen_days = list(dict.fromkeys(list(train_days_for_K) + list(validation_days)))
    static_obs_paths_by_lean: dict | None = None
    if use_static_obs_cache:
        static_obs_paths_by_lean = {}
        leans = sorted({bool(getattr(a.genes, "predictor_lean_obs", False))
                        for a in agents})
        for _lean in leans:
            static_obs_paths_by_lean[_lean] = prebuild_static_obs_cache(
                gen_days, data_dir=cfg.data_dir,
                cache_dir=cfg.output_dir / (
                    "mp_static_obs_cache_lean" if _lean
                    else "mp_static_obs_cache_full"),
                predictor_bundle=cfg.predictor_bundle,
                use_race_outcome_predictor=bool(cfg.use_race_outcome_predictor),
                use_direction_predictor=bool(cfg.use_direction_predictor),
                predictor_lean_obs=bool(_lean),
                scorer_dir=cfg.scorer_dir,
            )
        # Memory-budget guard — shared path (one copy/day across workers).
        _dc_bytes = [
            Path(npy).stat().st_size
            for variant in static_obs_paths_by_lean.values()
            for (npy, _side) in variant.values()
        ]
        if _dc_bytes:
            assert_day_cache_fits(
                day_cache_bytes=_dc_bytes,
                n_workers=int(cfg.parallel_agents or 1), shared=True)

    mp_predictor_manifests = (
        tuple(cfg.predictor_manifests) if cfg.predictor_bundle is not None
        else None
    )
    if (cfg.parallel_agents and int(cfg.parallel_agents) > 0
            and cfg.predictor_bundle is not None and not cfg.predictor_manifests):
        raise ValueError(
            "multiprocess run_tranche with a predictor bundle requires "
            "cfg.predictor_manifests (the worker rebuilds from manifest paths).")

    store_paths = model_store_paths(cfg.model_store) if cfg.model_store else None

    specs = [
        _build_spec(
            a, tranche_K=int(tranche_K), train_days=train_days_for_K,
            validation_days=validation_days, cfg=cfg,
            static_obs_paths_by_lean=static_obs_paths_by_lean,
            store_paths=store_paths, mp_predictor_manifests=mp_predictor_manifests,
            idx=i, n_agents=n_agents)
        for i, a in enumerate(agents)
    ]

    # ── Execute. NO selection of any kind in this block. ──
    if cfg.parallel_agents and int(cfg.parallel_agents) > 0:
        own_pool = None
        if executor is None:
            own_pool = executor = make_pool(int(cfg.parallel_agents))
        try:
            raw = train_cluster_fn(
                specs, n_workers=int(cfg.parallel_agents), executor=executor)
        finally:
            if own_pool is not None:
                own_pool.shutdown()
    else:
        # Sequential fallback (parallel_agents=0) — also the test path. Strip
        # the worker-only spec keys and call train_one_agent directly.
        raw = []
        for spec in specs:
            s = dict(spec)
            for k in ("_feature_cache_day_paths", "_static_obs_day_paths",
                      "_model_store_paths", "_predictor_manifests",
                      "_num_threads", "gpu_lane_max_concurrent"):
                s.pop(k, None)
            if cfg.model_store is not None:
                s["model_store"] = cfg.model_store
            if cfg.predictor_bundle is not None:
                s["predictor_bundle"] = cfg.predictor_bundle
            raw.append(train_one_agent_fn(**s))

    out: list[TrancheResult] = []
    for a, res in zip(agents, raw):
        if res is None:
            out.append(TrancheResult(
                agent_id=a.agent_id, lineage_id=a.lineage_id,
                tranche_K=int(tranche_K), weights_path="", result=res,
                validation_locked=float("nan"), validation_naked=float("nan"),
                validation_day_pnl=float("nan"), composite_score=float("nan"),
                error="worker returned None"))
            continue
        wp = getattr(res, "weights_path", "") or ""
        if wp:
            _write_genehash_sidecar(
                wp, a.genes, lineage_id=a.lineage_id, tranche_K=int(tranche_K))
        ev = res.eval
        out.append(TrancheResult(
            agent_id=a.agent_id, lineage_id=a.lineage_id,
            tranche_K=int(tranche_K), weights_path=wp, result=res,
            validation_locked=float(getattr(ev, "locked_pnl", float("nan"))),
            validation_naked=float(getattr(ev, "naked_pnl", float("nan"))),
            validation_day_pnl=float(getattr(ev, "day_pnl", float("nan"))),
            composite_score=float(
                getattr(res, "composite_score", float("nan"))
                if hasattr(res, "composite_score") else float("nan")),
        ))
    return out
