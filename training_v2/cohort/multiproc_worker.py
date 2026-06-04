"""Multiprocess cluster training — training-speedup-v2 R5.

The big speed lever for this CPU-bound, 20-core box: run the cluster's
agents as PARALLEL PROCESSES, each a solo ``train_one_agent`` on the
optimized single-agent path. Measured ~7.7× parallel speedup (8 agents),
~5.8× cluster-day — beating the GPU-batched path (R1+R2, 2.55×), because
it parallelises the WHOLE per-agent rollout (env + forward + update)
across the cores, not just the forward.

**Bit-identical by construction:** each worker runs the canonical solo
``train_one_agent`` (the golden path) at its own seed. Parallel vs
sequential produces identical ``AgentResult``s (deterministic per seed) —
gated by ``tests/test_v2_multiproc_cluster.py``.

Single-threading per worker (``MKL_NUM_THREADS=1`` + ``torch.set_num_threads(1)``)
is load-bearing: without it, N workers each spawn BLAS thread-pools and
oversubscribe the cores → no parallel gain. Set at module import (before
any torch/numpy import in the spawned worker) so the env var takes effect.

model_store is the caller's responsibility: pass ``model_store=None`` in
the specs (registry writes are deferred to the parent, which is concurrency-
safe — SQLite-WAL concurrent writes from N processes are avoided).
"""
from __future__ import annotations

import os

# Per-process BLAS/OMP cap. Set BEFORE torch/numpy import so the spawned
# worker (which re-imports this module) picks it up. Default "1" == single
# thread per worker — the load-bearing no-oversubscription invariant AND the
# bit-identity guarantee. When the runner enables size-aware threading
# (--big-model-threads N) it sets SES_MP_MAX_THREADS=N in the PARENT before
# spawning the pool, so each worker inherits it and raises the cap to N; the
# actual per-AGENT thread count is then set at runtime by
# torch.set_num_threads() from each spec's ``_num_threads`` (big models get N,
# the rest stay 1, so their large matmuls spread onto the cores small agents
# free as they finish). Unconditional (not setdefault) so an inherited
# multi-thread value can't leak in when threading is OFF.
_mp_max_threads = os.environ.get("SES_MP_MAX_THREADS", "1")
for _v in (
    "MKL_NUM_THREADS", "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_v] = _mp_max_threads

import logging
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "train_cluster_multiproc",
    "default_worker_count",
    "prebuild_feature_cache",
    "save_shared_cache",
    "save_shared_cache_per_day",
    "prebuild_static_obs_cache",
    "model_store_paths",
    "make_pool",
]

# ── Per-worker engineered-feature day cache (warm persistent pool) ──────────
# When the cohort runner reuses ONE ProcessPoolExecutor across generations
# (instead of re-spawning per generation), each worker process stays alive
# between tasks. This module-global dict therefore persists across tasks in
# that worker, so a day engineered + deserialised on generation 1 is reused
# on every later generation — the worker skips both the ~30s torch re-import
# (process stays warm) AND the ~50s 548 MB cache deserialise (day cached).
# Keyed by day string; values are the engineer_day output (identical across
# agents — pure fn of day + cohort-fixed params — so reuse is bit-identical).
# LRU-bounded so a long cohort with many rotating eval days can't grow RAM
# without limit; fixed training days are touched every task so they're MRU
# and never evicted.
_WORKER_DAY_CACHE: dict = {}
# 2026-06-02 (shared-memory-day-cache): RESTORED 4 -> 16. The 4 was a firefight
# band-aid: predictors-ON + full obs dicts are ~1.4 GB each, so 16 days x N
# workers OOM'd a 128 GB box at N>=4. That OOM is now fixed at the source —
# predictors-ON multiprocess uses the SHARED static_obs memmap path
# (_WORKER_STATIC_OBS_CACHE, below), which holds cheap page-cache-shared views
# (~2.2 GB private/worker, measured), NOT per-worker GB dict copies. This dict
# cache (_WORKER_DAY_CACHE) is now reached ONLY on the legacy predictor-OFF
# multiprocess path or the static_obs graceful-fallback (HC#3). 16 restores the
# pre-band-aid reuse depth for those paths. NB: predictor-OFF dicts are still
# ~1.4 GB, so a predictor-OFF cohort at very high N could still pressure RAM —
# route predictor-OFF through the static_obs path too if that ever bites.
_WORKER_DAY_CACHE_MAX = 16


def _worker_load_day(day: str, path: str):
    """Return the engineered features for ``day``, from the per-worker cache
    if present (move-to-MRU), else deserialise the per-day file and cache it.
    """
    cache = _WORKER_DAY_CACHE
    if day in cache:
        cache[day] = cache.pop(day)  # mark most-recently-used
        return cache[day]
    with open(path, "rb") as fh:
        feats = pickle.load(fh)
    cache[day] = feats
    while len(cache) > _WORKER_DAY_CACHE_MAX:
        del cache[next(iter(cache))]  # evict least-recently-used
    return feats


# ── Per-worker static_obs artifact cache (shared-memory-day-cache) ──────────
# Holds DayStaticObs objects, NOT the GB-sized dicts of _WORKER_DAY_CACHE.
# Each value is a thin handle: a read-only memmap VIEW (the actual array data
# lives in the OS page cache, shared across processes) + the small gate-cache
# sidecar. So the per-worker RAM cost is bytes, not GBs — the cap can be high
# (just bounds open-handle / sidecar accumulation over a long cohort with many
# rotating eval days). Keyed by day string; bit-identical across agents (pure
# fn of day + cohort-fixed params + bundle).
_WORKER_STATIC_OBS_CACHE: dict = {}
_WORKER_STATIC_OBS_CACHE_MAX = 64  # cheap views, not the GB dict copies


def _worker_load_static_obs(day: str, npy_path: str, sidecar_path: str):
    """Return the ``DayStaticObs`` for ``day`` (memmapped, read-only), from
    the per-worker cache if present (move-to-MRU), else load it.

    Graceful fallback (HC#3): on a memmap error retry a full (non-mmap) read
    into RAM (slower + more RAM, still correct); on any other failure
    (missing file, schema/contract mismatch, corruption) return ``None`` so
    the caller omits this day from ``static_obs_cache`` and the env rebuilds
    it from scratch (``engineer_day`` + inference). Shared memory is an
    optimisation — its failure must degrade, never kill the cohort.
    """
    cache = _WORKER_STATIC_OBS_CACHE
    # Key by (day, npy_path), NOT day alone: a single warm worker may serve
    # agents on DIFFERENT obs representations (lean vs full are a fresh-blood
    # gene) whose static_obs for the same day live in different cache dirs.
    # Keying on day alone returned the first-loaded variant for both -> a
    # StaticObsCacheMismatch crash (pbt-breeding 2026-06-04). The npy_path
    # encodes the variant. Guarded by
    # tests/test_v2_static_obs_variant_cache.py.
    key = (day, npy_path)
    if key in cache:
        cache[key] = cache.pop(key)  # mark most-recently-used
        return cache[key]
    from training_v2.cohort.static_obs_cache import DayStaticObs

    art = None
    try:
        art = DayStaticObs.load(npy_path, sidecar_path, mmap=True)
    except Exception as exc:  # noqa: BLE001 — degrade on ANY load failure
        # One retry without mmap handles a pure mmap/page-mapping error
        # (rare on Windows); a contract/schema mismatch or missing file
        # fails again here and drops to the from-scratch fallback below.
        try:
            art = DayStaticObs.load(npy_path, sidecar_path, mmap=False)
            logger.warning(
                "static_obs cache for %s: mmap failed (%s); read full into "
                "RAM (slower + more RAM, still correct)", day, exc,
            )
        except Exception as exc2:  # noqa: BLE001
            logger.warning(
                "static_obs cache load failed for %s (%s); falling back to "
                "from-scratch env build for this day", day, exc2,
            )
            return None
    cache[key] = art
    while len(cache) > _WORKER_STATIC_OBS_CACHE_MAX:
        del cache[next(iter(cache))]  # evict least-recently-used
    return art


# ── Per-worker predictor-bundle cache (predictor support, 2026-06-02) ───────
# The predictor bundle (LightGBM + sklearn + torch heads) is loaded by
# REFERENCE — the worker rebuilds it from its manifest paths rather than
# receiving a pickled object across the spawn boundary. This (a) sidesteps
# the "is the bundle picklable" risk entirely, (b) avoids serialising a large
# object into every spawn-arg, and (c) is bit-identical to the parent's
# bundle (same manifests → same files → same deterministic models). Cached
# per-worker keyed by the manifests tuple, so a warm worker loads the bundle
# ONCE across all waves + generations.
_WORKER_PREDICTOR_BUNDLE: dict = {}


def _worker_load_bundle(manifests: tuple):
    """Return the PredictorBundle for ``manifests`` (champion, ranker,
    direction paths), from the per-worker cache if present else loaded once.
    """
    key = tuple(str(m) for m in manifests)
    bundle = _WORKER_PREDICTOR_BUNDLE.get(key)
    if bundle is None:
        from predictors import PredictorBundle
        bundle = PredictorBundle.from_manifests(
            champion_manifest=key[0],
            ranker_manifest=key[1],
            direction_manifest=key[2],
        )
        _WORKER_PREDICTOR_BUNDLE[key] = bundle
    return bundle


def make_pool(n_workers: int) -> "ProcessPoolExecutor":
    """Create a persistent ProcessPoolExecutor for the cohort runner to reuse
    across generations. MKL/OMP single-threading is already pinned at this
    module's import (top of file), which the spawned workers inherit.
    """
    return ProcessPoolExecutor(max_workers=int(n_workers))


def model_store_paths(store) -> dict:
    """Extract a ModelStore's paths into a picklable dict for a worker spec.

    The runner holds a live ModelStore (un-picklable — it owns a sqlite
    connection). Put ``spec["_model_store_paths"] = model_store_paths(store)``
    and each worker rebuilds an equivalent store pointing at the same files.
    """
    return {
        "db_path": str(store.db_path),
        "weights_dir": str(store.weights_dir),
        "bet_logs_dir": str(store.bet_logs_dir),
    }


def default_worker_count(n_agents: int) -> int:
    """Pick a worker count: one per agent, capped so we don't oversubscribe.

    Memory-bandwidth-bound, not just core-bound — the per-agent rollout
    slows under contention, so the THROUGHPUT (agents/sec) curve rises then
    plateaus well below the core count. Measured on the 20-core dev box
    (``tools``/``measure_optimal_n.py``, cached days, per-agent ~72 s solo):

        K=4 0.043  K=8 0.070  K=12 0.084  K=16 0.094(peak)  K=20 0.093 ag/s

    i.e. a flat plateau K≈12-20; the only bad choice is going too LOW (K=4
    is <½ the peak). ``cpu_count-2`` (18 here) sits on the plateau, so it is
    a safe generic default. For best results an operator should run the
    sweep on their own hardware and pin ``--parallel-agents`` to the
    measured peak (the curve shape is machine-specific — beefier memory
    subsystems plateau higher). The caller can override.
    """
    try:
        cpu = os.cpu_count() or 4
    except Exception:
        cpu = 4
    return max(1, min(int(n_agents), cpu - 2))


def _train_agent_worker(spec: dict):
    """Top-level (picklable) worker: a single solo ``train_one_agent``.

    ``spec`` is a kwargs dict for ``train_one_agent`` (must include
    ``model_store=None``). Runs single-threaded; returns the AgentResult.

    Feature-cache injection (skips ``engineer_day`` in the worker's env
    build) is by PATH, not by value — passing the big dict as a spawn-arg
    would pickle it N times. Two forms:

    * ``_feature_cache_day_paths`` (dict ``{day: path}``) — the warm-pool
      form: each day is loaded via the per-worker LRU ``_WORKER_DAY_CACHE``,
      so across generations a reused worker deserialises each day at most
      once. Use this with a persistent executor.
    * ``_feature_cache_path`` (single pickled dict) — the simple form: load
      the whole file each call (no cross-task reuse). Used by the probes.
    """
    import torch
    from training_v2.cohort.gpu_slot import gpu_slot
    from training_v2.cohort.worker import gpu_slot_cap_for, train_one_agent

    spec = dict(spec)
    # Size-aware threading (default 1 == single-thread == bit-identical). The
    # master sets ``_num_threads`` per agent from its hidden_size: big LSTMs get
    # N so their 1024-wide matmuls spread across the cores small agents free as
    # they finish; everything else stays 1. The import-time SES_MP_MAX_THREADS
    # cap must be >= this for it to take effect (the runner sets both together).
    torch.set_num_threads(max(1, int(spec.pop("_num_threads", 1))))
    static_obs_paths = spec.pop("_static_obs_day_paths", None)
    day_paths = spec.pop("_feature_cache_day_paths", None)
    cache_path = spec.pop("_feature_cache_path", None)
    if static_obs_paths is not None:
        # shared-memory-day-cache: load each day's DayStaticObs (memmapped,
        # read-only → one physical copy shared across processes via the OS
        # page cache). A day that fails to load is omitted, so the env
        # rebuilds just that day from scratch (HC#3 graceful fallback).
        static_obs_cache = {}
        for day, (npy, side) in static_obs_paths.items():
            art = _worker_load_static_obs(day, npy, side)
            if art is not None:
                static_obs_cache[day] = art
        if static_obs_cache:
            spec["static_obs_cache"] = static_obs_cache
    elif day_paths is not None:
        spec["feature_cache"] = {
            day: _worker_load_day(day, path) for day, path in day_paths.items()
        }
    elif cache_path is not None:
        with open(cache_path, "rb") as fh:
            spec["feature_cache"] = pickle.load(fh)

    # Production registry writes: each worker constructs its OWN ModelStore
    # pointing at the SHARED db/weights paths and writes its rows directly
    # (WAL + busy_timeout serialise concurrent writers — see
    # ModelStore._get_conn). ModelStore holds a sqlite connection so it
    # can't be pickled across the spawn boundary; passing the paths and
    # rebuilding inside the worker is the picklable route.
    store_paths = spec.pop("_model_store_paths", None)
    if store_paths is not None:
        from registry.model_store import ModelStore
        spec["model_store"] = ModelStore(
            db_path=store_paths["db_path"],
            weights_dir=store_paths["weights_dir"],
            bet_logs_dir=store_paths.get("bet_logs_dir"),
        )

    # Predictor bundle by reference: rebuild from manifests in the worker
    # (cached), bit-identical to the parent's bundle. Injected as
    # predictor_bundle= so the env consumes predictor obs exactly as the
    # sequential path does.
    manifests = spec.pop("_predictor_manifests", None)
    if manifests is not None:
        spec["predictor_bundle"] = _worker_load_bundle(manifests)

    # GPU policy-lane concurrency cap: a GPU-lane agent (big-ctx transformer
    # under --gpu-policy-lane) takes one of N slots before it builds its CUDA
    # policy and holds it through eval, so at most N hold the card at once
    # (OOM guard — 4 ctx256 transformers peaked at 18.7/24 GB, d512/depth6 are
    # heavier). CPU-lane agents get cap 0 (no-op) and run unthrottled. The
    # `with` guarantees the OS lock releases even if train raises — vital on
    # the WARM pool where a leaked slot would wedge the lane for the worker's
    # whole life. Pop the cap key unconditionally so it never leaks into
    # **spec; mock specs without "genes" (unit tests) take the uncapped path.
    _gpu_max = int(spec.pop("gpu_lane_max_concurrent", 2))
    _genes = spec.get("genes")
    gpu_cap = (
        gpu_slot_cap_for(_genes, bool(spec.get("gpu_policy_lane", False)), _gpu_max)
        if _genes is not None else 0
    )
    t0 = time.perf_counter()
    with gpu_slot(gpu_cap, label=str(spec.get("agent_id", ""))):
        result = train_one_agent(**spec)
    # Stamp the per-agent wall (the cluster wall is max over workers).
    try:
        result.train.wall_time_sec = time.perf_counter() - t0
    except Exception:
        pass
    return result


def train_cluster_multiproc(
    specs: list[dict],
    *,
    n_workers: int | None = None,
    executor: "ProcessPoolExecutor | None" = None,
) -> list:
    """Train a cluster of agents in parallel processes.

    Parameters
    ----------
    specs:
        One ``train_one_agent`` kwargs dict per agent, in order. Each MUST
        set ``model_store=None`` (the parent writes the registry) and
        ``device="cpu"`` (multiprocess is CPU-parallel; the GPU is not
        shared across N processes). Predictor-bundle args, if used, must
        be picklable or loaded inside the worker.
    executor:
        Optional PERSISTENT pool (from ``make_pool``). When given, the pool
        is reused (NOT shut down here) so its workers stay warm across
        generations — torch is imported once per worker and the per-worker
        day cache survives, so generation 2+ skips both startup costs. When
        ``None``, a fresh pool is created and torn down for this call.

    Returns
    -------
    ``list[AgentResult]`` in input order (``ProcessPoolExecutor.map`` is
    order-preserving).
    """
    n = len(specs)
    if n == 0:
        return []
    t0 = time.perf_counter()
    if executor is not None:
        logger.info(
            "train_cluster_multiproc: %d agents on persistent pool (warm)", n,
        )
        results = list(executor.map(_train_agent_worker, specs))
        logger.info(
            "train_cluster_multiproc: %d agents done in %.0fs wall",
            n, time.perf_counter() - t0,
        )
        return results
    nw = int(n_workers) if n_workers else default_worker_count(n)
    logger.info(
        "train_cluster_multiproc: %d agents across %d worker processes "
        "(1 thread each, fresh pool)", n, nw,
    )
    with ProcessPoolExecutor(max_workers=nw) as ex:
        results = list(ex.map(_train_agent_worker, specs))
    logger.info(
        "train_cluster_multiproc: %d agents done in %.0fs wall",
        n, time.perf_counter() - t0,
    )
    return results


def prebuild_feature_cache(
    days: list[str],
    *,
    data_dir: "Path",
    scorer_dir: "Path | None" = None,
    into: dict | None = None,
) -> dict:
    """Pre-engineer the per-day ``feature_cache`` ONCE, in the parent.

    Returns a ``{day: engineer_day(...) output}`` dict identical to what each
    worker would compute on its own first build — the engineering params come
    from the cohort-fixed ``scalping_train_config()`` (obi_top_n, microprice,
    traded-delta windows, etc. are NOT per-agent genes), so the cache content
    is constant across the cohort. Building a throwaway env per unique day
    populates the cache via the canonical ``_build_env_for_day`` path; only
    ``engineer_day`` output is stored (predictor-off; per-env static-obs /
    runner-maps are not cached and are rebuilt cheaply per worker).

    Caller pickles the result with ``save_shared_cache`` and passes the path
    via each spec's ``_feature_cache_path`` — workers then skip
    ``engineer_day`` (the dominant build cost), recovering the cross-process
    cache-sharing that splitting agents into separate processes would
    otherwise lose. Bit-identical by construction (``engineer_day`` is a pure
    function of day + fixed params); gated end-to-end by the R5 cache-share
    parity test.
    """
    from training_v2.cohort.worker import (
        _build_env_for_day, DEFAULT_SCORER_DIR, scalping_train_config,
    )

    cfg = scalping_train_config()
    sdir = scorer_dir if scorer_dir is not None else DEFAULT_SCORER_DIR
    cache: dict = into if into is not None else {}
    for day in dict.fromkeys(days):  # unique, order-preserving
        if day in cache:  # already engineered (cross-generation reuse)
            continue
        t0 = time.perf_counter()
        _build_env_for_day(
            day_str=day, data_dir=data_dir, cfg=cfg, scorer_dir=sdir,
            feature_cache=cache,
        )
        logger.info(
            "prebuild_feature_cache: engineered %s in %.0fs", day,
            time.perf_counter() - t0,
        )
    return cache


def save_shared_cache(cache: dict, path: "Path") -> "Path":
    """Pickle a pre-built ``feature_cache`` to ``path`` for worker reuse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def save_shared_cache_per_day(
    cache: dict, cache_dir: "Path", days: list[str],
) -> dict:
    """Write ONE pickle per day under ``cache_dir`` and return ``{day: path}``.

    Per-day files (vs one big file) let a warm-pool worker load each day at
    most once across all generations via ``_WORKER_DAY_CACHE`` — a rotated
    eval day costs one deserialise the first time any worker sees it, never
    again. A day whose file already exists is left untouched (engineer_day
    is deterministic, so the bytes would be identical), so re-calling this
    each generation only writes genuinely-new days.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    for day in dict.fromkeys(days):
        p = cache_dir / f"mp_day_{day}.pkl"
        if not p.exists():
            with open(p, "wb") as fh:
                pickle.dump(cache[day], fh, protocol=pickle.HIGHEST_PROTOCOL)
        out[day] = str(p)
    return out


# ── Multiprocess day-cache memory-budget guard (2026-06-04) ───────────────
#
# The overnight pbt_long campaign OOM-crashed at gen 2 of every run (before
# the gauntlet ever reached R3 -> ZERO champions, a whole night wasted with
# NO guard): predictors-OFF caches each ~1.45GB engineered DAY in EVERY
# worker, so 16 workers x 12 days plateaus at ~128GB -> MemoryError. This
# estimator + assert catch that config BEFORE training starts. The
# per-worker DICT path multiplies day bytes by n_workers; the shared
# static_obs MEMMAP path (predictors-ON) holds ONE copy across all workers.

# Unpickled engineered-day dicts measured ~5x their on-disk pickle (numpy +
# Python object overhead): a 0.30GB pickle -> 1.45GB resident.
_DICT_UNPICKLE_FACTOR = 5.0
# Rough non-day per-worker resident (torch + policy + PPO buffers + interp).
_PER_WORKER_OVERHEAD_GB = 2.5


def estimate_day_cache_peak_gb(
    *,
    day_cache_bytes: "list[int]",
    n_workers: int,
    shared: bool,
    per_worker_overhead_gb: float = _PER_WORKER_OVERHEAD_GB,
) -> float:
    """Estimate peak RAM (GB) the day cache adds across the worker pool.

    ``shared=True`` (static_obs memmap path): each day's ``.npy`` is loaded
    ``mmap_mode='r'`` so the OS page cache holds ONE physical copy across all
    workers -> ``sum(days) + n_workers * overhead``.

    ``shared=False`` (per-worker engineered-dict path): EVERY worker
    deserialises EVERY day it touches into its OWN dict cache ->
    ``n_workers * sum(days) * UNPICKLE_FACTOR + n_workers * overhead``. This
    is the OOM trap the overnight run fell into.
    """
    total_days_gb = sum(int(b) for b in day_cache_bytes) / 1e9
    if shared:
        return total_days_gb + n_workers * per_worker_overhead_gb
    return (
        n_workers * total_days_gb * _DICT_UNPICKLE_FACTOR
        + n_workers * per_worker_overhead_gb
    )


def assert_day_cache_fits(
    *,
    day_cache_bytes: "list[int]",
    n_workers: int,
    shared: bool,
    total_ram_gb: float | None = None,
    refuse_frac: float = 0.90,
    warn_frac: float = 0.70,
) -> float:
    """Refuse a multiprocess config whose projected day-cache RAM would OOM.

    Returns the estimate (GB). Raises :class:`MemoryError` (caught early,
    before any training) when the estimate exceeds ``refuse_frac`` of system
    RAM; logs a loud warning above ``warn_frac``. The message points at the
    fixes: fewer ``--parallel-agents`` / ``--days``, or predictors-ON (which
    shares one copy per day via the static_obs memmap cache).
    """
    if total_ram_gb is None:
        try:
            import psutil
            total_ram_gb = psutil.virtual_memory().total / 1e9
        except Exception:
            total_ram_gb = 0.0  # unknown -> estimate only, no hard refuse
    est = estimate_day_cache_peak_gb(
        day_cache_bytes=day_cache_bytes, n_workers=n_workers, shared=shared,
    )
    path = "shared static_obs memmap" if shared else "per-worker dict"
    n_days = len(day_cache_bytes)
    if total_ram_gb and est > refuse_frac * total_ram_gb:
        raise MemoryError(
            f"Projected day-cache RAM ~{est:.0f}GB ({path} path, "
            f"{n_workers} workers x {n_days} days) exceeds {refuse_frac:.0%} "
            f"of the {total_ram_gb:.0f}GB box -- this WILL OOM mid-run "
            f"(this is the bug that wasted the overnight pbt_long campaign). "
            f"Fix: reduce --parallel-agents or --days, or run predictors-ON "
            f"so the static_obs memmap cache holds ONE shared copy per day "
            f"instead of one per worker.",
        )
    if total_ram_gb and est > warn_frac * total_ram_gb:
        logger.warning(
            "Day-cache RAM estimate ~%.0fGB (%s path, %d workers x %d days) "
            "is %.0f%% of the %.0fGB box -- watch memory.",
            est, path, n_workers, n_days, 100 * est / total_ram_gb,
            total_ram_gb,
        )
    else:
        logger.info(
            "Day-cache RAM estimate ~%.0fGB (%s path, %d workers x %d days) "
            "-- fits the %.0fGB box.",
            est, path, n_workers, n_days, total_ram_gb or 0,
        )
    return est


def prebuild_static_obs_cache(
    days: list[str],
    *,
    data_dir: "Path",
    cache_dir: "Path",
    predictor_bundle: object,
    use_race_outcome_predictor: bool,
    use_direction_predictor: bool,
    predictor_lean_obs: bool = False,
    scorer_dir: "Path | None" = None,
) -> dict:
    """Bake each day's shareable ``static_obs`` artifact ONCE, in the master.

    The shared-memory-day-cache replacement for ``prebuild_feature_cache`` +
    ``save_shared_cache_per_day`` on the multiprocess path. For each unique
    day it builds the **canonical predictors-ON env** (the exact path a
    worker uses, via ``_build_env_for_day``), captures ``env._static_obs`` +
    the predictor gate caches into a
    :class:`training_v2.cohort.static_obs_cache.DayStaticObs`, and writes
    ``static_obs_{day}.npy`` (the big memmappable array, predictors baked in)
    + ``meta_{day}.pkl`` (gate caches + obs-contract manifest) under
    ``cache_dir``. Returns ``{day: (npy_path, sidecar_path)}``.

    Why this replaces the dict prebuild (see
    ``plans/shared-memory-day-cache/step0_structure.md``):

    * The old ``mp_day_{day}.pkl`` held the ``engineer_day`` DICTS (~1 GB/day
      of Python objects, NOT memmappable, copied master + N workers → OOM).
    * The ``.npy`` here holds the downstream ``static_obs`` arrays
      (~90 MB/day full-obs, ~10–20× smaller) that workers
      ``np.load(mmap_mode='r')`` so the OS page cache holds ONE physical copy
      shared across processes.
    * **Predictors are baked here** (the master holds the bundle), so workers
      skip ``engineer_day`` + ``_features_to_array`` + per-worker predictor
      inference — and the in-place dict mutation that made the dicts
      unshareable never happens on the shared object.

    Master RAM stays ~one day's arrays (build → save → drop), not the
    all-days dict footprint. A day whose artifact already exists is left
    untouched (deterministic build, cross-generation reuse — same contract
    as ``save_shared_cache_per_day``). Bit-identity is gated by
    ``tests/test_env_golden_parity.py`` (the env consume-path reproduces the
    from-scratch build).
    """
    from training_v2.cohort.worker import (
        DEFAULT_SCORER_DIR,
        _build_env_for_day,
        scalping_train_config,
    )
    from training_v2.cohort.static_obs_cache import DayStaticObs

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sdir = scorer_dir if scorer_dir is not None else DEFAULT_SCORER_DIR

    # Mirror the worker's env config: predictor flags flow via
    # cfg["observations"] (train_one_agent sets them there and lets the env
    # resolve), so the prebuilt env is byte-identical to the worker's.
    # Action-side config (reward overrides, pwin thresholds, race-confidence,
    # direction gate) does NOT affect static_obs or the cached gate caches,
    # so it is intentionally omitted here (verified by golden parity).
    cfg = scalping_train_config()
    if use_race_outcome_predictor:
        cfg.setdefault("observations", {})["use_race_outcome_predictor"] = True
    if use_direction_predictor:
        cfg.setdefault("observations", {})["use_direction_predictor"] = True

    out: dict = {}
    for day in dict.fromkeys(days):
        npy_path = cache_dir / f"static_obs_{day}.npy"
        side_path = cache_dir / f"meta_{day}.pkl"
        if npy_path.exists() and side_path.exists():
            out[day] = (str(npy_path), str(side_path))
            continue
        t0 = time.perf_counter()
        env, _shim = _build_env_for_day(
            day_str=day, data_dir=data_dir, cfg=cfg, scorer_dir=sdir,
            predictor_bundle=predictor_bundle,
            predictor_lean_obs=predictor_lean_obs,
        )
        artifact = DayStaticObs.from_env(env)
        artifact.save(npy_path, side_path)
        n_ticks = int(artifact.static_obs_flat.shape[0])
        mb = artifact.static_obs_flat.nbytes / 1e6
        del env, artifact
        out[day] = (str(npy_path), str(side_path))
        logger.info(
            "prebuild_static_obs_cache: baked %s in %.0fs "
            "(%d ticks, %.0f MB → %s)",
            day, time.perf_counter() - t0, n_ticks, mb, npy_path.name,
        )
    return out
