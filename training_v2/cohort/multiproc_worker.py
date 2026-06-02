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

# Single-thread BLAS/OMP per process. Set BEFORE torch/numpy import so the
# spawned worker (which re-imports this module) picks it up. Unconditional
# (not setdefault) so an inherited multi-thread value can't leak in.
for _v in (
    "MKL_NUM_THREADS", "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_v] = "1"

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
    "model_store_paths",
]


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

    Memory- and bandwidth-bound, not just core-bound — the per-agent
    rollout slows under contention (measured build 61→95s, rollout 28→47s
    at 8-way), so capping at ~cpu_count-2 leaves headroom for the OS + the
    parent. The caller can override.
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

    If ``spec`` carries the private key ``_feature_cache_path`` (a pickled
    pre-built ``feature_cache``), the worker loads it ONCE and injects it as
    ``feature_cache=`` so the env build skips ``engineer_day``. Passing the
    cache by PATH (not by value) avoids pickling the large dict into every
    spawn-arg N times — the parent serialises once to disk, workers
    deserialise in parallel.
    """
    import torch
    torch.set_num_threads(1)
    from training_v2.cohort.worker import train_one_agent

    spec = dict(spec)
    cache_path = spec.pop("_feature_cache_path", None)
    if cache_path is not None:
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

    t0 = time.perf_counter()
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

    Returns
    -------
    ``list[AgentResult]`` in input order (``ProcessPoolExecutor.map`` is
    order-preserving).
    """
    n = len(specs)
    if n == 0:
        return []
    nw = int(n_workers) if n_workers else default_worker_count(n)
    logger.info(
        "train_cluster_multiproc: %d agents across %d worker processes "
        "(1 thread each)", n, nw,
    )
    t0 = time.perf_counter()
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
