"""R5 multiprocess-cluster plumbing tests (training-speedup-v2).

These are the FAST, CI-friendly guards for ``multiproc_worker``: worker
count selection, path/cache serialisation, and — the load-bearing one —
that ``_train_agent_worker`` correctly injects the shared ``feature_cache``
(by path) and the per-worker ``ModelStore`` (by paths) into the
``train_one_agent`` call while popping the private spec keys.

The end-to-end BIT-IDENTITY gate (parallel == sequential, and shared-cache
parallel == sequential) is heavy (real-data training) and is proven by the
probes ``C:/tmp/measure_multiproc_train.py`` and
``measure_multiproc_shared.py`` — both PASS:

    [R5] GATE PASS  ... sequential 444s -> parallel 142s = 3.1x  (N=4, basic)
    [R5-share] GATE PASS  (shared-cache parallel == sequential)

A monkeypatch on ``train_one_agent`` cannot cross the spawn boundary into a
pool worker, so the orchestration itself is exercised in-process here and
end-to-end by the probes.
"""
from __future__ import annotations

import os
import pickle
import types

import pytest

from training_v2.cohort import multiproc_worker as mp


def test_default_worker_count_one_per_agent_capped():
    cpu = os.cpu_count() or 4
    cap = max(1, cpu - 2)
    assert mp.default_worker_count(1) == 1
    assert mp.default_worker_count(2) == min(2, cap)
    # many agents -> capped at cpu-2 (leave headroom for OS + parent)
    assert mp.default_worker_count(10_000) == cap


def test_model_store_paths_are_strings(tmp_path):
    store = types.SimpleNamespace(
        db_path=tmp_path / "m.db",
        weights_dir=tmp_path / "w",
        bet_logs_dir=tmp_path / "b",
    )
    p = mp.model_store_paths(store)
    assert set(p) == {"db_path", "weights_dir", "bet_logs_dir"}
    assert all(isinstance(v, str) for v in p.values())
    assert p["db_path"].endswith("m.db")


def test_save_shared_cache_roundtrip(tmp_path):
    cache = {"2026-05-09": ["features"], "2026-05-13": [1, 2, 3]}
    path = mp.save_shared_cache(cache, tmp_path / "sub" / "c.pkl")
    assert path.exists()
    with open(path, "rb") as fh:
        assert pickle.load(fh) == cache


def test_empty_specs_returns_empty():
    assert mp.train_cluster_multiproc([]) == []


def test_worker_injects_cache_and_store_and_pops_private_keys(
    tmp_path, monkeypatch,
):
    """The load-bearing plumbing test: ``_train_agent_worker`` loads the
    shared cache by path, builds a ModelStore from paths, injects both into
    the ``train_one_agent`` call, and removes the private ``_`` spec keys.
    """
    cache = {"d": ["engineered"]}
    cache_path = mp.save_shared_cache(cache, tmp_path / "c.pkl")

    captured: dict = {}

    def fake_train_one_agent(**kw):
        captured.update(kw)
        return types.SimpleNamespace(
            train=types.SimpleNamespace(wall_time_sec=0.0)
        )

    import training_v2.cohort.worker as worker
    monkeypatch.setattr(worker, "train_one_agent", fake_train_one_agent)

    spec = dict(
        agent_id="r0",
        seed=42,
        model_store=None,
        _feature_cache_path=str(cache_path),
        _model_store_paths={
            "db_path": str(tmp_path / "m.db"),
            "weights_dir": str(tmp_path / "w"),
            "bet_logs_dir": str(tmp_path / "b"),
        },
    )
    mp._train_agent_worker(spec)

    # cache loaded from file and injected
    assert captured["feature_cache"] == cache
    # store built from paths and injected (overrides the None in the spec)
    assert captured["model_store"] is not None
    assert str(captured["model_store"].db_path).endswith("m.db")
    # private keys popped — never forwarded to train_one_agent
    assert "_feature_cache_path" not in captured
    assert "_model_store_paths" not in captured
    # real kwargs preserved
    assert captured["agent_id"] == "r0"
    assert captured["seed"] == 42


def test_save_shared_cache_per_day_one_file_each_idempotent(tmp_path):
    cache = {"2026-05-09": ["a"], "2026-05-13": ["b"], "2026-05-14": ["c"]}
    paths = mp.save_shared_cache_per_day(
        cache, tmp_path / "c", ["2026-05-09", "2026-05-13"])
    assert set(paths) == {"2026-05-09", "2026-05-13"}
    for d, p in paths.items():
        assert os.path.exists(p)
        with open(p, "rb") as fh:
            assert pickle.load(fh) == cache[d]
    # re-call with an extra day: existing files untouched, new one written
    mtime_before = os.path.getmtime(paths["2026-05-09"])
    paths2 = mp.save_shared_cache_per_day(
        cache, tmp_path / "c", ["2026-05-09", "2026-05-14"])
    assert os.path.getmtime(paths2["2026-05-09"]) == mtime_before  # not rewritten
    assert os.path.exists(paths2["2026-05-14"])


def test_worker_day_cache_loads_once_then_reuses(tmp_path):
    mp._WORKER_DAY_CACHE.clear()
    p = tmp_path / "d.pkl"
    with open(p, "wb") as fh:
        pickle.dump(["feat"], fh)
    a = mp._worker_load_day("2026-05-09", str(p))
    assert a == ["feat"] and "2026-05-09" in mp._WORKER_DAY_CACHE
    # delete the file — a second load MUST come from the cache, not disk
    os.remove(p)
    b = mp._worker_load_day("2026-05-09", str(p))
    assert b is a  # exact cached object
    mp._WORKER_DAY_CACHE.clear()


def test_worker_day_cache_lru_evicts_lru_keeps_mru(tmp_path, monkeypatch):
    mp._WORKER_DAY_CACHE.clear()
    monkeypatch.setattr(mp, "_WORKER_DAY_CACHE_MAX", 2)

    def mkfile(d):
        p = tmp_path / f"{d}.pkl"
        with open(p, "wb") as fh:
            pickle.dump([d], fh)
        return str(p)

    mp._worker_load_day("A", mkfile("A"))
    mp._worker_load_day("B", mkfile("B"))   # cache: A, B
    mp._worker_load_day("A", mkfile("A"))   # touch A -> order B, A
    mp._worker_load_day("C", mkfile("C"))   # evict LRU (B); keep A, add C
    assert set(mp._WORKER_DAY_CACHE) == {"A", "C"}
    mp._WORKER_DAY_CACHE.clear()


def test_worker_injects_day_paths_via_cache(tmp_path, monkeypatch):
    mp._WORKER_DAY_CACHE.clear()
    paths = mp.save_shared_cache_per_day(
        {"d1": ["x"], "d2": ["y"]}, tmp_path / "c", ["d1", "d2"])
    captured: dict = {}

    def fake_train_one_agent(**kw):
        captured.update(kw)
        return types.SimpleNamespace(
            train=types.SimpleNamespace(wall_time_sec=0.0))

    import training_v2.cohort.worker as worker
    monkeypatch.setattr(worker, "train_one_agent", fake_train_one_agent)
    mp._train_agent_worker(dict(agent_id="r0", _feature_cache_day_paths=paths))
    assert captured["feature_cache"] == {"d1": ["x"], "d2": ["y"]}
    assert "_feature_cache_day_paths" not in captured
    mp._WORKER_DAY_CACHE.clear()


def test_make_pool_returns_executor():
    from concurrent.futures import ProcessPoolExecutor
    pool = mp.make_pool(2)
    try:
        assert isinstance(pool, ProcessPoolExecutor)
    finally:
        pool.shutdown(wait=False)


def test_train_cluster_reuses_executor_without_shutting_it_down():
    """A persistent (warm) pool passed via executor= must be reused and NOT
    shut down by the call — that's what keeps workers warm across generations.
    """
    calls: dict = {}

    class FakeExec:
        def map(self, fn, specs):
            calls["mapped"] = list(specs)
            return [f"res-{s['agent_id']}" for s in calls["mapped"]]

        def shutdown(self, *a, **k):
            calls["shutdown"] = True

    out = mp.train_cluster_multiproc(
        [{"agent_id": "a"}, {"agent_id": "b"}], executor=FakeExec())
    assert out == ["res-a", "res-b"]          # order preserved
    assert len(calls["mapped"]) == 2          # dispatched via the pool
    assert "shutdown" not in calls            # warm pool left alive


def test_resolve_parallel_agents_default_is_16():
    from training_v2.cohort.runner import _resolve_parallel_agents
    assert _resolve_parallel_agents(
        None, batched=False, has_predictor=False) == 16


def test_resolve_parallel_agents_explicit_value_kept():
    from training_v2.cohort.runner import _resolve_parallel_agents
    assert _resolve_parallel_agents(8, batched=False, has_predictor=False) == 8
    assert _resolve_parallel_agents(0, batched=False, has_predictor=False) == 0


def test_resolve_parallel_agents_batched_default_yields_to_zero():
    from training_v2.cohort.runner import _resolve_parallel_agents
    # default 16 + --batched -> 0 (batched wins, NO error)
    assert _resolve_parallel_agents(
        None, batched=True, has_predictor=False) == 0


def test_resolve_parallel_agents_batched_explicit_raises():
    from training_v2.cohort.runner import _resolve_parallel_agents
    with pytest.raises(SystemExit):
        _resolve_parallel_agents(8, batched=True, has_predictor=False)


def test_resolve_parallel_agents_predictor_default_yields_to_zero():
    from training_v2.cohort.runner import _resolve_parallel_agents
    # default 16 + predictors -> 0 (auto-disable to sequential, NO error)
    assert _resolve_parallel_agents(
        None, batched=False, has_predictor=True) == 0


def test_resolve_parallel_agents_predictor_explicit_raises():
    from training_v2.cohort.runner import _resolve_parallel_agents
    with pytest.raises(SystemExit):
        _resolve_parallel_agents(8, batched=False, has_predictor=True)


def test_worker_without_private_keys_is_passthrough(tmp_path, monkeypatch):
    """No ``_feature_cache_path`` / ``_model_store_paths`` -> the spec is
    forwarded unchanged (solo-equivalent, model_store stays as given)."""
    captured: dict = {}

    def fake_train_one_agent(**kw):
        captured.update(kw)
        return types.SimpleNamespace(
            train=types.SimpleNamespace(wall_time_sec=0.0)
        )

    import training_v2.cohort.worker as worker
    monkeypatch.setattr(worker, "train_one_agent", fake_train_one_agent)

    spec = dict(agent_id="r1", seed=7, model_store=None)
    mp._train_agent_worker(spec)

    assert captured == {"agent_id": "r1", "seed": 7, "model_store": None}
    assert "feature_cache" not in captured
