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
