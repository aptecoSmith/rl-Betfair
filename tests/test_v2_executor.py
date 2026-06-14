"""Tests for training_v2/cohort/executor.py — the gauntlet-pipeline executor.

The executor is the pure execution primitive (Phase 2). These tests exercise it
WITHOUT real training: the sequential path takes a stub ``train_one_agent_fn``.
They guard the load-bearing contracts — recipe purity (structural + sidecar),
no-selection ordering, leakage refusal, and spec-key drift vs ``train_one_agent``.
"""
from __future__ import annotations

import inspect
import types
from pathlib import Path

import pytest

from training_v2.cohort import executor as ex
from training_v2.cohort.genes import CohortGenes
from training_v2.cohort.worker import train_one_agent


def _genes(**kw) -> CohortGenes:
    base = dict(learning_rate=3e-4, entropy_coeff=1e-3, clip_range=0.2,
                gae_lambda=0.95, value_coeff=0.5, mini_batch_size=64,
                hidden_size=128)
    base.update(kw)
    return CohortGenes(**base)


def _fake_result(agent_id, weights_path, *, locked=5.0, naked=-3.0, pnl=1.0):
    ev = types.SimpleNamespace(locked_pnl=locked, naked_pnl=naked, day_pnl=pnl)
    return types.SimpleNamespace(
        agent_id=agent_id, weights_path=weights_path, eval=ev)


def _cfg(tmp_path) -> ex.TrancheExecConfig:
    # Predictor-less + sequential => no bundle/manifest/cache machinery needed.
    return ex.TrancheExecConfig(
        data_dir=tmp_path / "data", output_dir=tmp_path / "out",
        model_store=None, predictor_bundle=None,
        use_race_outcome_predictor=False, parallel_agents=0)


# ── config_hash ────────────────────────────────────────────────────────────


def test_config_hash_same_recipe_same_hash():
    assert ex.config_hash(_genes()) == ex.config_hash(_genes())


def test_config_hash_differs_on_any_gene():
    assert ex.config_hash(_genes()) != ex.config_hash(_genes(open_cost=1.0))


# ── recipe purity (structural) ─────────────────────────────────────────────


def test_t1_with_weights_raises(tmp_path):
    a = ex.RecipeAgent(agent_id="a", genes=_genes(), lineage_id="L",
                       origin="fresh", init_weights_path="x.pt")
    with pytest.raises(ex.RecipePurityError):
        ex.run_tranche([a], tranche_K=1, train_days_for_K=["d1"],
                       validation_days=["v1"], cfg=_cfg(tmp_path))


def test_tk_without_weights_raises(tmp_path):
    a = ex.RecipeAgent(agent_id="a", genes=_genes(), lineage_id="L",
                       origin="climber", init_weights_path=None)
    with pytest.raises(ex.RecipePurityError):
        ex.run_tranche([a], tranche_K=2, train_days_for_K=["d2"],
                       validation_days=["v1"], cfg=_cfg(tmp_path))


# ── recipe purity (sidecar) ────────────────────────────────────────────────


def test_genehash_sidecar_blocks_cross_recipe_warmstart(tmp_path):
    # Pretend a K-1 checkpoint exists, cooked under recipe A.
    wpath = tmp_path / "parent.pt"
    wpath.write_bytes(b"x")
    ex._write_genehash_sidecar(wpath, _genes(open_cost=1.0),
                               lineage_id="L", tranche_K=1)
    # An agent of lineage L but DIFFERENT genes tries to inherit it -> chimera.
    bad = ex.RecipeAgent(agent_id="a", genes=_genes(open_cost=2.0),
                         lineage_id="L", origin="climber",
                         init_weights_path=str(wpath))
    with pytest.raises(ex.RecipePurityError):
        ex.run_tranche([bad], tranche_K=2, train_days_for_K=["d2"],
                       validation_days=["v1"], cfg=_cfg(tmp_path))


def test_genehash_sidecar_allows_same_recipe_warmstart(tmp_path):
    wpath = tmp_path / "parent.pt"
    wpath.write_bytes(b"x")
    g = _genes(open_cost=1.5)
    ex._write_genehash_sidecar(wpath, g, lineage_id="L", tranche_K=1)
    good = ex.RecipeAgent(agent_id="a", genes=g, lineage_id="L",
                          origin="climber", init_weights_path=str(wpath))
    calls = []

    def stub(**spec):
        calls.append(spec)
        return _fake_result(spec["agent_id"], str(tmp_path / "child.pt"))

    out = ex.run_tranche([good], tranche_K=2, train_days_for_K=["d2"],
                         validation_days=["v1"], cfg=_cfg(tmp_path),
                         train_one_agent_fn=stub)
    assert len(out) == 1 and out[0].error is None
    # The warm-start weights were threaded through to train_one_agent.
    assert calls[0]["init_weights_path"] == str(wpath)


# ── leakage refusal ────────────────────────────────────────────────────────


def test_validation_train_overlap_raises(tmp_path):
    a = ex.RecipeAgent(agent_id="a", genes=_genes(), lineage_id="L",
                       origin="fresh")
    with pytest.raises(ValueError):
        ex.run_tranche([a], tranche_K=1, train_days_for_K=["d1", "shared"],
                       validation_days=["shared"], cfg=_cfg(tmp_path))


# ── no-selection ordering + validation extraction + sidecar write ──────────


def test_run_tranche_preserves_order_and_extracts_validation(tmp_path):
    agents = [
        ex.RecipeAgent(agent_id=f"a{i}", genes=_genes(open_cost=float(i)),
                       lineage_id=f"L{i}", origin="fresh", seed=i)
        for i in range(3)
    ]

    def stub(**spec):
        i = int(spec["agent_id"][1:])
        wp = str(tmp_path / f"w{i}.pt")
        Path(wp).write_bytes(b"x")
        return _fake_result(spec["agent_id"], wp, locked=float(i), naked=-float(i))

    out = ex.run_tranche(agents, tranche_K=1, train_days_for_K=["d1"],
                         validation_days=["v1"], cfg=_cfg(tmp_path),
                         train_one_agent_fn=stub)
    # Order preserved, NO ranking applied (a2 with best locked is still last).
    assert [r.agent_id for r in out] == ["a0", "a1", "a2"]
    assert [r.validation_locked for r in out] == [0.0, 1.0, 2.0]
    # A genehash sidecar was stamped next to every saved checkpoint.
    for i in range(3):
        side = ex._genehash_path(str(tmp_path / f"w{i}.pt"))
        assert side.exists()


# ── spec-key drift guard ───────────────────────────────────────────────────


def test_spec_keys_match_train_one_agent_signature(tmp_path):
    """Every non-worker spec key must be a real train_one_agent kwarg.

    The executor reuses the worker verbatim; if train_one_agent gains/renames a
    kwarg this catches the drift (mirrors the runner's same invariant).
    """
    a = ex.RecipeAgent(agent_id="a", genes=_genes(), lineage_id="L",
                       origin="fresh", seed=0)
    spec = ex._build_spec(
        a, tranche_K=1, train_days=["d1"], validation_days=["v1"],
        cfg=_cfg(tmp_path), static_obs_paths_by_lean=None,
        store_paths=None, mp_predictor_manifests=None, idx=0, n_agents=1)
    sig = set(inspect.signature(train_one_agent).parameters)
    # Worker-only keys the multiprocess worker pops before calling the worker.
    worker_only = {"_feature_cache_day_paths", "_static_obs_day_paths",
                   "_model_store_paths", "_predictor_manifests", "_num_threads",
                   "gpu_lane_max_concurrent"}
    for k in spec:
        if k in worker_only:
            continue
        assert k in sig, f"spec key {k!r} is not a train_one_agent parameter"
