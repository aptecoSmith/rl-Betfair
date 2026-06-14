"""Tests for tools/gene_register.py — the Phase 1 coverage-map tool.

Read-only tool, so the tests are pure-function unit checks on the binning /
config-hash / coverage aggregation, plus a smoke test that the real registry
loads and de-dups by config hash (not by model_id).
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from tools import gene_register as gr


def _rec(model_id, era="e", genes=None, **kw):
    return gr.AgentRecord(model_id=model_id, era=era, genes=genes or {}, **kw)


def test_config_hash_collapses_identical_recipes():
    a = {"learning_rate": 1e-4, "hidden_size": 128, "architecture": "lstm"}
    b = {"architecture": "lstm", "hidden_size": 128, "learning_rate": 1e-4}
    # Key order must not matter.
    assert gr._config_hash(a) == gr._config_hash(b)


def test_config_hash_float_noise_tolerated():
    a = {"learning_rate": 0.000100000001}
    b = {"learning_rate": 0.000100000002}
    assert gr._config_hash(a) == gr._config_hash(b)


def test_config_hash_distinguishes_real_differences():
    a = {"open_cost": 1.0}
    b = {"open_cost": 2.0}
    assert gr._config_hash(a) != gr._config_hash(b)


def test_choice_bins_one_column_per_value():
    spec = gr.GeneSpec("architecture", "choice", choices=("lstm", "transformer"))
    bins = gr._bins_for_spec(spec, ["lstm", "lstm", "transformer"], n_bins=8)
    labels = [b.label for b in bins]
    assert labels == ["lstm", "transformer"]


def test_choice_bins_append_unseen_observed_value():
    # A reeval override outside the declared sample set must not be dropped.
    spec = gr.GeneSpec("force_close_before_off_seconds", "choice",
                       choices=(0.0,))
    bins = gr._bins_for_spec(spec, [0.0, 120.0], n_bins=8)
    labels = [b.label for b in bins]
    assert "0.0" in labels and "120.0" in labels


def test_numeric_assign_bin_inclusive_high_edge():
    spec = gr.GeneSpec("clip_range", "float", 0.1, 0.3)
    bins = gr._bins_for_spec(spec, [0.1, 0.3], n_bins=4)
    # Value at the very top edge lands in the last bin (not dropped).
    assert gr._assign_bin(spec, bins, 0.3) == len(bins) - 1
    assert gr._assign_bin(spec, bins, 0.1) == 0


def test_logfloat_edges_are_log_spaced():
    edges = gr._bin_edges(1e-5, 1e-3, 2, log=True)
    # Geometric midpoint of [1e-5, 1e-3] is 1e-4.
    assert math.isclose(edges[1], 1e-4, rel_tol=1e-6)


def test_build_coverage_dedups_by_config_not_model_id():
    # Two models, SAME recipe -> one visited cell. A third, different recipe.
    g1 = {"open_cost": 1.0}
    g2 = {"open_cost": 3.0}
    recs = [
        _rec("m1", genes=dict(g1), holdout_locked=10.0),
        _rec("m2", genes=dict(g1), holdout_locked=20.0),  # clone of g1
        _rec("m3", genes=dict(g2), holdout_locked=5.0),
    ]
    coverage, n_configs = gr.build_coverage(recs, ["open_cost"], n_bins=8)
    assert n_configs == 2  # g1 (twice) collapses to one
    bins = coverage["open_cost"]
    total = sum(b.n_configs for b in bins)
    assert total == 2


def test_build_coverage_keeps_best_holdout_for_a_config():
    g1 = {"open_cost": 1.0}
    recs = [
        _rec("m1", genes=dict(g1), holdout_locked=10.0),
        _rec("m2", genes=dict(g1), holdout_locked=42.0),
    ]
    coverage, _ = gr.build_coverage(recs, ["open_cost"], n_bins=8)
    visited = [b for b in coverage["open_cost"] if b.n_configs]
    assert len(visited) == 1
    assert visited[0].holdout_locked == [42.0]


@pytest.mark.skipif(not Path("registry").exists(),
                    reason="no registry/ checkout")
def test_real_registry_loads_and_dedups():
    registry = Path("registry")
    records: dict[str, gr.AgentRecord] = {}
    gr._load_scoreboards(registry, records)
    gr._load_reevals(registry, records)
    gr._load_register_csvs(registry, records)
    rec_list = [r for r in records.values() if r.genes]
    assert rec_list, "expected at least one persisted gene config"
    coverage, n_configs = gr.build_coverage(rec_list, ["learning_rate"], 8)
    # De-dup by config hash strictly reduces model-count -> config-count.
    assert 0 < n_configs <= len(rec_list)
    # The legacy 7 genes must all be present somewhere in the data.
    seen = set()
    for r in rec_list:
        seen.update(r.genes)
    for g in ("learning_rate", "entropy_coeff", "clip_range", "gae_lambda",
              "value_coeff", "mini_batch_size", "hidden_size"):
        assert g in seen, f"{g} missing from persisted configs"
