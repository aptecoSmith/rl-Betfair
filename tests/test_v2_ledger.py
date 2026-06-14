"""Tests for training_v2/cohort/ledger.py — gauntlet ledger + queues (Phase 3)."""
from __future__ import annotations

import pytest

from training_v2.cohort.executor import config_hash
from training_v2.cohort.genes import CohortGenes
from training_v2.cohort.ledger import (
    DaySplit,
    GauntletLedger,
    assert_day_split_disjoint,
)


def _genes(**kw):
    base = dict(learning_rate=3e-4, entropy_coeff=1e-3, clip_range=0.2,
                gae_lambda=0.95, value_coeff=0.5, mini_batch_size=64,
                hidden_size=128)
    base.update(kw)
    return CohortGenes(**base)


def _split():
    return DaySplit(
        tranche_days=[["d1", "d2"], ["d3", "d4"]],
        validation_days=["v1", "v2"],
        final_test_days=["t1"])


# ── day split / leakage ────────────────────────────────────────────────────


def test_day_split_disjoint_ok():
    assert_day_split_disjoint(_split())  # no raise


def test_day_split_validation_train_leak_raises():
    bad = DaySplit(tranche_days=[["d1", "v1"]], validation_days=["v1"])
    with pytest.raises(AssertionError):
        assert_day_split_disjoint(bad)


def test_day_split_final_test_leak_raises():
    bad = DaySplit(tranche_days=[["d1"]], validation_days=["v1"],
                   final_test_days=["d1"])
    with pytest.raises(AssertionError):
        assert_day_split_disjoint(bad)


def test_train_days_for_is_1_indexed():
    s = _split()
    assert s.train_days_for(1) == ["d1", "d2"]
    assert s.train_days_for(2) == ["d3", "d4"]
    with pytest.raises(IndexError):
        s.train_days_for(3)


# ── queues / frontier ──────────────────────────────────────────────────────


def test_add_recipe_lands_in_needs_t1(tmp_path):
    led = GauntletLedger(tmp_path / "ledger.jsonl")
    g = _genes()
    led.add_recipe(g, origin="fresh", config_hash=config_hash(g))
    assert len(led.needs(1)) == 1
    assert led.needs(2) == []
    assert led.frontier_depth() == 0
    assert led.frontier() == []  # depth 0 -> no frontier yet


def test_record_tranche_advances_queue(tmp_path):
    led = GauntletLedger(tmp_path / "ledger.jsonl")
    g = _genes()
    e = led.add_recipe(g, origin="fresh", config_hash=config_hash(g))
    led.record_tranche(e.lineage_id, 1, weights_path="w1.pt", composite=2.0,
                       locked=5.0, naked=-3.0)
    # Now it needs T2, not T1.
    assert led.needs(1) == []
    assert [x.lineage_id for x in led.needs(2)] == [e.lineage_id]
    assert led.frontier_depth() == 1
    assert [x.lineage_id for x in led.frontier()] == [e.lineage_id]


def test_record_tranche_rejects_skip(tmp_path):
    led = GauntletLedger(tmp_path / "ledger.jsonl")
    g = _genes()
    e = led.add_recipe(g, origin="fresh", config_hash=config_hash(g))
    # Can't jump to tranche 2 before completing 1.
    with pytest.raises(ValueError):
        led.record_tranche(e.lineage_id, 2, weights_path="w.pt", composite=1.0,
                           locked=0.0, naked=0.0)


def test_culled_lineage_leaves_active_queues(tmp_path):
    led = GauntletLedger(tmp_path / "ledger.jsonl")
    g = _genes()
    e = led.add_recipe(g, origin="fresh", config_hash=config_hash(g))
    led.set_status(e.lineage_id, "culled")
    assert led.needs(1) == []
    assert led.needs(1, active_only=False)  # still visible when asked


def test_frontier_is_same_depth_only(tmp_path):
    led = GauntletLedger(tmp_path / "ledger.jsonl")
    deep = led.add_recipe(_genes(open_cost=1.0), origin="fresh",
                          config_hash="h1")
    shallow = led.add_recipe(_genes(open_cost=2.0), origin="fresh",
                             config_hash="h2")
    led.record_tranche(deep.lineage_id, 1, weights_path="w.pt", composite=1.0,
                       locked=1.0, naked=0.0)
    led.record_tranche(deep.lineage_id, 2, weights_path="w2.pt", composite=2.0,
                       locked=2.0, naked=0.0)
    # frontier depth is 2; only the deep lineage is in it (shallow at depth 0).
    assert led.frontier_depth() == 2
    fr = led.frontier()
    assert [x.lineage_id for x in fr] == [deep.lineage_id]
    assert shallow.tranches_completed == 0


# ── persistence / resume ───────────────────────────────────────────────────


def test_resume_round_trips_state(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = GauntletLedger(path, split=_split())
    led.set_split(_split())
    g = _genes()
    e = led.add_recipe(g, origin="fresh", config_hash=config_hash(g))
    led.record_tranche(e.lineage_id, 1, weights_path="w1.pt", composite=2.5,
                       locked=5.0, naked=-3.0, agent_id="aid1")

    led2 = GauntletLedger.load(path)
    assert led2.split is not None
    assert led2.split.validation_days == ["v1", "v2"]
    e2 = led2.get(e.lineage_id)
    assert e2.tranches_completed == 1
    assert e2.weights_path == "w1.pt"
    assert e2.validation_locked["1"] == 5.0
    assert e2.last_agent_id == "aid1"
    # The resumed ledger derives the same queue.
    assert [x.lineage_id for x in led2.needs(2)] == [e.lineage_id]


def test_compact_preserves_state(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = GauntletLedger(path, split=_split())
    led.set_split(_split())
    for i in range(3):
        g = _genes(open_cost=float(i))
        led.add_recipe(g, origin="fresh", config_hash=config_hash(g),
                       lineage_id=f"L{i}")
    led.record_tranche("L0", 1, weights_path="w.pt", composite=1.0,
                       locked=1.0, naked=0.0)
    n_lines_before = path.read_text().count("\n")
    led.compact()
    n_lines_after = path.read_text().count("\n")
    # Compaction shrinks the log (1 meta + 3 entries = 4 lines).
    assert n_lines_after == 4
    assert n_lines_after < n_lines_before
    led2 = GauntletLedger.load(path)
    assert led2.get("L0").tranches_completed == 1
    assert led2.frontier_depth() == 1
