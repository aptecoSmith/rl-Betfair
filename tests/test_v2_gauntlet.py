"""Tests for training_v2/cohort/gauntlet.py — the orchestrator (Phase 5).

Stub the executor (``run_tranche_fn``) so the pipeline is tested without real
training: scheduling, ledger advancement, uniform single-tranche batches, the
recipe-purity warm-start wiring, breeding cadence, and resume.
"""
from __future__ import annotations

import random
from pathlib import Path

from training_v2.cohort.breeder import BreedConfig
from training_v2.cohort.executor import TrancheExecConfig, TrancheResult
from training_v2.cohort.gauntlet import (
    GauntletConfig,
    climb_to_frontier,
    run_gauntlet,
    seed_population,
)
from training_v2.cohort.ledger import DaySplit, GauntletLedger


def _split(n_tranches=2):
    return DaySplit(
        tranche_days=[[f"t{k}d{i}" for i in range(2)] for k in range(n_tranches)],
        validation_days=["v1", "v2"],
        final_test_days=["seal1"])


def _exec_cfg(tmp_path):
    return TrancheExecConfig(
        data_dir=tmp_path / "data", output_dir=tmp_path / "out",
        model_store=None, predictor_bundle=None,
        use_race_outcome_predictor=False, parallel_agents=0)


def _make_stub(tmp_path, calls):
    """A run_tranche stand-in: writes a weights file per agent, scores by the
    open_cost gene (deterministic), records the call for assertions."""
    def stub(agents, *, tranche_K, train_days_for_K, validation_days, cfg,
             executor=None):
        calls.append({"K": tranche_K, "agents": list(agents),
                      "train_days": list(train_days_for_K)})
        out = []
        for a in agents:
            wp = str(tmp_path / f"{a.lineage_id}_K{tranche_K}.pt")
            Path(wp).write_bytes(b"x")
            score = float(a.genes.open_cost)  # higher open_cost == better (stub)
            out.append(TrancheResult(
                agent_id=a.agent_id, lineage_id=a.lineage_id,
                tranche_K=tranche_K, weights_path=wp, result=None,
                validation_locked=score, validation_naked=-1.0,
                validation_day_pnl=0.0, composite_score=score))
        return out
    return stub


def test_seed_population_into_needs_t1(tmp_path):
    led = GauntletLedger(tmp_path / "l.jsonl")
    cfg = GauntletConfig(n_recipes=5)
    n = seed_population(led, cfg, random.Random(0))
    assert n == 5
    assert len(led.needs(1)) == 5
    # Idempotent — won't reseed a non-empty ledger.
    assert seed_population(led, cfg, random.Random(0)) == 0


def test_climb_reaches_frontier_uniform_batches(tmp_path):
    led = GauntletLedger(tmp_path / "l.jsonl")
    seed_population(led, GauntletConfig(n_recipes=4), random.Random(0))
    calls = []
    runs = climb_to_frontier(led, _split(2), _exec_cfg(tmp_path),
                             run_tranche_fn=_make_stub(tmp_path, calls))
    # Two tranches -> two uniform runs (one per depth).
    assert runs == 2
    # Every run is a SINGLE tranche depth (uniform cost).
    assert [c["K"] for c in calls] == [1, 2]
    # All lineages reached the frontier depth.
    assert led.frontier_depth() == 2
    assert len(led.frontier()) == 4


def test_warm_start_wiring_is_recipe_pure(tmp_path):
    led = GauntletLedger(tmp_path / "l.jsonl")
    seed_population(led, GauntletConfig(n_recipes=3), random.Random(0))
    calls = []
    climb_to_frontier(led, _split(2), _exec_cfg(tmp_path),
                      run_tranche_fn=_make_stub(tmp_path, calls))
    # K=1: fresh start, no inherited weights.
    for a in calls[0]["agents"]:
        assert a.init_weights_path is None
    # K=2: each agent warm-starts ITS OWN lineage's K=1 weights.
    for a in calls[1]["agents"]:
        assert a.init_weights_path == str(tmp_path / f"{a.lineage_id}_K1.pt")


def test_train_days_are_the_tranche_for_K(tmp_path):
    led = GauntletLedger(tmp_path / "l.jsonl")
    seed_population(led, GauntletConfig(n_recipes=2), random.Random(0))
    calls = []
    split = _split(2)
    climb_to_frontier(led, split, _exec_cfg(tmp_path),
                      run_tranche_fn=_make_stub(tmp_path, calls))
    assert calls[0]["train_days"] == split.train_days_for(1)
    assert calls[1]["train_days"] == split.train_days_for(2)


def test_run_gauntlet_with_breeding_keeps_population_climbing(tmp_path):
    led = run_gauntlet(
        split=_split(2),
        exec_cfg=_exec_cfg(tmp_path),
        cfg=GauntletConfig(n_recipes=4, max_breed_rounds=1,
                           breed=BreedConfig(keep_fraction=0.5, min_quorum=2)),
        ledger_path=tmp_path / "l.jsonl",
        rng=random.Random(0),
        run_tranche_fn=_make_stub(tmp_path, []))
    # After climb + 1 breed round + re-climb, the frontier is full depth and
    # the bred replacements have climbed the WHOLE gauntlet too (full fair shot).
    assert led.frontier_depth() == 2
    active = [e for e in led.all_entries() if e.status == "active"]
    # 2 survivors + 2 freshly-bred (now also at depth 2) = 4 active at frontier.
    assert all(e.tranches_completed == 2 for e in active)
    assert len(active) == 4
    # Some lineages were culled (the breeding happened).
    culled = [e for e in led.all_entries() if e.status == "culled"]
    assert len(culled) == 2


def test_failed_tranche_culls_lineage(tmp_path):
    led = GauntletLedger(tmp_path / "l.jsonl")
    seed_population(led, GauntletConfig(n_recipes=3), random.Random(0))

    def stub(agents, *, tranche_K, train_days_for_K, validation_days, cfg,
             executor=None):
        out = []
        for i, a in enumerate(agents):
            if i == 0 and tranche_K == 1:  # first agent's worker "died" at T1
                out.append(TrancheResult(
                    agent_id=a.agent_id, lineage_id=a.lineage_id,
                    tranche_K=tranche_K, weights_path="", result=None,
                    validation_locked=float("nan"), validation_naked=float("nan"),
                    validation_day_pnl=float("nan"), composite_score=float("nan"),
                    error="worker died"))
            else:
                wp = str(tmp_path / f"{a.lineage_id}_K{tranche_K}.pt")
                Path(wp).write_bytes(b"x")
                out.append(TrancheResult(
                    agent_id=a.agent_id, lineage_id=a.lineage_id,
                    tranche_K=tranche_K, weights_path=wp, result=None,
                    validation_locked=1.0, validation_naked=-1.0,
                    validation_day_pnl=0.0, composite_score=1.0))
        return out

    climb_to_frontier(led, _split(2), _exec_cfg(tmp_path), run_tranche_fn=stub)
    culled = [e for e in led.all_entries() if e.status == "culled"]
    assert len(culled) == 1  # the failed lineage was culled, not crashed on


def test_resume_continues_from_ledger(tmp_path):
    path = tmp_path / "l.jsonl"
    split = _split(2)
    cfg = GauntletConfig(n_recipes=4, max_breed_rounds=0)
    # First run climbs to frontier.
    run_gauntlet(split=split, exec_cfg=_exec_cfg(tmp_path), cfg=cfg,
                 ledger_path=path, rng=random.Random(0),
                 run_tranche_fn=_make_stub(tmp_path, []))
    # Re-load: state persisted, frontier at depth 2, nothing left to climb.
    led2 = GauntletLedger.load(path)
    assert led2.frontier_depth() == 2
    calls = []
    runs = climb_to_frontier(led2, split, _exec_cfg(tmp_path),
                             run_tranche_fn=_make_stub(tmp_path, calls))
    assert runs == 0  # already at frontier, no work
