"""Guard: the cull-early gauntlet (climb_cull_per_tranche) — the "tick".

Operator 2026-06-19: cull AFTER EACH tranche (successive-halving), mutate
survivors to refill, and re-climb the mutants T1..TK so they rejoin the pool at
depth K. Contrast climb_to_frontier (full fair shot — no mid-climb culling).

Verified with a fake run_tranche_fn (no training): population dynamics, the
catch-up (mutants climb every prior tranche), and per-tranche leaderboards.
"""
from __future__ import annotations

import hashlib
import random
from types import SimpleNamespace

from training_v2.cohort.breeder import BreedConfig
from training_v2.cohort.gauntlet import (
    GauntletConfig,
    climb_cull_per_tranche,
    seed_population,
)
from training_v2.cohort.ledger import DaySplit, GauntletLedger


def _split(n_tranches=3):
    return DaySplit(
        tranche_days=[[f"t{t}d{i}" for i in range(2)] for t in range(n_tranches)],
        validation_days=["v1", "v2"],
        final_test_days=[],
    )


def _quality(lineage_id: str) -> float:
    # Deterministic per-lineage "held-out locked" so ranking is stable.
    return int(hashlib.md5(lineage_id.encode()).hexdigest()[:6], 16) / 1e6


def _make_fake_runner(calls: list):
    def fake(agents, *, tranche_K, train_days_for_K, validation_days, cfg, executor):
        calls.append((tranche_K, [a.lineage_id for a in agents]))
        out = []
        for a in agents:
            q = _quality(a.lineage_id)
            out.append(SimpleNamespace(
                error=None, weights_path=f"w/{a.lineage_id}/{tranche_K}",
                result=SimpleNamespace(quality=q), validation_locked=q,
                validation_naked=0.0, agent_id=a.agent_id))
        return out
    return fake


def _run(tmp_path, n_tranches=3, pool=4):
    led = GauntletLedger(tmp_path / "ledger.jsonl")
    split = _split(n_tranches)
    led.set_split(split)
    rng = random.Random(0)
    cfg = GauntletConfig(n_recipes=pool, enabled_set=frozenset())
    seed_population(led, cfg, rng)
    calls: list = []
    breed = BreedConfig(keep_fraction=0.5, min_quorum=2,
                        enabled_set=frozenset(), mutant_fraction=1.0)
    runs = climb_cull_per_tranche(
        led, split, exec_cfg=None, rng=rng, seed_base=1,
        run_tranche_fn=_make_fake_runner(calls),
        score_result_fn=lambda res: res.quality, breed_cfg=breed)
    return led, calls, runs


def test_pool_maintained_and_culls_every_tranche(tmp_path):
    led, calls, runs = _run(tmp_path, n_tranches=3, pool=4)
    # Pool of 4, keep 2 / cull 2 each tranche → frontier always refilled to 4.
    assert len(led.frontier(3)) == 4, "frontier pool not refilled to P at depth N"
    culled = [e for e in led.all_entries() if e.status == "culled"]
    assert len(culled) == 6, "expected 2 culled per tranche × 3 tranches"
    # 4 seed + 2 refills × 3 tranches = 10 lineages total.
    assert len(led.all_entries()) == 10


def test_catch_up_climbs_every_prior_tranche(tmp_path):
    # A mutant born after tranche K must record a score at EVERY depth 1..K
    # (it re-climbed the whole ladder). record_tranche enforces sequential
    # completion, so a skipped catch-up tranche would have raised.
    led, calls, runs = _run(tmp_path, n_tranches=3, pool=4)
    for e in led.frontier(3):  # the survivors + caught-up mutants at the frontier
        assert set(e.validation_score) == {"1", "2", "3"}, (
            f"lineage {e.lineage_id} missing a tranche score: {e.validation_score}")
        assert e.tranches_completed == 3


def test_per_tranche_leaderboards_recorded(tmp_path):
    led, calls, runs = _run(tmp_path, n_tranches=3, pool=4)
    # Per-depth leaderboard sizes. Each tranche K: 4 ran the main batch; each
    # breed emits 2 mutants that catch up through every prior depth. So:
    #   depth1 = 4(seed) + 2(K1) + 2(K2) + 2(K3) = 10
    #   depth2 = 4(K2 main) + 2(K2 catch-up) + 2(K3 catch-up) = 8
    #   depth3 = 4(K3 main) + 2(K3 catch-up) = 6
    by_depth = {1: 0, 2: 0, 3: 0}
    for e in led.all_entries():
        for k in e.validation_score:
            by_depth[int(k)] += 1
    assert by_depth == {1: 10, 2: 8, 3: 6}
    # Run count: K=1 → main+1 catch-up; K=2 → main+2; K=3 → main+3 = 2+3+4 = 9.
    assert runs == 9
    assert len(calls) == 9


def test_stop_and_resume_completes_without_double_culling(tmp_path):
    # The architecture must be stop/resumable: kill mid-training, reload the
    # ledger from disk, continue → same final state, and crucially NO depth is
    # bred twice (the bred-depths marker is the idempotency guard).
    split = _split(3)
    ledger_path = tmp_path / "ledger.jsonl"

    # Run 1: crash after a handful of tranche runs (simulates a mid-training kill).
    led1 = GauntletLedger(ledger_path)
    led1.set_split(split)
    seed_population(led1, GauntletConfig(n_recipes=4, enabled_set=frozenset()),
                    random.Random(0))
    calls1: list = []
    fake1 = _make_fake_runner(calls1)

    def crashing(*a, **k):
        if len(calls1) >= 4:           # die partway through (during a drain)
            raise RuntimeError("simulated mid-training kill")
        return fake1(*a, **k)

    breed = BreedConfig(keep_fraction=0.5, min_quorum=2,
                        enabled_set=frozenset(), mutant_fraction=1.0)
    try:
        climb_cull_per_tranche(led1, split, exec_cfg=None, rng=random.Random(0),
                               seed_base=1, run_tranche_fn=crashing,
                               score_result_fn=lambda r: r.quality, breed_cfg=breed)
    except RuntimeError:
        pass
    assert len(calls1) == 4  # confirmed it died mid-way

    # Run 2: RELOAD the ledger from disk (the resume) and finish cleanly.
    led2 = GauntletLedger.load(ledger_path)
    assert led2.split is not None  # split persisted + reloaded
    calls2: list = []
    climb_cull_per_tranche(led2, led2.split, exec_cfg=None, rng=random.Random(0),
                           seed_base=1, run_tranche_fn=_make_fake_runner(calls2),
                           score_result_fn=lambda r: r.quality, breed_cfg=breed)

    # Final state identical to an uninterrupted run, and no depth double-bred.
    assert led2.bred_depths() == {1, 2, 3}
    assert len(led2.frontier(3)) == 4
    culled = [e for e in led2.all_entries() if e.status == "culled"]
    assert len(culled) == 6, "double-cull on resume would push this above 6"
    assert len(led2.all_entries()) == 10


def test_depth1_cull_is_truncation_on_score(tmp_path):
    # The depth-1 elimination is truncation on score: among the 4 SEEDS (the
    # only fresh-origin lineages — refills are mutants), the 2 with the higher
    # depth-1 score advanced (completed >= 2) and the 2 lower were culled at
    # depth 1. (breed_frontier's truncation has its own unit tests; this
    # confirms it fires per-tranche on the same-depth seed pool.)
    led, _calls, _runs = _run(tmp_path, n_tranches=3, pool=4)
    seeds = [e for e in led.all_entries() if e.origin == "fresh"]
    assert len(seeds) == 4
    culled = [e for e in seeds if e.status == "culled" and e.tranches_completed == 1]
    advanced = [e for e in seeds if e.tranches_completed >= 2]
    assert len(culled) == 2 and len(advanced) == 2
    assert max(e.validation_score["1"] for e in culled) <= \
        min(e.validation_score["1"] for e in advanced) + 1e-9
