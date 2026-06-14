"""Tests for training_v2/cohort/breeder.py — gauntlet selection + mutation."""
from __future__ import annotations

import random

from training_v2.cohort.breeder import BreedConfig, breed_frontier, genes_from_dict
from training_v2.cohort.executor import config_hash
from training_v2.cohort.genes import CohortGenes
from training_v2.cohort.ledger import GauntletLedger


def _genes(**kw):
    base = dict(learning_rate=3e-4, entropy_coeff=1e-3, clip_range=0.2,
                gae_lambda=0.95, value_coeff=0.5, mini_batch_size=64,
                hidden_size=128)
    base.update(kw)
    return CohortGenes(**base)


def _ledger_with_frontier(tmp_path, n, depth=1, scores=None):
    """Build a ledger with n lineages all at `depth`, given validation scores."""
    led = GauntletLedger(tmp_path / "ledger.jsonl")
    scores = scores or list(range(n))
    ids = []
    for i in range(n):
        g = _genes(open_cost=float(i))
        e = led.add_recipe(g, origin="fresh", config_hash=config_hash(g),
                           lineage_id=f"L{i}")
        for k in range(1, depth + 1):
            led.record_tranche(e.lineage_id, k, weights_path=f"w{i}_{k}.pt",
                               composite=float(scores[i]), locked=float(scores[i]),
                               naked=0.0)
        ids.append(e.lineage_id)
    return led, ids


def test_below_quorum_skips(tmp_path):
    led, _ = _ledger_with_frontier(tmp_path, 1, depth=1)
    res = breed_frontier(led, random.Random(0),
                         cfg=BreedConfig(min_quorum=2))
    assert res.bred is False


def test_keeps_top_fraction_culls_rest(tmp_path):
    led, ids = _ledger_with_frontier(tmp_path, 4, depth=1,
                                     scores=[10, 1, 8, 2])
    res = breed_frontier(led, random.Random(0),
                         cfg=BreedConfig(keep_fraction=0.5, min_quorum=2))
    assert res.bred is True
    # Top 2 by composite are L0 (10) and L2 (8).
    assert set(res.survivors) == {"L0", "L2"}
    assert set(res.culled) == {"L1", "L3"}
    # Survivors stay active; culled are marked culled.
    assert led.get("L0").status == "active"
    assert led.get("L1").status == "culled"


def test_replacements_land_in_needs_t1(tmp_path):
    led, _ = _ledger_with_frontier(tmp_path, 4, depth=1, scores=[10, 1, 8, 2])
    res = breed_frontier(led, random.Random(0),
                         cfg=BreedConfig(keep_fraction=0.5, min_quorum=2))
    # One replacement per culled slot, all at depth 0 (needs-T1).
    assert len(res.new_lineage_ids) == len(res.culled)
    needs1 = {e.lineage_id for e in led.needs(1)}
    assert set(res.new_lineage_ids) <= needs1
    for lid in res.new_lineage_ids:
        assert led.get(lid).tranches_completed == 0


def test_mutants_carry_parent_lineage(tmp_path):
    led, _ = _ledger_with_frontier(tmp_path, 4, depth=1, scores=[10, 1, 8, 2])
    res = breed_frontier(led, random.Random(1),
                         cfg=BreedConfig(keep_fraction=0.5, min_quorum=2,
                                         mutant_fraction=1.0))
    for lid in res.new_lineage_ids:
        e = led.get(lid)
        assert e.origin == "mutant"
        assert e.parent_lineage_id in res.survivors


def test_fresh_blood_when_mutant_fraction_zero(tmp_path):
    led, _ = _ledger_with_frontier(tmp_path, 4, depth=1, scores=[10, 1, 8, 2])
    res = breed_frontier(led, random.Random(2),
                         cfg=BreedConfig(keep_fraction=0.5, min_quorum=2,
                                         mutant_fraction=0.0))
    for lid in res.new_lineage_ids:
        e = led.get(lid)
        assert e.origin == "fresh"
        assert e.parent_lineage_id is None


def test_full_fair_shot_shallow_climbers_untouched(tmp_path):
    """A lineage still climbing at a shallower depth is NEVER culled, even with
    the worst score — only the frontier is eligible."""
    led = GauntletLedger(tmp_path / "ledger.jsonl")
    # Two deep (depth 2), one shallow (depth 1) with a terrible score.
    for i, depth, sc in [(0, 2, 9.0), (1, 2, 8.0), (2, 1, -999.0)]:
        g = _genes(open_cost=float(i))
        e = led.add_recipe(g, origin="fresh", config_hash=config_hash(g),
                           lineage_id=f"L{i}")
        for k in range(1, depth + 1):
            led.record_tranche(e.lineage_id, k, weights_path=f"w{i}_{k}.pt",
                               composite=sc, locked=sc, naked=0.0)
    res = breed_frontier(led, random.Random(0),
                         cfg=BreedConfig(keep_fraction=0.5, min_quorum=2))
    assert res.depth == 2
    # The shallow lineage L2 is at depth 1 — not in the frontier, never culled.
    assert "L2" not in res.culled
    assert led.get("L2").status == "active"


def test_sigma_leg_ceiling_filters(tmp_path):
    led, _ = _ledger_with_frontier(tmp_path, 4, depth=1, scores=[10, 9, 8, 7])
    # L0 has the best composite but a blown σ_leg (£50 > £30 ceiling).
    sigma = {"L0": 50.0, "L1": 10.0, "L2": 12.0, "L3": 11.0}

    def sigma_leg_fn(e):
        return sigma[e.lineage_id]

    res = breed_frontier(led, random.Random(0),
                         cfg=BreedConfig(keep_fraction=0.5, min_quorum=2,
                                         sigma_leg_ceiling=30.0),
                         sigma_leg_fn=sigma_leg_fn)
    # L0 is filtered out by the ceiling; survivors are the next-best within σ.
    assert "L0" not in res.survivors
    assert set(res.survivors) == {"L1", "L2"}


def test_genes_from_dict_drops_legacy_keys():
    d = _genes().to_dict()
    d["arb_spread_scale"] = 1.2  # a dead legacy gene from an old era
    g = genes_from_dict(d)
    assert isinstance(g, CohortGenes)
    assert not hasattr(g, "arb_spread_scale")


def test_deterministic_given_rng(tmp_path):
    led1, _ = _ledger_with_frontier(tmp_path / "a", 4, depth=1,
                                    scores=[10, 1, 8, 2])
    led2, _ = _ledger_with_frontier(tmp_path / "b", 4, depth=1,
                                    scores=[10, 1, 8, 2])
    r1 = breed_frontier(led1, random.Random(7), cfg=BreedConfig(min_quorum=2))
    r2 = breed_frontier(led2, random.Random(7), cfg=BreedConfig(min_quorum=2))
    g1 = [led1.get(lid).config_hash for lid in r1.new_lineage_ids]
    g2 = [led2.get(lid).config_hash for lid in r2.new_lineage_ids]
    assert g1 == g2
