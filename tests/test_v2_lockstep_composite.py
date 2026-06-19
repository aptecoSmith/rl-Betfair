"""Maturation-aware composite + the lockstep tock (band-seed) flow.

Guards plans/lockstep-cohort/ step 4 (the `locked_maturation` GA selector +
breeding-aware defaults) and the tock path (a lockstep era seeded at gen 0 with
`--seed-gene` bands, then concentrating via truncation+mutation).
"""
from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from training_v2.cohort.lockstep import init_lockstep_population
from training_v2.cohort.runner import (
    COMPOSITE_SCORE_MODE_LOCKED_MATURATION,
    COMPOSITE_SCORE_MODE_LOCKED_PER_STD,
    COMPOSITE_SCORE_MODE_LOCKED_WEIGHTED,
    COMPOSITE_SCORE_MODE_TOTAL_REWARD,
    _composite_score_base,
    _resolve_composite_defaults,
    _resolve_seed_bands,
)


# ── locked_maturation formula ──────────────────────────────────────────────

def test_locked_maturation_formula():
    es = SimpleNamespace(locked_pnl=10.0, naked_pnl=-4.0, arbs_completed=5)
    # locked + 0.25·naked + w·arbs_completed = 10 + (-1) + 2·5 = 19
    score = _composite_score_base(
        es, 2.0, COMPOSITE_SCORE_MODE_LOCKED_MATURATION)
    assert score == pytest.approx(19.0)


def test_locked_maturation_weight_zero_equals_locked_weighted():
    es = SimpleNamespace(locked_pnl=10.0, naked_pnl=-4.0, arbs_completed=5)
    a = _composite_score_base(es, 0.0, COMPOSITE_SCORE_MODE_LOCKED_MATURATION)
    b = _composite_score_base(es, 0.0, COMPOSITE_SCORE_MODE_LOCKED_WEIGHTED)
    assert a == b == pytest.approx(9.0)  # 10 + 0.25·(-4)


def test_locked_maturation_ignores_agent_closed():
    # Two agents, same locked/naked, same matured count, DIFFERENT agent-closed:
    # the maturation term must NOT move (agent-closed is not maturation).
    base = dict(locked_pnl=5.0, naked_pnl=0.0, arbs_completed=3)
    es1 = SimpleNamespace(**base, arbs_closed=0)
    es2 = SimpleNamespace(**base, arbs_closed=40)
    s1 = _composite_score_base(es1, 1.0, COMPOSITE_SCORE_MODE_LOCKED_MATURATION)
    s2 = _composite_score_base(es2, 1.0, COMPOSITE_SCORE_MODE_LOCKED_MATURATION)
    assert s1 == s2 == pytest.approx(5.0 + 3.0)


# ── breeding-aware defaults (the _resolve_<knob> both-sources test) ─────────

def test_resolve_composite_defaults_lockstep_is_maturation_aware():
    assert _resolve_composite_defaults("lockstep", None, None) == (
        COMPOSITE_SCORE_MODE_LOCKED_MATURATION, 1.0)


def test_resolve_composite_defaults_ga_pbt_byte_identical():
    assert _resolve_composite_defaults("ga", None, None) == (
        COMPOSITE_SCORE_MODE_TOTAL_REWARD, 0.0)
    assert _resolve_composite_defaults("pbt", None, None) == (
        COMPOSITE_SCORE_MODE_TOTAL_REWARD, 0.0)


def test_resolve_composite_defaults_gauntlet_is_scalper_selector():
    # Gauntlet defaults to the scalper selector (locked_per_std = tnv2,
    # locked/(1+σ_naked), never reads naked-sign) so breeding can't be driven
    # by naked luck. weight 0.0 (locked_per_std ignores maturation weight).
    assert _resolve_composite_defaults("gauntlet", None, None) == (
        COMPOSITE_SCORE_MODE_LOCKED_PER_STD, 0.0)
    # Explicit CLI value still overrides the gauntlet default.
    assert _resolve_composite_defaults("gauntlet", "locked_weighted", None) == (
        COMPOSITE_SCORE_MODE_LOCKED_WEIGHTED, 0.0)


def test_resolve_composite_defaults_explicit_cli_wins():
    # Explicit mode + weight override the breeding default, both directions.
    assert _resolve_composite_defaults("lockstep", "locked_weighted", 3.0) == (
        "locked_weighted", 3.0)
    # Explicit weight 0 on lockstep is respected (not bumped to 1.0).
    assert _resolve_composite_defaults("lockstep", None, 0.0) == (
        COMPOSITE_SCORE_MODE_LOCKED_MATURATION, 0.0)
    assert _resolve_composite_defaults("ga", "sortino", None) == (
        "sortino", 0.0)


# ── tock flow: band-seed allowed for lockstep, concentrates gen 0 ──────────

def test_seed_gene_allowed_for_lockstep_rejected_for_ga():
    # Band within clip_range's declared range [0.1, 0.3].
    bands, _enabled = _resolve_seed_bands(
        ["clip_range=0.25:0.29"], breeding="lockstep",
        reward_overrides={}, enabled_set=frozenset())
    assert bands["clip_range"] == (0.25, 0.29)
    # The gene-only GA path still refuses --seed-gene.
    with pytest.raises(SystemExit):
        _resolve_seed_bands(
            ["clip_range=0.25:0.29"], breeding="ga",
            reward_overrides={}, enabled_set=frozenset())


def test_lockstep_tock_seeds_gen0_within_band():
    # A tock specifies the configs that go in at t1: gen-0 fresh blood is drawn
    # INSIDE the seeded band (then concentrates via truncation+mutation).
    pop = init_lockstep_population(
        random.Random(1), 8, seed_bands={"clip_range": (0.25, 0.29)})
    assert len(pop) == 8
    assert all(s.origin == "fresh" for s in pop)
    assert all(0.25 <= s.genes.clip_range <= 0.29 for s in pop), \
        [s.genes.clip_range for s in pop]
