"""Tests for Sprint 3 Session 1: mutation cap (Issue 11) + breeding pool (Issue 08).

Kept in a separate file from test_genetic_operators.py to avoid a noisy diff
on the shared helpers; reuses _make_config / _make_score / PARENT_A_HP via
import from the existing test module.
"""

from __future__ import annotations

import logging
import random

from agents.population_manager import PopulationManager, SelectionResult
from registry.model_store import ModelStore

from tests.test_genetic_operators import (
    PARENT_A_HP,
    _make_config,
    _make_score,
)


# ── Issue 11: Mutation count cap ─────────────────────────────────────────────


def _count_mutated(deltas: dict) -> int:
    return sum(1 for v in deltas.values() if v is not None)


class TestMutationCap:
    def test_cap_zero_mutates_none(self):
        pm = PopulationManager(_make_config(), model_store=None)
        hp = dict(PARENT_A_HP)
        original = dict(hp)
        _, deltas = pm.mutate(
            hp, mutation_rate=1.0, rng=random.Random(0), max_mutations=0,
        )
        assert _count_mutated(deltas) == 0
        for spec in pm.hp_specs:
            assert hp[spec.name] == original[spec.name]

    def test_cap_one_mutates_exactly_one(self):
        pm = PopulationManager(_make_config(), model_store=None)
        for seed in range(20):
            hp = dict(PARENT_A_HP)
            _, deltas = pm.mutate(
                hp, mutation_rate=1.0, rng=random.Random(seed), max_mutations=1,
            )
            assert _count_mutated(deltas) == 1, (
                f"seed={seed} produced {_count_mutated(deltas)} mutations"
            )

    def test_cap_two_mutates_exactly_two(self):
        pm = PopulationManager(_make_config(), model_store=None)
        for seed in range(20):
            hp = dict(PARENT_A_HP)
            _, deltas = pm.mutate(
                hp, mutation_rate=1.0, rng=random.Random(seed), max_mutations=2,
            )
            assert _count_mutated(deltas) == 2

    def test_cap_none_uses_coin_flip(self):
        """max_mutations=None preserves the legacy per-gene coin-flip path."""
        pm = PopulationManager(_make_config(), model_store=None)
        hp = dict(PARENT_A_HP)
        _, deltas = pm.mutate(
            hp, mutation_rate=1.0, rng=random.Random(42), max_mutations=None,
        )
        for spec in pm.hp_specs:
            assert deltas[spec.name] is not None

    def test_cap_above_eligible_mutates_all(self):
        pm = PopulationManager(_make_config(), model_store=None)
        n_specs = len(pm.hp_specs)
        hp = dict(PARENT_A_HP)
        _, deltas = pm.mutate(
            hp, mutation_rate=1.0, rng=random.Random(0),
            max_mutations=n_specs * 10,
        )
        assert _count_mutated(deltas) == n_specs

    def test_cap_respects_arch_cooldown(self):
        # Inject architecture_name into the search ranges so the
        # cooldown rule is exercised — the default _make_config does
        # not include it.
        cfg = _make_config()
        cfg["hyperparameters"]["search_ranges"]["architecture_name"] = {
            "type": "str_choice",
            "choices": ["ppo_lstm_v1", "ppo_time_lstm_v1"],
        }
        pm = PopulationManager(cfg, model_store=None)
        spec_names = {s.name for s in pm.hp_specs}
        assert "architecture_name" in spec_names
        n_specs = len(pm.hp_specs)
        for seed in range(20):
            hp = dict(PARENT_A_HP)
            hp["arch_change_cooldown"] = 1
            _, deltas = pm.mutate(
                hp, mutation_rate=1.0, rng=random.Random(seed),
                max_mutations=n_specs,  # ask to mutate everything
            )
            # Architecture cooled down → its delta is None.
            assert deltas["architecture_name"] is None
            # Every other gene was eligible and so mutated.
            assert _count_mutated(deltas) == n_specs - 1

    def test_cap_backfills_missing_keys(self):
        """Missing HP keys must still be backfilled when a cap excludes them."""
        pm = PopulationManager(_make_config(), model_store=None)
        for seed in range(20):
            hp = {k: v for k, v in PARENT_A_HP.items() if k != "mlp_layers"}
            pm.mutate(
                hp, mutation_rate=1.0, rng=random.Random(seed), max_mutations=1,
            )
            assert "mlp_layers" in hp

    def test_cap_legacy_path_rng_identical(self):
        """max_mutations=None must consume the RNG identically to default arg."""
        pm = PopulationManager(_make_config(), model_store=None)
        hp1 = dict(PARENT_A_HP)
        hp2 = dict(PARENT_A_HP)
        pm.mutate(hp1, mutation_rate=0.3, rng=random.Random(123), max_mutations=None)
        pm.mutate(hp2, mutation_rate=0.3, rng=random.Random(123))
        for spec in pm.hp_specs:
            assert hp1[spec.name] == hp2[spec.name]

    def test_breed_passes_cap_through(self, tmp_path):
        """breed(max_mutations=N) caps each child's mutations to N."""
        config = _make_config(pop_size=6, n_elite=2, top_pct=0.5)
        store = ModelStore(str(tmp_path / "test.db"), str(tmp_path / "weights"))
        pm = PopulationManager(config, model_store=store)
        agents = pm.initialise_population(generation=0, seed=42)
        scores = [
            _make_score(agents[i].model_id, composite=(i + 1) * 0.1)
            for i in range(6)
        ]
        result = pm.select(scores)
        children, records = pm.breed(
            result, generation=1, mutation_rate=1.0, seed=99, max_mutations=2,
        )
        for record in records:
            mutated = sum(1 for v in record.deltas.values() if v is not None)
            assert mutated <= 2


# ── Issue 08: Breeding pool scope (external parents) ─────────────────────────


class TestBreedingPool:
    def test_external_parents_dont_take_slots(self, tmp_path):
        """n_children depends on len(survivors), not survivors+external."""
        config = _make_config(pop_size=8, n_elite=2, top_pct=0.5)
        store = ModelStore(str(tmp_path / "test.db"), str(tmp_path / "weights"))
        pm = PopulationManager(config, model_store=store)
        agents = pm.initialise_population(generation=0, seed=42)
        run_survivors = [agents[0].model_id, agents[1].model_id]
        external = [a.model_id for a in agents[2:]]
        scores = [_make_score(a.model_id, composite=0.5) for a in agents]
        result = SelectionResult(
            elites=run_survivors,
            survivors=run_survivors,
            eliminated=[],
            ranked_scores=scores,
            external_parents=external,
        )
        children, _records = pm.breed(
            result, generation=1, mutation_rate=0.3, seed=77,
            external_parent_ids=external,
        )
        # n_children = pop_size - run_survivors (external don't take slots).
        assert len(children) == 8 - 2

    def test_external_parents_can_be_chosen(self, tmp_path):
        """External parents must be reachable from the parent pool."""
        config = _make_config(pop_size=6, n_elite=1, top_pct=0.5)
        store = ModelStore(str(tmp_path / "test.db"), str(tmp_path / "weights"))
        pm = PopulationManager(config, model_store=store)
        agents = pm.initialise_population(generation=0, seed=42)
        run_survivors = [agents[0].model_id]
        external = [a.model_id for a in agents[1:]]
        scores = [_make_score(a.model_id, composite=0.5) for a in agents]
        result = SelectionResult(
            elites=run_survivors,
            survivors=run_survivors,
            eliminated=[],
            ranked_scores=scores,
            external_parents=external,
        )
        _children, records = pm.breed(
            result, generation=1, mutation_rate=0.0, seed=11,
            external_parent_ids=external,
        )
        all_parents = set()
        for r in records:
            all_parents.add(r.parent_a_id)
            all_parents.add(r.parent_b_id)
        assert all_parents.intersection(set(external))

    def test_no_children_warning_logged(self, tmp_path, caplog):
        """A warning must be emitted when n_children <= 0 (hard constraint)."""
        config = _make_config(pop_size=3, n_elite=1, top_pct=1.0)
        store = ModelStore(str(tmp_path / "test.db"), str(tmp_path / "weights"))
        pm = PopulationManager(config, model_store=store)
        agents = pm.initialise_population(generation=0, seed=1)
        scores = [_make_score(a.model_id, composite=0.5) for a in agents]
        result = pm.select(scores)
        # All 3 are survivors → n_children = 0.
        assert len(result.survivors) == 3
        with caplog.at_level(logging.WARNING, logger="agents.population_manager"):
            children, _records = pm.breed(result, generation=1, seed=0)
        assert len(children) == 0
        assert any("no children to breed" in r.message for r in caplog.records)
