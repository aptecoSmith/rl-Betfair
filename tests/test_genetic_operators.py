"""Unit tests for genetic operators (crossover, mutation, breeding, logging)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from agents.population_manager import (
    BreedingRecord,
    PopulationManager,
    SelectionResult,
    parse_search_ranges,
    validate_hyperparams,
)
from registry.model_store import GeneticEventRecord, ModelStore
from registry.scoreboard import ModelScore


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_config(pop_size=6, n_elite=2, top_pct=0.5, logs_dir=None):
    cfg = {
        "population": {
            "size": pop_size,
            "n_elite": n_elite,
            "selection_top_pct": top_pct,
            "mutation_rate": 0.3,
        },
        "training": {
            "architecture": "ppo_lstm_v1",
            "max_runners": 14,
            "starting_budget": 100.0,
        },
        "hyperparameters": {
            "search_ranges": {
                "learning_rate": {"type": "float_log", "min": 1e-5, "max": 5e-4},
                "ppo_clip_epsilon": {"type": "float", "min": 0.1, "max": 0.3},
                "entropy_coefficient": {"type": "float", "min": 0.001, "max": 0.05},
                "lstm_hidden_size": {"type": "int_choice", "choices": [64, 128, 256, 512, 1024, 2048]},
                "mlp_hidden_size": {"type": "int_choice", "choices": [64, 128, 256]},
                "mlp_layers": {"type": "int", "min": 1, "max": 3},
                "observation_window_ticks": {"type": "int", "min": 3, "max": 360},
                "reward_early_pick_bonus": {"type": "float", "min": 1.0, "max": 1.5},
                "reward_efficiency_penalty": {"type": "float", "min": 0.001, "max": 0.05},
                "reward_precision_bonus": {"type": "float", "min": 0.0, "max": 3.0},
            }
        },
        "discard_policy": {
            "min_win_rate": 0.35,
            "min_mean_pnl": 0.0,
            "min_sharpe": -0.5,
        },
    }
    if logs_dir:
        cfg["paths"] = {"logs": str(logs_dir)}
    return cfg


def _make_score(model_id, composite, win_rate=0.5, mean_daily_pnl=5.0, sharpe=1.0):
    return ModelScore(
        model_id=model_id,
        win_rate=win_rate,
        mean_daily_pnl=mean_daily_pnl,
        sharpe=sharpe,
        bet_precision=0.5,
        pnl_per_bet=1.0,
        efficiency=0.5,
        composite_score=composite,
        test_days=10,
        profitable_days=int(win_rate * 10),
    )


PARENT_A_HP = {
    "learning_rate": 1e-4,
    "ppo_clip_epsilon": 0.2,
    "entropy_coefficient": 0.01,
    "lstm_hidden_size": 256,
    "mlp_hidden_size": 128,
    "mlp_layers": 2,
    "observation_window_ticks": 60,
    "reward_early_pick_bonus": 1.3,
    "reward_efficiency_penalty": 0.01,
    "reward_precision_bonus": 1.0,
    "architecture_name": "ppo_lstm_v1",
}

PARENT_B_HP = {
    "learning_rate": 3e-4,
    "ppo_clip_epsilon": 0.15,
    "entropy_coefficient": 0.03,
    "lstm_hidden_size": 512,
    "mlp_hidden_size": 64,
    "mlp_layers": 1,
    "observation_window_ticks": 120,
    "reward_early_pick_bonus": 1.1,
    "reward_efficiency_penalty": 0.03,
    "reward_precision_bonus": 2.0,
    "architecture_name": "ppo_lstm_v1",
}


# ── Crossover ────────────────────────────────────────────────────────────────


class TestCrossover:
    def test_child_has_all_params(self):
        pm = PopulationManager(_make_config(), model_store=None)
        child, inh = pm.crossover(PARENT_A_HP, PARENT_B_HP, rng=random.Random(42))
        for spec in pm.hp_specs:
            assert spec.name in child

    def test_each_param_from_one_parent(self):
        pm = PopulationManager(_make_config(), model_store=None)
        child, inh = pm.crossover(PARENT_A_HP, PARENT_B_HP, rng=random.Random(42))
        for spec in pm.hp_specs:
            name = spec.name
            assert inh[name] in ("A", "B")
            if inh[name] == "A":
                assert child[name] == PARENT_A_HP[name]
            else:
                assert child[name] == PARENT_B_HP[name]

    def test_architecture_inherited(self):
        pm = PopulationManager(_make_config(), model_store=None)
        child, inh = pm.crossover(PARENT_A_HP, PARENT_B_HP, rng=random.Random(42))
        assert "architecture_name" in child
        assert inh["architecture_name"] in ("A", "B")

    def test_reproducibility(self):
        pm = PopulationManager(_make_config(), model_store=None)
        c1, _ = pm.crossover(PARENT_A_HP, PARENT_B_HP, rng=random.Random(99))
        c2, _ = pm.crossover(PARENT_A_HP, PARENT_B_HP, rng=random.Random(99))
        assert c1 == c2

    def test_mix_of_parents(self):
        """Over many trials, both parents should contribute."""
        pm = PopulationManager(_make_config(), model_store=None)
        a_count = 0
        b_count = 0
        for seed in range(100):
            _, inh = pm.crossover(PARENT_A_HP, PARENT_B_HP, rng=random.Random(seed))
            for v in inh.values():
                if v == "A":
                    a_count += 1
                else:
                    b_count += 1
        assert a_count > 100
        assert b_count > 100


# ── Mutation ─────────────────────────────────────────────────────────────────


class TestMutation:
    def test_mutated_params_in_range(self):
        pm = PopulationManager(_make_config(), model_store=None)
        specs = pm.hp_specs
        for seed in range(50):
            hp = dict(PARENT_A_HP)
            pm.mutate(hp, mutation_rate=1.0, rng=random.Random(seed))
            validate_hyperparams(hp, specs)

    def test_mutation_rate_zero_no_changes(self):
        pm = PopulationManager(_make_config(), model_store=None)
        hp = dict(PARENT_A_HP)
        original = dict(hp)
        _, deltas = pm.mutate(hp, mutation_rate=0.0, rng=random.Random(42))
        for spec in pm.hp_specs:
            assert hp[spec.name] == original[spec.name]
            assert deltas[spec.name] is None

    def test_mutation_rate_one_all_attempted(self):
        """With rate=1.0, every param should be attempted (though some may have delta=0)."""
        pm = PopulationManager(_make_config(), model_store=None)
        hp = dict(PARENT_A_HP)
        _, deltas = pm.mutate(hp, mutation_rate=1.0, rng=random.Random(42))
        # All should have non-None delta (even if 0 for clamped edge cases)
        for spec in pm.hp_specs:
            assert deltas[spec.name] is not None

    def test_float_mutation_gaussian(self):
        """Float mutations should produce different values."""
        pm = PopulationManager(_make_config(), model_store=None)
        changed = False
        for seed in range(20):
            hp = dict(PARENT_A_HP)
            pm.mutate(hp, mutation_rate=1.0, rng=random.Random(seed))
            if hp["ppo_clip_epsilon"] != PARENT_A_HP["ppo_clip_epsilon"]:
                changed = True
                break
        assert changed

    def test_int_choice_mutation_adjacent(self):
        """Int choice mutation should produce an adjacent choice."""
        pm = PopulationManager(_make_config(), model_store=None)
        choices = [64, 128, 256, 512, 1024, 2048]
        for seed in range(50):
            hp = dict(PARENT_A_HP)
            hp["lstm_hidden_size"] = 256  # middle of choices
            pm.mutate(hp, mutation_rate=1.0, rng=random.Random(seed))
            assert hp["lstm_hidden_size"] in choices
            # Should be adjacent to 256 (i.e. 128 or 512) or same if clamped
            idx_orig = choices.index(256)
            idx_new = choices.index(hp["lstm_hidden_size"])
            assert abs(idx_new - idx_orig) <= 1

    def test_int_mutation_bounded(self):
        """Int at boundary should not go out of range."""
        pm = PopulationManager(_make_config(), model_store=None)
        for seed in range(50):
            hp = dict(PARENT_A_HP)
            hp["mlp_layers"] = 1  # at minimum
            pm.mutate(hp, mutation_rate=1.0, rng=random.Random(seed))
            assert 1 <= hp["mlp_layers"] <= 3

    def test_float_log_stays_in_range(self):
        pm = PopulationManager(_make_config(), model_store=None)
        for seed in range(50):
            hp = dict(PARENT_A_HP)
            hp["learning_rate"] = 1e-5  # at minimum
            pm.mutate(hp, mutation_rate=1.0, rng=random.Random(seed))
            assert 1e-5 <= hp["learning_rate"] <= 5e-4

    def test_deltas_dict_has_all_params(self):
        pm = PopulationManager(_make_config(), model_store=None)
        hp = dict(PARENT_A_HP)
        _, deltas = pm.mutate(hp, mutation_rate=0.5, rng=random.Random(42))
        for spec in pm.hp_specs:
            assert spec.name in deltas

    def test_reproducibility(self):
        pm = PopulationManager(_make_config(), model_store=None)
        hp1 = dict(PARENT_A_HP)
        hp2 = dict(PARENT_A_HP)
        pm.mutate(hp1, mutation_rate=0.5, rng=random.Random(42))
        pm.mutate(hp2, mutation_rate=0.5, rng=random.Random(42))
        for spec in pm.hp_specs:
            assert hp1[spec.name] == hp2[spec.name]


# ── Breed ────────────────────────────────────────────────────────────────────


class TestBreed:
    @pytest.fixture
    def bred_result(self, tmp_path):
        """Create 6 agents, score them, select, breed to fill back to 6."""
        config = _make_config(pop_size=6, n_elite=2, top_pct=0.5)
        store = ModelStore(str(tmp_path / "test.db"), str(tmp_path / "weights"))
        pm = PopulationManager(config, model_store=store)
        agents = pm.initialise_population(generation=0, seed=42)

        # Score: agents indexed by creation order, higher index = higher score
        scores = [
            _make_score(agents[i].model_id, composite=(i + 1) * 0.1)
            for i in range(6)
        ]
        result = pm.select(scores)
        children, records = pm.breed(result, generation=1, mutation_rate=0.3, seed=99)
        return pm, store, agents, result, children, records

    def test_children_count_fills_population(self, bred_result):
        pm, store, agents, result, children, records = bred_result
        assert len(children) + len(result.survivors) == pm.population_size

    def test_children_have_valid_hyperparams(self, bred_result):
        pm, store, agents, result, children, records = bred_result
        for child in children:
            validate_hyperparams(child.hyperparameters, pm.hp_specs)

    def test_children_have_policies(self, bred_result):
        pm, store, agents, result, children, records = bred_result
        from agents.policy_network import BasePolicy
        for child in children:
            assert isinstance(child.policy, BasePolicy)

    def test_children_forward_pass(self, bred_result):
        pm, store, agents, result, children, records = bred_result
        obs = torch.randn(1, pm.obs_dim)
        for child in children:
            out = child.policy(obs)
            assert out.action_mean.shape == (1, pm.action_dim)

    def test_children_registered_in_store(self, bred_result):
        pm, store, agents, result, children, records = bred_result
        for child in children:
            record = store.get_model(child.model_id)
            assert record is not None
            assert record.generation == 1
            assert record.parent_a_id is not None
            assert record.parent_b_id is not None

    def test_children_parents_are_survivors(self, bred_result):
        pm, store, agents, result, children, records = bred_result
        for child in children:
            record = store.get_model(child.model_id)
            assert record.parent_a_id in result.survivors
            assert record.parent_b_id in result.survivors

    def test_breeding_records_correct_count(self, bred_result):
        pm, store, agents, result, children, records = bred_result
        assert len(records) == len(children)

    def test_breeding_records_have_inheritance(self, bred_result):
        pm, store, agents, result, children, records = bred_result
        for br in records:
            assert isinstance(br, BreedingRecord)
            for spec in pm.hp_specs:
                assert spec.name in br.inheritance
                assert br.inheritance[spec.name] in ("A", "B")

    def test_breeding_records_have_deltas(self, bred_result):
        pm, store, agents, result, children, records = bred_result
        for br in records:
            for spec in pm.hp_specs:
                assert spec.name in br.deltas

    def test_no_two_children_identical(self, bred_result):
        pm, store, agents, result, children, records = bred_result
        hp_tuples = []
        for c in children:
            hp_no_arch = {k: v for k, v in c.hyperparameters.items() if k != "architecture_name"}
            hp_tuples.append(tuple(sorted(hp_no_arch.items())))
        # With 3 children and random crossover/mutation, extremely unlikely to be identical
        assert len(set(hp_tuples)) == len(hp_tuples)

    def test_breed_seed_reproducibility(self, tmp_path):
        config = _make_config(pop_size=6, n_elite=2, top_pct=0.5)
        store1 = ModelStore(str(tmp_path / "s1.db"), str(tmp_path / "w1"))
        store2 = ModelStore(str(tmp_path / "s2.db"), str(tmp_path / "w2"))

        pm1 = PopulationManager(config, model_store=store1)
        pm2 = PopulationManager(config, model_store=store2)
        agents1 = pm1.initialise_population(generation=0, seed=42)
        agents2 = pm2.initialise_population(generation=0, seed=42)

        scores1 = [_make_score(agents1[i].model_id, composite=(i+1)*0.1) for i in range(6)]
        scores2 = [_make_score(agents2[i].model_id, composite=(i+1)*0.1) for i in range(6)]

        r1 = pm1.select(scores1)
        r2 = pm2.select(scores2)

        c1, _ = pm1.breed(r1, generation=1, seed=77)
        c2, _ = pm2.breed(r2, generation=1, seed=77)

        for a, b in zip(c1, c2):
            hp_a = {k: v for k, v in a.hyperparameters.items() if k != "architecture_name"}
            hp_b = {k: v for k, v in b.hyperparameters.items() if k != "architecture_name"}
            assert hp_a == hp_b


# ── Genetic logging ──────────────────────────────────────────────────────────


class TestGeneticLogging:
    @pytest.fixture
    def logged_result(self, tmp_path):
        config = _make_config(pop_size=6, n_elite=2, top_pct=0.5, logs_dir=str(tmp_path / "logs"))
        store = ModelStore(str(tmp_path / "test.db"), str(tmp_path / "weights"))
        pm = PopulationManager(config, model_store=store)
        agents = pm.initialise_population(generation=0, seed=42)

        scores = [
            _make_score(agents[0].model_id, 0.05, win_rate=0.1, mean_daily_pnl=-10.0, sharpe=-2.0),
            _make_score(agents[1].model_id, 0.1, win_rate=0.2, mean_daily_pnl=-5.0, sharpe=-1.0),
            _make_score(agents[2].model_id, 0.2),
            _make_score(agents[3].model_id, 0.4),
            _make_score(agents[4].model_id, 0.6),
            _make_score(agents[5].model_id, 0.8),
        ]
        result = pm.select(scores)
        discarded = pm.apply_discard_policy(
            [s for s in scores if s.model_id in result.eliminated]
        )
        children, records = pm.breed(result, generation=1, seed=99)
        pm.log_generation(
            generation=1,
            selection_result=result,
            breeding_records=records,
            discarded=discarded,
        )
        return pm, store, agents, result, children, records, discarded, tmp_path

    def test_log_file_created(self, logged_result):
        *_, tmp_path = logged_result
        log_files = list((tmp_path / "logs" / "genetics").glob("gen_1_*.log"))
        assert len(log_files) == 1

    def test_log_file_contains_selection(self, logged_result):
        *_, tmp_path = logged_result
        log_files = list((tmp_path / "logs" / "genetics").glob("gen_1_*.log"))
        content = log_files[0].read_text(encoding="utf-8")
        assert "SELECTION" in content
        assert "elite" in content.lower()

    def test_log_file_contains_breeding(self, logged_result):
        *_, tmp_path = logged_result
        log_files = list((tmp_path / "logs" / "genetics").glob("gen_1_*.log"))
        content = log_files[0].read_text(encoding="utf-8")
        assert "BREEDING" in content
        assert "Parent A:" in content
        assert "Parent B:" in content

    def test_log_file_contains_discard(self, logged_result):
        pm, store, agents, result, children, records, discarded, tmp_path = logged_result
        if discarded:
            log_files = list((tmp_path / "logs" / "genetics").glob("gen_1_*.log"))
            content = log_files[0].read_text(encoding="utf-8")
            assert "Discarded" in content

    def test_genetic_events_in_sqlite(self, logged_result):
        pm, store, *_ = logged_result
        events = store.get_genetic_events(generation=1)
        assert len(events) > 0

    def test_selection_events_recorded(self, logged_result):
        pm, store, agents, result, *_ = logged_result
        events = store.get_genetic_events(generation=1)
        selection_events = [e for e in events if e.event_type == "selection"]
        # Should have one per survivor
        assert len(selection_events) == len(result.survivors)

    def test_crossover_events_recorded(self, logged_result):
        pm, store, agents, result, children, records, *_ = logged_result
        events = store.get_genetic_events(generation=1)
        crossover_events = [e for e in events if e.event_type == "crossover"]
        # One per param per child
        assert len(crossover_events) == len(children) * len(pm.hp_specs)

    def test_crossover_events_have_parent_ids(self, logged_result):
        pm, store, *_ = logged_result
        events = store.get_genetic_events(generation=1)
        crossover_events = [e for e in events if e.event_type == "crossover"]
        for e in crossover_events:
            assert e.parent_a_id is not None
            assert e.parent_b_id is not None
            assert e.child_model_id is not None

    def test_discard_events_recorded(self, logged_result):
        pm, store, agents, result, children, records, discarded, *_ = logged_result
        events = store.get_genetic_events(generation=1)
        discard_events = [e for e in events if e.event_type == "discard"]
        assert len(discard_events) == len(discarded)

    def test_events_queryable_by_child(self, logged_result):
        pm, store, agents, result, children, *_ = logged_result
        if children:
            events = store.get_genetic_events(child_model_id=children[0].model_id)
            assert len(events) == len(pm.hp_specs)  # one crossover event per param
            for e in events:
                assert e.child_model_id == children[0].model_id

    def test_human_summary_populated(self, logged_result):
        pm, store, *_ = logged_result
        events = store.get_genetic_events(generation=1)
        for e in events:
            assert e.human_summary is not None
            assert len(e.human_summary) > 0
