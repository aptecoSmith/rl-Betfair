"""Integration tests for genetic operators with real config.

Creates a population from real config, scores it with fake evaluations,
runs the full select → breed → log pipeline, and verifies genetic log
files and SQLite events are correct.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from agents.population_manager import (
    PopulationManager,
    parse_search_ranges,
    validate_hyperparams,
)
from registry.model_store import EvaluationDayRecord, ModelStore
from registry.scoreboard import ModelScore, Scoreboard

pytestmark = pytest.mark.integration

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@pytest.fixture(scope="module")
def real_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _record_fake_evaluation(store, model_id, day_pnls):
    dates = [f"2026-03-{20 + i:02d}" for i in range(len(day_pnls))]
    run_id = store.create_evaluation_run(
        model_id=model_id,
        train_cutoff_date="2026-03-19",
        test_days=dates,
    )
    for date, pnl in zip(dates, day_pnls):
        profitable = pnl > 0
        bet_count = 10
        winning = 6 if profitable else 3
        store.record_evaluation_day(
            EvaluationDayRecord(
                run_id=run_id, date=date, day_pnl=pnl, bet_count=bet_count,
                winning_bets=winning, bet_precision=winning / bet_count,
                pnl_per_bet=pnl / bet_count, early_picks=2 if profitable else 0,
                profitable=profitable,
            )
        )


class TestIntegrationGeneticOperators:
    @pytest.fixture
    def setup(self, real_config, tmp_path):
        """Full pipeline: create pop → evaluate → score → select → breed → log."""
        config = dict(real_config)
        config["population"] = {
            "size": 8,
            "n_elite": 2,
            "selection_top_pct": 0.5,
            "mutation_rate": 0.3,
        }
        config["paths"] = {
            **config.get("paths", {}),
            "logs": str(tmp_path / "logs"),
        }

        store = ModelStore(str(tmp_path / "test.db"), str(tmp_path / "weights"))
        pm = PopulationManager(config, model_store=store)
        agents = pm.initialise_population(generation=0, seed=42)

        # Varied performance: ascending quality
        pnl_profiles = [
            [-20, -15, -10],   # terrible
            [-5, -3, 2],       # bad
            [-2, 1, -1],       # mediocre
            [2, 3, -1],        # ok
            [5, 4, 3],         # decent
            [8, 7, 5],         # good
            [12, 10, 8],       # very good
            [15, 12, 18],      # best
        ]
        for i, agent in enumerate(agents):
            _record_fake_evaluation(store, agent.model_id, pnl_profiles[i])

        board = Scoreboard(store, config)
        rankings = board.rank_all()
        result = pm.select(rankings)
        discarded = pm.apply_discard_policy(
            [s for s in rankings if s.model_id in result.eliminated]
        )
        children, records = pm.breed(result, generation=1, seed=99)
        pm.log_generation(1, result, records, discarded)

        return config, store, pm, agents, result, children, records, discarded, tmp_path

    def test_children_fill_population(self, setup):
        config, store, pm, agents, result, children, records, discarded, _ = setup
        assert len(children) + len(result.survivors) == 8

    def test_all_children_hyperparams_valid(self, setup):
        config, store, pm, agents, result, children, *_ = setup
        specs = parse_search_ranges(config["hyperparameters"]["search_ranges"])
        for child in children:
            validate_hyperparams(child.hyperparameters, specs)

    def test_children_forward_pass_on_real_obs(self, setup):
        config, store, pm, agents, result, children, *_ = setup
        obs = torch.randn(1, pm.obs_dim)
        for child in children:
            out = child.policy(obs)
            assert out.action_mean.shape == (1, pm.action_dim)
            assert not torch.isnan(out.action_mean).any()

    def test_log_file_written(self, setup):
        *_, tmp_path = setup
        log_files = list((tmp_path / "logs" / "genetics").glob("gen_1_*.log"))
        assert len(log_files) == 1
        content = log_files[0].read_text(encoding="utf-8")
        assert "=== Generation 1" in content
        assert "SELECTION" in content
        assert "BREEDING" in content

    def test_genetic_events_table_populated(self, setup):
        _, store, *_ = setup
        events = store.get_genetic_events(generation=1)
        assert len(events) > 0

    def test_selection_events_match_survivors(self, setup):
        _, store, pm, agents, result, *_ = setup
        events = store.get_genetic_events(generation=1)
        selection = [e for e in events if e.event_type == "selection"]
        assert len(selection) == len(result.survivors)

    def test_crossover_events_per_child(self, setup):
        _, store, pm, agents, result, children, *_ = setup
        events = store.get_genetic_events(generation=1)
        crossover = [e for e in events if e.event_type == "crossover"]
        assert len(crossover) == len(children) * len(pm.hp_specs)

    def test_discard_events_match(self, setup):
        _, store, pm, agents, result, children, records, discarded, _ = setup
        events = store.get_genetic_events(generation=1)
        discard_evts = [e for e in events if e.event_type == "discard"]
        assert len(discard_evts) == len(discarded)

    def test_child_lineage_queryable(self, setup):
        _, store, pm, agents, result, children, *_ = setup
        if children:
            child = children[0]
            events = store.get_genetic_events(child_model_id=child.model_id)
            assert len(events) == len(pm.hp_specs)
            record = store.get_model(child.model_id)
            assert record.parent_a_id in result.survivors
            assert record.parent_b_id in result.survivors

    def test_children_weights_loadable(self, setup):
        _, store, pm, agents, result, children, *_ = setup
        for child in children:
            loaded = pm.load_agent(child.model_id)
            obs = torch.randn(1, pm.obs_dim)
            torch.manual_seed(0)
            out_orig = child.policy(obs)
            torch.manual_seed(0)
            out_loaded = loaded.policy(obs)
            assert torch.allclose(out_orig.action_mean, out_loaded.action_mean, atol=1e-6)
