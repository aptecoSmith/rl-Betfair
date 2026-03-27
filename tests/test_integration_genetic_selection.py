"""Integration tests for genetic selection with real config and model store.

Creates a small population, records fake evaluation results, scores them
via Scoreboard, then runs selection and discard. Verifies the full pipeline
from population creation through selection works end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agents.population_manager import PopulationManager, SelectionResult
from registry.model_store import EvaluationDayRecord, ModelStore
from registry.scoreboard import Scoreboard

pytestmark = pytest.mark.integration

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@pytest.fixture(scope="module")
def real_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _record_fake_evaluation(
    store: ModelStore,
    model_id: str,
    day_pnls: list[float],
) -> None:
    """Record fake evaluation results so the model can be scored."""
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
                run_id=run_id,
                date=date,
                day_pnl=pnl,
                bet_count=bet_count,
                winning_bets=winning,
                bet_precision=winning / bet_count,
                pnl_per_bet=pnl / bet_count,
                early_picks=2 if profitable else 0,
                profitable=profitable,
            )
        )


class TestIntegrationGeneticSelection:
    @pytest.fixture
    def setup(self, real_config, tmp_path):
        """Create a small population with varied evaluation results."""
        # Override to small pop for speed
        config = dict(real_config)
        config["population"] = {
            "size": 6,
            "n_elite": 2,
            "selection_top_pct": 0.5,
        }

        store = ModelStore(str(tmp_path / "test.db"), str(tmp_path / "weights"))
        pm = PopulationManager(config, model_store=store)
        agents = pm.initialise_population(generation=0, seed=42)

        # Record evaluations with varied performance:
        # Agent 0: terrible (should be discard candidate)
        _record_fake_evaluation(store, agents[0].model_id, [-20.0, -15.0, -10.0, -8.0, -12.0])
        # Agent 1: bad but not terrible
        _record_fake_evaluation(store, agents[1].model_id, [-5.0, -3.0, 2.0, -4.0, -6.0])
        # Agent 2: mediocre
        _record_fake_evaluation(store, agents[2].model_id, [3.0, -2.0, 5.0, -1.0, 4.0])
        # Agent 3: decent
        _record_fake_evaluation(store, agents[3].model_id, [8.0, 5.0, -2.0, 10.0, 7.0])
        # Agent 4: good
        _record_fake_evaluation(store, agents[4].model_id, [12.0, 8.0, 10.0, -1.0, 15.0])
        # Agent 5: best
        _record_fake_evaluation(store, agents[5].model_id, [15.0, 12.0, 18.0, 10.0, 20.0])

        return config, store, pm, agents

    def test_scoreboard_scores_all_agents(self, setup):
        config, store, pm, agents = setup
        board = Scoreboard(store, config)
        rankings = board.rank_all()
        assert len(rankings) == 6

    def test_correct_number_of_survivors(self, setup):
        config, store, pm, agents = setup
        board = Scoreboard(store, config)
        rankings = board.rank_all()
        result = pm.select(rankings)
        assert len(result.survivors) == 3  # 50% of 6

    def test_correct_number_eliminated(self, setup):
        config, store, pm, agents = setup
        board = Scoreboard(store, config)
        rankings = board.rank_all()
        result = pm.select(rankings)
        assert len(result.eliminated) == 3

    def test_elites_are_top_two(self, setup):
        config, store, pm, agents = setup
        board = Scoreboard(store, config)
        rankings = board.rank_all()
        result = pm.select(rankings)
        # Best and second best should be elites
        assert len(result.elites) == 2
        assert result.elites[0] == agents[5].model_id  # best
        assert result.elites[1] == agents[4].model_id  # good

    def test_elites_preserved_in_survivors(self, setup):
        config, store, pm, agents = setup
        board = Scoreboard(store, config)
        rankings = board.rank_all()
        result = pm.select(rankings)
        for elite_id in result.elites:
            assert elite_id in result.survivors

    def test_worst_agent_eliminated(self, setup):
        config, store, pm, agents = setup
        board = Scoreboard(store, config)
        rankings = board.rank_all()
        result = pm.select(rankings)
        assert agents[0].model_id in result.eliminated

    def test_discard_policy_catches_worst(self, setup):
        config, store, pm, agents = setup
        board = Scoreboard(store, config)
        rankings = board.rank_all()
        result = pm.select(rankings)
        eliminated_scores = [s for s in rankings if s.model_id in result.eliminated]
        discarded = pm.apply_discard_policy(eliminated_scores)
        # Agent 0 (all negative PnL, low win_rate, bad sharpe) should be discarded
        assert agents[0].model_id in discarded

    def test_discarded_status_in_store(self, setup):
        config, store, pm, agents = setup
        board = Scoreboard(store, config)
        rankings = board.rank_all()
        result = pm.select(rankings)
        eliminated_scores = [s for s in rankings if s.model_id in result.eliminated]
        pm.apply_discard_policy(eliminated_scores)
        # Agent 0 should now be discarded in the store
        record = store.get_model(agents[0].model_id)
        assert record.status == "discarded"

    def test_non_discarded_stay_active(self, setup):
        config, store, pm, agents = setup
        board = Scoreboard(store, config)
        rankings = board.rank_all()
        result = pm.select(rankings)
        eliminated_scores = [s for s in rankings if s.model_id in result.eliminated]
        discarded = pm.apply_discard_policy(eliminated_scores)
        # Survivors should still be active
        for mid in result.survivors:
            record = store.get_model(mid)
            assert record.status == "active"
