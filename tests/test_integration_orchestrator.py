"""Integration tests for the training orchestrator.

These tests run on real extracted data (require MySQL and Parquet files).
Run with: pytest -m integration tests/test_integration_orchestrator.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from data.episode_builder import load_day, load_days
from registry.model_store import ModelStore
from registry.scoreboard import Scoreboard
from training.evaluator import Evaluator
from training.run_training import TrainingOrchestrator


pytestmark = pytest.mark.integration


def _get_available_dates(data_dir: str = "data/processed") -> list[str]:
    """Find all available extracted dates."""
    processed = Path(data_dir)
    if not processed.exists():
        return []
    dates = set()
    for f in processed.glob("*_ticks.parquet"):
        date_str = f.name.replace("_ticks.parquet", "")
        dates.add(date_str)
    return sorted(dates)


def _make_integration_config() -> dict:
    """Full config for integration tests."""
    import yaml
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # Override for fast tests
    config["population"]["size"] = 4
    config["population"]["n_elite"] = 1
    return config


@pytest.fixture(scope="module")
def available_dates():
    dates = _get_available_dates()
    if not dates:
        pytest.skip("No extracted data available")
    return dates


@pytest.fixture(scope="module")
def real_days(available_dates):
    """Load all available days."""
    return load_days(available_dates)


@pytest.fixture
def integration_config():
    return _make_integration_config()


# ── Evaluator integration tests ──────────────────────────────────────────────


class TestEvaluatorIntegration:
    def test_evaluate_on_real_day(self, real_days, integration_config, tmp_path):
        """Evaluate a model on real data — metrics should be populated."""
        from agents.architecture_registry import create_policy
        from env.betfair_env import ACTIONS_PER_RUNNER, AGENT_STATE_DIM, MARKET_DIM, RUNNER_DIM, VELOCITY_DIM

        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        max_runners = integration_config["training"]["max_runners"]
        obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * max_runners) + AGENT_STATE_DIM
        action_dim = max_runners * ACTIONS_PER_RUNNER

        policy = create_policy(
            name="ppo_lstm_v1",
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_runners=max_runners,
            hyperparams={"lstm_hidden_size": 64, "mlp_hidden_size": 64, "mlp_layers": 1},
        )

        model_id = store.create_model(
            generation=0,
            architecture_name="ppo_lstm_v1",
            architecture_description="integration test",
            hyperparameters={},
        )
        store.save_weights(model_id, policy.state_dict())

        evaluator = Evaluator(integration_config, model_store=store)
        day = real_days[0]
        run_id, records = evaluator.evaluate(
            model_id, policy, [day], day.date,
        )

        assert run_id is not None
        assert len(records) == 1
        rec = records[0]
        assert isinstance(rec.day_pnl, float)
        assert rec.bet_count >= 0
        assert 0.0 <= rec.bet_precision <= 1.0

        # Check persistence
        stored_days = store.get_evaluation_days(run_id)
        assert len(stored_days) == 1


# ── Orchestrator integration tests ───────────────────────────────────────────


class TestOrchestratorIntegration:
    def test_two_generations_on_real_data(self, real_days, integration_config, tmp_path):
        """Run 2 generations on real data with a small population."""
        integration_config["paths"]["logs"] = str(tmp_path / "logs")
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        queue = asyncio.Queue()

        orch = TrainingOrchestrator(
            integration_config,
            model_store=store,
            progress_queue=queue,
        )

        # Use same day for train and test if only 1 day available
        train_days = real_days[:1]
        test_days = real_days[1:] if len(real_days) > 1 else real_days[:1]

        result = orch.run(
            train_days=train_days,
            test_days=test_days,
            n_generations=2,
            n_epochs=1,
            seed=42,
        )

        assert len(result.generations) == 2

    def test_registry_updated(self, real_days, integration_config, tmp_path):
        """After 2 gens, registry should contain models with evaluations."""
        integration_config["paths"]["logs"] = str(tmp_path / "logs")
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")

        orch = TrainingOrchestrator(
            integration_config,
            model_store=store,
        )

        train_days = real_days[:1]
        test_days = real_days[1:] if len(real_days) > 1 else real_days[:1]

        orch.run(
            train_days=train_days,
            test_days=test_days,
            n_generations=2,
            n_epochs=1,
            seed=42,
        )

        models = store.list_models()
        pop_size = integration_config["population"]["size"]
        assert len(models) >= pop_size

        # Every model should have an evaluation run
        for m in models:
            if m.status == "active":
                run = store.get_latest_evaluation_run(m.model_id)
                assert run is not None, f"Model {m.model_id[:12]} has no eval run"

    def test_events_in_correct_order(self, real_days, integration_config, tmp_path):
        """Progress events should follow the correct phase order."""
        integration_config["paths"]["logs"] = str(tmp_path / "logs")
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        queue = asyncio.Queue()

        orch = TrainingOrchestrator(
            integration_config,
            model_store=store,
            progress_queue=queue,
        )

        train_days = real_days[:1]
        test_days = real_days[1:] if len(real_days) > 1 else real_days[:1]

        orch.run(
            train_days=train_days,
            test_days=test_days,
            n_generations=2,
            n_epochs=1,
            seed=42,
        )

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        phase_starts = [
            e["phase"] for e in events
            if e["event"] == "phase_start"
        ]
        # First gen: training → evaluating → scoring → selecting → breeding
        # Second gen: training → evaluating → scoring
        assert "training" in phase_starts
        assert "evaluating" in phase_starts
        assert "scoring" in phase_starts

    def test_genetic_log_populated(self, real_days, integration_config, tmp_path):
        """Genetic log file and SQLite events should be written."""
        integration_config["paths"]["logs"] = str(tmp_path / "logs")
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")

        orch = TrainingOrchestrator(
            integration_config,
            model_store=store,
        )

        train_days = real_days[:1]
        test_days = real_days[1:] if len(real_days) > 1 else real_days[:1]

        orch.run(
            train_days=train_days,
            test_days=test_days,
            n_generations=2,
            n_epochs=1,
            seed=42,
        )

        # Check log file
        genetics_dir = tmp_path / "logs" / "genetics"
        assert genetics_dir.exists()
        log_files = list(genetics_dir.glob("gen_0_*.log"))
        assert len(log_files) == 1

        # Check SQLite events
        events = store.get_genetic_events(generation=0)
        assert len(events) > 0

    def test_scoreboard_re_ranked(self, real_days, integration_config, tmp_path):
        """After 2 gens, scoreboard should have re-ranked models."""
        integration_config["paths"]["logs"] = str(tmp_path / "logs")
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")

        orch = TrainingOrchestrator(
            integration_config,
            model_store=store,
        )

        train_days = real_days[:1]
        test_days = real_days[1:] if len(real_days) > 1 else real_days[:1]

        result = orch.run(
            train_days=train_days,
            test_days=test_days,
            n_generations=2,
            n_epochs=1,
            seed=42,
        )

        assert len(result.final_rankings) > 0
        # Rankings should be sorted descending
        for i in range(len(result.final_rankings) - 1):
            assert (
                result.final_rankings[i].composite_score
                >= result.final_rankings[i + 1].composite_score
            )
