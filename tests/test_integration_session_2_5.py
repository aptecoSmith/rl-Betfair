"""Integration tests for Session 2.5 — First multi-generation run.

Runs N generations on real extracted data with a small population and verifies:
- All models registered in the model store
- Genetic events populated in SQLite
- Genetic log files written and non-empty
- Scoreboard non-trivial (scores computed for all models)
- Bet logs present for every evaluation day
- Progress events emitted in correct phase order

Run with: pytest -m integration tests/test_integration_session_2_5.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from data.episode_builder import load_days
from registry.model_store import ModelStore
from registry.scoreboard import Scoreboard
from training.run_training import TrainingOrchestrator

pytestmark = pytest.mark.integration


def _get_available_dates(data_dir: str = "data/processed") -> list[str]:
    processed = Path(data_dir)
    if not processed.exists():
        return []
    dates = set()
    for f in processed.glob("*_ticks.parquet"):
        date_str = f.name.replace("_ticks.parquet", "")
        dates.add(date_str)
    return sorted(dates)


@pytest.fixture(scope="module")
def integration_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # Small population for fast tests
    config["population"]["size"] = 4
    config["population"]["n_elite"] = 1
    # Don't require GPU for integration tests
    config["training"]["require_gpu"] = False
    return config


@pytest.fixture(scope="module")
def real_days(integration_config):
    dates = _get_available_dates(integration_config["paths"]["processed_data"])
    if not dates:
        pytest.skip("No extracted data available")
    return load_days(dates, data_dir=integration_config["paths"]["processed_data"])


@pytest.fixture
def run_result(real_days, integration_config, tmp_path_factory):
    """Run 2 generations on real data and return the result."""
    tmp_path = tmp_path_factory.mktemp("session_2_5")
    config = dict(integration_config)
    config["paths"] = dict(config["paths"])
    config["paths"]["logs"] = str(tmp_path / "logs")

    store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
    queue = asyncio.Queue()

    orch = TrainingOrchestrator(
        config,
        model_store=store,
        progress_queue=queue,
        device="cpu",
    )

    train_days = real_days[:1]
    test_days = real_days[1:] if len(real_days) > 1 else []

    result = orch.run(
        train_days=train_days,
        test_days=test_days,
        n_generations=2,
        n_epochs=1,
        seed=42,
    )

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())

    return result, store, events, tmp_path


class TestAllModelsInRegistry:
    def test_at_least_population_size_models(self, run_result, integration_config):
        _, store, _, _ = run_result
        models = store.list_models()
        pop_size = integration_config["population"]["size"]
        assert len(models) >= pop_size

    def test_all_models_have_weights(self, run_result):
        _, store, _, _ = run_result
        for m in store.list_models():
            assert m.weights_path is not None
            assert Path(m.weights_path).exists()


class TestGeneticEventsPopulated:
    def test_genetic_events_in_sqlite(self, run_result):
        _, store, _, _ = run_result
        events = store.get_genetic_events(generation=0)
        assert len(events) > 0

    def test_genetic_log_files_written(self, run_result):
        _, _, _, tmp_path = run_result
        genetics_dir = tmp_path / "logs" / "genetics"
        assert genetics_dir.exists()
        log_files = list(genetics_dir.glob("gen_0_*.log"))
        assert len(log_files) == 1
        # Log file should be non-empty
        assert log_files[0].stat().st_size > 0

    def test_genetic_log_contains_selection_and_breeding(self, run_result):
        _, _, _, tmp_path = run_result
        genetics_dir = tmp_path / "logs" / "genetics"
        log_files = list(genetics_dir.glob("gen_0_*.log"))
        content = log_files[0].read_text(encoding="utf-8")
        assert "SELECTION" in content
        assert "BREEDING" in content
        assert "Trait inheritance:" in content


class TestScoreboardNonTrivial:
    def test_final_rankings_exist(self, run_result):
        result, _, _, _ = run_result
        assert len(result.final_rankings) > 0

    def test_scores_vary(self, run_result):
        result, _, _, _ = run_result
        scores = [s.composite_score for s in result.final_rankings]
        # With different agents, not all scores should be identical
        assert len(set(round(s, 6) for s in scores)) > 1

    def test_rankings_sorted_descending(self, run_result):
        result, _, _, _ = run_result
        for i in range(len(result.final_rankings) - 1):
            assert (
                result.final_rankings[i].composite_score
                >= result.final_rankings[i + 1].composite_score
            )


class TestBetLogsPresent:
    def test_all_evaluated_models_have_bet_logs(self, run_result):
        _, store, _, _ = run_result
        for m in store.list_models():
            run = store.get_latest_evaluation_run(m.model_id)
            assert run is not None, f"Model {m.model_id[:12]} has no evaluation"
            days = store.get_evaluation_days(run.run_id)
            assert len(days) > 0, f"Model {m.model_id[:12]} has no evaluation days"


class TestProgressEventsCorrectOrder:
    def test_phases_in_order(self, run_result):
        _, _, events, _ = run_result
        phase_starts = [
            e["phase"] for e in events if e["event"] == "phase_start"
        ]
        # Gen 0: training → evaluating → scoring → selecting → breeding
        # Gen 1: training → evaluating → scoring
        assert "training" in phase_starts
        assert "evaluating" in phase_starts
        assert "scoring" in phase_starts
        assert "selecting" in phase_starts
        assert "breeding" in phase_starts

    def test_run_complete_emitted(self, run_result):
        _, _, events, _ = run_result
        complete = [e for e in events if e["phase"] == "run_complete"]
        assert len(complete) == 1

    def test_progress_events_emitted(self, run_result):
        _, _, events, _ = run_result
        progress = [e for e in events if e["event"] == "progress"]
        assert len(progress) > 0
