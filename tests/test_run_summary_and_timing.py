"""Tests for Session 4 ETA overhaul + training-end-summary features.

Covers:
- overall progress field on progress events
- last_run_timing.json persistence
- enriched run_complete summary (best_model, top_5, population_summary, etc)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from registry.model_store import ModelStore
from training.run_training import (
    HISTORICAL_TIMING_PATH,
    TrainingOrchestrator,
)

# Reuse the synthetic data helpers from test_orchestrator.
from tests.test_orchestrator import _make_day, _make_full_config


@pytest.fixture
def isolated_timing_path(tmp_path, monkeypatch):
    """Redirect the historical-timing file to a per-test temp path."""
    target = tmp_path / "last_run_timing.json"
    monkeypatch.setattr(
        "training.run_training.HISTORICAL_TIMING_PATH", target,
    )
    return target


def _drain(queue: asyncio.Queue) -> list[dict]:
    events: list[dict] = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


class TestOverallProgressField:
    def test_progress_events_include_overall(self, tmp_path):
        config = _make_full_config()
        config["population"]["size"] = 2
        store = ModelStore(db_path=tmp_path / "t.db", weights_dir=tmp_path / "w")
        queue: asyncio.Queue = asyncio.Queue()
        orch = TrainingOrchestrator(config, model_store=store, progress_queue=queue)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)
        orch.run(train_days=[train], test_days=[test], n_generations=1, n_epochs=1, seed=42)

        events = _drain(queue)
        progress_events = [e for e in events if e["event"] == "progress"]
        assert progress_events, "expected at least one progress event"
        # All progress events post-init should carry an overall snapshot.
        with_overall = [e for e in progress_events if "overall" in e]
        assert with_overall, "no progress event carried the overall field"
        overall = with_overall[-1]["overall"]
        # Matches ProgressTracker schema.
        assert {"label", "completed", "total", "pct"} <= set(overall.keys())


class TestHistoricalTimingPersistence:
    def test_writes_timing_file_after_run(self, tmp_path, isolated_timing_path):
        config = _make_full_config()
        config["population"]["size"] = 2
        store = ModelStore(db_path=tmp_path / "t.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)
        orch.run(train_days=[train], test_days=[test], n_generations=1, n_epochs=1, seed=42)

        assert isolated_timing_path.exists()
        data = json.loads(isolated_timing_path.read_text())
        assert "train_seconds_per_agent_per_day" in data
        assert "eval_seconds_per_agent_per_day" in data
        # Both rates should be positive finite numbers.
        assert data["train_seconds_per_agent_per_day"] > 0
        assert data["eval_seconds_per_agent_per_day"] > 0

    def test_missing_file_not_fatal(self, tmp_path, monkeypatch):
        """If the historical-timing path can't be written, the run still completes."""
        # Point at a path inside a file (always fails to create) — ensures
        # the try/except shields the run.
        broken = tmp_path / "a_file"
        broken.write_text("blocker")
        monkeypatch.setattr(
            "training.run_training.HISTORICAL_TIMING_PATH", broken / "timing.json",
        )
        config = _make_full_config()
        config["population"]["size"] = 2
        store = ModelStore(db_path=tmp_path / "t.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)
        # Should not raise.
        result = orch.run(
            train_days=[train], test_days=[test],
            n_generations=1, n_epochs=1, seed=42,
        )
        assert len(result.generations) == 1


class TestRunCompleteSummary:
    def test_run_complete_has_enriched_summary(self, tmp_path, isolated_timing_path):
        config = _make_full_config()
        config["population"]["size"] = 2
        store = ModelStore(db_path=tmp_path / "t.db", weights_dir=tmp_path / "w")
        queue: asyncio.Queue = asyncio.Queue()
        orch = TrainingOrchestrator(config, model_store=store, progress_queue=queue)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)
        orch.run(train_days=[train], test_days=[test], n_generations=1, n_epochs=1, seed=42)

        events = _drain(queue)
        complete = [
            e for e in events
            if e.get("event") == "phase_complete" and e.get("phase") == "run_complete"
        ]
        assert len(complete) == 1
        summary = complete[0]["summary"]
        # Required enrichment keys.
        for key in [
            "run_id", "status", "generations_completed", "generations_requested",
            "total_agents_trained", "total_agents_evaluated", "wall_time_seconds",
            "best_model", "top_5", "population_summary", "error_message",
        ]:
            assert key in summary, f"run_complete summary missing {key}"
        assert summary["status"] == "completed"
        assert summary["generations_requested"] == 1
        assert summary["wall_time_seconds"] > 0
        assert 0 < summary["wall_time_seconds"] < 86400
        # best_model matches the top of top_5 by composite score.
        if summary["top_5"]:
            top = summary["top_5"][0]
            assert summary["best_model"] is not None
            assert summary["best_model"]["model_id"] == top["model_id"]
        # Population summary keys.
        pop = summary["population_summary"]
        assert {"survived", "discarded", "garaged"} == set(pop.keys())
