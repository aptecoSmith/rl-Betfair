"""
Session 4.9 — Start/stop training from UI tests.

Tests the training start/stop API endpoints and orchestrator stop_event.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def config():
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)


# ── Orchestrator stop_event tests ───────────────────────────────────────────


class TestOrchestratorStopEvent:

    def test_stop_event_accepted(self, config):
        """TrainingOrchestrator accepts a stop_event parameter."""
        from training.run_training import TrainingOrchestrator

        config["training"]["require_gpu"] = False
        stop = threading.Event()
        orch = TrainingOrchestrator(
            config=config, model_store=None, device="cpu",
            stop_event=stop,
        )
        assert orch.stop_event is stop

    def test_stop_event_none_by_default(self, config):
        """Without stop_event, it defaults to None."""
        from training.run_training import TrainingOrchestrator

        config["training"]["require_gpu"] = False
        orch = TrainingOrchestrator(
            config=config, model_store=None, device="cpu",
        )
        assert orch.stop_event is None

    def test_check_stop_returns_false_when_not_set(self, config):
        from training.run_training import TrainingOrchestrator

        config["training"]["require_gpu"] = False
        stop = threading.Event()
        orch = TrainingOrchestrator(
            config=config, model_store=None, device="cpu",
            stop_event=stop,
        )
        assert orch._check_stop() is False

    def test_check_stop_returns_true_when_set(self, config):
        from training.run_training import TrainingOrchestrator

        config["training"]["require_gpu"] = False
        stop = threading.Event()
        orch = TrainingOrchestrator(
            config=config, model_store=None, device="cpu",
            stop_event=stop,
        )
        stop.set()
        assert orch._check_stop() is True

    def test_stop_emits_run_stopped_event(self, config):
        """When stop is detected, a run_stopped phase_complete event is emitted."""
        from training.run_training import TrainingOrchestrator

        config["training"]["require_gpu"] = False
        stop = threading.Event()
        queue = asyncio.Queue()
        orch = TrainingOrchestrator(
            config=config, model_store=None, device="cpu",
            stop_event=stop, progress_queue=queue,
        )
        stop.set()
        orch._check_stop()

        # Drain events
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        stopped_events = [e for e in events if e.get("phase") == "run_stopped"]
        assert len(stopped_events) == 1

    def test_check_stop_only_emits_once(self, config):
        """Multiple _check_stop calls only emit one run_stopped event."""
        from training.run_training import TrainingOrchestrator

        config["training"]["require_gpu"] = False
        stop = threading.Event()
        queue = asyncio.Queue()
        orch = TrainingOrchestrator(
            config=config, model_store=None, device="cpu",
            stop_event=stop, progress_queue=queue,
        )
        stop.set()
        orch._check_stop()
        orch._check_stop()
        orch._check_stop()

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        stopped = [e for e in events if e.get("phase") == "run_stopped"]
        assert len(stopped) == 1


class TestOrchestratorFinishEvent:

    def test_finish_event_accepted(self, config):
        """TrainingOrchestrator accepts a finish_event parameter."""
        from training.run_training import TrainingOrchestrator

        config["training"]["require_gpu"] = False
        finish = threading.Event()
        orch = TrainingOrchestrator(
            config=config, model_store=None, device="cpu",
            finish_event=finish,
        )
        assert orch.finish_event is finish

    def test_finish_event_none_by_default(self, config):
        """Without finish_event, it defaults to None."""
        from training.run_training import TrainingOrchestrator

        config["training"]["require_gpu"] = False
        orch = TrainingOrchestrator(
            config=config, model_store=None, device="cpu",
        )
        assert orch.finish_event is None

    def test_check_finish_returns_false_when_not_set(self, config):
        from training.run_training import TrainingOrchestrator

        config["training"]["require_gpu"] = False
        finish = threading.Event()
        orch = TrainingOrchestrator(
            config=config, model_store=None, device="cpu",
            finish_event=finish,
        )
        assert orch._check_finish() is False

    def test_check_finish_returns_true_when_set(self, config):
        from training.run_training import TrainingOrchestrator

        config["training"]["require_gpu"] = False
        finish = threading.Event()
        orch = TrainingOrchestrator(
            config=config, model_store=None, device="cpu",
            finish_event=finish,
        )
        finish.set()
        assert orch._check_finish() is True

    def test_check_finish_false_when_none(self, config):
        """_check_finish should return False when finish_event is None."""
        from training.run_training import TrainingOrchestrator

        config["training"]["require_gpu"] = False
        orch = TrainingOrchestrator(
            config=config, model_store=None, device="cpu",
        )
        assert orch._check_finish() is False


# ── API endpoint tests ──────────────────────────────────────────────────────


def _make_test_app():
    """Create a test FastAPI app with minimal state."""
    import threading
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routers import training

    app = FastAPI()
    app.include_router(training.router)
    app.state.config = yaml.safe_load(open("config.yaml"))
    app.state.config["training"]["require_gpu"] = False
    app.state.store = None
    app.state.progress_queue = asyncio.Queue()
    app.state.training_state = {"running": False, "latest_event": None}
    app.state.stop_event = threading.Event()
    app.state.training_task = None

    return TestClient(app), app


class TestStartEndpoint:

    def test_start_rejects_when_running(self):
        """POST /training/start returns 409 if already running."""
        client, app = _make_test_app()
        app.state.training_state["running"] = True

        resp = client.post("/training/start", json={"n_generations": 1, "n_epochs": 1})
        assert resp.status_code == 409

    def test_start_rejects_no_data(self, tmp_path):
        """POST /training/start returns 400 if no extracted data."""
        client, app = _make_test_app()
        app.state.config["paths"]["processed_data"] = str(tmp_path)

        resp = client.post("/training/start", json={"n_generations": 1, "n_epochs": 1})
        assert resp.status_code == 400
        assert "No extracted data" in resp.json()["detail"]

    def test_start_returns_run_config(self):
        """POST /training/start returns train/test split and params."""
        client, app = _make_test_app()

        # Mock asyncio.create_task to prevent actual training from running
        with patch("api.routers.training.asyncio.create_task") as mock_task:
            mock_task.return_value = MagicMock()
            resp = client.post("/training/start", json={"n_generations": 2, "n_epochs": 1})

        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert data["n_generations"] == 2
        assert data["n_epochs"] == 1
        assert len(data["train_days"]) > 0
        assert len(data["test_days"]) > 0


class TestStopEndpoint:

    def test_stop_rejects_when_not_running(self):
        """POST /training/stop returns 409 if no run in progress."""
        client, app = _make_test_app()
        resp = client.post("/training/stop")
        assert resp.status_code == 409

    def test_stop_sets_event(self):
        """POST /training/stop sets the stop_event."""
        client, app = _make_test_app()
        app.state.training_state["running"] = True

        resp = client.post("/training/stop")
        assert resp.status_code == 200
        assert app.state.stop_event.is_set()
        assert "Stop requested" in resp.json()["detail"]


# ── Pydantic schema tests ──────────────────────────────────────────────────


class TestTrainingSchemas:

    def test_start_request_defaults(self):
        from api.schemas import StartTrainingRequest
        req = StartTrainingRequest()
        assert req.n_generations == 3
        assert req.n_epochs == 3
        assert req.seed is None

    def test_start_request_custom(self):
        from api.schemas import StartTrainingRequest
        req = StartTrainingRequest(n_generations=5, n_epochs=2, seed=42)
        assert req.n_generations == 5
        assert req.n_epochs == 2
        assert req.seed == 42

    def test_start_response_fields(self):
        from api.schemas import StartTrainingResponse
        resp = StartTrainingResponse(
            run_id="abc", train_days=["2026-01-01"],
            test_days=["2026-01-02"], n_generations=3, n_epochs=3,
        )
        assert resp.run_id == "abc"

    def test_stop_response_fields(self):
        from api.schemas import StopTrainingResponse
        resp = StopTrainingResponse(detail="Stopped")
        assert resp.detail == "Stopped"
