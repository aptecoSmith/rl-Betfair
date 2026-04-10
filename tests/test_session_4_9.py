"""
Session 4.9 — Start/stop training from UI tests.

Tests the training start/stop API endpoints and orchestrator stop_event.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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


class TestEmitInfo:

    def test_emit_info_puts_progress_event_on_queue(self, config):
        """_emit_info should push a progress event with phase='info' and detail."""
        from training.run_training import TrainingOrchestrator

        config["training"]["require_gpu"] = False
        queue = asyncio.Queue()
        orch = TrainingOrchestrator(
            config=config, model_store=None, device="cpu",
            progress_queue=queue,
        )
        orch._emit_info("test message")

        assert queue.qsize() == 1
        event = queue.get_nowait()
        assert event["event"] == "progress"
        assert event["phase"] == "info"
        assert event["detail"] == "test message"
        assert "timestamp" in event

    def test_emit_info_without_queue_does_not_crash(self, config):
        """_emit_info should be a no-op when progress_queue is None."""
        from training.run_training import TrainingOrchestrator

        config["training"]["require_gpu"] = False
        orch = TrainingOrchestrator(
            config=config, model_store=None, device="cpu",
        )
        # Should not raise
        orch._emit_info("test message")

    def test_multiple_info_events_all_queued(self, config):
        """Multiple _emit_info calls should all end up on the queue in order."""
        from training.run_training import TrainingOrchestrator

        config["training"]["require_gpu"] = False
        queue = asyncio.Queue()
        orch = TrainingOrchestrator(
            config=config, model_store=None, device="cpu",
            progress_queue=queue,
        )
        orch._emit_info("message 1")
        orch._emit_info("message 2")
        orch._emit_info("message 3")

        messages = []
        while not queue.empty():
            ev = queue.get_nowait()
            messages.append(ev["detail"])
        assert messages == ["message 1", "message 2", "message 3"]


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

        # Mock _send_to_worker since start now delegates to the worker process
        mock_resp = {"type": "ack", "run_id": "test-run-123"}
        with patch("api.routers.training._send_to_worker", new_callable=AsyncMock, return_value=mock_resp):
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
        """POST /training/stop sends stop command to worker."""
        client, app = _make_test_app()
        app.state.training_state["running"] = True

        # Mock _send_to_worker since stop now delegates to the worker process
        mock_resp = {"type": "ack"}
        with patch("api.routers.training._send_to_worker", new_callable=AsyncMock, return_value=mock_resp):
            resp = client.post("/training/stop")
        assert resp.status_code == 200
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

    def test_start_request_with_dates(self):
        from api.schemas import StartTrainingRequest
        req = StartTrainingRequest(
            train_dates=["2026-01-01", "2026-01-02"],
            test_dates=["2026-01-03"],
        )
        assert req.train_dates == ["2026-01-01", "2026-01-02"]
        assert req.test_dates == ["2026-01-03"]

    def test_start_request_dates_default_none(self):
        from api.schemas import StartTrainingRequest
        req = StartTrainingRequest()
        assert req.train_dates is None
        assert req.test_dates is None

    def test_finish_response_fields(self):
        from api.schemas import FinishTrainingResponse
        resp = FinishTrainingResponse(detail="Finishing")
        assert resp.detail == "Finishing"

    def test_status_includes_worker_connected(self):
        from api.schemas import TrainingStatus
        status = TrainingStatus(running=False, worker_connected=True)
        assert status.worker_connected is True

    def test_status_worker_connected_defaults_false(self):
        from api.schemas import TrainingStatus
        status = TrainingStatus(running=False)
        assert status.worker_connected is False


class TestDateSelectionEndpoint:

    def test_start_with_explicit_dates(self, tmp_path):
        """POST /training/start with explicit dates passes them through."""
        client, app = _make_test_app()
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        # Create parquet files
        import pandas as pd
        for d in ["2026-01-01", "2026-01-02", "2026-01-03"]:
            pd.DataFrame({"x": [1]}).to_parquet(data_dir / f"{d}.parquet")
        app.state.config["paths"]["processed_data"] = str(data_dir)

        mock_send = AsyncMock(return_value={"type": "started", "run_id": "test123"})
        with patch("api.routers.training._send_to_worker", mock_send):
            resp = client.post("/training/start", json={
                "n_generations": 1,
                "n_epochs": 1,
                "train_dates": ["2026-01-01"],
                "test_dates": ["2026-01-02", "2026-01-03"],
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["train_days"] == ["2026-01-01"]
        assert data["test_days"] == ["2026-01-02", "2026-01-03"]

    def test_start_with_invalid_dates_returns_400(self, tmp_path):
        """POST /training/start with missing dates returns 400."""
        client, app = _make_test_app()
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        import pandas as pd
        pd.DataFrame({"x": [1]}).to_parquet(data_dir / "2026-01-01.parquet")
        app.state.config["paths"]["processed_data"] = str(data_dir)

        resp = client.post("/training/start", json={
            "n_generations": 1,
            "n_epochs": 1,
            "train_dates": ["2026-99-99"],  # doesn't exist
        })
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"].lower()


class TestPopulationSizeOverride:

    def test_population_size_applied_to_config(self):
        """When population_size is set, the worker config should be overridden."""
        import copy
        config = yaml.safe_load(open("config.yaml"))
        assert config["population"]["size"] == 50  # default

        # Simulate what the worker does
        run_config = copy.deepcopy(config)
        population_size = 5
        if population_size is not None:
            run_config["population"]["size"] = population_size
            run_config["population"]["n_elite"] = max(1, population_size // 10)

        assert run_config["population"]["size"] == 5
        assert run_config["population"]["n_elite"] == 1

    def test_population_size_none_uses_default(self):
        """When population_size is None, the config default should be used."""
        import copy
        config = yaml.safe_load(open("config.yaml"))

        run_config = copy.deepcopy(config)
        population_size = None
        if population_size is not None:
            run_config["population"]["size"] = population_size

        assert run_config["population"]["size"] == 50  # unchanged
