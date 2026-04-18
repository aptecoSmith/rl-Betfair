"""Unit tests for api/routers/training.py — status endpoint and WebSocket."""

from __future__ import annotations

import asyncio
import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import training


# ── Helpers ──────────────────────────────────────────────────────────

# Short keepalive so test_ws_connect_idle doesn't wait 30 s.
_KEEPALIVE_TIMEOUT = 1.0


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Queue consumer identical to main.py but with a fast keepalive for tests."""
    state = app.state.training_state
    queue = app.state.progress_queue

    async def _queue_consumer():
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=_KEEPALIVE_TIMEOUT)
            except asyncio.TimeoutError:
                ping = json.dumps({"event": "ping"})
                dead = set()
                for send_fn in app.state.ws_clients:
                    try:
                        await send_fn(ping)
                    except Exception:
                        dead.add(send_fn)
                app.state.ws_clients -= dead
                continue
            except asyncio.CancelledError:
                break

            state["latest_event"] = event
            if event.get("process"):
                state["latest_process"] = event["process"]
            if event.get("item"):
                state["latest_item"] = event["item"]

            if event.get("event") == "phase_start":
                state["running"] = True
                state["latest_item"] = None
            elif (
                event.get("event") == "run_complete"
                or (
                    event.get("event") == "phase_complete"
                    and event.get("phase") in (
                        "run_complete", "run_stopped", "run_error",
                        "extracting",
                    )
                )
            ):
                state["running"] = False
                state["latest_process"] = None
                state["latest_item"] = None

            msg = json.dumps(event)
            dead = set()
            for send_fn in app.state.ws_clients:
                try:
                    await send_fn(msg)
                except Exception:
                    dead.add(send_fn)
            app.state.ws_clients -= dead

    consumer_task = asyncio.create_task(_queue_consumer())
    yield
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass


def _make_app(training_state: dict | None = None) -> tuple[TestClient, FastAPI]:
    app = FastAPI(lifespan=_lifespan)
    app.include_router(training.router)

    app.state.training_state = training_state or {
        "running": False,
        "latest_event": None,
        "latest_process": None,
        "latest_item": None,
    }
    app.state.progress_queue = asyncio.Queue()
    app.state.ws_clients = set()
    app.state.worker_connected = app.state.training_state.get("running", False)

    # Use context manager so the lifespan (and queue consumer) actually starts
    client = TestClient(app)
    client.__enter__()
    return client, app


# ── Status Endpoint Tests ────────────────────────────────────────────


class TestTrainingStatus:
    def test_idle_status(self):
        client, _ = _make_app()
        resp = client.get("/training/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is False
        assert data["phase"] is None

    def test_running_no_event(self):
        client, _ = _make_app({"running": True, "latest_event": None, "latest_process": None, "latest_item": None})
        resp = client.get("/training/status")
        data = resp.json()
        assert data["running"] is True
        assert data["phase"] is None

    def test_running_with_progress(self):
        process_snap = {
            "label": "Generation 3 — training 20 agents",
            "completed": 7,
            "total": 20,
            "pct": 35.0,
            "item_eta_human": "6 min",
            "process_eta_human": "1h 18m",
        }
        item_snap = {
            "label": "Training agent model_xyz",
            "completed": 312,
            "total": 1000,
            "pct": 31.2,
            "item_eta_human": "4m 12s",
            "process_eta_human": "6m 05s",
        }
        state = {
            "running": True,
            "latest_event": {
                "event": "progress",
                "phase": "training",
                "generation": 3,
                "process": process_snap,
                "item": item_snap,
                "detail": "Episode 312 | reward=+1.24",
            },
            "latest_process": process_snap,
            "latest_item": item_snap,
        }
        client, _ = _make_app(state)
        resp = client.get("/training/status")
        data = resp.json()
        assert data["running"] is True
        assert data["phase"] == "training"
        assert data["generation"] == 3
        assert data["process"]["completed"] == 7
        assert data["process"]["total"] == 20
        assert data["item"]["completed"] == 312
        assert data["detail"] == "Episode 312 | reward=+1.24"


# ── WebSocket Tests ──────────────────────────────────────────────────


class TestTrainingWebSocket:
    def test_ws_connect_idle(self):
        """WebSocket connects and receives ping keepalive when idle."""
        client, _ = _make_app()
        with client.websocket_connect("/ws/training") as ws:
            # Should receive ping within timeout
            data = ws.receive_json()
            assert data["event"] == "ping"

    def test_ws_receives_latest_on_connect(self):
        """Mid-run client gets latest state immediately."""
        latest = {
            "event": "progress",
            "phase": "evaluating",
            "process": {"label": "Evaluating", "completed": 3, "total": 10},
        }
        client, _ = _make_app({"running": True, "latest_event": latest, "latest_process": None, "latest_item": None})
        with client.websocket_connect("/ws/training") as ws:
            data = ws.receive_json()
            assert data["event"] == "progress"
            assert data["phase"] == "evaluating"

    def test_ws_broadcasts_queue_events(self):
        """Events put on the queue are broadcast to connected client."""
        client, app = _make_app()
        queue: asyncio.Queue = app.state.progress_queue

        event = {
            "event": "phase_start",
            "phase": "training",
            "timestamp": 1234567890.0,
            "summary": {"generation": 1},
        }

        with client.websocket_connect("/ws/training") as ws:
            # Put event on the queue — the WS handler should pick it up
            queue.put_nowait(event)
            data = ws.receive_json()
            assert data["event"] == "phase_start"
            assert data["phase"] == "training"

    def test_ws_run_complete_sets_not_running(self):
        """run_complete event sets training_state.running to False."""
        client, app = _make_app({"running": True, "latest_event": None, "latest_process": None, "latest_item": None})
        queue: asyncio.Queue = app.state.progress_queue

        event = {"event": "run_complete", "timestamp": 1234567890.0}
        with client.websocket_connect("/ws/training") as ws:
            queue.put_nowait(event)
            data = ws.receive_json()
            assert data["event"] == "run_complete"

        assert app.state.training_state["running"] is False

    def test_ws_phase_start_sets_running(self):
        """phase_start event sets running to True."""
        client, app = _make_app({"running": False, "latest_event": None, "latest_process": None, "latest_item": None})
        queue: asyncio.Queue = app.state.progress_queue

        event = {"event": "phase_start", "phase": "training", "timestamp": 123.0}
        with client.websocket_connect("/ws/training") as ws:
            queue.put_nowait(event)
            data = ws.receive_json()
            assert data["event"] == "phase_start"

        assert app.state.training_state["running"] is True

    def test_ws_updates_latest_event(self):
        """Each event updates the latest_event for future connections."""
        client, app = _make_app()
        queue: asyncio.Queue = app.state.progress_queue

        event1 = {"event": "progress", "phase": "training", "detail": "ep 1"}
        event2 = {"event": "progress", "phase": "training", "detail": "ep 2"}

        with client.websocket_connect("/ws/training") as ws:
            queue.put_nowait(event1)
            ws.receive_json()
            queue.put_nowait(event2)
            ws.receive_json()

        assert app.state.training_state["latest_event"]["detail"] == "ep 2"


# ── Worker Connected Tests ──────────────────────────────────────────


class TestWorkerConnected:
    def test_status_includes_worker_connected_when_idle(self):
        """Status response should include worker_connected field."""
        client, app = _make_app()
        app.state.worker_connected = True
        resp = client.get("/training/status")
        data = resp.json()
        assert data["worker_connected"] is True

    def test_status_worker_disconnected_default(self):
        """worker_connected should default to False when not set."""
        client, _ = _make_app()
        resp = client.get("/training/status")
        data = resp.json()
        assert data["worker_connected"] is False

    def test_status_worker_disconnected_while_running(self):
        """When running but worker disconnected, phase should be worker_disconnected."""
        client, app = _make_app({
            "running": True,
            "latest_event": None,
            "latest_process": None,
            "latest_item": None,
        })
        app.state.worker_connected = False
        resp = client.get("/training/status")
        data = resp.json()
        assert data["running"] is True
        assert data["phase"] == "worker_disconnected"
        assert data["worker_connected"] is False

    def test_status_worker_connected_while_running(self):
        """When running and worker connected, worker_connected should be True."""
        client, app = _make_app({
            "running": True,
            "latest_event": {"event": "progress", "phase": "training"},
            "latest_process": None,
            "latest_item": None,
        })
        app.state.worker_connected = True
        resp = client.get("/training/status")
        data = resp.json()
        assert data["running"] is True
        assert data["worker_connected"] is True
        assert data["phase"] == "training"


# ── Architectures and Genetics Endpoints ─────────────────────────────


class TestArchitecturesEndpoint:
    def test_returns_list_of_architectures(self):
        """GET /training/architectures returns all registered architectures."""
        # Import policy_network to populate the registry (decorator side effects)
        import agents.policy_network  # noqa: F401

        client, app = _make_app()
        app.state.config = {
            "hyperparameters": {
                "search_ranges": {
                    "architecture_name": {"choices": ["ppo_lstm_v1"]}
                }
            }
        }
        resp = client.get("/training/architectures")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        # Each entry has name + description
        for item in data:
            assert "name" in item
            assert "description" in item
            assert item["description"]  # non-empty
        # Known architectures are present
        names = [a["name"] for a in data]
        assert "ppo_lstm_v1" in names

    def test_architecture_defaults_endpoint(self):
        """GET /training/architectures/defaults returns config choices."""
        client, app = _make_app()
        app.state.config = {
            "hyperparameters": {
                "search_ranges": {
                    "architecture_name": {
                        "choices": ["ppo_lstm_v1", "ppo_time_lstm_v1"]
                    }
                }
            }
        }
        resp = client.get("/training/architectures/defaults")
        assert resp.status_code == 200
        data = resp.json()
        assert data["defaults"] == ["ppo_lstm_v1", "ppo_time_lstm_v1"]


class TestHyperparameterSchemaEndpoint:
    def test_returns_full_schema(self):
        """GET /training/hyperparameter-schema returns every gene from config."""
        client, app = _make_app()
        app.state.config = {
            "hyperparameters": {
                "search_ranges": {
                    "learning_rate": {"type": "float_log", "min": 1e-5, "max": 5e-4},
                    "lstm_layer_norm": {"type": "int_choice", "choices": [0, 1]},
                    "architecture_name": {
                        "type": "str_choice",
                        "choices": ["ppo_lstm_v1", "ppo_time_lstm_v1"],
                    },
                }
            }
        }
        resp = client.get("/training/hyperparameter-schema")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 3
        names = [e["name"] for e in data]
        assert "learning_rate" in names
        assert "lstm_layer_norm" in names
        assert "architecture_name" in names

    def test_schema_entry_shape(self):
        """Each entry contains name/type/min/max/choices/source_file."""
        client, app = _make_app()
        app.state.config = {
            "hyperparameters": {
                "search_ranges": {
                    "learning_rate": {"type": "float_log", "min": 1e-5, "max": 5e-4},
                    "lstm_num_layers": {"type": "int_choice", "choices": [1, 2]},
                }
            }
        }
        resp = client.get("/training/hyperparameter-schema")
        data = {e["name"]: e for e in resp.json()}
        lr = data["learning_rate"]
        assert lr["type"] == "float_log"
        assert lr["min"] == 1e-5
        assert lr["max"] == 5e-4
        assert lr["choices"] is None
        assert lr["source_file"] == "config.yaml#hyperparameters.search_ranges.learning_rate"
        ll = data["lstm_num_layers"]
        assert ll["type"] == "int_choice"
        assert ll["choices"] == [1, 2]
        assert ll["min"] is None
        assert ll["max"] is None

    def test_empty_config_returns_empty_list(self):
        """No search_ranges -> empty list (not 500)."""
        client, app = _make_app()
        app.state.config = {}
        resp = client.get("/training/hyperparameter-schema")
        assert resp.status_code == 200
        assert resp.json() == []


class TestGeneticsEndpoint:
    def test_returns_genetics_info(self):
        """GET /training/genetics returns population config."""
        client, app = _make_app()
        app.state.config = {
            "population": {
                "size": 50,
                "n_elite": 5,
                "selection_top_pct": 0.5,
                "mutation_rate": 0.3,
            }
        }
        resp = client.get("/training/genetics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["population_size"] == 50
        assert data["n_elite"] == 5
        assert data["selection_top_pct"] == 0.5
        assert data["mutation_rate"] == 0.3

    def test_returns_defaults_when_config_missing(self):
        """Missing population config returns sensible defaults."""
        client, app = _make_app()
        app.state.config = {}
        resp = client.get("/training/genetics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["population_size"] == 50
        assert data["n_elite"] == 5


class TestStopGranularity:
    """Tests for the stop endpoint with granularity parameter."""

    def test_stop_defaults_to_immediate(self):
        """POST /training/stop without granularity defaults to immediate."""
        client, app = _make_app({"running": True, "latest_event": None, "latest_process": None, "latest_item": None})
        # Mock worker_ws to capture the sent command
        from unittest.mock import AsyncMock
        import asyncio
        ws_mock = AsyncMock()
        ws_mock.send = AsyncMock()
        app.state.worker_ws = ws_mock

        # The pending response future will capture what was sent
        sent_messages = []
        original_send = ws_mock.send
        async def capture_send(msg):
            sent_messages.append(json.loads(msg))
            # Resolve the pending future
            fut = app.state.worker_pending_response
            if fut and not fut.done():
                fut.set_result({"type": "status", "running": True})
        ws_mock.send = capture_send

        resp = client.post("/training/stop")
        assert resp.status_code == 200
        # Verify the IPC message contained granularity=immediate
        assert len(sent_messages) == 1
        assert sent_messages[0]["granularity"] == "immediate"

    def test_stop_with_eval_all(self):
        """POST /training/stop?granularity=eval_all sends correct IPC."""
        client, app = _make_app({"running": True, "latest_event": None, "latest_process": None, "latest_item": None})
        from unittest.mock import AsyncMock
        ws_mock = AsyncMock()
        sent_messages = []
        async def capture_send(msg):
            sent_messages.append(json.loads(msg))
            fut = app.state.worker_pending_response
            if fut and not fut.done():
                fut.set_result({"type": "status", "running": True})
        ws_mock.send = capture_send
        app.state.worker_ws = ws_mock

        resp = client.post("/training/stop?granularity=eval_all")
        assert resp.status_code == 200
        assert sent_messages[0]["granularity"] == "eval_all"

    def test_stop_with_invalid_granularity(self):
        """POST /training/stop?granularity=bogus returns 422."""
        client, app = _make_app({"running": True, "latest_event": None, "latest_process": None, "latest_item": None})
        resp = client.post("/training/stop?granularity=bogus")
        assert resp.status_code == 422


class TestStatusEvalFields:
    """Status endpoint includes unevaluated_count and eval_rate_s."""

    def test_status_includes_eval_fields_when_evaluating(self):
        state = {
            "running": True,
            "latest_event": {
                "event": "progress",
                "phase": "evaluating",
                "unevaluated_count": 5,
                "eval_rate_s": 45.2,
            },
            "latest_process": None,
            "latest_item": None,
        }
        client, app = _make_app(state)
        app.state.worker_connected = True
        resp = client.get("/training/status")
        data = resp.json()
        assert data["unevaluated_count"] == 5
        assert data["eval_rate_s"] == 45.2

    def test_status_eval_fields_null_when_training(self):
        state = {
            "running": True,
            "latest_event": {
                "event": "progress",
                "phase": "training",
            },
            "latest_process": None,
            "latest_item": None,
        }
        client, app = _make_app(state)
        app.state.worker_connected = True
        resp = client.get("/training/status")
        data = resp.json()
        assert data["unevaluated_count"] is None
        assert data["eval_rate_s"] is None


class TestStartWithOverrides:
    def test_start_rejects_unknown_architecture(self):
        """POST /training/start with unknown architecture returns 400."""
        import tempfile
        from pathlib import Path
        import pandas as pd

        client, app = _make_app()
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            pd.DataFrame({"x": [1]}).to_parquet(data_dir / "2026-01-01.parquet")
            app.state.config = {
                "paths": {"processed_data": str(data_dir)},
                "population": {"size": 2},
                "training": {},
                "hyperparameters": {"search_ranges": {"architecture_name": {"choices": []}}},
            }

            resp = client.post("/training/start", json={
                "n_generations": 1,
                "n_epochs": 1,
                "architectures": ["nonexistent_arch_xyz"],
            })
            assert resp.status_code == 400
            assert "unknown" in resp.json()["detail"].lower() or "architectures" in resp.json()["detail"].lower()

    def test_start_rejects_empty_architectures(self):
        """POST /training/start with empty architectures list returns 400."""
        import tempfile
        from pathlib import Path
        import pandas as pd

        client, app = _make_app()
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            pd.DataFrame({"x": [1]}).to_parquet(data_dir / "2026-01-01.parquet")
            app.state.config = {
                "paths": {"processed_data": str(data_dir)},
                "population": {"size": 2},
                "training": {},
                "hyperparameters": {"search_ranges": {"architecture_name": {"choices": []}}},
            }

            resp = client.post("/training/start", json={
                "n_generations": 1,
                "n_epochs": 1,
                "architectures": [],
            })
            assert resp.status_code == 400

    def test_start_accepts_valid_architectures_without_torch(self):
        """Regression: validation must accept architecture names listed in the
        config choices, without requiring agents.policy_network (and torch) to
        have been imported in the API process.

        Previously the validation imported REGISTRY from architecture_registry,
        which is only populated when policy_network is imported — and the API
        deliberately avoids importing torch.  The result was that every
        architecture name was rejected as "unknown", breaking the training
        wizard on the final step.
        """
        import tempfile
        from pathlib import Path
        import pandas as pd

        client, app = _make_app()
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            pd.DataFrame({"x": [1]}).to_parquet(data_dir / "2026-01-01.parquet")
            app.state.config = {
                "paths": {"processed_data": str(data_dir)},
                "population": {"size": 2},
                "training": {},
                "hyperparameters": {
                    "search_ranges": {
                        "architecture_name": {
                            "choices": ["ppo_lstm_v1", "ppo_time_lstm_v1"],
                        }
                    }
                },
            }
            # No worker_ws set — _send_to_worker will raise 503.  We only care
            # that validation passes, i.e. the response is NOT a 400.
            app.state.worker_ws = None

            resp = client.post("/training/start", json={
                "n_generations": 1,
                "n_epochs": 1,
                "architectures": ["ppo_lstm_v1", "ppo_time_lstm_v1"],
            })
            assert resp.status_code != 400, (
                f"Expected validation to pass, got 400: {resp.json()}"
            )
            assert resp.status_code == 503  # worker not available — validation passed

    def test_start_validates_against_config_not_registry(self):
        """An architecture that exists in the runtime REGISTRY but not in
        config.choices must be rejected, and vice-versa — validation is
        driven by config, not by the runtime registry."""
        import tempfile
        from pathlib import Path
        import pandas as pd

        client, app = _make_app()
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            pd.DataFrame({"x": [1]}).to_parquet(data_dir / "2026-01-01.parquet")
            app.state.config = {
                "paths": {"processed_data": str(data_dir)},
                "population": {"size": 2},
                "training": {},
                "hyperparameters": {
                    "search_ranges": {
                        "architecture_name": {"choices": ["only_this_one"]}
                    }
                },
            }
            app.state.worker_ws = None

            # "ppo_lstm_v1" is a real registered arch but not in this config's
            # choices → must be rejected.
            resp = client.post("/training/start", json={
                "n_generations": 1,
                "n_epochs": 1,
                "architectures": ["ppo_lstm_v1"],
            })
            assert resp.status_code == 400


# ── Smoke-test gate (Session 04, naked-clip-and-stability) ─────────


class TestSmokeTestGateDTO:
    """The HTTP-layer plumbing for the smoke-test gate.

    hard_constraints §15 says default ON. These tests verify:

    - ``smoke_test_first`` defaults to True on the request DTO.
    - The flag flows through to the worker start command.
    - A worker EVT_ERROR carrying ``smoke_test_result`` surfaces as a
      409 whose ``detail`` is a structured dict the frontend can
      render into the failure modal without inspecting strings.
    """

    def _make_app_with_fake_worker(self, reply: dict):
        """_make_app plus a fake worker_ws that resolves the pending
        response future with ``reply``. ``reply`` is dispatched in the
        background so the HTTP call can await it synchronously."""
        from pathlib import Path
        import pandas as pd

        client, app = _make_app()
        tmp = tempfile.mkdtemp()
        data_dir = Path(tmp)
        for d in ("2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"):
            pd.DataFrame({"x": [1]}).to_parquet(data_dir / f"{d}.parquet")
        app.state.config = {
            "paths": {"processed_data": str(data_dir)},
            "population": {"size": 2},
            "training": {},
            "hyperparameters": {
                "search_ranges": {
                    "architecture_name": {"choices": ["ppo_lstm_v1"]},
                },
            },
        }

        sent_messages: list[dict] = []

        class _FakeWorkerWS:
            async def send(self, raw: str):
                sent_messages.append(json.loads(raw))
                # Resolve the pending future on the next loop tick.
                loop = asyncio.get_event_loop()
                def _resolve():
                    pending = app.state.worker_pending_response
                    if pending and not pending.done():
                        pending.set_result(reply)
                loop.call_soon(_resolve)

        app.state.worker_ws = _FakeWorkerWS()
        app.state.worker_pending_response = None
        return client, app, sent_messages

    def test_smoke_test_first_defaults_true_on_dto(self):
        """Pydantic default is True — clients that don't send the field
        still trigger the gate (hard_constraints §15)."""
        from api.schemas import StartTrainingRequest
        req = StartTrainingRequest()
        assert req.smoke_test_first is True

    def test_flag_forwarded_to_worker(self):
        reply = {
            "type": "started",
            "run_id": "r1",
            "train_days": [],
            "test_days": [],
            "plan_id": None,
            "smoke_test_result": {
                "passed": True,
                "assertions": [],
                "probe_model_ids": ["smoke-transformer", "smoke-lstm"],
            },
        }
        client, app, sent = self._make_app_with_fake_worker(reply)
        resp = client.post("/training/start", json={
            "n_generations": 1,
            "n_epochs": 1,
            "architectures": ["ppo_lstm_v1"],
            "smoke_test_first": True,
        })
        assert resp.status_code == 200, resp.text
        assert resp.json()["smoke_test_result"]["passed"] is True
        assert sent and sent[0]["smoke_test_first"] is True

    def test_smoke_fail_surfaces_as_structured_409(self):
        """Worker EVT_ERROR with smoke_test_result → 409 + detail dict.

        The frontend relies on the ``detail.smoke_test_result`` shape
        — regression-proofing that contract so future refactors can't
        silently drop it."""
        failure_payload = {
            "passed": False,
            "assertions": [
                {
                    "name": "ep1_policy_loss",
                    "passed": False,
                    "observed": 1.0e17,
                    "threshold": 100.0,
                    "detail": "ep1 policy_loss blew up on the transformer",
                },
            ],
            "probe_model_ids": ["smoke-ppo_transformer_v1"],
        }
        reply = {
            "type": "error",
            "message": "Smoke test failed — full population not launched",
            "smoke_test_result": failure_payload,
        }
        client, _app, _sent = self._make_app_with_fake_worker(reply)
        resp = client.post("/training/start", json={
            "n_generations": 1,
            "n_epochs": 1,
            "architectures": ["ppo_lstm_v1"],
            "smoke_test_first": True,
        })
        assert resp.status_code == 409, resp.text
        detail = resp.json()["detail"]
        assert isinstance(detail, dict)
        assert detail["smoke_test_result"]["passed"] is False
        assert detail["smoke_test_result"]["assertions"][0]["name"] == "ep1_policy_loss"

    def test_ipc_make_start_cmd_carries_flag(self):
        from training.ipc import make_start_cmd
        raw = make_start_cmd(smoke_test_first=True)
        payload = json.loads(raw)
        assert payload["smoke_test_first"] is True
        # Default is False — an absent flag (legacy call) stays disabled.
        default = json.loads(make_start_cmd())
        assert default["smoke_test_first"] is False

    def test_make_error_msg_passes_smoke_result_through(self):
        from training.ipc import make_error_msg
        payload = {"passed": False, "assertions": [], "probe_model_ids": []}
        raw = make_error_msg("fail", smoke_test_result=payload)
        parsed = json.loads(raw)
        assert parsed["smoke_test_result"] == payload
        # Legacy one-arg form stays identical to pre-change output.
        legacy = json.loads(make_error_msg("fail"))
        assert "smoke_test_result" not in legacy


# ── Episodes Endpoint Tests ──────────────────────────────────────────


class TestTrainingEpisodes:
    """GET /training/episodes backs the learning-curves UI."""

    def _mount(self, tmp_path, lines: list[str]):
        client, app = _make_app()
        app.state.config = {"paths": {"logs": str(tmp_path)}}
        log_dir = tmp_path / "training"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "episodes.jsonl"
        log_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return client, app

    def _row(self, episode: int, ts: float, **overrides) -> str:
        base = {
            "episode": episode,
            "model_id": "m1",
            "day_date": "2026-04-06",
            "total_reward": -100.0,
            "total_pnl": -50.0,
            "bet_count": 10,
            "policy_loss": 0.15,
            "value_loss": 5.0,
            "entropy": 100.0,
            "arbs_completed": 2,
            "arbs_naked": 8,
            "timestamp": ts,
        }
        base.update(overrides)
        return json.dumps(base)

    def test_returns_empty_when_file_missing(self, tmp_path):
        client, app = _make_app()
        app.state.config = {"paths": {"logs": str(tmp_path / "does-not-exist")}}
        resp = client.get("/training/episodes")
        assert resp.status_code == 200
        assert resp.json() == {"episodes": [], "latest_ts": None, "truncated": False}

    def test_returns_all_rows_when_under_limit(self, tmp_path):
        client, _ = self._mount(tmp_path, [
            self._row(1, 1000.0),
            self._row(2, 1060.0),
            self._row(3, 1120.0),
        ])
        resp = client.get("/training/episodes")
        body = resp.json()
        assert len(body["episodes"]) == 3
        assert body["latest_ts"] == 1120.0
        assert body["truncated"] is False

    def test_respects_limit_parameter(self, tmp_path):
        rows = [self._row(i, 1000.0 + i * 60) for i in range(1, 6)]
        client, _ = self._mount(tmp_path, rows)
        resp = client.get("/training/episodes?limit=2")
        body = resp.json()
        assert len(body["episodes"]) == 2
        # When limited, returns the *most recent* rows.
        assert body["episodes"][0]["episode"] == 4
        assert body["episodes"][1]["episode"] == 5

    def test_since_ts_numeric_returns_only_newer_rows(self, tmp_path):
        client, _ = self._mount(tmp_path, [
            self._row(1, 1000.0),
            self._row(2, 1060.0),
            self._row(3, 1120.0),
            self._row(4, 1180.0),
        ])
        resp = client.get("/training/episodes?since_ts=1060")
        body = resp.json()
        assert [e["episode"] for e in body["episodes"]] == [3, 4]
        assert body["latest_ts"] == 1180.0

    def test_since_ts_accepts_iso_string(self, tmp_path):
        """ISO timestamp in since_ts is parsed to epoch on the fly."""
        # 1000.0 epoch = 1970-01-01T00:16:40Z. Two rows either side.
        client, _ = self._mount(tmp_path, [
            self._row(1, 900.0),
            self._row(2, 1100.0),
        ])
        # since_ts = 1000 epoch → 1970-01-01T00:16:40Z
        resp = client.get("/training/episodes?since_ts=1970-01-01T00:16:40Z")
        body = resp.json()
        assert [e["episode"] for e in body["episodes"]] == [2]

    def test_since_ts_strict_inequality(self, tmp_path):
        """since_ts == some row's ts excludes that row (next poll shouldn't
        double-count the last seen row)."""
        client, _ = self._mount(tmp_path, [
            self._row(1, 1000.0),
            self._row(2, 1060.0),
        ])
        resp = client.get("/training/episodes?since_ts=1060")
        assert resp.json()["episodes"] == []

    def test_skips_malformed_json_lines(self, tmp_path):
        client, _ = self._mount(tmp_path, [
            self._row(1, 1000.0),
            "{not valid json",
            "",
            self._row(2, 1120.0),
        ])
        resp = client.get("/training/episodes")
        body = resp.json()
        assert [e["episode"] for e in body["episodes"]] == [1, 2]

    def test_invalid_since_ts_treated_as_no_filter(self, tmp_path):
        """A garbage since_ts falls through rather than 400ing; a stale
        or malformed client cursor shouldn't hang the UI."""
        client, _ = self._mount(tmp_path, [
            self._row(1, 1000.0),
            self._row(2, 1120.0),
        ])
        resp = client.get("/training/episodes?since_ts=not-a-number")
        body = resp.json()
        assert [e["episode"] for e in body["episodes"]] == [1, 2]
