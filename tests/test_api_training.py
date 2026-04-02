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
