"""Tests for the manual evaluation API (POST /evaluate, GET /evaluate/status)."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import evaluation, training


@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield


def _make_app(
    *,
    running: bool = False,
    available_dates: list[str] | None = None,
    models: dict[str, dict] | None = None,
) -> tuple[TestClient, FastAPI, list]:
    """Build a minimal FastAPI app wired to the evaluation router.

    Returns (client, app, sent_messages) — sent_messages captures any IPC
    commands routed through the mock worker WS so individual tests can
    assert on the wire format.
    """
    app = FastAPI(lifespan=_lifespan)
    app.include_router(training.router)
    app.include_router(evaluation.router)

    app.state.training_state = {
        "running": running,
        "latest_event": None,
        "latest_process": None,
        "latest_item": None,
        "latest_overall": None,
        "plan_id": None,
    }
    app.state.progress_queue = asyncio.Queue()
    app.state.ws_clients = set()
    app.state.worker_connected = True
    app.state.worker_pending_response = None

    # Mock the model store
    store = MagicMock()
    models = models or {}

    def _get_model(mid: str):
        rec = models.get(mid)
        if rec is None:
            return None
        m = MagicMock()
        m.weights_path = rec.get("weights_path", "/tmp/fake.pt")
        m.hyperparameters = rec.get("hyperparameters", {})
        m.architecture_name = rec.get("architecture_name", "ppo_lstm_v1")
        return m

    store.get_model.side_effect = _get_model
    app.state.store = store

    # Provide a config + a fake processed-data dir so the date validator works
    tmp_data = Path(__file__).parent / "_evaluate_tmp_data"
    tmp_data.mkdir(exist_ok=True)
    # Cleanup any previous parquet stubs
    for p in tmp_data.glob("*.parquet"):
        p.unlink()
    for d in available_dates or []:
        (tmp_data / f"{d}.parquet").write_bytes(b"")

    app.state.config = {
        "paths": {"processed_data": str(tmp_data)},
        "training": {"max_runners": 14},
    }

    # Mock the worker WebSocket — every send() captures the message and
    # immediately resolves the pending future with a started ack.
    sent_messages: list = []
    ws_mock = AsyncMock()

    async def _capture_send(msg):
        sent_messages.append(json.loads(msg))
        fut = app.state.worker_pending_response
        if fut and not fut.done():
            fut.set_result({
                "type": "started",
                "kind": "evaluate",
                "run_id": "job-xyz",
            })

    ws_mock.send = _capture_send
    app.state.worker_ws = ws_mock

    client = TestClient(app)
    client.__enter__()
    return client, app, sent_messages


# ── POST /evaluate ──────────────────────────────────────────────────


class TestEvaluateRequest:
    def test_rejects_when_worker_busy(self):
        client, _, _ = _make_app(running=True, available_dates=["2026-03-01"],
                                 models={"m1": {}})
        resp = client.post("/evaluate", json={"model_ids": ["m1"]})
        assert resp.status_code == 409

    def test_rejects_empty_model_ids(self):
        client, _, _ = _make_app(available_dates=["2026-03-01"])
        resp = client.post("/evaluate", json={"model_ids": []})
        assert resp.status_code == 400

    def test_rejects_unknown_model_ids(self):
        client, _, _ = _make_app(available_dates=["2026-03-01"], models={"m1": {}})
        resp = client.post("/evaluate", json={"model_ids": ["m1", "missing"]})
        assert resp.status_code == 400
        assert "missing" in resp.json()["detail"]

    def test_rejects_models_without_weights(self):
        client, _, _ = _make_app(
            available_dates=["2026-03-01"],
            models={"m1": {"weights_path": ""}},
        )
        resp = client.post("/evaluate", json={"model_ids": ["m1"]})
        assert resp.status_code == 400
        assert "weights" in resp.json()["detail"]

    def test_rejects_unknown_test_dates(self):
        client, _, _ = _make_app(
            available_dates=["2026-03-01", "2026-03-02"],
            models={"m1": {}},
        )
        resp = client.post(
            "/evaluate",
            json={"model_ids": ["m1"], "test_dates": ["2026-03-99"]},
        )
        assert resp.status_code == 400

    def test_accepts_with_all_dates_when_test_dates_null(self):
        client, _, sent = _make_app(
            available_dates=["2026-03-01", "2026-03-02"],
            models={"m1": {}},
        )
        resp = client.post("/evaluate", json={"model_ids": ["m1"]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["accepted"] is True
        assert body["model_count"] == 1
        assert body["day_count"] == 2

        # Worker received a CMD_EVALUATE with all dates expanded
        assert len(sent) == 1
        cmd = sent[0]
        assert cmd["type"] == "evaluate"
        assert cmd["model_ids"] == ["m1"]
        assert cmd["test_dates"] == ["2026-03-01", "2026-03-02"]

    def test_accepts_with_explicit_dates(self):
        client, _, sent = _make_app(
            available_dates=["2026-03-01", "2026-03-02", "2026-03-03"],
            models={"m1": {}, "m2": {}},
        )
        resp = client.post(
            "/evaluate",
            json={"model_ids": ["m1", "m2"], "test_dates": ["2026-03-01", "2026-03-03"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_count"] == 2
        assert body["day_count"] == 2
        assert sent[0]["test_dates"] == ["2026-03-01", "2026-03-03"]


# ── GET /evaluate/status ────────────────────────────────────────────


class TestEvaluateStatus:
    def test_idle_status(self):
        client, _, _ = _make_app(available_dates=["2026-03-01"])
        resp = client.get("/evaluate/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["running"] is False
        assert body["phase"] is None
        assert body["manual_evaluation"] is False

    def test_running_with_progress(self):
        client, app, _ = _make_app(running=True, available_dates=["2026-03-01"])
        process_snap = {
            "label": "Evaluating 3 models",
            "completed": 1,
            "total": 3,
            "pct": 33.3,
            "item_eta_human": "30s",
            "process_eta_human": "1m",
        }
        app.state.training_state.update({
            "latest_event": {
                "event": "progress",
                "phase": "evaluating",
                "process": process_snap,
                "detail": "Evaluating model abc (1/3)",
                "summary": {"manual_evaluation": True},
            },
            "latest_process": process_snap,
        })
        resp = client.get("/evaluate/status")
        body = resp.json()
        assert body["running"] is True
        assert body["phase"] == "evaluating"
        assert body["process"]["completed"] == 1
        assert body["process"]["total"] == 3
        assert body["manual_evaluation"] is True
