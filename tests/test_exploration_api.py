"""Tests for the exploration / coverage dashboard API endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import exploration as exploration_router
from agents.population_manager import parse_search_ranges
from registry.model_store import ModelStore


@pytest.fixture
def hp_ranges() -> dict[str, dict]:
    return {
        "learning_rate": {"type": "float_log", "min": 1.0e-5, "max": 5.0e-4},
        "gamma": {"type": "float", "min": 0.95, "max": 0.999},
        "architecture_name": {
            "type": "str_choice",
            "choices": ["ppo_lstm_v1", "ppo_time_lstm_v1"],
        },
    }


@pytest.fixture
def store(tmp_path: Path) -> ModelStore:
    return ModelStore(
        db_path=tmp_path / "test.db",
        weights_dir=tmp_path / "weights",
    )


@pytest.fixture
def client(store: ModelStore, hp_ranges: dict) -> TestClient:
    app = FastAPI()
    app.include_router(exploration_router.router, prefix="/api")
    app.state.store = store
    app.state.config = {"hyperparameters": {"search_ranges": hp_ranges}}
    return TestClient(app)


class TestExplorationAPI:
    def test_history_empty(self, client: TestClient):
        resp = client.get("/api/exploration/history")
        assert resp.status_code == 200
        body = resp.json()
        assert body["runs"] == []
        assert body["count"] == 0

    def test_history_with_runs(self, client: TestClient, store: ModelStore):
        store.record_exploration_run("r1", {"lr": 0.01}, strategy="sobol")
        store.record_exploration_run("r2", {"lr": 0.02}, strategy="coverage")
        resp = client.get("/api/exploration/history")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        assert body["runs"][0]["strategy"] == "sobol"

    def test_coverage_endpoint(self, client: TestClient):
        resp = client.get("/api/exploration/coverage")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_agents"] == 0
        assert isinstance(body["genes"], list)
        assert isinstance(body["arch_counts"], dict)

    def test_suggested_seed_endpoint(self, client: TestClient):
        resp = client.get("/api/exploration/suggested-seed")
        assert resp.status_code == 200
        body = resp.json()
        assert "seed_point" in body
        assert isinstance(body["seed_point"], dict)
        assert "poorly_covered_genes" in body
