"""Unit tests for api/routers/models.py — scoreboard, detail, lineage, genetics."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from registry.model_store import ModelStore, EvaluationDayRecord, GeneticEventRecord
from registry.scoreboard import Scoreboard


# ── Helpers ──────────────────────────────────────────────────────────


def _make_app(store: ModelStore, config: dict) -> TestClient:
    """Create a TestClient with a real store + scoreboard wired in."""
    from fastapi import FastAPI
    from api.routers import models

    app = FastAPI()
    app.include_router(models.router)

    scoreboard = Scoreboard(store=store, config=config)
    app.state.store = store
    app.state.scoreboard = scoreboard

    return TestClient(app)


def _test_config() -> dict:
    return {
        "reward": {
            "coefficients": {
                "win_rate": 0.35,
                "sharpe": 0.30,
                "mean_daily_pnl": 0.15,
                "efficiency": 0.20,
            }
        },
        "training": {"starting_budget": 100.0},
    }


def _create_store(tmp_dir: str) -> ModelStore:
    db_path = str(Path(tmp_dir) / "test.db")
    weights_dir = str(Path(tmp_dir) / "weights")
    bet_logs_dir = str(Path(tmp_dir) / "bet_logs")
    return ModelStore(db_path=db_path, weights_dir=weights_dir, bet_logs_dir=bet_logs_dir)


def _seed_models(store: ModelStore) -> tuple[str, str, str]:
    """Create 3 models: grandparent → parent → child. Returns (gp_id, p_id, c_id)."""
    gp_id = store.create_model(
        generation=0,
        architecture_name="ppo_lstm_v1",
        architecture_description="Test arch",
        hyperparameters={"learning_rate": 0.001, "lstm_hidden_size": 128},
    )
    p_id = store.create_model(
        generation=1,
        architecture_name="ppo_lstm_v1",
        architecture_description="Test arch",
        hyperparameters={"learning_rate": 0.0005, "lstm_hidden_size": 256},
        parent_a_id=gp_id,
    )
    c_id = store.create_model(
        generation=2,
        architecture_name="ppo_lstm_v1",
        architecture_description="Test arch",
        hyperparameters={"learning_rate": 0.0003, "lstm_hidden_size": 512},
        parent_a_id=p_id,
        parent_b_id=gp_id,
    )
    return gp_id, p_id, c_id


def _seed_evaluation(store: ModelStore, model_id: str):
    """Create an evaluation run with 2 test days."""
    run_id = store.create_evaluation_run(
        model_id=model_id,
        train_cutoff_date="2026-03-25",
        test_days=["2026-03-26", "2026-03-27"],
    )
    store.record_evaluation_day(
        EvaluationDayRecord(
            run_id=run_id,
            date="2026-03-26",
            day_pnl=50.0,
            bet_count=10,
            winning_bets=7,
            bet_precision=0.7,
            pnl_per_bet=5.0,
            early_picks=2,
            profitable=True,
        )
    )
    store.record_evaluation_day(
        EvaluationDayRecord(
            run_id=run_id,
            date="2026-03-27",
            day_pnl=-20.0,
            bet_count=8,
            winning_bets=3,
            bet_precision=0.375,
            pnl_per_bet=-2.5,
            early_picks=1,
            profitable=False,
        )
    )
    return run_id


# ── Scoreboard Tests ─────────────────────────────────────────────────


class TestScoreboard:
    def test_empty_scoreboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            client = _make_app(store, _test_config())
            resp = client.get("/models")
            assert resp.status_code == 200
            data = resp.json()
            assert data["models"] == []

    def test_scoreboard_returns_ranked_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            gp_id, p_id, c_id = _seed_models(store)
            _seed_evaluation(store, gp_id)
            _seed_evaluation(store, p_id)

            client = _make_app(store, _test_config())
            resp = client.get("/models")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["models"]) >= 2
            # All entries have required fields
            for m in data["models"]:
                assert "model_id" in m
                assert "generation" in m
                assert "architecture_name" in m
                assert "composite_score" in m
                assert "win_rate" in m
                assert "sharpe" in m

    def test_scoreboard_sorted_by_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            gp_id, p_id, _ = _seed_models(store)
            _seed_evaluation(store, gp_id)
            _seed_evaluation(store, p_id)

            client = _make_app(store, _test_config())
            resp = client.get("/models")
            models = resp.json()["models"]
            if len(models) >= 2:
                scores = [m["composite_score"] for m in models if m["composite_score"] is not None]
                assert scores == sorted(scores, reverse=True)


# ── Model Detail Tests ───────────────────────────────────────────────


class TestModelDetail:
    def test_model_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            client = _make_app(store, _test_config())
            resp = client.get("/models/nonexistent-id")
            assert resp.status_code == 404

    def test_model_detail_returns_correct_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            gp_id, _, c_id = _seed_models(store)
            _seed_evaluation(store, c_id)

            client = _make_app(store, _test_config())
            resp = client.get(f"/models/{c_id}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["model_id"] == c_id
            assert data["generation"] == 2
            assert data["architecture_name"] == "ppo_lstm_v1"
            assert data["hyperparameters"]["lstm_hidden_size"] == 512
            assert data["parent_a_id"] is not None
            assert data["parent_b_id"] == gp_id

    def test_model_detail_includes_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            gp_id, _, _ = _seed_models(store)
            _seed_evaluation(store, gp_id)

            client = _make_app(store, _test_config())
            resp = client.get(f"/models/{gp_id}")
            data = resp.json()
            assert len(data["metrics_history"]) == 2
            day1 = data["metrics_history"][0]
            assert day1["date"] == "2026-03-26"
            assert day1["day_pnl"] == 50.0
            assert day1["profitable"] is True

    def test_model_detail_no_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            gp_id, _, _ = _seed_models(store)

            client = _make_app(store, _test_config())
            resp = client.get(f"/models/{gp_id}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["metrics_history"] == []


# ── Lineage Tests ────────────────────────────────────────────────────


class TestLineage:
    def test_lineage_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            client = _make_app(store, _test_config())
            resp = client.get("/models/nonexistent/lineage")
            assert resp.status_code == 404

    def test_lineage_root_model(self):
        """A generation-0 model with no parents returns just itself."""
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            gp_id, _, _ = _seed_models(store)

            client = _make_app(store, _test_config())
            resp = client.get(f"/models/{gp_id}/lineage")
            assert resp.status_code == 200
            nodes = resp.json()["nodes"]
            assert len(nodes) == 1
            assert nodes[0]["model_id"] == gp_id
            assert nodes[0]["parent_a_id"] is None

    def test_lineage_traverses_parents(self):
        """Child with two parents → should return child + both parents + grandparent."""
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            gp_id, p_id, c_id = _seed_models(store)

            client = _make_app(store, _test_config())
            resp = client.get(f"/models/{c_id}/lineage")
            nodes = resp.json()["nodes"]
            node_ids = {n["model_id"] for n in nodes}
            # Child has parent_a=p_id, parent_b=gp_id
            # p_id has parent_a=gp_id
            # So we should see: c_id, p_id, gp_id
            assert c_id in node_ids
            assert p_id in node_ids
            assert gp_id in node_ids

    def test_lineage_no_duplicates(self):
        """Even with shared ancestry, each node appears once."""
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            gp_id, p_id, c_id = _seed_models(store)

            client = _make_app(store, _test_config())
            resp = client.get(f"/models/{c_id}/lineage")
            nodes = resp.json()["nodes"]
            ids = [n["model_id"] for n in nodes]
            assert len(ids) == len(set(ids))

    def test_lineage_includes_hyperparams(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            gp_id, _, _ = _seed_models(store)

            client = _make_app(store, _test_config())
            resp = client.get(f"/models/{gp_id}/lineage")
            node = resp.json()["nodes"][0]
            assert "learning_rate" in node["hyperparameters"]


# ── Genetics Tests ───────────────────────────────────────────────────


class TestGenetics:
    def test_genetics_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            client = _make_app(store, _test_config())
            resp = client.get("/models/nonexistent/genetics")
            assert resp.status_code == 404

    def test_genetics_empty(self):
        """Model with no genetic events returns empty list."""
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            gp_id, _, _ = _seed_models(store)

            client = _make_app(store, _test_config())
            resp = client.get(f"/models/{gp_id}/genetics")
            assert resp.status_code == 200
            assert resp.json()["events"] == []

    def test_genetics_returns_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            gp_id, p_id, c_id = _seed_models(store)

            # Record genetic events for the child
            store.record_genetic_event(
                GeneticEventRecord(
                    event_id="evt-001",
                    generation=2,
                    event_type="crossover",
                    child_model_id=c_id,
                    parent_a_id=p_id,
                    parent_b_id=gp_id,
                    hyperparameter="learning_rate",
                    parent_a_value="0.0005",
                    parent_b_value="0.001",
                    inherited_from="A",
                    final_value="0.0005",
                    human_summary="Inherited LR from parent A",
                )
            )
            store.record_genetic_event(
                GeneticEventRecord(
                    event_id="evt-002",
                    generation=2,
                    event_type="mutation",
                    child_model_id=c_id,
                    parent_a_id=p_id,
                    hyperparameter="lstm_hidden_size",
                    parent_a_value="256",
                    inherited_from="A",
                    mutation_delta=256.0,
                    final_value="512",
                    human_summary="Mutated LSTM hidden size +256",
                )
            )

            client = _make_app(store, _test_config())
            resp = client.get(f"/models/{c_id}/genetics")
            assert resp.status_code == 200
            events = resp.json()["events"]
            assert len(events) == 2
            assert events[0]["event_type"] == "crossover"
            assert events[0]["hyperparameter"] == "learning_rate"
            assert events[1]["mutation_delta"] == 256.0
