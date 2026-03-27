"""Integration tests for the API against real registry DB and extracted data.

These tests create a temporary registry, seed it with real evaluation data
from the extracted Parquet files, and verify the API returns correct results.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from registry.model_store import (
    ModelStore,
    EvaluationDayRecord,
    EvaluationBetRecord,
    GeneticEventRecord,
)
from registry.scoreboard import Scoreboard
from api.routers import models, training, replay

# Real data paths
DATA_DIR = Path("data/processed")
REAL_DATE = "2026-03-26"
REAL_PARQUET = DATA_DIR / f"{REAL_DATE}.parquet"

pytestmark = pytest.mark.skipif(
    not REAL_PARQUET.exists(),
    reason=f"Real Parquet data not available at {REAL_PARQUET}",
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def real_markets() -> list[str]:
    """Get market IDs from real data."""
    df = pd.read_parquet(REAL_PARQUET)
    return sorted(df["market_id"].unique().tolist())[:3]


@pytest.fixture
def integration_env(real_markets):
    """Set up a temporary registry with seeded models and evaluation data."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ModelStore(
            db_path=str(Path(tmp) / "test.db"),
            weights_dir=str(Path(tmp) / "weights"),
            bet_logs_dir=str(Path(tmp) / "bet_logs"),
        )
        config = {
            "paths": {
                "processed_data": str(DATA_DIR),
                "registry_db": str(Path(tmp) / "test.db"),
                "model_weights": str(Path(tmp) / "weights"),
            },
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

        # Create a lineage: grandparent → parent → child
        gp_id = store.create_model(
            generation=0,
            architecture_name="ppo_lstm_v1",
            architecture_description="PPO with LSTM v1",
            hyperparameters={
                "learning_rate": 0.001,
                "lstm_hidden_size": 128,
                "mlp_hidden_size": 64,
            },
        )
        p_id = store.create_model(
            generation=1,
            architecture_name="ppo_lstm_v1",
            architecture_description="PPO with LSTM v1",
            hyperparameters={
                "learning_rate": 0.0005,
                "lstm_hidden_size": 256,
                "mlp_hidden_size": 128,
            },
            parent_a_id=gp_id,
        )
        c_id = store.create_model(
            generation=2,
            architecture_name="ppo_lstm_v1",
            architecture_description="PPO with LSTM v1",
            hyperparameters={
                "learning_rate": 0.0003,
                "lstm_hidden_size": 512,
                "mlp_hidden_size": 128,
            },
            parent_a_id=p_id,
            parent_b_id=gp_id,
        )

        # Seed evaluations for all 3 models
        market_id = real_markets[0]
        for mid, pnl_mult in [(gp_id, 1.0), (p_id, 0.5), (c_id, 1.5)]:
            run_id = store.create_evaluation_run(
                model_id=mid,
                train_cutoff_date="2026-03-25",
                test_days=[REAL_DATE],
            )
            store.record_evaluation_day(
                EvaluationDayRecord(
                    run_id=run_id,
                    date=REAL_DATE,
                    day_pnl=50.0 * pnl_mult,
                    bet_count=10,
                    winning_bets=6,
                    bet_precision=0.6,
                    pnl_per_bet=5.0 * pnl_mult,
                    early_picks=2,
                    profitable=True,
                )
            )
            # Write a bet to Parquet for replay tests
            bets = [
                EvaluationBetRecord(
                    run_id=run_id,
                    date=REAL_DATE,
                    market_id=market_id,
                    tick_timestamp="2026-03-26T14:00:05",
                    seconds_to_off=1795.0,
                    runner_id=101,
                    runner_name="Test Runner",
                    action="back",
                    price=3.5,
                    stake=10.0,
                    matched_size=10.0,
                    outcome="won",
                    pnl=25.0 * pnl_mult,
                ),
            ]
            store.write_bet_logs_parquet(run_id, REAL_DATE, bets)

        # Seed genetic events for the child
        store.record_genetic_event(
            GeneticEventRecord(
                event_id="int-evt-001",
                generation=2,
                event_type="crossover",
                child_model_id=c_id,
                parent_a_id=p_id,
                parent_b_id=gp_id,
                hyperparameter="learning_rate",
                parent_a_value="0.0005",
                parent_b_value="0.001",
                inherited_from="A",
                final_value="0.0003",
                human_summary="Inherited LR from parent A, mutated",
            )
        )

        # Build the full app
        app = FastAPI()
        app.include_router(models.router)
        app.include_router(training.router)
        app.include_router(replay.router)

        import asyncio

        scoreboard = Scoreboard(store=store, config=config)
        app.state.store = store
        app.state.scoreboard = scoreboard
        app.state.config = config
        app.state.progress_queue = asyncio.Queue()
        app.state.training_state = {"running": False, "latest_event": None}

        client = TestClient(app)
        yield {
            "client": client,
            "store": store,
            "gp_id": gp_id,
            "p_id": p_id,
            "c_id": c_id,
            "market_id": market_id,
        }


# ── Scoreboard Integration ──────────────────────────────────────────


class TestScoreboardIntegration:
    def test_scoreboard_returns_all_evaluated_models(self, integration_env):
        client = integration_env["client"]
        resp = client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        # All 3 models should be scored
        assert len(data["models"]) == 3

    def test_scoreboard_ranked_by_composite_score(self, integration_env):
        client = integration_env["client"]
        resp = client.get("/models")
        models_list = resp.json()["models"]
        scores = [m["composite_score"] for m in models_list]
        assert scores == sorted(scores, reverse=True)

    def test_scoreboard_entries_have_real_data(self, integration_env):
        client = integration_env["client"]
        resp = client.get("/models")
        for m in resp.json()["models"]:
            assert m["architecture_name"] == "ppo_lstm_v1"
            assert m["win_rate"] > 0
            assert m["test_days"] == 1


# ── Model Detail Integration ────────────────────────────────────────


class TestModelDetailIntegration:
    def test_model_detail_includes_real_hyperparams(self, integration_env):
        client = integration_env["client"]
        c_id = integration_env["c_id"]
        resp = client.get(f"/models/{c_id}")
        data = resp.json()
        assert data["hyperparameters"]["lstm_hidden_size"] == 512
        assert data["hyperparameters"]["learning_rate"] == 0.0003

    def test_model_detail_includes_per_day_metrics(self, integration_env):
        client = integration_env["client"]
        gp_id = integration_env["gp_id"]
        resp = client.get(f"/models/{gp_id}")
        metrics = resp.json()["metrics_history"]
        assert len(metrics) == 1
        assert metrics[0]["date"] == REAL_DATE
        assert metrics[0]["day_pnl"] == 50.0


# ── Lineage Integration ─────────────────────────────────────────────


class TestLineageIntegration:
    def test_lineage_traverses_correct_parent_chain(self, integration_env):
        client = integration_env["client"]
        c_id = integration_env["c_id"]
        p_id = integration_env["p_id"]
        gp_id = integration_env["gp_id"]

        resp = client.get(f"/models/{c_id}/lineage")
        nodes = resp.json()["nodes"]
        node_ids = {n["model_id"] for n in nodes}
        assert c_id in node_ids
        assert p_id in node_ids
        assert gp_id in node_ids

    def test_lineage_nodes_have_hyperparams(self, integration_env):
        client = integration_env["client"]
        c_id = integration_env["c_id"]
        resp = client.get(f"/models/{c_id}/lineage")
        for node in resp.json()["nodes"]:
            assert "learning_rate" in node["hyperparameters"]


# ── Genetics Integration ────────────────────────────────────────────


class TestGeneticsIntegration:
    def test_genetics_for_child_model(self, integration_env):
        client = integration_env["client"]
        c_id = integration_env["c_id"]
        resp = client.get(f"/models/{c_id}/genetics")
        events = resp.json()["events"]
        assert len(events) == 1
        assert events[0]["event_type"] == "crossover"
        assert events[0]["hyperparameter"] == "learning_rate"


# ── Replay Integration ──────────────────────────────────────────────


class TestReplayIntegration:
    def test_replay_day_returns_real_races(self, integration_env):
        client = integration_env["client"]
        gp_id = integration_env["gp_id"]
        resp = client.get(f"/replay/{gp_id}/{REAL_DATE}")
        assert resp.status_code == 200
        data = resp.json()
        # Real data has 53 markets
        assert len(data["races"]) > 0
        # Races have real venues
        venues = {r["venue"] for r in data["races"]}
        assert len(venues) > 0

    def test_replay_race_tick_sequence_matches_parquet(self, integration_env):
        """Verify tick count matches raw Parquet data."""
        client = integration_env["client"]
        gp_id = integration_env["gp_id"]
        market_id = integration_env["market_id"]

        # Count ticks in Parquet
        df = pd.read_parquet(REAL_PARQUET)
        expected_ticks = len(df[df["market_id"] == market_id])

        resp = client.get(f"/replay/{gp_id}/{REAL_DATE}/{market_id}")
        assert resp.status_code == 200
        actual_ticks = len(resp.json()["ticks"])
        assert actual_ticks == expected_ticks

    def test_replay_race_bet_events_at_correct_timestamps(self, integration_env):
        client = integration_env["client"]
        gp_id = integration_env["gp_id"]
        market_id = integration_env["market_id"]

        resp = client.get(f"/replay/{gp_id}/{REAL_DATE}/{market_id}")
        data = resp.json()
        # Should have our seeded bet
        assert len(data["all_bets"]) == 1
        bet = data["all_bets"][0]
        assert bet["tick_timestamp"] == "2026-03-26T14:00:05"
        assert bet["action"] == "back"
        assert bet["price"] == 3.5

    def test_replay_race_pnl_matches_registry(self, integration_env):
        client = integration_env["client"]
        gp_id = integration_env["gp_id"]
        market_id = integration_env["market_id"]

        resp = client.get(f"/replay/{gp_id}/{REAL_DATE}/{market_id}")
        data = resp.json()
        # Our seeded bet has pnl=25.0 (gp_id has pnl_mult=1.0)
        assert data["race_pnl"] == 25.0
