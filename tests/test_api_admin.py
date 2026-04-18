"""Unit tests for api/routers/admin.py — admin tools endpoints."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from registry.model_store import (
    EvaluationBetRecord,
    EvaluationDayRecord,
    GeneticEventRecord,
    ModelStore,
)
from registry.scoreboard import Scoreboard


# ── Helpers ──────────────────────────────────────────────────────────


def _test_config(tmp_dir: str) -> dict:
    processed = str(Path(tmp_dir) / "processed")
    backup = str(Path(tmp_dir) / "backup")
    Path(processed).mkdir(parents=True, exist_ok=True)
    Path(backup).mkdir(parents=True, exist_ok=True)
    return {
        "paths": {
            "processed_data": processed,
            "backup_data": backup,
            "model_weights": str(Path(tmp_dir) / "weights"),
            "registry_db": str(Path(tmp_dir) / "registry" / "models.db"),
            "logs": str(Path(tmp_dir) / "logs"),
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
        "database": {"host": "localhost", "port": 3306, "cold_data_db": "coldData", "hot_data_db": "hotDataRefactored"},
    }


def _create_store(tmp_dir: str) -> ModelStore:
    db_path = str(Path(tmp_dir) / "registry" / "models.db")
    weights_dir = str(Path(tmp_dir) / "weights")
    bet_logs_dir = str(Path(tmp_dir) / "registry" / "bet_logs")
    return ModelStore(db_path=db_path, weights_dir=weights_dir, bet_logs_dir=bet_logs_dir)


def _make_app(store: ModelStore, config: dict) -> TestClient:
    """Create a TestClient with admin router wired in."""
    from api.routers import admin

    app = FastAPI()
    app.include_router(admin.router)

    scoreboard = Scoreboard(store=store, config=config)
    app.state.config = config
    app.state.store = store
    app.state.scoreboard = scoreboard
    app.state.progress_queue = asyncio.Queue()
    app.state.training_state = {"running": False, "latest_event": None}

    return TestClient(app)


def _create_parquet(path: Path, market_ids: list[str] | None = None):
    """Create a minimal ticks Parquet file."""
    if market_ids is None:
        market_ids = ["1.234567890"]
    rows = []
    for mid in market_ids:
        for i in range(10):
            rows.append({"market_id": mid, "timestamp": f"2026-03-26T{10+i}:00:00", "sequence_number": i})
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _seed_model_with_eval(store: ModelStore, model_id: str = "test-model-1"):
    """Create a model with an evaluation run, days, bet logs, and genetic events."""
    import uuid as _uuid
    mid = store.create_model(
        generation=0,
        architecture_name="ppo_lstm_v1",
        architecture_description="Test",
        hyperparameters={"lr": 0.001},
        model_id=model_id,
    )

    # Save fake weights
    import torch
    store.save_weights(mid, {"layer": torch.zeros(2)})

    # Create evaluation run
    run_id = store.create_evaluation_run(mid, "2026-03-25", ["2026-03-26"], run_id=f"run-{model_id}")

    # Record eval day
    store.record_evaluation_day(EvaluationDayRecord(
        run_id=run_id, date="2026-03-26", day_pnl=10.0,
        bet_count=5, winning_bets=3, bet_precision=0.6,
        pnl_per_bet=2.0, early_picks=1, profitable=True,
    ))

    # Write bet log Parquet
    store.write_bet_logs_parquet(run_id, "2026-03-26", [
        EvaluationBetRecord(
            run_id=run_id, date="2026-03-26", market_id="1.234",
            tick_timestamp="2026-03-26T10:00:00", seconds_to_off=300,
            runner_id=12345, runner_name="Test Horse", action="back",
            price=3.0, stake=10.0, matched_size=10.0, outcome="won", pnl=20.0,
        ),
    ])

    # Record genetic event
    store.record_genetic_event(GeneticEventRecord(
        event_id=f"evt-{model_id}", generation=0, event_type="selection",
        child_model_id=mid, selection_reason="elite",
        human_summary="Test model survived as elite",
    ))

    return mid, run_id


# ── GET /admin/days tests ────────────────────────────────────────────


class TestListExtractedDays:
    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            client = _make_app(store, config)

            resp = client.get("/admin/days")
            assert resp.status_code == 200
            assert resp.json()["days"] == []

    def test_returns_day_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            processed = Path(config["paths"]["processed_data"])

            _create_parquet(processed / "2026-03-26.parquet", ["1.111", "1.222"])
            # Also create runners file — should be ignored
            _create_parquet(processed / "2026-03-26_runners.parquet")

            client = _make_app(store, config)
            resp = client.get("/admin/days")
            assert resp.status_code == 200

            days = resp.json()["days"]
            assert len(days) == 1
            assert days[0]["date"] == "2026-03-26"
            assert days[0]["tick_count"] == 20  # 2 markets × 10 ticks
            assert days[0]["race_count"] == 2
            assert days[0]["file_size_bytes"] > 0

    def test_multiple_days_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            processed = Path(config["paths"]["processed_data"])

            _create_parquet(processed / "2026-03-27.parquet")
            _create_parquet(processed / "2026-03-25.parquet")
            _create_parquet(processed / "2026-03-26.parquet")

            client = _make_app(store, config)
            resp = client.get("/admin/days")
            dates = [d["date"] for d in resp.json()["days"]]
            assert dates == ["2026-03-25", "2026-03-26", "2026-03-27"]


# ── GET /admin/backup-days tests ─────────────────────────────────────


class TestListBackupDays:
    def test_empty_backup_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            client = _make_app(store, config)

            resp = client.get("/admin/backup-days")
            assert resp.status_code == 200
            assert resp.json()["days"] == []

    def test_returns_only_new_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            processed = Path(config["paths"]["processed_data"])
            backup = Path(config["paths"]["backup_data"])

            # Already extracted
            _create_parquet(processed / "2026-03-25.parquet")
            # Available in backup
            _create_parquet(backup / "2026-03-25.parquet")  # already exists
            _create_parquet(backup / "2026-03-26.parquet")  # new
            _create_parquet(backup / "2026-03-27.parquet")  # new

            client = _make_app(store, config)
            resp = client.get("/admin/backup-days")
            dates = [d["date"] for d in resp.json()["days"]]
            assert "2026-03-25" not in dates
            assert "2026-03-26" in dates
            assert "2026-03-27" in dates

    def test_backup_dir_not_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            # Remove backup dir
            shutil.rmtree(config["paths"]["backup_data"])
            store = _create_store(tmp)
            client = _make_app(store, config)

            resp = client.get("/admin/backup-days")
            assert resp.status_code == 200
            assert resp.json()["days"] == []


# ── GET /admin/agents tests ──────────────────────────────────────────


class TestListAgents:
    def test_empty_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            client = _make_app(store, config)

            resp = client.get("/admin/agents")
            assert resp.status_code == 200
            assert resp.json()["agents"] == []

    def test_returns_all_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)

            m1 = store.create_model(0, "ppo_lstm_v1", "Test", {"lr": 0.001})
            m2 = store.create_model(1, "ppo_lstm_v1", "Test", {"lr": 0.002})
            store.update_model_status(m2, "discarded")

            client = _make_app(store, config)
            resp = client.get("/admin/agents")
            agents = resp.json()["agents"]
            assert len(agents) == 2

            ids = {a["model_id"] for a in agents}
            assert m1 in ids
            assert m2 in ids

            statuses = {a["model_id"]: a["status"] for a in agents}
            assert statuses[m1] == "active"
            assert statuses[m2] == "discarded"


# ── DELETE /admin/days/{date} tests ──────────────────────────────────


class TestDeleteDay:
    def test_delete_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            processed = Path(config["paths"]["processed_data"])
            _create_parquet(processed / "2026-03-26.parquet")

            client = _make_app(store, config)
            resp = client.delete("/admin/days/2026-03-26")
            assert resp.status_code == 400
            assert "confirm" in resp.json()["detail"].lower()
            # File should still exist
            assert (processed / "2026-03-26.parquet").exists()

    def test_delete_existing_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            processed = Path(config["paths"]["processed_data"])

            _create_parquet(processed / "2026-03-26.parquet")
            _create_parquet(processed / "2026-03-26_runners.parquet")

            # Add eval day record for this date
            mid = store.create_model(0, "ppo_lstm_v1", "Test", {})
            rid = store.create_evaluation_run(mid, "2026-03-25", ["2026-03-26"])
            store.record_evaluation_day(EvaluationDayRecord(
                run_id=rid, date="2026-03-26", day_pnl=5.0,
                bet_count=2, winning_bets=1, bet_precision=0.5,
                pnl_per_bet=2.5, early_picks=0, profitable=True,
            ))

            client = _make_app(store, config)
            resp = client.delete("/admin/days/2026-03-26?confirm=true")
            assert resp.status_code == 200
            assert resp.json()["deleted"] is True

            # Verify files deleted
            assert not (processed / "2026-03-26.parquet").exists()
            assert not (processed / "2026-03-26_runners.parquet").exists()

            # Verify eval days deleted
            days = store.get_evaluation_days(rid)
            assert len(days) == 0

    def test_delete_nonexistent_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            client = _make_app(store, config)

            resp = client.delete("/admin/days/2026-01-01?confirm=true")
            assert resp.status_code == 404

    def test_delete_invalid_date_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            client = _make_app(store, config)

            resp = client.delete("/admin/days/not-a-date?confirm=true")
            assert resp.status_code == 400


# ── DELETE /admin/agents/{model_id} tests ────────────────────────────


class TestDeleteAgent:
    def test_delete_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            mid = store.create_model(0, "ppo_lstm_v1", "Test", {})

            client = _make_app(store, config)
            resp = client.delete(f"/admin/agents/{mid}")
            assert resp.status_code == 400
            assert "confirm" in resp.json()["detail"].lower()
            # Model should still exist
            assert store.get_model(mid) is not None

    def test_delete_model_with_all_artefacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            mid, run_id = _seed_model_with_eval(store)

            client = _make_app(store, config)
            resp = client.delete(f"/admin/agents/{mid}?confirm=true")
            assert resp.status_code == 200
            assert resp.json()["deleted"] is True

            # Verify model gone
            assert store.get_model(mid) is None

            # Verify weights deleted
            assert not (store.weights_dir / f"{mid}.pt").exists()

            # Verify eval runs gone
            assert store.get_latest_evaluation_run(mid) is None

            # Verify eval days gone
            assert store.get_evaluation_days(run_id) == []

            # Verify bet logs gone
            assert not (store.bet_logs_dir / run_id).exists()

            # Verify genetic events gone
            events = store.get_genetic_events(child_model_id=mid)
            assert len(events) == 0

    def test_delete_nonexistent_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            client = _make_app(store, config)

            resp = client.delete("/admin/agents/nonexistent-id?confirm=true")
            assert resp.status_code == 404

    def test_delete_does_not_cascade_parent_refs(self):
        """Deleting a child does not remove the parent's record."""
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)

            parent_id = store.create_model(0, "ppo_lstm_v1", "Test", {})
            child_id = store.create_model(
                1, "ppo_lstm_v1", "Test", {},
                parent_a_id=parent_id,
            )

            client = _make_app(store, config)
            resp = client.delete(f"/admin/agents/{child_id}?confirm=true")
            assert resp.status_code == 200

            # Parent still exists
            assert store.get_model(parent_id) is not None


# ── POST /admin/import-day tests ─────────────────────────────────────


class TestImportDay:
    def test_import_day_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            client = _make_app(store, config)

            with patch("data.extractor.DataExtractor") as MockExtractor:
                instance = MockExtractor.return_value
                instance.extract_date.return_value = True

                resp = client.post("/admin/import-day", json={"date": "2026-03-26"})
                assert resp.status_code == 200
                data = resp.json()
                assert data["success"] is True
                assert data["date"] == "2026-03-26"

                instance.extract_date.assert_called_once()

    def test_import_day_no_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            client = _make_app(store, config)

            with patch("data.extractor.DataExtractor") as MockExtractor:
                instance = MockExtractor.return_value
                instance.extract_date.return_value = False

                resp = client.post("/admin/import-day", json={"date": "2026-03-26"})
                assert resp.status_code == 200
                assert resp.json()["success"] is False

    def test_import_day_invalid_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            client = _make_app(store, config)

            resp = client.post("/admin/import-day", json={"date": "not-a-date"})
            assert resp.status_code == 400

    def test_import_day_extractor_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            client = _make_app(store, config)

            with patch("data.extractor.DataExtractor") as MockExtractor:
                instance = MockExtractor.return_value
                instance.extract_date.side_effect = RuntimeError("DB down")

                resp = client.post("/admin/import-day", json={"date": "2026-03-26"})
                assert resp.status_code == 200
                assert resp.json()["success"] is False
                assert "DB down" in resp.json()["detail"]


# ── POST /admin/import-range tests ───────────────────────────────────


class TestImportRange:
    def test_import_range_queues_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            client = _make_app(store, config)

            resp = client.post("/admin/import-range", json={
                "start_date": "2026-03-25",
                "end_date": "2026-03-27",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["dates_queued"] == 3
            assert data["job_id"] != ""

    def test_import_range_skips_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            processed = Path(config["paths"]["processed_data"])

            _create_parquet(processed / "2026-03-25.parquet")
            _create_parquet(processed / "2026-03-26.parquet")

            client = _make_app(store, config)
            resp = client.post("/admin/import-range", json={
                "start_date": "2026-03-25",
                "end_date": "2026-03-27",
            })
            assert resp.status_code == 200
            assert resp.json()["dates_queued"] == 1  # only 2026-03-27

    def test_import_range_force_reimport(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            processed = Path(config["paths"]["processed_data"])

            _create_parquet(processed / "2026-03-25.parquet")

            client = _make_app(store, config)
            resp = client.post("/admin/import-range", json={
                "start_date": "2026-03-25",
                "end_date": "2026-03-25",
                "force": True,
            })
            assert resp.status_code == 200
            assert resp.json()["dates_queued"] == 1

    def test_import_range_all_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            processed = Path(config["paths"]["processed_data"])

            _create_parquet(processed / "2026-03-25.parquet")

            client = _make_app(store, config)
            resp = client.post("/admin/import-range", json={
                "start_date": "2026-03-25",
                "end_date": "2026-03-25",
            })
            assert resp.status_code == 200
            assert resp.json()["dates_queued"] == 0

    def test_import_range_invalid_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            client = _make_app(store, config)

            resp = client.post("/admin/import-range", json={
                "start_date": "bad",
                "end_date": "2026-03-27",
            })
            assert resp.status_code == 400

    def test_import_range_start_after_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            client = _make_app(store, config)

            resp = client.post("/admin/import-range", json={
                "start_date": "2026-03-27",
                "end_date": "2026-03-25",
            })
            assert resp.status_code == 400


# ── POST /admin/reset tests ─────────────────────────────────────────


class TestReset:
    def test_reset_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            client = _make_app(store, config)

            resp = client.post("/admin/reset", json={"confirm": "wrong"})
            assert resp.status_code == 400

    def test_reset_clears_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            processed = Path(config["paths"]["processed_data"])

            # Create model with all artefacts
            mid, run_id = _seed_model_with_eval(store)

            # Create extracted data (should be preserved)
            _create_parquet(processed / "2026-03-26.parquet")

            # Seed the per-episode training log that feeds the
            # Training Monitor's Learning-diagnostics cards. Reset
            # should truncate this so the charts start clean.
            training_log_dir = Path(config["paths"]["logs"]) / "training"
            training_log_dir.mkdir(parents=True, exist_ok=True)
            episodes_path = training_log_dir / "episodes.jsonl"
            episodes_path.write_text('{"episode": 1}\n{"episode": 2}\n', encoding="utf-8")
            # Operator-archived sibling that must NOT be touched.
            archived_path = training_log_dir / "episodes.pre-naked-asymmetry-20260418.jsonl"
            archived_path.write_text('{"archived": true}\n', encoding="utf-8")

            client = _make_app(store, config)
            resp = client.post("/admin/reset", json={"confirm": "DELETE_EVERYTHING"})
            assert resp.status_code == 200
            assert resp.json()["reset"] is True

            # Verify all models gone
            assert store.list_models() == []

            # Verify genetic events gone
            assert store.get_genetic_events() == []

            # Verify weights deleted
            assert list(store.weights_dir.glob("*.pt")) == []

            # Verify bet logs deleted
            bet_log_dirs = [d for d in store.bet_logs_dir.iterdir() if d.is_dir()] if store.bet_logs_dir.exists() else []
            assert len(bet_log_dirs) == 0

            # Verify extracted Parquet PRESERVED
            assert (processed / "2026-03-26.parquet").exists()

            # Verify episodes.jsonl truncated (file still exists, empty)
            assert episodes_path.exists()
            assert episodes_path.read_text(encoding="utf-8") == ""
            # Archived sibling untouched
            assert archived_path.read_text(encoding="utf-8") == '{"archived": true}\n'

    def test_reset_empty_registry(self):
        """Reset on an already-empty registry should succeed."""
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            client = _make_app(store, config)

            resp = client.post("/admin/reset", json={"confirm": "DELETE_EVERYTHING"})
            assert resp.status_code == 200
            assert resp.json()["reset"] is True

    def test_reset_preserves_garaged_models(self):
        """Garaged models should survive a reset when clear_garage is false."""
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)

            # Create two models, garage one
            mid_keep = store.create_model(0, "arch", "d", {"lr": 0.001})
            mid_delete = store.create_model(0, "arch", "d", {"lr": 0.002})
            import torch
            store.save_weights(mid_keep, {"w": torch.randn(2, 2)})
            store.save_weights(mid_delete, {"w": torch.randn(2, 2)})
            store.set_garaged(mid_keep, True)

            client = _make_app(store, config)
            resp = client.post("/admin/reset", json={"confirm": "DELETE_EVERYTHING"})
            assert resp.status_code == 200

            # Garaged model survives
            assert store.get_model(mid_keep) is not None
            assert store.get_model(mid_keep).garaged is True
            # Non-garaged model deleted
            assert store.get_model(mid_delete) is None
            # Garaged weights survive
            assert Path(store.get_model(mid_keep).weights_path).exists()

    def test_reset_clear_garage_deletes_everything(self):
        """With clear_garage=true, garaged models are also deleted."""
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)

            mid = store.create_model(0, "arch", "d", {"lr": 0.001})
            store.set_garaged(mid, True)

            client = _make_app(store, config)
            resp = client.post("/admin/reset", json={
                "confirm": "DELETE_EVERYTHING",
                "clear_garage": True,
            })
            assert resp.status_code == 200
            assert store.get_model(mid) is None

    def test_purge_discarded_endpoint(self):
        """POST /admin/purge-discarded removes discarded non-garaged models."""
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)

            mid_active = store.create_model(0, "arch", "d", {"lr": 0.001})
            mid_disc = store.create_model(0, "arch", "d", {"lr": 0.002})
            mid_garaged = store.create_model(0, "arch", "d", {"lr": 0.003})
            store.update_model_status(mid_disc, "discarded")
            store.update_model_status(mid_garaged, "discarded")
            store.set_garaged(mid_garaged, True)

            client = _make_app(store, config)
            resp = client.post("/admin/purge-discarded")
            assert resp.status_code == 200

            assert store.get_model(mid_active) is not None
            assert store.get_model(mid_disc) is None
            assert store.get_model(mid_garaged) is not None


# ── Integration tests ────────────────────────────────────────────────


class TestAdminIntegration:
    """Full create → delete → verify cycle."""

    def test_create_then_delete_day(self):
        """Create test data → delete day → verify cleanup."""
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            processed = Path(config["paths"]["processed_data"])

            # Create day + eval data referencing it
            _create_parquet(processed / "2026-03-26.parquet")
            _create_parquet(processed / "2026-03-26_runners.parquet")
            mid = store.create_model(0, "ppo_lstm_v1", "Test", {})
            rid = store.create_evaluation_run(mid, "2026-03-25", ["2026-03-26"])
            store.record_evaluation_day(EvaluationDayRecord(
                run_id=rid, date="2026-03-26", day_pnl=5.0,
                bet_count=2, winning_bets=1, bet_precision=0.5,
                pnl_per_bet=2.5, early_picks=0, profitable=True,
            ))
            # Also add a day that shouldn't be deleted
            store.record_evaluation_day(EvaluationDayRecord(
                run_id=rid, date="2026-03-27", day_pnl=3.0,
                bet_count=1, winning_bets=1, bet_precision=1.0,
                pnl_per_bet=3.0, early_picks=0, profitable=True,
            ))

            client = _make_app(store, config)
            resp = client.delete("/admin/days/2026-03-26?confirm=true")
            assert resp.status_code == 200

            # Files gone
            assert not (processed / "2026-03-26.parquet").exists()
            assert not (processed / "2026-03-26_runners.parquet").exists()

            # Only the 2026-03-26 eval day deleted, 2026-03-27 preserved
            days = store.get_evaluation_days(rid)
            assert len(days) == 1
            assert days[0].date == "2026-03-27"

    def test_create_then_delete_agent(self):
        """Create model with full artefacts → delete → verify all cleaned up."""
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            mid, run_id = _seed_model_with_eval(store)

            client = _make_app(store, config)
            resp = client.delete(f"/admin/agents/{mid}?confirm=true")
            assert resp.status_code == 200

            # Everything gone
            assert store.get_model(mid) is None
            assert store.get_latest_evaluation_run(mid) is None
            assert store.get_evaluation_days(run_id) == []
            assert not (store.bet_logs_dir / run_id).exists()
            assert store.get_genetic_events(child_model_id=mid) == []

    def test_create_then_reset(self):
        """Create models → reset → verify empty registry."""
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            processed = Path(config["paths"]["processed_data"])

            _seed_model_with_eval(store, "model-a")
            _seed_model_with_eval(store, "model-b")
            _create_parquet(processed / "2026-03-26.parquet")

            client = _make_app(store, config)
            resp = client.post("/admin/reset", json={"confirm": "DELETE_EVERYTHING"})
            assert resp.status_code == 200

            assert store.list_models() == []
            assert store.get_genetic_events() == []
            # Parquet preserved
            assert (processed / "2026-03-26.parquet").exists()

    def test_list_days_after_delete(self):
        """Days listing updates after deletion."""
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            processed = Path(config["paths"]["processed_data"])

            _create_parquet(processed / "2026-03-25.parquet")
            _create_parquet(processed / "2026-03-26.parquet")

            client = _make_app(store, config)

            resp = client.get("/admin/days")
            assert len(resp.json()["days"]) == 2

            client.delete("/admin/days/2026-03-25?confirm=true")

            resp = client.get("/admin/days")
            days = resp.json()["days"]
            assert len(days) == 1
            assert days[0]["date"] == "2026-03-26"


# ── Betting Constraints Endpoints ──────────────────────────────────────


class TestBettingConstraints:
    def test_get_defaults(self):
        """GET /admin/config/constraints returns defaults when none configured."""
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            client = _make_app(store, config)

            resp = client.get("/admin/config/constraints")
            assert resp.status_code == 200
            data = resp.json()
            assert data["max_back_price"] is None
            assert data["max_lay_price"] is None
            assert data["min_seconds_before_off"] == 0

    def test_get_with_existing_values(self):
        """GET returns previously configured constraint values."""
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            config["training"]["betting_constraints"] = {
                "max_back_price": 50.0,
                "max_lay_price": 30.0,
                "min_seconds_before_off": 300,
            }
            store = _create_store(tmp)
            client = _make_app(store, config)

            resp = client.get("/admin/config/constraints")
            assert resp.status_code == 200
            data = resp.json()
            assert data["max_back_price"] == 50.0
            assert data["max_lay_price"] == 30.0
            assert data["min_seconds_before_off"] == 300

    def test_post_updates_config(self):
        """POST /admin/config/constraints updates in-memory config."""
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            client = _make_app(store, config)
            client.app.state.config_path = str(Path(tmp) / "config.yaml")

            resp = client.post("/admin/config/constraints", json={
                "max_back_price": 100.0,
                "max_lay_price": None,
                "min_seconds_before_off": 600,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["max_back_price"] == 100.0
            assert data["max_lay_price"] is None
            assert data["min_seconds_before_off"] == 600

            # Verify the in-memory config was updated
            assert config["training"]["betting_constraints"]["max_back_price"] == 100.0
            assert config["training"]["betting_constraints"]["min_seconds_before_off"] == 600

    def test_post_persists_to_yaml(self):
        """POST writes constraints to config.yaml on disk."""
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)

            # Write initial config.yaml so the POST can persist to it
            config_path = Path(tmp) / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            client = _make_app(store, config)
            # Set config_path so the endpoint writes to our temp file
            client.app.state.config_path = str(config_path)

            client.post("/admin/config/constraints", json={
                "max_back_price": 75.0,
                "max_lay_price": 25.0,
                "min_seconds_before_off": 120,
            })

            # Re-read from disk
            with open(config_path) as f:
                on_disk = yaml.safe_load(f)
            assert on_disk["training"]["betting_constraints"]["max_back_price"] == 75.0
            assert on_disk["training"]["betting_constraints"]["max_lay_price"] == 25.0
            assert on_disk["training"]["betting_constraints"]["min_seconds_before_off"] == 120

    def test_get_reflects_post(self):
        """GET after POST returns the updated values."""
        with tempfile.TemporaryDirectory() as tmp:
            config = _test_config(tmp)
            store = _create_store(tmp)
            client = _make_app(store, config)
            # Point config_path to temp so we don't corrupt the real config.yaml
            client.app.state.config_path = str(Path(tmp) / "config.yaml")

            client.post("/admin/config/constraints", json={
                "max_back_price": 42.0,
                "max_lay_price": 10.0,
                "min_seconds_before_off": 900,
            })

            resp = client.get("/admin/config/constraints")
            data = resp.json()
            assert data["max_back_price"] == 42.0
            assert data["max_lay_price"] == 10.0
            assert data["min_seconds_before_off"] == 900
