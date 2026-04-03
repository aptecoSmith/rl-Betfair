"""Unit tests for registry/model_store.py -- SQLite model registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from registry.model_store import (
    EvaluationBetRecord,
    EvaluationDayRecord,
    EvaluationRunRecord,
    ModelRecord,
    ModelStore,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path) -> ModelStore:
    """Create a fresh ModelStore backed by a temp directory."""
    return ModelStore(
        db_path=tmp_path / "test.db",
        weights_dir=tmp_path / "weights",
    )


def _sample_hyperparams() -> dict:
    return {
        "learning_rate": 3e-4,
        "lstm_hidden_size": 256,
        "mlp_hidden_size": 128,
        "mlp_layers": 2,
    }


# ── Table creation ────────────────────────────────────────────────────────────


class TestInit:
    """Test database initialisation."""

    def test_db_created(self, store: ModelStore):
        assert store.db_path.exists()

    def test_weights_dir_created(self, store: ModelStore):
        assert store.weights_dir.exists()

    def test_idempotent_init(self, tmp_path: Path):
        """Calling init twice should not fail."""
        s1 = ModelStore(tmp_path / "test.db", tmp_path / "w")
        s2 = ModelStore(tmp_path / "test.db", tmp_path / "w")
        # both work
        s1.create_model(0, "arch", "desc", {})
        models = s2.list_models()
        assert len(models) == 1


# ── Model CRUD ────────────────────────────────────────────────────────────────


class TestModelCRUD:
    """Test create, read, list, update for models."""

    def test_create_model(self, store: ModelStore):
        mid = store.create_model(
            generation=1,
            architecture_name="ppo_lstm_v1",
            architecture_description="Test arch",
            hyperparameters=_sample_hyperparams(),
        )
        assert isinstance(mid, str)
        assert len(mid) > 0

    def test_create_with_explicit_id(self, store: ModelStore):
        mid = store.create_model(
            generation=1,
            architecture_name="ppo_lstm_v1",
            architecture_description="desc",
            hyperparameters={},
            model_id="my-custom-id",
        )
        assert mid == "my-custom-id"

    def test_get_model(self, store: ModelStore):
        mid = store.create_model(1, "ppo_lstm_v1", "desc", _sample_hyperparams())
        model = store.get_model(mid)

        assert model is not None
        assert model.model_id == mid
        assert model.generation == 1
        assert model.architecture_name == "ppo_lstm_v1"
        assert model.status == "active"
        assert model.hyperparameters == _sample_hyperparams()

    def test_get_nonexistent(self, store: ModelStore):
        assert store.get_model("nonexistent") is None

    def test_list_models(self, store: ModelStore):
        store.create_model(1, "arch", "d", {})
        store.create_model(2, "arch", "d", {})
        models = store.list_models()
        assert len(models) == 2

    def test_list_models_by_status(self, store: ModelStore):
        mid1 = store.create_model(1, "arch", "d", {})
        mid2 = store.create_model(2, "arch", "d", {})
        store.update_model_status(mid2, "discarded")

        active = store.list_models(status="active")
        assert len(active) == 1
        assert active[0].model_id == mid1

        discarded = store.list_models(status="discarded")
        assert len(discarded) == 1
        assert discarded[0].model_id == mid2

    def test_update_status(self, store: ModelStore):
        mid = store.create_model(1, "arch", "d", {})
        store.update_model_status(mid, "discarded")
        model = store.get_model(mid)
        assert model is not None
        assert model.status == "discarded"

    def test_update_composite_score(self, store: ModelStore):
        mid = store.create_model(1, "arch", "d", {})
        store.update_composite_score(mid, 0.75)
        model = store.get_model(mid)
        assert model is not None
        assert model.composite_score == pytest.approx(0.75)
        assert model.last_evaluated_at is not None

    def test_parents_stored(self, store: ModelStore):
        parent_a = store.create_model(1, "arch", "d", {})
        parent_b = store.create_model(1, "arch", "d", {})
        child = store.create_model(
            2, "arch", "d", {},
            parent_a_id=parent_a, parent_b_id=parent_b,
        )
        model = store.get_model(child)
        assert model is not None
        assert model.parent_a_id == parent_a
        assert model.parent_b_id == parent_b

    def test_hyperparameters_json_roundtrip(self, store: ModelStore):
        hp = {"lr": 1e-4, "layers": [64, 128], "flag": True}
        mid = store.create_model(1, "arch", "d", hp)
        model = store.get_model(mid)
        assert model is not None
        assert model.hyperparameters == hp


# ── Weights I/O ───────────────────────────────────────────────────────────────


class TestWeights:
    """Test save/load of PyTorch state dicts."""

    def test_save_and_load(self, store: ModelStore):
        mid = store.create_model(1, "arch", "d", {})
        sd = {"layer.weight": torch.randn(10, 5), "layer.bias": torch.randn(10)}

        path = store.save_weights(mid, sd)
        assert Path(path).exists()

        loaded = store.load_weights(mid)
        assert set(loaded.keys()) == set(sd.keys())
        for key in sd:
            assert torch.allclose(loaded[key], sd[key])

    def test_save_updates_weights_path(self, store: ModelStore):
        mid = store.create_model(1, "arch", "d", {})
        store.save_weights(mid, {"x": torch.tensor(1.0)})
        model = store.get_model(mid)
        assert model is not None
        assert model.weights_path is not None
        assert mid in model.weights_path

    def test_load_nonexistent_model(self, store: ModelStore):
        with pytest.raises(ValueError, match="not found"):
            store.load_weights("nonexistent")

    def test_load_no_weights(self, store: ModelStore):
        mid = store.create_model(1, "arch", "d", {})
        with pytest.raises(ValueError, match="no saved weights"):
            store.load_weights(mid)


# ── Evaluation runs ───────────────────────────────────────────────────────────


class TestEvaluationRuns:
    """Test evaluation run creation and querying."""

    def test_create_run(self, store: ModelStore):
        mid = store.create_model(1, "arch", "d", {})
        rid = store.create_evaluation_run(
            model_id=mid,
            train_cutoff_date="2026-03-20",
            test_days=["2026-03-21", "2026-03-22"],
        )
        assert isinstance(rid, str)

    def test_get_latest_run(self, store: ModelStore):
        mid = store.create_model(1, "arch", "d", {})
        rid1 = store.create_evaluation_run(mid, "2026-03-20", ["2026-03-21"])
        rid2 = store.create_evaluation_run(mid, "2026-03-22", ["2026-03-23"])

        latest = store.get_latest_evaluation_run(mid)
        assert latest is not None
        assert latest.run_id == rid2

    def test_no_runs(self, store: ModelStore):
        mid = store.create_model(1, "arch", "d", {})
        assert store.get_latest_evaluation_run(mid) is None


# ── Evaluation days ───────────────────────────────────────────────────────────


class TestEvaluationDays:
    """Test per-day evaluation metrics."""

    def test_record_and_retrieve(self, store: ModelStore):
        mid = store.create_model(1, "arch", "d", {})
        rid = store.create_evaluation_run(mid, "2026-03-20", ["2026-03-21"])

        day = EvaluationDayRecord(
            run_id=rid,
            date="2026-03-21",
            day_pnl=5.50,
            bet_count=10,
            winning_bets=4,
            bet_precision=0.4,
            pnl_per_bet=0.55,
            early_picks=1,
            profitable=True,
        )
        store.record_evaluation_day(day)

        days = store.get_evaluation_days(rid)
        assert len(days) == 1
        assert days[0].day_pnl == pytest.approx(5.50)
        assert days[0].bet_count == 10
        assert days[0].profitable is True

    def test_multiple_days(self, store: ModelStore):
        mid = store.create_model(1, "arch", "d", {})
        rid = store.create_evaluation_run(mid, "2026-03-18", ["2026-03-19", "2026-03-20"])

        for i, date in enumerate(["2026-03-19", "2026-03-20"]):
            store.record_evaluation_day(EvaluationDayRecord(
                run_id=rid, date=date, day_pnl=float(i), bet_count=5,
                winning_bets=2, bet_precision=0.4, pnl_per_bet=0.0,
                early_picks=0, profitable=i > 0,
            ))

        days = store.get_evaluation_days(rid)
        assert len(days) == 2
        assert days[0].date == "2026-03-19"


# ── Evaluation bets (Parquet) ─────────────────────────────────────────────────


class TestEvaluationBetsParquet:
    """Test bet records via Parquet read/write."""

    def test_write_and_read(self, store: ModelStore):
        mid = store.create_model(1, "arch", "d", {})
        rid = store.create_evaluation_run(mid, "2026-03-20", ["2026-03-21"])

        bet = EvaluationBetRecord(
            run_id=rid,
            date="2026-03-21",
            market_id="1.200000001",
            tick_timestamp="2026-03-21T14:00:00",
            seconds_to_off=300.0,
            runner_id=12345,
            runner_name="Horse1",
            action="back",
            price=4.0,
            stake=10.0,
            matched_size=10.0,
            outcome="won",
            pnl=30.0,
        )
        path = store.write_bet_logs_parquet(rid, "2026-03-21", [bet])
        assert path is not None
        assert path.exists()

        bets = store.get_evaluation_bets(rid)
        assert len(bets) == 1
        assert bets[0].action == "back"
        assert bets[0].pnl == pytest.approx(30.0)
        assert bets[0].runner_name == "Horse1"

    def test_multiple_bets(self, store: ModelStore):
        mid = store.create_model(1, "arch", "d", {})
        rid = store.create_evaluation_run(mid, "2026-03-20", ["2026-03-21"])

        bets = [
            EvaluationBetRecord(
                run_id=rid, date="2026-03-21", market_id="1.200000001",
                tick_timestamp=f"2026-03-21T14:0{i}:00",
                seconds_to_off=300.0 - i * 5,
                runner_id=i + 1, runner_name=f"Horse{i+1}",
                action="back" if i % 2 == 0 else "lay",
                price=3.0 + i, stake=5.0, matched_size=5.0,
                outcome="won" if i < 2 else "lost",
                pnl=10.0 if i < 2 else -5.0,
            )
            for i in range(5)
        ]
        store.write_bet_logs_parquet(rid, "2026-03-21", bets)

        stored = store.get_evaluation_bets(rid)
        assert len(stored) == 5
        backs = [b for b in stored if b.action == "back"]
        lays = [b for b in stored if b.action == "lay"]
        assert len(backs) == 3
        assert len(lays) == 2

    def test_multiple_days(self, store: ModelStore):
        mid = store.create_model(1, "arch", "d", {})
        rid = store.create_evaluation_run(mid, "2026-03-20", ["2026-03-21", "2026-03-22"])

        for date in ["2026-03-21", "2026-03-22"]:
            bets = [
                EvaluationBetRecord(
                    run_id=rid, date=date, market_id="1.200000001",
                    tick_timestamp=f"{date}T14:00:00",
                    seconds_to_off=300.0, runner_id=1, runner_name="Horse1",
                    action="back", price=4.0, stake=5.0, matched_size=5.0,
                    outcome="won", pnl=15.0,
                )
                for _ in range(3)
            ]
            store.write_bet_logs_parquet(rid, date, bets)

        stored = store.get_evaluation_bets(rid)
        assert len(stored) == 6
        # Should be sorted by date
        dates = [b.date for b in stored]
        assert dates == sorted(dates)

    def test_empty_records_returns_none(self, store: ModelStore):
        mid = store.create_model(1, "arch", "d", {})
        rid = store.create_evaluation_run(mid, "2026-03-20", ["2026-03-21"])
        result = store.write_bet_logs_parquet(rid, "2026-03-21", [])
        assert result is None

    def test_no_parquet_returns_empty(self, store: ModelStore):
        bets = store.get_evaluation_bets("nonexistent-run-id")
        assert bets == []

    def test_bet_logs_dir_created(self, store: ModelStore):
        assert store.bet_logs_dir.exists()

    def test_parquet_schema_correct(self, store: ModelStore):
        import pandas as pd

        mid = store.create_model(1, "arch", "d", {})
        rid = store.create_evaluation_run(mid, "2026-03-20", ["2026-03-21"])

        bet = EvaluationBetRecord(
            run_id=rid, date="2026-03-21", market_id="1.200000001",
            tick_timestamp="2026-03-21T14:00:00", seconds_to_off=300.0,
            runner_id=12345, runner_name="Horse1", action="back",
            price=4.0, stake=10.0, matched_size=10.0, outcome="won", pnl=30.0,
        )
        path = store.write_bet_logs_parquet(rid, "2026-03-21", [bet])
        df = pd.read_parquet(path)
        expected_cols = {
            "run_id", "date", "market_id", "tick_timestamp",
            "seconds_to_off", "runner_id", "runner_name", "action",
            "price", "stake", "matched_size", "outcome", "pnl",
            "opportunity_window_s",
        }
        assert set(df.columns) == expected_cols


# ── Garage ───────────────────────────────────────────────────────────────────


class TestGarage:
    """Tests for garage (star/persist) functionality."""

    def test_new_model_not_garaged(self, store: ModelStore):
        mid = store.create_model(0, "arch", "", _sample_hyperparams())
        rec = store.get_model(mid)
        assert rec.garaged is False

    def test_set_garaged_true(self, store: ModelStore):
        mid = store.create_model(0, "arch", "", _sample_hyperparams())
        store.set_garaged(mid, True)
        rec = store.get_model(mid)
        assert rec.garaged is True

    def test_set_garaged_false(self, store: ModelStore):
        mid = store.create_model(0, "arch", "", _sample_hyperparams())
        store.set_garaged(mid, True)
        store.set_garaged(mid, False)
        rec = store.get_model(mid)
        assert rec.garaged is False

    def test_list_garaged_models_empty(self, store: ModelStore):
        store.create_model(0, "arch", "", _sample_hyperparams())
        assert store.list_garaged_models() == []

    def test_list_garaged_models(self, store: ModelStore):
        m1 = store.create_model(0, "arch", "", _sample_hyperparams())
        m2 = store.create_model(0, "arch", "", _sample_hyperparams())
        store.set_garaged(m1, True)
        garaged = store.list_garaged_models()
        assert len(garaged) == 1
        assert garaged[0].model_id == m1

    def test_list_garaged_includes_discarded(self, store: ModelStore):
        mid = store.create_model(0, "arch", "", _sample_hyperparams())
        store.set_garaged(mid, True)
        store.update_model_status(mid, "discarded")
        garaged = store.list_garaged_models()
        assert len(garaged) == 1
        assert garaged[0].status == "discarded"


# ── Delete model ─────────────────────────────────────────────────────────────


class TestDeleteModel:
    """Tests for full model deletion."""

    def test_delete_model_removes_record(self, store: ModelStore):
        mid = store.create_model(0, "arch", "", _sample_hyperparams())
        assert store.delete_model(mid) is True
        assert store.get_model(mid) is None

    def test_delete_model_removes_weights(self, store: ModelStore):
        mid = store.create_model(0, "arch", "", _sample_hyperparams())
        sd = {"layer.weight": torch.randn(4, 4)}
        path = store.save_weights(mid, sd)
        assert Path(path).exists()
        store.delete_model(mid)
        assert not Path(path).exists()

    def test_delete_model_removes_eval_data(self, store: ModelStore):
        mid = store.create_model(0, "arch", "", _sample_hyperparams())
        rid = store.create_evaluation_run(mid, "2026-03-20", ["2026-03-21"])
        day = EvaluationDayRecord(
            run_id=rid, date="2026-03-21", day_pnl=5.0,
            bet_count=10, winning_bets=6, bet_precision=0.6,
            pnl_per_bet=0.5, early_picks=2, profitable=True,
        )
        store.record_evaluation_day(day)
        store.delete_model(mid)
        assert store.get_model(mid) is None
        assert store.get_latest_evaluation_run(mid) is None

    def test_delete_nonexistent_model(self, store: ModelStore):
        assert store.delete_model("nonexistent-id") is False


# ── Purge discarded ──────────────────────────────────────────────────────────


class TestPurgeDiscarded:
    """Tests for bulk purge of discarded non-garaged models."""

    def test_purge_removes_discarded(self, store: ModelStore):
        mid = store.create_model(0, "arch", "", _sample_hyperparams())
        store.update_model_status(mid, "discarded")
        purged = store.purge_discarded()
        assert mid in purged
        assert store.get_model(mid) is None

    def test_purge_preserves_active(self, store: ModelStore):
        mid = store.create_model(0, "arch", "", _sample_hyperparams())
        purged = store.purge_discarded()
        assert purged == []
        assert store.get_model(mid) is not None

    def test_purge_preserves_garaged_discarded(self, store: ModelStore):
        mid = store.create_model(0, "arch", "", _sample_hyperparams())
        store.update_model_status(mid, "discarded")
        store.set_garaged(mid, True)
        purged = store.purge_discarded()
        assert purged == []
        assert store.get_model(mid) is not None

    def test_purge_mixed(self, store: ModelStore):
        m_active = store.create_model(0, "arch", "", _sample_hyperparams())
        m_discarded = store.create_model(0, "arch", "", _sample_hyperparams())
        m_garaged_disc = store.create_model(0, "arch", "", _sample_hyperparams())
        store.update_model_status(m_discarded, "discarded")
        store.update_model_status(m_garaged_disc, "discarded")
        store.set_garaged(m_garaged_disc, True)

        purged = store.purge_discarded()
        assert m_discarded in purged
        assert m_active not in purged
        assert m_garaged_disc not in purged
        assert store.get_model(m_active) is not None
        assert store.get_model(m_garaged_disc) is not None
