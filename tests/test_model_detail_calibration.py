"""Tests for the model-detail calibration card.

Scalping-active-management §05. Covers:

1. Perfect predictions → MACE = 0.0.
2. Sparse-bucket exclusion — one bucket with 19 records, three with
   100+ → MACE averages the three dense buckets only.
3. Fewer than two buckets clear the threshold → MACE = None,
   insufficient_data = True.
4. Scatter includes completed pairs exactly once with the correct
   predicted/realised P&L values.
5. Stddev bucketing is self-scaling — a dataset with a single stddev
   value collapses every point to "med".
6. Directional run (no scalping bets) → ``calibration`` is None on
   the ModelDetail response.
7. API contract — GET /models/{id} returns the ``calibration`` field
   under the additive schema change.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.calibration import compute_calibration_stats
from registry.model_store import (
    EvaluationBetRecord,
    EvaluationDayRecord,
    ModelStore,
)
from registry.scoreboard import Scoreboard


# -- Helpers --------------------------------------------------------


def _make_app(store: ModelStore) -> TestClient:
    from fastapi import FastAPI
    from api.routers import models

    app = FastAPI()
    app.include_router(models.router)
    config = {
        "reward": {"coefficients": {
            "win_rate": 0.35, "sharpe": 0.30,
            "mean_daily_pnl": 0.15, "efficiency": 0.20,
        }},
        "training": {"starting_budget": 100.0},
    }
    app.state.store = store
    app.state.scoreboard = Scoreboard(store=store, config=config)
    return TestClient(app)


def _create_store(tmp_dir: str) -> ModelStore:
    return ModelStore(
        db_path=str(Path(tmp_dir) / "test.db"),
        weights_dir=str(Path(tmp_dir) / "weights"),
        bet_logs_dir=str(Path(tmp_dir) / "bet_logs"),
    )


def _make_pair_records(
    pair_id: str,
    run_id: str,
    date: str,
    fill_prob: float,
    completed: bool,
    *,
    predicted_pnl: float | None = 1.0,
    predicted_stddev: float | None = 2.0,
    back_price: float = 5.0,
    lay_price: float = 4.0,
    back_stake: float = 10.0,
    lay_stake: float = 10.0,
    aggressive_tick: str = "2026-03-26T10:00:00",
    passive_tick: str = "2026-03-26T10:00:05",
) -> list[EvaluationBetRecord]:
    """Build 1 (naked) or 2 (completed) EvaluationBetRecords forming
    a scalping pair with the given prediction values."""

    def _rec(
        side: str, price: float, stake: float, ts: str,
    ) -> EvaluationBetRecord:
        return EvaluationBetRecord(
            run_id=run_id,
            date=date,
            market_id="1.100",
            tick_timestamp=ts,
            seconds_to_off=300.0,
            runner_id=101,
            runner_name="Runner",
            action=side,
            price=price,
            stake=stake,
            matched_size=stake,
            outcome="won",
            pnl=0.0,
            pair_id=pair_id,
            fill_prob_at_placement=fill_prob,
            predicted_locked_pnl_at_placement=predicted_pnl,
            predicted_locked_stddev_at_placement=predicted_stddev,
        )

    # Aggressive = back @ back_price (first). Passive = lay @ lay_price.
    recs = [_rec("back", back_price, back_stake, aggressive_tick)]
    if completed:
        recs.append(_rec("lay", lay_price, lay_stake, passive_tick))
    return recs


def _perfect_bucket_records(
    run_id: str, count: int, fill_prob: float,
    completed_count: int, bucket_tag: str,
) -> list[EvaluationBetRecord]:
    """Emit ``count`` pairs whose fill-prob lands in the given bucket.

    ``completed_count`` of them are completed (two legs); the rest are
    naked (one leg). ``fill_prob`` must lie in the target bucket.
    """
    out: list[EvaluationBetRecord] = []
    for i in range(count):
        pair_id = f"{bucket_tag}-{i}"
        out.extend(_make_pair_records(
            pair_id=pair_id,
            run_id=run_id,
            date="2026-03-26",
            fill_prob=fill_prob,
            completed=(i < completed_count),
            aggressive_tick=f"2026-03-26T10:{i:02d}:00",
            passive_tick=f"2026-03-26T10:{i:02d}:05",
        ))
    return out


# -- Direct calibration-function tests -------------------------------


class TestCalibrationStats:
    def test_perfect_predictions_yield_zero_mace(self):
        """Each bucket has predictions dead-centre and an observed
        rate matching the midpoint exactly → every bucket's abs
        error is 0 → MACE = 0.0."""
        records: list[EvaluationBetRecord] = []
        # 80 pairs per bucket chosen so every midpoint × 80 is an
        # exact integer (0.125→10, 0.375→30, 0.625→50, 0.875→70).
        for fill_prob, midpoint, tag in (
            (0.125, 0.125, "b0"),
            (0.375, 0.375, "b1"),
            (0.625, 0.625, "b2"),
            (0.875, 0.875, "b3"),
        ):
            completed = int(midpoint * 80)
            records.extend(_perfect_bucket_records(
                run_id="R1", count=80, fill_prob=fill_prob,
                completed_count=completed, bucket_tag=tag,
            ))
        stats = compute_calibration_stats(records)
        assert stats is not None
        assert stats.insufficient_data is False
        assert stats.mace == pytest.approx(0.0, abs=1e-9)
        for b in stats.reliability_buckets:
            assert b.abs_calibration_error == pytest.approx(0.0, abs=1e-9)

    def test_sparse_bucket_excluded_from_mace(self):
        """Three dense buckets + one bucket with 19 records → MACE
        averages the three dense buckets only. The sparse bucket
        still appears in reliability_buckets."""
        records: list[EvaluationBetRecord] = []
        # Dense bucket 0: 100 pairs, all completed (observed=1.0,
        # midpoint=0.125, err=0.875).
        records.extend(_perfect_bucket_records(
            run_id="R1", count=100, fill_prob=0.10,
            completed_count=100, bucket_tag="b0",
        ))
        # Dense bucket 1: 100 pairs, observed=midpoint=0.375.
        records.extend(_perfect_bucket_records(
            run_id="R1", count=100, fill_prob=0.40,
            completed_count=38, bucket_tag="b1",
        ))
        # Dense bucket 2: 100 pairs, observed=midpoint=0.625.
        records.extend(_perfect_bucket_records(
            run_id="R1", count=100, fill_prob=0.60,
            completed_count=62, bucket_tag="b2",
        ))
        # Sparse bucket 3: 19 pairs. Whatever error it has is excluded.
        records.extend(_perfect_bucket_records(
            run_id="R1", count=19, fill_prob=0.90,
            completed_count=0, bucket_tag="b3",
        ))
        stats = compute_calibration_stats(records)
        assert stats is not None
        assert stats.insufficient_data is False
        # MACE = mean of the three dense buckets' errors only.
        dense_errors = [
            stats.reliability_buckets[0].abs_calibration_error,
            stats.reliability_buckets[1].abs_calibration_error,
            stats.reliability_buckets[2].abs_calibration_error,
        ]
        expected = sum(dense_errors) / 3
        assert stats.mace == pytest.approx(expected, abs=1e-9)
        # Sparse bucket is still reported.
        assert stats.reliability_buckets[3].count == 19

    def test_insufficient_data_when_fewer_than_two_buckets_clear(self):
        """Only one bucket with count >= 20 → MACE=None,
        insufficient_data=True."""
        # One dense bucket (50 pairs), three sparse (5 each).
        records: list[EvaluationBetRecord] = []
        records.extend(_perfect_bucket_records(
            run_id="R1", count=50, fill_prob=0.10,
            completed_count=5, bucket_tag="b0",
        ))
        records.extend(_perfect_bucket_records(
            run_id="R1", count=5, fill_prob=0.40,
            completed_count=0, bucket_tag="b1",
        ))
        records.extend(_perfect_bucket_records(
            run_id="R1", count=5, fill_prob=0.60,
            completed_count=0, bucket_tag="b2",
        ))
        records.extend(_perfect_bucket_records(
            run_id="R1", count=5, fill_prob=0.90,
            completed_count=0, bucket_tag="b3",
        ))
        stats = compute_calibration_stats(records)
        assert stats is not None
        assert stats.insufficient_data is True
        assert stats.mace is None

    def test_scatter_contains_completed_pair_exactly_once(self):
        """A completed pair with predicted_pnl=1.0 appears once; the
        realised_pnl reflects the locked-floor math."""
        # Build a completed pair with a clear locked-floor value.
        # back @ 5.0 stake 10, lay @ 4.0 stake 10, commission 5%.
        # win_pnl = 10*(5-1)*(1-0.05) - 10*(4-1) = 38 - 30 = 8.
        # lose_pnl = -10 + 10*(1-0.05) = -10 + 9.5 = -0.5.
        # locked = max(0, min(8, -0.5)) = 0.0.
        records = _make_pair_records(
            pair_id="P1", run_id="R1", date="2026-03-26",
            fill_prob=0.6, completed=True,
            predicted_pnl=1.0, predicted_stddev=2.0,
            back_price=5.0, lay_price=4.0,
            back_stake=10.0, lay_stake=10.0,
        )
        stats = compute_calibration_stats(records)
        assert stats is not None
        # Single completed pair → scatter has one point.
        assert len(stats.scatter) == 1
        point = stats.scatter[0]
        assert point.predicted_pnl == pytest.approx(1.0)
        assert point.realised_pnl == pytest.approx(0.0)

    def test_scatter_point_with_positive_realised_pnl(self):
        """Properly-sized asymmetric hedge locks a positive amount —
        a record with predicted=1.0 and realised=1.2 appears exactly
        once in the scatter list."""
        # Choose a hedge where the locked floor is exactly 1.2.
        # back @ 5.0 stake 10; require lay stake L at price P so that
        # win_pnl = lose_pnl = 1.2.
        # win_pnl = 10*4*0.95 - L*(P-1) = 38 - L*(P-1) = 1.2
        # lose_pnl = -10 + L*0.95 = 1.2 → L = 11.2/0.95 ≈ 11.7894737
        # 38 - 11.7894737*(P-1) = 1.2 → (P-1) = 36.8/11.7894737 ≈ 3.1214
        # P ≈ 4.1214.
        lay_stake = 11.2 / 0.95
        lay_price = 1.0 + (36.8 / lay_stake)
        records = _make_pair_records(
            pair_id="P1", run_id="R1", date="2026-03-26",
            fill_prob=0.6, completed=True,
            predicted_pnl=1.0, predicted_stddev=2.0,
            back_price=5.0, lay_price=lay_price,
            back_stake=10.0, lay_stake=lay_stake,
        )
        stats = compute_calibration_stats(records)
        assert stats is not None
        assert len(stats.scatter) == 1
        point = stats.scatter[0]
        assert point.predicted_pnl == pytest.approx(1.0)
        assert point.realised_pnl == pytest.approx(1.2, abs=1e-6)

    def test_single_stddev_value_collapses_to_med(self):
        """If every completed pair has identical predicted stddev,
        every scatter point is ``med`` (p25 == p75 == the value)."""
        records: list[EvaluationBetRecord] = []
        for i in range(5):
            records.extend(_make_pair_records(
                pair_id=f"P{i}", run_id="R1", date="2026-03-26",
                fill_prob=0.6, completed=True,
                predicted_pnl=float(i), predicted_stddev=2.0,
                aggressive_tick=f"2026-03-26T10:{i:02d}:00",
                passive_tick=f"2026-03-26T10:{i:02d}:05",
            ))
        stats = compute_calibration_stats(records)
        assert stats is not None
        assert len(stats.scatter) == 5
        assert all(p.stddev_bucket == "med" for p in stats.scatter)

    def test_directional_run_returns_none(self):
        """Records without pair_id / fill-prob → no scalping pairs →
        compute_calibration_stats returns None."""
        record = EvaluationBetRecord(
            run_id="R1", date="2026-03-26", market_id="1.100",
            tick_timestamp="2026-03-26T10:00:00",
            seconds_to_off=300.0, runner_id=1, runner_name="R",
            action="back", price=5.0, stake=10.0, matched_size=10.0,
            outcome="won", pnl=40.0, pair_id=None,
            fill_prob_at_placement=None,
        )
        assert compute_calibration_stats([record]) is None


# -- API contract tests ---------------------------------------------


def _seed_scalping_model_with_bets(
    store: ModelStore, bets_for_run: list[EvaluationBetRecord],
) -> str:
    """Create a model, an evaluation run, and attach the given bet
    records as its eval parquet. Returns the model id."""
    model_id = store.create_model(
        generation=0,
        architecture_name="ppo_lstm_v1",
        architecture_description="Scalping test",
        hyperparameters={"scalping_mode": True},
    )
    run_id = store.create_evaluation_run(
        model_id=model_id,
        train_cutoff_date="2026-03-25",
        test_days=["2026-03-26"],
    )
    # Rewrite each record with the real run_id.
    records = [
        EvaluationBetRecord(**{**b.__dict__, "run_id": run_id})
        for b in bets_for_run
    ]
    store.record_evaluation_day(EvaluationDayRecord(
        run_id=run_id, date="2026-03-26", day_pnl=0.0,
        bet_count=len(records), winning_bets=0,
        bet_precision=0.0, pnl_per_bet=0.0, early_picks=0,
        profitable=False, arbs_completed=sum(1 for _ in records) // 2,
    ))
    store.write_bet_logs_parquet(run_id, "2026-03-26", records)
    return model_id


class TestCalibrationApi:
    def test_directional_model_returns_null_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            model_id = store.create_model(
                generation=0, architecture_name="ppo_lstm_v1",
                architecture_description="Directional",
                hyperparameters={},
            )
            client = _make_app(store)
            resp = client.get(f"/models/{model_id}")
            assert resp.status_code == 200
            data = resp.json()
            assert "calibration" in data
            assert data["calibration"] is None

    def test_scalping_model_insufficient_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            # Five pairs, all in one bucket — insufficient_data=True.
            records = _perfect_bucket_records(
                run_id="pending", count=5, fill_prob=0.6,
                completed_count=3, bucket_tag="b2",
            )
            model_id = _seed_scalping_model_with_bets(store, records)
            client = _make_app(store)
            resp = client.get(f"/models/{model_id}")
            assert resp.status_code == 200
            data = resp.json()
            cal = data["calibration"]
            assert cal is not None
            assert cal["insufficient_data"] is True
            assert cal["mace"] is None
            # The four buckets are always reported.
            assert len(cal["reliability_buckets"]) == 4

    def test_scalping_model_populated_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            # Four dense buckets, 40 pairs per bucket, observed =
            # midpoint so MACE = 0. Midpoints × 40 are all integer.
            records: list[EvaluationBetRecord] = []
            buckets = (
                (0.10, 0.125, "b0"), (0.40, 0.375, "b1"),
                (0.60, 0.625, "b2"), (0.90, 0.875, "b3"),
            )
            n = 40
            for fp, mid, tag in buckets:
                records.extend(_perfect_bucket_records(
                    run_id="pending", count=n, fill_prob=fp,
                    completed_count=int(mid * n),
                    bucket_tag=tag,
                ))
            model_id = _seed_scalping_model_with_bets(store, records)
            client = _make_app(store)
            resp = client.get(f"/models/{model_id}")
            assert resp.status_code == 200
            cal = resp.json()["calibration"]
            assert cal is not None
            assert cal["insufficient_data"] is False
            assert cal["mace"] == pytest.approx(0.0, abs=1e-9)
            # Scatter has one entry per completed pair.
            total_completed = sum(int(m * n) for _, m, _ in buckets)
            assert len(cal["scatter"]) == total_completed
