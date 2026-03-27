"""Unit tests for training/evaluator.py -- Model evaluation on test days."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import numpy as np
import pytest
import torch

from agents.architecture_registry import create_policy
from data.episode_builder import Day, PriceSize, Race, RunnerMeta, RunnerSnap, Tick
from env.betfair_env import (
    AGENT_STATE_DIM,
    MARKET_DIM,
    RUNNER_DIM,
    VELOCITY_DIM,
)
from registry.model_store import EvaluationDayRecord, ModelStore
from training.evaluator import Evaluator


# ── Synthetic data helpers ────────────────────────────────────────────────────


def _make_runner_meta(selection_id: int, name: str = "Horse") -> RunnerMeta:
    return RunnerMeta(
        selection_id=selection_id,
        runner_name=name,
        sort_priority="1",
        handicap="0",
        sire_name="Sire",
        dam_name="Dam",
        damsire_name="DamSire",
        bred="GB",
        official_rating="85",
        adjusted_rating="85",
        age="4",
        sex_type="GELDING",
        colour_type="BAY",
        weight_value="140",
        weight_units="LB",
        jockey_name="J Smith",
        jockey_claim="0",
        trainer_name="T Jones",
        owner_name="Owner",
        stall_draw="3",
        cloth_number="1",
        form="1234",
        days_since_last_run="14",
        wearing="",
        forecastprice_numerator="3",
        forecastprice_denominator="1",
    )


def _make_runner_snap(
    selection_id: int,
    ltp: float = 4.0,
    back_price: float = 4.0,
    lay_price: float = 4.2,
    size: float = 100.0,
    status: str = "ACTIVE",
) -> RunnerSnap:
    return RunnerSnap(
        selection_id=selection_id,
        status=status,
        last_traded_price=ltp,
        total_matched=500.0,
        starting_price_near=0.0,
        starting_price_far=0.0,
        adjustment_factor=None,
        bsp=None,
        sort_priority=1,
        removal_date=None,
        available_to_back=[PriceSize(price=back_price, size=size)],
        available_to_lay=[PriceSize(price=lay_price, size=size)],
    )


def _make_tick(
    market_id: str,
    seq: int,
    runners: list[RunnerSnap],
    start_time: datetime | None = None,
    timestamp: datetime | None = None,
    in_play: bool = False,
    winner: int | None = None,
) -> Tick:
    if start_time is None:
        start_time = datetime(2026, 3, 26, 14, 0, 0)
    if timestamp is None:
        timestamp = start_time - timedelta(seconds=600 - seq * 5)
    return Tick(
        market_id=market_id,
        timestamp=timestamp,
        sequence_number=seq,
        venue="Newmarket",
        market_start_time=start_time,
        number_of_active_runners=len(runners),
        traded_volume=10000.0,
        in_play=in_play,
        winner_selection_id=winner,
        race_status=None,
        temperature=15.0,
        precipitation=0.0,
        wind_speed=5.0,
        wind_direction=180.0,
        humidity=60.0,
        weather_code=0,
        runners=runners,
    )


def _make_race(
    market_id: str = "1.200000001",
    start_time: datetime | None = None,
    n_ticks: int = 5,
    n_runners: int = 3,
    winner_sid: int = 1,
) -> Race:
    if start_time is None:
        start_time = datetime(2026, 3, 26, 14, 0, 0)
    runner_ids = list(range(1, n_runners + 1))
    runners = [_make_runner_snap(sid, ltp=3.0 + sid) for sid in runner_ids]
    ticks: list[Tick] = []

    for i in range(n_ticks):
        ts = start_time - timedelta(seconds=600 - i * 5)
        ticks.append(_make_tick(
            market_id, seq=i, runners=runners,
            start_time=start_time, timestamp=ts,
            in_play=False, winner=winner_sid,
        ))
    # Add one in-play tick
    ticks.append(_make_tick(
        market_id, seq=n_ticks, runners=runners,
        start_time=start_time,
        timestamp=start_time + timedelta(seconds=5),
        in_play=True, winner=winner_sid,
    ))

    runner_meta = {sid: _make_runner_meta(sid, f"Horse{sid}") for sid in runner_ids}
    return Race(
        market_id=market_id,
        venue="Newmarket",
        market_start_time=start_time,
        winner_selection_id=winner_sid,
        ticks=ticks,
        runner_metadata=runner_meta,
    )


def _make_day(
    date: str = "2026-03-26",
    n_races: int = 2,
    n_ticks: int = 5,
    n_runners: int = 3,
) -> Day:
    races = []
    for i in range(n_races):
        start = datetime(2026, 3, 26, 14 + i, 0, 0)
        races.append(_make_race(
            market_id=f"1.{200000001 + i}",
            start_time=start,
            n_ticks=n_ticks,
            n_runners=n_runners,
            winner_sid=1,
        ))
    return Day(date=date, races=races)


def _make_config() -> dict:
    return {
        "training": {
            "architecture": "ppo_lstm_v1",
            "starting_budget": 100.0,
            "max_runners": 14,
        },
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "efficiency_penalty": 0.01,
            "coefficients": {
                "win_rate": 0.35,
                "sharpe": 0.30,
                "mean_daily_pnl": 0.15,
                "efficiency": 0.20,
            },
        },
        "paths": {
            "processed_data": "data/processed",
            "model_weights": "registry/weights",
            "logs": "logs",
            "registry_db": "registry/models.db",
        },
    }


def _make_policy(config: dict):
    max_runners = config["training"]["max_runners"]
    obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * max_runners) + AGENT_STATE_DIM
    action_dim = max_runners * 2
    return create_policy(
        name="ppo_lstm_v1",
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_runners=max_runners,
        hyperparams={"lstm_hidden_size": 64, "mlp_hidden_size": 64, "mlp_layers": 1},
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestEvaluatorInit:
    """Test Evaluator construction."""

    def test_creates_without_store(self):
        config = _make_config()
        evaluator = Evaluator(config)
        assert evaluator.model_store is None

    def test_creates_with_store(self, tmp_path):
        config = _make_config()
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        evaluator = Evaluator(config, model_store=store)
        assert evaluator.model_store is store


class TestEvaluateEmpty:
    """Test evaluation with empty inputs."""

    def test_empty_test_days_returns_empty(self):
        config = _make_config()
        evaluator = Evaluator(config)
        policy = _make_policy(config)
        run_id, records = evaluator.evaluate("test-id", policy, [], "2026-03-26")
        assert run_id is None
        assert records == []


class TestEvaluateOnSyntheticDay:
    """Test evaluation on synthetic data without persistence."""

    def test_returns_day_records(self):
        config = _make_config()
        evaluator = Evaluator(config)
        policy = _make_policy(config)
        day = _make_day()
        run_id, records = evaluator.evaluate("test-id", policy, [day], "2026-03-26")
        assert run_id is None  # no store
        assert len(records) == 1
        rec = records[0]
        assert rec.date == "2026-03-26"
        assert isinstance(rec.day_pnl, float)
        assert isinstance(rec.bet_count, int)
        assert rec.bet_count >= 0

    def test_multiple_days(self):
        config = _make_config()
        evaluator = Evaluator(config)
        policy = _make_policy(config)
        days = [_make_day("2026-03-26"), _make_day("2026-03-27")]
        _, records = evaluator.evaluate("test-id", policy, days, "2026-03-25")
        assert len(records) == 2
        assert records[0].date == "2026-03-26"
        assert records[1].date == "2026-03-27"

    def test_profitable_flag(self):
        config = _make_config()
        evaluator = Evaluator(config)
        policy = _make_policy(config)
        day = _make_day()
        _, records = evaluator.evaluate("test-id", policy, [day], "2026-03-25")
        rec = records[0]
        assert rec.profitable == (rec.day_pnl > 0)

    def test_bet_precision_no_bets(self):
        """When no bets placed, precision should be 0."""
        config = _make_config()
        evaluator = Evaluator(config)
        policy = _make_policy(config)
        day = _make_day(n_ticks=1)  # very short, unlikely to place bets
        _, records = evaluator.evaluate("test-id", policy, [day], "2026-03-25")
        rec = records[0]
        if rec.bet_count == 0:
            assert rec.bet_precision == 0.0
            assert rec.pnl_per_bet == 0.0

    def test_uses_deterministic_actions(self):
        """Two evaluations of the same policy on the same day should match."""
        config = _make_config()
        policy = _make_policy(config)
        day = _make_day()

        eval1 = Evaluator(config)
        _, records1 = eval1.evaluate("id1", policy, [day], "2026-03-25")

        eval2 = Evaluator(config)
        _, records2 = eval2.evaluate("id2", policy, [day], "2026-03-25")

        assert records1[0].day_pnl == records2[0].day_pnl
        assert records1[0].bet_count == records2[0].bet_count


class TestEvaluateWithStore:
    """Test evaluation persists to ModelStore."""

    def test_creates_evaluation_run(self, tmp_path):
        config = _make_config()
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        policy = _make_policy(config)

        # Create model first
        model_id = store.create_model(
            generation=0,
            architecture_name="ppo_lstm_v1",
            architecture_description="test",
            hyperparameters={},
        )
        store.save_weights(model_id, policy.state_dict())

        evaluator = Evaluator(config, model_store=store)
        day = _make_day()
        run_id, records = evaluator.evaluate(model_id, policy, [day], "2026-03-25")

        assert run_id is not None
        run = store.get_latest_evaluation_run(model_id)
        assert run is not None
        assert run.run_id == run_id

    def test_persists_day_records(self, tmp_path):
        config = _make_config()
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        policy = _make_policy(config)

        model_id = store.create_model(
            generation=0,
            architecture_name="ppo_lstm_v1",
            architecture_description="test",
            hyperparameters={},
        )
        store.save_weights(model_id, policy.state_dict())

        evaluator = Evaluator(config, model_store=store)
        days = [_make_day("2026-03-26"), _make_day("2026-03-27")]
        run_id, _ = evaluator.evaluate(model_id, policy, days, "2026-03-25")

        stored_days = store.get_evaluation_days(run_id)
        assert len(stored_days) == 2

    def test_persists_bet_records_to_parquet(self, tmp_path):
        config = _make_config()
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        policy = _make_policy(config)

        model_id = store.create_model(
            generation=0,
            architecture_name="ppo_lstm_v1",
            architecture_description="test",
            hyperparameters={},
        )
        store.save_weights(model_id, policy.state_dict())

        evaluator = Evaluator(config, model_store=store)
        day = _make_day(n_ticks=20)  # more ticks = more chance of bets
        run_id, records = evaluator.evaluate(model_id, policy, [day], "2026-03-25")

        stored_bets = store.get_evaluation_bets(run_id)
        # Bet count should match records
        assert len(stored_bets) == records[0].bet_count

        # Verify Parquet files exist
        if records[0].bet_count > 0:
            parquet_dir = store.bet_logs_dir / run_id
            assert parquet_dir.exists()
            parquet_files = list(parquet_dir.glob("*.parquet"))
            assert len(parquet_files) == 1


class TestEvaluateProgressEvents:
    """Test that progress events are emitted correctly."""

    def test_emits_progress_events(self):
        config = _make_config()
        queue = asyncio.Queue()
        evaluator = Evaluator(config, progress_queue=queue)
        policy = _make_policy(config)
        days = [_make_day("2026-03-26"), _make_day("2026-03-27")]
        evaluator.evaluate("test-id", policy, days, "2026-03-25")

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        assert len(events) == 2  # one per day
        for e in events:
            assert e["event"] == "progress"
            assert e["phase"] == "evaluating"
            assert "item" in e
            assert "detail" in e

    def test_progress_events_in_order(self):
        config = _make_config()
        queue = asyncio.Queue()
        evaluator = Evaluator(config, progress_queue=queue)
        policy = _make_policy(config)
        days = [_make_day("2026-03-26"), _make_day("2026-03-27")]
        evaluator.evaluate("test-id", policy, days, "2026-03-25")

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        # Progress should show increasing completion
        assert events[0]["item"]["completed"] == 1
        assert events[1]["item"]["completed"] == 2

    def test_no_events_when_no_queue(self):
        """Evaluator should not fail when no queue is provided."""
        config = _make_config()
        evaluator = Evaluator(config)
        policy = _make_policy(config)
        day = _make_day()
        _, records = evaluator.evaluate("test-id", policy, [day], "2026-03-25")
        assert len(records) == 1


class TestEvaluateDayMetrics:
    """Test per-day metric correctness."""

    def test_day_pnl_is_float(self):
        config = _make_config()
        evaluator = Evaluator(config)
        policy = _make_policy(config)
        day = _make_day()
        _, records = evaluator.evaluate("test-id", policy, [day], "2026-03-25")
        assert isinstance(records[0].day_pnl, float)

    def test_winning_bets_leq_bet_count(self):
        config = _make_config()
        evaluator = Evaluator(config)
        policy = _make_policy(config)
        day = _make_day()
        _, records = evaluator.evaluate("test-id", policy, [day], "2026-03-25")
        assert records[0].winning_bets <= records[0].bet_count

    def test_bet_precision_in_range(self):
        config = _make_config()
        evaluator = Evaluator(config)
        policy = _make_policy(config)
        day = _make_day()
        _, records = evaluator.evaluate("test-id", policy, [day], "2026-03-25")
        assert 0.0 <= records[0].bet_precision <= 1.0

    def test_early_picks_non_negative(self):
        config = _make_config()
        evaluator = Evaluator(config)
        policy = _make_policy(config)
        day = _make_day()
        _, records = evaluator.evaluate("test-id", policy, [day], "2026-03-25")
        assert records[0].early_picks >= 0
