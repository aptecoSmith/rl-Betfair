"""Unit tests for training/run_training.py -- Training orchestrator."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.architecture_registry import create_policy
from agents.population_manager import PopulationManager
from data.episode_builder import Day, PriceSize, Race, RunnerMeta, RunnerSnap, Tick
from env.betfair_env import (
    AGENT_STATE_DIM,
    MARKET_DIM,
    RUNNER_DIM,
    VELOCITY_DIM,
)
from registry.model_store import ModelStore
from registry.scoreboard import Scoreboard
from training.run_training import (
    GenerationResult,
    TrainingOrchestrator,
    TrainingRunResult,
)


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


def _make_full_config() -> dict:
    """Config with all sections needed for the orchestrator."""
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
        "population": {
            "size": 4,   # small for fast tests
            "n_elite": 1,
            "selection_top_pct": 0.5,
            "mutation_rate": 0.3,
        },
        "discard_policy": {
            "min_win_rate": 0.35,
            "min_mean_pnl": 0.0,
            "min_sharpe": -0.5,
        },
        "hyperparameters": {
            "search_ranges": {
                "learning_rate": {"type": "float_log", "min": 1e-5, "max": 5e-4},
                "ppo_clip_epsilon": {"type": "float", "min": 0.1, "max": 0.3},
                "entropy_coefficient": {"type": "float", "min": 0.001, "max": 0.05},
                "lstm_hidden_size": {"type": "int_choice", "choices": [64, 128]},
                "mlp_hidden_size": {"type": "int_choice", "choices": [64, 128]},
                "mlp_layers": {"type": "int", "min": 1, "max": 2},
                "observation_window_ticks": {"type": "int", "min": 3, "max": 20},
                "reward_early_pick_bonus": {"type": "float", "min": 1.0, "max": 1.5},
                "reward_efficiency_penalty": {"type": "float", "min": 0.001, "max": 0.05},
            },
        },
    }


# ── Tests: Orchestrator init ─────────────────────────────────────────────────


class TestOrchestratorInit:
    def test_creates_without_store(self):
        config = _make_full_config()
        orch = TrainingOrchestrator(config)
        assert orch.model_store is None
        assert orch.scoreboard is None

    def test_creates_with_store(self, tmp_path):
        config = _make_full_config()
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)
        assert orch.model_store is store
        assert orch.scoreboard is not None


# ── Tests: Empty inputs ──────────────────────────────────────────────────────


class TestOrchestratorEmpty:
    def test_no_train_days_returns_empty(self, tmp_path):
        config = _make_full_config()
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)
        result = orch.run(train_days=[], test_days=[], n_generations=1)
        assert len(result.generations) == 0

    def test_no_test_days_uses_train_days(self, tmp_path):
        """When no test days, training days are used for eval with warning."""
        config = _make_full_config()
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)
        train_day = _make_day("2026-03-26", n_races=1, n_ticks=3)
        result = orch.run(
            train_days=[train_day],
            test_days=[],
            n_generations=1,
            n_epochs=1,
            seed=42,
        )
        assert len(result.generations) == 1


# ── Tests: Single generation ─────────────────────────────────────────────────


class TestSingleGeneration:
    def test_runs_one_generation(self, tmp_path):
        config = _make_full_config()
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)

        result = orch.run(
            train_days=[train],
            test_days=[test],
            n_generations=1,
            n_epochs=1,
            seed=42,
        )
        assert len(result.generations) == 1
        gen = result.generations[0]
        assert gen.generation == 0
        assert len(gen.training_stats) == config["population"]["size"]

    def test_models_registered_in_store(self, tmp_path):
        config = _make_full_config()
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)

        orch.run(
            train_days=[train],
            test_days=[test],
            n_generations=1,
            n_epochs=1,
            seed=42,
        )
        models = store.list_models()
        assert len(models) >= config["population"]["size"]

    def test_scores_computed(self, tmp_path):
        config = _make_full_config()
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)

        result = orch.run(
            train_days=[train],
            test_days=[test],
            n_generations=1,
            n_epochs=1,
            seed=42,
        )
        gen = result.generations[0]
        assert len(gen.scores) > 0

    def test_no_selection_on_single_gen(self, tmp_path):
        """Last generation should not do selection/breeding."""
        config = _make_full_config()
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)

        result = orch.run(
            train_days=[train],
            test_days=[test],
            n_generations=1,
            n_epochs=1,
            seed=42,
        )
        gen = result.generations[0]
        assert gen.selection is None
        assert gen.children == []


# ── Tests: Two generations ───────────────────────────────────────────────────


class TestTwoGenerations:
    def test_runs_two_generations(self, tmp_path):
        config = _make_full_config()
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)

        result = orch.run(
            train_days=[train],
            test_days=[test],
            n_generations=2,
            n_epochs=1,
            seed=42,
        )
        assert len(result.generations) == 2

    def test_gen0_has_selection(self, tmp_path):
        config = _make_full_config()
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)

        result = orch.run(
            train_days=[train],
            test_days=[test],
            n_generations=2,
            n_epochs=1,
            seed=42,
        )
        gen0 = result.generations[0]
        assert gen0.selection is not None
        assert len(gen0.selection.survivors) > 0

    def test_gen0_has_children(self, tmp_path):
        config = _make_full_config()
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)

        result = orch.run(
            train_days=[train],
            test_days=[test],
            n_generations=2,
            n_epochs=1,
            seed=42,
        )
        gen0 = result.generations[0]
        assert len(gen0.children) > 0

    def test_gen1_is_last_no_selection(self, tmp_path):
        config = _make_full_config()
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)

        result = orch.run(
            train_days=[train],
            test_days=[test],
            n_generations=2,
            n_epochs=1,
            seed=42,
        )
        gen1 = result.generations[1]
        assert gen1.selection is None

    def test_registry_has_all_models(self, tmp_path):
        config = _make_full_config()
        pop_size = config["population"]["size"]
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)

        orch.run(
            train_days=[train],
            test_days=[test],
            n_generations=2,
            n_epochs=1,
            seed=42,
        )
        # Gen 0 creates pop_size models. Gen 0 breeds children to refill.
        # Gen 1 trains survivors + children.
        # So total models = pop_size (gen0) + children (gen0 breeding)
        models = store.list_models()
        assert len(models) >= pop_size

    def test_genetic_log_written(self, tmp_path):
        config = _make_full_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)

        orch.run(
            train_days=[train],
            test_days=[test],
            n_generations=2,
            n_epochs=1,
            seed=42,
        )

        genetics_dir = tmp_path / "logs" / "genetics"
        assert genetics_dir.exists()
        log_files = list(genetics_dir.glob("gen_0_*.log"))
        assert len(log_files) == 1

    def test_genetic_events_in_db(self, tmp_path):
        config = _make_full_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)

        orch.run(
            train_days=[train],
            test_days=[test],
            n_generations=2,
            n_epochs=1,
            seed=42,
        )

        events = store.get_genetic_events(generation=0)
        assert len(events) > 0

    def test_final_rankings_populated(self, tmp_path):
        config = _make_full_config()
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)

        result = orch.run(
            train_days=[train],
            test_days=[test],
            n_generations=2,
            n_epochs=1,
            seed=42,
        )
        assert len(result.final_rankings) > 0

    def test_evaluation_runs_created(self, tmp_path):
        config = _make_full_config()
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)

        result = orch.run(
            train_days=[train],
            test_days=[test],
            n_generations=2,
            n_epochs=1,
            seed=42,
        )

        # Every trained agent should have an evaluation run
        models = store.list_models()
        for m in models:
            run = store.get_latest_evaluation_run(m.model_id)
            assert run is not None, f"Model {m.model_id[:12]} has no evaluation run"


# ── Tests: Progress events ───────────────────────────────────────────────────


class TestOrchestratorProgress:
    def test_emits_phase_events(self, tmp_path):
        config = _make_full_config()
        config["population"]["size"] = 2  # minimal for speed
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        queue = asyncio.Queue()
        orch = TrainingOrchestrator(config, model_store=store, progress_queue=queue)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)

        orch.run(
            train_days=[train],
            test_days=[test],
            n_generations=1,
            n_epochs=1,
            seed=42,
        )

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        # Should have phase_start and phase_complete events
        event_types = [e["event"] for e in events]
        assert "phase_start" in event_types
        assert "phase_complete" in event_types

    def test_phase_order_single_gen(self, tmp_path):
        config = _make_full_config()
        config["population"]["size"] = 2
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        queue = asyncio.Queue()
        orch = TrainingOrchestrator(config, model_store=store, progress_queue=queue)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)

        orch.run(
            train_days=[train],
            test_days=[test],
            n_generations=1,
            n_epochs=1,
            seed=42,
        )

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        # Extract phase transitions
        phases = [
            (e["event"], e["phase"])
            for e in events
            if e["event"] in ("phase_start", "phase_complete")
        ]

        # Should see: training start → training complete → eval start → eval complete → scoring start → scoring complete
        phase_names = [p[1] for p in phases if p[0] == "phase_start"]
        assert "training" in phase_names
        assert "evaluating" in phase_names
        assert "scoring" in phase_names

    def test_progress_events_emitted_for_agents(self, tmp_path):
        config = _make_full_config()
        config["population"]["size"] = 2
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        queue = asyncio.Queue()
        orch = TrainingOrchestrator(config, model_store=store, progress_queue=queue)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)

        orch.run(
            train_days=[train],
            test_days=[test],
            n_generations=1,
            n_epochs=1,
            seed=42,
        )

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        progress_events = [e for e in events if e["event"] == "progress"]
        assert len(progress_events) > 0

    def test_run_complete_event_emitted(self, tmp_path):
        config = _make_full_config()
        config["population"]["size"] = 2
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        queue = asyncio.Queue()
        orch = TrainingOrchestrator(config, model_store=store, progress_queue=queue)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)

        orch.run(
            train_days=[train],
            test_days=[test],
            n_generations=1,
            n_epochs=1,
            seed=42,
        )

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        complete_events = [e for e in events if e["phase"] == "run_complete"]
        assert len(complete_events) == 1

    def test_two_gen_has_selection_events(self, tmp_path):
        config = _make_full_config()
        config["population"]["size"] = 4
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        queue = asyncio.Queue()
        orch = TrainingOrchestrator(config, model_store=store, progress_queue=queue)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)

        orch.run(
            train_days=[train],
            test_days=[test],
            n_generations=2,
            n_epochs=1,
            seed=42,
        )

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        phase_starts = [e["phase"] for e in events if e["event"] == "phase_start"]
        assert "selecting" in phase_starts
        assert "breeding" in phase_starts


# ── Tests: Result structure ──────────────────────────────────────────────────


class TestResultStructure:
    def test_run_id_is_uuid(self, tmp_path):
        config = _make_full_config()
        config["population"]["size"] = 2
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)

        result = orch.run(
            train_days=[train],
            test_days=[test],
            n_generations=1,
            n_epochs=1,
            seed=42,
        )
        assert len(result.run_id) == 36  # UUID format

    def test_generation_result_has_training_stats(self, tmp_path):
        config = _make_full_config()
        config["population"]["size"] = 2
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)

        result = orch.run(
            train_days=[train],
            test_days=[test],
            n_generations=1,
            n_epochs=1,
            seed=42,
        )
        gen = result.generations[0]
        assert len(gen.training_stats) == 2
        for model_id, stats in gen.training_stats.items():
            assert stats.episodes_completed > 0

    def test_weights_saved_after_training(self, tmp_path):
        config = _make_full_config()
        config["population"]["size"] = 2
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)

        train = _make_day("2026-03-26", n_races=1, n_ticks=3)
        test = _make_day("2026-03-27", n_races=1, n_ticks=3)

        orch.run(
            train_days=[train],
            test_days=[test],
            n_generations=1,
            n_epochs=1,
            seed=42,
        )

        models = store.list_models()
        for m in models:
            assert m.weights_path is not None
            assert Path(m.weights_path).exists()


# ── Tests: GPU detection & enforcement ───────────────────────────────────────


class TestGPUDetection:
    def test_auto_detects_device(self, tmp_path):
        """Orchestrator should auto-detect CUDA or CPU."""
        config = _make_full_config()
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert orch.device == expected

    def test_explicit_device_overrides(self, tmp_path):
        """Explicitly passing device should override auto-detection."""
        config = _make_full_config()
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store, device="cpu")
        assert orch.device == "cpu"

    def test_require_gpu_raises_on_cpu(self, tmp_path):
        """When require_gpu=true and no GPU, should raise RuntimeError."""
        if torch.cuda.is_available():
            pytest.skip("GPU is available — this test checks the CPU fallback path")
        config = _make_full_config()
        config["training"]["require_gpu"] = True
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        with pytest.raises(RuntimeError, match="require_gpu"):
            TrainingOrchestrator(config, model_store=store)

    def test_require_gpu_passes_with_gpu(self, tmp_path):
        """When require_gpu=true and GPU is available, should succeed."""
        if not torch.cuda.is_available():
            pytest.skip("No GPU available")
        config = _make_full_config()
        config["training"]["require_gpu"] = True
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store)
        assert orch.device == "cuda"

    def test_require_gpu_false_allows_cpu(self, tmp_path):
        """When require_gpu=false, CPU fallback is allowed."""
        config = _make_full_config()
        config["training"]["require_gpu"] = False
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        orch = TrainingOrchestrator(config, model_store=store, device="cpu")
        assert orch.device == "cpu"


# ── Tests: Parquet bet logs ──────────────────────────────────────────────────


class TestParquetBetLogs:
    def test_write_and_read_100_bets(self, tmp_path):
        """Write 100 bets to Parquet and read them back."""
        from registry.model_store import EvaluationBetRecord

        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        model_id = store.create_model(
            generation=0,
            architecture_name="ppo_lstm_v1",
            architecture_description="test",
            hyperparameters={},
        )
        run_id = store.create_evaluation_run(model_id, "2026-03-25", ["2026-03-26"])

        bets = [
            EvaluationBetRecord(
                run_id=run_id, date="2026-03-26", market_id=f"1.{i}",
                tick_timestamp="", seconds_to_off=0.0, runner_id=1,
                runner_name="Horse", action="back", price=3.0,
                stake=10.0, matched_size=10.0, outcome="won", pnl=20.0,
            )
            for i in range(100)
        ]
        store.write_bet_logs_parquet(run_id, "2026-03-26", bets)

        stored = store.get_evaluation_bets(run_id)
        assert len(stored) == 100

    def test_empty_write_is_noop(self, tmp_path):
        store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
        result = store.write_bet_logs_parquet("run-1", "2026-03-26", [])
        assert result is None
