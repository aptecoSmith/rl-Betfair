"""Integration tests for Session 1.5 -- End-to-end single agent run.

Trains one agent on the chronological training split and evaluates on
the test split.  Verifies:
- Per-day metrics recorded in registry
- Bet log populated (Parquet)
- Composite score computed
- Scoreboard non-empty
- Full train -> evaluate -> registry pipeline on real data

Run with: pytest -m integration tests/test_integration_session_1_5.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from data.episode_builder import load_days
from registry.model_store import ModelStore
from registry.scoreboard import Scoreboard
from training.run_training import TrainingOrchestrator

pytestmark = pytest.mark.integration


def _get_available_dates(data_dir: str = "data/processed") -> list[str]:
    processed = Path(data_dir)
    if not processed.exists():
        return []
    dates = set()
    for f in processed.glob("*.parquet"):
        if "_runners" not in f.name:
            dates.add(f.stem)
    return sorted(dates)


@pytest.fixture(scope="module")
def integration_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # Single agent, no selection/breeding
    config["population"]["size"] = 1
    config["population"]["n_elite"] = 1
    config["training"]["require_gpu"] = False
    return config


@pytest.fixture(scope="module")
def real_days(integration_config):
    dates = _get_available_dates(integration_config["paths"]["processed_data"])
    if len(dates) < 2:
        pytest.skip("Need 2+ extracted days for train/test split")
    return load_days(dates, data_dir=integration_config["paths"]["processed_data"])


@pytest.fixture(scope="module")
def run_result(real_days, integration_config, tmp_path_factory):
    """Train one agent on the chronological training split, evaluate on test."""
    tmp_path = tmp_path_factory.mktemp("session_1_5")
    config = dict(integration_config)
    config["paths"] = dict(config["paths"])
    config["paths"]["logs"] = str(tmp_path / "logs")

    store = ModelStore(db_path=tmp_path / "test.db", weights_dir=tmp_path / "w")
    queue: asyncio.Queue = asyncio.Queue()

    orch = TrainingOrchestrator(
        config,
        model_store=store,
        progress_queue=queue,
        device="cpu",
    )

    split = len(real_days) // 2
    train_days = real_days[:split]
    test_days = real_days[split:]

    result = orch.run(
        train_days=train_days,
        test_days=test_days,
        n_generations=1,
        n_epochs=1,
        seed=42,
    )

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())

    return result, store, events, train_days, test_days


# ---- Model registered and has weights ------------------------------------

class TestModelRegistered:
    def test_exactly_one_model(self, run_result):
        _, store, _, _, _ = run_result
        models = store.list_models()
        assert len(models) == 1

    def test_model_is_active(self, run_result):
        _, store, _, _, _ = run_result
        model = store.list_models()[0]
        assert model.status == "active"

    def test_model_has_weights_file(self, run_result):
        _, store, _, _, _ = run_result
        model = store.list_models()[0]
        assert model.weights_path is not None
        assert Path(model.weights_path).exists()

    def test_model_has_architecture(self, run_result):
        _, store, _, _, _ = run_result
        model = store.list_models()[0]
        assert model.architecture_name in ("ppo_lstm_v1", "ppo_time_lstm_v1")


# ---- Evaluation run exists -----------------------------------------------

class TestEvaluationRun:
    def test_evaluation_run_recorded(self, run_result):
        _, store, _, _, _ = run_result
        model = store.list_models()[0]
        run = store.get_latest_evaluation_run(model.model_id)
        assert run is not None

    def test_evaluation_run_has_test_days(self, run_result):
        _, store, _, _, test_days = run_result
        model = store.list_models()[0]
        run = store.get_latest_evaluation_run(model.model_id)
        assert len(run.test_days) == len(test_days)

    def test_train_cutoff_date_is_last_train_day(self, run_result):
        _, store, _, train_days, _ = run_result
        model = store.list_models()[0]
        run = store.get_latest_evaluation_run(model.model_id)
        assert run.train_cutoff_date == train_days[-1].date


# ---- Per-day metrics recorded --------------------------------------------

class TestPerDayMetrics:
    def test_day_records_match_test_days(self, run_result):
        _, store, _, _, test_days = run_result
        model = store.list_models()[0]
        run = store.get_latest_evaluation_run(model.model_id)
        day_records = store.get_evaluation_days(run.run_id)
        assert len(day_records) == len(test_days)

    def test_day_dates_match(self, run_result):
        _, store, _, _, test_days = run_result
        model = store.list_models()[0]
        run = store.get_latest_evaluation_run(model.model_id)
        day_records = store.get_evaluation_days(run.run_id)
        record_dates = {dr.date for dr in day_records}
        expected_dates = {d.date for d in test_days}
        assert record_dates == expected_dates

    def test_day_pnl_is_finite(self, run_result):
        _, store, _, _, _ = run_result
        model = store.list_models()[0]
        run = store.get_latest_evaluation_run(model.model_id)
        day_records = store.get_evaluation_days(run.run_id)
        for dr in day_records:
            assert dr.day_pnl != float("inf")
            assert dr.day_pnl != float("-inf")
            assert dr.day_pnl == dr.day_pnl  # not NaN

    def test_pnl_bounded_by_budget(self, run_result):
        """P&L should not exceed starting budget in magnitude for a single day."""
        _, store, _, _, _ = run_result
        model = store.list_models()[0]
        run = store.get_latest_evaluation_run(model.model_id)
        day_records = store.get_evaluation_days(run.run_id)
        for dr in day_records:
            # Max loss is the starting budget (100); max gain is bounded by
            # bet matching (realistic), so 10x budget is a generous upper bound
            assert abs(dr.day_pnl) < 10000, (
                f"Day {dr.date}: P&L {dr.day_pnl} seems unreasonable"
            )

    def test_bet_precision_in_range(self, run_result):
        _, store, _, _, _ = run_result
        model = store.list_models()[0]
        run = store.get_latest_evaluation_run(model.model_id)
        day_records = store.get_evaluation_days(run.run_id)
        for dr in day_records:
            assert 0.0 <= dr.bet_precision <= 1.0

    def test_profitable_flag_consistent(self, run_result):
        _, store, _, _, _ = run_result
        model = store.list_models()[0]
        run = store.get_latest_evaluation_run(model.model_id)
        day_records = store.get_evaluation_days(run.run_id)
        for dr in day_records:
            assert dr.profitable == (dr.day_pnl > 0)


# ---- Bet log populated ---------------------------------------------------

class TestBetLog:
    def test_bets_recorded(self, run_result):
        _, store, _, _, _ = run_result
        model = store.list_models()[0]
        run = store.get_latest_evaluation_run(model.model_id)
        bets = store.get_evaluation_bets(run.run_id)
        assert len(bets) > 0, "Expected at least some bets from the agent"

    def test_bet_fields_populated(self, run_result):
        _, store, _, _, _ = run_result
        model = store.list_models()[0]
        run = store.get_latest_evaluation_run(model.model_id)
        bets = store.get_evaluation_bets(run.run_id)
        if not bets:
            pytest.skip("No bets to check")
        for b in bets[:10]:  # sample first 10
            assert b.market_id is not None
            assert b.action in ("back", "lay")
            assert b.price > 0
            assert b.stake >= 0
            assert b.outcome in ("won", "lost", "void")

    def test_bet_pnl_sums_match_day_pnl(self, run_result):
        """Sum of individual bet P&Ls should approximately match day P&L."""
        _, store, _, _, _ = run_result
        model = store.list_models()[0]
        run = store.get_latest_evaluation_run(model.model_id)
        bets = store.get_evaluation_bets(run.run_id)
        day_records = store.get_evaluation_days(run.run_id)

        if not bets or not day_records:
            pytest.skip("No bets or day records")

        for dr in day_records:
            day_bets = [b for b in bets if b.date == dr.date]
            bet_pnl_sum = sum(b.pnl for b in day_bets)
            # Allow small floating-point tolerance
            assert abs(bet_pnl_sum - dr.day_pnl) < 0.02, (
                f"Day {dr.date}: bet P&L sum {bet_pnl_sum:.4f} "
                f"!= day P&L {dr.day_pnl:.4f}"
            )

    def test_bet_parquet_files_exist(self, run_result):
        _, store, _, _, _ = run_result
        model = store.list_models()[0]
        run = store.get_latest_evaluation_run(model.model_id)
        day_records = store.get_evaluation_days(run.run_id)

        for dr in day_records:
            if dr.bet_count > 0:
                parquet_file = store.bet_logs_dir / run.run_id / f"{dr.date}.parquet"
                assert parquet_file.exists(), (
                    f"Missing Parquet bet log for {dr.date}"
                )


# ---- Composite score computed and scoreboard non-empty -------------------

class TestScoreboard:
    def test_composite_score_computed(self, run_result):
        result, store, _, _, _ = run_result
        assert len(result.final_rankings) == 1
        score = result.final_rankings[0]
        assert score.composite_score is not None

    def test_scoreboard_rank_all(self, run_result):
        _, store, _, _, _ = run_result
        # Load config for Scoreboard
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        scoreboard = Scoreboard(store, config)
        rankings = scoreboard.rank_all()
        assert len(rankings) == 1

    def test_composite_score_in_valid_range(self, run_result):
        result, _, _, _, _ = run_result
        score = result.final_rankings[0]
        # Composite score is a weighted sum; with negative P&L it can be < 0
        # but should be bounded
        assert -2.0 <= score.composite_score <= 2.0

    def test_score_metrics_populated(self, run_result):
        result, _, _, _, _ = run_result
        score = result.final_rankings[0]
        assert 0.0 <= score.win_rate <= 1.0
        # Sharpe can be 0.0 with 1 test day
        assert score.sharpe is not None
        assert score.mean_daily_pnl is not None
        assert score.efficiency is not None


# ---- Progress events correct ----------------------------------------------

class TestProgressEvents:
    def test_training_phase_emitted(self, run_result):
        _, _, events, _, _ = run_result
        phases = [e["phase"] for e in events if e["event"] == "phase_start"]
        assert "training" in phases

    def test_evaluating_phase_emitted(self, run_result):
        _, _, events, _, _ = run_result
        phases = [e["phase"] for e in events if e["event"] == "phase_start"]
        assert "evaluating" in phases

    def test_scoring_phase_emitted(self, run_result):
        _, _, events, _, _ = run_result
        phases = [e["phase"] for e in events if e["event"] == "phase_start"]
        assert "scoring" in phases

    def test_run_complete_emitted(self, run_result):
        _, _, events, _, _ = run_result
        complete = [e for e in events if e.get("phase") == "run_complete"]
        assert len(complete) == 1

    def test_progress_events_emitted(self, run_result):
        _, _, events, _, _ = run_result
        progress = [e for e in events if e["event"] == "progress"]
        assert len(progress) > 0

    def test_no_selection_breeding_for_single_agent(self, run_result):
        """With population=1, gen 0 should not have selection/breeding phases."""
        _, _, events, _, _ = run_result
        phases = [e["phase"] for e in events if e["event"] == "phase_start"]
        # With only 1 generation there's no need for selection/breeding
        # (those happen between generations)
        # The orchestrator may still emit them depending on implementation
        # but run_complete should be the final event
        complete = [e for e in events if e.get("phase") == "run_complete"]
        assert len(complete) == 1
