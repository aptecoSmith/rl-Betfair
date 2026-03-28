"""
Session 4.7 — Opportunity window metric tests.

Tests the compute_opportunity_window function and the wiring of
opportunity_window_s through bet records, day records, and scoreboard.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.episode_builder import (
    Day, PriceSize, Race, RunnerMeta, RunnerSnap, Tick,
)
from env.bet_manager import Bet, BetOutcome, BetSide
from registry.model_store import EvaluationBetRecord, EvaluationDayRecord, ModelStore
from registry.scoreboard import ModelScore, Scoreboard
from training.evaluator import compute_opportunity_window


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_tick(
    seq: int,
    timestamp: datetime,
    in_play: bool = False,
    back_price: float = 0.0,
    back_size: float = 50.0,
    lay_price: float = 0.0,
    lay_size: float = 50.0,
    selection_id: int = 1,
) -> Tick:
    """Create a minimal Tick with one runner."""
    atb = [PriceSize(back_price, back_size)] if back_price > 0 else []
    atl = [PriceSize(lay_price, lay_size)] if lay_price > 0 else []
    runner = RunnerSnap(
        selection_id=selection_id,
        status="ACTIVE",
        last_traded_price=back_price or lay_price or 5.0,
        total_matched=100.0,
        starting_price_near=0.0,
        starting_price_far=0.0,
        adjustment_factor=None,
        bsp=None,
        sort_priority=1,
        removal_date=None,
        available_to_back=atb,
        available_to_lay=atl,
    )
    return Tick(
        market_id="m1",
        timestamp=timestamp,
        sequence_number=seq,
        venue="Test",
        market_start_time=datetime(2026, 1, 1, 14, 0),
        number_of_active_runners=1,
        traded_volume=100.0,
        in_play=in_play,
        winner_selection_id=None,
        race_status=None,
        temperature=None, precipitation=None, wind_speed=None,
        wind_direction=None, humidity=None, weather_code=None,
        runners=[runner],
    )


def _make_race(ticks: list[Tick]) -> Race:
    return Race(
        market_id="m1",
        venue="Test",
        market_start_time=datetime(2026, 1, 1, 14, 0),
        winner_selection_id=1,
        ticks=ticks,
        runner_metadata={},
        winning_selection_ids={1},
    )


# ── compute_opportunity_window tests ────────────────────────────────────────


class TestComputeOpportunityWindow:

    def test_back_price_available_5_ticks(self):
        """Back price 4.0 available on ticks 0-4 (25s), bet at tick 2."""
        base = datetime(2026, 1, 1, 13, 50)
        ticks = [
            _make_tick(i, base + timedelta(seconds=i * 5), back_price=4.0)
            for i in range(5)
        ]
        race = _make_race(ticks)
        window = compute_opportunity_window(race, 2, 1, "back", 4.0)
        # ticks 0-4 span 20 seconds (tick 0 to tick 4)
        assert window == 20.0

    def test_back_price_better_counts(self):
        """Back at 4.0; tick has price 5.0 (better for backer) — should count."""
        base = datetime(2026, 1, 1, 13, 50)
        ticks = [
            _make_tick(0, base, back_price=5.0),
            _make_tick(1, base + timedelta(seconds=5), back_price=4.0),
            _make_tick(2, base + timedelta(seconds=10), back_price=5.0),
        ]
        race = _make_race(ticks)
        window = compute_opportunity_window(race, 1, 1, "back", 4.0)
        # All 3 ticks have price >= 4.0, spanning 10 seconds
        assert window == 10.0

    def test_back_price_disappears_mid_race(self):
        """Price 4.0 on ticks 0-2, drops to 3.0 on tick 3."""
        base = datetime(2026, 1, 1, 13, 50)
        ticks = [
            _make_tick(0, base, back_price=4.0),
            _make_tick(1, base + timedelta(seconds=5), back_price=4.0),
            _make_tick(2, base + timedelta(seconds=10), back_price=4.0),
            _make_tick(3, base + timedelta(seconds=15), back_price=3.0),
        ]
        race = _make_race(ticks)
        # Bet at tick 1 — backward hits tick 0, forward hits tick 2 (3.0 < 4.0 stops)
        window = compute_opportunity_window(race, 1, 1, "back", 4.0)
        assert window == 10.0  # tick 0 to tick 2

    def test_lay_price_available(self):
        """Lay at 5.0; prices <= 5.0 count as available."""
        base = datetime(2026, 1, 1, 13, 50)
        ticks = [
            _make_tick(0, base, lay_price=4.5),
            _make_tick(1, base + timedelta(seconds=5), lay_price=5.0),
            _make_tick(2, base + timedelta(seconds=10), lay_price=4.8),
            _make_tick(3, base + timedelta(seconds=15), lay_price=6.0),  # worse
        ]
        race = _make_race(ticks)
        window = compute_opportunity_window(race, 1, 1, "lay", 5.0)
        # Ticks 0-2 all have lay price <= 5.0 (10 seconds)
        assert window == 10.0

    def test_single_tick_race(self):
        """Race with only 1 tick — window is 0 seconds."""
        base = datetime(2026, 1, 1, 13, 50)
        ticks = [_make_tick(0, base, back_price=4.0)]
        race = _make_race(ticks)
        window = compute_opportunity_window(race, 0, 1, "back", 4.0)
        assert window == 0.0

    def test_tick_index_negative(self):
        """tick_index == -1 returns 0.0."""
        base = datetime(2026, 1, 1, 13, 50)
        ticks = [_make_tick(0, base, back_price=4.0)]
        race = _make_race(ticks)
        assert compute_opportunity_window(race, -1, 1, "back", 4.0) == 0.0

    def test_forward_stops_at_in_play(self):
        """Forward scan stops at in-play ticks."""
        base = datetime(2026, 1, 1, 13, 50)
        ticks = [
            _make_tick(0, base, back_price=4.0),
            _make_tick(1, base + timedelta(seconds=5), back_price=4.0),
            _make_tick(2, base + timedelta(seconds=10), back_price=4.0, in_play=True),
            _make_tick(3, base + timedelta(seconds=15), back_price=4.0, in_play=True),
        ]
        race = _make_race(ticks)
        window = compute_opportunity_window(race, 0, 1, "back", 4.0)
        # tick 0 + tick 1 pre-race (5 seconds); tick 2+ is in-play, stops
        assert window == 5.0

    def test_runner_not_found(self):
        """Runner not present in tick — window is 0."""
        base = datetime(2026, 1, 1, 13, 50)
        ticks = [
            _make_tick(0, base, back_price=4.0, selection_id=99),
            _make_tick(1, base + timedelta(seconds=5), back_price=4.0, selection_id=99),
        ]
        race = _make_race(ticks)
        # Looking for selection_id=1 but ticks have selection_id=99
        window = compute_opportunity_window(race, 0, 1, "back", 4.0)
        assert window == 0.0

    def test_empty_race(self):
        """Empty race returns 0.0."""
        race = _make_race([])
        assert compute_opportunity_window(race, 0, 1, "back", 4.0) == 0.0


# ── Bet dataclass tick_index tests ──────────────────────────────────────────


class TestBetTickIndex:

    def test_bet_default_tick_index(self):
        bet = Bet(
            selection_id=1, side=BetSide.BACK, requested_stake=10.0,
            matched_stake=10.0, average_price=4.0, market_id="m1",
        )
        assert bet.tick_index == -1

    def test_bet_tick_index_settable(self):
        bet = Bet(
            selection_id=1, side=BetSide.BACK, requested_stake=10.0,
            matched_stake=10.0, average_price=4.0, market_id="m1",
        )
        bet.tick_index = 5
        assert bet.tick_index == 5


# ── EvaluationBetRecord tests ──────────────────────────────────────────────


class TestEvalBetRecordOpportunityWindow:

    def test_default_opportunity_window(self):
        rec = EvaluationBetRecord(
            run_id="r1", date="2026-01-01", market_id="m1",
            tick_timestamp="", seconds_to_off=0.0, runner_id=1,
            runner_name="H", action="back", price=4.0, stake=10.0,
            matched_size=10.0, outcome="won", pnl=30.0,
        )
        assert rec.opportunity_window_s == 0.0

    def test_opportunity_window_set(self):
        rec = EvaluationBetRecord(
            run_id="r1", date="2026-01-01", market_id="m1",
            tick_timestamp="", seconds_to_off=0.0, runner_id=1,
            runner_name="H", action="back", price=4.0, stake=10.0,
            matched_size=10.0, outcome="won", pnl=30.0,
            opportunity_window_s=25.0,
        )
        assert rec.opportunity_window_s == 25.0


# ── EvaluationDayRecord tests ─────────────────────────────────────────────


class TestEvalDayRecordOpportunityWindow:

    def test_default_window_fields(self):
        rec = EvaluationDayRecord(
            run_id="r1", date="2026-01-01", day_pnl=10.0, bet_count=5,
            winning_bets=2, bet_precision=0.4, pnl_per_bet=2.0,
            early_picks=0, profitable=True,
        )
        assert rec.mean_opportunity_window_s == 0.0
        assert rec.median_opportunity_window_s == 0.0

    def test_window_fields_settable(self):
        rec = EvaluationDayRecord(
            run_id="r1", date="2026-01-01", day_pnl=10.0, bet_count=5,
            winning_bets=2, bet_precision=0.4, pnl_per_bet=2.0,
            early_picks=0, profitable=True,
            mean_opportunity_window_s=15.0,
            median_opportunity_window_s=12.0,
        )
        assert rec.mean_opportunity_window_s == 15.0
        assert rec.median_opportunity_window_s == 12.0


# ── ModelScore tests ───────────────────────────────────────────────────────


class TestModelScoreOpportunityWindow:

    def test_default_mean_opp_window(self):
        score = ModelScore(
            model_id="m1", win_rate=0.5, mean_daily_pnl=10.0,
            sharpe=1.0, bet_precision=0.4, pnl_per_bet=2.0,
            efficiency=0.5, composite_score=0.5, test_days=10,
            profitable_days=5,
        )
        assert score.mean_opportunity_window_s == 0.0

    def test_scoreboard_computes_mean_opp_window(self):
        """Scoreboard.compute_score populates mean_opportunity_window_s."""
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        scoreboard = Scoreboard(store=None, config=config)

        days = [
            EvaluationDayRecord(
                run_id="r1", date="2026-01-01", day_pnl=10.0, bet_count=5,
                winning_bets=2, bet_precision=0.4, pnl_per_bet=2.0,
                early_picks=0, profitable=True,
                mean_opportunity_window_s=20.0,
            ),
            EvaluationDayRecord(
                run_id="r1", date="2026-01-02", day_pnl=-5.0, bet_count=3,
                winning_bets=1, bet_precision=0.33, pnl_per_bet=-1.67,
                early_picks=0, profitable=False,
                mean_opportunity_window_s=10.0,
            ),
        ]
        score = scoreboard.compute_score(days)
        assert score is not None
        assert score.mean_opportunity_window_s == 15.0  # mean of 20 and 10


# ── Parquet round-trip tests ───────────────────────────────────────────────


class TestParquetOpportunityWindow:

    def test_parquet_roundtrip_with_window(self, tmp_path):
        store = ModelStore(
            db_path=str(tmp_path / "test.db"),
            weights_dir=str(tmp_path / "weights"),
            bet_logs_dir=str(tmp_path / "bet_logs"),
        )
        mid = store.create_model(1, "arch", "d", {})
        rid = store.create_evaluation_run(mid, "2026-01-01", ["2026-01-02"])

        rec = EvaluationBetRecord(
            run_id=rid, date="2026-01-02", market_id="m1",
            tick_timestamp="2026-01-02T13:00:00", seconds_to_off=300.0,
            runner_id=1, runner_name="Horse", action="back",
            price=4.0, stake=10.0, matched_size=10.0,
            outcome="won", pnl=30.0,
            opportunity_window_s=25.5,
        )
        store.write_bet_logs_parquet(rid, "2026-01-02", [rec])
        bets = store.get_evaluation_bets(rid)
        assert len(bets) == 1
        assert bets[0].opportunity_window_s == 25.5

    def test_parquet_backward_compat_no_window_column(self, tmp_path):
        """Old Parquet files without opportunity_window_s column still load."""
        import pandas as pd

        store = ModelStore(
            db_path=str(tmp_path / "test.db"),
            weights_dir=str(tmp_path / "weights"),
            bet_logs_dir=str(tmp_path / "bet_logs"),
        )
        mid = store.create_model(1, "arch", "d", {})
        rid = store.create_evaluation_run(mid, "2026-01-01", ["2026-01-02"])

        # Write a Parquet WITHOUT the opportunity_window_s column
        run_dir = tmp_path / "bet_logs" / rid
        run_dir.mkdir(parents=True)
        df = pd.DataFrame([{
            "run_id": rid, "date": "2026-01-02", "market_id": "m1",
            "tick_timestamp": "", "seconds_to_off": 0.0,
            "runner_id": 1, "runner_name": "Horse", "action": "back",
            "price": 4.0, "stake": 10.0, "matched_size": 10.0,
            "outcome": "won", "pnl": 30.0,
            # No opportunity_window_s column
        }])
        df.to_parquet(run_dir / "2026-01-02.parquet", index=False)

        bets = store.get_evaluation_bets(rid)
        assert len(bets) == 1
        assert bets[0].opportunity_window_s == 0.0


# ── SQLite day record round-trip ───────────────────────────────────────────


class TestSQLiteDayRecordWindow:

    def test_day_record_roundtrip_with_window(self, tmp_path):
        store = ModelStore(
            db_path=str(tmp_path / "test.db"),
            weights_dir=str(tmp_path / "weights"),
        )
        mid = store.create_model(1, "arch", "d", {})
        rid = store.create_evaluation_run(mid, "2026-01-01", ["2026-01-02"])

        rec = EvaluationDayRecord(
            run_id=rid, date="2026-01-02", day_pnl=10.0, bet_count=5,
            winning_bets=2, bet_precision=0.4, pnl_per_bet=2.0,
            early_picks=0, profitable=True,
            mean_opportunity_window_s=15.0,
            median_opportunity_window_s=12.0,
        )
        store.record_evaluation_day(rec)
        days = store.get_evaluation_days(rid)
        assert len(days) == 1
        assert days[0].mean_opportunity_window_s == 15.0
        assert days[0].median_opportunity_window_s == 12.0
