"""Tests for load_days() progress queue emission."""

from __future__ import annotations

import queue
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from data.episode_builder import load_days


def _write_minimal_parquet(data_dir: Path, date_str: str) -> None:
    """Write a minimal valid parquet pair for a single date."""
    ticks_path = data_dir / f"{date_str}.parquet"
    runners_path = data_dir / f"{date_str}_runners.parquet"

    # Minimal tick data — just enough for load_day to not crash
    ticks_df = pd.DataFrame({
        "market_id": ["1.200000001"] * 2,
        "timestamp": pd.to_datetime(["2026-01-01 14:00:00", "2026-01-01 14:05:00"]),
        "sequence_number": [0, 1],
        "venue": ["Newmarket"] * 2,
        "market_start_time": pd.to_datetime(["2026-01-01 14:10:00"] * 2),
        "market_type": ["WIN"] * 2,
        "market_name": ["Race 1"] * 2,
        "number_of_active_runners": [3] * 2,
        "traded_volume": [10000.0] * 2,
        "in_play": [False, True],
        "snap_json": ['{"runners":[]}'] * 2,
        "winner_selection_id": [101] * 2,
        "race_status": [None] * 2,
        "temperature": [15.0] * 2,
        "precipitation": [0.0] * 2,
        "wind_speed": [5.0] * 2,
        "wind_direction": [180.0] * 2,
        "humidity": [60.0] * 2,
        "weather_code": [0] * 2,
    })
    ticks_df.to_parquet(ticks_path)

    runners_df = pd.DataFrame({
        "market_id": ["1.200000001"],
        "selection_id": [101],
        "runner_name": ["Horse A"],
        "sort_priority": [1],
        "handicap": [0.0],
    })
    runners_df.to_parquet(runners_path)


class TestLoadDaysProgress:
    def test_emits_progress_per_day(self):
        """load_days should emit one progress event per day loaded."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _write_minimal_parquet(data_dir, "2026-01-01")
            _write_minimal_parquet(data_dir, "2026-01-02")

            q: queue.Queue = queue.Queue()
            days = load_days(
                ["2026-01-01", "2026-01-02"],
                data_dir=data_dir,
                progress_queue=q,
            )

            assert len(days) == 2
            events = []
            while not q.empty():
                events.append(q.get_nowait())

            assert len(events) == 2
            assert events[0]["phase"] == "building"
            assert events[0]["process"]["completed"] == 1
            assert events[0]["process"]["total"] == 2
            assert events[1]["process"]["completed"] == 2
            assert "2026-01-02" in events[1]["detail"]

    def test_no_error_without_queue(self):
        """load_days should work fine without a progress_queue."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _write_minimal_parquet(data_dir, "2026-01-01")

            days = load_days(["2026-01-01"], data_dir=data_dir)
            assert len(days) == 1

    def test_skipped_dates_still_emit_progress(self):
        """Missing dates should still tick the progress tracker."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _write_minimal_parquet(data_dir, "2026-01-01")

            q: queue.Queue = queue.Queue()
            days = load_days(
                ["2026-01-01", "2026-01-99"],  # second date doesn't exist
                data_dir=data_dir,
                progress_queue=q,
            )

            assert len(days) == 1  # only one loaded
            events = []
            while not q.empty():
                events.append(q.get_nowait())
            assert len(events) == 2  # both dates emitted progress
