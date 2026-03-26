"""Tests: ProgressTracker — ETA calculation, formatting, edge cases."""

from __future__ import annotations

import time
from collections import deque
from unittest.mock import patch

import pytest

from training.progress_tracker import ProgressTracker, _fmt


# ─── _fmt helper ──────────────────────────────────────────────────────────────

class TestFmt:
    def test_none_returns_unknown(self):
        assert _fmt(None) == "unknown"

    def test_zero_seconds(self):
        assert _fmt(0) == "0s"

    def test_negative_seconds(self):
        assert _fmt(-5) == "0s"

    def test_seconds_only(self):
        assert _fmt(45) == "45s"

    def test_exactly_one_minute(self):
        assert _fmt(60) == "1m"

    def test_minutes_and_seconds(self):
        assert _fmt(125) == "2m 5s"

    def test_minutes_no_seconds(self):
        assert _fmt(120) == "2m"

    def test_exactly_one_hour(self):
        assert _fmt(3600) == "1h"

    def test_hours_and_minutes(self):
        assert _fmt(3900) == "1h 5m"

    def test_hours_no_minutes(self):
        assert _fmt(7200) == "2h"

    def test_large_value(self):
        # 2h 18m 35s → only hours+minutes shown
        assert _fmt(2 * 3600 + 18 * 60 + 35) == "2h 18m"


# ─── ProgressTracker ──────────────────────────────────────────────────────────

class TestProgressTrackerInit:
    def test_initial_completed_is_zero(self):
        t = ProgressTracker(total=10, label="test")
        assert t.completed == 0

    def test_initial_total(self):
        t = ProgressTracker(total=10, label="test")
        assert t.total == 10

    def test_initial_label(self):
        t = ProgressTracker(total=10, label="my label")
        assert t.label == "my label"

    def test_initial_item_eta_is_none(self):
        t = ProgressTracker(total=10, label="test")
        assert t.item_eta_seconds is None

    def test_initial_process_eta_is_none(self):
        t = ProgressTracker(total=10, label="test")
        assert t.process_eta_seconds is None

    def test_initial_pct_is_zero(self):
        t = ProgressTracker(total=10, label="test")
        assert t.pct == 0.0

    def test_zero_total_pct_is_zero(self):
        t = ProgressTracker(total=0, label="test")
        assert t.pct == 0.0


class TestProgressTrackerTick:
    def test_tick_increments_completed(self):
        t = ProgressTracker(total=10, label="test")
        t.tick()
        assert t.completed == 1

    def test_tick_populates_times(self):
        t = ProgressTracker(total=10, label="test")
        t.tick()
        assert len(t._times) == 1

    def test_tick_duration_is_non_negative(self):
        t = ProgressTracker(total=10, label="test")
        t.tick()
        assert t._times[0] >= 0.0

    def test_multiple_ticks(self):
        t = ProgressTracker(total=10, label="test")
        for _ in range(5):
            t.tick()
        assert t.completed == 5
        assert len(t._times) == 5

    def test_rolling_window_caps_times(self):
        t = ProgressTracker(total=20, label="test", rolling_window=3)
        for _ in range(10):
            t.tick()
        assert len(t._times) <= 3


class TestProgressTrackerETA:
    """ETA tests use known values injected directly into _times."""

    def _tracker_with_times(
        self, times: list[float], completed: int, total: int, window: int = 10
    ) -> ProgressTracker:
        t = ProgressTracker(total=total, label="test", rolling_window=window)
        t._times = deque(times, maxlen=window)
        t.completed = completed
        return t

    def test_item_eta_is_rolling_mean(self):
        t = self._tracker_with_times([1.0, 2.0, 3.0], completed=3, total=10)
        assert t.item_eta_seconds == pytest.approx(2.0)

    def test_process_eta_is_mean_times_remaining(self):
        # avg=2.0, remaining=7 → 14.0
        t = self._tracker_with_times([1.0, 2.0, 3.0], completed=3, total=10)
        assert t.process_eta_seconds == pytest.approx(14.0)

    def test_process_eta_zero_when_complete(self):
        t = self._tracker_with_times([1.0, 2.0], completed=10, total=10)
        assert t.process_eta_seconds == pytest.approx(0.0)

    def test_item_eta_single_sample(self):
        t = self._tracker_with_times([5.0], completed=1, total=10)
        assert t.item_eta_seconds == pytest.approx(5.0)

    def test_rolling_window_only_uses_last_n(self):
        # window=2; last two times are [4.0, 6.0], avg=5.0
        t = self._tracker_with_times([4.0, 6.0], completed=5, total=10, window=2)
        assert t.item_eta_seconds == pytest.approx(5.0)


class TestProgressTrackerPct:
    def test_pct_halfway(self):
        t = ProgressTracker(total=10, label="test")
        t._times = deque([1.0], maxlen=10)
        t.completed = 5
        assert t.pct == 50.0

    def test_pct_complete(self):
        t = ProgressTracker(total=4, label="test")
        t._times = deque([1.0], maxlen=10)
        t.completed = 4
        assert t.pct == 100.0

    def test_pct_rounds_to_one_decimal(self):
        t = ProgressTracker(total=3, label="test")
        t._times = deque([1.0], maxlen=10)
        t.completed = 1
        # 1/3 * 100 = 33.333... → 33.3
        assert t.pct == 33.3


class TestProgressTrackerToDict:
    def test_to_dict_keys_present(self):
        t = ProgressTracker(total=10, label="test")
        d = t.to_dict()
        expected_keys = {
            "label", "completed", "total", "pct",
            "item_eta_s", "process_eta_s", "item_eta_human", "process_eta_human",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_zero_completed_edge_case(self):
        t = ProgressTracker(total=10, label="test")
        d = t.to_dict()
        assert d["item_eta_s"] is None
        assert d["process_eta_s"] is None
        assert d["item_eta_human"] == "unknown"
        assert d["process_eta_human"] == "unknown"
        assert d["completed"] == 0
        assert d["pct"] == 0.0

    def test_to_dict_human_eta_formatted(self):
        t = ProgressTracker(total=10, label="test")
        t._times = deque([60.0], maxlen=10)   # 1 min per item
        t.completed = 1
        d = t.to_dict()
        # item ETA ≈ 60s → "1m"
        assert d["item_eta_human"] == "1m"
        # process ETA: 9 remaining × 60s = 540s → "9m"
        assert d["process_eta_human"] == "9m"

    def test_to_dict_label_matches(self):
        t = ProgressTracker(total=5, label="Extracting days")
        d = t.to_dict()
        assert d["label"] == "Extracting days"

    def test_to_dict_total_matches(self):
        t = ProgressTracker(total=42, label="test")
        d = t.to_dict()
        assert d["total"] == 42


class TestProgressTrackerResetTimer:
    def test_reset_timer_does_not_change_completed(self):
        t = ProgressTracker(total=10, label="test")
        t.tick()
        t.reset_timer()
        assert t.completed == 1

    def test_reset_timer_updates_last_tick(self):
        t = ProgressTracker(total=10, label="test")
        old = t._last_tick
        time.sleep(0.01)
        t.reset_timer()
        assert t._last_tick > old
