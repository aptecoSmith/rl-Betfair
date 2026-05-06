"""Regression tests for the offline direction-label scan
(``training_v2.direction_label_scan``) — phase-13 S02.

Mirrors the v2 oracle test pattern (``tests/test_v2_oracle.py``).
Synthetic days are built directly from ``data.episode_builder``
dataclasses so the scan can be exercised without parquet I/O.

Tests:

1. ``test_scan_day_emits_one_row_per_priceable_runner_tick``
2. ``test_label_back_positive_when_ltp_drops_to_threshold``
3. ``test_label_back_zero_when_ltp_never_drops_to_threshold``
4. ``test_label_lay_symmetric_to_label_back``
5. ``test_priceability_at_open_tick_required``
6. ``test_horizon_ticks_caps_search``
7. ``test_force_close_horizon_bounds_search``
8. ``test_in_play_truncates_search``
9. ``test_determinism``
10. ``test_round_trip_save_load_strict``
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from data.episode_builder import Day, PriceSize, Race, RunnerSnap, Tick
from env.tick_ladder import tick_offset
from training_v2.direction_label_scan import (
    LABEL_VERSION,
    _cache_dir,
    _cache_stem,
    load_labels,
    save_labels,
    scan_day,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

_START = datetime(2026, 4, 10, 14, 0, 0)

_MINIMAL_CONFIG: dict = {
    "training": {
        "max_runners": 5,
        "starting_budget": 100.0,
        "max_bets_per_race": 20,
        "scalping_mode": True,
        "betting_constraints": {
            "max_back_price": 50.0,
            "max_lay_price": None,
        },
    },
    "actions": {"force_aggressive": True},
    "reward": {
        "early_pick_bonus_min": 1.2,
        "early_pick_bonus_max": 1.5,
        "early_pick_min_seconds": 300,
        "efficiency_penalty": 0.01,
        "commission": 0.05,
    },
}


def _make_runner(
    sid: int = 101,
    ltp: float = 5.0,
    back_price: float | None = None,
    back_size: float = 100.0,
    lay_price: float | None = None,
    lay_size: float = 100.0,
    status: str = "ACTIVE",
) -> RunnerSnap:
    bp = back_price if back_price is not None else ltp
    lp = lay_price if lay_price is not None else ltp + 0.1
    return RunnerSnap(
        selection_id=sid,
        status=status,
        last_traded_price=ltp,
        total_matched=1000.0,
        starting_price_near=0.0,
        starting_price_far=0.0,
        adjustment_factor=None,
        bsp=None,
        sort_priority=1,
        removal_date=None,
        available_to_back=[PriceSize(price=bp, size=back_size)],
        available_to_lay=[PriceSize(price=lp, size=lay_size)],
    )


def _make_tick(
    market_id: str,
    seq: int,
    runners: list[RunnerSnap],
    in_play: bool = False,
    seconds_before_off: int = 300,
    tick_spacing_seconds: int = 5,
) -> Tick:
    ts = _START - timedelta(
        seconds=seconds_before_off - seq * tick_spacing_seconds,
    )
    return Tick(
        market_id=market_id,
        timestamp=ts,
        sequence_number=seq,
        venue="Newmarket",
        market_start_time=_START,
        number_of_active_runners=len(runners),
        traded_volume=5000.0,
        in_play=in_play,
        winner_selection_id=101,
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
    market_id: str,
    pre_ticks: list[list[RunnerSnap]],
    in_play_after: int | None = None,
    seconds_before_off_at_t0: int = 300,
    tick_spacing_seconds: int = 5,
    winner_sid: int = 101,
) -> Race:
    """Build a Race. ``in_play_after`` is the tick index from which
    onwards ticks are flagged in_play."""
    ticks = []
    for i, runners in enumerate(pre_ticks):
        ip = (in_play_after is not None and i >= in_play_after)
        ticks.append(
            _make_tick(
                market_id,
                i,
                runners,
                in_play=ip,
                seconds_before_off=seconds_before_off_at_t0,
                tick_spacing_seconds=tick_spacing_seconds,
            )
        )
    from tests.test_betfair_env import _make_runner_meta  # type: ignore[attr-defined]
    all_sids = {r.selection_id for t in ticks for r in t.runners}
    meta = {sid: _make_runner_meta(sid) for sid in all_sids}
    return Race(
        market_id=market_id,
        venue="Newmarket",
        market_start_time=_START,
        winner_selection_id=winner_sid,
        ticks=ticks,
        runner_metadata=meta,
        winning_selection_ids={winner_sid},
    )


def _scan_synthetic(
    runners_per_tick: list[list[RunnerSnap]],
    *,
    horizon_ticks: int = 60,
    threshold_ticks: int = 5,
    force_close_seconds: float = 0.0,
    in_play_after: int | None = None,
    seconds_before_off_at_t0: int = 300,
    tick_spacing_seconds: int = 5,
    config: dict | None = None,
):
    cfg = config or _MINIMAL_CONFIG
    race = _make_race(
        "1.999000001",
        runners_per_tick,
        in_play_after=in_play_after,
        seconds_before_off_at_t0=seconds_before_off_at_t0,
        tick_spacing_seconds=tick_spacing_seconds,
    )
    day = Day(date="2026-04-10", races=[race])
    import data.episode_builder as _eb
    orig = _eb.load_day

    def _fake_load(date, data_dir=None):  # noqa: ANN001
        return day

    _eb.load_day = _fake_load  # type: ignore[assignment]
    try:
        return scan_day(
            "2026-04-10",
            Path("data/processed"),
            cfg,
            direction_horizon_ticks=horizon_ticks,
            direction_threshold_ticks=threshold_ticks,
            force_close_before_off_seconds=force_close_seconds,
        )
    finally:
        _eb.load_day = orig  # type: ignore[assignment]


# ── Test 1: shape ─────────────────────────────────────────────────────────────


class TestScanDayEmitsOneRowPerPriceableRunnerTick:
    def test_two_pre_race_ticks_one_runner_emits_two_rows(self):
        runner = _make_runner(sid=101, ltp=5.0)
        labels = _scan_synthetic([[runner], [runner]])
        # Both pre-race ticks are priceable for the runner.
        assert len(labels) == 2
        assert labels[0].tick_index == 0
        assert labels[0].runner_idx == 0
        assert labels[1].tick_index == 1
        assert labels[1].runner_idx == 0


# ── Test 2: positive label_back ───────────────────────────────────────────────


class TestLabelBackPositiveWhenLtpDropsToThreshold:
    def test_ltp_drops_by_threshold_yields_label_back_one(self):
        ltp_T = 5.0
        threshold = 5
        target = tick_offset(ltp_T, threshold, -1)
        # Tick 0: LTP=5.0. Tick 1: LTP=midway (not enough).
        # Tick 2: LTP=target → favourable cross.
        runners = [
            [_make_runner(sid=101, ltp=ltp_T)],
            [_make_runner(sid=101, ltp=tick_offset(ltp_T, 2, -1))],
            [_make_runner(sid=101, ltp=target)],
        ]
        labels = _scan_synthetic(
            runners, threshold_ticks=threshold, horizon_ticks=10,
        )
        # Row at tick 0 should have label_back == 1.
        row0 = next(r for r in labels if r.tick_index == 0)
        assert row0.label_back == 1.0
        assert row0.first_back_fav_tick == 2


# ── Test 3: zero label_back ───────────────────────────────────────────────────


class TestLabelBackZeroWhenLtpNeverDropsToThreshold:
    def test_flat_ltp_yields_label_back_zero(self):
        ltp_T = 5.0
        # All flat — never crosses.
        runners = [
            [_make_runner(sid=101, ltp=ltp_T)],
            [_make_runner(sid=101, ltp=ltp_T)],
            [_make_runner(sid=101, ltp=ltp_T)],
        ]
        labels = _scan_synthetic(
            runners, threshold_ticks=5, horizon_ticks=10,
        )
        row0 = next(r for r in labels if r.tick_index == 0)
        assert row0.label_back == 0.0
        assert row0.first_back_fav_tick == -1


# ── Test 4: symmetric label_lay ───────────────────────────────────────────────


class TestLabelLaySymmetricToLabelBack:
    def test_ltp_rises_by_threshold_yields_label_lay_one(self):
        ltp_T = 5.0
        threshold = 5
        target = tick_offset(ltp_T, threshold, +1)
        runners = [
            [_make_runner(sid=101, ltp=ltp_T)],
            [_make_runner(sid=101, ltp=tick_offset(ltp_T, 2, +1))],
            [_make_runner(sid=101, ltp=target)],
        ]
        labels = _scan_synthetic(
            runners, threshold_ticks=threshold, horizon_ticks=10,
        )
        row0 = next(r for r in labels if r.tick_index == 0)
        assert row0.label_lay == 1.0
        assert row0.first_lay_fav_tick == 2
        # Symmetric: label_back stays 0 for a rising series.
        assert row0.label_back == 0.0


# ── Test 5: priceability ──────────────────────────────────────────────────────


class TestPriceabilityAtOpenTickRequired:
    def test_atb_failing_junk_filter_skips_row(self):
        # LTP=5.0 but ATB=15.0 (3x LTP, outside ±50% junk filter) AND
        # ATL=15.5 — both sides fail. No row.
        ltp_T = 5.0
        runner_t0 = _make_runner(
            sid=101, ltp=ltp_T, back_price=15.0, lay_price=15.5,
        )
        runner_t1 = _make_runner(sid=101, ltp=ltp_T)
        labels = _scan_synthetic(
            [[runner_t0], [runner_t1]],
            threshold_ticks=5, horizon_ticks=10,
        )
        # Row at tick 0 should be absent (both sides un-priceable);
        # tick 1 still emits.
        assert all(r.tick_index != 0 for r in labels)
        assert any(r.tick_index == 1 for r in labels)


# ── Test 6: horizon caps search ───────────────────────────────────────────────


class TestHorizonTicksCapsSearch:
    def test_drop_after_horizon_does_not_set_label(self):
        ltp_T = 5.0
        threshold = 5
        target = tick_offset(ltp_T, threshold, -1)
        # Drop happens at tick 5; horizon = 2 → search ends at tick 2.
        runners = [
            [_make_runner(sid=101, ltp=ltp_T)],
            [_make_runner(sid=101, ltp=ltp_T)],
            [_make_runner(sid=101, ltp=ltp_T)],
            [_make_runner(sid=101, ltp=ltp_T)],
            [_make_runner(sid=101, ltp=ltp_T)],
            [_make_runner(sid=101, ltp=target)],
        ]
        labels = _scan_synthetic(
            runners, threshold_ticks=threshold, horizon_ticks=2,
        )
        row0 = next(r for r in labels if r.tick_index == 0)
        assert row0.label_back == 0.0


# ── Test 7: force-close bounds ────────────────────────────────────────────────


class TestForceCloseHorizonBoundsSearch:
    def test_drop_inside_force_close_window_excluded(self):
        # Each tick is 5s apart; tick 0 is 300s before off. Tick 5 is
        # 275s before off. force_close_seconds=290 → ticks at <=290s
        # before off are inside the cutoff. So scan stops at the LAST
        # tick whose time_to_off > 290s, which is tick 1 (295s).
        ltp_T = 5.0
        threshold = 5
        target = tick_offset(ltp_T, threshold, -1)
        runners = [
            [_make_runner(sid=101, ltp=ltp_T)],
            [_make_runner(sid=101, ltp=ltp_T)],
            [_make_runner(sid=101, ltp=ltp_T)],
            [_make_runner(sid=101, ltp=ltp_T)],
            [_make_runner(sid=101, ltp=target)],   # tick 4: 280s before off
        ]
        labels = _scan_synthetic(
            runners,
            threshold_ticks=threshold,
            horizon_ticks=10,
            force_close_seconds=290.0,
            seconds_before_off_at_t0=300,
            tick_spacing_seconds=5,
        )
        row0 = next(r for r in labels if r.tick_index == 0)
        assert row0.label_back == 0.0


# ── Test 8: in-play truncates ─────────────────────────────────────────────────


class TestInPlayTruncatesSearch:
    def test_drop_after_in_play_does_not_set_label(self):
        ltp_T = 5.0
        threshold = 5
        target = tick_offset(ltp_T, threshold, -1)
        # in_play_after=3 → ticks 3, 4, 5 are in_play. Drop at tick 4
        # therefore not seen. tick 0 + 1 + 2 are pre-race.
        runners = [
            [_make_runner(sid=101, ltp=ltp_T)],
            [_make_runner(sid=101, ltp=ltp_T)],
            [_make_runner(sid=101, ltp=ltp_T)],
            [_make_runner(sid=101, ltp=target)],
            [_make_runner(sid=101, ltp=target)],
            [_make_runner(sid=101, ltp=target)],
        ]
        labels = _scan_synthetic(
            runners,
            threshold_ticks=threshold,
            horizon_ticks=10,
            in_play_after=3,
        )
        row0 = next(r for r in labels if r.tick_index == 0)
        assert row0.label_back == 0.0


# ── Test 9: determinism ───────────────────────────────────────────────────────


class TestDeterminism:
    def test_two_scans_byte_identical(self):
        runner = _make_runner(sid=101, ltp=5.0)
        a = _scan_synthetic([[runner], [runner], [runner]])
        b = _scan_synthetic([[runner], [runner], [runner]])
        assert len(a) == len(b)
        for ra, rb in zip(a, b):
            assert ra.tick_index == rb.tick_index
            assert ra.runner_idx == rb.runner_idx
            assert ra.label_back == rb.label_back
            assert ra.label_lay == rb.label_lay
            assert ra.ltp_at_open == rb.ltp_at_open
            assert ra.threshold_back == rb.threshold_back
            assert ra.threshold_lay == rb.threshold_lay
            assert ra.first_back_fav_tick == rb.first_back_fav_tick
            assert ra.first_lay_fav_tick == rb.first_lay_fav_tick


# ── Test 10: round-trip + strict header ───────────────────────────────────────


class TestRoundTripSaveLoadStrict:
    def test_save_then_load_returns_same_rows(self, tmp_path):
        runner = _make_runner(sid=101, ltp=5.0)
        labels = _scan_synthetic([[runner], [runner]])
        assert len(labels) == 2

        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        save_labels(
            labels,
            "2026-04-10",
            data_dir,
            _MINIMAL_CONFIG,
            direction_horizon_ticks=60,
            direction_threshold_ticks=5,
            force_close_before_off_seconds=0.0,
            total_pre_race_ticks=2,
        )

        loaded = load_labels(
            "2026-04-10",
            data_dir,
            direction_horizon_ticks=60,
            direction_threshold_ticks=5,
            force_close_before_off_seconds=0.0,
            strict=True,
        )
        assert len(loaded) == len(labels)
        for orig, back in zip(labels, loaded):
            assert orig.tick_index == back.tick_index
            assert orig.runner_idx == back.runner_idx
            assert orig.label_back == back.label_back
            assert orig.label_lay == back.label_lay

    def test_threshold_mismatch_raises(self, tmp_path):
        runner = _make_runner(sid=101, ltp=5.0)
        labels = _scan_synthetic([[runner]], threshold_ticks=5)
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        save_labels(
            labels,
            "2026-04-10",
            data_dir,
            _MINIMAL_CONFIG,
            direction_horizon_ticks=60,
            direction_threshold_ticks=5,
            force_close_before_off_seconds=0.0,
            total_pre_race_ticks=1,
        )
        with pytest.raises(FileNotFoundError):
            # Different stem → cache file does not exist.
            load_labels(
                "2026-04-10",
                data_dir,
                direction_horizon_ticks=60,
                direction_threshold_ticks=8,
                force_close_before_off_seconds=0.0,
                strict=True,
            )

    def test_horizon_mismatch_in_header_raises_in_strict(self, tmp_path):
        runner = _make_runner(sid=101, ltp=5.0)
        labels = _scan_synthetic([[runner]])
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        save_labels(
            labels,
            "2026-04-10",
            data_dir,
            _MINIMAL_CONFIG,
            direction_horizon_ticks=60,
            direction_threshold_ticks=5,
            force_close_before_off_seconds=0.0,
            total_pre_race_ticks=1,
        )
        # Manually corrupt the header so the file path resolves but the
        # header value disagrees.
        import json
        cd = _cache_dir(data_dir, "2026-04-10")
        stem = _cache_stem(60, 5, 0.0)
        header_path = cd / f"{stem}_header.json"
        h = json.loads(header_path.read_text())
        h["direction_horizon_ticks"] = 999
        header_path.write_text(json.dumps(h))
        with pytest.raises(ValueError, match="horizon"):
            load_labels(
                "2026-04-10",
                data_dir,
                direction_horizon_ticks=60,
                direction_threshold_ticks=5,
                force_close_before_off_seconds=0.0,
                strict=True,
            )

    def test_label_version_recorded(self, tmp_path):
        runner = _make_runner(sid=101, ltp=5.0)
        labels = _scan_synthetic([[runner]])
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        save_labels(
            labels,
            "2026-04-10",
            data_dir,
            _MINIMAL_CONFIG,
            direction_horizon_ticks=60,
            direction_threshold_ticks=5,
            force_close_before_off_seconds=0.0,
            total_pre_race_ticks=1,
        )
        import json
        cd = _cache_dir(data_dir, "2026-04-10")
        stem = _cache_stem(60, 5, 0.0)
        header = json.loads((cd / f"{stem}_header.json").read_text())
        assert header["label_version"] == LABEL_VERSION
