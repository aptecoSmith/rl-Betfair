"""Tests for P1c: traded_delta and mid_drift windowed features (session 21).

All tests are CPU-only.  Tests 1–7 target the pure functions in
``env/features.py``.  Tests 8–10 exercise the env integration.
"""

from __future__ import annotations

import math
from collections import deque

import pytest

from env.features import (
    betfair_tick_size,
    compute_mid_drift,
    compute_traded_delta,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _entry(ts: float, mp: float, delta: float = 0.0):
    """Build a (timestamp, microprice, vol_delta) tuple."""
    return (ts, mp, delta)


def _tick_size_mock(price: float) -> float:
    """Fixed tick size of 0.02 for all prices — simplifies arithmetic."""
    return 0.02


# ── betfair_tick_size sanity ──────────────────────────────────────────────────


class TestBetfairTickSize:
    """Sanity checks for the standard Betfair tick ladder."""

    def test_below_2(self):
        assert betfair_tick_size(1.5) == pytest.approx(0.01)

    def test_boundary_2_to_3(self):
        assert betfair_tick_size(2.0) == pytest.approx(0.02)
        assert betfair_tick_size(2.99) == pytest.approx(0.02)

    def test_boundary_3_to_4(self):
        assert betfair_tick_size(3.0) == pytest.approx(0.05)

    def test_boundary_4_to_6(self):
        assert betfair_tick_size(4.0) == pytest.approx(0.10)
        assert betfair_tick_size(5.5) == pytest.approx(0.10)

    def test_boundary_6_to_10(self):
        assert betfair_tick_size(6.0) == pytest.approx(0.20)

    def test_over_100(self):
        assert betfair_tick_size(200.0) == pytest.approx(10.0)


# ── Test 1 — First-tick values are zero ──────────────────────────────────────


class TestFirstTickZero:
    """Fresh history → both functions return exactly 0.0."""

    def test_traded_delta_empty_history(self):
        assert compute_traded_delta([], ref_mp := 3.0, 60.0, 1000.0) == 0.0

    def test_mid_drift_empty_history(self):
        assert compute_mid_drift([], 60.0, 1000.0, _tick_size_mock) == 0.0

    def test_traded_delta_single_entry_zero_delta(self):
        """One entry, vol_delta=0 → result is 0.0 (first tick behaviour)."""
        hist = [_entry(1000.0, 3.0, 0.0)]
        result = compute_traded_delta(hist, 3.0, 60.0, 1000.0)
        assert result == 0.0

    def test_mid_drift_single_entry_no_baseline(self):
        """One entry in history — no entry at-or-before cutoff → 0.0."""
        # now_ts=1000, cutoff=940; entry is at ts=1000 (after cutoff)
        hist = [_entry(1000.0, 3.0, 0.0)]
        result = compute_mid_drift(hist, 60.0, 1000.0, _tick_size_mock)
        assert result == 0.0


# ── Test 2 — Traded delta sign: backers hitting lays ────────────────────────


class TestTradedDeltaPositive:
    """Volume at mp ≤ reference → positive result."""

    def test_volume_at_reference_price_counts_positive(self):
        # reference = 3.0; entry mp = 3.0 (equal → counts positive)
        hist = [_entry(950.0, 3.0, 100.0)]
        result = compute_traded_delta(hist, 3.0, 60.0, 1000.0)
        assert result == pytest.approx(100.0)

    def test_volume_below_reference_counts_positive(self):
        # reference = 3.5; entry mp = 3.0 (below reference → positive)
        hist = [_entry(950.0, 3.0, 200.0)]
        result = compute_traded_delta(hist, 3.5, 60.0, 1000.0)
        assert result == pytest.approx(200.0)

    def test_multiple_entries_all_positive(self):
        hist = [
            _entry(940.0, 3.0, 50.0),
            _entry(960.0, 3.1, 80.0),
        ]
        result = compute_traded_delta(hist, 3.5, 60.0, 1000.0)
        assert result == pytest.approx(130.0)


# ── Test 3 — Traded delta sign: layers hitting backs ────────────────────────


class TestTradedDeltaNegative:
    """Volume at mp > reference → negative result."""

    def test_volume_above_reference_counts_negative(self):
        # reference = 3.0; entry mp = 3.5 (above → negative)
        hist = [_entry(950.0, 3.5, 150.0)]
        result = compute_traded_delta(hist, 3.0, 60.0, 1000.0)
        assert result == pytest.approx(-150.0)

    def test_mixed_entries_net_result(self):
        # Two entries: 100 backing (mp ≤ ref), 200 laying (mp > ref)
        hist = [
            _entry(940.0, 3.0, 100.0),  # mp == ref → positive
            _entry(960.0, 3.5, 200.0),  # mp > ref  → negative
        ]
        result = compute_traded_delta(hist, 3.0, 60.0, 1000.0)
        # 100 - 200 = -100
        assert result == pytest.approx(-100.0)


# ── Test 4 — Traded delta window edge ────────────────────────────────────────


class TestTradedDeltaWindowEdge:
    """Events at the exact window boundary are handled correctly."""

    def test_entry_just_inside_window_contributes(self):
        # window_seconds=60; now_ts=1000; cutoff=940
        # entry at ts=940.001 → ts >= cutoff → included
        hist = [_entry(940.001, 3.0, 99.0)]
        result = compute_traded_delta(hist, 3.0, 60.0, 1000.0)
        assert result == pytest.approx(99.0)

    def test_entry_just_outside_window_excluded(self):
        # entry at ts=939.999 → ts < cutoff → excluded
        hist = [_entry(939.999, 3.0, 99.0)]
        result = compute_traded_delta(hist, 3.0, 60.0, 1000.0)
        assert result == 0.0

    def test_entry_exactly_at_cutoff_included(self):
        # ts == cutoff → ts < cutoff is False → included
        hist = [_entry(940.0, 3.0, 55.0)]
        result = compute_traded_delta(hist, 3.0, 60.0, 1000.0)
        assert result == pytest.approx(55.0)


# ── Test 5 — Mid drift: rising microprice ────────────────────────────────────


class TestMidDriftRising:
    """Rising microprice → positive tick delta."""

    def test_rising_price_positive_drift(self):
        # baseline at ts=940 (≤ cutoff=940): mp=3.0
        # current (last entry): mp=3.1
        # diff = 0.1; tick_size=0.02 → 5 ticks
        hist = [
            _entry(940.0, 3.0, 0.0),
            _entry(1000.0, 3.1, 0.0),
        ]
        result = compute_mid_drift(hist, 60.0, 1000.0, _tick_size_mock)
        assert result == pytest.approx(5.0)

    def test_large_rise_many_ticks(self):
        # cutoff = now_ts - window_seconds = 1000 - 90 = 910
        # entry at ts=900 ≤ cutoff → IS baseline
        hist = [
            _entry(900.0, 2.0, 0.0),
            _entry(1000.0, 2.4, 0.0),
        ]
        # diff = 0.4; tick_size = 0.02 → 20 ticks
        result = compute_mid_drift(hist, 90.0, 1000.0, _tick_size_mock)
        assert result == pytest.approx(20.0)


# ── Test 6 — Mid drift: falling microprice ───────────────────────────────────


class TestMidDriftFalling:
    """Falling microprice → negative tick delta."""

    def test_falling_price_negative_drift(self):
        hist = [
            _entry(930.0, 4.0, 0.0),
            _entry(1000.0, 3.5, 0.0),
        ]
        # cutoff = 940; baseline ts=930 ≤ cutoff → baseline mp=4.0
        # diff = 3.5 - 4.0 = -0.5; tick_size=0.02 → -25 ticks
        result = compute_mid_drift(hist, 60.0, 1000.0, _tick_size_mock)
        assert result == pytest.approx(-25.0)


# ── Test 7 — Mid drift window edge ───────────────────────────────────────────


class TestMidDriftWindowEdge:
    """Baseline selection respects the cutoff (ts ≤ now_ts - window_seconds)."""

    def test_entry_just_outside_window_is_baseline(self):
        # cutoff = 940; entry at ts=939.999 → ts ≤ cutoff → IS baseline
        hist = [
            _entry(939.999, 3.0, 0.0),
            _entry(1000.0, 3.1, 0.0),
        ]
        result = compute_mid_drift(hist, 60.0, 1000.0, _tick_size_mock)
        # diff = 0.1 / 0.02 = 5 ticks
        assert result == pytest.approx(5.0)

    def test_entry_just_inside_window_not_baseline(self):
        # cutoff = 940; entry at ts=940.001 → ts > cutoff → NOT baseline
        # → no valid baseline → 0.0
        hist = [
            _entry(940.001, 3.0, 0.0),
            _entry(1000.0, 3.1, 0.0),
        ]
        result = compute_mid_drift(hist, 60.0, 1000.0, _tick_size_mock)
        assert result == 0.0

    def test_multiple_entries_latest_at_or_before_cutoff_is_baseline(self):
        # Two entries before cutoff; the LATER one (ts=935) should be baseline.
        hist = [
            _entry(920.0, 2.8, 0.0),
            _entry(935.0, 3.0, 0.0),  # <- latest at-or-before cutoff=940
            _entry(950.0, 3.1, 0.0),
            _entry(1000.0, 3.2, 0.0),
        ]
        # baseline mp = 3.0; current mp = 3.2; diff = 0.2; ticks = 10
        result = compute_mid_drift(hist, 60.0, 1000.0, _tick_size_mock)
        assert result == pytest.approx(10.0)


# ── Tests 8–10 — Env integration ─────────────────────────────────────────────


def _make_windowed_day(n_ticks: int = 5):
    """Build a minimal Day whose ticks have incrementing timestamps and volume.

    Ticks are spread 30 seconds apart so that after 3+ ticks we exceed a
    60-second window and mid_drift can find a baseline.
    """
    from datetime import datetime, timedelta, timezone

    from data.episode_builder import Day, PriceSize, Race, RunnerMeta, RunnerSnap, Tick

    def _ps(price: float, size: float) -> PriceSize:
        return PriceSize(price=price, size=size)

    def _runner(sid: int, ltp: float, total_matched: float) -> RunnerSnap:
        return RunnerSnap(
            selection_id=sid,
            status="ACTIVE",
            last_traded_price=ltp,
            total_matched=total_matched,
            starting_price_near=0.0,
            starting_price_far=0.0,
            adjustment_factor=None,
            bsp=None,
            sort_priority=1,
            removal_date=None,
            available_to_back=[_ps(ltp - 0.1, 200.0)],
            available_to_lay=[_ps(ltp + 0.1, 200.0)],
        )

    def _meta(sid: int) -> RunnerMeta:
        return RunnerMeta(
            selection_id=sid, runner_name=f"Runner {sid}",
            sort_priority="1", handicap="", sire_name="", dam_name="",
            damsire_name="", bred="", official_rating="", adjusted_rating="",
            age="4", sex_type="G", colour_type="B", weight_value="",
            weight_units="", jockey_name="", jockey_claim="", trainer_name="",
            owner_name="", stall_draw="", cloth_number="", form="",
            days_since_last_run="", wearing="", forecastprice_numerator="",
            forecastprice_denominator="",
        )

    start = datetime(2026, 4, 8, 14, 30, tzinfo=timezone.utc)
    base_ts = datetime(2026, 4, 8, 14, 28, 0, tzinfo=timezone.utc)
    ticks = []
    for i in range(n_ticks):
        ts = base_ts + timedelta(seconds=i * 30)
        in_play = (i == n_ticks - 1)
        winner = 1001 if in_play else None
        status_1001 = "WINNER" if in_play else "ACTIVE"
        status_1002 = "LOSER" if in_play else "ACTIVE"
        ticks.append(Tick(
            market_id="1.99999",
            timestamp=ts,
            sequence_number=i + 1,
            venue="Ascot",
            market_start_time=start,
            number_of_active_runners=2,
            traded_volume=float(500 + i * 200),
            in_play=in_play,
            winner_selection_id=winner,
            race_status="off" if in_play else None,
            temperature=None, precipitation=None, wind_speed=None,
            wind_direction=None, humidity=None, weather_code=None,
            runners=[
                RunnerSnap(
                    selection_id=1001, status=status_1001,
                    last_traded_price=3.0,
                    total_matched=float(300 + i * 50),
                    starting_price_near=0.0, starting_price_far=0.0,
                    adjustment_factor=None,
                    bsp=3.1 if in_play else None,
                    sort_priority=1, removal_date=None,
                    available_to_back=[] if in_play else [_ps(2.9, 200.0)],
                    available_to_lay=[] if in_play else [_ps(3.1, 200.0)],
                ),
                RunnerSnap(
                    selection_id=1002, status=status_1002,
                    last_traded_price=5.0,
                    total_matched=float(200 + i * 30),
                    starting_price_near=0.0, starting_price_far=0.0,
                    adjustment_factor=None,
                    bsp=5.5 if in_play else None,
                    sort_priority=2, removal_date=None,
                    available_to_back=[] if in_play else [_ps(4.9, 100.0)],
                    available_to_lay=[] if in_play else [_ps(5.1, 100.0)],
                ),
            ],
        ))

    race = Race(
        market_id="1.99999",
        venue="Ascot",
        market_start_time=start,
        winner_selection_id=1001,
        winning_selection_ids={1001},
        ticks=ticks,
        runner_metadata={1001: _meta(1001), 1002: _meta(1002)},
        market_type="WIN",
        market_name="Test race",
        n_runners=2,
    )
    return Day(date="2026-04-08", races=[race])


def _windowed_config(traded_delta_window_s: float = 60.0, mid_drift_window_s: float = 60.0) -> dict:
    return {
        "training": {
            "max_runners": 4,
            "starting_budget": 100.0,
            "max_bets_per_race": 5,
            "require_gpu": False,
            "betting_constraints": {
                "max_back_price": 100.0,
                "max_lay_price": None,
                "min_seconds_before_off": 0,
            },
        },
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 60,
            "terminal_bonus_weight": 1.0,
            "efficiency_penalty": 0.01,
            "precision_bonus": 1.0,
            "commission": 0.05,
            "drawdown_shaping_weight": 0.0,
        },
        "features": {
            "obi_top_n": 3,
            "microprice_top_n": 3,
            "traded_delta_window_s": traded_delta_window_s,
            "mid_drift_window_s": mid_drift_window_s,
        },
    }


class TestEnvWindowed:
    """Test 8: Env smoke — features appear in debug_features and are sensible."""

    def test_windowed_features_in_debug_features(self):
        """Both windowed features appear in debug_features throughout the race."""
        from env.betfair_env import BetfairEnv

        # 6 ticks * 30s apart gives a 150s race — well beyond the 60s window.
        day = _make_windowed_day(n_ticks=6)
        env = BetfairEnv(day, _windowed_config())
        _, info = env.reset()

        first_step_done = False
        found_nonzero = False

        while True:
            action = env.action_space.sample() * 0.0
            _, _, terminated, _, info = env.step(action)
            debug = info.get("debug_features", {})
            for sid, feats in debug.items():
                assert "traded_delta" in feats, f"traded_delta missing for runner {sid}"
                assert "mid_drift" in feats, f"mid_drift missing for runner {sid}"

                if not first_step_done:
                    # First step: history has only one tick (vol_delta=0),
                    # so traded_delta=0 and mid_drift=0.
                    assert feats["traded_delta"] == pytest.approx(0.0), (
                        f"Expected traded_delta=0 on first step, got {feats['traded_delta']}"
                    )
                    assert feats["mid_drift"] == pytest.approx(0.0), (
                        f"Expected mid_drift=0 on first step, got {feats['mid_drift']}"
                    )

            first_step_done = True

            # Check for non-zero on mid-race ticks (after window fills)
            for sid, feats in debug.items():
                if feats.get("traded_delta", 0.0) != 0.0 or feats.get("mid_drift", 0.0) != 0.0:
                    found_nonzero = True

            if terminated:
                break

        assert found_nonzero, (
            "Neither traded_delta nor mid_drift was ever non-zero mid-race. "
            "Window logic is broken."
        )


class TestDeterminism:
    """Test 9: Same race replayed twice → byte-identical feature values."""

    def test_windowed_features_deterministic(self):
        from env.betfair_env import BetfairEnv

        day = _make_windowed_day(n_ticks=5)
        cfg = _windowed_config()

        def _collect():
            env = BetfairEnv(day, cfg)
            env.reset()
            results = []
            while True:
                action = env.action_space.sample() * 0.0
                _, _, terminated, _, info = env.step(action)
                debug = info.get("debug_features", {})
                for sid in sorted(debug):
                    feats = debug[sid]
                    results.append((sid, feats.get("traded_delta"), feats.get("mid_drift")))
                if terminated:
                    break
            return results

        run1 = _collect()
        run2 = _collect()

        assert run1, "No features collected"
        assert run1 == run2, f"Non-determinism detected:\nrun1={run1}\nrun2={run2}"


class TestHistoryBufferBounded:
    """Test 10: Per-runner deque length is ≤ the configured bound."""

    def test_deque_length_bounded(self):
        from env.betfair_env import BetfairEnv

        # Small window so bound is 2*30+20=80 entries
        day = _make_windowed_day(n_ticks=5)
        cfg = _windowed_config(traded_delta_window_s=30.0, mid_drift_window_s=30.0)
        env = BetfairEnv(day, cfg)
        env.reset()

        while True:
            action = env.action_space.sample() * 0.0
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                break

        expected_maxlen = env._windowed_maxlen
        for sid, buf in env._windowed_history.items():
            assert len(buf) <= expected_maxlen, (
                f"Runner {sid} deque length {len(buf)} exceeds bound {expected_maxlen}"
            )
            # maxlen is enforced on the deque itself
            assert buf.maxlen == expected_maxlen, (
                f"Runner {sid} deque.maxlen={buf.maxlen}, expected {expected_maxlen}"
            )


# ── Schema version test ───────────────────────────────────────────────────────


class TestSchemaVersionRefusesPreP1c:
    """Loader refuses P1b checkpoints (version 3) after P1c bumps to 4."""

    def test_refuses_p1b_checkpoint(self):
        from env.betfair_env import OBS_SCHEMA_VERSION, validate_obs_schema

        # P1c bumped to 4; later bumps keep this ≥ 4. The intent of this
        # test is that a P1b checkpoint (version 3) is refused, regardless
        # of how many subsequent schema bumps have landed.
        assert OBS_SCHEMA_VERSION >= 4, (
            f"Expected OBS_SCHEMA_VERSION>=4 after P1c bump, got {OBS_SCHEMA_VERSION}"
        )
        p1b_checkpoint = {"obs_schema_version": 3, "weights": {}}
        with pytest.raises(ValueError, match="obs_schema_version"):
            validate_obs_schema(p1b_checkpoint)

    def test_accepts_current_version(self):
        from env.betfair_env import OBS_SCHEMA_VERSION, validate_obs_schema

        valid = {"obs_schema_version": OBS_SCHEMA_VERSION, "weights": {}}
        validate_obs_schema(valid)  # must not raise
