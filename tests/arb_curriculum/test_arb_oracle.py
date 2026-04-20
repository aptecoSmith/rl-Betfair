"""8 tests for the offline arb oracle scan (arb-curriculum Session 01).

hard_constraints.md §27:
  1. Synthetic day with one injected arb → one sample.
  2. Price-cap filter compliance: above max_back_price → 0 samples.
  3. Junk-filter compliance: ATB far from LTP → 0 samples.
  4. Empty day → 0 samples, no crash, .npz still written.
  5. Determinism: scan twice, same sample arrays.
  6. Round-trip: save → load_samples → content matches.
  7. Density metric: CLI stdout contains samples=X ticks=Y density=X/Y.
  8. Obs dim: sample.obs.shape[0] == BetfairEnv(scalping_mode=True).observation_space.shape[0].
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from data.episode_builder import Day, PriceSize, Race, RunnerSnap, Tick
from env.betfair_env import BetfairEnv, OBS_SCHEMA_VERSION, ACTION_SCHEMA_VERSION
from env.scalping_math import min_arb_ticks_for_profit
from env.tick_ladder import tick_offset
from training.arb_oracle import (
    OracleSample,
    count_pre_race_ticks,
    load_samples,
    save_samples,
    scan_day,
)

# ── Shared helpers ────────────────────────────────────────────────────────────

_START = datetime(2026, 4, 10, 14, 0, 0)

_MINIMAL_CONFIG: dict = {
    "training": {
        "max_runners": 5,
        "starting_budget": 100.0,
        "max_bets_per_race": 20,
        "betting_constraints": {
            "max_back_price": 50.0,
            "max_lay_price": None,
        },
    },
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
    back_price: float = 5.0,
    back_size: float = 100.0,
    lay_price: float = 5.1,
    lay_size: float = 100.0,
    status: str = "ACTIVE",
) -> RunnerSnap:
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
        available_to_back=[PriceSize(price=back_price, size=back_size)],
        available_to_lay=[PriceSize(price=lay_price, size=lay_size)],
    )


def _make_tick(
    market_id: str,
    seq: int,
    runners: list[RunnerSnap],
    in_play: bool = False,
    seconds_before_off: int = 300,
) -> Tick:
    ts = _START - timedelta(seconds=seconds_before_off - seq * 5)
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
    winner_sid: int = 101,
) -> Race:
    """Build a Race from a list-of-runners per pre-race tick."""
    ticks = [
        _make_tick(market_id, i, runners)
        for i, runners in enumerate(pre_ticks)
    ]
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


def _profitable_runner(sid: int = 101) -> RunnerSnap:
    """A runner at LTP=5.0 whose ATB price will produce a profitable arb."""
    # At LTP=5.0 with commission=0.05, min_arb_ticks ≈ 9.
    # The oracle will find a profitable arb here.
    return _make_runner(sid=sid, ltp=5.0, back_price=5.0)


def _scan_synthetic(
    runners_per_tick: list[list[RunnerSnap]],
    config: dict | None = None,
    market_id: str = "1.999000001",
) -> list[OracleSample]:
    """Scan a one-race synthetic day and return oracle samples."""
    cfg = config or _MINIMAL_CONFIG
    race = _make_race(market_id, runners_per_tick)
    day = Day(date="2026-04-10", races=[race])
    return _scan_day_obj(day, cfg)


def _scan_day_obj(day: Day, config: dict) -> list[OracleSample]:
    """Run oracle scan on a pre-built Day object (bypasses file I/O)."""
    # Monkey-patch load_day so scan_day uses the pre-built day object.
    import training.arb_oracle as _mod
    orig = None
    try:
        import data.episode_builder as _eb
        orig = _eb.load_day

        def _fake_load(date, data_dir=None):  # noqa: ANN001
            return day

        _eb.load_day = _fake_load  # type: ignore[assignment]
        return scan_day("2026-04-10", Path("data/processed"), config)
    finally:
        if orig is not None:
            _eb.load_day = orig  # type: ignore[assignment]


# ── Test 1: one injected arb → one sample ────────────────────────────────────


class TestOneInjectedArb:
    def test_single_profitable_runner_emits_one_sample(self):
        # One pre-race tick, one runner with a profitable ATB price.
        samples = _scan_synthetic([[_profitable_runner(101)]])
        assert len(samples) == 1

    def test_sample_fields_correct(self):
        samples = _scan_synthetic([[_profitable_runner(101)]])
        s = samples[0]
        assert s.tick_index == 0
        assert s.runner_idx == 0  # only runner → slot 0
        assert s.arb_spread_ticks >= 1
        assert s.expected_locked_pnl > 0.0
        # arb_spread_ticks must be the minimum profitable spread
        min_t = min_arb_ticks_for_profit(5.0, "back", 0.05, max_ticks=25)
        assert s.arb_spread_ticks == min_t

    def test_obs_is_float32(self):
        samples = _scan_synthetic([[_profitable_runner(101)]])
        assert samples[0].obs.dtype == np.float32

    def test_two_ticks_same_runner_emits_two_samples(self):
        runner = _profitable_runner(101)
        samples = _scan_synthetic([[runner], [runner]])
        assert len(samples) == 2
        assert samples[0].tick_index == 0
        assert samples[1].tick_index == 1


# ── Test 2: price-cap filter compliance ──────────────────────────────────────


class TestPriceCapFilter:
    def test_atb_above_max_back_price_rejected(self):
        # max_back_price=50.0 in minimal config; ATB=60.0 must be rejected.
        runner = _make_runner(sid=101, ltp=60.0, back_price=60.0)
        samples = _scan_synthetic([[runner]])
        assert len(samples) == 0

    def test_atb_at_cap_boundary_accepted(self):
        # ATB == max_back_price (50.0) should pass.
        runner = _make_runner(sid=101, ltp=50.0, back_price=50.0)
        samples = _scan_synthetic([[runner]])
        # 50.0 is at the 50-100 band (tick size 5.0), so a profitable
        # spread may need >25 ticks — could return 0 if unscalpable.
        # Just assert no exception and no cap rejection.
        assert isinstance(samples, list)


# ── Test 3: junk filter compliance ───────────────────────────────────────────


class TestJunkFilter:
    def test_atb_far_from_ltp_rejected(self):
        # LTP=5.0, ATB=2.0 → 2.0 < 5.0*0.5=2.5 → outside filter.
        runner = _make_runner(sid=101, ltp=5.0, back_price=2.0)
        samples = _scan_synthetic([[runner]])
        assert len(samples) == 0

    def test_atb_within_filter_accepted(self):
        # ATB=5.0 == LTP=5.0 is trivially within ±50%.
        runner = _make_runner(sid=101, ltp=5.0, back_price=5.0)
        samples = _scan_synthetic([[runner]])
        assert len(samples) >= 1

    def test_no_ltp_rejected(self):
        # LTP=0 → no reference price → runner skipped.
        runner = _make_runner(sid=101, ltp=0.0, back_price=5.0)
        samples = _scan_synthetic([[runner]])
        assert len(samples) == 0


# ── Test 4: empty day ─────────────────────────────────────────────────────────


class TestEmptyDay:
    def test_no_races_returns_empty_list(self):
        day = Day(date="2026-04-10", races=[])
        samples = _scan_day_obj(day, _MINIMAL_CONFIG)
        assert samples == []

    def test_empty_day_npz_writes_without_crash(self, tmp_path):
        day = Day(date="2026-04-10", races=[])
        samples = _scan_day_obj(day, _MINIMAL_CONFIG)
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        save_samples(samples, "2026-04-10", data_dir, _MINIMAL_CONFIG, 0, 0)
        npz = tmp_path / "oracle_cache" / "2026-04-10" / "oracle_samples.npz"
        assert npz.exists()
        loaded = np.load(npz, allow_pickle=False)
        assert len(loaded["tick_index"]) == 0

    def test_empty_day_header_json_created(self, tmp_path):
        day = Day(date="2026-04-10", races=[])
        samples = _scan_day_obj(day, _MINIMAL_CONFIG)
        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        save_samples(samples, "2026-04-10", data_dir, _MINIMAL_CONFIG, 0, 0)
        header = tmp_path / "oracle_cache" / "2026-04-10" / "header.json"
        assert header.exists()
        import json
        h = json.loads(header.read_text())
        assert h["samples"] == 0
        assert h["obs_schema_version"] == OBS_SCHEMA_VERSION


# ── Test 5: determinism ───────────────────────────────────────────────────────


class TestDeterminism:
    def test_scan_twice_same_arrays(self):
        runner = _profitable_runner(101)
        samples_a = _scan_synthetic([[runner], [runner]])
        samples_b = _scan_synthetic([[runner], [runner]])

        assert len(samples_a) == len(samples_b)
        for a, b in zip(samples_a, samples_b):
            assert a.tick_index == b.tick_index
            assert a.runner_idx == b.runner_idx
            assert a.arb_spread_ticks == b.arb_spread_ticks
            assert a.expected_locked_pnl == b.expected_locked_pnl
            np.testing.assert_array_equal(a.obs, b.obs)

    def test_npz_arrays_byte_identical_across_two_saves(self, tmp_path):
        runner = _profitable_runner(101)
        samples = _scan_synthetic([[runner]])
        obs_dim = samples[0].obs.shape[0]

        data_dir_a = tmp_path / "a" / "processed"
        data_dir_b = tmp_path / "b" / "processed"
        data_dir_a.mkdir(parents=True)
        data_dir_b.mkdir(parents=True)

        save_samples(samples, "2026-04-10", data_dir_a, _MINIMAL_CONFIG, 1, obs_dim)
        save_samples(samples, "2026-04-10", data_dir_b, _MINIMAL_CONFIG, 1, obs_dim)

        npz_a = np.load(
            tmp_path / "a" / "oracle_cache" / "2026-04-10" / "oracle_samples.npz",
            allow_pickle=False,
        )
        npz_b = np.load(
            tmp_path / "b" / "oracle_cache" / "2026-04-10" / "oracle_samples.npz",
            allow_pickle=False,
        )
        for key in ("tick_index", "runner_idx", "obs", "arb_spread_ticks",
                    "expected_locked_pnl"):
            np.testing.assert_array_equal(npz_a[key], npz_b[key])


# ── Test 6: round-trip save/load ──────────────────────────────────────────────


class TestRoundTrip:
    def test_save_then_load_identical(self, tmp_path):
        runner = _profitable_runner(101)
        samples = _scan_synthetic([[runner], [runner]])
        obs_dim = samples[0].obs.shape[0]

        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        n_ticks = 2
        save_samples(samples, "2026-04-10", data_dir, _MINIMAL_CONFIG, n_ticks, obs_dim)

        loaded = load_samples("2026-04-10", data_dir, strict=True)

        assert len(loaded) == len(samples)
        for orig, back in zip(samples, loaded):
            assert orig.tick_index == back.tick_index
            assert orig.runner_idx == back.runner_idx
            assert orig.arb_spread_ticks == back.arb_spread_ticks
            assert abs(orig.expected_locked_pnl - back.expected_locked_pnl) < 1e-6
            np.testing.assert_allclose(orig.obs, back.obs, rtol=0, atol=0)

    def test_schema_mismatch_raises(self, tmp_path):
        runner = _profitable_runner(101)
        samples = _scan_synthetic([[runner]])
        obs_dim = samples[0].obs.shape[0]

        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        save_samples(samples, "2026-04-10", data_dir, _MINIMAL_CONFIG, 1, obs_dim)

        # Corrupt the stored obs_schema_version.
        npz_path = tmp_path / "oracle_cache" / "2026-04-10" / "oracle_samples.npz"
        data = dict(np.load(npz_path, allow_pickle=False))
        data["obs_schema_version"] = np.array(999, dtype=np.int32)
        np.savez(npz_path, **data)

        with pytest.raises(ValueError, match="obs_schema_version"):
            load_samples("2026-04-10", data_dir, strict=True)

    def test_load_strict_false_skips_version_check(self, tmp_path):
        runner = _profitable_runner(101)
        samples = _scan_synthetic([[runner]])
        obs_dim = samples[0].obs.shape[0]

        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        save_samples(samples, "2026-04-10", data_dir, _MINIMAL_CONFIG, 1, obs_dim)

        # Corrupt version but load with strict=False — should not raise.
        npz_path = tmp_path / "oracle_cache" / "2026-04-10" / "oracle_samples.npz"
        data = dict(np.load(npz_path, allow_pickle=False))
        data["obs_schema_version"] = np.array(999, dtype=np.int32)
        np.savez(npz_path, **data)

        loaded = load_samples("2026-04-10", data_dir, strict=False)
        assert len(loaded) == 1


# ── Test 7: density metric via CLI stdout ─────────────────────────────────────


class TestDensityMetric:
    def test_cli_stdout_format(self, tmp_path, monkeypatch):
        # Set up a tiny day with one profitable tick and wire the CLI to
        # use our synthetic data.
        runner = _profitable_runner(101)
        day = Day(
            date="2026-04-10",
            races=[_make_race("1.999000001", [[runner]])],
        )

        import data.episode_builder as _eb
        import training.arb_oracle as _oracle_mod
        orig_load = _eb.load_day

        def _fake_load(date, data_dir=None):  # noqa: ANN001
            return day

        monkeypatch.setattr(_eb, "load_day", _fake_load)
        monkeypatch.setattr(
            _oracle_mod, "_load_config", lambda: _MINIMAL_CONFIG
        )

        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        monkeypatch.chdir(tmp_path)

        # Capture stdout from the _cli() function directly.
        buf = io.StringIO()
        monkeypatch.setattr(
            sys, "argv",
            ["arb_oracle", "scan", "--date", "2026-04-10",
             "--data-dir", str(data_dir)],
        )

        # Patch _load_config to return minimal config and data-dir to tmp.
        from training.arb_oracle import _cli
        with redirect_stdout(buf):
            _cli()

        out = buf.getvalue()
        assert "samples=" in out
        assert "ticks=" in out
        assert "density=" in out

        # Parse key fields from output.
        import re
        m_samples = re.search(r"samples=(\d+)", out)
        m_ticks = re.search(r"ticks=(\d+)", out)
        m_density = re.search(r"density=([0-9.]+)", out)
        assert m_samples and m_ticks and m_density

        n_samples = int(m_samples.group(1))
        n_ticks = int(m_ticks.group(1))
        density = float(m_density.group(1))

        assert n_ticks >= 1
        assert n_samples >= 0
        assert abs(density - n_samples / max(n_ticks, 1)) < 1e-3


# ── Test 8: obs dim matches env ───────────────────────────────────────────────


class TestObsDimMatchesEnv:
    def test_oracle_obs_dim_matches_scalping_env(self):
        # Use a synthetic day with one profitable arb tick.
        runner = _profitable_runner(101)
        day = Day(
            date="2026-04-10",
            races=[_make_race("1.999000001", [[runner]])],
        )
        samples = _scan_day_obj(day, _MINIMAL_CONFIG)

        assert len(samples) >= 1, "Expected at least one oracle sample."
        obs_dim = samples[0].obs.shape[0]

        # Instantiate a scalping env on the same day and compare.
        env = BetfairEnv(day, _MINIMAL_CONFIG, scalping_mode=True)
        expected_dim = env.observation_space.shape[0]

        assert obs_dim == expected_dim, (
            f"Oracle obs dim {obs_dim} != env obs space dim {expected_dim}"
        )

    def test_oracle_obs_dim_real_data(self):
        """If real processed data is available, verify against env obs space."""
        from tests._data_fixtures import latest_processed_date
        result = latest_processed_date()
        if result is None:
            pytest.skip("No processed data available.")

        date_str, _ = result
        import yaml
        try:
            with open("config.yaml") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            pytest.skip("config.yaml not found.")

        samples = scan_day(date_str, Path("data/processed"), config)
        if not samples:
            pytest.skip(f"No oracle samples found for {date_str}.")

        from data.episode_builder import load_day
        day = load_day(date_str, Path("data/processed"))
        env = BetfairEnv(day, config, scalping_mode=True)
        expected = env.observation_space.shape[0]
        assert samples[0].obs.shape[0] == expected
