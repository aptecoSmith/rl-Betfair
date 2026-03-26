"""Tests for data/episode_builder.py — Parquet → Episode objects."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from data.episode_builder import (
    Day,
    PriceSize,
    Race,
    RunnerMeta,
    RunnerSnap,
    Tick,
    _build_day,
    _build_runner_meta,
    _opt_float,
    _opt_int,
    load_day,
    load_days,
    parse_snap_json,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_snap_json(
    runners: list[dict] | None = None,
    n_runners: int = 3,
) -> str:
    """Build a valid snap_json string for testing."""
    if runners is not None:
        return json.dumps({"Runners": runners})
    # Default: n_runners with distinct prices
    result = []
    for i in range(n_runners):
        result.append({
            "SelectionId": 1000 + i,
            "Status": "ACTIVE",
            "LastTradedPrice": 3.0 + i * 0.5,
            "TotalMatched": 500.0 + i * 100,
            "StartingPriceNear": 2.8 + i * 0.5,
            "StartingPriceFar": 3.5 + i * 0.5,
            "AdjustmentFactor": 100.0,
            "Bsp": None,
            "SortPriority": i + 1,
            "RemovalDate": None,
            "AvailableToBack": [
                {"Price": 3.0 + i * 0.5, "Size": 100.0},
                {"Price": 2.9 + i * 0.5, "Size": 50.0},
            ],
            "AvailableToLay": [
                {"Price": 3.1 + i * 0.5, "Size": 80.0},
                {"Price": 3.2 + i * 0.5, "Size": 40.0},
            ],
        })
    return json.dumps({"Runners": result})


def _make_ticks_df(
    n_ticks: int = 5,
    market_id: str = "1.234567890",
    venue: str = "Newmarket",
    start_time: str = "2026-03-26 14:00:00",
    n_runners: int = 3,
    in_play: bool = False,
) -> pd.DataFrame:
    """Build a synthetic ticks DataFrame matching the extractor schema."""
    rows = []
    for i in range(n_ticks):
        rows.append({
            "market_id": market_id,
            "timestamp": pd.Timestamp(f"2026-03-26 13:{50+i}:00"),
            "sequence_number": pd.array([100 + i], dtype=pd.Int64Dtype())[0],
            "venue": venue,
            "market_start_time": pd.Timestamp(start_time),
            "number_of_active_runners": pd.array([n_runners], dtype=pd.Int32Dtype())[0],
            "traded_volume": 10000.0 + i * 500,
            "in_play": in_play,
            "snap_json": _make_snap_json(n_runners=n_runners),
            "winner_selection_id": pd.array([1001], dtype=pd.Int64Dtype())[0],
            "temperature": 15.0,
            "precipitation": 0.0,
            "wind_speed": 5.0,
            "wind_direction": 180.0,
            "humidity": 65.0,
            "weather_code": pd.array([0], dtype=pd.Int32Dtype())[0],
        })
    return pd.DataFrame(rows)


def _make_runners_df(
    market_id: str = "1.234567890",
    n_runners: int = 3,
) -> pd.DataFrame:
    """Build a synthetic runners DataFrame matching the extractor schema."""
    rows = []
    for i in range(n_runners):
        rows.append({
            "market_id": market_id,
            "selection_id": str(1000 + i),
            "runner_name": f"Horse_{i}",
            "sort_priority": str(i + 1),
            "handicap": "0.0",
            "SIRE_NAME": f"Sire_{i}",
            "DAM_NAME": f"Dam_{i}",
            "DAMSIRE_NAME": f"Damsire_{i}",
            "SIRE_YEAR_BORN": "2015",
            "DAM_YEAR_BORN": "2016",
            "DAMSIRE_YEAR_BORN": "2010",
            "SIRE_BRED": "GB",
            "DAM_BRED": "GB",
            "DAMSIRE_BRED": "IRE",
            "BRED": "GB",
            "OFFICIAL_RATING": str(80 + i * 5),
            "ADJUSTED_RATING": str(82 + i * 5),
            "AGE": "4",
            "SEX_TYPE": "Gelding",
            "COLOUR_TYPE": "Bay",
            "WEIGHT_VALUE": str(130 + i),
            "WEIGHT_UNITS": "LB",
            "JOCKEY_NAME": f"Jockey_{i}",
            "JOCKEY_CLAIM": "0",
            "TRAINER_NAME": f"Trainer_{i}",
            "OWNER_NAME": f"Owner_{i}",
            "STALL_DRAW": str(i + 1),
            "CLOTH_NUMBER": str(i + 1),
            "CLOTH_NUMBER_ALPHA": "",
            "FORM": "1234-21",
            "DAYS_SINCE_LAST_RUN": str(14 + i),
            "WEARING": "" if i > 0 else "Blinkers",
            "FORECASTPRICE_NUMERATOR": "5",
            "FORECASTPRICE_DENOMINATOR": "2",
            "COLOURS_DESCRIPTION": "Red and white",
            "COLOURS_FILENAME": "colors.png",
            "runner_id": f"runner_{i}",
        })
    return pd.DataFrame(rows)


def _write_parquet_pair(
    tmp_path: Path,
    date_str: str = "2026-03-26",
    n_ticks: int = 5,
    n_runners: int = 3,
    n_markets: int = 1,
    in_play: bool = False,
) -> Path:
    """Write ticks and runners Parquet files into tmp_path, return the dir."""
    all_ticks = []
    all_runners = []
    for m in range(n_markets):
        mid = f"1.{234567890 + m}"
        st = f"2026-03-26 {14 + m}:00:00"
        ticks = _make_ticks_df(
            n_ticks=n_ticks,
            market_id=mid,
            start_time=st,
            n_runners=n_runners,
            in_play=in_play,
        )
        runners = _make_runners_df(market_id=mid, n_runners=n_runners)
        all_ticks.append(ticks)
        all_runners.append(runners)

    pd.concat(all_ticks).to_parquet(tmp_path / f"{date_str}.parquet", index=False)
    pd.concat(all_runners).to_parquet(
        tmp_path / f"{date_str}_runners.parquet", index=False
    )
    return tmp_path


# ── Tests: _opt_float / _opt_int ─────────────────────────────────────────────


class TestOptHelpers:
    def test_opt_float_valid(self):
        assert _opt_float("3.14") == 3.14

    def test_opt_float_none(self):
        assert _opt_float(None) is None

    def test_opt_float_empty(self):
        assert _opt_float("") is None

    def test_opt_float_invalid(self):
        assert _opt_float("abc") is None

    def test_opt_int_valid(self):
        assert _opt_int("42") == 42

    def test_opt_int_none(self):
        assert _opt_int(None) is None

    def test_opt_int_empty(self):
        assert _opt_int("") is None

    def test_opt_int_invalid(self):
        assert _opt_int("xyz") is None


# ── Tests: parse_snap_json ───────────────────────────────────────────────────


class TestParseSnapJson:
    def test_basic_parse(self):
        snap = _make_snap_json(n_runners=2)
        runners = parse_snap_json(snap)
        assert len(runners) == 2
        assert isinstance(runners[0], RunnerSnap)

    def test_selection_id(self):
        snap = _make_snap_json(n_runners=1)
        runners = parse_snap_json(snap)
        assert runners[0].selection_id == 1000

    def test_ltp(self):
        snap = _make_snap_json(n_runners=1)
        runners = parse_snap_json(snap)
        assert runners[0].last_traded_price == 3.0

    def test_available_to_back(self):
        snap = _make_snap_json(n_runners=1)
        runners = parse_snap_json(snap)
        assert len(runners[0].available_to_back) == 2
        assert runners[0].available_to_back[0].price == 3.0
        assert runners[0].available_to_back[0].size == 100.0

    def test_available_to_lay(self):
        snap = _make_snap_json(n_runners=1)
        runners = parse_snap_json(snap)
        assert len(runners[0].available_to_lay) == 2
        assert runners[0].available_to_lay[0].price == 3.1
        assert runners[0].available_to_lay[0].size == 80.0

    def test_status(self):
        snap = _make_snap_json(runners=[
            {"SelectionId": 1, "Status": "REMOVED", "LastTradedPrice": 0}
        ])
        runners = parse_snap_json(snap)
        assert runners[0].status == "REMOVED"

    def test_bsp_null(self):
        snap = _make_snap_json(n_runners=1)
        runners = parse_snap_json(snap)
        assert runners[0].bsp is None

    def test_bsp_present(self):
        snap = _make_snap_json(runners=[
            {"SelectionId": 1, "Bsp": 4.5, "Status": "ACTIVE", "LastTradedPrice": 4.0}
        ])
        runners = parse_snap_json(snap)
        assert runners[0].bsp == 4.5

    def test_empty_runners(self):
        snap = json.dumps({"Runners": []})
        runners = parse_snap_json(snap)
        assert runners == []

    def test_camelcase_keys(self):
        """Parser supports both PascalCase and camelCase."""
        snap = json.dumps({"runners": [
            {"selectionId": 99, "status": "ACTIVE", "ltp": 5.0, "tv": 200.0}
        ]})
        runners = parse_snap_json(snap)
        assert runners[0].selection_id == 99
        assert runners[0].last_traded_price == 5.0

    def test_missing_atb_atl(self):
        """Runners with no order book levels produce empty lists."""
        snap = json.dumps({"Runners": [
            {"SelectionId": 1, "Status": "ACTIVE", "LastTradedPrice": 3.0}
        ]})
        runners = parse_snap_json(snap)
        assert runners[0].available_to_back == []
        assert runners[0].available_to_lay == []

    def test_total_matched(self):
        snap = _make_snap_json(n_runners=1)
        runners = parse_snap_json(snap)
        assert runners[0].total_matched == 500.0

    def test_starting_price_near_far(self):
        snap = _make_snap_json(n_runners=1)
        runners = parse_snap_json(snap)
        assert runners[0].starting_price_near == 2.8
        assert runners[0].starting_price_far == 3.5

    def test_adjustment_factor(self):
        snap = _make_snap_json(n_runners=1)
        runners = parse_snap_json(snap)
        assert runners[0].adjustment_factor == 100.0

    def test_sort_priority(self):
        snap = _make_snap_json(n_runners=1)
        runners = parse_snap_json(snap)
        assert runners[0].sort_priority == 1

    def test_removal_date_none(self):
        snap = _make_snap_json(n_runners=1)
        runners = parse_snap_json(snap)
        assert runners[0].removal_date is None


# ── Tests: RunnerMeta construction ───────────────────────────────────────────


class TestRunnerMeta:
    def test_builds_from_row(self):
        df = _make_runners_df(n_runners=1)
        meta = _build_runner_meta(df.iloc[0])
        assert isinstance(meta, RunnerMeta)
        assert meta.selection_id == 1000
        assert meta.runner_name == "Horse_0"

    def test_string_fields_preserved(self):
        df = _make_runners_df(n_runners=1)
        meta = _build_runner_meta(df.iloc[0])
        assert meta.official_rating == "80"
        assert meta.stall_draw == "1"
        assert meta.days_since_last_run == "14"

    def test_empty_string_for_missing(self):
        df = _make_runners_df(n_runners=1)
        df.loc[0, "DAYS_SINCE_LAST_RUN"] = ""
        meta = _build_runner_meta(df.iloc[0])
        assert meta.days_since_last_run == ""

    def test_nan_becomes_empty_string(self):
        df = _make_runners_df(n_runners=1)
        df.loc[0, "STALL_DRAW"] = None
        meta = _build_runner_meta(df.iloc[0])
        assert meta.stall_draw == ""

    def test_forecast_price_fields(self):
        df = _make_runners_df(n_runners=1)
        meta = _build_runner_meta(df.iloc[0])
        assert meta.forecastprice_numerator == "5"
        assert meta.forecastprice_denominator == "2"

    def test_wearing_field(self):
        df = _make_runners_df(n_runners=1)
        meta = _build_runner_meta(df.iloc[0])
        assert meta.wearing == "Blinkers"


# ── Tests: Tick construction ─────────────────────────────────────────────────


class TestTickConstruction:
    def test_row_to_tick(self):
        from data.episode_builder import _row_to_tick

        df = _make_ticks_df(n_ticks=1, n_runners=2)
        tick = _row_to_tick(df.iloc[0])
        assert isinstance(tick, Tick)
        assert tick.market_id == "1.234567890"
        assert len(tick.runners) == 2

    def test_weather_fields(self):
        from data.episode_builder import _row_to_tick

        df = _make_ticks_df(n_ticks=1)
        tick = _row_to_tick(df.iloc[0])
        assert tick.temperature == 15.0
        assert tick.precipitation == 0.0
        assert tick.wind_speed == 5.0
        assert tick.humidity == 65.0
        assert tick.weather_code == 0

    def test_winner_selection_id(self):
        from data.episode_builder import _row_to_tick

        df = _make_ticks_df(n_ticks=1)
        tick = _row_to_tick(df.iloc[0])
        assert tick.winner_selection_id == 1001

    def test_null_winner(self):
        from data.episode_builder import _row_to_tick

        df = _make_ticks_df(n_ticks=1)
        df["winner_selection_id"] = pd.array([pd.NA], dtype=pd.Int64Dtype())
        tick = _row_to_tick(df.iloc[0])
        assert tick.winner_selection_id is None

    def test_in_play_false(self):
        from data.episode_builder import _row_to_tick

        df = _make_ticks_df(n_ticks=1, in_play=False)
        tick = _row_to_tick(df.iloc[0])
        assert tick.in_play is False


# ── Tests: _build_day ────────────────────────────────────────────────────────


class TestBuildDay:
    def test_single_market(self):
        ticks = _make_ticks_df(n_ticks=5, n_runners=3)
        runners = _make_runners_df(n_runners=3)
        day = _build_day("2026-03-26", ticks, runners)
        assert isinstance(day, Day)
        assert day.date == "2026-03-26"
        assert len(day.races) == 1
        assert len(day.races[0].ticks) == 5

    def test_multiple_markets(self):
        ticks1 = _make_ticks_df(n_ticks=3, market_id="1.100", start_time="2026-03-26 14:00:00")
        ticks2 = _make_ticks_df(n_ticks=4, market_id="1.200", start_time="2026-03-26 15:00:00")
        runners1 = _make_runners_df(market_id="1.100")
        runners2 = _make_runners_df(market_id="1.200")
        ticks = pd.concat([ticks1, ticks2])
        runners = pd.concat([runners1, runners2])
        day = _build_day("2026-03-26", ticks, runners)
        assert len(day.races) == 2
        # Races sorted by market_start_time
        assert day.races[0].market_id == "1.100"
        assert day.races[1].market_id == "1.200"

    def test_ticks_sorted_by_sequence_number(self):
        ticks = _make_ticks_df(n_ticks=5)
        # Shuffle the rows
        ticks = ticks.sample(frac=1, random_state=42)
        runners = _make_runners_df()
        day = _build_day("2026-03-26", ticks, runners)
        seq_nums = [t.sequence_number for t in day.races[0].ticks]
        assert seq_nums == sorted(seq_nums)

    def test_in_play_raises(self):
        ticks = _make_ticks_df(n_ticks=3, in_play=True)
        runners = _make_runners_df()
        with pytest.raises(ValueError, match="in-play"):
            _build_day("2026-03-26", ticks, runners)

    def test_empty_ticks(self):
        ticks = pd.DataFrame(columns=["market_id", "in_play"])
        runners = pd.DataFrame()
        day = _build_day("2026-03-26", ticks, runners)
        assert len(day.races) == 0

    def test_runner_metadata_attached(self):
        ticks = _make_ticks_df(n_ticks=1, n_runners=2)
        runners = _make_runners_df(n_runners=2)
        day = _build_day("2026-03-26", ticks, runners)
        race = day.races[0]
        assert 1000 in race.runner_metadata
        assert 1001 in race.runner_metadata
        assert race.runner_metadata[1000].runner_name == "Horse_0"

    def test_missing_runners_parquet_ok(self):
        """Day builds fine with empty runners DataFrame."""
        ticks = _make_ticks_df(n_ticks=2)
        day = _build_day("2026-03-26", ticks, pd.DataFrame())
        assert len(day.races) == 1
        assert day.races[0].runner_metadata == {}

    def test_race_level_fields(self):
        ticks = _make_ticks_df(n_ticks=2)
        runners = _make_runners_df()
        day = _build_day("2026-03-26", ticks, runners)
        race = day.races[0]
        assert race.market_id == "1.234567890"
        assert race.venue == "Newmarket"
        assert race.winner_selection_id == 1001

    def test_races_sorted_by_start_time(self):
        """Races must be ordered by market_start_time, not by market_id."""
        # Create race B first (earlier start) then race A (later start)
        ticks_b = _make_ticks_df(
            n_ticks=2, market_id="1.999", start_time="2026-03-26 13:00:00"
        )
        ticks_a = _make_ticks_df(
            n_ticks=2, market_id="1.111", start_time="2026-03-26 16:00:00"
        )
        ticks = pd.concat([ticks_a, ticks_b])
        runners = pd.DataFrame()
        day = _build_day("2026-03-26", ticks, runners)
        assert day.races[0].market_id == "1.999"  # Earlier start
        assert day.races[1].market_id == "1.111"


# ── Tests: load_day ──────────────────────────────────────────────────────────


class TestLoadDay:
    def test_load_from_parquet(self, tmp_path):
        _write_parquet_pair(tmp_path)
        day = load_day("2026-03-26", data_dir=tmp_path)
        assert isinstance(day, Day)
        assert len(day.races) == 1
        assert len(day.races[0].ticks) == 5

    def test_missing_ticks_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_day("2099-01-01", data_dir=tmp_path)

    def test_missing_runners_file_ok(self, tmp_path):
        """If runners Parquet missing, Day still builds (no metadata)."""
        ticks = _make_ticks_df(n_ticks=2)
        ticks.to_parquet(tmp_path / "2026-03-26.parquet", index=False)
        # No runners file written
        day = load_day("2026-03-26", data_dir=tmp_path)
        assert len(day.races) == 1
        assert day.races[0].runner_metadata == {}

    def test_multi_market_day(self, tmp_path):
        _write_parquet_pair(tmp_path, n_markets=3)
        day = load_day("2026-03-26", data_dir=tmp_path)
        assert len(day.races) == 3

    def test_in_play_ticks_rejected(self, tmp_path):
        _write_parquet_pair(tmp_path, in_play=True)
        with pytest.raises(ValueError, match="in-play"):
            load_day("2026-03-26", data_dir=tmp_path)

    def test_snap_json_parsed(self, tmp_path):
        _write_parquet_pair(tmp_path, n_runners=4)
        day = load_day("2026-03-26", data_dir=tmp_path)
        tick = day.races[0].ticks[0]
        assert len(tick.runners) == 4
        assert tick.runners[0].selection_id == 1000


# ── Tests: load_days ─────────────────────────────────────────────────────────


class TestLoadDays:
    def test_load_multiple_days(self, tmp_path):
        _write_parquet_pair(tmp_path, date_str="2026-03-25")
        _write_parquet_pair(tmp_path, date_str="2026-03-26")
        days = load_days(["2026-03-25", "2026-03-26"], data_dir=tmp_path)
        assert len(days) == 2
        assert days[0].date == "2026-03-25"
        assert days[1].date == "2026-03-26"

    def test_skips_missing_dates(self, tmp_path):
        _write_parquet_pair(tmp_path, date_str="2026-03-26")
        days = load_days(["2026-03-25", "2026-03-26"], data_dir=tmp_path)
        assert len(days) == 1
        assert days[0].date == "2026-03-26"

    def test_empty_list(self, tmp_path):
        days = load_days([], data_dir=tmp_path)
        assert days == []


# ── Tests: PriceSize dataclass ───────────────────────────────────────────────


class TestPriceSize:
    def test_frozen(self):
        ps = PriceSize(price=3.0, size=100.0)
        with pytest.raises(AttributeError):
            ps.price = 4.0  # type: ignore[misc]

    def test_values(self):
        ps = PriceSize(price=5.5, size=200.0)
        assert ps.price == 5.5
        assert ps.size == 200.0


# ── Tests: RunnerSnap dataclass ──────────────────────────────────────────────


class TestRunnerSnapDataclass:
    def test_frozen(self):
        snap = parse_snap_json(_make_snap_json(n_runners=1))[0]
        with pytest.raises(AttributeError):
            snap.selection_id = 999  # type: ignore[misc]

    def test_slots(self):
        snap = parse_snap_json(_make_snap_json(n_runners=1))[0]
        assert hasattr(snap, "__slots__")


# ── Tests: Tick dataclass ────────────────────────────────────────────────────


class TestTickDataclass:
    def test_frozen(self):
        from data.episode_builder import _row_to_tick

        df = _make_ticks_df(n_ticks=1)
        tick = _row_to_tick(df.iloc[0])
        with pytest.raises(AttributeError):
            tick.market_id = "changed"  # type: ignore[misc]


# ── Tests: edge cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_runner(self, tmp_path):
        _write_parquet_pair(tmp_path, n_runners=1)
        day = load_day("2026-03-26", data_dir=tmp_path)
        assert len(day.races[0].ticks[0].runners) == 1

    def test_single_tick(self, tmp_path):
        _write_parquet_pair(tmp_path, n_ticks=1)
        day = load_day("2026-03-26", data_dir=tmp_path)
        assert len(day.races[0].ticks) == 1

    def test_many_markets(self, tmp_path):
        _write_parquet_pair(tmp_path, n_markets=10)
        day = load_day("2026-03-26", data_dir=tmp_path)
        assert len(day.races) == 10

    def test_removed_runner_in_snap(self):
        """A removed runner should still be in the snap with status REMOVED."""
        snap = _make_snap_json(runners=[
            {"SelectionId": 1, "Status": "REMOVED", "LastTradedPrice": 0,
             "RemovalDate": "2026-03-26T10:00:00"},
            {"SelectionId": 2, "Status": "ACTIVE", "LastTradedPrice": 3.0},
        ])
        runners = parse_snap_json(snap)
        assert len(runners) == 2
        assert runners[0].status == "REMOVED"
        assert runners[0].removal_date == "2026-03-26T10:00:00"
        assert runners[1].status == "ACTIVE"

    def test_snap_json_with_missing_prices(self):
        """Runners with all zeros for prices should parse without error."""
        snap = json.dumps({"Runners": [
            {"SelectionId": 1, "Status": "ACTIVE", "LastTradedPrice": 0,
             "TotalMatched": 0, "StartingPriceNear": 0, "StartingPriceFar": 0,
             "AvailableToBack": [], "AvailableToLay": []}
        ]})
        runners = parse_snap_json(snap)
        assert runners[0].last_traded_price == 0.0
        assert runners[0].available_to_back == []

    def test_null_weather(self, tmp_path):
        """Weather fields can be null (failed fetch)."""
        ticks = _make_ticks_df(n_ticks=1)
        ticks["temperature"] = [None]
        ticks["precipitation"] = [None]
        ticks["wind_speed"] = [None]
        ticks["humidity"] = [None]
        ticks["weather_code"] = pd.array([pd.NA], dtype=pd.Int32Dtype())
        runners = _make_runners_df()
        ticks.to_parquet(tmp_path / "2026-03-26.parquet", index=False)
        runners.to_parquet(tmp_path / "2026-03-26_runners.parquet", index=False)

        day = load_day("2026-03-26", data_dir=tmp_path)
        tick = day.races[0].ticks[0]
        assert tick.temperature is None
        assert tick.weather_code is None

    def test_large_selection_id(self):
        """Betfair selection IDs can exceed 2^31."""
        snap = json.dumps({"Runners": [
            {"SelectionId": 3000000000, "Status": "ACTIVE", "LastTradedPrice": 5.0}
        ]})
        runners = parse_snap_json(snap)
        assert runners[0].selection_id == 3000000000

    def test_mixed_in_play_raises(self):
        """Even a single in-play tick in the mix should raise."""
        ticks = _make_ticks_df(n_ticks=3, in_play=False)
        extra = _make_ticks_df(n_ticks=1, in_play=True)
        extra["sequence_number"] = pd.array([999], dtype=pd.Int64Dtype())
        mixed = pd.concat([ticks, extra])
        with pytest.raises(ValueError, match="1 in-play"):
            _build_day("2026-03-26", mixed, pd.DataFrame())
