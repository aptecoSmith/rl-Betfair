"""Tests for data/extractor.py.

All tests use mock objects — no live MySQL connection is required.
The strategy is to mock at the DataExtractor._query_ticks /
_query_runners boundary so that the SQL, dtype-casting, Parquet output,
and ProgressTracker logic are all exercised without needing a real DB.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from data.extractor import (
    RUNNERS_COLUMNS,
    SQL_EW_SUBQUERY_FULL,
    SQL_EW_SUBQUERY_NULL,
    SQL_POLLED_TICKS_TEMPLATE,
    SQL_RUNNERS,
    SQL_TICKS_TEMPLATE,
    TICKS_COLUMNS,
    DataExtractor,
    _cast_ticks,
    _column_exists,
    _optional_column_expr,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def minimal_config(tmp_path):
    """Minimal config dict pointing output at a temp directory."""
    return {
        "database": {
            "host": "localhost",
            "port": 3307,
            "cold_data_db": "coldData",
            "hot_data_db": "hotDataRefactored",
        },
        "paths": {
            "processed_data": str(tmp_path / "processed"),
        },
    }


def _make_ticks_df(n: int = 3, winner_id: int | None = 12345678901) -> pd.DataFrame:
    """Return a minimal ticks DataFrame that passes _cast_ticks."""
    return pd.DataFrame({
        "market_id": [f"1.{i}" for i in range(n)],
        "timestamp": pd.to_datetime(["2026-03-01 10:00:00"] * n),
        "sequence_number": [100 + i for i in range(n)],
        "venue": ["Newmarket"] * n,
        "market_start_time": pd.to_datetime(["2026-03-01 11:30:00"] * n),
        "market_type": ["WIN"] * n,
        "number_of_active_runners": [8] * n,
        "traded_volume": [5000.0] * n,
        "in_play": [0] * n,           # MySQL TINYINT — cast to bool
        "snap_json": ['{"runners":[]}'] * n,
        "winner_selection_id": [winner_id] * n,
        "each_way_divisor": [None] * n,
        "number_of_each_way_places": [None] * n,
        "temperature": [12.5] * n,
        "precipitation": [0.0] * n,
        "wind_speed": [3.2] * n,
        "wind_direction": [180.0] * n,
        "humidity": [72.0] * n,
        "weather_code": [0] * n,
    })


def _make_runners_df(market_ids: list[str] | None = None) -> pd.DataFrame:
    """Return a minimal runners DataFrame."""
    market_ids = market_ids or ["1.0"]
    rows = []
    for mid in market_ids:
        rows.append({col: None for col in RUNNERS_COLUMNS})
        rows[-1].update({
            "market_id": mid,
            "selection_id": 12345678901,
            "runner_name": "Arkle",
            "sort_priority": 1,
            "handicap": 0.0,
            "JOCKEY_NAME": "F. Dettori",
            "OFFICIAL_RATING": "110",
            "STALL_DRAW": "3",
            "FORECASTPRICE_NUMERATOR": "4",
            "FORECASTPRICE_DENOMINATOR": "1",
            "DAYS_SINCE_LAST_RUN": "14",
        })
    return pd.DataFrame(rows)


def _make_extractor(config, ticks_df=None, runners_df=None, polled=False) -> DataExtractor:
    """Return a DataExtractor with _query_ticks/_query_runners/_query_market_names mocked.

    When *polled* is False (default), ``_has_polled_date`` is mocked to return
    False so the legacy ``ResolvedMarketSnaps`` path is used.
    """
    mock_engine = MagicMock()
    extractor = DataExtractor(config, engine=mock_engine)

    # Default: polled source not available → legacy path
    extractor._has_polled_date = MagicMock(return_value=polled)

    if ticks_df is not None:
        extractor._query_ticks = MagicMock(return_value=_cast_ticks(ticks_df.copy()))
    if runners_df is not None:
        extractor._query_runners = MagicMock(return_value=runners_df.copy())

    # Default market names mock — returns names for all market IDs in the ticks
    if ticks_df is not None and not ticks_df.empty:
        market_ids = list(ticks_df["market_id"].unique())
        names_df = pd.DataFrame({
            "market_id": market_ids,
            "market_name": [f"Race {i+1}" for i in range(len(market_ids))],
        })
    else:
        names_df = pd.DataFrame(columns=["market_id", "market_name"])
    extractor._query_market_names = MagicMock(return_value=names_df)

    return extractor


# ── SQL content tests ─────────────────────────────────────────────────────────

class TestSqlContent:
    def test_ticks_query_uses_resolved_market_snaps_as_driver(self):
        """InPlay filtering is now done in Python after parsing SnapJson."""
        sql = SQL_TICKS_TEMPLATE
        assert "FROM ResolvedMarketSnaps rms" in sql

    def test_ticks_query_joins_resolved_market_snaps(self):
        sql = SQL_TICKS_TEMPLATE
        assert "ResolvedMarketSnaps" in sql

    def test_ticks_query_selects_snap_json(self):
        sql = SQL_TICKS_TEMPLATE
        assert "SnapJson" in sql
        assert "snap_json" in sql

    def test_ticks_query_casts_winner_selection_id(self):
        sql = SQL_TICKS_TEMPLATE
        assert "CAST" in sql
        assert "WinnerSelectionId" in sql
        assert "SIGNED" in sql

    def test_ticks_query_filters_pre_race_weather(self):
        sql = SQL_TICKS_TEMPLATE
        assert "PRE_RACE" in sql
        assert "WeatherObservations" in sql

    def test_ticks_query_uses_date_parameter(self):
        sql = SQL_TICKS_TEMPLATE
        assert ":target_date" in sql

    def test_ticks_query_orders_by_sequence(self):
        sql = SQL_TICKS_TEMPLATE
        assert "SequenceNumber" in sql
        assert "ORDER BY" in sql.upper()

    def test_ticks_query_references_cold_data_db(self):
        sql = SQL_TICKS_TEMPLATE
        assert "coldData." in sql

    def test_runners_query_joins_runner_metadata(self):
        assert "RunnerMetaData" in SQL_RUNNERS
        assert "runnerdescription" in SQL_RUNNERS

    def test_runners_query_uses_market_ids_parameter(self):
        assert ":market_ids" in SQL_RUNNERS

    def test_runners_query_selects_selection_id(self):
        assert "SelectionId" in SQL_RUNNERS
        assert "selection_id" in SQL_RUNNERS

    def test_runners_query_references_cold_data_db(self):
        assert "coldData." in SQL_RUNNERS


# ── Column schema tests ───────────────────────────────────────────────────────

class TestColumnSchemas:
    def test_ticks_columns_complete(self):
        required = {
            "market_id", "timestamp", "sequence_number", "venue",
            "market_start_time", "market_type", "market_name",
            "number_of_active_runners", "traded_volume",
            "in_play", "snap_json", "winner_selection_id",
            "each_way_divisor", "number_of_each_way_places",
            "race_status",
            "temperature", "precipitation", "wind_speed", "wind_direction",
            "humidity", "weather_code",
        }
        assert required == set(TICKS_COLUMNS)

    def test_runners_columns_include_metadata_fields(self):
        required_metadata = {
            "OFFICIAL_RATING", "STALL_DRAW", "JOCKEY_NAME", "TRAINER_NAME",
            "FORECASTPRICE_NUMERATOR", "FORECASTPRICE_DENOMINATOR",
            "DAYS_SINCE_LAST_RUN", "AGE", "WEIGHT_VALUE", "FORM",
        }
        assert required_metadata.issubset(set(RUNNERS_COLUMNS))

    def test_runners_columns_include_identity_fields(self):
        for col in ("market_id", "selection_id", "runner_name"):
            assert col in RUNNERS_COLUMNS


# ── _cast_ticks dtype tests ───────────────────────────────────────────────────

class TestCastTicks:
    def test_winner_selection_id_becomes_int64(self):
        df = _make_ticks_df()
        result = _cast_ticks(df)
        assert result["winner_selection_id"].dtype == pd.Int64Dtype()

    def test_winner_selection_id_preserves_large_value(self):
        """Betfair selection IDs exceed int32 max (2_147_483_647)."""
        large_id = 99_999_999_999
        df = _make_ticks_df(winner_id=large_id)
        result = _cast_ticks(df)
        assert result["winner_selection_id"].iloc[0] == large_id

    def test_winner_selection_id_nullable_when_null(self):
        df = _make_ticks_df(winner_id=None)
        result = _cast_ticks(df)
        assert result["winner_selection_id"].isna().all()

    def test_sequence_number_becomes_int64(self):
        df = _make_ticks_df()
        result = _cast_ticks(df)
        assert result["sequence_number"].dtype == pd.Int64Dtype()

    def test_number_of_active_runners_becomes_int32(self):
        df = _make_ticks_df()
        result = _cast_ticks(df)
        assert result["number_of_active_runners"].dtype == pd.Int32Dtype()

    def test_weather_code_becomes_int32(self):
        df = _make_ticks_df()
        result = _cast_ticks(df)
        assert result["weather_code"].dtype == pd.Int32Dtype()

    def test_in_play_becomes_bool(self):
        df = _make_ticks_df()
        result = _cast_ticks(df)
        assert result["in_play"].dtype == bool

    def test_in_play_false_from_zero(self):
        df = _make_ticks_df()
        result = _cast_ticks(df)
        assert not result["in_play"].any()

    def test_empty_dataframe_returned_unchanged(self):
        df = pd.DataFrame(columns=TICKS_COLUMNS)
        result = _cast_ticks(df)
        assert result.empty

    def test_weather_nulls_preserved(self):
        df = _make_ticks_df()
        df["temperature"] = None
        df["weather_code"] = None
        result = _cast_ticks(df)
        assert result["temperature"].isna().all()
        assert result["weather_code"].isna().all()


# ── extract_date tests ────────────────────────────────────────────────────────

class TestExtractDate:
    def test_returns_true_when_ticks_found(self, minimal_config):
        extractor = _make_extractor(
            minimal_config,
            ticks_df=_make_ticks_df(),
            runners_df=_make_runners_df(["1.0", "1.1", "1.2"]),
        )
        # mock connect() context manager
        extractor._engine.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        extractor._engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        result = extractor.extract_date(date(2026, 3, 1))
        assert result is True

    def test_returns_false_when_no_ticks(self, minimal_config):
        extractor = _make_extractor(
            minimal_config,
            ticks_df=pd.DataFrame(columns=TICKS_COLUMNS),
            runners_df=_make_runners_df(),
        )
        extractor._engine.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        extractor._engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        result = extractor.extract_date(date(2026, 3, 1))
        assert result is False

    def test_no_parquet_written_when_no_ticks(self, minimal_config, tmp_path):
        extractor = _make_extractor(
            minimal_config,
            ticks_df=pd.DataFrame(columns=TICKS_COLUMNS),
            runners_df=_make_runners_df(),
        )
        extractor._engine.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        extractor._engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        extractor.extract_date(date(2026, 3, 1))
        assert not list(Path(minimal_config["paths"]["processed_data"]).glob("*.parquet"))

    def test_ticks_parquet_written(self, minimal_config):
        extractor = _make_extractor(
            minimal_config,
            ticks_df=_make_ticks_df(),
            runners_df=_make_runners_df(["1.0", "1.1", "1.2"]),
        )
        extractor._engine.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        extractor._engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        extractor.extract_date(date(2026, 3, 1))
        out = Path(minimal_config["paths"]["processed_data"])
        assert (out / "2026-03-01.parquet").exists()

    def test_runners_parquet_written(self, minimal_config):
        extractor = _make_extractor(
            minimal_config,
            ticks_df=_make_ticks_df(),
            runners_df=_make_runners_df(["1.0", "1.1", "1.2"]),
        )
        extractor._engine.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        extractor._engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        extractor.extract_date(date(2026, 3, 1))
        out = Path(minimal_config["paths"]["processed_data"])
        assert (out / "2026-03-01_runners.parquet").exists()

    def test_ticks_parquet_schema(self, minimal_config):
        extractor = _make_extractor(
            minimal_config,
            ticks_df=_make_ticks_df(),
            runners_df=_make_runners_df(["1.0", "1.1", "1.2"]),
        )
        extractor._engine.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        extractor._engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        extractor.extract_date(date(2026, 3, 1))
        out = Path(minimal_config["paths"]["processed_data"])
        df = pd.read_parquet(out / "2026-03-01.parquet")
        for col in TICKS_COLUMNS:
            assert col in df.columns, f"Missing column in ticks Parquet: {col}"

    def test_runners_parquet_schema(self, minimal_config):
        extractor = _make_extractor(
            minimal_config,
            ticks_df=_make_ticks_df(),
            runners_df=_make_runners_df(["1.0", "1.1", "1.2"]),
        )
        extractor._engine.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        extractor._engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        extractor.extract_date(date(2026, 3, 1))
        out = Path(minimal_config["paths"]["processed_data"])
        df = pd.read_parquet(out / "2026-03-01_runners.parquet")
        for col in RUNNERS_COLUMNS:
            assert col in df.columns, f"Missing column in runners Parquet: {col}"

    def test_ticks_parquet_winner_id_is_int64(self, minimal_config):
        extractor = _make_extractor(
            minimal_config,
            ticks_df=_make_ticks_df(winner_id=99_999_999_999),
            runners_df=_make_runners_df(["1.0", "1.1", "1.2"]),
        )
        extractor._engine.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        extractor._engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        extractor.extract_date(date(2026, 3, 1))
        out = Path(minimal_config["paths"]["processed_data"])
        df = pd.read_parquet(out / "2026-03-01.parquet")
        # Arrow reads Int64 back as int64 or pandas Int64
        assert str(df["winner_selection_id"].dtype) in ("Int64", "int64")
        assert df["winner_selection_id"].iloc[0] == 99_999_999_999

    def test_query_ticks_called_with_correct_date(self, minimal_config):
        extractor = _make_extractor(
            minimal_config,
            ticks_df=_make_ticks_df(),
            runners_df=_make_runners_df(["1.0", "1.1", "1.2"]),
        )
        extractor._engine.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        extractor._engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        target = date(2026, 3, 1)
        extractor.extract_date(target)
        extractor._query_ticks.assert_called_once()
        assert extractor._query_ticks.call_args[0][0] == target

    def test_query_runners_called_with_market_ids(self, minimal_config):
        ticks = _make_ticks_df(n=3)
        ticks["market_id"] = ["1.100", "1.101", "1.102"]
        extractor = _make_extractor(
            minimal_config,
            ticks_df=ticks,
            runners_df=_make_runners_df(["1.100", "1.101", "1.102"]),
        )
        extractor._engine.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        extractor._engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        extractor.extract_date(date(2026, 3, 1))
        extractor._query_runners.assert_called_once()
        actual_ids = set(extractor._query_runners.call_args[0][0])
        assert actual_ids == {"1.100", "1.101", "1.102"}


# ── extract_all + ProgressTracker tests ──────────────────────────────────────

class TestExtractAll:
    def _make_extractor_with_dates(self, config, available_dates, ticks_df, runners_df):
        extractor = _make_extractor(config, ticks_df=ticks_df, runners_df=runners_df)
        extractor.get_available_dates = MagicMock(return_value=available_dates)
        extractor._engine.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        extractor._engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        return extractor

    def test_returns_zero_when_no_dates(self, minimal_config):
        extractor = self._make_extractor_with_dates(
            minimal_config, [], _make_ticks_df(), _make_runners_df()
        )
        assert extractor.extract_all() == 0

    def test_returns_count_of_successful_days(self, minimal_config):
        dates = [date(2026, 3, 1), date(2026, 3, 2)]
        extractor = self._make_extractor_with_dates(
            minimal_config, dates, _make_ticks_df(), _make_runners_df()
        )
        result = extractor.extract_all()
        assert result == 2

    def test_start_date_filter(self, minimal_config):
        dates = [date(2026, 3, 1), date(2026, 3, 2), date(2026, 3, 3)]
        extractor = self._make_extractor_with_dates(
            minimal_config, dates, _make_ticks_df(), _make_runners_df()
        )
        result = extractor.extract_all(start_date=date(2026, 3, 2))
        assert result == 2  # only 2026-03-02 and 2026-03-03

    def test_end_date_filter(self, minimal_config):
        dates = [date(2026, 3, 1), date(2026, 3, 2), date(2026, 3, 3)]
        extractor = self._make_extractor_with_dates(
            minimal_config, dates, _make_ticks_df(), _make_runners_df()
        )
        result = extractor.extract_all(end_date=date(2026, 3, 2))
        assert result == 2  # only 2026-03-01 and 2026-03-02

    def test_date_range_filter(self, minimal_config):
        dates = [date(2026, 3, 1), date(2026, 3, 2), date(2026, 3, 3)]
        extractor = self._make_extractor_with_dates(
            minimal_config, dates, _make_ticks_df(), _make_runners_df()
        )
        result = extractor.extract_all(
            start_date=date(2026, 3, 2), end_date=date(2026, 3, 2)
        )
        assert result == 1

    def test_progress_tracker_ticks_once_per_date(self, minimal_config):
        dates = [date(2026, 3, 1), date(2026, 3, 2), date(2026, 3, 3)]
        extractor = self._make_extractor_with_dates(
            minimal_config, dates, _make_ticks_df(), _make_runners_df()
        )
        with patch("data.extractor.ProgressTracker") as MockTracker:
            mock_tracker = MagicMock()
            mock_tracker.to_dict.return_value = {
                "completed": 1, "total": 3, "process_eta_human": "1m"
            }
            MockTracker.return_value = mock_tracker
            extractor.extract_all()
        assert mock_tracker.tick.call_count == 3

    def test_progress_tracker_total_matches_date_count(self, minimal_config):
        dates = [date(2026, 3, 1), date(2026, 3, 2)]
        extractor = self._make_extractor_with_dates(
            minimal_config, dates, _make_ticks_df(), _make_runners_df()
        )
        with patch("data.extractor.ProgressTracker") as MockTracker:
            mock_tracker = MagicMock()
            mock_tracker.to_dict.return_value = {
                "completed": 1, "total": 2, "process_eta_human": "30s"
            }
            MockTracker.return_value = mock_tracker
            extractor.extract_all()
        MockTracker.assert_called_once()
        assert MockTracker.call_args.kwargs.get("total", MockTracker.call_args.args[0] if MockTracker.call_args.args else None) == 2

    def test_empty_ticks_day_counted_as_failure(self, minimal_config):
        dates = [date(2026, 3, 1)]
        extractor = self._make_extractor_with_dates(
            minimal_config,
            dates,
            pd.DataFrame(columns=TICKS_COLUMNS),  # empty — no data
            _make_runners_df(),
        )
        result = extractor.extract_all()
        assert result == 0


# ── get_available_dates tests ─────────────────────────────────────────────────

class TestGetAvailableDates:
    def test_returns_list_of_dates(self, minimal_config):
        mock_engine = MagicMock()
        # Simulate two rows returned from the DB
        row1 = MagicMock()
        row1.race_date = date(2026, 3, 1)
        row2 = MagicMock()
        row2.race_date = date(2026, 3, 2)
        mock_conn = MagicMock()
        mock_conn.execute.return_value = iter([row1, row2])
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        extractor = DataExtractor(minimal_config, engine=mock_engine)
        result = extractor.get_available_dates()
        assert result == [date(2026, 3, 1), date(2026, 3, 2)]

    def test_returns_empty_list_when_no_data(self, minimal_config):
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value = iter([])
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        extractor = DataExtractor(minimal_config, engine=mock_engine)
        result = extractor.get_available_dates()
        assert result == []


# ── Parquet round-trip tests ──────────────────────────────────────────────────

class TestParquetRoundTrip:
    def test_ticks_parquet_row_count_preserved(self, minimal_config):
        ticks = _make_ticks_df(n=5)
        extractor = _make_extractor(
            minimal_config,
            ticks_df=ticks,
            runners_df=_make_runners_df(list(ticks["market_id"].unique())),
        )
        extractor._engine.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        extractor._engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        extractor.extract_date(date(2026, 3, 1))
        out = Path(minimal_config["paths"]["processed_data"])
        df = pd.read_parquet(out / "2026-03-01.parquet")
        assert len(df) == 5

    def test_snap_json_preserved_exactly(self, minimal_config):
        ticks = _make_ticks_df(n=1)
        snap = '{"runners":[{"id":123,"ltp":4.5}]}'
        ticks["snap_json"] = [snap]
        extractor = _make_extractor(
            minimal_config,
            ticks_df=ticks,
            runners_df=_make_runners_df(["1.0"]),
        )
        extractor._engine.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        extractor._engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        extractor.extract_date(date(2026, 3, 1))
        out = Path(minimal_config["paths"]["processed_data"])
        df = pd.read_parquet(out / "2026-03-01.parquet")
        assert df["snap_json"].iloc[0] == snap

    def test_runner_metadata_strings_not_parsed(self, minimal_config):
        """RunnerMetaData numeric fields stay as strings — no parsing in extractor."""
        ticks = _make_ticks_df(n=1)
        runners = _make_runners_df(["1.0"])
        runners["OFFICIAL_RATING"] = "110"   # string, not int
        runners["STALL_DRAW"] = "3"           # string, not int
        extractor = _make_extractor(minimal_config, ticks_df=ticks, runners_df=runners)
        extractor._engine.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        extractor._engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        extractor.extract_date(date(2026, 3, 1))
        out = Path(minimal_config["paths"]["processed_data"])
        df = pd.read_parquet(out / "2026-03-01_runners.parquet")
        # Must come back as string (or object), never as int/float
        assert df["OFFICIAL_RATING"].iloc[0] == "110"
        assert df["STALL_DRAW"].iloc[0] == "3"

    def test_days_since_last_run_empty_string_preserved(self, minimal_config):
        """DAYS_SINCE_LAST_RUN = '' (first-time runners) must survive round-trip."""
        ticks = _make_ticks_df(n=1)
        runners = _make_runners_df(["1.0"])
        runners["DAYS_SINCE_LAST_RUN"] = ""
        extractor = _make_extractor(minimal_config, ticks_df=ticks, runners_df=runners)
        extractor._engine.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        extractor._engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        extractor.extract_date(date(2026, 3, 1))
        out = Path(minimal_config["paths"]["processed_data"])
        df = pd.read_parquet(out / "2026-03-01_runners.parquet")
        assert df["DAYS_SINCE_LAST_RUN"].iloc[0] == ""


# ── Schema-tolerant query tests ──────────────────────────────────────────────

def _make_column_exists_conn(existing: set[tuple[str, str]]) -> MagicMock:
    """Build a mock sa.Connection whose execute() simulates
    ``information_schema.COLUMNS`` lookups.

    *existing* is a set of ``(table_name, column_name)`` tuples that should
    be reported as present. Everything else returns no rows.
    """
    conn = MagicMock()

    def _execute(stmt, params=None):
        result = MagicMock()
        if params and "table_name" in params and "column_name" in params:
            key = (params["table_name"], params["column_name"])
            result.fetchone.return_value = (1,) if key in existing else None
        else:
            result.fetchone.return_value = None
        return result

    conn.execute.side_effect = _execute
    return conn


class TestColumnExists:
    def test_returns_true_when_column_present(self):
        conn = _make_column_exists_conn({("PolledMarketSnapshots", "EachWayDivisor")})
        assert _column_exists(conn, "PolledMarketSnapshots", "EachWayDivisor") is True

    def test_returns_false_when_column_missing(self):
        conn = _make_column_exists_conn(set())
        assert _column_exists(conn, "PolledMarketSnapshots", "EachWayDivisor") is False

    def test_returns_false_when_table_missing(self):
        conn = _make_column_exists_conn({("OtherTable", "EachWayDivisor")})
        assert _column_exists(conn, "PolledMarketSnapshots", "EachWayDivisor") is False

    def test_returns_false_on_execute_error(self):
        """If information_schema lookup itself errors, treat as missing rather than crash."""
        conn = MagicMock()
        conn.execute.side_effect = RuntimeError("information_schema unavailable")
        assert _column_exists(conn, "PolledMarketSnapshots", "EachWayDivisor") is False


class TestOptionalColumnExpr:
    def test_returns_real_column_when_present(self):
        conn = _make_column_exists_conn({("PolledMarketSnapshots", "EachWayDivisor")})
        expr = _optional_column_expr(
            conn, "PolledMarketSnapshots", "pms", "EachWayDivisor", "each_way_divisor",
        )
        assert expr == "pms.EachWayDivisor AS each_way_divisor"

    def test_returns_null_placeholder_when_missing(self):
        conn = _make_column_exists_conn(set())
        expr = _optional_column_expr(
            conn, "PolledMarketSnapshots", "pms", "EachWayDivisor", "each_way_divisor",
        )
        assert expr == "NULL AS each_way_divisor"

    def test_alias_preserved_in_both_branches(self):
        """Alias must match regardless of whether column exists — downstream DataFrame
        schema must stay stable."""
        present_conn = _make_column_exists_conn(
            {("updates", "NumberOfEachWayPlaces")},
        )
        missing_conn = _make_column_exists_conn(set())
        present = _optional_column_expr(
            present_conn, "updates", "u", "NumberOfEachWayPlaces", "number_of_each_way_places",
        )
        missing = _optional_column_expr(
            missing_conn, "updates", "u", "NumberOfEachWayPlaces", "number_of_each_way_places",
        )
        assert present.endswith("AS number_of_each_way_places")
        assert missing.endswith("AS number_of_each_way_places")


class TestSqlTemplatePlaceholders:
    """The template constants must declare placeholders that the query methods
    substitute at runtime. If these names drift, _query_ticks /
    _query_polled_ticks will raise KeyError."""

    def test_ticks_template_has_ew_subquery_placeholder(self):
        assert "{ew_subquery}" in SQL_TICKS_TEMPLATE

    def test_polled_template_has_each_way_placeholders(self):
        assert "{each_way_divisor_expr}" in SQL_POLLED_TICKS_TEMPLATE
        assert "{number_of_each_way_places_expr}" in SQL_POLLED_TICKS_TEMPLATE

    def test_ew_subquery_full_matches_original_semantics(self):
        """The 'full' variant must still MAX() the two columns grouped by MarketId,
        preserving the original pre-refactor behaviour."""
        assert "MAX(EachWayDivisor)" in SQL_EW_SUBQUERY_FULL
        assert "MAX(NumberOfEachWayPlaces)" in SQL_EW_SUBQUERY_FULL
        assert "GROUP BY MarketId" in SQL_EW_SUBQUERY_FULL
        assert "FROM updates" in SQL_EW_SUBQUERY_FULL

    def test_ew_subquery_null_emits_null_columns(self):
        """The fallback variant must emit NULL columns with the expected aliases."""
        assert "NULL" in SQL_EW_SUBQUERY_NULL
        assert "AS EachWayDivisor" in SQL_EW_SUBQUERY_NULL
        assert "AS NumberOfEachWayPlaces" in SQL_EW_SUBQUERY_NULL
        assert "FROM updates" in SQL_EW_SUBQUERY_NULL

    def test_polled_template_formats_successfully_with_optional_exprs(self):
        """Sanity check: the polled template must format without error when
        substituted with either real columns or NULL placeholders."""
        sql = SQL_POLLED_TICKS_TEMPLATE.format(
            each_way_divisor_expr="NULL AS each_way_divisor",
            number_of_each_way_places_expr="NULL AS number_of_each_way_places",
        )
        assert "NULL AS each_way_divisor" in sql
        assert "NULL AS number_of_each_way_places" in sql
        # Unsubstituted placeholders would leave {curly braces} in the output.
        assert "{" not in sql and "}" not in sql


# ── Extract_date progress callback tests ─────────────────────────────────────

class TestExtractDateProgressCallback:
    """DataExtractor.extract_date must invoke on_progress at each sub-step so
    the admin restore wizard can render live sub-progress and log messages."""

    def test_on_progress_called_in_order_for_happy_path(self, minimal_config):
        ticks = _make_ticks_df(n=3)
        runners = _make_runners_df(["1.0", "1.1", "1.2"])
        extractor = _make_extractor(minimal_config, ticks_df=ticks, runners_df=runners)
        extractor._engine.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        extractor._engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        events: list[tuple[int, int, str]] = []
        extractor.extract_date(
            date(2026, 3, 1),
            on_progress=lambda step, total, msg: events.append((step, total, msg)),
        )

        # Steps must be 1-indexed and strictly monotonic.
        steps = [e[0] for e in events]
        assert steps == sorted(steps)
        assert steps[0] >= 1
        # Total must be consistent across all reports.
        totals = {e[1] for e in events}
        assert len(totals) == 1
        # Final step should equal total (i.e. "writing parquet" fires).
        assert steps[-1] == events[-1][1]
        # Messages must be non-empty strings.
        assert all(isinstance(e[2], str) and e[2] for e in events)

    def test_on_progress_reports_short_circuit_for_empty_ticks(self, minimal_config):
        """When no ticks are found the extractor should still report *some*
        progress (the user needs to know the short-circuit happened), and it
        must not raise when on_progress is passed."""
        empty_ticks = _make_ticks_df(n=0)
        extractor = _make_extractor(minimal_config, ticks_df=empty_ticks)
        extractor._engine.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        extractor._engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        events: list[tuple[int, int, str]] = []
        result = extractor.extract_date(
            date(2026, 3, 1),
            on_progress=lambda step, total, msg: events.append((step, total, msg)),
        )
        assert result is False
        assert len(events) >= 1

    def test_on_progress_exception_does_not_break_extraction(self, minimal_config):
        """A misbehaving callback must not prevent the extract from completing
        (the admin.py callback runs on a thread boundary and any raise would
        otherwise surface as a thread crash)."""
        ticks = _make_ticks_df(n=1)
        runners = _make_runners_df(["1.0"])
        extractor = _make_extractor(minimal_config, ticks_df=ticks, runners_df=runners)
        extractor._engine.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        extractor._engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        def _boom(step: int, total: int, msg: str) -> None:
            raise RuntimeError("callback exploded")

        # Must NOT raise
        result = extractor.extract_date(date(2026, 3, 1), on_progress=_boom)
        assert result is True
        out = Path(minimal_config["paths"]["processed_data"])
        assert (out / "2026-03-01.parquet").exists()

    def test_on_progress_optional(self, minimal_config):
        """Backwards compatibility: calling extract_date without on_progress
        must still work (CLI / tests that don't care about progress)."""
        ticks = _make_ticks_df(n=1)
        runners = _make_runners_df(["1.0"])
        extractor = _make_extractor(minimal_config, ticks_df=ticks, runners_df=runners)
        extractor._engine.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        extractor._engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        result = extractor.extract_date(date(2026, 3, 1))  # no callback
        assert result is True
