"""Tests for Session 2.7a — PolledMarketSnapshots + RaceStatusEvents.

Covers:
- Polled RunnersJson → SnapJson normalisation
- Race status join (as-of merge)
- Extractor auto-detect (polled vs legacy)
- Episode builder race_status field
- Feature engineer race status one-hot + time_since_status_change
- Env obs_dim updated correctly
- Backward compatibility with old Parquet files (no race_status column)
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data.episode_builder import Day, PriceSize, Race, RunnerMeta, RunnerSnap, Tick, load_day, parse_snap_json
from data.extractor import (
    TICKS_COLUMNS,
    DataExtractor,
    _join_race_status,
    _polled_runners_to_snap_json,
)
from data.feature_engineer import (
    RACE_STATUSES,
    TickHistory,
    engineer_tick,
    market_tick_features,
)
from env.betfair_env import (
    AGENT_STATE_DIM,
    MARKET_DIM,
    MARKET_KEYS,
    MARKET_VELOCITY_KEYS,
    RUNNER_DIM,
    VELOCITY_DIM,
    BetfairEnv,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _sample_polled_runners_json() -> str:
    """Return a sample RunnersJson string in the polled format."""
    return json.dumps([
        {
            "selectionId": 12345,
            "handicap": 0.0,
            "state": {
                "adjustmentFactor": 10.5,
                "sortPriority": 1,
                "lastPriceTraded": 3.5,
                "totalMatched": 50000.0,
                "status": "ACTIVE",
            },
            "exchange": {
                "availableToBack": [
                    {"price": 3.4, "size": 100.0},
                    {"price": 3.3, "size": 200.0},
                ],
                "availableToLay": [
                    {"price": 3.55, "size": 50.0},
                    {"price": 3.6, "size": 75.0},
                ],
            },
        },
        {
            "selectionId": 67890,
            "handicap": 0.0,
            "state": {
                "adjustmentFactor": 8.2,
                "sortPriority": 2,
                "lastPriceTraded": 6.0,
                "totalMatched": 25000.0,
                "status": "ACTIVE",
            },
            "exchange": {
                "availableToBack": [{"price": 5.8, "size": 80.0}],
                "availableToLay": [{"price": 6.2, "size": 40.0}],
            },
        },
    ])


def _make_tick(race_status: str | None = None) -> Tick:
    """Create a minimal Tick with optional race_status."""
    return Tick(
        market_id="1.100",
        timestamp=datetime(2026, 3, 27, 13, 55),
        sequence_number=1,
        venue="Newmarket",
        market_start_time=datetime(2026, 3, 27, 14, 0),
        number_of_active_runners=2,
        traded_volume=10000.0,
        in_play=False,
        winner_selection_id=None,
        race_status=race_status,
        temperature=15.0,
        precipitation=0.0,
        wind_speed=5.0,
        wind_direction=180.0,
        humidity=60.0,
        weather_code=0,
        runners=[
            RunnerSnap(
                selection_id=101, status="ACTIVE",
                last_traded_price=3.5, total_matched=5000.0,
                starting_price_near=0.0, starting_price_far=0.0,
                adjustment_factor=10.0, bsp=None, sort_priority=1,
                removal_date=None,
                available_to_back=[PriceSize(3.4, 100.0)],
                available_to_lay=[PriceSize(3.6, 50.0)],
            ),
        ],
    )


# ── Polled RunnersJson → SnapJson conversion ──────────────────────────────────


class TestPolledRunnersToSnapJson:
    def test_converts_to_market_runners_format(self):
        result = json.loads(_polled_runners_to_snap_json(_sample_polled_runners_json()))
        assert "MarketRunners" in result
        assert len(result["MarketRunners"]) == 2

    def test_selection_id_mapped(self):
        result = json.loads(_polled_runners_to_snap_json(_sample_polled_runners_json()))
        r0 = result["MarketRunners"][0]
        assert r0["RunnerId"]["SelectionId"] == 12345

    def test_status_mapped(self):
        result = json.loads(_polled_runners_to_snap_json(_sample_polled_runners_json()))
        r0 = result["MarketRunners"][0]
        assert r0["Definition"]["Status"] == "ACTIVE"

    def test_ltp_mapped(self):
        result = json.loads(_polled_runners_to_snap_json(_sample_polled_runners_json()))
        r0 = result["MarketRunners"][0]
        assert r0["Prices"]["LastTradedPrice"] == 3.5

    def test_total_matched_mapped(self):
        result = json.loads(_polled_runners_to_snap_json(_sample_polled_runners_json()))
        r0 = result["MarketRunners"][0]
        assert r0["Prices"]["TradedVolume"] == 50000.0

    def test_adjustment_factor_mapped(self):
        result = json.loads(_polled_runners_to_snap_json(_sample_polled_runners_json()))
        r0 = result["MarketRunners"][0]
        assert r0["Definition"]["AdjustmentFactor"] == 10.5

    def test_sort_priority_mapped(self):
        result = json.loads(_polled_runners_to_snap_json(_sample_polled_runners_json()))
        r0 = result["MarketRunners"][0]
        assert r0["Definition"]["SortPriority"] == 1

    def test_back_prices_mapped(self):
        result = json.loads(_polled_runners_to_snap_json(_sample_polled_runners_json()))
        r0 = result["MarketRunners"][0]
        backs = r0["Prices"]["AvailableToBack"]
        assert len(backs) == 2
        assert backs[0]["Price"] == 3.4
        assert backs[0]["Size"] == 100.0

    def test_lay_prices_mapped(self):
        result = json.loads(_polled_runners_to_snap_json(_sample_polled_runners_json()))
        r0 = result["MarketRunners"][0]
        lays = r0["Prices"]["AvailableToLay"]
        assert len(lays) == 2
        assert lays[0]["Price"] == 3.55

    def test_null_input(self):
        result = json.loads(_polled_runners_to_snap_json(None))
        assert result["MarketRunners"] == []

    def test_empty_string_input(self):
        result = json.loads(_polled_runners_to_snap_json(""))
        assert result["MarketRunners"] == []

    def test_invalid_json_input(self):
        result = json.loads(_polled_runners_to_snap_json("{bad json"))
        assert result["MarketRunners"] == []

    def test_non_list_input(self):
        result = json.loads(_polled_runners_to_snap_json('{"not": "a list"}'))
        assert result["MarketRunners"] == []

    def test_parse_snap_json_reads_converted_output(self):
        """Verify parse_snap_json correctly parses the converted SnapJson."""
        snap_json = _polled_runners_to_snap_json(_sample_polled_runners_json())
        runners = parse_snap_json(snap_json)
        assert len(runners) == 2
        assert runners[0].selection_id == 12345
        assert runners[0].last_traded_price == 3.5
        assert runners[0].total_matched == 50000.0
        assert runners[0].status == "ACTIVE"
        assert runners[0].adjustment_factor == 10.5
        assert runners[0].sort_priority == 1
        assert len(runners[0].available_to_back) == 2
        assert len(runners[0].available_to_lay) == 2
        assert runners[0].available_to_back[0].price == 3.4
        assert runners[0].available_to_lay[0].price == 3.55

    def test_missing_state_fields_default(self):
        """Runners with missing state fields should get defaults."""
        raw = json.dumps([{"selectionId": 999}])
        snap = _polled_runners_to_snap_json(raw)
        runners = parse_snap_json(snap)
        assert len(runners) == 1
        assert runners[0].selection_id == 999
        assert runners[0].status == "ACTIVE"
        assert runners[0].last_traded_price == 0.0


# ── Race status join ──────────────────────────────────────────────────────────


class TestRaceStatusJoin:
    def test_join_adds_race_status_column(self):
        """As-of join should add race_status to ticks."""
        ticks = pd.DataFrame({
            "market_id": ["1.100", "1.100", "1.100"],
            "timestamp": pd.to_datetime([
                "2026-03-27 13:50:00",
                "2026-03-27 13:55:00",
                "2026-03-27 13:58:00",
            ]),
            "sequence_number": [1, 2, 3],
        })
        events = pd.DataFrame({
            "market_id": ["1.100", "1.100"],
            "timestamp": pd.to_datetime([
                "2026-03-27 13:52:00",
                "2026-03-27 13:56:00",
            ]),
            "status": ["parading", "going down"],
        })

        conn = MagicMock()
        result_mock = MagicMock()
        result_mock.fetchall.return_value = list(events.itertuples(index=False, name=None))
        result_mock.keys.return_value = ["market_id", "timestamp", "status"]
        conn.execute.return_value = result_mock

        from datetime import date
        merged = _join_race_status(ticks, date(2026, 3, 27), conn)
        assert "race_status" in merged.columns
        # Tick 1 at 13:50 → before any event → None
        assert pd.isna(merged.iloc[0]["race_status"])
        # Tick 2 at 13:55 → after "parading" at 13:52 → parading
        assert merged.iloc[1]["race_status"] == "parading"
        # Tick 3 at 13:58 → after "going down" at 13:56 → going down
        assert merged.iloc[2]["race_status"] == "going down"

    def test_join_handles_no_events(self):
        ticks = pd.DataFrame({
            "market_id": ["1.100"],
            "timestamp": pd.to_datetime(["2026-03-27 13:50:00"]),
            "sequence_number": [1],
        })
        conn = MagicMock()
        result_mock = MagicMock()
        result_mock.fetchall.return_value = []
        result_mock.keys.return_value = ["market_id", "timestamp", "status"]
        conn.execute.return_value = result_mock

        from datetime import date
        merged = _join_race_status(ticks, date(2026, 3, 27), conn)
        assert "race_status" in merged.columns
        assert merged["race_status"].isna().all()

    def test_join_handles_empty_ticks(self):
        ticks = pd.DataFrame(columns=["market_id", "timestamp", "sequence_number"])
        conn = MagicMock()
        from datetime import date
        merged = _join_race_status(ticks, date(2026, 3, 27), conn)
        assert "race_status" in merged.columns

    def test_join_handles_multiple_markets(self):
        ticks = pd.DataFrame({
            "market_id": ["1.100", "1.200"],
            "timestamp": pd.to_datetime([
                "2026-03-27 13:55:00",
                "2026-03-27 14:55:00",
            ]),
            "sequence_number": [1, 1],
        })
        events = pd.DataFrame({
            "market_id": ["1.100", "1.200"],
            "timestamp": pd.to_datetime([
                "2026-03-27 13:50:00",
                "2026-03-27 14:50:00",
            ]),
            "status": ["parading", "under orders"],
        })

        conn = MagicMock()
        result_mock = MagicMock()
        result_mock.fetchall.return_value = list(events.itertuples(index=False, name=None))
        result_mock.keys.return_value = ["market_id", "timestamp", "status"]
        conn.execute.return_value = result_mock

        from datetime import date
        merged = _join_race_status(ticks, date(2026, 3, 27), conn)
        mkt100 = merged[merged["market_id"] == "1.100"].iloc[0]
        mkt200 = merged[merged["market_id"] == "1.200"].iloc[0]
        assert mkt100["race_status"] == "parading"
        assert mkt200["race_status"] == "under orders"


# ── Extractor auto-detect ────────────────────────────────────────────────────


class TestExtractorAutoDetect:
    @pytest.fixture()
    def config(self, tmp_path):
        return {
            "database": {
                "host": "localhost", "port": 3306,
                "cold_data_db": "coldData", "hot_data_db": "hotDataRefactored",
            },
            "paths": {"processed_data": str(tmp_path / "processed")},
        }

    def test_has_polled_data_returns_false_when_no_table(self, config):
        mock_engine = MagicMock()
        extractor = DataExtractor(config, engine=mock_engine)
        # Simulate table not existing
        conn = MagicMock()
        conn.execute.side_effect = Exception("Table doesn't exist")
        assert extractor._has_polled_date(pd.Timestamp("2026-03-27").date(), conn) is False

    def test_has_polled_data_returns_false_when_no_rows(self, config):
        mock_engine = MagicMock()
        extractor = DataExtractor(config, engine=mock_engine)
        conn = MagicMock()
        result = MagicMock()
        result.fetchone.return_value = None
        conn.execute.return_value = result
        assert extractor._has_polled_date(pd.Timestamp("2026-03-27").date(), conn) is False

    def test_has_polled_data_returns_true_when_rows_exist(self, config):
        mock_engine = MagicMock()
        extractor = DataExtractor(config, engine=mock_engine)
        conn = MagicMock()
        result = MagicMock()
        result.fetchone.return_value = (1,)
        conn.execute.return_value = result
        assert extractor._has_polled_date(pd.Timestamp("2026-03-27").date(), conn) is True

    def test_extract_date_adds_race_status_none_for_legacy(self, config):
        """Legacy path should produce race_status=None in output."""
        mock_engine = MagicMock()
        extractor = DataExtractor(config, engine=mock_engine)
        extractor._has_polled_date = MagicMock(return_value=False)

        ticks_df = pd.DataFrame({
            "market_id": ["1.100"],
            "timestamp": pd.to_datetime(["2026-03-27 13:50:00"]),
            "sequence_number": [1],
            "venue": ["Newmarket"],
            "market_start_time": pd.to_datetime(["2026-03-27 14:00:00"]),
            "market_type": ["WIN"],
            "number_of_active_runners": [8],
            "traded_volume": [5000.0],
            "in_play": [False],
            "snap_json": ['{"MarketRunners": []}'],
            "winner_selection_id": pd.array([None], dtype=pd.Int64Dtype()),
            "temperature": [15.0],
            "precipitation": [0.0],
            "wind_speed": [5.0],
            "wind_direction": [180.0],
            "humidity": [60.0],
            "weather_code": pd.array([0], dtype=pd.Int32Dtype()),
        })
        from data.extractor import _cast_ticks
        extractor._query_ticks = MagicMock(return_value=_cast_ticks(ticks_df))
        extractor._query_runners = MagicMock(return_value=pd.DataFrame())
        extractor._query_market_names = MagicMock(
            return_value=pd.DataFrame(columns=["market_id", "market_name"])
        )

        from datetime import date
        ok = extractor.extract_date(date(2026, 3, 27))
        assert ok is True

        out_path = Path(config["paths"]["processed_data"]) / "2026-03-27.parquet"
        result = pd.read_parquet(out_path)
        assert "race_status" in result.columns
        assert result["race_status"].isna().all()


# ── Episode builder race_status ──────────────────────────────────────────────


class TestTickRaceStatus:
    def test_tick_has_race_status_field(self):
        tick = _make_tick(race_status="parading")
        assert tick.race_status == "parading"

    def test_tick_race_status_none_by_default(self):
        tick = _make_tick()
        assert tick.race_status is None

    def test_tick_race_status_all_values(self):
        for status in RACE_STATUSES:
            tick = _make_tick(race_status=status)
            assert tick.race_status == status

    def test_load_day_backward_compat(self, tmp_path):
        """Loading a Parquet file without race_status column should work."""
        # Create a minimal Parquet without race_status
        df = pd.DataFrame({
            "market_id": ["1.100"],
            "timestamp": pd.to_datetime(["2026-03-27 13:50:00"]),
            "sequence_number": [1],
            "venue": ["Newmarket"],
            "market_start_time": pd.to_datetime(["2026-03-27 14:00:00"]),
            "market_type": ["WIN"],
            "market_name": ["Test Race"],
            "number_of_active_runners": [1],
            "traded_volume": [5000.0],
            "in_play": [False],
            "snap_json": ['{"MarketRunners": [{"RunnerId": {"SelectionId": 101}, '
                          '"Definition": {"Status": "ACTIVE"}, '
                          '"Prices": {"LastTradedPrice": 3.5, "TradedVolume": 1000, '
                          '"StartingPriceNear": 0, "StartingPriceFar": 0, '
                          '"AvailableToBack": [{"Price": 3.4, "Size": 100}], '
                          '"AvailableToLay": [{"Price": 3.6, "Size": 50}]}}]}'],
            "winner_selection_id": pd.array([None], dtype=pd.Int64Dtype()),
            "temperature": [15.0],
            "precipitation": [0.0],
            "wind_speed": [5.0],
            "wind_direction": [180.0],
            "humidity": [60.0],
            "weather_code": pd.array([0], dtype=pd.Int32Dtype()),
        })
        df.to_parquet(tmp_path / "2026-03-27.parquet", index=False)

        day = load_day("2026-03-27", data_dir=tmp_path)
        assert len(day.races) == 1
        assert day.races[0].ticks[0].race_status is None


# ── Feature engineer race status features ──────────────────────────────────


class TestRaceStatusFeatures:
    def test_race_status_one_hot_all_zeros_when_none(self):
        tick = _make_tick(race_status=None)
        feats = market_tick_features(tick)
        for s in RACE_STATUSES:
            key = f"race_status_{s.replace(' ', '_')}"
            assert feats[key] == 0.0, f"Expected 0.0 for {key}"

    def test_race_status_one_hot_parading(self):
        tick = _make_tick(race_status="parading")
        feats = market_tick_features(tick)
        assert feats["race_status_parading"] == 1.0
        assert feats["race_status_going_down"] == 0.0
        assert feats["race_status_off"] == 0.0

    def test_race_status_one_hot_under_orders(self):
        tick = _make_tick(race_status="under orders")
        feats = market_tick_features(tick)
        assert feats["race_status_under_orders"] == 1.0
        assert feats["race_status_parading"] == 0.0

    def test_race_status_one_hot_off(self):
        tick = _make_tick(race_status="off")
        feats = market_tick_features(tick)
        assert feats["race_status_off"] == 1.0

    def test_race_status_case_insensitive(self):
        tick = _make_tick(race_status="PARADING")
        feats = market_tick_features(tick)
        assert feats["race_status_parading"] == 1.0

    def test_all_six_statuses_produce_correct_one_hot(self):
        for status in RACE_STATUSES:
            tick = _make_tick(race_status=status)
            feats = market_tick_features(tick)
            key = f"race_status_{status.replace(' ', '_')}"
            assert feats[key] == 1.0
            other_keys = [
                f"race_status_{s.replace(' ', '_')}"
                for s in RACE_STATUSES if s != status
            ]
            for ok in other_keys:
                assert feats[ok] == 0.0

    def test_race_statuses_constant(self):
        assert len(RACE_STATUSES) == 6
        assert "parading" in RACE_STATUSES
        assert "off" in RACE_STATUSES


class TestTimeSinceStatusChange:
    def test_initial_time_since_status_change_is_zero(self):
        history = TickHistory()
        tick = _make_tick(race_status="parading")
        mkt_feats = market_tick_features(tick)
        history.update(tick, mkt_feats)
        vel = history.market_velocity_features()
        assert vel["time_since_status_change"] == 0.0

    def test_time_since_increases_with_ticks(self):
        history = TickHistory()
        tick = _make_tick(race_status="parading")
        mkt_feats = market_tick_features(tick)

        # Update 5 times with same status
        for _ in range(5):
            history.update(tick, mkt_feats)

        vel = history.market_velocity_features()
        # 4 ticks since last change (change was on tick 1, now on tick 5)
        expected = 4 * 5.0 / 1800.0
        assert abs(vel["time_since_status_change"] - expected) < 1e-6

    def test_time_resets_on_status_change(self):
        history = TickHistory()
        tick1 = _make_tick(race_status="parading")
        tick2 = _make_tick(race_status="going down")
        mkt_feats = market_tick_features(tick1)

        history.update(tick1, mkt_feats)
        history.update(tick1, mkt_feats)
        history.update(tick2, mkt_feats)
        vel = history.market_velocity_features()
        # Just changed on this tick → 0
        assert vel["time_since_status_change"] == 0.0

    def test_time_clamped_at_one(self):
        history = TickHistory()
        tick = _make_tick(race_status="parading")
        mkt_feats = market_tick_features(tick)

        # 400 ticks * 5s = 2000s > 1800s → clamped at 1.0
        for _ in range(400):
            history.update(tick, mkt_feats)
        vel = history.market_velocity_features()
        assert vel["time_since_status_change"] == 1.0

    def test_reset_clears_status_tracking(self):
        history = TickHistory()
        tick = _make_tick(race_status="parading")
        mkt_feats = market_tick_features(tick)
        history.update(tick, mkt_feats)
        history.update(tick, mkt_feats)

        history.reset()
        tick2 = _make_tick(race_status="going down")
        history.update(tick2, mkt_feats)
        vel = history.market_velocity_features()
        assert vel["time_since_status_change"] == 0.0


# ── Environment dimension updates ────────────────────────────────────────────


class TestEnvDimensions:
    def test_market_dim_includes_race_status(self):
        assert MARKET_DIM == 37  # 25 + 6 race status + 6 market type/each-way

    def test_velocity_dim_includes_time_since_change(self):
        assert VELOCITY_DIM == 11  # 6 + 1 time_since_status_change + 4 market velocity (Session 2.8)

    def test_market_keys_contain_race_status(self):
        for s in RACE_STATUSES:
            key = f"race_status_{s.replace(' ', '_')}"
            assert key in MARKET_KEYS, f"{key} missing from MARKET_KEYS"

    def test_velocity_keys_contain_time_since_change(self):
        assert "time_since_status_change" in MARKET_VELOCITY_KEYS

    def test_obs_dim_correct(self):
        from env.betfair_env import POSITION_DIM
        obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * 14) + AGENT_STATE_DIM + (POSITION_DIM * 14)
        assert obs_dim == 1636  # +6 market type / each-way features

    def test_env_backward_compat_with_none_race_status(self):
        """Env should handle ticks with race_status=None gracefully."""
        tick = _make_tick(race_status=None)
        race = Race(
            market_id="1.100",
            venue="Newmarket",
            market_start_time=datetime(2026, 3, 27, 14, 0),
            winner_selection_id=None,
            ticks=[tick],
            runner_metadata={},
        )
        day = Day(date="2026-03-27", races=[race])
        config = {
            "training": {"max_runners": 14, "starting_budget": 100.0, "max_bets_per_race": 20},
            "reward": {
                "early_pick_bonus_min": 1.2, "early_pick_bonus_max": 1.5,
                "early_pick_min_seconds": 300, "efficiency_penalty": 0.01,
                "commission": 0.05,
            },
        }
        env = BetfairEnv(day, config)
        obs, info = env.reset()
        assert obs.shape == (1636,)  # +6 market type / each-way features
        # Race status features should all be 0
        # Market features start at index 0, race status at indices 25-30
        for i in range(25, 31):
            assert obs[i] == 0.0, f"obs[{i}] should be 0.0 for None race_status"


# ── TICKS_COLUMNS updated ────────────────────────────────────────────────────


class TestTicksColumnsUpdated:
    def test_race_status_in_ticks_columns(self):
        assert "race_status" in TICKS_COLUMNS

    def test_ticks_columns_count(self):
        # 19 prior + each_way_divisor + number_of_each_way_places
        assert len(TICKS_COLUMNS) == 21
