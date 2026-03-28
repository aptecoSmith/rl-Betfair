"""Integration tests for Session 2.7a — PolledMarketSnapshots + RaceStatusEvents.

Tests against real extracted data (2026-03-26) and the live MySQL database.
Skip gracefully when DB or data isn't available.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml

from data.episode_builder import Day, load_day
from data.extractor import DataExtractor, TICKS_COLUMNS
from data.feature_engineer import RACE_STATUSES, engineer_race, market_tick_features
from env.betfair_env import BetfairEnv, MARKET_DIM, VELOCITY_DIM, RUNNER_DIM, AGENT_STATE_DIM

# ── Fixtures ──────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


@pytest.fixture(scope="module")
def config():
    if not CONFIG_PATH.exists():
        pytest.skip("config.yaml not found")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def extractor(config):
    """Create a DataExtractor connected to the real MySQL DB."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        ext = DataExtractor(config)
        # Test connection by listing dates
        ext.get_available_dates()
        return ext
    except Exception as e:
        pytest.skip(f"Cannot connect to MySQL: {e}")


@pytest.fixture(scope="module")
def real_day() -> Day:
    """Load the real 2026-03-26 day."""
    parquet_path = DATA_DIR / "2026-03-26.parquet"
    if not parquet_path.exists():
        pytest.skip("2026-03-26.parquet not found")
    return load_day("2026-03-26", data_dir=DATA_DIR)


# ── Auto-detect tests ────────────────────────────────────────────────────────


class TestAutoDetect:
    def test_has_polled_data_for_existing_date(self, extractor):
        """Check polled data detection on a date we know has legacy data."""
        # 2026-03-26 should exist in legacy but likely not in polled (yet)
        result = extractor.has_polled_data(date(2026, 3, 26))
        # Either True or False is fine — we just verify no error
        assert isinstance(result, bool)

    def test_get_available_dates_includes_legacy(self, extractor):
        """available_dates should include at least 2026-03-26 from legacy source."""
        dates = extractor.get_available_dates()
        assert date(2026, 3, 26) in dates

    def test_extract_date_legacy_fallback(self, extractor, tmp_path):
        """When polled data is absent, extraction should use legacy and succeed."""
        # Use a fresh output dir
        extractor._output_dir = tmp_path
        ok = extractor.extract_date(date(2026, 3, 26))
        assert ok is True

        # Verify output
        ticks_path = tmp_path / "2026-03-26.parquet"
        assert ticks_path.exists()
        df = pd.read_parquet(ticks_path)
        assert len(df) > 0

        # race_status column should exist (added by extract_date)
        assert "race_status" in df.columns


# ── Backward compatibility ────────────────────────────────────────────────────


class TestBackwardCompat:
    def test_load_day_works_without_race_status(self, real_day):
        """Old Parquet files without race_status should still load."""
        assert len(real_day.races) > 0
        for race in real_day.races:
            for tick in race.ticks:
                # Old data won't have race_status → should be None
                assert tick.race_status is None

    def test_feature_engineer_handles_none_race_status(self, real_day):
        """Feature engineering should work with race_status=None."""
        race = real_day.races[0]
        feats = engineer_race(race)
        assert len(feats) == len(race.ticks)

        # Check race status features are all 0.0
        for tick_feats in feats:
            mkt = tick_feats["market"]
            for s in RACE_STATUSES:
                key = f"race_status_{s.replace(' ', '_')}"
                assert key in mkt
                assert mkt[key] == 0.0

    def test_feature_engineer_has_time_since_status_change(self, real_day):
        """time_since_status_change should be present in velocity features."""
        race = real_day.races[0]
        feats = engineer_race(race)
        assert "time_since_status_change" in feats[0]["market_velocity"]

    def test_env_runs_full_episode_with_old_data(self, real_day, config):
        """Full episode on legacy data should work with new obs_dim."""
        env = BetfairEnv(real_day, config)
        obs, info = env.reset()

        expected_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * 14) + AGENT_STATE_DIM
        assert obs.shape == (expected_dim,)
        assert expected_dim == 1587  # was 1583 (Session 2.8: +4 market velocity)

        # Step through a few ticks
        import numpy as np
        for _ in range(min(10, len(real_day.races[0].ticks))):
            action = np.zeros(28, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
            assert obs.shape == (expected_dim,)
