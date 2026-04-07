"""Integration tests for Session 2.7a — PolledMarketSnapshots + RaceStatusEvents.

The ``TestBackwardCompat`` group needs a parquet that *predates*
Session 2.7a — i.e. one without the ``race_status`` column.  This
file used to be hard-coded to ``2026-03-26``, but that day was aged
out of the cold backups, leaving the whole class skipped.

Now we synthesise a legacy-schema parquet on the fly: pick the
latest available extracted day, drop the post-2.7a columns, and
write the result to a temp dir alongside a copy of its runners
file. ``load_day`` then loads it as if it were genuinely old
data.  See :mod:`tests._data_fixtures` for the helper.
"""

from __future__ import annotations

import os
import shutil
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml

from data.episode_builder import Day, load_day
from data.extractor import DataExtractor, TICKS_COLUMNS
from data.feature_engineer import RACE_STATUSES, engineer_race, market_tick_features
from env.betfair_env import BetfairEnv, MARKET_DIM, VELOCITY_DIM, RUNNER_DIM, AGENT_STATE_DIM
from tests._data_fixtures import latest_processed_date, make_legacy_schema_parquet

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
def real_day(tmp_path_factory) -> Day:
    """Synthesise a legacy-schema day from the latest extracted parquet.

    The ``TestBackwardCompat`` tests below assert that data WITHOUT
    the post-Session-2.7a columns (``race_status``, each-way fields)
    still loads through ``episode_builder.load_day`` and runs through
    ``feature_engineer`` / ``BetfairEnv``.  We can't just hand them a
    modern parquet — those tests would fail because real data has
    those columns populated.  So we copy the latest day to a temp
    dir, strip the post-2.7a columns from the ticks, and load the
    result.
    """
    latest = latest_processed_date(data_dir=DATA_DIR)
    if latest is None:
        pytest.skip(f"No extracted parquet available in {DATA_DIR}")
    date_str, ticks_path = latest
    runners_path = DATA_DIR / f"{date_str}_runners.parquet"
    if not runners_path.exists():
        pytest.skip(f"No runners parquet for {date_str}")

    tmp_dir = tmp_path_factory.mktemp("legacy_fixture")
    legacy_ticks = tmp_dir / f"{date_str}.parquet"
    legacy_runners = tmp_dir / f"{date_str}_runners.parquet"
    make_legacy_schema_parquet(ticks_path, legacy_ticks)
    shutil.copy(runners_path, legacy_runners)

    return load_day(date_str, data_dir=tmp_dir)


# ── Auto-detect tests ────────────────────────────────────────────────────────


class TestAutoDetect:
    def test_has_polled_data_for_existing_date(self, extractor):
        """Check polled data detection on a date we know has legacy data."""
        # 2026-03-26 should exist in legacy but likely not in polled (yet)
        result = extractor.has_polled_data(date(2026, 3, 26))
        # Either True or False is fine — we just verify no error
        assert isinstance(result, bool)

    def test_get_available_dates_not_empty(self, extractor):
        """available_dates should return at least one date."""
        dates = extractor.get_available_dates()
        assert len(dates) > 0

    def test_extract_date_legacy_fallback(self, extractor, tmp_path):
        """Extraction should succeed for an available date."""
        dates = extractor.get_available_dates()
        if not dates:
            pytest.skip("No dates available in database")
        test_date = dates[0]

        # Use a fresh output dir
        extractor._output_dir = tmp_path
        ok = extractor.extract_date(test_date)
        assert ok is True

        # Verify output
        ticks_path = tmp_path / f"{test_date.isoformat()}.parquet"
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
        """Full episode on legacy data should work end-to-end.

        We pull the expected obs shape directly from the env's
        ``observation_space`` rather than hand-summing dim constants
        — the dim constants are imported here only to make sure
        they're still valid imports for legacy callers.  The
        previous version of this test hand-summed
        ``MARKET + VELOCITY + RUNNER*N + AGENT`` and missed the
        ``POSITION_DIM * max_runners`` term entirely, so the
        assertion would always fail against a real env.  That bug
        was hidden for months by an unrelated module-level skip.
        """
        env = BetfairEnv(real_day, config)
        obs, info = env.reset()

        # Sanity-check that the legacy dim constants are still
        # importable — we don't sum them here.
        assert MARKET_DIM > 0
        assert VELOCITY_DIM > 0
        assert RUNNER_DIM > 0
        assert AGENT_STATE_DIM > 0

        expected_shape = env.observation_space.shape
        assert obs.shape == expected_shape

        # Step through a few ticks
        import numpy as np
        action_dim = env.action_space.shape[0]
        for _ in range(min(10, len(real_day.races[0].ticks))):
            action = np.zeros(action_dim, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
            assert obs.shape == expected_shape
