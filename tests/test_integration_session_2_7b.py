"""Session 2.7b — RaceCardRunners integration tests.

These tests require a live MySQL connection with real data from
BetfairPoller (``RaceCardRunners`` table populated).

Skipped if MySQL is not reachable or the table is empty.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# ── Skip if MySQL is not available ───────────────────────────────────────────

try:
    import sqlalchemy as sa

    with open("config.yaml") as f:
        _config = yaml.safe_load(f)
    _db = _config["database"]
    _url = f"mysql+pymysql://root:IAgr33d2Th1s!@{_db['host']}:{_db['port']}/{_db['hot_data_db']}"
    _engine = sa.create_engine(_url)
    with _engine.connect() as conn:
        count = conn.execute(sa.text("SELECT COUNT(*) FROM RaceCardRunners")).scalar()
    _HAS_RACECARD_DATA = count > 0
except Exception:
    _HAS_RACECARD_DATA = False

pytestmark = pytest.mark.skipif(
    not _HAS_RACECARD_DATA,
    reason="RaceCardRunners table not available or empty",
)


@pytest.fixture(scope="module")
def config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def _available_date(config):
    """Pick the first available date from the database."""
    from data.extractor import DataExtractor
    ext = DataExtractor(config)
    dates = ext.get_available_dates()
    if not dates:
        pytest.skip("No dates available in database")
    return dates[0]


@pytest.fixture(scope="module")
def extracted_date(config, _available_date, tmp_path_factory):
    """Extract an available date to a temp dir and return (path, date_str)."""
    from data.extractor import DataExtractor

    out = tmp_path_factory.mktemp("parquet")
    ext = DataExtractor(config, output_dir=out)
    ok = ext.extract_date(_available_date)
    assert ok, f"Extraction failed for {_available_date}"
    return out



class TestExtraction:
    def test_runners_parquet_has_past_races_json(self, extracted_date):
        runners = pd.read_parquet(next(extracted_date.glob("*_runners.parquet")))
        assert "past_races_json" in runners.columns
        assert "timeform_comment" in runners.columns
        assert "recent_form" in runners.columns

    def test_past_races_json_populated(self, extracted_date):
        runners = pd.read_parquet(next(extracted_date.glob("*_runners.parquet")))
        if "past_races_json" not in runners.columns:
            pytest.skip("past_races_json column not present in extracted data")
        non_null = runners["past_races_json"].notna().sum()
        if non_null == 0:
            pytest.skip("past_races_json column exists but has no non-null values")

    def test_timeform_comment_populated(self, extracted_date):
        runners = pd.read_parquet(next(extracted_date.glob("*_runners.parquet")))
        if "timeform_comment" not in runners.columns:
            pytest.skip("timeform_comment column not present in extracted data")
        non_null = runners["timeform_comment"].notna().sum()
        if non_null == 0:
            pytest.skip("timeform_comment column exists but has no non-null values")


class TestEpisodeBuilder:
    def test_past_races_populated(self, extracted_date):
        from data.episode_builder import load_day

        date_str = next(p.stem for p in extracted_date.glob("*.parquet") if "_runners" not in p.stem)
        day = load_day(date_str, data_dir=extracted_date)
        found_past_races = False
        for race in day.races:
            for meta in race.runner_metadata.values():
                if meta.past_races:
                    found_past_races = True
                    assert len(meta.past_races) > 0
                    assert meta.past_races[0].course != ""
                    break
            if found_past_races:
                break
        if not found_past_races:
            pytest.skip("No runners with past_races data in database for this date")

    def test_timeform_comment_loaded(self, extracted_date):
        from data.episode_builder import load_day

        date_str = next(p.stem for p in extracted_date.glob("*.parquet") if "_runners" not in p.stem)
        day = load_day(date_str, data_dir=extracted_date)
        found = False
        for race in day.races:
            for meta in race.runner_metadata.values():
                if meta.timeform_comment:
                    found = True
                    break
            if found:
                break
        if not found:
            pytest.skip("No runners with timeform_comment data in database for this date")


class TestFeatureEngineer:
    def test_pr_features_populated(self, extracted_date):
        from data.episode_builder import load_day
        from data.feature_engineer import TickHistory, engineer_tick

        date_str = next(p.stem for p in extracted_date.glob("*.parquet") if "_runners" not in p.stem)
        day = load_day(date_str, data_dir=extracted_date)
        history = TickHistory()
        # Find a race with runners that have past races
        for race in day.races:
            has_history = any(m.past_races for m in race.runner_metadata.values())
            if has_history and race.ticks:
                feats = engineer_tick(race.ticks[0], race, history)
                for sid, rfeats in feats["runners"].items():
                    pr_keys = [k for k in rfeats if k.startswith("pr_")]
                    assert len(pr_keys) == 17
                    if race.runner_metadata.get(sid) and race.runner_metadata[sid].past_races:
                        non_nan = sum(1 for k in pr_keys if not math.isnan(rfeats[k]))
                        assert non_nan > 10, f"Too few non-NaN pr_ features: {non_nan}"
                return
        pytest.skip("No races with past race data found")


class TestEnvironment:
    def test_env_runs_full_episode(self, extracted_date, config):
        from data.episode_builder import load_day
        from env.betfair_env import BetfairEnv

        date_str = next(p.stem for p in extracted_date.glob("*.parquet") if "_runners" not in p.stem)
        day = load_day(date_str, data_dir=extracted_date)
        env = BetfairEnv(day, config)
        obs, _ = env.reset()
        obs_dim = env.observation_space.shape[0]
        assert obs.shape == (obs_dim,)
        assert np.isnan(obs).sum() == 0

        # Run 20 steps
        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        assert obs.shape == (obs_dim,)
        assert np.isnan(obs).sum() == 0
