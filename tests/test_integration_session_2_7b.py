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
def extracted_date(config, tmp_path_factory):
    """Extract 2026-03-28 to a temp dir and return the path."""
    from data.extractor import DataExtractor

    out = tmp_path_factory.mktemp("parquet")
    ext = DataExtractor(config, output_dir=out)
    ok = ext.extract_date(date(2026, 3, 28))
    assert ok, "Extraction failed — no polled data for 2026-03-28?"
    return out


class TestExtraction:
    def test_runners_parquet_has_past_races_json(self, extracted_date):
        runners = pd.read_parquet(extracted_date / "2026-03-28_runners.parquet")
        assert "past_races_json" in runners.columns
        assert "timeform_comment" in runners.columns
        assert "recent_form" in runners.columns

    def test_past_races_json_populated(self, extracted_date):
        runners = pd.read_parquet(extracted_date / "2026-03-28_runners.parquet")
        non_null = runners["past_races_json"].notna().sum()
        assert non_null > 0, "No past_races_json values populated"

    def test_timeform_comment_populated(self, extracted_date):
        runners = pd.read_parquet(extracted_date / "2026-03-28_runners.parquet")
        non_null = runners["timeform_comment"].notna().sum()
        assert non_null > 0


class TestEpisodeBuilder:
    def test_past_races_populated(self, extracted_date):
        from data.episode_builder import load_day

        day = load_day("2026-03-28", data_dir=extracted_date)
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
        assert found_past_races, "No runners with past races found"

    def test_timeform_comment_loaded(self, extracted_date):
        from data.episode_builder import load_day

        day = load_day("2026-03-28", data_dir=extracted_date)
        found = False
        for race in day.races:
            for meta in race.runner_metadata.values():
                if meta.timeform_comment:
                    found = True
                    break
            if found:
                break
        assert found, "No runners with timeform_comment found"


class TestFeatureEngineer:
    def test_pr_features_populated(self, extracted_date):
        from data.episode_builder import load_day
        from data.feature_engineer import TickHistory, engineer_tick

        day = load_day("2026-03-28", data_dir=extracted_date)
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

        day = load_day("2026-03-28", data_dir=extracted_date)
        env = BetfairEnv(day, config)
        obs, _ = env.reset()
        assert obs.shape == (1587,)
        assert np.isnan(obs).sum() == 0

        # Run 20 steps
        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        assert obs.shape == (1587,)
        assert np.isnan(obs).sum() == 0
