"""Tests for the market_type_filter gene (Issue 04).

Covers env filtering, zero-race handling, and gene sampling.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
import pytest
import yaml

from data.episode_builder import Day, Race, Tick
from env.betfair_env import BetfairEnv

# Re-use the synthetic constructors from the env test suite.
from tests.test_betfair_env import _make_race, _make_day


# -- Helpers ------------------------------------------------------------------


def _make_race_with_type(
    market_id: str,
    market_type: str,
    offset_hours: int = 0,
) -> Race:
    """Build a race and stamp its market_type."""
    race = _make_race(market_id=market_id)
    # Offset start time so races in the same day don't collide.
    if offset_hours:
        delta = timedelta(hours=offset_hours)
        race.market_start_time = race.market_start_time + delta
        race.ticks = [
            Tick(
                market_id=t.market_id,
                timestamp=t.timestamp + delta,
                sequence_number=t.sequence_number,
                venue=t.venue,
                market_start_time=t.market_start_time + delta,
                number_of_active_runners=t.number_of_active_runners,
                traded_volume=t.traded_volume,
                in_play=t.in_play,
                winner_selection_id=t.winner_selection_id,
                race_status=t.race_status,
                temperature=t.temperature,
                precipitation=t.precipitation,
                wind_speed=t.wind_speed,
                wind_direction=t.wind_direction,
                humidity=t.humidity,
                weather_code=t.weather_code,
                runners=t.runners,
            )
            for t in race.ticks
        ]
    race.market_type = market_type
    return race


def _make_mixed_day() -> Day:
    """Day with 2 WIN and 2 EACH_WAY races."""
    return Day(
        date="2026-04-10",
        races=[
            _make_race_with_type("1.001", "WIN", offset_hours=0),
            _make_race_with_type("1.002", "EACH_WAY", offset_hours=1),
            _make_race_with_type("1.003", "WIN", offset_hours=2),
            _make_race_with_type("1.004", "EACH_WAY", offset_hours=3),
        ],
    )


@pytest.fixture
def config():
    return {
        "training": {
            "max_runners": 14,
            "starting_budget": 100.0,
            "max_bets_per_race": 20,
        },
        "actions": {
            "force_aggressive": True,
        },
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "efficiency_penalty": 0.01,
        },
    }


# -- Tests --------------------------------------------------------------------


class TestMarketTypeFilter:
    """Environment filtering by market type."""

    def test_win_filter(self, config):
        """filter=WIN → only WIN races survive."""
        day = _make_mixed_day()
        env = BetfairEnv(day, config, market_type_filter="WIN")
        assert len(env.day.races) == 2
        assert all(r.market_type == "WIN" for r in env.day.races)

    def test_ew_filter(self, config):
        """filter=EACH_WAY → only EW races survive."""
        day = _make_mixed_day()
        env = BetfairEnv(day, config, market_type_filter="EACH_WAY")
        assert len(env.day.races) == 2
        assert all(r.market_type == "EACH_WAY" for r in env.day.races)

    def test_both_keeps_all(self, config):
        """filter=BOTH → all 4 races."""
        day = _make_mixed_day()
        env = BetfairEnv(day, config, market_type_filter="BOTH")
        assert len(env.day.races) == 4

    def test_free_choice_keeps_all(self, config):
        """filter=FREE_CHOICE → all 4 races (same as BOTH)."""
        day = _make_mixed_day()
        env = BetfairEnv(day, config, market_type_filter="FREE_CHOICE")
        assert len(env.day.races) == 4

    def test_default_is_both(self, config):
        """No explicit filter → all races kept."""
        day = _make_mixed_day()
        env = BetfairEnv(day, config)
        assert len(env.day.races) == 4

    def test_zero_races_completes_gracefully(self, config):
        """WIN-only day with EW filter → 0 races, episode ends immediately."""
        day = Day(
            date="2026-04-10",
            races=[
                _make_race_with_type("1.001", "WIN", offset_hours=0),
                _make_race_with_type("1.002", "WIN", offset_hours=1),
            ],
        )
        env = BetfairEnv(day, config, market_type_filter="EACH_WAY")
        assert len(env.day.races) == 0

        obs, info = env.reset()
        # With zero races, first step should immediately terminate
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated or truncated
        assert info.get("day_pnl", 0.0) == 0.0

    def test_case_insensitive(self, config):
        """Filter value is case-insensitive."""
        day = _make_mixed_day()
        env = BetfairEnv(day, config, market_type_filter="win")
        assert len(env.day.races) == 2
        assert all(r.market_type == "WIN" for r in env.day.races)


class TestMarketTypeFilterGene:
    """Gene definition in config.yaml."""

    def test_gene_exists_in_config(self):
        """market_type_filter is defined in search_ranges."""
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        ranges = cfg["hyperparameters"]["search_ranges"]
        assert "market_type_filter" in ranges

    def test_gene_choices(self):
        """Gene has the correct choices."""
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        spec = cfg["hyperparameters"]["search_ranges"]["market_type_filter"]
        assert spec["type"] == "str_choice"
        assert set(spec["choices"]) == {"WIN", "EACH_WAY", "BOTH", "FREE_CHOICE"}

    def test_gene_sampled_by_population_manager(self):
        """PopulationManager samples market_type_filter as a gene."""
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        from agents.population_manager import parse_search_ranges, sample_hyperparams
        import random

        specs = parse_search_ranges(cfg["hyperparameters"]["search_ranges"])
        rng = random.Random(42)
        hp = sample_hyperparams(specs, rng)
        assert "market_type_filter" in hp
        assert hp["market_type_filter"] in ("WIN", "EACH_WAY", "BOTH", "FREE_CHOICE")
