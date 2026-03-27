"""Integration tests for population manager with real config and data.

These tests initialise a population from the real config.yaml and verify
that every agent can produce a forward pass on real extracted data.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from agents.population_manager import PopulationManager, validate_hyperparams, parse_search_ranges
from data.episode_builder import load_day
from data.feature_engineer import engineer_day
from env.betfair_env import (
    AGENT_STATE_DIM,
    MARKET_DIM,
    RUNNER_DIM,
    VELOCITY_DIM,
    BetfairEnv,
)
from registry.model_store import ModelStore

pytestmark = pytest.mark.integration

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


@pytest.fixture(scope="module")
def real_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_real_date() -> str | None:
    if not PROCESSED_DIR.exists():
        return None
    for f in sorted(PROCESSED_DIR.glob("*_runners.parquet")):
        date_str = f.stem.replace("_runners", "")
        ticks_file = PROCESSED_DIR / f"{date_str}.parquet"
        if ticks_file.exists():
            return date_str
    return None


@pytest.fixture(scope="module")
def real_date() -> str:
    date = _find_real_date()
    if date is None:
        pytest.skip("No real extracted data available")
    return date


@pytest.fixture(scope="module")
def real_day(real_date: str):
    day = load_day(str(PROCESSED_DIR), real_date)
    return engineer_day(day)


class TestPopulationFromRealConfig:
    def test_population_initialises_correct_size(self, real_config: dict):
        pm = PopulationManager(real_config, model_store=None)
        agents = pm.initialise_population(generation=0, seed=42)
        assert len(agents) == real_config["population"]["size"]

    def test_all_hyperparams_valid(self, real_config: dict):
        pm = PopulationManager(real_config, model_store=None)
        agents = pm.initialise_population(generation=0, seed=42)
        specs = parse_search_ranges(real_config["hyperparameters"]["search_ranges"])
        for a in agents:
            validate_hyperparams(a.hyperparameters, specs)

    def test_no_two_agents_identical(self, real_config: dict):
        pm = PopulationManager(real_config, model_store=None)
        agents = pm.initialise_population(generation=0, seed=42)
        hp_tuples = []
        for a in agents:
            hp_no_arch = {k: v for k, v in a.hyperparameters.items() if k != "architecture_name"}
            hp_tuples.append(tuple(sorted(hp_no_arch.items())))
        assert len(set(hp_tuples)) == len(hp_tuples)

    def test_forward_pass_on_real_observation(self, real_config: dict, real_day):
        """Every agent can forward-pass on a real observation vector."""
        pm = PopulationManager(real_config, model_store=None)
        agents = pm.initialise_population(generation=0, seed=42)

        # Build a real observation from the env
        env = BetfairEnv(
            days=[real_day],
            starting_budget=real_config["training"]["starting_budget"],
            max_runners=real_config["training"]["max_runners"],
        )
        obs, _ = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        for a in agents:
            out = a.policy(obs_tensor)
            assert out.action_mean.shape == (1, pm.action_dim)
            assert out.value.shape == (1, 1)
            assert not torch.isnan(out.action_mean).any()
            assert not torch.isnan(out.value).any()

    def test_round_trip_with_model_store(self, real_config: dict, tmp_path):
        """Create population → save to store → load back → forward pass matches."""
        store = ModelStore(
            str(tmp_path / "test.db"),
            str(tmp_path / "weights"),
        )
        pm = PopulationManager(real_config, model_store=store)
        agents = pm.initialise_population(generation=0, seed=42)

        obs = torch.randn(1, pm.obs_dim)

        for a in agents:
            loaded = pm.load_agent(a.model_id)
            torch.manual_seed(0)
            out_orig = a.policy(obs)
            torch.manual_seed(0)
            out_loaded = loaded.policy(obs)
            assert torch.allclose(out_orig.action_mean, out_loaded.action_mean, atol=1e-6)

    def test_all_agents_in_registry(self, real_config: dict, tmp_path):
        """All agents are persisted in the model store with correct metadata."""
        store = ModelStore(
            str(tmp_path / "test.db"),
            str(tmp_path / "weights"),
        )
        pm = PopulationManager(real_config, model_store=store)
        agents = pm.initialise_population(generation=0, seed=42)

        all_models = store.list_models()
        assert len(all_models) == len(agents)
        for a in agents:
            record = store.get_model(a.model_id)
            assert record.generation == 0
            assert record.architecture_name == "ppo_lstm_v1"
            assert record.status == "active"
