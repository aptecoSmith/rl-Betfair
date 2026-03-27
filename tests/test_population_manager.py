"""Unit tests for agents/population_manager.py."""

from __future__ import annotations

import math
import random

import pytest
import torch

from agents.architecture_registry import REGISTRY
from agents.policy_network import BasePolicy, PPOLSTMPolicy
from agents.population_manager import (
    AgentRecord,
    HyperparamSpec,
    PopulationManager,
    parse_search_ranges,
    sample_hyperparams,
    validate_hyperparams,
)
from env.betfair_env import (
    AGENT_STATE_DIM,
    MARKET_DIM,
    RUNNER_DIM,
    VELOCITY_DIM,
)

MAX_RUNNERS = 14
OBS_DIM = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * MAX_RUNNERS) + AGENT_STATE_DIM
ACTION_DIM = MAX_RUNNERS * 2


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def search_ranges_raw() -> dict:
    """Raw search ranges as they appear in config.yaml."""
    return {
        "learning_rate": {"type": "float_log", "min": 1e-5, "max": 5e-4},
        "ppo_clip_epsilon": {"type": "float", "min": 0.1, "max": 0.3},
        "entropy_coefficient": {"type": "float", "min": 0.001, "max": 0.05},
        "lstm_hidden_size": {"type": "int_choice", "choices": [64, 128, 256, 512, 1024, 2048]},
        "mlp_hidden_size": {"type": "int_choice", "choices": [64, 128, 256]},
        "mlp_layers": {"type": "int", "min": 1, "max": 3},
        "observation_window_ticks": {"type": "int", "min": 3, "max": 360},
        "reward_early_pick_bonus": {"type": "float", "min": 1.0, "max": 1.5},
        "reward_efficiency_penalty": {"type": "float", "min": 0.001, "max": 0.05},
    }


@pytest.fixture
def hp_specs(search_ranges_raw: dict) -> list[HyperparamSpec]:
    return parse_search_ranges(search_ranges_raw)


@pytest.fixture
def small_config(search_ranges_raw: dict) -> dict:
    """A minimal config dict for testing with population size 5."""
    return {
        "population": {"size": 5, "n_elite": 1, "selection_top_pct": 0.5},
        "training": {"architecture": "ppo_lstm_v1", "max_runners": 14, "starting_budget": 100.0},
        "hyperparameters": {"search_ranges": search_ranges_raw},
    }


# ── parse_search_ranges ──────────────────────────────────────────────────────


class TestParseSearchRanges:
    def test_returns_correct_count(self, search_ranges_raw: dict):
        specs = parse_search_ranges(search_ranges_raw)
        assert len(specs) == 9

    def test_float_log_spec(self, hp_specs: list[HyperparamSpec]):
        lr = next(s for s in hp_specs if s.name == "learning_rate")
        assert lr.type == "float_log"
        assert lr.min == 1e-5
        assert lr.max == 5e-4

    def test_float_spec(self, hp_specs: list[HyperparamSpec]):
        clip = next(s for s in hp_specs if s.name == "ppo_clip_epsilon")
        assert clip.type == "float"
        assert clip.min == 0.1
        assert clip.max == 0.3

    def test_int_choice_spec(self, hp_specs: list[HyperparamSpec]):
        lstm = next(s for s in hp_specs if s.name == "lstm_hidden_size")
        assert lstm.type == "int_choice"
        assert lstm.choices == [64, 128, 256, 512, 1024, 2048]

    def test_int_spec(self, hp_specs: list[HyperparamSpec]):
        layers = next(s for s in hp_specs if s.name == "mlp_layers")
        assert layers.type == "int"
        assert layers.min == 1
        assert layers.max == 3


# ── sample_hyperparams ───────────────────────────────────────────────────────


class TestSampleHyperparams:
    def test_returns_all_keys(self, hp_specs: list[HyperparamSpec]):
        hp = sample_hyperparams(hp_specs)
        expected = {s.name for s in hp_specs}
        assert set(hp.keys()) == expected

    def test_float_log_in_range(self, hp_specs: list[HyperparamSpec]):
        for _ in range(50):
            hp = sample_hyperparams(hp_specs)
            assert 1e-5 <= hp["learning_rate"] <= 5e-4

    def test_float_in_range(self, hp_specs: list[HyperparamSpec]):
        for _ in range(50):
            hp = sample_hyperparams(hp_specs)
            assert 0.1 <= hp["ppo_clip_epsilon"] <= 0.3
            assert 0.001 <= hp["entropy_coefficient"] <= 0.05

    def test_int_choice_valid(self, hp_specs: list[HyperparamSpec]):
        for _ in range(50):
            hp = sample_hyperparams(hp_specs)
            assert hp["lstm_hidden_size"] in [64, 128, 256, 512, 1024, 2048]
            assert hp["mlp_hidden_size"] in [64, 128, 256]

    def test_int_in_range(self, hp_specs: list[HyperparamSpec]):
        for _ in range(50):
            hp = sample_hyperparams(hp_specs)
            assert 1 <= hp["mlp_layers"] <= 3
            assert 3 <= hp["observation_window_ticks"] <= 360

    def test_reward_params_in_range(self, hp_specs: list[HyperparamSpec]):
        for _ in range(50):
            hp = sample_hyperparams(hp_specs)
            assert 1.0 <= hp["reward_early_pick_bonus"] <= 1.5
            assert 0.001 <= hp["reward_efficiency_penalty"] <= 0.05

    def test_seeded_reproducibility(self, hp_specs: list[HyperparamSpec]):
        hp1 = sample_hyperparams(hp_specs, rng=random.Random(42))
        hp2 = sample_hyperparams(hp_specs, rng=random.Random(42))
        assert hp1 == hp2

    def test_different_seeds_different_results(self, hp_specs: list[HyperparamSpec]):
        hp1 = sample_hyperparams(hp_specs, rng=random.Random(1))
        hp2 = sample_hyperparams(hp_specs, rng=random.Random(2))
        # Extremely unlikely all 9 params are identical with different seeds
        assert hp1 != hp2

    def test_float_log_distribution_covers_range(self, hp_specs: list[HyperparamSpec]):
        """Sampling on log scale should produce values across orders of magnitude."""
        rng = random.Random(123)
        values = [sample_hyperparams(hp_specs, rng)["learning_rate"] for _ in range(200)]
        # Should have some values in lower half (1e-5 to ~7e-5) and upper half
        below_geometric_mean = sum(1 for v in values if v < math.sqrt(1e-5 * 5e-4))
        assert below_geometric_mean > 20  # at least 10% in lower half

    def test_unknown_type_raises(self):
        specs = [HyperparamSpec(name="x", type="unknown")]
        with pytest.raises(ValueError, match="Unknown hyperparameter type"):
            sample_hyperparams(specs)


# ── validate_hyperparams ─────────────────────────────────────────────────────


class TestValidateHyperparams:
    def test_valid_params_pass(self, hp_specs: list[HyperparamSpec]):
        hp = sample_hyperparams(hp_specs, rng=random.Random(42))
        validate_hyperparams(hp, hp_specs)  # should not raise

    def test_float_below_min_raises(self, hp_specs: list[HyperparamSpec]):
        hp = sample_hyperparams(hp_specs, rng=random.Random(42))
        hp["ppo_clip_epsilon"] = 0.05  # below min of 0.1
        with pytest.raises(ValueError, match="ppo_clip_epsilon"):
            validate_hyperparams(hp, hp_specs)

    def test_float_above_max_raises(self, hp_specs: list[HyperparamSpec]):
        hp = sample_hyperparams(hp_specs, rng=random.Random(42))
        hp["ppo_clip_epsilon"] = 0.5  # above max of 0.3
        with pytest.raises(ValueError, match="ppo_clip_epsilon"):
            validate_hyperparams(hp, hp_specs)

    def test_int_choice_invalid_raises(self, hp_specs: list[HyperparamSpec]):
        hp = sample_hyperparams(hp_specs, rng=random.Random(42))
        hp["lstm_hidden_size"] = 999
        with pytest.raises(ValueError, match="lstm_hidden_size"):
            validate_hyperparams(hp, hp_specs)

    def test_int_below_min_raises(self, hp_specs: list[HyperparamSpec]):
        hp = sample_hyperparams(hp_specs, rng=random.Random(42))
        hp["mlp_layers"] = 0
        with pytest.raises(ValueError, match="mlp_layers"):
            validate_hyperparams(hp, hp_specs)

    def test_int_above_max_raises(self, hp_specs: list[HyperparamSpec]):
        hp = sample_hyperparams(hp_specs, rng=random.Random(42))
        hp["mlp_layers"] = 5
        with pytest.raises(ValueError, match="mlp_layers"):
            validate_hyperparams(hp, hp_specs)

    def test_extra_keys_ignored(self, hp_specs: list[HyperparamSpec]):
        hp = {"architecture_name": "ppo_lstm_v1", "extra_key": 42}
        validate_hyperparams(hp, hp_specs)  # should not raise

    def test_float_log_below_min_raises(self, hp_specs: list[HyperparamSpec]):
        hp = sample_hyperparams(hp_specs, rng=random.Random(42))
        hp["learning_rate"] = 1e-6  # below min of 1e-5
        with pytest.raises(ValueError, match="learning_rate"):
            validate_hyperparams(hp, hp_specs)


# ── PopulationManager.initialise_population ──────────────────────────────────


class TestPopulationManagerInit:
    def test_population_size(self, small_config: dict):
        pm = PopulationManager(small_config, model_store=None)
        agents = pm.initialise_population(generation=0, seed=42)
        assert len(agents) == 5

    def test_all_agents_are_agent_records(self, small_config: dict):
        pm = PopulationManager(small_config, model_store=None)
        agents = pm.initialise_population(generation=0, seed=42)
        for a in agents:
            assert isinstance(a, AgentRecord)

    def test_all_agents_have_policies(self, small_config: dict):
        pm = PopulationManager(small_config, model_store=None)
        agents = pm.initialise_population(generation=0, seed=42)
        for a in agents:
            assert isinstance(a.policy, BasePolicy)
            assert isinstance(a.policy, PPOLSTMPolicy)

    def test_all_hyperparams_within_range(self, small_config: dict, hp_specs: list[HyperparamSpec]):
        pm = PopulationManager(small_config, model_store=None)
        agents = pm.initialise_population(generation=0, seed=42)
        for a in agents:
            validate_hyperparams(a.hyperparameters, hp_specs)

    def test_no_two_agents_identical(self, small_config: dict):
        pm = PopulationManager(small_config, model_store=None)
        agents = pm.initialise_population(generation=0, seed=42)
        # Compare hyperparams (excluding architecture_name which is the same)
        hp_sets = []
        for a in agents:
            hp_no_arch = {k: v for k, v in a.hyperparameters.items() if k != "architecture_name"}
            hp_sets.append(tuple(sorted(hp_no_arch.items())))
        assert len(set(hp_sets)) == len(hp_sets), "All agents should have unique hyperparams"

    def test_generation_number_set(self, small_config: dict):
        pm = PopulationManager(small_config, model_store=None)
        agents = pm.initialise_population(generation=3, seed=42)
        for a in agents:
            assert a.generation == 3

    def test_architecture_name_set(self, small_config: dict):
        pm = PopulationManager(small_config, model_store=None)
        agents = pm.initialise_population(generation=0, seed=42)
        for a in agents:
            assert a.architecture_name == "ppo_lstm_v1"
            assert a.hyperparameters["architecture_name"] == "ppo_lstm_v1"

    def test_model_ids_unique(self, small_config: dict):
        pm = PopulationManager(small_config, model_store=None)
        agents = pm.initialise_population(generation=0, seed=42)
        ids = [a.model_id for a in agents]
        assert len(set(ids)) == len(ids)

    def test_seed_reproducibility(self, small_config: dict):
        pm = PopulationManager(small_config, model_store=None)
        agents1 = pm.initialise_population(generation=0, seed=99)
        agents2 = pm.initialise_population(generation=0, seed=99)
        for a, b in zip(agents1, agents2):
            # Hyperparams should be identical (model_ids will differ due to uuid)
            hp_a = {k: v for k, v in a.hyperparameters.items() if k != "architecture_name"}
            hp_b = {k: v for k, v in b.hyperparameters.items() if k != "architecture_name"}
            assert hp_a == hp_b

    def test_different_seeds_different_populations(self, small_config: dict):
        pm = PopulationManager(small_config, model_store=None)
        agents1 = pm.initialise_population(generation=0, seed=1)
        agents2 = pm.initialise_population(generation=0, seed=2)
        hp1 = [a.hyperparameters["learning_rate"] for a in agents1]
        hp2 = [a.hyperparameters["learning_rate"] for a in agents2]
        assert hp1 != hp2


# ── Forward pass works for all agents ────────────────────────────────────────


class TestPopulationForwardPass:
    def test_all_agents_forward_pass(self, small_config: dict):
        """Every agent in the population can produce a forward pass."""
        pm = PopulationManager(small_config, model_store=None)
        agents = pm.initialise_population(generation=0, seed=42)
        obs = torch.randn(1, OBS_DIM)
        for a in agents:
            out = a.policy(obs)
            assert out.action_mean.shape == (1, ACTION_DIM)
            assert out.value.shape == (1, 1)

    def test_agents_with_different_hidden_sizes(self, small_config: dict):
        """Agents with different LSTM/MLP sizes all produce valid outputs."""
        pm = PopulationManager(small_config, model_store=None)
        agents = pm.initialise_population(generation=0, seed=42)
        obs = torch.randn(1, OBS_DIM)
        hidden_sizes = set()
        for a in agents:
            hidden_sizes.add(a.hyperparameters["lstm_hidden_size"])
            hidden = a.policy.init_hidden(batch_size=1)
            assert hidden[0].shape[-1] == a.hyperparameters["lstm_hidden_size"]
            out = a.policy(obs, hidden)
            assert out.action_mean.shape == (1, ACTION_DIM)

    def test_agent_hidden_state_carries(self, small_config: dict):
        """Hidden state from one forward pass can be fed to the next."""
        pm = PopulationManager(small_config, model_store=None)
        agents = pm.initialise_population(generation=0, seed=42)
        obs = torch.randn(1, OBS_DIM)
        agent = agents[0]
        out1 = agent.policy(obs)
        out2 = agent.policy(obs, out1.hidden_state)
        # Hidden states should differ (LSTM updated)
        assert not torch.equal(out1.hidden_state[0], out2.hidden_state[0])


# ── PopulationManager with model store ───────────────────────────────────────


class TestPopulationManagerWithStore:
    @pytest.fixture
    def store(self, tmp_path):
        from registry.model_store import ModelStore

        db_path = str(tmp_path / "test.db")
        weights_dir = str(tmp_path / "weights")
        return ModelStore(db_path, weights_dir)

    def test_agents_registered_in_store(self, small_config: dict, store):
        pm = PopulationManager(small_config, model_store=store)
        agents = pm.initialise_population(generation=0, seed=42)
        for a in agents:
            record = store.get_model(a.model_id)
            assert record is not None
            assert record.architecture_name == "ppo_lstm_v1"
            assert record.generation == 0
            assert record.status == "active"

    def test_weights_saved_in_store(self, small_config: dict, store):
        pm = PopulationManager(small_config, model_store=store)
        agents = pm.initialise_population(generation=0, seed=42)
        for a in agents:
            state_dict = store.load_weights(a.model_id)
            assert len(state_dict) > 0

    def test_hyperparams_stored_correctly(self, small_config: dict, store):
        pm = PopulationManager(small_config, model_store=store)
        agents = pm.initialise_population(generation=0, seed=42)
        for a in agents:
            record = store.get_model(a.model_id)
            assert record.hyperparameters == a.hyperparameters

    def test_load_agent_round_trip(self, small_config: dict, store):
        pm = PopulationManager(small_config, model_store=store)
        agents = pm.initialise_population(generation=0, seed=42)
        original = agents[0]

        loaded = pm.load_agent(original.model_id)
        assert loaded.model_id == original.model_id
        assert loaded.generation == original.generation
        assert loaded.hyperparameters == original.hyperparameters
        assert loaded.architecture_name == original.architecture_name

        # Weights should match
        obs = torch.randn(1, OBS_DIM)
        torch.manual_seed(0)
        out_orig = original.policy(obs)
        torch.manual_seed(0)
        out_loaded = loaded.policy(obs)
        assert torch.allclose(out_orig.action_mean, out_loaded.action_mean, atol=1e-6)
        assert torch.allclose(out_orig.value, out_loaded.value, atol=1e-6)

    def test_load_agent_without_store_raises(self, small_config: dict):
        pm = PopulationManager(small_config, model_store=None)
        with pytest.raises(RuntimeError, match="Cannot load agent"):
            pm.load_agent("nonexistent-id")


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_population_size_one(self, search_ranges_raw: dict):
        config = {
            "population": {"size": 1, "n_elite": 1, "selection_top_pct": 0.5},
            "training": {"architecture": "ppo_lstm_v1", "max_runners": 14, "starting_budget": 100.0},
            "hyperparameters": {"search_ranges": search_ranges_raw},
        }
        pm = PopulationManager(config, model_store=None)
        agents = pm.initialise_population(generation=0, seed=42)
        assert len(agents) == 1

    def test_obs_dim_matches_env(self, small_config: dict):
        pm = PopulationManager(small_config, model_store=None)
        expected = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * 14) + AGENT_STATE_DIM
        assert pm.obs_dim == expected
        assert pm.obs_dim == 1345

    def test_action_dim_matches_env(self, small_config: dict):
        pm = PopulationManager(small_config, model_store=None)
        assert pm.action_dim == 28
