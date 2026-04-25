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
    perturb_from_seed,
    sample_hyperparams,
    validate_hyperparams,
)
from env.betfair_env import (
    ACTIONS_PER_RUNNER,
    AGENT_STATE_DIM,
    MARKET_DIM,
    POSITION_DIM,
    RUNNER_DIM,
    VELOCITY_DIM,
)

MAX_RUNNERS = 14
OBS_DIM = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * MAX_RUNNERS) + AGENT_STATE_DIM + (POSITION_DIM * MAX_RUNNERS)
ACTION_DIM = MAX_RUNNERS * ACTIONS_PER_RUNNER


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
        "early_pick_bonus_min": {"type": "float", "min": 1.0, "max": 1.2},
        "early_pick_bonus_max": {"type": "float", "min": 1.2, "max": 1.5},
        "reward_efficiency_penalty": {"type": "float", "min": 0.001, "max": 0.05},
        "reward_precision_bonus": {"type": "float", "min": 0.0, "max": 3.0},
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
        assert len(specs) == 10

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

    def test_reward_params_in_range(self, hp_specs: list[HyperparamSpec]):
        for _ in range(50):
            hp = sample_hyperparams(hp_specs)
            assert 1.0 <= hp["early_pick_bonus_min"] <= 1.2
            assert 1.2 <= hp["early_pick_bonus_max"] <= 1.5
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

    def test_load_agent_survives_hp_drift_via_shape_inference(
        self, small_config: dict, store,
    ):
        """Hyperparams can drift from saved weight shapes (e.g. breeding
        re-wrote genes on a child without re-initialising weights).
        load_agent must recover by inferring the architecture shape from
        the state dict and rebuilding the policy at matching dimensions.

        Diagnosed in the 2026-04-22 reeval failure on 308236be — stored
        hyperparams said hidden=512, ctx=128; saved weights were
        hidden=256, ctx=32, from an earlier training run. load_state_dict
        rejected with size mismatch; the fix retries with shape-inferred
        hyperparams.
        """
        pm = PopulationManager(small_config, model_store=store)
        agents = pm.initialise_population(generation=0, seed=42)
        original = agents[0]

        # Forge the drift: after weights are saved at the original hp,
        # mutate the record's hp to claim a different (bigger) hidden
        # size. A naive re-build would construct a hidden=1024 LSTM
        # that can't load the hidden=256 weights.
        original_hp = dict(original.hyperparameters)
        drifted_hp = {**original_hp}
        if original.architecture_name in ("ppo_lstm_v1", "ppo_time_lstm_v1"):
            drifted_hp["lstm_hidden_size"] = (
                int(original_hp.get("lstm_hidden_size", 256)) * 2
            )
        else:
            # transformer: bump ctx_ticks to a value that won't match
            # the position_embedding shape.
            drifted_hp["transformer_ctx_ticks"] = (
                int(original_hp.get("transformer_ctx_ticks", 32)) * 2
            )
        store.update_hyperparameters(original.model_id, drifted_hp)

        # Without the fix: naively rebuilding at drifted_hp and calling
        # load_state_dict would raise "size mismatch". With the fix:
        # load_agent detects the mismatch, infers the original shape
        # from the state dict, and rebuilds.
        loaded = pm.load_agent(original.model_id)

        # The loaded policy's weights match the original (shape + values).
        obs = torch.randn(1, OBS_DIM)
        torch.manual_seed(0)
        out_orig = original.policy(obs)
        torch.manual_seed(0)
        out_loaded = loaded.policy(obs)
        assert torch.allclose(
            out_orig.action_mean, out_loaded.action_mean, atol=1e-6,
        )

    def test_load_agent_survives_lstm_layer_norm_drift(
        self, small_config: dict, store,
    ):
        """When the hp record claims ``lstm_layer_norm=True`` but the
        saved weights were trained with ``lstm_layer_norm=False`` (or
        vice-versa), load_agent must rebuild with the structurally-
        correct value rather than crash on missing/unexpected
        ``lstm_output_norm.weight`` keys.

        Root cause of the 2026-04-25 ``post-kl-fix-reference`` gen-1
        crash: ``backfill_hyperparameters`` set
        ``lstm_layer_norm=True`` (midpoint default of an int_choice
        spec) on every gen-0 record at gen-1 start. The gen-0 weights
        had been trained with the spec absent or False → no LayerNorm
        params in state_dict → "Missing key: lstm_output_norm.weight"
        on rebuild. 8/12 agents skipped, gen-1 collapsed to 4 agents.

        See worker.log around line 1170310 (run cc2bd0ba) for the
        original trace.
        """
        from agents.architecture_registry import (
            infer_arch_hp_from_state_dict,
        )
        pm = PopulationManager(small_config, model_store=store)
        agents = pm.initialise_population(generation=0, seed=42)
        # Pick an LSTM-family agent — the only ones with the
        # lstm_output_norm gene.
        original = next(
            (a for a in agents if a.architecture_name in
             ("ppo_lstm_v1", "ppo_time_lstm_v1")),
            None,
        )
        if original is None:
            pytest.skip("test config produced no LSTM-family agents")
        original_hp = dict(original.hyperparameters)
        # Force the original to be saved with NO LayerNorm (Identity).
        # The state_dict will lack lstm_output_norm.weight.
        from agents.architecture_registry import create_policy
        clean_hp = {**original_hp, "lstm_layer_norm": False}
        clean_policy = create_policy(
            name=original.architecture_name,
            obs_dim=pm.obs_dim,
            action_dim=pm.action_dim,
            max_runners=pm.max_runners,
            hyperparams=clean_hp,
        )
        store.save_weights(
            original.model_id, clean_policy.state_dict(),
            obs_schema_version=pm._obs_schema_version,
        )

        # Inference should detect lstm_layer_norm=False from the
        # absent lstm_output_norm.weight key.
        sd = store.load_weights(
            original.model_id,
            expected_obs_schema_version=pm._obs_schema_version,
        )
        inferred = infer_arch_hp_from_state_dict(
            original.architecture_name, sd,
        )
        assert inferred.get("lstm_layer_norm") is False, (
            f"infer_arch_hp_from_state_dict failed to detect "
            f"lstm_layer_norm=False from absent lstm_output_norm.weight; "
            f"inferred={inferred}"
        )

        # Now forge the drift: hp record claims lstm_layer_norm=True
        # (the failing backfill case). load_agent must rebuild.
        drifted_hp = {**clean_hp, "lstm_layer_norm": True}
        store.update_hyperparameters(original.model_id, drifted_hp)
        loaded = pm.load_agent(original.model_id)

        # Verify the rebuilt policy has Identity, not LayerNorm.
        from torch import nn
        norm_module = getattr(loaded.policy, "lstm_output_norm", None)
        assert isinstance(norm_module, nn.Identity), (
            f"Expected Identity after rebuild with inferred "
            f"lstm_layer_norm=False; got {type(norm_module).__name__}"
        )

    def test_load_agent_inverse_layer_norm_drift(
        self, small_config: dict, store,
    ):
        """Symmetric case: hp claims False but weights have LayerNorm.

        The inverse of the gen-1 crash. Less likely in practice but
        the inference must be symmetric: presence of
        ``lstm_output_norm.weight`` → infer
        ``lstm_layer_norm=True``.
        """
        from agents.architecture_registry import (
            create_policy,
            infer_arch_hp_from_state_dict,
        )
        pm = PopulationManager(small_config, model_store=store)
        agents = pm.initialise_population(generation=0, seed=42)
        original = next(
            (a for a in agents if a.architecture_name in
             ("ppo_lstm_v1", "ppo_time_lstm_v1")),
            None,
        )
        if original is None:
            pytest.skip("test config produced no LSTM-family agents")
        original_hp = dict(original.hyperparameters)
        norm_hp = {**original_hp, "lstm_layer_norm": True}
        norm_policy = create_policy(
            name=original.architecture_name,
            obs_dim=pm.obs_dim,
            action_dim=pm.action_dim,
            max_runners=pm.max_runners,
            hyperparams=norm_hp,
        )
        store.save_weights(
            original.model_id, norm_policy.state_dict(),
            obs_schema_version=pm._obs_schema_version,
        )
        sd = store.load_weights(
            original.model_id,
            expected_obs_schema_version=pm._obs_schema_version,
        )
        inferred = infer_arch_hp_from_state_dict(
            original.architecture_name, sd,
        )
        assert inferred.get("lstm_layer_norm") is True, (
            f"infer failed to detect lstm_layer_norm=True from "
            f"present lstm_output_norm.weight; inferred={inferred}"
        )


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
        # Dimension constants are defined in env/betfair_env.py — update that
        # file if you need to change the observation layout. This test only
        # asserts that PopulationManager agrees with whatever env currently
        # exports, rather than hardcoding a fresh integer (which is what
        # went stale originally).
        expected = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * 14) + AGENT_STATE_DIM + (POSITION_DIM * 14)
        assert pm.obs_dim == expected

    def test_action_dim_matches_env(self, small_config: dict):
        pm = PopulationManager(small_config, model_store=None)
        from env.betfair_env import ACTIONS_PER_RUNNER
        assert pm.action_dim == 14 * ACTIONS_PER_RUNNER


# ── Seed-based population (Sprint 4, Session 04) ─────────────────────────────


class TestPerturbFromSeed:
    """Tests for ``perturb_from_seed``."""

    def test_all_values_within_bounds(self, hp_specs):
        seed_point = sample_hyperparams(hp_specs, random.Random(0))
        for i in range(20):
            hp = perturb_from_seed(seed_point, hp_specs, random.Random(i), sigma=0.1)
            for spec in hp_specs:
                val = hp[spec.name]
                if spec.type in ("float", "float_log"):
                    assert spec.min <= val <= spec.max, f"{spec.name}={val} out of bounds"
                elif spec.type == "int":
                    assert int(spec.min) <= val <= int(spec.max)
                elif spec.type in ("int_choice", "str_choice"):
                    assert val in spec.choices

    def test_values_close_to_seed(self, hp_specs):
        """Perturbed values should cluster around the seed."""
        seed_point = {
            "learning_rate": 1e-4,
            "ppo_clip_epsilon": 0.2,
            "entropy_coefficient": 0.02,
            "lstm_hidden_size": 256,
            "mlp_hidden_size": 128,
            "mlp_layers": 2,
            "early_pick_bonus_min": 1.1,
            "early_pick_bonus_max": 1.35,
            "reward_efficiency_penalty": 0.02,
            "reward_precision_bonus": 1.5,
        }
        clips = []
        for i in range(50):
            hp = perturb_from_seed(seed_point, hp_specs, random.Random(i), sigma=0.1)
            clips.append(hp["ppo_clip_epsilon"])
        avg = sum(clips) / len(clips)
        # Average should be close to seed value 0.2 (range is 0.1–0.3)
        assert abs(avg - 0.2) < 0.05, f"Average clip {avg} too far from seed 0.2"

    def test_spread_proportional_to_sigma(self, hp_specs):
        seed_point = sample_hyperparams(hp_specs, random.Random(0))
        # Small sigma → tight cluster
        small = [perturb_from_seed(seed_point, hp_specs, random.Random(i), sigma=0.01)["ppo_clip_epsilon"] for i in range(50)]
        # Large sigma → wider cluster
        large = [perturb_from_seed(seed_point, hp_specs, random.Random(i), sigma=0.3)["ppo_clip_epsilon"] for i in range(50)]

        import statistics
        std_small = statistics.stdev(small)
        std_large = statistics.stdev(large)
        assert std_large > std_small, "Larger sigma should produce more spread"


class TestInitialiseFromSeed:
    """Tests for seed_point parameter on initialise_population."""

    def test_seed_point_population(self, small_config):
        pm = PopulationManager(small_config, model_store=None)
        seed_point = sample_hyperparams(pm.hp_specs, random.Random(42))
        seed_point["architecture_name"] = "ppo_lstm_v1"
        agents = pm.initialise_population(
            generation=0, seed=99, seed_point=seed_point,
        )
        assert len(agents) == 5
        # All agents should have valid hyperparameters
        for a in agents:
            assert isinstance(a.policy, BasePolicy)

    def test_no_seed_point_unchanged(self, small_config):
        """Without seed_point, behaviour should be identical to before."""
        pm = PopulationManager(small_config, model_store=None)
        a = pm.initialise_population(generation=0, seed=42)
        b = pm.initialise_population(generation=0, seed=42)
        # Same seed → same hyperparameters
        for ag_a, ag_b in zip(a, b):
            assert ag_a.hyperparameters == ag_b.hyperparameters
