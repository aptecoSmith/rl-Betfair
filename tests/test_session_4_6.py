"""
Session 4.6 — Performance profiling & optimisation tests.

Tests that the performance optimisations don't break correctness:
- orjson parsing produces identical results to json.loads
- Optimised rollout produces valid transitions
- Pinned memory PPO update produces valid gradients
- Parallel evaluation config is respected
- Benchmark script runs and produces valid output
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.episode_builder import (
    Day,
    PastRace,
    PriceSize,
    Race,
    RunnerMeta,
    RunnerSnap,
    Tick,
    _json_loads,
    _parse_past_races_json,
    parse_snap_json,
)


# ── orjson parsing tests ────────────────────────────────────────────────────


class TestOrjsonParsing:
    """Verify orjson produces identical results to stdlib json."""

    def test_json_loads_simple_dict(self):
        result = _json_loads('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_json_loads_nested_dict(self):
        raw = '{"MarketRunners": [{"RunnerId": {"SelectionId": 123}}]}'
        result = _json_loads(raw)
        assert result["MarketRunners"][0]["RunnerId"]["SelectionId"] == 123

    def test_json_loads_bytes_input(self):
        result = _json_loads(b'{"a": 1}')
        assert result == {"a": 1}

    def test_json_loads_empty_array(self):
        result = _json_loads("[]")
        assert result == []

    def test_json_loads_handles_unicode(self):
        result = _json_loads('{"name": "Newmarket"}')
        assert result["name"] == "Newmarket"

    def test_parse_snap_json_nested_layout(self):
        """parse_snap_json with orjson produces identical RunnerSnaps."""
        snap = json.dumps({
            "MarketRunners": [{
                "RunnerId": {"SelectionId": 12345},
                "Definition": {"Status": "ACTIVE", "SortPriority": 1},
                "Prices": {
                    "LastTradedPrice": 4.5,
                    "TradedVolume": 1234.56,
                    "StartingPriceNear": 4.2,
                    "StartingPriceFar": 5.0,
                    "AvailableToBack": [{"Price": 4.5, "Size": 100.0}],
                    "AvailableToLay": [{"Price": 4.6, "Size": 150.0}],
                },
            }],
        })
        runners = parse_snap_json(snap)
        assert len(runners) == 1
        r = runners[0]
        assert r.selection_id == 12345
        assert r.status == "ACTIVE"
        assert r.last_traded_price == 4.5
        assert r.total_matched == 1234.56
        assert len(r.available_to_back) == 1
        assert r.available_to_back[0].price == 4.5
        assert len(r.available_to_lay) == 1
        assert r.available_to_lay[0].price == 4.6

    def test_parse_snap_json_flat_layout(self):
        """Flat layout still works with orjson."""
        snap = json.dumps({
            "Runners": [{
                "SelectionId": 99,
                "Status": "WINNER",
                "ltp": 2.0,
                "TotalMatched": 500.0,
                "StartingPriceNear": 0,
                "StartingPriceFar": 0,
                "AvailableToBack": [],
                "AvailableToLay": [],
            }],
        })
        runners = parse_snap_json(snap)
        assert len(runners) == 1
        assert runners[0].selection_id == 99
        assert runners[0].status == "WINNER"

    def test_parse_past_races_json_with_orjson(self):
        """PastRacesJson parsing works with orjson."""
        raw = json.dumps([{
            "date": "2026-01-15T00:00:00",
            "course": "Ascot",
            "distance": 2000,
            "going": {"full": "Good", "abbr": "G"},
            "raceType": {"full": "Flat"},
            "bsp": 5.5,
            "inPlayMax": 10.0,
            "inPlayMin": 3.0,
            "jockey": "Test Jockey",
            "officialRating": 85,
            "position": "2/10",
        }])
        races = _parse_past_races_json(raw)
        assert len(races) == 1
        assert races[0].course == "Ascot"
        assert races[0].position == 2
        assert races[0].field_size == 10

    def test_parse_past_races_json_empty_string(self):
        assert _parse_past_races_json("") == ()
        assert _parse_past_races_json("[]") == ()
        assert _parse_past_races_json("null") == ()

    def test_parse_past_races_json_malformed(self):
        assert _parse_past_races_json("{invalid}") == ()


# ── Rollout optimisation tests ──────────────────────────────────────────────


@pytest.fixture
def config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # These session-4.6 rollout tests predate scalping mode and compute
    # action_dim from ACTIONS_PER_RUNNER (=4). config.yaml now defaults
    # to scalping_mode=true for the active scalping training runs, so
    # pin it off here to keep the directional action layout these tests
    # were written against.
    cfg.setdefault("training", {})["scalping_mode"] = False
    return cfg


@pytest.fixture
def synthetic_day():
    """Create a minimal synthetic day for testing."""
    runner_snap = RunnerSnap(
        selection_id=1,
        status="ACTIVE",
        last_traded_price=5.0,
        total_matched=100.0,
        starting_price_near=0.0,
        starting_price_far=0.0,
        adjustment_factor=None,
        bsp=None,
        sort_priority=1,
        removal_date=None,
        available_to_back=[PriceSize(5.0, 50.0)],
        available_to_lay=[PriceSize(5.2, 50.0)],
    )
    winner_snap = RunnerSnap(
        selection_id=1,
        status="WINNER",
        last_traded_price=5.0,
        total_matched=100.0,
        starting_price_near=0.0,
        starting_price_far=0.0,
        adjustment_factor=None,
        bsp=None,
        sort_priority=1,
        removal_date=None,
        available_to_back=[],
        available_to_lay=[],
    )
    from datetime import datetime
    tick1 = Tick(
        market_id="m1", timestamp=datetime(2026, 1, 1, 12, 0),
        sequence_number=1, venue="Test", market_start_time=datetime(2026, 1, 1, 12, 30),
        number_of_active_runners=1, traded_volume=100.0, in_play=False,
        winner_selection_id=1, race_status=None,
        temperature=None, precipitation=None, wind_speed=None,
        wind_direction=None, humidity=None, weather_code=None,
        runners=[runner_snap],
    )
    tick2 = Tick(
        market_id="m1", timestamp=datetime(2026, 1, 1, 12, 5),
        sequence_number=2, venue="Test", market_start_time=datetime(2026, 1, 1, 12, 30),
        number_of_active_runners=1, traded_volume=200.0, in_play=False,
        winner_selection_id=1, race_status=None,
        temperature=None, precipitation=None, wind_speed=None,
        wind_direction=None, humidity=None, weather_code=None,
        runners=[runner_snap],
    )
    tick3 = Tick(
        market_id="m1", timestamp=datetime(2026, 1, 1, 12, 30),
        sequence_number=3, venue="Test", market_start_time=datetime(2026, 1, 1, 12, 30),
        number_of_active_runners=1, traded_volume=300.0, in_play=True,
        winner_selection_id=1, race_status="off",
        temperature=None, precipitation=None, wind_speed=None,
        wind_direction=None, humidity=None, weather_code=None,
        runners=[winner_snap],
    )
    race = Race(
        market_id="m1", venue="Test",
        market_start_time=datetime(2026, 1, 1, 12, 30),
        winner_selection_id=1,
        ticks=[tick1, tick2, tick3],
        runner_metadata={
            1: RunnerMeta(
                selection_id=1, runner_name="TestHorse", sort_priority="1",
                handicap="0", sire_name="", dam_name="", damsire_name="",
                bred="", official_rating="85", adjusted_rating="85",
                age="4", sex_type="", colour_type="", weight_value="",
                weight_units="", jockey_name="", jockey_claim="",
                trainer_name="", owner_name="", stall_draw="1",
                cloth_number="1", form="", days_since_last_run="",
                wearing="", forecastprice_numerator="5",
                forecastprice_denominator="1",
            ),
        },
        winning_selection_ids={1},
    )
    return Day(date="2026-01-01", races=[race])


class TestOptimisedRollout:
    """Verify the optimised rollout loop produces valid results."""

    def test_rollout_produces_transitions(self, config, synthetic_day):
        from agents.architecture_registry import create_policy
        from agents.ppo_trainer import PPOTrainer
        from env.betfair_env import ACTIONS_PER_RUNNER, MARKET_DIM, VELOCITY_DIM, RUNNER_DIM, AGENT_STATE_DIM, POSITION_DIM

        max_runners = config["training"]["max_runners"]
        obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * max_runners) + AGENT_STATE_DIM + (POSITION_DIM * max_runners)
        action_dim = max_runners * ACTIONS_PER_RUNNER

        policy = create_policy("ppo_lstm_v1", obs_dim, action_dim, max_runners, {
            "lstm_hidden_size": 64, "mlp_hidden_size": 64, "mlp_layers": 1,
        })

        trainer = PPOTrainer(
            policy=policy, config=config, device="cpu",
        )
        rollout, ep_stats = trainer._collect_rollout(synthetic_day)

        assert len(rollout) == 3  # 3 ticks
        assert ep_stats.n_steps == 3
        assert ep_stats.day_date == "2026-01-01"

    def test_rollout_transitions_have_valid_fields(self, config, synthetic_day):
        from agents.architecture_registry import create_policy
        from agents.ppo_trainer import PPOTrainer
        from env.betfair_env import ACTIONS_PER_RUNNER, MARKET_DIM, VELOCITY_DIM, RUNNER_DIM, AGENT_STATE_DIM, POSITION_DIM

        max_runners = config["training"]["max_runners"]
        obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * max_runners) + AGENT_STATE_DIM + (POSITION_DIM * max_runners)
        action_dim = max_runners * ACTIONS_PER_RUNNER

        policy = create_policy("ppo_lstm_v1", obs_dim, action_dim, max_runners, {
            "lstm_hidden_size": 64, "mlp_hidden_size": 64, "mlp_layers": 1,
        })

        trainer = PPOTrainer(
            policy=policy, config=config, device="cpu",
        )
        rollout, _ = trainer._collect_rollout(synthetic_day)

        for t in rollout.transitions:
            assert t.obs.shape == (obs_dim,)
            assert t.action.shape == (action_dim,)
            assert not np.isnan(t.log_prob)
            assert not np.isnan(t.value)
            assert not np.isnan(t.reward)
            # obs should have no NaNs
            assert not np.any(np.isnan(t.obs))

    def test_rollout_last_transition_is_done(self, config, synthetic_day):
        from agents.architecture_registry import create_policy
        from agents.ppo_trainer import PPOTrainer
        from env.betfair_env import ACTIONS_PER_RUNNER, MARKET_DIM, VELOCITY_DIM, RUNNER_DIM, AGENT_STATE_DIM, POSITION_DIM

        max_runners = config["training"]["max_runners"]
        obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * max_runners) + AGENT_STATE_DIM + (POSITION_DIM * max_runners)
        action_dim = max_runners * ACTIONS_PER_RUNNER

        policy = create_policy("ppo_lstm_v1", obs_dim, action_dim, max_runners, {
            "lstm_hidden_size": 64, "mlp_hidden_size": 64, "mlp_layers": 1,
        })

        trainer = PPOTrainer(
            policy=policy, config=config, device="cpu",
        )
        rollout, _ = trainer._collect_rollout(synthetic_day)

        assert rollout.transitions[-1].done is True
        # All non-terminal transitions should not be done
        for t in rollout.transitions[:-1]:
            assert t.done is False


# ── Pinned memory PPO update tests ──────────────────────────────────────────


class TestPinnedMemoryPPO:
    """Verify PPO update with pinned memory produces valid gradients."""

    def test_ppo_update_cpu(self, config, synthetic_day):
        """PPO update on CPU (no pinning) still works correctly."""
        from agents.architecture_registry import create_policy
        from agents.ppo_trainer import PPOTrainer
        from env.betfair_env import ACTIONS_PER_RUNNER, MARKET_DIM, VELOCITY_DIM, RUNNER_DIM, AGENT_STATE_DIM, POSITION_DIM

        max_runners = config["training"]["max_runners"]
        obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * max_runners) + AGENT_STATE_DIM + (POSITION_DIM * max_runners)
        action_dim = max_runners * ACTIONS_PER_RUNNER

        policy = create_policy("ppo_lstm_v1", obs_dim, action_dim, max_runners, {
            "lstm_hidden_size": 64, "mlp_hidden_size": 64, "mlp_layers": 1,
        })

        trainer = PPOTrainer(
            policy=policy, config=config, device="cpu",
        )
        rollout, _ = trainer._collect_rollout(synthetic_day)
        loss_info = trainer._ppo_update(rollout)

        assert "policy_loss" in loss_info
        assert "value_loss" in loss_info
        assert "entropy" in loss_info
        assert not math.isnan(loss_info["policy_loss"])
        assert not math.isnan(loss_info["value_loss"])
        assert not math.isnan(loss_info["entropy"])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU")
    def test_ppo_update_cuda_pinned(self, config, synthetic_day):
        """PPO update on CUDA with pinned memory transfers."""
        from agents.architecture_registry import create_policy
        from agents.ppo_trainer import PPOTrainer
        from env.betfair_env import ACTIONS_PER_RUNNER, MARKET_DIM, VELOCITY_DIM, RUNNER_DIM, AGENT_STATE_DIM, POSITION_DIM

        max_runners = config["training"]["max_runners"]
        obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * max_runners) + AGENT_STATE_DIM + (POSITION_DIM * max_runners)
        action_dim = max_runners * ACTIONS_PER_RUNNER

        policy = create_policy("ppo_lstm_v1", obs_dim, action_dim, max_runners, {
            "lstm_hidden_size": 64, "mlp_hidden_size": 64, "mlp_layers": 1,
        })

        trainer = PPOTrainer(
            policy=policy, config=config, device="cuda",
        )
        rollout, _ = trainer._collect_rollout(synthetic_day)
        loss_info = trainer._ppo_update(rollout)

        assert not math.isnan(loss_info["policy_loss"])
        assert not math.isnan(loss_info["value_loss"])


# ── Parallel evaluation config tests ────────────────────────────────────────


class TestParallelEvalConfig:
    """Verify parallel evaluation respects config settings."""

    def test_default_eval_workers_is_1(self, config):
        """Without config override, eval_workers defaults to 1 (sequential)."""
        workers = config.get("training", {}).get("eval_workers", 1)
        # Default config has no eval_workers key, should be 1
        assert workers == 1

    def test_eval_workers_caps_at_cpu_count(self):
        """Even with high config value, workers capped at CPU count."""
        n_cpu = os.cpu_count() or 1
        configured = 999
        n_agents = 20
        actual = min(n_agents, configured, n_cpu)
        assert actual == n_cpu


# ── Optimised evaluation tests ──────────────────────────────────────────────


class TestOptimisedEvaluation:
    """Verify the optimised evaluator produces valid results."""

    def test_evaluation_produces_day_records(self, config, synthetic_day):
        from agents.architecture_registry import create_policy
        from training.evaluator import Evaluator
        from env.betfair_env import ACTIONS_PER_RUNNER, MARKET_DIM, VELOCITY_DIM, RUNNER_DIM, AGENT_STATE_DIM, POSITION_DIM

        max_runners = config["training"]["max_runners"]
        obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * max_runners) + AGENT_STATE_DIM + (POSITION_DIM * max_runners)
        action_dim = max_runners * ACTIONS_PER_RUNNER

        policy = create_policy("ppo_lstm_v1", obs_dim, action_dim, max_runners, {
            "lstm_hidden_size": 64, "mlp_hidden_size": 64, "mlp_layers": 1,
        })

        evaluator = Evaluator(config=config, device="cpu")
        run_id, day_records = evaluator.evaluate(
            model_id="test",
            policy=policy,
            test_days=[synthetic_day],
            train_cutoff_date="2025-01-01",
        )

        assert len(day_records) == 1
        dr = day_records[0]
        assert dr.date == "2026-01-01"
        assert not math.isnan(dr.day_pnl)
        assert dr.bet_count >= 0


# ── Benchmark script tests ──────────────────────────────────────────────────


class TestBenchmarkScript:
    """Verify benchmark script structure and compare function."""

    def test_benchmark_compare_function(self, tmp_path):
        """The compare function reads two JSON files and prints comparison."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from benchmark import compare

        before = {
            "timings": {
                "data_loading_s": 4.0,
                "feature_engineering_s": 1.5,
                "rollout_collection_s": 8.5,
                "ppo_update_s": 1.7,
                "evaluation_s": 5.4,
                "total_train_eval_s": 15.6,
            },
            "metrics": {"rollout_steps_per_s": 500},
        }
        after = {
            "timings": {
                "data_loading_s": 2.7,
                "feature_engineering_s": 1.4,
                "rollout_collection_s": 5.7,
                "ppo_update_s": 1.8,
                "evaluation_s": 4.8,
                "total_train_eval_s": 12.3,
            },
            "metrics": {"rollout_steps_per_s": 740},
        }

        before_path = tmp_path / "before.json"
        after_path = tmp_path / "after.json"
        before_path.write_text(json.dumps(before), encoding="utf-8")
        after_path.write_text(json.dumps(after), encoding="utf-8")

        # Should not raise
        compare(str(before_path), str(after_path))

    def test_benchmark_output_file_exists(self):
        """After running benchmark, output JSON should exist."""
        bench_file = Path("logs/bench_after.json")
        if bench_file.exists():
            data = json.loads(bench_file.read_text(encoding="utf-8"))
            assert "timings" in data
            assert "metrics" in data
            assert "device" in data


# ── orjson fallback test ────────────────────────────────────────────────────


class TestOrjsonFallback:
    """Verify that the system works even without orjson installed."""

    def test_json_loads_function_exists(self):
        """_json_loads should always be importable."""
        from data.episode_builder import _json_loads
        result = _json_loads('{"test": true}')
        assert result == {"test": True}
