"""
Session 4.6 — Integration tests for performance optimisations.

Tests that the optimised pipeline produces correct results on real data.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.episode_builder import load_day

# Skip all tests if no real data is available
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
REAL_DATES = sorted(
    f.stem for f in PROCESSED_DIR.glob("*.parquet")
    if not f.stem.endswith("_runners")
) if PROCESSED_DIR.exists() else []

pytestmark = pytest.mark.skipif(
    len(REAL_DATES) < 1,
    reason="No extracted data available for integration tests",
)


@pytest.fixture(scope="module")
def config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Session-4.6 integration tests assume directional action layout.
    # config.yaml defaults scalping_mode=true for active training runs,
    # so pin it off here.
    cfg.setdefault("training", {})["scalping_mode"] = False
    return cfg


@pytest.fixture(scope="module")
def real_day():
    """Load the largest available day."""
    days = []
    for d in REAL_DATES:
        day = load_day(d)
        days.append(day)
    return max(days, key=lambda d: sum(len(r.ticks) for r in d.races))


class TestOrjsonRealData:
    """Verify orjson parses real SnapJson correctly."""

    def test_all_ticks_parsed(self, real_day):
        """Every tick in the real day has parsed runners."""
        for race in real_day.races:
            for tick in race.ticks:
                # Should have at least 1 runner per tick
                assert len(tick.runners) > 0, (
                    f"Tick {tick.sequence_number} in {tick.market_id} has no runners"
                )

    def test_runner_prices_are_valid(self, real_day):
        """Runner prices should be positive floats."""
        for race in real_day.races:
            for tick in race.ticks:
                for runner in tick.runners:
                    if runner.last_traded_price > 0:
                        assert runner.last_traded_price < 1100, (
                            f"LTP {runner.last_traded_price} out of range"
                        )
                    for ps in runner.available_to_back:
                        assert ps.price > 0
                        assert ps.size >= 0
                    for ps in runner.available_to_lay:
                        assert ps.price > 0
                        assert ps.size >= 0


class TestRolloutOnRealData:
    """Verify optimised rollout works on real extracted data."""

    def test_rollout_completes_on_real_day(self, config, real_day):
        """Full rollout on real data produces valid transitions."""
        from agents.architecture_registry import create_policy
        from agents.ppo_trainer import PPOTrainer
        from data.feature_engineer import engineer_day
        from env.betfair_env import ACTIONS_PER_RUNNER, MARKET_DIM, VELOCITY_DIM, RUNNER_DIM, AGENT_STATE_DIM

        max_runners = config["training"]["max_runners"]
        obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * max_runners) + AGENT_STATE_DIM
        action_dim = max_runners * ACTIONS_PER_RUNNER

        policy = create_policy("ppo_lstm_v1", obs_dim, action_dim, max_runners, {
            "lstm_hidden_size": 64, "mlp_hidden_size": 64, "mlp_layers": 1,
        })

        fc = {real_day.date: engineer_day(real_day)}

        trainer = PPOTrainer(
            policy=policy, config=config, device="cpu", feature_cache=fc,
        )
        rollout, ep_stats = trainer._collect_rollout(real_day)

        total_ticks = sum(len(r.ticks) for r in real_day.races)
        assert ep_stats.n_steps == total_ticks
        assert len(rollout) == total_ticks

        # No NaN observations
        for t in rollout.transitions:
            assert not np.any(np.isnan(t.obs)), "NaN found in observation"
            assert not np.isnan(t.log_prob), "NaN in log_prob"
            assert not np.isnan(t.value), "NaN in value"

    @pytest.mark.timeout(120)
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU")
    def test_ppo_update_no_nans_on_real_data(self, config, real_day):
        """PPO update on real data with pinned memory produces no NaNs."""
        from agents.architecture_registry import create_policy
        from agents.ppo_trainer import PPOTrainer
        from data.feature_engineer import engineer_day
        from env.betfair_env import ACTIONS_PER_RUNNER, MARKET_DIM, VELOCITY_DIM, RUNNER_DIM, AGENT_STATE_DIM

        max_runners = config["training"]["max_runners"]
        obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * max_runners) + AGENT_STATE_DIM
        action_dim = max_runners * ACTIONS_PER_RUNNER

        policy = create_policy("ppo_lstm_v1", obs_dim, action_dim, max_runners, {
            "lstm_hidden_size": 64, "mlp_hidden_size": 64, "mlp_layers": 1,
        })

        fc = {real_day.date: engineer_day(real_day)}

        trainer = PPOTrainer(
            policy=policy, config=config, device="cuda", feature_cache=fc,
        )
        rollout, _ = trainer._collect_rollout(real_day)
        loss_info = trainer._ppo_update(rollout)

        assert not math.isnan(loss_info["policy_loss"])
        assert not math.isnan(loss_info["value_loss"])
        assert not math.isnan(loss_info["entropy"])

    def test_evaluation_on_real_day(self, config, real_day):
        """Evaluation with optimised loop on real data."""
        from agents.architecture_registry import create_policy
        from training.evaluator import Evaluator
        from data.feature_engineer import engineer_day
        from env.betfair_env import ACTIONS_PER_RUNNER, MARKET_DIM, VELOCITY_DIM, RUNNER_DIM, AGENT_STATE_DIM

        max_runners = config["training"]["max_runners"]
        obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * max_runners) + AGENT_STATE_DIM
        action_dim = max_runners * ACTIONS_PER_RUNNER

        policy = create_policy("ppo_lstm_v1", obs_dim, action_dim, max_runners, {
            "lstm_hidden_size": 64, "mlp_hidden_size": 64, "mlp_layers": 1,
        })

        fc = {real_day.date: engineer_day(real_day)}

        evaluator = Evaluator(config=config, device="cpu", feature_cache=fc)
        _, day_records = evaluator.evaluate(
            model_id="test",
            policy=policy,
            test_days=[real_day],
            train_cutoff_date="2025-01-01",
        )

        assert len(day_records) == 1
        dr = day_records[0]
        assert not math.isnan(dr.day_pnl)
        assert dr.bet_count >= 0


class TestBenchmarkResults:
    """Verify benchmark output files contain valid data."""

    def test_before_benchmark_exists(self):
        path = Path("logs/bench_before.json")
        assert path.exists(), "Run benchmark.py --output logs/bench_before.json first"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["timings"]["data_loading_s"] > 0
        assert data["timings"]["rollout_collection_s"] > 0

    def test_after_benchmark_exists(self):
        path = Path("logs/bench_after.json")
        assert path.exists(), "Run benchmark.py --output logs/bench_after.json first"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["timings"]["data_loading_s"] > 0
        assert data["timings"]["rollout_collection_s"] > 0

    def test_after_is_faster_than_before(self):
        """Optimised pipeline should be faster overall."""
        before_path = Path("logs/bench_before.json")
        after_path = Path("logs/bench_after.json")
        if not before_path.exists() or not after_path.exists():
            pytest.skip("Benchmark files not available")

        before = json.loads(before_path.read_text(encoding="utf-8"))
        after = json.loads(after_path.read_text(encoding="utf-8"))

        # Data loading should be faster (orjson)
        assert after["timings"]["data_loading_s"] < before["timings"]["data_loading_s"]
        # Rollout should be faster
        assert after["timings"]["rollout_collection_s"] < before["timings"]["rollout_collection_s"]
        # Total should be faster
        assert after["timings"]["total_train_eval_s"] < before["timings"]["total_train_eval_s"]
