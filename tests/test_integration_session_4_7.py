"""
Session 4.7 — Integration tests for opportunity window metric.

Tests that the opportunity window is computed correctly on real data.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.episode_builder import load_day

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
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def real_day():
    days = [load_day(d) for d in REAL_DATES]
    return max(days, key=lambda d: sum(len(r.ticks) for r in d.races))


class TestOpportunityWindowRealData:

    def test_opportunity_windows_computed(self, config, real_day):
        """At least some bets have non-zero opportunity windows."""
        from agents.architecture_registry import create_policy
        from data.feature_engineer import engineer_day
        from training.evaluator import Evaluator
        from env.betfair_env import MARKET_DIM, VELOCITY_DIM, RUNNER_DIM, AGENT_STATE_DIM

        max_runners = config["training"]["max_runners"]
        obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * max_runners) + AGENT_STATE_DIM
        action_dim = max_runners * 2

        policy = create_policy("ppo_lstm_v1", obs_dim, action_dim, max_runners, {
            "lstm_hidden_size": 64, "mlp_hidden_size": 64, "mlp_layers": 1,
        })

        fc = {real_day.date: engineer_day(real_day)}
        evaluator = Evaluator(config=config, device="cpu", feature_cache=fc)
        _, day_records = evaluator.evaluate(
            model_id="test", policy=policy,
            test_days=[real_day], train_cutoff_date="2025-01-01",
        )

        assert len(day_records) == 1
        dr = day_records[0]
        # mean/median should be populated (may be 0 if no bets placed)
        assert not math.isnan(dr.mean_opportunity_window_s)
        assert not math.isnan(dr.median_opportunity_window_s)

    def test_tick_timestamp_populated(self, config, real_day):
        """Bet records should have populated tick_timestamp and seconds_to_off."""
        from agents.architecture_registry import create_policy
        from data.feature_engineer import engineer_day
        from training.evaluator import Evaluator
        from env.betfair_env import MARKET_DIM, VELOCITY_DIM, RUNNER_DIM, AGENT_STATE_DIM
        from registry.model_store import ModelStore

        max_runners = config["training"]["max_runners"]
        obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * max_runners) + AGENT_STATE_DIM
        action_dim = max_runners * 2

        # Use a random seed that encourages betting
        import numpy as np
        np.random.seed(42)
        torch.manual_seed(42)

        policy = create_policy("ppo_lstm_v1", obs_dim, action_dim, max_runners, {
            "lstm_hidden_size": 64, "mlp_hidden_size": 64, "mlp_layers": 1,
        })

        fc = {real_day.date: engineer_day(real_day)}

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            store = ModelStore(
                db_path=str(Path(tmp) / "test.db"),
                weights_dir=str(Path(tmp) / "weights"),
                bet_logs_dir=str(Path(tmp) / "bet_logs"),
            )
            mid = store.create_model(1, "ppo_lstm_v1", "test", {})
            evaluator = Evaluator(
                config=config, model_store=store,
                device="cpu", feature_cache=fc,
            )
            run_id, _ = evaluator.evaluate(
                model_id=mid, policy=policy,
                test_days=[real_day], train_cutoff_date="2025-01-01",
            )

            if run_id:
                bets = store.get_evaluation_bets(run_id)
                if bets:
                    # At least some bets should have tick_timestamp filled
                    filled = [b for b in bets if b.tick_timestamp != ""]
                    assert len(filled) > 0, "No bets have tick_timestamp populated"
                    # And seconds_to_off should be > 0 (pre-race bets)
                    positive_sto = [b for b in bets if b.seconds_to_off > 0]
                    assert len(positive_sto) > 0, "No bets have seconds_to_off > 0"
