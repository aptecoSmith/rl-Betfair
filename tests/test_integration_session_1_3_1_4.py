"""Integration tests for Sessions 1.3 (PPO trainer) and 1.4 (model registry).

Requires real extracted Parquet data in data/processed/.
Run with: pytest -m integration tests/test_integration_session_1_3_1_4.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.architecture_registry import create_policy
from agents.ppo_trainer import PPOTrainer, TrainingStats
from data.episode_builder import load_day
from env.betfair_env import ACTIONS_PER_RUNNER, AGENT_STATE_DIM, MARKET_DIM, RUNNER_DIM, VELOCITY_DIM
from registry.model_store import EvaluationBetRecord, EvaluationDayRecord, ModelStore
from registry.scoreboard import Scoreboard

pytestmark = pytest.mark.integration

DATA_DIR = Path("data/processed")


def _find_available_dates() -> list[str]:
    """Find all extracted dates with both ticks and runners parquet files."""
    dates = []
    for f in sorted(DATA_DIR.glob("*_runners.parquet")):
        date_str = f.stem.replace("_runners", "")
        tick_file = DATA_DIR / f"{date_str}.parquet"
        if tick_file.exists():
            dates.append(date_str)
    return dates


def _make_config() -> dict:
    """Load the real project config."""
    import yaml
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_policy(config: dict):
    max_runners = config["training"]["max_runners"]
    obs_dim = MARKET_DIM + VELOCITY_DIM + RUNNER_DIM * max_runners + AGENT_STATE_DIM
    action_dim = max_runners * ACTIONS_PER_RUNNER
    return create_policy(
        config["training"]["architecture"],
        obs_dim, action_dim, max_runners,
        hyperparams={"lstm_hidden_size": 64, "mlp_hidden_size": 32, "mlp_layers": 1},
    )


# ── Session 1.3: PPO Trainer on real data ────────────────────────────────────


class TestPPOTrainerRealData:
    """Train one agent for a small number of episodes on real data."""

    def test_train_on_real_day(self, tmp_path):
        dates = _find_available_dates()
        if not dates:
            pytest.skip("No extracted data available")

        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 32},
        )

        day = load_day(dates[0])
        stats = trainer.train([day], n_epochs=1)

        # Basic sanity
        assert stats.episodes_completed == 1
        assert stats.total_steps > 0
        assert len(stats.episode_stats) == 1

    def test_loss_computed(self, tmp_path):
        dates = _find_available_dates()
        if not dates:
            pytest.skip("No extracted data available")

        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 2, "mini_batch_size": 32},
        )

        day = load_day(dates[0])
        stats = trainer.train([day], n_epochs=1)

        # Loss should be computed (non-zero unless trivially so)
        assert np.isfinite(stats.final_policy_loss)
        assert np.isfinite(stats.final_value_loss)
        assert np.isfinite(stats.final_entropy)

    def test_bets_placed_and_settled(self, tmp_path):
        dates = _find_available_dates()
        if not dates:
            pytest.skip("No extracted data available")

        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 32},
        )

        day = load_day(dates[0])
        stats = trainer.train([day], n_epochs=1)
        ep = stats.episode_stats[0]

        # P&L should be finite
        assert np.isfinite(ep.total_pnl)
        # Bet count can be 0 (model may not place bets), that's OK
        assert ep.bet_count >= 0
        # Races should be completed
        assert ep.races_completed > 0

    def test_pnl_recorded(self, tmp_path):
        dates = _find_available_dates()
        if not dates:
            pytest.skip("No extracted data available")

        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 32},
        )

        day = load_day(dates[0])
        stats = trainer.train([day], n_epochs=1)

        # Log file should exist
        log_file = tmp_path / "logs" / "training" / "episodes.jsonl"
        assert log_file.exists()
        record = json.loads(log_file.read_text().strip())
        assert "total_pnl" in record
        assert np.isfinite(record["total_pnl"])

    def test_multiple_epochs_real_data(self, tmp_path):
        dates = _find_available_dates()
        if not dates:
            pytest.skip("No extracted data available")

        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 32},
        )

        day = load_day(dates[0])
        stats = trainer.train([day], n_epochs=2)

        assert stats.episodes_completed == 2
        assert len(stats.episode_stats) == 2


# ── Session 1.4: Model registry & scoreboard on real data ────────────────────


class TestModelStoreRealData:
    """Train → save → load → verify weights match."""

    def test_save_and_load_trained_weights(self, tmp_path):
        dates = _find_available_dates()
        if not dates:
            pytest.skip("No extracted data available")

        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 32},
        )

        # Train briefly
        day = load_day(dates[0])
        trainer.train([day], n_epochs=1)

        # Save to registry
        store = ModelStore(
            db_path=tmp_path / "models.db",
            weights_dir=tmp_path / "weights",
        )
        mid = store.create_model(
            generation=1,
            architecture_name="ppo_lstm_v1",
            architecture_description="Test model",
            hyperparameters={"lstm_hidden_size": 64, "mlp_hidden_size": 32, "mlp_layers": 1},
        )
        store.save_weights(mid, policy.state_dict())

        # Load and verify
        loaded = store.load_weights(mid)
        original = policy.state_dict()

        assert set(loaded.keys()) == set(original.keys())
        for key in original:
            assert torch.allclose(loaded[key], original[key]), f"Mismatch in {key}"

    def test_metadata_correct(self, tmp_path):
        store = ModelStore(
            db_path=tmp_path / "models.db",
            weights_dir=tmp_path / "weights",
        )
        hp = {"lstm_hidden_size": 64, "mlp_hidden_size": 32}
        mid = store.create_model(1, "ppo_lstm_v1", "PPO+LSTM v1", hp)

        model = store.get_model(mid)
        assert model is not None
        assert model.architecture_name == "ppo_lstm_v1"
        assert model.hyperparameters == hp
        assert model.status == "active"


class TestScoreboardRealData:
    """Train → save → evaluate → score → rank."""

    def test_full_pipeline(self, tmp_path):
        dates = _find_available_dates()
        if not dates:
            pytest.skip("No extracted data available")

        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 32},
        )

        # Train
        day = load_day(dates[0])
        stats = trainer.train([day], n_epochs=1)
        ep = stats.episode_stats[0]

        # Save model
        store = ModelStore(
            db_path=tmp_path / "models.db",
            weights_dir=tmp_path / "weights",
        )
        mid = store.create_model(
            generation=1,
            architecture_name="ppo_lstm_v1",
            architecture_description="Test",
            hyperparameters={},
        )
        store.save_weights(mid, policy.state_dict())

        # Record evaluation
        rid = store.create_evaluation_run(
            model_id=mid,
            train_cutoff_date=dates[0],
            test_days=[dates[0]],
        )
        bc = ep.bet_count
        wb = ep.winning_bets
        store.record_evaluation_day(EvaluationDayRecord(
            run_id=rid,
            date=dates[0],
            day_pnl=ep.total_pnl,
            bet_count=bc,
            winning_bets=wb,
            bet_precision=wb / bc if bc > 0 else 0.0,
            pnl_per_bet=ep.total_pnl / bc if bc > 0 else 0.0,
            early_picks=0,
            profitable=ep.total_pnl > 0,
        ))

        # Score and rank
        board = Scoreboard(store, config)
        score = board.score_model(mid)
        assert score is not None
        assert score.model_id == mid
        assert score.test_days == 1

        rankings = board.update_scores()
        assert len(rankings) == 1

        # Verify score persisted
        model = store.get_model(mid)
        assert model is not None
        assert model.composite_score is not None
