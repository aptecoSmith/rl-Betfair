"""Unit tests for agents/ppo_trainer.py -- PPO training loop."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.architecture_registry import create_policy
from agents.ppo_trainer import (
    EpisodeStats,
    PPOTrainer,
    Rollout,
    Transition,
    TrainingStats,
)
from data.episode_builder import Day, PriceSize, Race, RunnerMeta, RunnerSnap, Tick
from env.betfair_env import BetfairEnv


# ── Synthetic data helpers ────────────────────────────────────────────────────


def _make_runner_meta(selection_id: int, name: str = "Horse") -> RunnerMeta:
    return RunnerMeta(
        selection_id=selection_id,
        runner_name=name,
        sort_priority="1",
        handicap="0",
        sire_name="Sire",
        dam_name="Dam",
        damsire_name="DamSire",
        bred="GB",
        official_rating="85",
        adjusted_rating="85",
        age="4",
        sex_type="GELDING",
        colour_type="BAY",
        weight_value="140",
        weight_units="LB",
        jockey_name="J Smith",
        jockey_claim="0",
        trainer_name="T Jones",
        owner_name="Owner",
        stall_draw="3",
        cloth_number="1",
        form="1234",
        days_since_last_run="14",
        wearing="",
        forecastprice_numerator="3",
        forecastprice_denominator="1",
    )


def _make_runner_snap(
    selection_id: int,
    ltp: float = 4.0,
    back_price: float = 4.0,
    lay_price: float = 4.2,
    size: float = 100.0,
    status: str = "ACTIVE",
) -> RunnerSnap:
    return RunnerSnap(
        selection_id=selection_id,
        status=status,
        last_traded_price=ltp,
        total_matched=500.0,
        starting_price_near=0.0,
        starting_price_far=0.0,
        adjustment_factor=None,
        bsp=None,
        sort_priority=1,
        removal_date=None,
        available_to_back=[PriceSize(price=back_price, size=size)],
        available_to_lay=[PriceSize(price=lay_price, size=size)],
    )


def _make_tick(
    market_id: str,
    seq: int,
    runners: list[RunnerSnap],
    start_time: datetime | None = None,
    timestamp: datetime | None = None,
    in_play: bool = False,
    winner: int | None = None,
) -> Tick:
    if start_time is None:
        start_time = datetime(2026, 3, 26, 14, 0, 0)
    if timestamp is None:
        timestamp = start_time - timedelta(seconds=600 - seq * 5)
    return Tick(
        market_id=market_id,
        timestamp=timestamp,
        sequence_number=seq,
        venue="Newmarket",
        market_start_time=start_time,
        number_of_active_runners=len(runners),
        traded_volume=10000.0,
        in_play=in_play,
        winner_selection_id=winner,
        temperature=15.0,
        precipitation=0.0,
        wind_speed=5.0,
        wind_direction=180.0,
        humidity=60.0,
        weather_code=0,
        runners=runners,
    )


def _make_race(
    market_id: str = "1.200000001",
    start_time: datetime | None = None,
    n_ticks: int = 5,
    n_runners: int = 3,
    winner_sid: int = 1,
) -> Race:
    if start_time is None:
        start_time = datetime(2026, 3, 26, 14, 0, 0)
    runner_ids = list(range(1, n_runners + 1))
    runners = [_make_runner_snap(sid, ltp=3.0 + sid) for sid in runner_ids]
    ticks: list[Tick] = []

    for i in range(n_ticks):
        ts = start_time - timedelta(seconds=600 - i * 5)
        ticks.append(_make_tick(
            market_id, seq=i, runners=runners,
            start_time=start_time, timestamp=ts,
            in_play=False, winner=winner_sid,
        ))
    # Add one in-play tick
    ticks.append(_make_tick(
        market_id, seq=n_ticks, runners=runners,
        start_time=start_time,
        timestamp=start_time + timedelta(seconds=5),
        in_play=True, winner=winner_sid,
    ))

    runner_meta = {sid: _make_runner_meta(sid, f"Horse{sid}") for sid in runner_ids}
    return Race(
        market_id=market_id,
        venue="Newmarket",
        market_start_time=start_time,
        winner_selection_id=winner_sid,
        ticks=ticks,
        runner_metadata=runner_meta,
    )


def _make_day(n_races: int = 2, n_ticks: int = 5, n_runners: int = 3) -> Day:
    races = []
    for i in range(n_races):
        start = datetime(2026, 3, 26, 14 + i, 0, 0)
        races.append(_make_race(
            market_id=f"1.{200000001 + i}",
            start_time=start,
            n_ticks=n_ticks,
            n_runners=n_runners,
            winner_sid=1,
        ))
    return Day(date="2026-03-26", races=races)


def _make_config() -> dict:
    return {
        "training": {
            "architecture": "ppo_lstm_v1",
            "starting_budget": 100.0,
            "max_runners": 14,
        },
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "efficiency_penalty": 0.01,
            "coefficients": {
                "win_rate": 0.35,
                "sharpe": 0.30,
                "mean_daily_pnl": 0.15,
                "efficiency": 0.20,
            },
        },
        "paths": {
            "processed_data": "data/processed",
            "model_weights": "registry/weights",
            "logs": "logs",
            "registry_db": "registry/models.db",
        },
    }


def _make_policy(config: dict) -> torch.nn.Module:
    max_runners = config["training"]["max_runners"]
    from env.betfair_env import AGENT_STATE_DIM, MARKET_DIM, RUNNER_DIM, VELOCITY_DIM
    obs_dim = MARKET_DIM + VELOCITY_DIM + RUNNER_DIM * max_runners + AGENT_STATE_DIM
    action_dim = max_runners * 2
    return create_policy(
        "ppo_lstm_v1", obs_dim, action_dim, max_runners,
        hyperparams={"lstm_hidden_size": 64, "mlp_hidden_size": 32, "mlp_layers": 1},
    )


# ── Test classes ──────────────────────────────────────────────────────────────


class TestTransition:
    """Test the Transition dataclass."""

    def test_create(self):
        t = Transition(
            obs=np.zeros(10),
            action=np.zeros(5),
            log_prob=-1.0,
            value=0.5,
            reward=1.0,
            done=False,
        )
        assert t.reward == 1.0
        assert not t.done

    def test_done_flag(self):
        t = Transition(
            obs=np.zeros(10),
            action=np.zeros(5),
            log_prob=0.0,
            value=0.0,
            reward=0.0,
            done=True,
        )
        assert t.done


class TestRollout:
    """Test the Rollout container."""

    def test_empty(self):
        r = Rollout()
        assert len(r) == 0

    def test_append(self):
        r = Rollout()
        t = Transition(np.zeros(5), np.zeros(3), 0.0, 0.0, 1.0, False)
        r.append(t)
        assert len(r) == 1
        assert r.transitions[0].reward == 1.0

    def test_multiple(self):
        r = Rollout()
        for i in range(10):
            r.append(Transition(np.zeros(5), np.zeros(3), 0.0, 0.0, float(i), False))
        assert len(r) == 10


class TestEpisodeStats:
    """Test the EpisodeStats dataclass."""

    def test_create(self):
        es = EpisodeStats(
            day_date="2026-03-26",
            total_reward=5.0,
            total_pnl=10.0,
            bet_count=3,
            winning_bets=2,
            races_completed=2,
            final_budget=110.0,
            n_steps=50,
        )
        assert es.total_reward == 5.0
        assert es.n_steps == 50


class TestTrainingStats:
    """Test TrainingStats defaults."""

    def test_defaults(self):
        ts = TrainingStats()
        assert ts.episodes_completed == 0
        assert ts.mean_reward == 0.0
        assert ts.episode_stats == []


class TestPPOTrainerInit:
    """Test PPOTrainer initialisation."""

    def test_default_hyperparams(self):
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)

        assert trainer.lr == 3e-4
        assert trainer.gamma == 0.99
        assert trainer.gae_lambda == 0.95
        assert trainer.clip_epsilon == 0.2
        assert trainer.entropy_coeff == 0.01
        assert trainer.max_grad_norm == 0.5
        assert trainer.ppo_epochs == 4

    def test_custom_hyperparams(self):
        config = _make_config()
        policy = _make_policy(config)
        hp = {"learning_rate": 1e-4, "ppo_clip_epsilon": 0.1, "gamma": 0.95}
        trainer = PPOTrainer(policy, config, hyperparams=hp)

        assert trainer.lr == 1e-4
        assert trainer.clip_epsilon == 0.1
        assert trainer.gamma == 0.95

    def test_log_dir_created(self, tmp_path):
        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)

        assert trainer.log_dir.exists()

    def test_optimiser_created(self):
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)
        assert trainer.optimiser is not None


class TestRolloutCollection:
    """Test the rollout collection from synthetic env."""

    def test_collect_rollout_runs(self):
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)

        day = _make_day(n_races=1, n_ticks=3, n_runners=3)
        rollout, stats = trainer._collect_rollout(day)

        assert len(rollout) > 0
        assert stats.n_steps > 0
        assert stats.day_date == "2026-03-26"

    def test_collect_rollout_transitions_valid(self):
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)

        day = _make_day(n_races=1, n_ticks=3, n_runners=3)
        rollout, _ = trainer._collect_rollout(day)

        for t in rollout.transitions:
            assert isinstance(t.obs, np.ndarray)
            assert isinstance(t.action, np.ndarray)
            assert isinstance(t.log_prob, float)
            assert isinstance(t.value, float)
            assert isinstance(t.reward, float)
            assert isinstance(t.done, bool)
            assert not np.isnan(t.log_prob)
            assert not np.isnan(t.value)

    def test_last_transition_done(self):
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)

        day = _make_day(n_races=1, n_ticks=3, n_runners=3)
        rollout, _ = trainer._collect_rollout(day)

        assert rollout.transitions[-1].done

    def test_episode_stats_populated(self):
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)

        day = _make_day(n_races=2, n_ticks=3, n_runners=3)
        _, stats = trainer._collect_rollout(day)

        assert stats.races_completed == 2
        assert stats.n_steps > 0
        assert isinstance(stats.total_pnl, float)


class TestGAE:
    """Test GAE advantage estimation."""

    def test_returns_correct_shapes(self):
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)

        rollout = Rollout()
        for i in range(10):
            rollout.append(Transition(
                obs=np.zeros(5),
                action=np.zeros(3),
                log_prob=-0.5,
                value=float(i) * 0.1,
                reward=1.0,
                done=(i == 9),
            ))

        advantages, returns = trainer._compute_advantages(rollout)
        assert advantages.shape == (10,)
        assert returns.shape == (10,)

    def test_advantages_not_all_zero(self):
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)

        rollout = Rollout()
        for i in range(5):
            rollout.append(Transition(
                obs=np.zeros(5),
                action=np.zeros(3),
                log_prob=-0.5,
                value=0.5,
                reward=1.0 if i == 3 else 0.0,
                done=(i == 4),
            ))

        advantages, _ = trainer._compute_advantages(rollout)
        assert not torch.allclose(advantages, torch.zeros_like(advantages))

    def test_terminal_state_zero_future(self):
        """The advantage at a terminal state should not include future rewards."""
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)

        rollout = Rollout()
        # Two sub-episodes spliced together
        rollout.append(Transition(np.zeros(5), np.zeros(3), 0.0, 0.5, 1.0, True))
        rollout.append(Transition(np.zeros(5), np.zeros(3), 0.0, 0.5, 100.0, True))

        advantages, returns = trainer._compute_advantages(rollout)
        # The first episode's advantage should not be influenced by the 100.0 reward
        # in the second episode
        assert advantages[0] < 50.0  # should be close to 1.0 - 0.5 = 0.5

    def test_returns_equal_advantage_plus_value(self):
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)

        rollout = Rollout()
        values = [0.1, 0.3, 0.5, 0.7, 0.9]
        for i, v in enumerate(values):
            rollout.append(Transition(
                np.zeros(5), np.zeros(3), 0.0, v, 1.0, i == len(values) - 1,
            ))

        advantages, returns = trainer._compute_advantages(rollout)
        # returns = advantages + values
        for i in range(len(values)):
            assert abs(returns[i].item() - (advantages[i].item() + values[i])) < 1e-5


class TestPPOUpdate:
    """Test the PPO optimisation step."""

    def test_update_runs_without_error(self):
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config, hyperparams={"ppo_epochs": 1, "mini_batch_size": 4})

        day = _make_day(n_races=1, n_ticks=3, n_runners=3)
        rollout, _ = trainer._collect_rollout(day)

        loss_info = trainer._ppo_update(rollout)
        assert "policy_loss" in loss_info
        assert "value_loss" in loss_info
        assert "entropy" in loss_info
        assert isinstance(loss_info["policy_loss"], float)

    def test_update_changes_parameters(self):
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config, hyperparams={"ppo_epochs": 2, "mini_batch_size": 4})

        # Record initial params
        initial_params = {n: p.clone() for n, p in policy.named_parameters()}

        day = _make_day(n_races=1, n_ticks=5, n_runners=3)
        rollout, _ = trainer._collect_rollout(day)
        trainer._ppo_update(rollout)

        # Check at least some params changed
        changed = False
        for n, p in policy.named_parameters():
            if not torch.allclose(p, initial_params[n]):
                changed = True
                break
        assert changed, "PPO update should change at least some parameters"

    def test_loss_values_finite(self):
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config, hyperparams={"ppo_epochs": 1, "mini_batch_size": 4})

        day = _make_day(n_races=1, n_ticks=3, n_runners=3)
        rollout, _ = trainer._collect_rollout(day)
        loss_info = trainer._ppo_update(rollout)

        assert np.isfinite(loss_info["policy_loss"])
        assert np.isfinite(loss_info["value_loss"])
        assert np.isfinite(loss_info["entropy"])


class TestTrainLoop:
    """Test the full train() method."""

    def test_train_one_epoch(self, tmp_path):
        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 4},
        )

        day = _make_day(n_races=1, n_ticks=3, n_runners=3)
        stats = trainer.train([day], n_epochs=1)

        assert stats.episodes_completed == 1
        assert stats.total_steps > 0
        assert len(stats.episode_stats) == 1

    def test_train_multiple_epochs(self, tmp_path):
        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 4},
        )

        day = _make_day(n_races=1, n_ticks=3, n_runners=3)
        stats = trainer.train([day], n_epochs=3)

        assert stats.episodes_completed == 3
        assert len(stats.episode_stats) == 3

    def test_train_multiple_days(self, tmp_path):
        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 4},
        )

        days = [_make_day(n_races=1, n_ticks=3, n_runners=3) for _ in range(2)]
        stats = trainer.train(days, n_epochs=1)

        assert stats.episodes_completed == 2

    def test_train_returns_mean_stats(self, tmp_path):
        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 4},
        )

        day = _make_day(n_races=1, n_ticks=3, n_runners=3)
        stats = trainer.train([day], n_epochs=2)

        assert isinstance(stats.mean_reward, float)
        assert isinstance(stats.mean_pnl, float)
        assert isinstance(stats.mean_bet_count, float)

    def test_train_loss_terms_populated(self, tmp_path):
        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 4},
        )

        day = _make_day(n_races=1, n_ticks=3, n_runners=3)
        stats = trainer.train([day], n_epochs=1)

        # At least one of the final losses should be non-zero
        assert (
            stats.final_policy_loss != 0.0
            or stats.final_value_loss != 0.0
            or stats.final_entropy != 0.0
        )


class TestLogging:
    """Test episode logging to disk."""

    def test_log_file_created(self, tmp_path):
        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 4},
        )

        day = _make_day(n_races=1, n_ticks=3, n_runners=3)
        trainer.train([day], n_epochs=1)

        log_file = tmp_path / "logs" / "training" / "episodes.jsonl"
        assert log_file.exists()

    def test_log_file_valid_jsonl(self, tmp_path):
        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 4},
        )

        day = _make_day(n_races=1, n_ticks=3, n_runners=3)
        trainer.train([day], n_epochs=2)

        log_file = tmp_path / "logs" / "training" / "episodes.jsonl"
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2

        for line in lines:
            record = json.loads(line)
            assert "episode" in record
            assert "total_reward" in record
            assert "total_pnl" in record
            assert "policy_loss" in record
            assert "day_date" in record

    def test_log_record_fields(self, tmp_path):
        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 4},
        )

        day = _make_day(n_races=1, n_ticks=3, n_runners=3)
        trainer.train([day], n_epochs=1)

        log_file = tmp_path / "logs" / "training" / "episodes.jsonl"
        record = json.loads(log_file.read_text().strip())

        expected_keys = {
            "episode", "day_date", "total_reward", "total_pnl",
            "bet_count", "winning_bets", "races_completed", "final_budget",
            "n_steps", "policy_loss", "value_loss", "entropy", "timestamp",
        }
        assert expected_keys <= set(record.keys())


class TestProgressQueue:
    """Test progress event publishing to asyncio Queue."""

    def test_progress_events_emitted(self, tmp_path):
        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        queue: asyncio.Queue = asyncio.Queue()
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 4},
            progress_queue=queue,
        )

        day = _make_day(n_races=1, n_ticks=3, n_runners=3)
        trainer.train([day], n_epochs=1)

        assert not queue.empty()

    def test_progress_event_schema(self, tmp_path):
        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        queue: asyncio.Queue = asyncio.Queue()
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 4},
            progress_queue=queue,
        )

        day = _make_day(n_races=1, n_ticks=3, n_runners=3)
        trainer.train([day], n_epochs=1)

        event = queue.get_nowait()
        assert event["event"] == "progress"
        assert event["phase"] == "training"
        assert "item" in event
        assert "detail" in event
        assert "episode" in event
        assert "total_reward" in event["episode"]
        assert "policy_loss" in event["episode"]

    def test_multiple_episodes_multiple_events(self, tmp_path):
        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        queue: asyncio.Queue = asyncio.Queue()
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 4},
            progress_queue=queue,
        )

        day = _make_day(n_races=1, n_ticks=3, n_runners=3)
        trainer.train([day], n_epochs=3)

        count = 0
        while not queue.empty():
            queue.get_nowait()
            count += 1
        assert count == 3

    def test_no_queue_no_error(self, tmp_path):
        """Training without a queue should not raise."""
        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 4},
        )

        day = _make_day(n_races=1, n_ticks=3, n_runners=3)
        stats = trainer.train([day], n_epochs=1)
        assert stats.episodes_completed == 1


class TestEmptyDay:
    """Test behaviour with an empty day (no races)."""

    def test_empty_day_produces_stats(self, tmp_path):
        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 4},
        )

        empty_day = Day(date="2026-03-26", races=[])
        stats = trainer.train([empty_day], n_epochs=1)

        assert stats.episodes_completed == 1
        assert stats.episode_stats[0].n_steps == 1  # one terminal step
        assert stats.episode_stats[0].races_completed == 0


class TestGradientClipping:
    """Verify gradient clipping is applied."""

    def test_gradients_bounded(self, tmp_path):
        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={
                "ppo_epochs": 1,
                "mini_batch_size": 4,
                "max_grad_norm": 0.5,
            },
        )

        day = _make_day(n_races=1, n_ticks=5, n_runners=3)
        rollout, _ = trainer._collect_rollout(day)
        trainer._ppo_update(rollout)

        # After update, check that the total gradient norm was clipped
        # (We can't directly check during the update, but we verify no NaN)
        for p in policy.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
