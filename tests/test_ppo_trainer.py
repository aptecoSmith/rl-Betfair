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
from env.betfair_env import ACTIONS_PER_RUNNER, BetfairEnv


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
        race_status=None,
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
    action_dim = max_runners * ACTIONS_PER_RUNNER
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
        # entropy-control-v2: entropy_coeff is now log_alpha.exp() —
        # bit-exact equality to the literal doesn't survive the
        # log/exp round trip, so compare to machine-epsilon tolerance.
        assert trainer.entropy_coeff == pytest.approx(0.005, rel=1e-12)
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
            "raw_pnl_reward", "shaped_bonus",
            "bet_count", "winning_bets", "races_completed", "final_budget",
            "n_steps", "policy_loss", "value_loss", "entropy", "timestamp",
        }
        assert expected_keys <= set(record.keys())
        # Raw + shaped split must approximately reconstruct total_reward.
        assert record["raw_pnl_reward"] + record["shaped_bonus"] == pytest.approx(
            record["total_reward"], abs=1e-3,
        )


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


class TestEntropyAndCentering:
    """Session 03 — entropy coefficient default halved + reward centering.

    plans/naked-clip-and-stability/session_prompts/03_entropy_and_centering.md
    """

    def test_entropy_default_is_halved(self):
        """Fresh PPOTrainer with no explicit ``entropy_coefficient`` hp
        picks up the halved default (0.005 per §13). Post
        entropy-control-v2 the value is ``log_alpha.exp()`` and we
        tolerate a machine-epsilon drift from the literal."""
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)
        assert trainer._log_alpha.exp().item() == pytest.approx(
            0.005, rel=1e-12,
        )
        assert trainer.entropy_coeff == pytest.approx(0.005, rel=1e-12)

    def test_entropy_explicit_hp_overrides_default(self):
        """Explicit ``entropy_coefficient`` hp still wins (GA path)."""
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config, hyperparams={"entropy_coefficient": 0.02},
        )
        assert trainer.entropy_coeff == pytest.approx(0.02, rel=1e-12)

    def test_reward_baseline_initialises_on_first_episode(self):
        """First call sets EMA to the observed value exactly; second
        call applies the EMA blend (α=0.01)."""
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)

        assert trainer._reward_ema == 0.0
        assert trainer._reward_ema_initialised is False

        trainer._update_reward_baseline(42.0)
        assert trainer._reward_ema == 42.0
        assert trainer._reward_ema_initialised is True

        trainer._update_reward_baseline(-58.0)
        expected = 0.99 * 42.0 + 0.01 * (-58.0)
        assert abs(trainer._reward_ema - expected) < 1e-9

    def test_reward_baseline_ema_update_is_monotonic(self):
        """A monotonically-increasing reward sequence yields a
        monotonically-increasing EMA."""
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)

        rewards = [0.0, 10.0, 20.0, 50.0, 100.0, 200.0]
        ema_trace: list[float] = []
        for r in rewards:
            trainer._update_reward_baseline(r)
            ema_trace.append(trainer._reward_ema)

        for i in range(1, len(ema_trace)):
            assert ema_trace[i] >= ema_trace[i - 1]

    def test_centering_preserves_advantage_ordering(self):
        """Synthetic rollout where every transition is terminal
        (done=True, value=0). Delta_t = r_t - ema, last_gae resets at
        each done → advantage_t = r_t - ema. Per-mini-batch
        normalisation subtracts the mean and divides by std; the
        constant ``ema`` cancels, so centered and uncentered
        normalised advantages agree to floating-point tolerance."""
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)

        rewards = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        rollout = Rollout()
        for r in rewards:
            rollout.append(Transition(
                obs=np.zeros(10, dtype=np.float32),
                action=np.zeros(1, dtype=np.float32),
                log_prob=0.0,
                value=0.0,
                reward=r,
                done=True,
                training_reward=r,
            ))

        # Uncentered pass: EMA = 0 (not initialised), so no subtraction.
        adv_uncentered, _ = trainer._compute_advantages(rollout)

        # Centered pass: force a non-zero EMA.
        trainer._reward_ema = 4.5
        trainer._reward_ema_initialised = True
        adv_centered, _ = trainer._compute_advantages(rollout)

        def normalise(t: torch.Tensor) -> torch.Tensor:
            return (t - t.mean()) / (t.std() + 1e-8)

        norm_unc = normalise(adv_uncentered)
        norm_cen = normalise(adv_centered)
        assert torch.allclose(norm_unc, norm_cen, atol=1e-5)

    def test_centering_fixes_uniformly_negative_rewards(self):
        """Rollout with all rewards in [-900, -200] (transformer
        0a8cacd3 scale). After centering + per-mini-batch
        normalisation, advantages have mean ≈ 0 (normalisation does
        that regardless); without centering the UN-NORMALISED
        advantage tensor is strongly negative-biased. This is the
        sanity-check that centering shifts the pre-normalisation
        signal the way we expect."""
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)

        rewards = [-900.0, -750.0, -600.0, -400.0, -350.0, -250.0, -220.0, -210.0]
        rollout = Rollout()
        for r in rewards:
            rollout.append(Transition(
                obs=np.zeros(10, dtype=np.float32),
                action=np.zeros(1, dtype=np.float32),
                log_prob=0.0,
                value=0.0,
                reward=r,
                done=True,
                training_reward=r,
            ))

        adv_uncentered, _ = trainer._compute_advantages(rollout)
        assert adv_uncentered.mean().item() < -300.0  # strongly negative

        trainer._reward_ema = float(np.mean(rewards))
        trainer._reward_ema_initialised = True
        adv_centered, _ = trainer._compute_advantages(rollout)

        assert abs(adv_centered.mean().item()) < 1e-3  # centered ≈ 0

    # -- Units regression: EMA must be per-step, not episode-sum ----
    #
    # Bug observed in the 2026-04-18 smoke probe (plans/naked-clip-
    # and-stability/lessons_learnt.md): ``_ppo_update`` was calling
    # ``self._update_reward_baseline(rollout_reward)`` where
    # ``rollout_reward`` was ``sum(tr.training_reward for tr in
    # transitions)`` — the episode SUM. But
    # ``_compute_advantages`` subtracts ``_reward_ema`` from each
    # PER-STEP ``tr.training_reward``. On a 4742-step rollout with
    # total reward −1551, the EMA landed at −1551, then on the next
    # rollout every per-step reward got shifted by +1551, GAE
    # accumulated to ~26,000, and value_loss exploded to 6.76e+08.
    # The pre-existing centering tests passed because they set
    # ``_reward_ema = np.mean(rewards)`` directly — exercising the
    # correct-units path while production used wrong units. These
    # tests lock the contract.

    # These tests exercise the CALLER contract of
    # ``_update_reward_baseline`` — they build a rollout and invoke
    # the exact code path ``_ppo_update`` uses to populate the EMA,
    # without needing the full policy forward pass. The rollout
    # shape matches what the real trainer produces; only the policy
    # network is bypassed.

    def _rollout_with_constant_reward(
        self, obs_dim: int, per_step: float, n_steps: int,
    ) -> Rollout:
        rollout = Rollout()
        for _ in range(n_steps):
            rollout.append(Transition(
                obs=np.zeros(obs_dim, dtype=np.float32),
                action=np.zeros(1, dtype=np.float32),
                log_prob=0.0,
                value=0.0,
                reward=per_step,
                done=False,
                training_reward=per_step,
            ))
        return rollout

    def _simulate_baseline_update_from_rollout(
        self, trainer: PPOTrainer, rollout: Rollout,
    ) -> None:
        """Run the exact ``_ppo_update`` lines that populate the EMA.

        Mirrors ``_ppo_update`` at agents/ppo_trainer.py:1304-ish so
        a future change to the aggregation (sum vs mean) fails this
        test immediately, without paying the cost of a real policy
        forward/backward."""
        transitions = list(rollout.transitions)
        per_step_mean_reward = (
            float(sum(tr.training_reward for tr in transitions))
            / max(1, len(transitions))
        )
        trainer._update_reward_baseline(per_step_mean_reward)

    def test_reward_baseline_stores_per_step_mean_not_episode_sum(
        self,
    ):
        """After a rollout with per-step rewards near zero and a long
        step count, the EMA must end up near the per-step mean
        (~rollout_sum / n_steps), NOT near the episode sum. Would
        have caught the 2026-04-18 probe bug immediately — the old
        code stored the sum, producing an EMA of -1551 from a
        4742-step rollout with per-step reward ~-0.33."""
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)

        n_steps = 1000
        per_step_reward = -0.03
        rollout = self._rollout_with_constant_reward(
            obs_dim=10, per_step=per_step_reward, n_steps=n_steps,
        )
        self._simulate_baseline_update_from_rollout(trainer, rollout)

        expected = per_step_reward
        assert abs(trainer._reward_ema - expected) < 1e-6, (
            f"_reward_ema={trainer._reward_ema} expected per-step mean "
            f"{expected}; episode sum would be "
            f"{per_step_reward * n_steps}. If the EMA is close to "
            f"that, the unit mismatch is back."
        )
        # Belt-and-braces: EMA magnitude stays small relative to the
        # episode sum (the bug magnitude).
        assert abs(trainer._reward_ema) < abs(per_step_reward * n_steps) * 0.1

    def test_centering_does_not_explode_returns_across_rollouts(self):
        """Two back-to-back rollouts with the same per-step reward
        scale. If the EMA stored episode-sum (the bug), ep2's returns
        blow up by ~n_steps× and the value head with it. With the
        fix, ep2 returns stay in the same O(1/(1-γλ)) magnitude as
        the centered per-step reward."""
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)

        per_step = -0.05
        n = 2000

        # Ep1: establish the EMA using the production aggregation.
        rollout1 = self._rollout_with_constant_reward(
            obs_dim=10, per_step=per_step, n_steps=n,
        )
        self._simulate_baseline_update_from_rollout(trainer, rollout1)
        ema_after_ep1 = trainer._reward_ema

        # Ep2: compute advantages under the post-ep1 EMA.
        rollout2 = self._rollout_with_constant_reward(
            obs_dim=10, per_step=per_step, n_steps=n,
        )
        _advantages2, returns2 = trainer._compute_advantages(rollout2)

        # With correct units (EMA ~= per_step), centered reward per
        # step is ~0, GAE converges near 0, returns stay O(1).
        # With the old bug (EMA == episode sum), centered reward per
        # step jumps by ~100, returns accumulate to ~1700. Threshold
        # 50 is well below the bug magnitude and well above the
        # fixed magnitude.
        returns_mean = returns2.abs().mean().item()
        assert returns_mean < 50.0, (
            f"ep2 |returns|.mean()={returns_mean} — centering bug "
            f"likely re-introduced (EMA stored as episode sum rather "
            f"than per-step mean). ep1 EMA was {ema_after_ep1}."
        )

    def test_centering_units_match_between_update_and_advantage(self):
        """Semantic lock: the quantity fed into
        ``_update_reward_baseline`` must match the unit used inside
        ``_compute_advantages``. After simulating the production
        aggregation on a rollout whose per-step reward is constant
        C, the EMA should equal exactly C (first-call init), not
        C × n_steps."""
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)

        C = 0.123
        n = 500
        rollout = self._rollout_with_constant_reward(
            obs_dim=10, per_step=C, n_steps=n,
        )
        self._simulate_baseline_update_from_rollout(trainer, rollout)
        assert abs(trainer._reward_ema - C) < 1e-6, (
            f"_reward_ema={trainer._reward_ema} expected {C}; "
            f"episode sum = {C * n}, ratio = "
            f"{trainer._reward_ema / C if C else 'inf'}"
        )

    def test_real_ppo_update_feeds_per_step_mean_to_baseline(self):
        """End-to-end lock: spy on ``_update_reward_baseline`` while
        running the REAL ``_ppo_update`` against a collected rollout.
        The spy captures the exact value production code passes;
        asserting it equals ``sum / n`` (per-step mean) instead of
        ``sum`` (episode total) fails immediately if anyone reverts
        the call-site fix.

        This is the one test that protects against the two-file-drift
        failure mode — a future refactor can't skip updating the
        helper in the unit tests because this test doesn't use the
        helper at all."""
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 4},
        )

        day = _make_day(n_races=1, n_ticks=5, n_runners=3)
        rollout, _ = trainer._collect_rollout(day)
        transitions = list(rollout.transitions)
        assert len(transitions) > 1, (
            "expected a multi-step rollout to meaningfully "
            "distinguish per-step-mean from episode-sum"
        )
        episode_sum = sum(tr.training_reward for tr in transitions)
        expected_per_step_mean = episode_sum / len(transitions)

        recorded: list[float] = []
        orig = trainer._update_reward_baseline

        def spy(x: float) -> None:
            recorded.append(float(x))
            return orig(x)

        trainer._update_reward_baseline = spy  # type: ignore[assignment]
        trainer._ppo_update(rollout)

        assert len(recorded) == 1, (
            f"expected one baseline update per _ppo_update; got "
            f"{len(recorded)}"
        )
        passed_value = recorded[0]
        assert abs(passed_value - expected_per_step_mean) < 1e-6, (
            f"_update_reward_baseline received {passed_value}; "
            f"expected per-step mean {expected_per_step_mean} "
            f"(n={len(transitions)}, sum={episode_sum}). "
            f"If the value matches the episode sum, the 2026-04-18 "
            f"unit-mismatch bug has returned."
        )
        # The episode sum itself should be distinctly different —
        # guards against the degenerate n=1 case.
        assert abs(passed_value - episode_sum) > 1e-6 or len(transitions) == 1


# ── Target-entropy controller (entropy-control-v2 Session 01) ──────────────


class TestTargetEntropyController:
    """SAC-style target-entropy controller (``_update_entropy_coefficient``).

    Replaces the fixed ``entropy_coefficient`` with a learned
    ``log_alpha`` variable that a small separate Adam optimiser drives
    to hold the policy's forward-pass entropy at
    ``self._target_entropy``. See
    ``plans/entropy-control-v2/session_prompts/01_target_entropy_controller.md``.
    """

    def test_log_alpha_initialises_from_entropy_coefficient(self):
        """hp={'entropy_coefficient': 0.01} → log_alpha.exp() ≈ 0.01.
        Default (no hp) → 0.005."""
        import math
        config = _make_config()
        policy = _make_policy(config)

        trainer = PPOTrainer(
            policy, config, hyperparams={"entropy_coefficient": 0.01},
        )
        assert trainer._log_alpha.exp().item() == pytest.approx(
            0.01, rel=1e-6,
        )
        assert trainer._log_alpha.item() == pytest.approx(
            math.log(0.01), abs=1e-6,
        )
        assert trainer.entropy_coeff == pytest.approx(0.01, rel=1e-6)

        # Default path — no explicit entropy_coefficient hp.
        policy2 = _make_policy(config)
        trainer2 = PPOTrainer(policy2, config)
        assert trainer2._log_alpha.exp().item() == pytest.approx(
            0.005, rel=1e-6,
        )
        assert trainer2.entropy_coeff == pytest.approx(0.005, rel=1e-6)

    def test_controller_shrinks_alpha_when_entropy_above_target(self):
        """target=100, feed current_entropy=200 → log_alpha strictly
        smaller than before. Sign: target-current = -100, alpha_loss =
        -log_alpha × (-100) = 100 × log_alpha; gradient descent on a
        function whose derivative wrt log_alpha is +100 pushes log_alpha
        DOWN."""
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={
                "entropy_coefficient": 0.01,
                "target_entropy": 100.0,
                "alpha_lr": 1e-2,  # raise LR so one step moves meaningfully
            },
        )
        before = float(trainer._log_alpha.item())

        trainer._update_entropy_coefficient(current_entropy=200.0)

        after = float(trainer._log_alpha.item())
        assert after < before, (
            f"log_alpha should shrink when entropy > target; "
            f"before={before} after={after}"
        )
        assert trainer.entropy_coeff < 0.01

    def test_controller_grows_alpha_when_entropy_below_target(self):
        """target=100, feed current_entropy=50 → log_alpha strictly
        larger than before. Symmetric to the shrink case."""
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={
                "entropy_coefficient": 0.01,
                "target_entropy": 100.0,
                "alpha_lr": 1e-2,
            },
        )
        before = float(trainer._log_alpha.item())

        trainer._update_entropy_coefficient(current_entropy=50.0)

        after = float(trainer._log_alpha.item())
        assert after > before, (
            f"log_alpha should grow when entropy < target; "
            f"before={before} after={after}"
        )
        assert trainer.entropy_coeff > 0.01

    def test_log_alpha_clamped_within_bounds(self):
        """Stress with a pathological entropy mismatch to pull
        log_alpha hard toward each bound. After one step log_alpha
        sits at the clamp bound exactly, not outside it. Symmetric
        across upper and lower bounds."""
        import math
        # Lower clamp: pathological entropy >>> target with a huge LR
        # drives log_alpha to log(1e-5).
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={
                "entropy_coefficient": 0.01,
                "target_entropy": 100.0,
                "alpha_lr": 1e6,  # enormous so one step saturates
            },
        )
        trainer._update_entropy_coefficient(current_entropy=1e6)
        assert trainer._log_alpha.item() == pytest.approx(
            math.log(1e-5), abs=1e-6,
        )

        # Upper clamp: pathological entropy <<< target with a huge LR
        # drives log_alpha to log(0.1).
        policy2 = _make_policy(config)
        trainer2 = PPOTrainer(
            policy2, config,
            hyperparams={
                "entropy_coefficient": 0.01,
                "target_entropy": 100.0,
                "alpha_lr": 1e6,
            },
        )
        trainer2._update_entropy_coefficient(current_entropy=-1e6)
        assert trainer2._log_alpha.item() == pytest.approx(
            math.log(0.1), abs=1e-6,
        )

    def test_controller_optimizer_separate_from_policy(self):
        """The alpha controller is orthogonal to the policy optimiser
        — running ``_update_entropy_coefficient`` must NOT mutate the
        policy optimiser's state_dict. The alpha controller DOES
        move ``log_alpha``.

        Post Session 05 the alpha optimiser is SGD(momentum=0) so
        its state_dict is effectively empty — the essential
        assertion is therefore that ``log_alpha.item()`` changed,
        not that the optimiser's internal state did."""
        import copy
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"alpha_lr": 1e-3, "target_entropy": 100.0},
        )

        policy_state_before = copy.deepcopy(
            trainer.optimiser.state_dict()
        )
        log_alpha_before = float(trainer._log_alpha.item())

        trainer._update_entropy_coefficient(current_entropy=200.0)

        policy_state_after = trainer.optimiser.state_dict()
        log_alpha_after = float(trainer._log_alpha.item())

        # Policy optimiser state is unchanged — the alpha update does
        # not touch the policy's autograd graph.
        assert (
            policy_state_before["state"] == policy_state_after["state"]
        ), "policy optimiser state was mutated by the alpha update"

        # log_alpha DID move — the controller actually took a step.
        assert log_alpha_before != log_alpha_after, (
            "log_alpha did not change — did the controller actually "
            "take a step?"
        )

    def test_effective_entropy_coeff_matches_log_alpha_exp(self):
        """After ``_update_entropy_coefficient`` returns,
        ``self.entropy_coeff == self._log_alpha.exp().item()`` within
        floating-point epsilon."""
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={
                "entropy_coefficient": 0.01,
                "target_entropy": 100.0,
                "alpha_lr": 1e-3,
            },
        )

        # Drive the controller through a handful of steps at different
        # entropies so the coefficient meaningfully moves.
        for ent in (150.0, 80.0, 120.0, 95.0):
            trainer._update_entropy_coefficient(current_entropy=ent)
            assert trainer.entropy_coeff == pytest.approx(
                float(trainer._log_alpha.exp().item()), rel=1e-9,
            )

    def test_real_ppo_update_updates_log_alpha(self):
        """End-to-end: a real ``_ppo_update`` call must move
        ``log_alpha`` from its init value. Exercises the wired-in code
        path — unit tests on the controller method alone are
        insufficient per the 2026-04-18 units-mismatch lesson
        (plans/naked-clip-and-stability/lessons_learnt.md)."""
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={
                "entropy_coefficient": 0.005,
                "target_entropy": 112.0,
                "alpha_lr": 1e-4,
                "ppo_epochs": 1,
                "mini_batch_size": 4,
            },
        )
        before = float(trainer._log_alpha.item())

        day = _make_day(n_races=1, n_ticks=5, n_runners=3)
        rollout, _ = trainer._collect_rollout(day)
        trainer._ppo_update(rollout)

        after = float(trainer._log_alpha.item())
        assert after != before, (
            "log_alpha did not move during a real _ppo_update — the "
            "controller is not wired into the update path."
        )
        # Sanity: the new entropy_coeff matches log_alpha.exp() after
        # the update.
        assert trainer.entropy_coeff == pytest.approx(
            float(trainer._log_alpha.exp().item()), rel=1e-9,
        )

    def test_target_entropy_default_matches_session_06(self):
        """Default ``target_entropy`` is 150.0 (Session 06, 2026-04-19).
        The original 112.0 target (80% of fresh-init ep-1 entropy 139.6)
        sat below the action space's natural entropy floor — no alpha
        value could coax entropy below the floor, so the controller
        drove alpha all the way to the lower clamp without ever
        stabilising entropy. Target 150 sits ~+8% above fresh-init
        139, giving the controller real authority from ep1 onward.
        Any change to this default needs to co-land with a plan
        update."""
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)
        assert trainer._target_entropy == pytest.approx(150.0)

    def test_alpha_lr_default_matches_session_05(self):
        """Default ``alpha_lr`` is 1e-2 (entropy-control-v2 Session 05).
        The default moved 1e-4 → 3e-2 (Session 04) when the Adam
        version proved too timid, and 3e-2 → 1e-2 (Session 05) when
        the controller was reformulated as proportional SGD — SGD
        multiplies lr by the raw entropy error (O(20-50)), so
        lr=1e-2 gives per-episode log_alpha steps of O(0.2-0.5),
        enough to tighten alpha across several episodes without
        saturating in a single update."""
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)
        assert trainer._alpha_optimizer.param_groups[0]["lr"] == (
            pytest.approx(1e-2, rel=1e-9)
        )

    def test_alpha_optimizer_is_sgd_proportional_controller(self):
        """The alpha controller is plain SGD (momentum=0), not Adam.
        Proportional control needs the step size to scale with the
        entropy error; Adam's adaptive normalisation destroys that
        property. Pin the optimiser class so a future refactor
        can't silently revert to Adam."""
        import torch
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)
        assert isinstance(trainer._alpha_optimizer, torch.optim.SGD), (
            f"alpha optimiser is "
            f"{type(trainer._alpha_optimizer).__name__}; expected SGD"
        )
        assert trainer._alpha_optimizer.param_groups[0]["momentum"] == 0.0

    def test_controller_step_is_proportional_to_error(self):
        """SGD → step size scales linearly with the entropy error.
        Two controller invocations with errors differing by 10x
        should produce log_alpha deltas differing by 10x, in the
        same direction."""
        config = _make_config()
        policy = _make_policy(config)
        trainer_small = PPOTrainer(
            policy, config,
            hyperparams={
                "entropy_coefficient": 0.02,
                "target_entropy": 100.0,
                "alpha_lr": 1e-3,
            },
        )
        before_small = float(trainer_small._log_alpha.item())
        trainer_small._update_entropy_coefficient(current_entropy=110.0)
        delta_small = float(trainer_small._log_alpha.item()) - before_small

        policy2 = _make_policy(config)
        trainer_big = PPOTrainer(
            policy2, config,
            hyperparams={
                "entropy_coefficient": 0.02,
                "target_entropy": 100.0,
                "alpha_lr": 1e-3,
            },
        )
        before_big = float(trainer_big._log_alpha.item())
        trainer_big._update_entropy_coefficient(current_entropy=200.0)
        delta_big = float(trainer_big._log_alpha.item()) - before_big

        # Same direction (both down — entropy above target).
        assert delta_small < 0 and delta_big < 0
        # delta_big / delta_small ≈ error_big / error_small = 100/10 = 10.
        ratio = delta_big / delta_small
        assert 9.5 < ratio < 10.5, (
            f"expected ~10x ratio (proportional control); got {ratio}"
        )

    def test_alpha_lr_explicit_hp_overrides_default(self):
        """Explicit ``alpha_lr`` hp still wins — lets tests and
        pathological configs pin the controller's aggressiveness."""
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config, hyperparams={"alpha_lr": 1e-4},
        )
        assert trainer._alpha_optimizer.param_groups[0]["lr"] == (
            pytest.approx(1e-4, rel=1e-9)
        )

    def test_log_episode_includes_alpha_and_log_alpha(self, tmp_path):
        """Per-episode JSONL row carries ``alpha``, ``log_alpha``, and
        ``target_entropy`` fields so the learning-curves panel can
        plot the controller trajectory alongside entropy."""
        import math
        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={
                "entropy_coefficient": 0.01,
                "target_entropy": 112.0,
            },
        )

        ep = EpisodeStats(
            day_date="2026-04-19",
            total_reward=0.0,
            total_pnl=0.0,
            bet_count=0,
            winning_bets=0,
            races_completed=0,
            final_budget=100.0,
            n_steps=0,
        )
        loss_info = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
        }

        class _Tracker:
            completed = 1
            total = 1

            def to_dict(self):
                return {"completed": 1, "total": 1}

        trainer._log_episode(ep, loss_info, _Tracker())

        log_file = trainer.log_dir / "episodes.jsonl"
        row = json.loads(log_file.read_text().strip().splitlines()[-1])
        assert "alpha" in row
        assert "log_alpha" in row
        assert "target_entropy" in row
        assert row["alpha"] == pytest.approx(0.01, rel=1e-6)
        assert row["log_alpha"] == pytest.approx(math.log(0.01), abs=1e-4)
        assert row["target_entropy"] == pytest.approx(112.0)


# ── ppo-kl-fix (2026-04-24): stateful rollout ↔ stateful update ────────────
#
# Regression guards for the fix documented in
# ``plans/ppo-kl-fix/purpose.md``. Before this fix the PPO update
# evaluated the policy statelessly (zero-init hidden) while the
# rollout evaluated it statefully (carried hidden across ticks),
# so ``new_log_probs`` and ``old_log_probs`` came from different
# distributions and ``approx_kl`` blew up to 1e3–1e6 on epoch 0 of
# every update — see
# ``plans/ppo-stability-and-force-close-investigation/findings.md``.
#
# All three tests below are INTEGRATION-level (real trainer, real
# policy, real rollout, no mocks on the forward pass) per the
# 2026-04-18 units-mismatch lesson: a unit test that mocks the
# forward away silently passes a broken implementation.


class TestRecurrentStateThroughPpoUpdate:
    """Verify the PPO update conditions on rollout-time hidden state."""

    def test_collect_rollout_captures_hidden_state_in_on_every_transition(
        self,
    ):
        """Rollout must stash the INCOMING hidden state on every
        transition. The stored state is what the update feeds back
        into the policy's forward pass to reproduce the rollout-time
        distribution — missing it on any transition silently drops
        that mini-batch back onto the stateless-lobotomised path.
        """
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(policy, config)

        day = _make_day(n_races=1, n_ticks=5, n_runners=3)
        rollout, _ = trainer._collect_rollout(day)
        transitions = list(rollout.transitions)

        assert len(transitions) > 1
        for i, tr in enumerate(transitions):
            assert tr.hidden_state_in is not None, (
                f"transition {i}: hidden_state_in is None — the "
                f"rollout did not capture the pre-forward hidden "
                f"state, so ``_ppo_update`` will drop back to the "
                f"stateless path that ``plans/ppo-kl-fix/`` exists "
                f"to remove."
            )
            h0, h1 = tr.hidden_state_in
            # LSTM hidden shape is ``(num_layers, 1, hidden_size)``.
            assert h0.shape == h1.shape
            assert h0.ndim == 3 and h0.shape[1] == 1

        # First transition's hidden state must be zero (init_hidden).
        first = transitions[0].hidden_state_in
        assert first is not None
        assert np.all(first[0] == 0.0)
        assert np.all(first[1] == 0.0)

        # Any subsequent transition should have a NON-zero hidden
        # state (the LSTM has processed at least one tick). A
        # trivially zero state across all transitions would indicate
        # the capture is happening AFTER the reassignment — wrong
        # timing per ``plans/ppo-kl-fix/hard_constraints.md §7``.
        any_nonzero = any(
            not np.all(tr.hidden_state_in[0] == 0.0)  # type: ignore[index]
            for tr in transitions[1:]
        )
        assert any_nonzero, (
            "every stored hidden state was all-zero — likely the "
            "capture is reading the POST-forward state. See "
            "plans/ppo-kl-fix/hard_constraints.md §7."
        )

    def test_ppo_update_approx_kl_small_on_first_epoch_lstm(self):
        """The REAL integration guard.

        A fresh LSTM policy + one PPO update → ``approx_kl`` must
        land in the literature-normal range. Before the fix this
        value was 1e3–1e6 in observed probe runs; after the fix it
        should be O(1e-2) to O(1). The 1.0 threshold is
        deliberately loose: the test's purpose is to catch the
        stateful/stateless mismatch recurring, not to regulate the
        exact magnitude of healthy PPO drift.
        """
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 4},
        )

        day = _make_day(n_races=1, n_ticks=5, n_runners=3)
        rollout, _ = trainer._collect_rollout(day)
        assert len(rollout.transitions) > 1

        trainer._ppo_update(rollout)

        assert trainer._last_approx_kl < 1.0, (
            f"approx_kl={trainer._last_approx_kl:.4f} on epoch 0 "
            f"of a fresh policy. If this is in the thousands, the "
            f"rollout↔update state mismatch documented in "
            f"plans/ppo-kl-fix/ has returned: ``_ppo_update`` is "
            f"evaluating the policy statelessly while "
            f"``_collect_rollout`` evaluates it statefully."
        )
        # And it should not have hit the early-stop on epoch 0
        # (threshold 0.03 by default). A trained policy might; a
        # fresh one on its first update should not.
        assert trainer._last_kl_early_stop_epoch != 0, (
            f"fresh policy hit the 0.03 KL early-stop on epoch 0 "
            f"with approx_kl={trainer._last_approx_kl:.4f}."
        )

    def test_ppo_update_approx_kl_matches_old_logp_before_any_gradient_step(
        self,
    ):
        """Tighter lock: if we monkey-patch out the optimiser step
        (so weights don't move) then run ``_ppo_update``, the KL
        between rollout-time ``old_log_probs`` and the update's
        ``new_log_probs`` MUST be effectively zero — same weights,
        same state, same obs → same log-prob. Before the fix this
        was non-zero because the update used zero-init hidden
        state instead of the rollout's carried state.

        This is the most direct test of the fix: the KL is a
        function of (policy, obs, state), and holding policy and
        obs constant, the KL is exactly the signature of whether
        state is being fed through correctly.
        """
        config = _make_config()
        policy = _make_policy(config)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 4},
        )

        day = _make_day(n_races=1, n_ticks=5, n_runners=3)
        rollout, _ = trainer._collect_rollout(day)
        assert len(rollout.transitions) > 1

        # Freeze the optimiser: zero gradient application so the
        # network weights don't move during the update.
        def _noop_step(closure=None):
            return None

        trainer.optimiser.step = _noop_step  # type: ignore[assignment]
        trainer._alpha_optimizer.step = _noop_step  # type: ignore[assignment]

        trainer._ppo_update(rollout)

        # With weights frozen, KL should be near-machine-epsilon.
        # Float32 precision + pin-memory/device hops allow a tiny
        # drift; 1e-3 is generous but still 4 orders of magnitude
        # below the 0.03 early-stop threshold and ~1e7× tighter
        # than the pre-fix observed median of 12,740.
        assert abs(trainer._last_approx_kl) < 1e-3, (
            f"approx_kl={trainer._last_approx_kl} with weights "
            f"frozen — the update is evaluating the policy under "
            f"a different hidden state than the rollout did. See "
            f"plans/ppo-kl-fix/."
        )
