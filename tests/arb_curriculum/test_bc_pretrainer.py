"""Tests for arb-curriculum Session 04: BC pretrainer + controller handshake.

Categories (per hard_constraints.md §30):
1. Per-agent independence — same genes, different seeds → weights diverge after BC.
2. Only actor_head changes — non-actor_head params bit-identical post-BC.
3. Loss decreases on synthetic samples — 50 consistent samples, 20 steps.
4. Empty oracle → skip cleanly — returns empty BCLossHistory, no param changes.
5. Gene-zero skip — n_steps=0 → empty BCLossHistory immediately.
6. All three architectures — BC runs without crash on lstm, time_lstm, transformer.
7. Schema mismatch hard-fails — load_samples raises ValueError on version mismatch.
8. Integration test: post-BC PPO update — spy on _update_reward_baseline asserts
   per-step mean argument (2026-04-18 units-mismatch regression guard).
9. Controller handshake — _effective_target_entropy interpolates 40→150 over 5 eps.
"""

from __future__ import annotations

import math
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.architecture_registry import create_policy
from agents.bc_pretrainer import (
    BCLossHistory,
    BCPretrainer,
    _is_bc_target_head,
    measure_entropy,
)
from agents.ppo_trainer import PPOTrainer
from data.episode_builder import Day, PriceSize, Race, RunnerMeta, RunnerSnap, Tick
from env.betfair_env import (
    AGENT_STATE_DIM,
    MARKET_DIM,
    POSITION_DIM,
    RUNNER_DIM,
    SCALPING_ACTIONS_PER_RUNNER,
    SCALPING_AGENT_STATE_DIM,
    SCALPING_POSITION_DIM,
    VELOCITY_DIM,
)
from training.arb_oracle import OracleSample

# ---------------------------------------------------------------------------
# Shared constants for scalping mode (7 dims per runner)
# ---------------------------------------------------------------------------

_MAX_RUNNERS = 2
_PER_RUNNER_ACTION_DIM = SCALPING_ACTIONS_PER_RUNNER  # 7
_ACTION_DIM = _MAX_RUNNERS * _PER_RUNNER_ACTION_DIM   # 14

# Scalping obs_dim (the env in scalping mode produces this size)
_OBS_DIM = (
    MARKET_DIM
    + VELOCITY_DIM
    + _MAX_RUNNERS * RUNNER_DIM
    + AGENT_STATE_DIM
    + SCALPING_AGENT_STATE_DIM
    + _MAX_RUNNERS * (POSITION_DIM + SCALPING_POSITION_DIM)
)

_ALL_ARCHS = ("ppo_lstm_v1", "ppo_time_lstm_v1", "ppo_transformer_v1")

_MARKET_START = datetime(2026, 4, 20, 14, 0, 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_policy(arch: str = "ppo_lstm_v1"):
    hp = {
        "lstm_hidden_size": 32,
        "mlp_hidden_size": 16,
        "mlp_layers": 1,
        "transformer_heads": 2,
        "transformer_depth": 1,
        "transformer_ctx_ticks": 8,
    }
    return create_policy(
        name=arch,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        max_runners=_MAX_RUNNERS,
        hyperparams=hp,
    )


def _make_oracle_sample(runner_idx: int = 0, spread_ticks: int = 5) -> OracleSample:
    """Synthetic oracle sample with random obs of the right size."""
    obs = np.random.default_rng(42).standard_normal(_OBS_DIM).astype(np.float32)
    return OracleSample(
        tick_index=0,
        runner_idx=runner_idx,
        obs=obs,
        arb_spread_ticks=spread_ticks,
        expected_locked_pnl=0.05,
    )


def _make_samples(n: int, runner_idx: int = 0) -> list[OracleSample]:
    rng = np.random.default_rng(99)
    return [
        OracleSample(
            tick_index=i,
            runner_idx=runner_idx,
            obs=rng.standard_normal(_OBS_DIM).astype(np.float32),
            arb_spread_ticks=5,
            expected_locked_pnl=0.05,
        )
        for i in range(n)
    ]


def _consistent_samples(n: int, runner_idx: int = 0) -> list[OracleSample]:
    """All samples share the same obs — gradient is consistent for testing loss drop."""
    obs = np.random.default_rng(7).standard_normal(_OBS_DIM).astype(np.float32)
    return [
        OracleSample(
            tick_index=i,
            runner_idx=runner_idx,
            obs=obs.copy(),
            arb_spread_ticks=5,
            expected_locked_pnl=0.05,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Day / race helpers for the integration test (test 8)
# ---------------------------------------------------------------------------


def _make_runner_meta(sid: int) -> RunnerMeta:
    return RunnerMeta(
        selection_id=sid,
        runner_name="Horse",
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


def _make_runner_snap(sid: int, ltp: float = 5.0) -> RunnerSnap:
    return RunnerSnap(
        selection_id=sid,
        status="ACTIVE",
        last_traded_price=ltp,
        total_matched=1000.0,
        starting_price_near=0.0,
        starting_price_far=0.0,
        adjustment_factor=None,
        bsp=None,
        sort_priority=1,
        removal_date=None,
        available_to_back=[PriceSize(price=ltp, size=100.0)],
        available_to_lay=[PriceSize(price=ltp + 0.1, size=100.0)],
    )


def _make_stub_day(n_ticks: int = 4) -> Day:
    sid = 101
    ticks = []
    for i in range(n_ticks):
        ts = _MARKET_START - timedelta(seconds=300 - i * 5)
        ticks.append(Tick(
            market_id="1.999000001",
            timestamp=ts,
            sequence_number=i,
            venue="Newmarket",
            market_start_time=_MARKET_START,
            number_of_active_runners=1,
            traded_volume=5000.0,
            in_play=False,
            winner_selection_id=sid,
            race_status=None,
            temperature=15.0,
            precipitation=0.0,
            wind_speed=5.0,
            wind_direction=180.0,
            humidity=60.0,
            weather_code=0,
            runners=[_make_runner_snap(sid)],
        ))
    # Add in-play tick so the race settles
    ticks.append(Tick(
        market_id="1.999000001",
        timestamp=_MARKET_START + timedelta(seconds=5),
        sequence_number=n_ticks,
        venue="Newmarket",
        market_start_time=_MARKET_START,
        number_of_active_runners=1,
        traded_volume=5000.0,
        in_play=True,
        winner_selection_id=sid,
        race_status=None,
        temperature=15.0,
        precipitation=0.0,
        wind_speed=5.0,
        wind_direction=180.0,
        humidity=60.0,
        weather_code=0,
        runners=[_make_runner_snap(sid)],
    ))
    meta = {sid: _make_runner_meta(sid)}
    race = Race(
        market_id="1.999000001",
        venue="Newmarket",
        market_start_time=_MARKET_START,
        winner_selection_id=sid,
        ticks=ticks,
        runner_metadata=meta,
        winning_selection_ids={sid},
    )
    return Day(date="2026-04-20", races=[race])


def _scalping_config() -> dict:
    return {
        "training": {
            "architecture": "ppo_lstm_v1",
            "starting_budget": 100.0,
            "max_runners": _MAX_RUNNERS,
            "scalping_mode": True,
            "betting_constraints": {
                "max_back_price": 50.0,
                "max_lay_price": None,
                "min_seconds_before_off": 0,
            },
            "max_bets_per_race": 20,
        },
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "efficiency_penalty": 0.01,
            "commission": 0.05,
            "naked_loss_scale": 1.0,
        },
        "paths": {
            "processed_data": "data/processed",
            "model_weights": "registry/weights",
            "logs": "logs",
            "registry_db": "registry/models.db",
        },
    }


# ===========================================================================
# Test 1: Per-agent independence
# ===========================================================================


class TestPerAgentIndependence:
    def test_same_genes_different_seeds_diverge(self):
        """Two policies, same architecture, different seeds → different params post-BC."""
        torch.manual_seed(0)
        policy_a = _make_policy()
        torch.manual_seed(1)
        policy_b = _make_policy()

        samples = _make_samples(50)
        bc = BCPretrainer(lr=1e-3, batch_size=16)

        torch.manual_seed(10)
        bc.pretrain(policy_a, samples, n_steps=5)
        torch.manual_seed(20)
        bc.pretrain(policy_b, samples, n_steps=5)

        # Actor head parameters must differ (different starting weights + different
        # batch sampling due to different random states).
        for (name_a, p_a), (name_b, p_b) in zip(
            policy_a.named_parameters(), policy_b.named_parameters()
        ):
            assert name_a == name_b
            if _is_bc_target_head(name_a):
                if not torch.equal(p_a, p_b):
                    return  # found at least one divergent param
        pytest.fail("All actor_head parameters are equal — BC is sharing state")


# ===========================================================================
# Test 2: Only actor_head changes
# ===========================================================================


class TestOnlyActorHeadChanges:
    def test_non_actor_head_params_bit_identical(self):
        """BC freezes all non-actor_head params; they must be bit-identical after."""
        policy = _make_policy()

        # Snapshot all non-actor_head parameters.
        snapshots = {
            name: p.detach().clone()
            for name, p in policy.named_parameters()
            if not _is_bc_target_head(name)
        }
        assert snapshots, "no non-actor_head params found — check _is_bc_target_head"

        samples = _make_samples(50)
        bc = BCPretrainer(lr=1e-3, batch_size=16)
        bc.pretrain(policy, samples, n_steps=5)

        for name, original in snapshots.items():
            current = dict(policy.named_parameters())[name].detach()
            assert torch.equal(current, original), (
                f"Non-actor_head param '{name}' changed during BC"
            )

    def test_requires_grad_restored_after_bc(self):
        """All parameters have requires_grad=True after BC (PPO can train them)."""
        policy = _make_policy()
        samples = _make_samples(20)
        BCPretrainer(lr=1e-3).pretrain(policy, samples, n_steps=3)

        for name, p in policy.named_parameters():
            assert p.requires_grad, (
                f"param '{name}' has requires_grad=False after BC — "
                "PPO's optimiser cannot update it"
            )


# ===========================================================================
# Test 3: Loss decreases on consistent synthetic samples
# ===========================================================================


class TestLossDecreases:
    def test_signal_loss_drops_over_20_steps(self):
        """Consistent samples with signal target +1.0: loss must fall sharply."""
        policy = _make_policy()
        samples = _consistent_samples(50, runner_idx=0)
        bc = BCPretrainer(lr=5e-3, batch_size=16, arb_spread_weight=0.0)

        torch.manual_seed(0)
        history = bc.pretrain(policy, samples, n_steps=20)

        assert history.signal_losses, "no losses recorded"
        first = history.signal_losses[0]
        last = history.signal_losses[-1]
        assert last < first * 0.5, (
            f"signal_loss did not halve after 20 steps: "
            f"first={first:.4f} last={last:.4f}"
        )

    def test_history_lengths_match_n_steps(self):
        """signal_losses / arb_spread_losses / total_losses all have len == n_steps."""
        policy = _make_policy()
        samples = _make_samples(30)
        history = BCPretrainer(lr=1e-3).pretrain(policy, samples, n_steps=7)

        assert len(history.signal_losses) == 7
        assert len(history.arb_spread_losses) == 7
        assert len(history.total_losses) == 7
        assert math.isclose(history.final_signal_loss, history.signal_losses[-1])
        assert math.isclose(history.final_arb_spread_loss, history.arb_spread_losses[-1])


# ===========================================================================
# Test 4: Empty oracle → skip cleanly
# ===========================================================================


class TestEmptyOracleSkip:
    def test_empty_samples_returns_empty_history(self):
        """pretrain(policy, [], n_steps=100) returns BCLossHistory with no losses."""
        policy = _make_policy()
        history = BCPretrainer().pretrain(policy, [], n_steps=100)

        assert history.signal_losses == []
        assert history.arb_spread_losses == []
        assert history.total_losses == []
        assert math.isclose(history.final_signal_loss, 0.0)
        assert math.isclose(history.final_arb_spread_loss, 0.0)

    def test_empty_samples_params_unchanged(self):
        """Empty pretrain call does not alter any policy parameter."""
        policy = _make_policy()
        snapshots = {
            name: p.detach().clone()
            for name, p in policy.named_parameters()
        }
        BCPretrainer().pretrain(policy, [], n_steps=100)

        for name, original in snapshots.items():
            current = dict(policy.named_parameters())[name].detach()
            assert torch.equal(current, original), (
                f"param '{name}' changed despite empty oracle"
            )


# ===========================================================================
# Test 5: Gene-zero skip
# ===========================================================================


class TestGeneZeroSkip:
    def test_n_steps_zero_returns_empty_history(self):
        """n_steps=0 returns empty BCLossHistory without touching the policy."""
        policy = _make_policy()
        snapshots = {
            name: p.detach().clone()
            for name, p in policy.named_parameters()
        }
        history = BCPretrainer(lr=1e-3).pretrain(policy, _make_samples(10), n_steps=0)

        assert history.signal_losses == []
        for name, original in snapshots.items():
            current = dict(policy.named_parameters())[name].detach()
            assert torch.equal(current, original)

    def test_negative_n_steps_returns_empty_history(self):
        """n_steps < 0 treated same as 0."""
        history = BCPretrainer().pretrain(_make_policy(), _make_samples(10), n_steps=-5)
        assert history.signal_losses == []


# ===========================================================================
# Test 6: All three architectures
# ===========================================================================


class TestAllArchitectures:
    @pytest.mark.parametrize("arch", _ALL_ARCHS)
    def test_bc_runs_without_crash(self, arch: str):
        """BC runs on each architecture and returns a non-empty loss history."""
        policy = _make_policy(arch)
        samples = _make_samples(30)
        history = BCPretrainer(lr=1e-4, batch_size=8).pretrain(
            policy, samples, n_steps=3,
        )
        assert len(history.signal_losses) == 3
        assert all(math.isfinite(v) for v in history.signal_losses)
        assert all(math.isfinite(v) for v in history.arb_spread_losses)

    @pytest.mark.parametrize("arch", _ALL_ARCHS)
    def test_measure_entropy_returns_finite(self, arch: str):
        """measure_entropy returns a positive finite scalar on all architectures."""
        policy = _make_policy(arch)
        samples = _make_samples(10)
        ent = measure_entropy(policy, samples)
        assert math.isfinite(ent)
        assert ent > 0.0


# ===========================================================================
# Test 7: Schema mismatch hard-fails
# ===========================================================================


class TestSchemaMismatch:
    def test_load_samples_raises_on_version_mismatch(self, tmp_path: Path):
        """load_samples with wrong obs_schema_version raises ValueError.

        load_samples(date, data_dir) looks in data_dir.parent/oracle_cache/date/,
        so we create the cache one level below tmp_path.
        """
        from env.betfair_env import OBS_SCHEMA_VERSION, ACTION_SCHEMA_VERSION
        from training.arb_oracle import load_samples

        # data_dir = tmp_path/processed; oracle_cache at tmp_path/oracle_cache/
        data_dir = tmp_path / "processed"
        cache_dir = tmp_path / "oracle_cache" / "2026-04-20"
        cache_dir.mkdir(parents=True)

        wrong_version = OBS_SCHEMA_VERSION + 999
        np.savez(
            cache_dir / "oracle_samples.npz",
            obs_schema_version=np.array(wrong_version),
            action_schema_version=np.array(ACTION_SCHEMA_VERSION),
            tick_index=np.zeros(0, dtype=np.int64),
            runner_idx=np.zeros(0, dtype=np.int64),
            obs=np.zeros((0, 10), dtype=np.float32),
            arb_spread_ticks=np.zeros(0, dtype=np.int64),
            expected_locked_pnl=np.zeros(0, dtype=np.float64),
        )

        with pytest.raises(ValueError, match="obs_schema_version"):
            load_samples("2026-04-20", data_dir, strict=True)

    def test_load_samples_raises_on_action_version_mismatch(self, tmp_path: Path):
        """load_samples with wrong action_schema_version raises ValueError."""
        from env.betfair_env import OBS_SCHEMA_VERSION, ACTION_SCHEMA_VERSION
        from training.arb_oracle import load_samples

        data_dir = tmp_path / "processed"
        cache_dir = tmp_path / "oracle_cache" / "2026-04-20"
        cache_dir.mkdir(parents=True)

        np.savez(
            cache_dir / "oracle_samples.npz",
            obs_schema_version=np.array(OBS_SCHEMA_VERSION),
            action_schema_version=np.array(ACTION_SCHEMA_VERSION + 999),
            tick_index=np.zeros(0, dtype=np.int64),
            runner_idx=np.zeros(0, dtype=np.int64),
            obs=np.zeros((0, 10), dtype=np.float32),
            arb_spread_ticks=np.zeros(0, dtype=np.int64),
            expected_locked_pnl=np.zeros(0, dtype=np.float64),
        )

        with pytest.raises(ValueError, match="action_schema_version"):
            load_samples("2026-04-20", data_dir, strict=True)


# ===========================================================================
# Test 8: Integration — post-BC _ppo_update passes per-step mean to baseline
# ===========================================================================


class TestPostBCPPOUpdateUnits:
    def test_update_reward_baseline_receives_per_step_mean(self):
        """Load-bearing regression guard (2026-04-18 units-mismatch lesson).

        After BC runs, the first _ppo_update must pass per-step mean reward
        to _update_reward_baseline — NOT the episode sum.  A future refactor
        that reverts the call-site will immediately fail this test.
        """
        config = _scalping_config()
        policy = _make_policy("ppo_lstm_v1")

        # Run BC on synthetic samples (establishes post-BC state).
        samples = _make_samples(20)
        bc = BCPretrainer(lr=1e-4, batch_size=8)
        torch.manual_seed(0)
        bc.pretrain(policy, samples, n_steps=3)

        trainer = PPOTrainer(
            policy=policy,
            config=config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 4},
        )
        # Simulate post-BC state on the trainer.
        trainer._bc_loss_history = BCLossHistory(
            signal_losses=[0.5, 0.4, 0.3],
            arb_spread_losses=[0.1, 0.09, 0.08],
            total_losses=[0.6, 0.49, 0.38],
            final_signal_loss=0.3,
            final_arb_spread_loss=0.08,
        )
        trainer._bc_pretrain_steps_done = 3
        trainer._post_bc_entropy = 40.0
        trainer._bc_target_entropy_warmup_eps = 5

        day = _make_stub_day(n_ticks=4)
        rollout, _ = trainer._collect_rollout(day)
        transitions = list(rollout.transitions)
        assert len(transitions) > 1, "need multi-step rollout to distinguish mean vs sum"

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
            f"expected one baseline update per _ppo_update; got {len(recorded)}"
        )
        passed_value = recorded[0]
        assert abs(passed_value - expected_per_step_mean) < 1e-6, (
            f"_update_reward_baseline received {passed_value}; "
            f"expected per-step mean {expected_per_step_mean} "
            f"(n={len(transitions)}, sum={episode_sum}). "
            f"If the value matches the episode sum, the 2026-04-18 "
            f"units-mismatch bug has returned."
        )


# ===========================================================================
# Test 9: Controller handshake — _effective_target_entropy interpolation
# ===========================================================================


class TestControllerHandshake:
    def test_effective_target_entropy_interpolates(self):
        """post_bc_entropy=40, target=150, warmup=5 → [40, 62, 84, 106, 128, 150, 150]."""
        config = _scalping_config()
        policy = _make_policy()
        trainer = PPOTrainer(
            policy=policy,
            config=config,
            hyperparams={"target_entropy": 150.0},
        )
        trainer._post_bc_entropy = 40.0
        trainer._bc_target_entropy_warmup_eps = 5

        expected = [40.0, 62.0, 84.0, 106.0, 128.0, 150.0, 150.0]
        for ep in range(7):
            trainer._eps_since_bc = ep
            result = trainer._effective_target_entropy()
            assert math.isclose(result, expected[ep], abs_tol=1e-6), (
                f"ep={ep}: expected {expected[ep]}, got {result}"
            )

    def test_no_bc_returns_configured_target(self):
        """Without BC (_post_bc_entropy=None), returns _target_entropy unchanged."""
        config = _scalping_config()
        policy = _make_policy()
        trainer = PPOTrainer(
            policy=policy,
            config=config,
            hyperparams={"target_entropy": 150.0},
        )
        # Default state: _post_bc_entropy is None.
        assert trainer._post_bc_entropy is None
        assert math.isclose(trainer._effective_target_entropy(), 150.0)

    def test_warmup_zero_disables_handshake(self):
        """bc_target_entropy_warmup_eps=0 → always returns _target_entropy."""
        config = _scalping_config()
        policy = _make_policy()
        trainer = PPOTrainer(
            policy=policy,
            config=config,
            hyperparams={"target_entropy": 150.0},
        )
        trainer._post_bc_entropy = 40.0
        trainer._bc_target_entropy_warmup_eps = 0

        for ep in range(10):
            trainer._eps_since_bc = ep
            assert math.isclose(trainer._effective_target_entropy(), 150.0), (
                f"ep={ep}: warmup=0 should disable handshake"
            )

    def test_warmup_complete_returns_configured_target(self):
        """Once eps_since_bc >= warmup_eps, returns _target_entropy."""
        config = _scalping_config()
        policy = _make_policy()
        trainer = PPOTrainer(
            policy=policy,
            config=config,
            hyperparams={"target_entropy": 150.0},
        )
        trainer._post_bc_entropy = 40.0
        trainer._bc_target_entropy_warmup_eps = 5
        trainer._eps_since_bc = 5  # exactly at end

        assert math.isclose(trainer._effective_target_entropy(), 150.0)

        trainer._eps_since_bc = 10  # well past end
        assert math.isclose(trainer._effective_target_entropy(), 150.0)
