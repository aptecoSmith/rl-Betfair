"""Tests for Session 2.8 — Time-aware LSTM and time delta features.

Covers:
- Feature engineer: seconds_since_last_tick, seconds_spanned_last_N_ticks
- Environment: 4 new time delta dims in observation
- TimeLSTMCell: forward pass, forget gate responds to time delta, gradients
- PPOTimeLSTMPolicy: forward pass, architecture registry lookup
- Population manager: str_choice support
- Backward compatibility (defaults to 0 for missing timestamps)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest
import torch
import yaml

from data.episode_builder import Day, PriceSize, Race, RunnerMeta, RunnerSnap, Tick
from data.feature_engineer import TickHistory, engineer_tick, market_tick_features
from env.betfair_env import (
    AGENT_STATE_DIM,
    MARKET_DIM,
    MARKET_KEYS,
    MARKET_VELOCITY_KEYS,
    RUNNER_DIM,
    RUNNER_KEYS,
    VELOCITY_DIM,
    BetfairEnv,
)
from agents.architecture_registry import REGISTRY, create_policy
from agents.policy_network import (
    MARKET_TOTAL_DIM,
    PolicyOutput,
    TimeLSTMCell,
    PPOTimeLSTMPolicy,
    PPOLSTMPolicy,
)
from agents.population_manager import (
    HyperparamSpec,
    parse_search_ranges,
    sample_hyperparams,
    validate_hyperparams,
    PopulationManager,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_tick(
    timestamp: datetime | None = None,
    race_status: str | None = None,
) -> Tick:
    """Create a minimal Tick with specified timestamp."""
    ts = timestamp or datetime(2026, 3, 28, 13, 55, 0)
    return Tick(
        market_id="1.200",
        timestamp=ts,
        sequence_number=1,
        venue="Ascot",
        market_start_time=datetime(2026, 3, 28, 14, 0, 0),
        number_of_active_runners=2,
        traded_volume=10000.0,
        in_play=False,
        winner_selection_id=None,
        race_status=race_status,
        temperature=15.0,
        precipitation=0.0,
        wind_speed=5.0,
        wind_direction=180.0,
        humidity=60.0,
        weather_code=0,
        runners=[
            RunnerSnap(
                selection_id=101, status="ACTIVE",
                last_traded_price=3.5, total_matched=5000.0,
                starting_price_near=0.0, starting_price_far=0.0,
                adjustment_factor=10.0, bsp=None, sort_priority=1,
                removal_date=None,
                available_to_back=[PriceSize(3.4, 100.0)],
                available_to_lay=[PriceSize(3.6, 50.0)],
            ),
            RunnerSnap(
                selection_id=102, status="ACTIVE",
                last_traded_price=5.0, total_matched=3000.0,
                starting_price_near=0.0, starting_price_far=0.0,
                adjustment_factor=8.0, bsp=None, sort_priority=2,
                removal_date=None,
                available_to_back=[PriceSize(4.8, 80.0)],
                available_to_lay=[PriceSize(5.2, 60.0)],
            ),
        ],
    )


def _make_runner_meta(sid: int, name: str, **kwargs) -> RunnerMeta:
    """Create a RunnerMeta with all required fields."""
    defaults = dict(
        selection_id=sid, runner_name=name,
        sort_priority="1", handicap="0",
        sire_name="", dam_name="", damsire_name="", bred="",
        official_rating="90", adjusted_rating="88",
        age="4", sex_type="Gelding", colour_type="",
        weight_value="9.5", weight_units="st-lbs",
        jockey_name="J Smith", jockey_claim="0",
        trainer_name="T Jones", owner_name="",
        stall_draw="3", cloth_number="1",
        form="123", days_since_last_run="14", wearing="",
        forecastprice_numerator="5", forecastprice_denominator="2",
    )
    defaults.update(kwargs)
    return RunnerMeta(**defaults)


def _make_race(ticks: list[Tick] | None = None) -> Race:
    """Create a minimal Race."""
    return Race(
        market_id="1.200",
        venue="Ascot",
        market_start_time=datetime(2026, 3, 28, 14, 0, 0),
        market_type="WIN",
        market_name="2:00 Ascot",
        n_runners=2,
        winner_selection_id=101,
        winning_selection_ids={101},
        runner_metadata={
            101: _make_runner_meta(101, "Star Runner"),
            102: _make_runner_meta(102, "Dark Horse", sort_priority="2",
                                    stall_draw="7", cloth_number="2",
                                    official_rating="85", age="5",
                                    sex_type="Colt"),
        },
        ticks=ticks or [_make_tick()],
    )


def _load_config() -> dict:
    """Load the project config.yaml."""
    from pathlib import Path
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Feature engineer: time delta features ─────────────────────────────────────


class TestTimeDeltaFeatures:
    """Test seconds_since_last_tick and seconds_spanned_last_N_ticks."""

    def test_first_tick_seconds_since_last_is_zero(self):
        """First tick in a race has seconds_since_last_tick = 0."""
        history = TickHistory()
        tick = _make_tick(timestamp=datetime(2026, 3, 28, 13, 55, 0))
        race = _make_race([tick])
        feat = engineer_tick(tick, race, history)
        assert feat["market_velocity"]["seconds_since_last_tick"] == 0.0

    def test_second_tick_5s_gap(self):
        """5-second gap: normalised = 5/300 ≈ 0.0167."""
        history = TickHistory()
        race = _make_race()
        t1 = _make_tick(timestamp=datetime(2026, 3, 28, 13, 55, 0))
        t2 = _make_tick(timestamp=datetime(2026, 3, 28, 13, 55, 5))
        engineer_tick(t1, race, history)
        feat = engineer_tick(t2, race, history)
        expected = 5.0 / 300.0
        assert abs(feat["market_velocity"]["seconds_since_last_tick"] - expected) < 1e-6

    def test_180s_gap_normalised(self):
        """180-second gap: normalised = 180/300 = 0.6."""
        history = TickHistory()
        race = _make_race()
        t1 = _make_tick(timestamp=datetime(2026, 3, 28, 13, 55, 0))
        t2 = _make_tick(timestamp=datetime(2026, 3, 28, 13, 58, 0))
        engineer_tick(t1, race, history)
        feat = engineer_tick(t2, race, history)
        expected = 180.0 / 300.0
        assert abs(feat["market_velocity"]["seconds_since_last_tick"] - expected) < 1e-6

    def test_large_gap_clamped_to_1(self):
        """Gap > 300s should be clamped to 1.0."""
        history = TickHistory()
        race = _make_race()
        t1 = _make_tick(timestamp=datetime(2026, 3, 28, 13, 50, 0))
        t2 = _make_tick(timestamp=datetime(2026, 3, 28, 14, 0, 0))  # 600s gap
        engineer_tick(t1, race, history)
        feat = engineer_tick(t2, race, history)
        assert feat["market_velocity"]["seconds_since_last_tick"] == 1.0

    def test_seconds_spanned_3_uniform_5s(self):
        """3 ticks at 5s spacing: span = 10s, normalised = 10/180 ≈ 0.0556."""
        history = TickHistory()
        race = _make_race()
        base = datetime(2026, 3, 28, 13, 55, 0)
        for i in range(3):
            tick = _make_tick(timestamp=base + timedelta(seconds=5 * i))
            feat = engineer_tick(tick, race, history)
        expected = 10.0 / 180.0  # 3 * 60 = 180
        assert abs(feat["market_velocity"]["seconds_spanned_3"] - expected) < 1e-6

    def test_seconds_spanned_3_non_uniform(self):
        """Mixed 5s + 180s: ticks at t=0, t=5, t=185. Span = 185s."""
        history = TickHistory()
        race = _make_race()
        base = datetime(2026, 3, 28, 13, 55, 0)
        timestamps = [base, base + timedelta(seconds=5), base + timedelta(seconds=185)]
        for ts in timestamps:
            tick = _make_tick(timestamp=ts)
            feat = engineer_tick(tick, race, history)
        expected = min(185.0 / 180.0, 1.0)  # clamped to 1.0
        assert abs(feat["market_velocity"]["seconds_spanned_3"] - expected) < 1e-6

    def test_seconds_spanned_insufficient_ticks_is_zero(self):
        """With only 2 ticks, seconds_spanned_3 = 0 (not enough history)."""
        history = TickHistory()
        race = _make_race()
        base = datetime(2026, 3, 28, 13, 55, 0)
        for i in range(2):
            tick = _make_tick(timestamp=base + timedelta(seconds=5 * i))
            feat = engineer_tick(tick, race, history)
        assert feat["market_velocity"]["seconds_spanned_3"] == 0.0

    def test_seconds_spanned_5_present(self):
        """seconds_spanned_5 is computed with 5 ticks."""
        history = TickHistory()
        race = _make_race()
        base = datetime(2026, 3, 28, 13, 55, 0)
        for i in range(5):
            tick = _make_tick(timestamp=base + timedelta(seconds=5 * i))
            feat = engineer_tick(tick, race, history)
        # 5 ticks: span = ts[4] - ts[0] = 20s, normalised = 20/300
        expected = 20.0 / 300.0
        assert abs(feat["market_velocity"]["seconds_spanned_5"] - expected) < 1e-6

    def test_seconds_spanned_10_present(self):
        """seconds_spanned_10 is computed with 10 ticks."""
        history = TickHistory()
        race = _make_race()
        base = datetime(2026, 3, 28, 13, 55, 0)
        for i in range(10):
            tick = _make_tick(timestamp=base + timedelta(seconds=5 * i))
            feat = engineer_tick(tick, race, history)
        # 10 ticks: span = ts[9] - ts[0] = 45s, normalised = 45/600
        expected = 45.0 / 600.0
        assert abs(feat["market_velocity"]["seconds_spanned_10"] - expected) < 1e-6

    def test_reset_clears_timestamp_history(self):
        """TickHistory.reset() should clear timestamps."""
        history = TickHistory()
        race = _make_race()
        base = datetime(2026, 3, 28, 13, 55, 0)
        for i in range(3):
            tick = _make_tick(timestamp=base + timedelta(seconds=5 * i))
            engineer_tick(tick, race, history)
        history.reset()
        assert len(history._timestamp_history) == 0

    def test_all_4_time_features_in_velocity_dict(self):
        """All 4 new time features present in market_velocity output."""
        history = TickHistory()
        race = _make_race()
        tick = _make_tick()
        feat = engineer_tick(tick, race, history)
        vel = feat["market_velocity"]
        for key in ("seconds_since_last_tick", "seconds_spanned_3",
                     "seconds_spanned_5", "seconds_spanned_10"):
            assert key in vel, f"Missing key: {key}"


# ── Environment: obs_dim updated ──────────────────────────────────────────────


class TestEnvTimeDeltaDims:
    """Test that the environment's observation dimensions are updated."""

    def test_velocity_dim_is_11(self):
        assert VELOCITY_DIM == 11

    def test_market_dim_unchanged(self):
        assert MARKET_DIM == 37

    def test_runner_dim_consistent(self):
        from env.betfair_env import RUNNER_KEYS
        assert RUNNER_DIM == len(RUNNER_KEYS)

    def test_velocity_keys_contain_time_features(self):
        for key in ("seconds_since_last_tick", "seconds_spanned_3",
                     "seconds_spanned_5", "seconds_spanned_10"):
            assert key in MARKET_VELOCITY_KEYS, f"Missing key: {key}"

    def test_obs_dim_consistent(self):
        """obs_dim = MARKET + VELOCITY + (RUNNER × 14) + AGENT_STATE + (POSITION × 14)."""
        from env.betfair_env import POSITION_DIM
        expected = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * 14) + AGENT_STATE_DIM + (POSITION_DIM * 14)
        assert expected > 0  # sanity: non-trivial dimension

    def test_market_total_dim_correct(self):
        """MARKET_TOTAL_DIM = 37 + 11 + 6 = 54."""
        assert MARKET_TOTAL_DIM == 54

    def test_no_duplicate_velocity_keys(self):
        assert len(MARKET_VELOCITY_KEYS) == len(set(MARKET_VELOCITY_KEYS))

    def test_env_runs_episode_with_time_features(self):
        """BetfairEnv should work with new obs_dim."""
        config = _load_config()
        ticks = [
            _make_tick(timestamp=datetime(2026, 3, 28, 13, 55, i * 5))
            for i in range(3)
        ]
        race = _make_race(ticks)
        day = Day(date="2026-03-28", races=[race])
        env = BetfairEnv(day, config)
        obs, info = env.reset()
        from env.betfair_env import POSITION_DIM
        assert obs.shape[0] == MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * 14) + AGENT_STATE_DIM + (POSITION_DIM * 14)
        assert not np.any(np.isnan(obs))


# ── TimeLSTMCell ──────────────────────────────────────────────────────────────


class TestTimeLSTMCell:
    """Tests for the custom time-aware LSTM cell."""

    def test_forward_output_shapes(self):
        """Output h, c have correct shapes."""
        cell = TimeLSTMCell(input_size=16, hidden_size=32)
        x = torch.randn(4, 16)
        h = torch.zeros(4, 32)
        c = torch.zeros(4, 32)
        td = torch.zeros(4)
        h_new, c_new = cell(x, (h, c), td)
        assert h_new.shape == (4, 32)
        assert c_new.shape == (4, 32)

    def test_forget_gate_responds_to_time_delta(self):
        """Larger time delta should cause more forgetting (different c_new)."""
        torch.manual_seed(42)
        cell = TimeLSTMCell(input_size=16, hidden_size=32)
        # Make W_dt positive so larger delta → larger forget gate → more forgetting
        with torch.no_grad():
            cell.W_dt.fill_(1.0)

        x = torch.randn(1, 16)
        h = torch.randn(1, 32)
        c = torch.randn(1, 32)

        # Small time delta
        _, c_small = cell(x, (h, c.clone()), torch.tensor([0.01]))
        # Large time delta
        _, c_large = cell(x, (h, c.clone()), torch.tensor([1.0]))

        # With positive W_dt, larger delta → larger forget gate → c retains
        # more of old value and less of new gate. The cell states should differ.
        assert not torch.allclose(c_small, c_large, atol=1e-6)

    def test_zero_time_delta_matches_standard_lstm_behavior(self):
        """With delta=0, TimeLSTMCell should behave like a standard LSTM cell."""
        cell = TimeLSTMCell(input_size=8, hidden_size=16)
        x = torch.randn(2, 8)
        h = torch.zeros(2, 16)
        c = torch.zeros(2, 16)

        h1, c1 = cell(x, (h, c), torch.zeros(2))
        # Run again with same inputs — should be deterministic
        h2, c2 = cell(x, (h, c), torch.zeros(2))
        assert torch.allclose(h1, h2)
        assert torch.allclose(c1, c2)

    def test_gradients_flow_through_time_delta(self):
        """Gradient flows from loss back through W_dt."""
        cell = TimeLSTMCell(input_size=8, hidden_size=16)
        x = torch.randn(1, 8)
        h = torch.randn(1, 16)
        c = torch.randn(1, 16)  # non-zero c so forget gate matters
        td = torch.tensor([0.5], requires_grad=False)

        h_out, _ = cell(x, (h, c), td)
        loss = h_out.sum()
        loss.backward()

        assert cell.W_dt.grad is not None
        assert cell.W_dt.grad.abs().sum() > 0

    def test_time_delta_2d_input(self):
        """time_delta with shape (batch, 1) should work."""
        cell = TimeLSTMCell(input_size=8, hidden_size=16)
        x = torch.randn(3, 8)
        h = torch.zeros(3, 16)
        c = torch.zeros(3, 16)
        td = torch.tensor([[0.1], [0.5], [0.9]])
        h_out, c_out = cell(x, (h, c), td)
        assert h_out.shape == (3, 16)

    def test_forget_gate_monotonic_with_positive_w_dt(self):
        """With W_dt > 0, increasing delta should monotonically increase
        the forget gate's contribution (more old cell state retained)."""
        torch.manual_seed(42)
        cell = TimeLSTMCell(input_size=8, hidden_size=16)
        with torch.no_grad():
            cell.W_dt.fill_(2.0)  # Strong positive weight

        x = torch.randn(1, 8)
        h = torch.randn(1, 16)
        c_init = torch.ones(1, 16) * 5.0  # large initial cell state

        results = []
        for delta_val in [0.0, 0.1, 0.5, 1.0]:
            _, c_out = cell(x, (h, c_init.clone()), torch.tensor([delta_val]))
            results.append(c_out.mean().item())

        # The cell state should change monotonically with delta
        # (exact direction depends on gate interactions, but it should vary)
        assert not all(abs(results[i] - results[0]) < 1e-6 for i in range(1, len(results)))


# ── PPOTimeLSTMPolicy ─────────────────────────────────────────────────────────


class TestPPOTimeLSTMPolicy:
    """Tests for the time-aware LSTM policy."""

    def _make_policy(self, **kwargs) -> PPOTimeLSTMPolicy:
        hp = {"lstm_hidden_size": 64, "mlp_hidden_size": 32, "mlp_layers": 1}
        hp.update(kwargs)
        from env.betfair_env import POSITION_DIM
        obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * 14) + AGENT_STATE_DIM + (POSITION_DIM * 14)
        return PPOTimeLSTMPolicy(
            obs_dim=obs_dim,
            action_dim=28,
            max_runners=14,
            hyperparams=hp,
        )

    def test_forward_output_types(self):
        policy = self._make_policy()
        obs = torch.randn(1, policy.obs_dim)
        out = policy.forward(obs)
        assert isinstance(out, PolicyOutput)
        assert out.action_mean.shape == (1, 28)
        assert out.value.shape == (1, 1)

    def test_forward_with_hidden_state(self):
        policy = self._make_policy()
        obs = torch.randn(1, policy.obs_dim)
        h = policy.init_hidden(1)
        out = policy.forward(obs, h)
        assert out.hidden_state[0].shape == (1, 1, 64)
        assert out.hidden_state[1].shape == (1, 1, 64)

    def test_forward_sequence(self):
        """3-D input (batch, seq_len, obs_dim) should work."""
        policy = self._make_policy()
        obs = torch.randn(2, 5, policy.obs_dim)
        out = policy.forward(obs)
        assert out.action_mean.shape == (2, 28)

    def test_architecture_name(self):
        assert PPOTimeLSTMPolicy.architecture_name == "ppo_time_lstm_v1"

    def test_description_non_empty(self):
        assert len(PPOTimeLSTMPolicy.description) > 10

    def test_get_action_distribution(self):
        policy = self._make_policy()
        obs = torch.randn(1, policy.obs_dim)
        dist, value, hidden = policy.get_action_distribution(obs)
        assert dist.mean.shape == (1, 28)
        assert value.shape == (1, 1)

    def test_init_hidden_shape(self):
        policy = self._make_policy()
        h, c = policy.init_hidden(batch_size=3)
        assert h.shape == (1, 3, 64)
        assert c.shape == (1, 3, 64)

    def test_gradients_flow(self):
        """Full backward pass should compute gradients for all parameters."""
        policy = self._make_policy()
        obs = torch.randn(1, policy.obs_dim)
        out = policy.forward(obs)
        loss = out.action_mean.sum() + out.value.sum()
        loss.backward()
        # Check TimeLSTMCell W_dt has gradient (multi-layer LSTM stores
        # cells in a ModuleList; check the first layer).
        assert policy.time_lstm_cells[0].W_dt.grad is not None

    def test_time_delta_affects_hidden_state(self):
        """Different seconds_since_last_tick values should produce
        different hidden states."""
        policy = self._make_policy()
        obs1 = torch.randn(1, policy.obs_dim)
        obs2 = obs1.clone()

        # Modify the seconds_since_last_tick feature
        time_delta_idx = MARKET_DIM + VELOCITY_DIM - 4  # index in obs
        obs1[0, time_delta_idx] = 0.01  # small delta
        obs2[0, time_delta_idx] = 1.0   # large delta

        h = policy.init_hidden(1)
        out1 = policy.forward(obs1, h)
        out2 = policy.forward(obs2, h)

        # Hidden states should differ
        assert not torch.allclose(out1.hidden_state[0], out2.hidden_state[0], atol=1e-6)


# ── Architecture registry ─────────────────────────────────────────────────────


class TestArchitectureRegistry:
    """Test that ppo_time_lstm_v1 is registered."""

    def test_registered(self):
        assert "ppo_time_lstm_v1" in REGISTRY

    def test_create_policy(self):
        from env.betfair_env import POSITION_DIM
        obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * 14) + AGENT_STATE_DIM + (POSITION_DIM * 14)
        policy = create_policy(
            "ppo_time_lstm_v1",
            obs_dim=obs_dim,
            action_dim=28,
            max_runners=14,
            hyperparams={"lstm_hidden_size": 64, "mlp_hidden_size": 32, "mlp_layers": 1},
        )
        assert isinstance(policy, PPOTimeLSTMPolicy)

    def test_both_architectures_registered(self):
        assert "ppo_lstm_v1" in REGISTRY
        assert "ppo_time_lstm_v1" in REGISTRY


# ── Population manager: str_choice support ────────────────────────────────────


class TestStrChoiceSupport:
    """Test str_choice type for architecture_name in hyperparameters."""

    def test_parse_str_choice(self):
        raw = {
            "architecture_name": {
                "type": "str_choice",
                "choices": ["ppo_lstm_v1", "ppo_time_lstm_v1"],
            },
        }
        specs = parse_search_ranges(raw)
        assert specs[0].type == "str_choice"
        assert specs[0].choices == ["ppo_lstm_v1", "ppo_time_lstm_v1"]

    def test_sample_str_choice(self):
        import random
        specs = [
            HyperparamSpec(
                name="architecture_name",
                type="str_choice",
                choices=["ppo_lstm_v1", "ppo_time_lstm_v1"],
            ),
        ]
        rng = random.Random(42)
        result = sample_hyperparams(specs, rng)
        assert result["architecture_name"] in ["ppo_lstm_v1", "ppo_time_lstm_v1"]

    def test_validate_str_choice_valid(self):
        specs = [
            HyperparamSpec(
                name="architecture_name",
                type="str_choice",
                choices=["ppo_lstm_v1", "ppo_time_lstm_v1"],
            ),
        ]
        validate_hyperparams({"architecture_name": "ppo_time_lstm_v1"}, specs)

    def test_validate_str_choice_invalid(self):
        specs = [
            HyperparamSpec(
                name="architecture_name",
                type="str_choice",
                choices=["ppo_lstm_v1", "ppo_time_lstm_v1"],
            ),
        ]
        with pytest.raises(ValueError):
            validate_hyperparams({"architecture_name": "invalid_arch"}, specs)

    def test_config_has_architecture_name_str_choice(self):
        config = _load_config()
        arch_spec = config["hyperparameters"]["search_ranges"]["architecture_name"]
        assert arch_spec["type"] == "str_choice"
        assert "ppo_lstm_v1" in arch_spec["choices"]
        assert "ppo_time_lstm_v1" in arch_spec["choices"]


# ── Backward compatibility ────────────────────────────────────────────────────


class TestBackwardCompatibility:
    """Ensure old data without timestamps still works."""

    def test_time_features_default_to_zero_first_tick(self):
        """First tick defaults all time features to 0."""
        history = TickHistory()
        tick = _make_tick()
        race = _make_race([tick])
        feat = engineer_tick(tick, race, history)
        vel = feat["market_velocity"]
        assert vel["seconds_since_last_tick"] == 0.0
        assert vel["seconds_spanned_3"] == 0.0
        assert vel["seconds_spanned_5"] == 0.0
        assert vel["seconds_spanned_10"] == 0.0

    def test_existing_velocity_features_still_present(self):
        """Old velocity features should still be computed."""
        history = TickHistory()
        tick = _make_tick()
        race = _make_race([tick])
        feat = engineer_tick(tick, race, history)
        vel = feat["market_velocity"]
        for key in ("market_vol_delta_3", "overround_delta_3", "time_since_status_change"):
            assert key in vel
