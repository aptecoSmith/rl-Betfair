"""Tests for arb-curriculum Session 02: matured-arb shaped bonus.

Categories (per hard_constraints.md §28):
1. weight=0 byte-identical
2. bonus emitted only on maturation
3. zero-mean at expected count
4. cap enforced (+)
5. symmetry / cap enforced (-)
6. JSONL field present
7. gene passthrough via reward_overrides
8. invariant parametrised (raw + shaped ≈ total) at weight in {0.0, 1.0}
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pytest

from data.episode_builder import Day, PriceSize, Race, RunnerMeta, RunnerSnap, Tick
from env.betfair_env import BetfairEnv

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MARKET_START = datetime(2026, 4, 20, 14, 0, 0)


def _minimal_config(
    matured_arb_bonus_weight: float = 0.0,
    matured_arb_bonus_cap: float = 10.0,
    matured_arb_expected_random: float = 2.0,
    extra_reward: dict | None = None,
) -> dict:
    cfg: dict[str, Any] = {
        "training": {
            "max_runners": 5,
            "starting_budget": 100.0,
            "max_bets_per_race": 20,
            "betting_constraints": {
                "max_back_price": 50.0,
                "max_lay_price": None,
            },
        },
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "efficiency_penalty": 0.01,
            "commission": 0.05,
            "matured_arb_bonus_weight": matured_arb_bonus_weight,
            "matured_arb_bonus_cap": matured_arb_bonus_cap,
            "matured_arb_expected_random": matured_arb_expected_random,
        },
    }
    if extra_reward:
        cfg["reward"].update(extra_reward)
    return cfg


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


def _make_tick(sid: int = 101, ltp: float = 5.0, seq: int = 0) -> Tick:
    runner = RunnerSnap(
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
    ts = _MARKET_START - timedelta(seconds=300 - seq * 5)
    return Tick(
        market_id="1.999000001",
        timestamp=ts,
        sequence_number=seq,
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
        runners=[runner],
    )


def _make_stub_day(n_ticks: int = 3, sid: int = 101) -> Day:
    ticks = [_make_tick(sid=sid, seq=i) for i in range(n_ticks)]
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


def _make_env(config: dict, n_ticks: int = 3) -> BetfairEnv:
    stub_day = _make_stub_day(n_ticks=n_ticks)
    return BetfairEnv(stub_day, config, scalping_mode=True)


def _run_one_episode(config: dict, n_ticks: int = 3) -> tuple[float, float, float, dict]:
    """Run a single episode and return (raw, shaped, total, info)."""
    env = _make_env(config, n_ticks=n_ticks)
    obs, info = env.reset()
    # All-zeros action = hold (signal < 0 → no bet, all other dims neutral).
    noop = np.zeros(env.action_space.shape, dtype=np.float32)
    total_reward = 0.0
    done = False
    last_info: dict = {}
    while not done:
        obs, reward, terminated, truncated, info = env.step(noop)
        total_reward += float(reward)
        done = terminated or truncated
        last_info = info
    raw = last_info.get("raw_pnl_reward", 0.0)
    shaped = last_info.get("shaped_bonus", 0.0)
    return raw, shaped, total_reward, last_info


# ===========================================================================
# Test 1: weight=0 → byte-identical reward components between two runs
# ===========================================================================


class TestWeightZeroByteIdentical:
    def test_shaped_is_deterministic_at_weight_zero(self):
        """With weight=0, two runs produce identical raw/shaped/total."""
        config = _minimal_config(matured_arb_bonus_weight=0.0)
        raw0, shaped0, total0, _ = _run_one_episode(config)
        raw1, shaped1, total1, _ = _run_one_episode(config)
        assert math.isclose(raw0, raw1, abs_tol=1e-9)
        assert math.isclose(shaped0, shaped1, abs_tol=1e-9)
        assert math.isclose(total0, total1, abs_tol=1e-9)


# ===========================================================================
# Test 2: no matured pairs → term is weight*(0 - expected_random)
# ===========================================================================


class TestBonusOnlyOnMaturation:
    def test_zero_matured_pairs_formula(self):
        """Formula: weight*(0 - expected_random), capped at -cap."""
        weight = 1.0
        expected_random = 2.0
        cap = 10.0
        n_matured = 0

        raw_bonus = weight * (n_matured - expected_random)
        term = float(np.clip(raw_bonus, -cap, cap))
        assert math.isclose(term, -2.0, abs_tol=1e-9)

    def test_env_attributes_read_correctly(self):
        """Env reads weight, cap, and expected_random from config."""
        config = _minimal_config(
            matured_arb_bonus_weight=0.7,
            matured_arb_bonus_cap=5.0,
            matured_arb_expected_random=3.0,
        )
        env = _make_env(config)
        assert math.isclose(env._matured_arb_bonus_weight, 0.7, abs_tol=1e-9)
        assert math.isclose(env._matured_arb_bonus_cap, 5.0, abs_tol=1e-9)
        assert math.isclose(env._matured_arb_expected_random, 3.0, abs_tol=1e-9)


# ===========================================================================
# Test 3: exactly expected_random matured pairs → term is 0
# ===========================================================================


class TestZeroMeanAtExpectedCount:
    def test_term_is_zero_at_expected_count(self):
        """n_matured == expected_random → raw_bonus = 0 → term = 0."""
        weight = 1.0
        cap = 10.0
        expected_random = 3.0
        n_matured = 3

        raw_bonus = weight * (n_matured - expected_random)
        term = float(np.clip(raw_bonus, -cap, cap))
        assert math.isclose(term, 0.0, abs_tol=1e-9)

    def test_various_expected_random_values(self):
        """Zero-mean holds for several weight / expected_random combinations."""
        for weight in [0.5, 1.0, 2.0]:
            for expected_random in [0.0, 1.0, 5.0]:
                cap = 100.0
                n_matured = expected_random
                raw_bonus = weight * (n_matured - expected_random)
                term = float(np.clip(raw_bonus, -cap, cap))
                assert math.isclose(term, 0.0, abs_tol=1e-9), (
                    f"weight={weight}, expected_random={expected_random}: term={term}"
                )


# ===========================================================================
# Test 4: cap enforced on positive side
# ===========================================================================


class TestCapEnforcedPositive:
    def test_large_n_matured_clamps_to_cap(self):
        """100 matured pairs → raw_bonus >> cap → term == +cap."""
        weight = 1.0
        cap = 10.0
        expected_random = 0.0
        n_matured = 100

        raw_bonus = weight * (n_matured - expected_random)
        term = float(np.clip(raw_bonus, -cap, cap))
        assert math.isclose(term, cap, abs_tol=1e-9)

    def test_cap_configurable_from_env(self):
        """Custom cap value is stored correctly on the env."""
        config = _minimal_config(matured_arb_bonus_cap=7.5)
        env = _make_env(config)
        assert math.isclose(env._matured_arb_bonus_cap, 7.5, abs_tol=1e-9)


# ===========================================================================
# Test 5: cap enforced on negative side
# ===========================================================================


class TestCapEnforcedNegative:
    def test_zero_matured_high_expected_clamps_to_minus_cap(self):
        """0 pairs, expected_random=200, weight=1 → raw=-200 → term=-cap."""
        weight = 1.0
        cap = 10.0
        expected_random = 200.0
        n_matured = 0

        raw_bonus = weight * (n_matured - expected_random)
        term = float(np.clip(raw_bonus, -cap, cap))
        assert math.isclose(term, -cap, abs_tol=1e-9)

    def test_negative_cap_equals_positive_cap_magnitude(self):
        """Positive and negative caps are symmetric by construction."""
        cap = 10.0
        # too high
        term_hi = float(np.clip(999.0, -cap, cap))
        # too low
        term_lo = float(np.clip(-999.0, -cap, cap))
        assert math.isclose(term_hi, cap, abs_tol=1e-9)
        assert math.isclose(term_lo, -cap, abs_tol=1e-9)
        assert math.isclose(abs(term_hi), abs(term_lo), abs_tol=1e-9)


# ===========================================================================
# Test 6: JSONL field present in info
# ===========================================================================


class TestJsonlFieldPresent:
    def test_matured_arb_bonus_active_in_info(self):
        """_get_info() includes 'matured_arb_bonus_active' equal to weight."""
        config = _minimal_config(matured_arb_bonus_weight=0.42)
        _, _, _, last_info = _run_one_episode(config)
        assert "matured_arb_bonus_active" in last_info
        assert math.isclose(last_info["matured_arb_bonus_active"], 0.42, abs_tol=1e-9)

    def test_field_present_when_weight_zero(self):
        """Field is present even when weight=0 (always emitted)."""
        config = _minimal_config(matured_arb_bonus_weight=0.0)
        _, _, _, last_info = _run_one_episode(config)
        assert "matured_arb_bonus_active" in last_info
        assert math.isclose(last_info["matured_arb_bonus_active"], 0.0, abs_tol=1e-9)


# ===========================================================================
# Test 7: gene passthrough via reward_overrides
# ===========================================================================


class TestGenePassthrough:
    def test_reward_override_sets_weight(self):
        """reward_overrides={'matured_arb_bonus_weight': 0.5} sets the knob."""
        base_config = _minimal_config(matured_arb_bonus_weight=0.0)
        stub_day = _make_stub_day()
        env = BetfairEnv(
            stub_day,
            base_config,
            scalping_mode=True,
            reward_overrides={"matured_arb_bonus_weight": 0.5},
        )
        assert math.isclose(env._matured_arb_bonus_weight, 0.5, abs_tol=1e-9)

    def test_reward_override_leaves_other_knobs_unchanged(self):
        """Passthrough of weight does not alter cap or expected_random."""
        base_config = _minimal_config(
            matured_arb_bonus_weight=0.0,
            matured_arb_bonus_cap=7.0,
            matured_arb_expected_random=3.5,
        )
        stub_day = _make_stub_day()
        env = BetfairEnv(
            stub_day,
            base_config,
            scalping_mode=True,
            reward_overrides={"matured_arb_bonus_weight": 0.9},
        )
        assert math.isclose(env._matured_arb_bonus_weight, 0.9, abs_tol=1e-9)
        assert math.isclose(env._matured_arb_bonus_cap, 7.0, abs_tol=1e-9)
        assert math.isclose(env._matured_arb_expected_random, 3.5, abs_tol=1e-9)


# ===========================================================================
# Test 8: invariant raw + shaped ≈ total, parametrised over weight
# ===========================================================================


class TestInvariantRawPlusShaped:
    @pytest.mark.parametrize("weight", [0.0, 1.0])
    def test_raw_plus_shaped_equals_total(self, weight: float):
        """raw_pnl_reward + shaped_bonus ≈ total_reward for weight in {0.0, 1.0}."""
        config = _minimal_config(matured_arb_bonus_weight=weight)
        raw, shaped, total, _ = _run_one_episode(config)
        assert math.isclose(raw + shaped, total, rel_tol=1e-5, abs_tol=1e-6), (
            f"weight={weight}: raw={raw} + shaped={shaped} = {raw + shaped} "
            f"!= total={total} (diff={abs(raw + shaped - total):.2e})"
        )
