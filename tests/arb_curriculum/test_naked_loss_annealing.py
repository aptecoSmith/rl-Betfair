"""Tests for arb-curriculum Session 03: naked-loss scale gene + annealing.

Categories (per hard_constraints.md §29):
1. scale=1.0 byte-identical (no-op)
2. scale=0.5 halves loss magnitude in raw reward
3. scale=0.0 zeros losses but preserves winners
4. winner side untouched across all scales
5. invariant raw+shaped≈total preserved at scale ∈ {0.5, 1.0}
6. annealing interpolation over 5 generations of a {start:0, end:4} schedule
7. annealing degenerate: {start:0, end:0} → always 1.0
8. JSONL field present in info
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pytest

from data.episode_builder import Day, PriceSize, Race, RunnerMeta, RunnerSnap, Tick
from env.betfair_env import BetfairEnv, _compute_scalping_reward_terms
from training.arb_annealing import anneal_factor, effective_naked_loss_scale

# ---------------------------------------------------------------------------
# Shared helpers (same pattern as test_matured_arb_bonus.py)
# ---------------------------------------------------------------------------

_MARKET_START = datetime(2026, 4, 20, 14, 0, 0)


def _minimal_config(
    naked_loss_scale: float = 1.0,
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
            "naked_loss_scale": naked_loss_scale,
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


def _run_one_episode(config: dict, n_ticks: int = 3) -> tuple[float, float, float, dict]:
    stub_day = _make_stub_day(n_ticks=n_ticks)
    env = BetfairEnv(stub_day, config, scalping_mode=True)
    obs, info = env.reset()
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
# Test 1: scale=1.0 is byte-identical to no-scale
# ===========================================================================


class TestScaleOneByteIdentical:
    def test_two_runs_at_scale_one_match(self):
        """scale=1.0 produces identical raw/shaped/total on two runs."""
        config = _minimal_config(naked_loss_scale=1.0)
        raw0, shaped0, total0, _ = _run_one_episode(config)
        raw1, shaped1, total1, _ = _run_one_episode(config)
        assert math.isclose(raw0, raw1, abs_tol=1e-9)
        assert math.isclose(shaped0, shaped1, abs_tol=1e-9)
        assert math.isclose(total0, total1, abs_tol=1e-9)

    def test_compute_terms_scale_one_no_op(self):
        """_compute_scalping_reward_terms with scale=1.0 leaves race_pnl unchanged."""
        race_pnl = -50.0
        naked_per_pair = [-20.0, -30.0]
        raw_1, _ = _compute_scalping_reward_terms(
            race_pnl=race_pnl,
            naked_per_pair=naked_per_pair,
            n_close_signal_successes=0,
            naked_loss_scale=1.0,
        )
        assert math.isclose(raw_1, race_pnl, abs_tol=1e-9)


# ===========================================================================
# Test 2: scale=0.5 halves loss magnitude
# ===========================================================================


class TestScaleHalfLoss:
    def test_half_scale_reduces_loss_by_half(self):
        """scale=0.5 → raw = race_pnl - 0.5 * loss_sum."""
        race_pnl = -50.0           # locked=0, closed=0, naked=-50
        naked_per_pair = [-50.0]   # single naked loss of -50
        loss_sum = -50.0

        raw_half, _ = _compute_scalping_reward_terms(
            race_pnl=race_pnl,
            naked_per_pair=naked_per_pair,
            n_close_signal_successes=0,
            naked_loss_scale=0.5,
        )
        expected = race_pnl - (1.0 - 0.5) * loss_sum  # -50 + 25 = -25
        assert math.isclose(raw_half, expected, abs_tol=1e-9)
        assert math.isclose(raw_half, -25.0, abs_tol=1e-9)

    def test_half_scale_mixed_pairs(self):
        """scale=0.5 halves only losses; winner contribution unchanged."""
        naked_per_pair = [+100.0, -80.0]
        race_pnl = 100.0 - 80.0  # = 20.0 (ignoring locked/closed)
        loss_sum = -80.0

        raw, _ = _compute_scalping_reward_terms(
            race_pnl=race_pnl,
            naked_per_pair=naked_per_pair,
            n_close_signal_successes=0,
            naked_loss_scale=0.5,
        )
        expected = race_pnl - 0.5 * loss_sum  # 20 + 40 = 60
        assert math.isclose(raw, expected, abs_tol=1e-9)


# ===========================================================================
# Test 3: scale=0.0 zeros losses, preserves winners
# ===========================================================================


class TestScaleZero:
    def test_zero_scale_removes_losses(self):
        """scale=0.0 → losses contribute 0 to raw; winners unchanged."""
        naked_per_pair = [-40.0, -10.0]
        # race_pnl = 0 (locked) + 0 (closed) + naked_sum = -50
        race_pnl = -50.0
        loss_sum = -50.0

        raw, _ = _compute_scalping_reward_terms(
            race_pnl=race_pnl,
            naked_per_pair=naked_per_pair,
            n_close_signal_successes=0,
            naked_loss_scale=0.0,
        )
        expected = race_pnl - 1.0 * loss_sum  # -50 + 50 = 0
        assert math.isclose(raw, expected, abs_tol=1e-9)
        assert math.isclose(raw, 0.0, abs_tol=1e-9)

    def test_zero_scale_only_naked_winners(self):
        """scale=0.0 does NOT change naked winners (only loss side)."""
        naked_per_pair = [+100.0]  # winner only
        race_pnl = 100.0
        loss_sum = 0.0

        raw, _ = _compute_scalping_reward_terms(
            race_pnl=race_pnl,
            naked_per_pair=naked_per_pair,
            n_close_signal_successes=0,
            naked_loss_scale=0.0,
        )
        expected = race_pnl - 1.0 * loss_sum  # 100 - 0 = 100
        assert math.isclose(raw, expected, abs_tol=1e-9)


# ===========================================================================
# Test 4: winner side untouched across all scales
# ===========================================================================


class TestWinnerSideUntouched:
    @pytest.mark.parametrize("scale", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_winner_raw_independent_of_scale(self, scale: float):
        """race_pnl with only naked winners is independent of naked_loss_scale."""
        naked_per_pair = [+50.0, +30.0]
        race_pnl = 80.0  # all winners, no losses

        raw, _ = _compute_scalping_reward_terms(
            race_pnl=race_pnl,
            naked_per_pair=naked_per_pair,
            n_close_signal_successes=0,
            naked_loss_scale=scale,
        )
        # loss_sum = 0 → no adjustment regardless of scale
        assert math.isclose(raw, race_pnl, abs_tol=1e-9), (
            f"scale={scale}: raw={raw} != race_pnl={race_pnl}"
        )


# ===========================================================================
# Test 5: invariant raw + shaped ≈ total at scale ∈ {0.5, 1.0}
# ===========================================================================


class TestInvariantRawPlusShapedPreserved:
    @pytest.mark.parametrize("scale", [0.5, 1.0])
    def test_raw_plus_shaped_equals_total(self, scale: float):
        """raw_pnl_reward + shaped_bonus ≈ total_reward for given scale."""
        config = _minimal_config(naked_loss_scale=scale)
        raw, shaped, total, _ = _run_one_episode(config)
        assert math.isclose(raw + shaped, total, rel_tol=1e-5, abs_tol=1e-6), (
            f"scale={scale}: raw={raw} + shaped={shaped} = {raw + shaped} "
            f"!= total={total} (diff={abs(raw + shaped - total):.2e})"
        )


# ===========================================================================
# Test 6: annealing interpolation — {start:0, end:4} over 5 generations
# ===========================================================================


class TestAnnealInterpolation:
    def test_factor_before_start(self):
        assert math.isclose(anneal_factor(current_gen=-1, start=0, end=4), 0.0)

    def test_factor_at_start(self):
        assert math.isclose(anneal_factor(current_gen=0, start=0, end=4), 0.0)

    def test_factor_midway(self):
        assert math.isclose(anneal_factor(current_gen=2, start=0, end=4), 0.5)

    def test_factor_at_end(self):
        assert math.isclose(anneal_factor(current_gen=4, start=0, end=4), 1.0)

    def test_factor_after_end(self):
        assert math.isclose(anneal_factor(current_gen=100, start=0, end=4), 1.0)

    def test_effective_scale_interpolates_toward_one(self):
        """effective_naked_loss_scale transitions gene→1.0 over [0,4)."""
        gene = 0.2
        schedule = {"start_gen": 0, "end_gen": 4}

        # gen=0: factor=0 → effective = 0.2 + (1-0.2)*0 = 0.2
        assert math.isclose(
            effective_naked_loss_scale(gene, 0, schedule), 0.2, abs_tol=1e-9
        )
        # gen=2: factor=0.5 → effective = 0.2 + 0.8*0.5 = 0.6
        assert math.isclose(
            effective_naked_loss_scale(gene, 2, schedule), 0.6, abs_tol=1e-9
        )
        # gen=4: factor=1.0 → effective = 0.2 + 0.8*1 = 1.0
        assert math.isclose(
            effective_naked_loss_scale(gene, 4, schedule), 1.0, abs_tol=1e-9
        )

    def test_effective_scale_no_schedule_returns_gene(self):
        """No schedule → returns gene unchanged."""
        assert math.isclose(
            effective_naked_loss_scale(0.3, 5, None), 0.3, abs_tol=1e-9
        )


# ===========================================================================
# Test 7: annealing degenerate — {start:0, end:0} → always 1.0
# ===========================================================================


class TestAnnealDegenerate:
    def test_start_equals_end_always_returns_one(self):
        """end <= start → factor=1.0 → effective_scale always 1.0."""
        schedule = {"start_gen": 0, "end_gen": 0}
        for gen in range(5):
            result = effective_naked_loss_scale(0.1, gen, schedule)
            assert math.isclose(result, 1.0, abs_tol=1e-9), (
                f"gen={gen}: expected 1.0, got {result}"
            )

    def test_degenerate_anneal_factor(self):
        assert math.isclose(anneal_factor(0, 2, 2), 1.0)
        assert math.isclose(anneal_factor(0, 5, 3), 1.0)


# ===========================================================================
# Test 8: JSONL field present in info
# ===========================================================================


class TestJsonlFieldPresent:
    def test_naked_loss_scale_active_in_info(self):
        """_get_info() includes 'naked_loss_scale_active' equal to config value."""
        config = _minimal_config(naked_loss_scale=0.7)
        _, _, _, last_info = _run_one_episode(config)
        assert "naked_loss_scale_active" in last_info
        assert math.isclose(last_info["naked_loss_scale_active"], 0.7, abs_tol=1e-9)

    def test_field_present_at_default_scale(self):
        """Field is present even at default scale=1.0."""
        config = _minimal_config(naked_loss_scale=1.0)
        _, _, _, last_info = _run_one_episode(config)
        assert "naked_loss_scale_active" in last_info
        assert math.isclose(last_info["naked_loss_scale_active"], 1.0, abs_tol=1e-9)

    def test_gene_passthrough_via_reward_overrides(self):
        """reward_overrides={'naked_loss_scale': 0.3} sets _naked_loss_scale."""
        base_config = _minimal_config(naked_loss_scale=1.0)
        stub_day = _make_stub_day()
        env = BetfairEnv(
            stub_day,
            base_config,
            scalping_mode=True,
            reward_overrides={"naked_loss_scale": 0.3},
        )
        assert math.isclose(env._naked_loss_scale, 0.3, abs_tol=1e-9)
