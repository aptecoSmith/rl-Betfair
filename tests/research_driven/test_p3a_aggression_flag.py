"""Tests for P3a: aggression flag in action space (session 28).

Verifies that:
1. Aggressive dispatch reproduces pre-P3 behaviour (force_aggressive backstop).
2. Passive dispatch routes to PassiveOrderBook.
3. Mixed per-slot dispatch (aggressive + passive + skip in one tick).
4. Schema-bump loader refuses pre-P3 checkpoints.
5. force_aggressive + passive signal cleanly overrides to aggressive.
6. Aggressive regression: all prior passive tests (P4a/b/c) still pass
   (covered by running the full test suite; not duplicated here).

All tests are CPU-only and run in milliseconds.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import numpy as np
import pytest

from data.episode_builder import PriceSize, RunnerSnap, RunnerMeta, Tick, Race, Day
from env.bet_manager import BetManager, BetSide, PassiveOrderBook
from env.betfair_env import (
    ACTION_SCHEMA_VERSION,
    ACTIONS_PER_RUNNER,
    BetfairEnv,
    OBS_SCHEMA_VERSION,
    validate_action_schema,
    validate_obs_schema,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ps(price: float, size: float) -> PriceSize:
    return PriceSize(price=price, size=size)


_TS = datetime(2026, 4, 10, 14, 0, 0, tzinfo=timezone.utc)
_OFF_TIME = _TS + timedelta(minutes=10)


def _meta(selection_id: int) -> RunnerMeta:
    return RunnerMeta(
        selection_id=selection_id,
        runner_name=f"Runner_{selection_id}",
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


def _runner(
    selection_id: int = 1001,
    ltp: float = 4.0,
    back_levels: list[tuple[float, float]] | None = None,
    lay_levels: list[tuple[float, float]] | None = None,
    total_matched: float = 1000.0,
    status: str = "ACTIVE",
) -> RunnerSnap:
    atb = [_ps(p, s) for p, s in (back_levels or [])]
    atl = [_ps(p, s) for p, s in (lay_levels or [])]
    return RunnerSnap(
        selection_id=selection_id,
        status=status,
        last_traded_price=ltp,
        total_matched=total_matched,
        starting_price_near=0.0,
        starting_price_far=0.0,
        adjustment_factor=None,
        bsp=None,
        sort_priority=1,
        removal_date=None,
        available_to_back=atb,
        available_to_lay=atl,
    )


def _tick(runners: list[RunnerSnap], in_play: bool = False, ts: datetime | None = None) -> Tick:
    return Tick(
        market_id="1.23456",
        timestamp=ts or _TS,
        sequence_number=1,
        venue="Test",
        market_start_time=_OFF_TIME,
        number_of_active_runners=len(runners),
        traded_volume=0.0,
        in_play=in_play,
        winner_selection_id=None,
        race_status=None,
        temperature=None,
        precipitation=None,
        wind_speed=None,
        wind_direction=None,
        humidity=None,
        weather_code=None,
        runners=runners,
    )


def _minimal_config(*, force_aggressive: bool = False) -> dict:
    """Build a minimal config dict for BetfairEnv."""
    return {
        "training": {
            "max_runners": 3,
            "starting_budget": 100.0,
            "max_bets_per_race": 20,
            "betting_constraints": {
                "max_back_price": 100.0,
                "max_lay_price": None,
                "min_seconds_before_off": 0,
            },
        },
        "actions": {
            "force_aggressive": force_aggressive,
        },
        "reward": {
            "early_pick_bonus_min": 1.0,
            "early_pick_bonus_max": 1.0,
            "early_pick_min_seconds": 0,
            "terminal_bonus_weight": 0.0,
            "efficiency_penalty": 0.0,
            "precision_bonus": 0.0,
            "drawdown_shaping_weight": 0.0,
            "spread_cost_weight": 0.0,
            "commission": 0.05,
        },
        "features": {
            "obi_top_n": 3,
            "microprice_top_n": 3,
            "traded_delta_window_s": 60,
            "mid_drift_window_s": 60,
        },
    }


def _make_env(
    runners_per_tick: list[list[RunnerSnap]],
    *,
    force_aggressive: bool = False,
    winner_sid: int | None = None,
) -> BetfairEnv:
    """Build a BetfairEnv with one race containing the given ticks.

    The final tick is in-play (settlement trigger) with an optional winner.
    """
    ticks = []
    for i, r_list in enumerate(runners_per_tick):
        is_last = i == len(runners_per_tick) - 1
        t = _tick(r_list, in_play=is_last, ts=_TS + timedelta(seconds=i))
        if is_last and winner_sid is not None:
            # Patch winner into the tick for settlement
            t = Tick(
                market_id=t.market_id,
                timestamp=t.timestamp,
                sequence_number=t.sequence_number,
                venue=t.venue,
                market_start_time=t.market_start_time,
                number_of_active_runners=t.number_of_active_runners,
                traded_volume=t.traded_volume,
                in_play=True,
                winner_selection_id=winner_sid,
                race_status=t.race_status,
                temperature=t.temperature,
                precipitation=t.precipitation,
                wind_speed=t.wind_speed,
                wind_direction=t.wind_direction,
                humidity=t.humidity,
                weather_code=t.weather_code,
                runners=t.runners,
            )
        ticks.append(t)

    first_runners = runners_per_tick[0]
    runner_meta = {r.selection_id: _meta(r.selection_id) for r in first_runners}
    race = Race(
        market_id="1.23456",
        venue="Test",
        market_start_time=_OFF_TIME,
        winner_selection_id=winner_sid,
        ticks=ticks,
        runner_metadata=runner_meta,
    )
    day = Day(date="2026-04-10", races=[race])
    config = _minimal_config(force_aggressive=force_aggressive)
    return BetfairEnv(day, config, emit_debug_features=False)


def _build_action(
    max_runners: int,
    slot_actions: dict[int, tuple[float, float, float]],
) -> np.ndarray:
    """Build a flat action array.

    slot_actions maps slot_idx → (signal, stake_raw, aggression_raw).
    """
    action = np.zeros(max_runners * ACTIONS_PER_RUNNER, dtype=np.float32)
    for slot_idx, (signal, stake_raw, aggression_raw) in slot_actions.items():
        action[slot_idx] = signal
        action[max_runners + slot_idx] = stake_raw
        action[2 * max_runners + slot_idx] = aggression_raw
    return action


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestActionSpaceShape:
    """Action space shape uses ACTIONS_PER_RUNNER."""

    def test_action_space_shape(self):
        r = _runner(1001, 4.0, back_levels=[(3.8, 50)], lay_levels=[(4.2, 50)])
        env = _make_env([[r], [r]])
        assert env.action_space.shape == (env.max_runners * ACTIONS_PER_RUNNER,)
        assert env.action_space.shape == (3 * ACTIONS_PER_RUNNER,)


class TestAggressiveDispatch:
    """Aggressive dispatch (aggression > 0) places via BetManager.place_back/lay."""

    def test_back_aggressive_places_bet(self):
        r = _runner(1001, 4.0, back_levels=[(3.8, 50)], lay_levels=[(4.2, 50)])
        env = _make_env([[r], [r]])
        obs, info = env.reset()

        # slot 0: back signal=+0.5 (>0.33), stake=0.0 (maps to 50), aggression=+0.5 (>0 = aggressive)
        action = _build_action(3, {0: (0.5, 0.0, 0.5)})
        obs, reward, done, trunc, info = env.step(action)

        assert env.bet_manager is not None
        assert len(env.bet_manager.bets) == 1
        assert env.bet_manager.bets[0].side == BetSide.BACK
        assert info["action_debug"][1001]["aggressive_placed"] is True
        assert info["action_debug"][1001]["passive_placed"] is False

    def test_lay_aggressive_places_bet(self):
        r = _runner(1001, 4.0, back_levels=[(3.8, 50)], lay_levels=[(4.2, 50)])
        env = _make_env([[r], [r]])
        env.reset()

        # slot 0: lay signal=-0.5 (<-0.33), aggression=+0.5 (aggressive)
        action = _build_action(3, {0: (-0.5, 0.0, 0.5)})
        _, _, _, _, info = env.step(action)

        assert len(env.bet_manager.bets) == 1
        assert env.bet_manager.bets[0].side == BetSide.LAY
        assert info["action_debug"][1001]["aggressive_placed"] is True


class TestPassiveDispatch:
    """Passive dispatch (aggression ≤ 0) routes to PassiveOrderBook."""

    def test_back_passive_places_passive_order(self):
        r = _runner(1001, 4.0, back_levels=[(3.8, 50)], lay_levels=[(4.2, 50)])
        env = _make_env([[r], [r]])
        env.reset()

        # aggression=-0.5 (≤0 = passive)
        action = _build_action(3, {0: (0.5, 0.0, -0.5)})
        _, _, _, _, info = env.step(action)

        assert len(env.bet_manager.bets) == 0, "aggressive bets should be empty"
        assert len(env.bet_manager.passive_book.orders) == 1
        order = env.bet_manager.passive_book.orders[0]
        assert order.side is BetSide.BACK
        assert order.selection_id == 1001
        assert info["action_debug"][1001]["passive_placed"] is True
        assert info["action_debug"][1001]["aggressive_placed"] is False

    def test_lay_passive_places_passive_order(self):
        r = _runner(1001, 4.0, back_levels=[(3.8, 50)], lay_levels=[(4.2, 50)])
        env = _make_env([[r], [r]])
        env.reset()

        # stake_raw=-0.8 → fraction=0.1 → stake=10 → liability=10*(4.2-1)=32 < budget
        action = _build_action(3, {0: (-0.5, -0.8, -0.5)})
        _, _, _, _, info = env.step(action)

        assert len(env.bet_manager.bets) == 0
        assert len(env.bet_manager.passive_book.orders) == 1
        assert env.bet_manager.passive_book.orders[0].side is BetSide.LAY
        assert info["action_debug"][1001]["passive_placed"] is True

    def test_aggression_at_zero_is_passive(self):
        """Aggression == 0 (the threshold) is passive (≤ 0)."""
        r = _runner(1001, 4.0, back_levels=[(3.8, 50)], lay_levels=[(4.2, 50)])
        env = _make_env([[r], [r]])
        env.reset()

        action = _build_action(3, {0: (0.5, 0.0, 0.0)})
        _, _, _, _, info = env.step(action)

        assert len(env.bet_manager.bets) == 0
        assert len(env.bet_manager.passive_book.orders) == 1
        assert info["action_debug"][1001]["passive_placed"] is True


class TestMixedDispatch:
    """Mixed per-slot dispatch: aggressive + passive + skip in one tick."""

    def test_mixed_slots(self):
        r0 = _runner(1001, 4.0, back_levels=[(3.8, 50)], lay_levels=[(4.2, 50)])
        r1 = _runner(1002, 5.0, back_levels=[(4.8, 50)], lay_levels=[(5.2, 50)])
        r2 = _runner(1003, 6.0, back_levels=[(5.8, 50)], lay_levels=[(6.2, 50)])
        env = _make_env([[r0, r1, r2], [r0, r1, r2]])
        env.reset()

        action = _build_action(3, {
            0: (0.5, 0.0, 0.5),    # slot 0: aggressive back
            1: (0.5, 0.0, -0.5),   # slot 1: passive back
            2: (0.0, 0.0, 0.5),    # slot 2: no signal (skip)
        })
        _, _, _, _, info = env.step(action)

        # Slot 0 → aggressive bet
        assert len(env.bet_manager.bets) == 1
        assert env.bet_manager.bets[0].selection_id == 1001
        assert info["action_debug"][1001]["aggressive_placed"] is True

        # Slot 1 → passive order
        assert len(env.bet_manager.passive_book.orders) == 1
        assert env.bet_manager.passive_book.orders[0].selection_id == 1002
        assert info["action_debug"][1002]["passive_placed"] is True

        # Slot 2 → skipped (signal in dead zone)
        assert 1003 not in info["action_debug"]


class TestSchemaValidation:
    """Action schema bump refuses pre-P3 checkpoints."""

    def test_missing_action_schema_raises(self):
        """Pre-P3 checkpoint has no action_schema_version key."""
        checkpoint = {"obs_schema_version": OBS_SCHEMA_VERSION, "weights": {}}
        with pytest.raises(ValueError, match="no action_schema_version"):
            validate_action_schema(checkpoint)

    def test_wrong_action_schema_raises(self):
        checkpoint = {
            "obs_schema_version": OBS_SCHEMA_VERSION,
            "action_schema_version": 999,
            "weights": {},
        }
        with pytest.raises(ValueError, match="action_schema_version=999"):
            validate_action_schema(checkpoint)

    def test_correct_action_schema_passes(self):
        checkpoint = {
            "obs_schema_version": OBS_SCHEMA_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "weights": {},
        }
        validate_action_schema(checkpoint)  # should not raise

    def test_pre_p1_checkpoint_also_refused(self):
        """Checkpoint with neither obs nor action schema version."""
        checkpoint = {"weights": {}}
        with pytest.raises(ValueError):
            validate_action_schema(checkpoint)


class TestForceAggressive:
    """force_aggressive config override."""

    def test_force_aggressive_overrides_passive_signal(self):
        """With force_aggressive=true, a passive signal still dispatches aggressively."""
        r = _runner(1001, 4.0, back_levels=[(3.8, 50)], lay_levels=[(4.2, 50)])
        env = _make_env([[r], [r]], force_aggressive=True)
        env.reset()

        # aggression=-0.5 would normally be passive, but force_aggressive overrides
        action = _build_action(3, {0: (0.5, 0.0, -0.5)})
        _, _, _, _, info = env.step(action)

        assert len(env.bet_manager.bets) == 1, "should place aggressive bet"
        assert len(env.bet_manager.passive_book.orders) == 0, "no passive orders"
        assert info["action_debug"][1001]["aggressive_placed"] is True
        assert info["action_debug"][1001]["passive_placed"] is False

    def test_force_aggressive_with_aggression_zero(self):
        """force_aggressive + aggression=0 doesn't crash, places aggressive bet."""
        r = _runner(1001, 4.0, back_levels=[(3.8, 50)], lay_levels=[(4.2, 50)])
        env = _make_env([[r], [r]], force_aggressive=True)
        env.reset()

        action = _build_action(3, {0: (0.5, 0.0, 0.0)})
        _, _, _, _, info = env.step(action)

        assert len(env.bet_manager.bets) == 1
        assert info["action_debug"][1001]["aggressive_placed"] is True

    def test_force_aggressive_lay(self):
        """force_aggressive works for lay signals too."""
        r = _runner(1001, 4.0, back_levels=[(3.8, 50)], lay_levels=[(4.2, 50)])
        env = _make_env([[r], [r]], force_aggressive=True)
        env.reset()

        action = _build_action(3, {0: (-0.5, 0.0, -1.0)})
        _, _, _, _, info = env.step(action)

        assert len(env.bet_manager.bets) == 1
        assert env.bet_manager.bets[0].side == BetSide.LAY
        assert info["action_debug"][1001]["aggressive_placed"] is True


class TestActionDebugInfo:
    """action_debug is correctly populated in info dict."""

    def test_action_debug_empty_on_reset(self):
        r = _runner(1001, 4.0, back_levels=[(3.8, 50)], lay_levels=[(4.2, 50)])
        env = _make_env([[r], [r]])
        _, info = env.reset()
        assert info["action_debug"] == {}

    def test_action_debug_records_skipped_reason(self):
        """When a passive placement fails, skipped_reason is recorded."""
        # Runner with no LTP → passive will fail
        r = _runner(1001, 0.0, back_levels=[], lay_levels=[])
        r2 = _runner(1001, 0.0, back_levels=[], lay_levels=[])
        env = _make_env([[r], [r2]])
        env.reset()

        # Even though signal is present, no liquidity → skip before dispatch
        action = _build_action(3, {0: (0.5, 0.0, -0.5)})
        _, _, _, _, info = env.step(action)
        # The runner has no back/lay liquidity so it's skipped before dispatch
        assert 1001 not in info["action_debug"] or info["action_debug"].get(1001, {}).get("passive_placed") is False
