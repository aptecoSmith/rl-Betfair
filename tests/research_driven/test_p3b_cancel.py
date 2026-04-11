"""Tests for P3b: cancel action in action space (session 29).

Verifies that:
1. Cancel of a resting order releases budget.
2. Cancel with nothing to cancel is a no-op.
3. Cancel oldest: two passives → oldest cancelled, newer survives.
4. Cancel + place in same tick (atomic move to new price).
5. Cancelled passive contributes zero P&L.
6. Efficiency-penalty interaction: cancelled passives count toward bet_count.
7. Cancel does not affect aggressive bets.
8. Schema-bump loader refuses pre-P3b (v1) checkpoints.
9. raw + shaped ≈ total_reward invariant holds.

All CPU, all fast.
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
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ps(price: float, size: float) -> PriceSize:
    return PriceSize(price=price, size=size)


_TS = datetime(2026, 4, 11, 14, 0, 0, tzinfo=timezone.utc)
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


def _minimal_config(*, efficiency_penalty: float = 0.0) -> dict:
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
            "force_aggressive": False,
        },
        "reward": {
            "early_pick_bonus_min": 1.0,
            "early_pick_bonus_max": 1.0,
            "early_pick_min_seconds": 0,
            "terminal_bonus_weight": 0.0,
            "efficiency_penalty": efficiency_penalty,
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
    winner_sid: int | None = None,
    efficiency_penalty: float = 0.0,
) -> BetfairEnv:
    """Build a BetfairEnv with one race containing the given ticks."""
    ticks = []
    for i, r_list in enumerate(runners_per_tick):
        is_last = i == len(runners_per_tick) - 1
        t = _tick(r_list, in_play=is_last, ts=_TS + timedelta(seconds=i))
        if is_last and winner_sid is not None:
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
    day = Day(date="2026-04-11", races=[race])
    config = _minimal_config(efficiency_penalty=efficiency_penalty)
    return BetfairEnv(day, config, emit_debug_features=False)


def _build_action(
    max_runners: int,
    slot_actions: dict[int, tuple[float, float, float, float]],
) -> np.ndarray:
    """Build a flat action array.

    slot_actions maps slot_idx → (signal, stake_raw, aggression_raw, cancel_raw).
    """
    action = np.zeros(max_runners * ACTIONS_PER_RUNNER, dtype=np.float32)
    for slot_idx, (signal, stake_raw, aggression_raw, cancel_raw) in slot_actions.items():
        action[slot_idx] = signal
        action[max_runners + slot_idx] = stake_raw
        action[2 * max_runners + slot_idx] = aggression_raw
        action[3 * max_runners + slot_idx] = cancel_raw
    return action


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestActionSpaceShape:
    """Action space shape reflects ACTIONS_PER_RUNNER=4."""

    def test_action_space_shape(self):
        r = _runner(1001, 4.0, back_levels=[(3.8, 50)], lay_levels=[(4.2, 50)])
        env = _make_env([[r], [r]])
        assert env.action_space.shape == (env.max_runners * ACTIONS_PER_RUNNER,)
        assert env.action_space.shape == (3 * 4,)  # 3 runners × 4 values


class TestCancelReleasesBudget:
    """Test 1: cancel of a resting order releases budget."""

    def test_back_cancel_releases_budget(self):
        r = _runner(1001, 4.0, back_levels=[(3.8, 50)], lay_levels=[(4.2, 50)])
        # tick0: place passive, tick1: cancel, tick2: settle
        env = _make_env([[r], [r], [r]])
        env.reset()

        budget_before = env.bet_manager.budget

        # Place passive back — budget decreases
        action_place = _build_action(3, {0: (0.5, 0.0, -0.5, -1.0)})  # no cancel
        env.step(action_place)
        assert len(env.bet_manager.passive_book.orders) == 1
        budget_after_place = env.bet_manager.budget
        assert budget_after_place < budget_before

        # Cancel — budget restored
        action_cancel = _build_action(3, {0: (0.0, 0.0, 0.0, 0.5)})  # cancel=+0.5
        env.step(action_cancel)
        assert len(env.bet_manager.passive_book.orders) == 0
        assert env.bet_manager.budget == pytest.approx(budget_before)

    def test_lay_cancel_releases_liability(self):
        r = _runner(1001, 4.0, back_levels=[(3.8, 50)], lay_levels=[(4.2, 50)])
        env = _make_env([[r], [r], [r]])
        env.reset()

        liability_before = env.bet_manager.open_liability
        assert liability_before == 0.0

        # Place passive lay — liability increases
        action_place = _build_action(3, {0: (-0.5, -0.8, -0.5, -1.0)})
        env.step(action_place)
        assert len(env.bet_manager.passive_book.orders) == 1
        assert env.bet_manager.open_liability > 0.0

        # Cancel — liability restored
        action_cancel = _build_action(3, {0: (0.0, 0.0, 0.0, 0.5)})
        env.step(action_cancel)
        assert len(env.bet_manager.passive_book.orders) == 0
        assert env.bet_manager.open_liability == pytest.approx(0.0)


class TestCancelNoopWhenEmpty:
    """Test 2: cancel with nothing to cancel is a no-op."""

    def test_cancel_empty_no_exception(self):
        r = _runner(1001, 4.0, back_levels=[(3.8, 50)], lay_levels=[(4.2, 50)])
        env = _make_env([[r], [r]])
        env.reset()

        budget_before = env.bet_manager.budget

        # Emit cancel on a slot with no passive orders — no error
        action = _build_action(3, {0: (0.0, 0.0, 0.0, 0.5)})
        obs, reward, done, trunc, info = env.step(action)

        assert env.bet_manager.budget == pytest.approx(budget_before)
        assert len(env.bet_manager.passive_book.cancelled_orders) == 0


class TestCancelOldest:
    """Test 3: cancel oldest — two passives, oldest cancelled, newer survives."""

    def test_cancel_oldest_of_two(self):
        r = _runner(1001, 4.0, back_levels=[(3.8, 50)], lay_levels=[(4.2, 50)])
        # Need 4 ticks: place1, place2, cancel, settle
        env = _make_env([[r], [r], [r], [r]])
        env.reset()

        # Place first passive (at price 3.8)
        action1 = _build_action(3, {0: (0.5, -0.8, -0.5, -1.0)})  # small stake
        env.step(action1)
        assert len(env.bet_manager.passive_book.orders) == 1
        first_order_tick = env.bet_manager.passive_book.orders[0].placed_tick_index

        # Place second passive
        action2 = _build_action(3, {0: (0.5, -0.8, -0.5, -1.0)})
        env.step(action2)
        assert len(env.bet_manager.passive_book.orders) == 2
        second_order_tick = env.bet_manager.passive_book.orders[1].placed_tick_index
        assert second_order_tick > first_order_tick

        # Cancel — should remove oldest
        action_cancel = _build_action(3, {0: (0.0, 0.0, 0.0, 0.5)})
        env.step(action_cancel)
        remaining = env.bet_manager.passive_book.orders
        assert len(remaining) == 1
        assert remaining[0].placed_tick_index == second_order_tick  # newer survives


class TestCancelAndPlaceSameTick:
    """Test 4: cancel + place in same tick — atomic price move."""

    def test_cancel_then_place_same_tick(self):
        r = _runner(1001, 4.0, back_levels=[(3.8, 50)], lay_levels=[(4.2, 50)])
        # tick0: place, tick1: cancel+place, tick2: settle
        env = _make_env([[r], [r], [r]])
        env.reset()

        budget_start = env.bet_manager.budget

        # Place passive back
        action_place = _build_action(3, {0: (0.5, -0.8, -0.5, -1.0)})
        env.step(action_place)
        assert len(env.bet_manager.passive_book.orders) == 1
        old_tick = env.bet_manager.passive_book.orders[0].placed_tick_index
        budget_after_first = env.bet_manager.budget

        # Cancel + place in same tick (cancel runs first, then place)
        action_both = _build_action(3, {0: (0.5, -0.8, -0.5, 0.5)})  # cancel=+0.5, signal=back
        _, _, _, _, info = env.step(action_both)

        orders = env.bet_manager.passive_book.orders
        assert len(orders) == 1, "should have exactly one order (old cancelled, new placed)"
        assert orders[0].placed_tick_index > old_tick, "new order on later tick"
        # Budget: the old stake was released, new stake reserved — net should be same
        assert env.bet_manager.budget == pytest.approx(budget_after_first)
        # action_debug should show both cancel and place
        assert info["action_debug"][1001]["cancelled"] is True
        assert info["action_debug"][1001]["passive_placed"] is True


class TestCancelledPassiveZeroPnL:
    """Test 5: cancelled passive contributes zero P&L."""

    def test_policy_cancelled_passive_zero_pnl(self):
        r = _runner(1001, 4.0, back_levels=[(3.8, 50)], lay_levels=[(4.2, 50)])
        # tick0: place, tick1: cancel, tick2: settle (in-play)
        env = _make_env([[r], [r], [r]], winner_sid=1001)
        env.reset()

        budget_start = env.bet_manager.budget

        # Place passive
        action_place = _build_action(3, {0: (0.5, -0.8, -0.5, -1.0)})
        env.step(action_place)

        # Cancel
        action_cancel = _build_action(3, {0: (0.0, 0.0, 0.0, 0.5)})
        env.step(action_cancel)

        # Step to settlement
        action_noop = _build_action(3, {})
        _, _, done, _, info = env.step(action_noop)

        assert done
        # No bets matched → P&L is zero
        assert info["day_pnl"] == pytest.approx(0.0)


class TestEfficiencyPenaltyInteraction:
    """Test 6: efficiency penalty includes policy-cancelled passives."""

    def test_cancel_counts_toward_efficiency(self):
        r = _runner(1001, 4.0, back_levels=[(3.8, 50)], lay_levels=[(4.2, 50)])
        penalty = 0.01
        # tick0: place, tick1: cancel, tick2: settle
        env = _make_env([[r], [r], [r]], winner_sid=1001, efficiency_penalty=penalty)
        env.reset()

        # Place passive
        action_place = _build_action(3, {0: (0.5, -0.8, -0.5, -1.0)})
        env.step(action_place)

        # Cancel via policy
        action_cancel = _build_action(3, {0: (0.0, 0.0, 0.0, 0.5)})
        env.step(action_cancel)

        # Settle
        action_noop = _build_action(3, {})
        _, _, done, _, info = env.step(action_noop)

        assert done
        # 1 policy cancel + race-off cancel_all (but that's 0 since we already cancelled)
        # The shaped_bonus should include the efficiency penalty for the cancel
        # Total cancel count: 1 (policy cancel). No matched bets.
        # efficiency_cost = (0 matched + 1 cancelled) × 0.01 = -0.01
        assert info["shaped_bonus"] == pytest.approx(-penalty)


class TestCancelDoesNotAffectAggressive:
    """Test 7: cancel does not affect aggressive bets."""

    def test_aggressive_unaffected_by_cancel_signal(self):
        r = _runner(1001, 4.0, back_levels=[(3.8, 50)], lay_levels=[(4.2, 50)])
        r2 = _runner(1002, 5.0, back_levels=[(4.8, 50)], lay_levels=[(5.2, 50)])
        # tick0: aggressive + cancel on different slots, tick1: settle
        env = _make_env([[r, r2], [r, r2]], winner_sid=1001)
        env.reset()

        # slot 0: aggressive back + cancel signal
        # slot 1: aggressive back, no cancel
        action = _build_action(3, {
            0: (0.5, 0.0, 0.5, 0.5),   # aggressive + cancel (nothing to cancel)
            1: (0.5, 0.0, 0.5, -1.0),  # aggressive, no cancel
        })
        _, _, _, _, info = env.step(action)

        assert len(env.bet_manager.bets) == 2
        assert env.bet_manager.bets[0].side == BetSide.BACK
        assert env.bet_manager.bets[1].side == BetSide.BACK
        # Cancel signal on slot 0 is a no-op (no passive to cancel)
        assert info["action_debug"][1001]["aggressive_placed"] is True
        assert info["action_debug"][1001]["cancelled"] is False  # nothing was cancelled


class TestSchemaBumpRefusesPreP3b:
    """Test 8: schema-bump loader refuses pre-P3b (v1) checkpoints."""

    def test_v1_checkpoint_refused(self):
        """Session-28-only checkpoint (v1) refused by v2 loader."""
        checkpoint = {
            "obs_schema_version": OBS_SCHEMA_VERSION,
            "action_schema_version": 1,
            "weights": {},
        }
        with pytest.raises(ValueError, match="action_schema_version=1"):
            validate_action_schema(checkpoint)

    def test_v2_checkpoint_accepted(self):
        checkpoint = {
            "obs_schema_version": OBS_SCHEMA_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "weights": {},
        }
        validate_action_schema(checkpoint)  # should not raise

    def test_no_schema_refused(self):
        checkpoint = {"obs_schema_version": OBS_SCHEMA_VERSION, "weights": {}}
        with pytest.raises(ValueError, match="no action_schema_version"):
            validate_action_schema(checkpoint)


class TestRawShapedInvariant:
    """Test 9: raw + shaped ≈ total_reward invariant holds."""

    def test_invariant_with_cancel(self):
        r = _runner(1001, 4.0, back_levels=[(3.8, 50)], lay_levels=[(4.2, 50)])
        penalty = 0.01
        env = _make_env([[r], [r], [r]], winner_sid=1001, efficiency_penalty=penalty)
        env.reset()

        total_reward = 0.0

        # Place passive
        action_place = _build_action(3, {0: (0.5, -0.8, -0.5, -1.0)})
        _, reward, _, _, _ = env.step(action_place)
        total_reward += reward

        # Cancel
        action_cancel = _build_action(3, {0: (0.0, 0.0, 0.0, 0.5)})
        _, reward, _, _, _ = env.step(action_cancel)
        total_reward += reward

        # Settle
        action_noop = _build_action(3, {})
        _, reward, done, _, info = env.step(action_noop)
        total_reward += reward

        assert done
        raw = info["raw_pnl_reward"]
        shaped = info["shaped_bonus"]
        assert raw + shaped == pytest.approx(total_reward, abs=1e-6)
