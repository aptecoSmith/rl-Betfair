"""Tests for ``agents_v2.action_space``.

Phase 1, Session 01 — locked discrete action space + masking helpers.
The tests exercise pure index math (no env required) and the masking
helper against a synthetic ``BetfairEnv`` built from the standard
test fixtures in ``tests.test_betfair_env``.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from data.episode_builder import Day, PriceSize, Race, RunnerSnap, Tick
from env.bet_manager import MIN_BET_STAKE
from env.betfair_env import BetfairEnv

from agents_v2.action_space import (
    ActionType,
    DiscreteActionSpace,
    compute_mask,
)
from tests.test_betfair_env import (
    _make_day,
    _make_runner_meta,
    _make_runner_snap,
)


# ── DiscreteActionSpace ─────────────────────────────────────────────────────


class TestDiscreteActionSpace:
    def test_n_matches_locked_formula(self):
        space = DiscreteActionSpace(max_runners=14)
        assert space.n == 1 + 3 * 14

    def test_max_runners_must_be_positive(self):
        with pytest.raises(ValueError):
            DiscreteActionSpace(max_runners=0)

    def test_index_layout_round_trip(self):
        """``encode(decode(i)) == i`` for every i in [0, n)."""
        space = DiscreteActionSpace(max_runners=14)
        for i in range(space.n):
            kind, runner_idx = space.decode(i)
            assert space.encode(kind, runner_idx) == i

    def test_decode_layout_boundaries(self):
        space = DiscreteActionSpace(max_runners=4)
        # 0 is no-op
        assert space.decode(0) == (ActionType.NOOP, None)
        # 1..4 are open_back_0..3
        assert space.decode(1) == (ActionType.OPEN_BACK, 0)
        assert space.decode(4) == (ActionType.OPEN_BACK, 3)
        # 5..8 are open_lay_0..3
        assert space.decode(5) == (ActionType.OPEN_LAY, 0)
        assert space.decode(8) == (ActionType.OPEN_LAY, 3)
        # 9..12 are close_0..3
        assert space.decode(9) == (ActionType.CLOSE, 0)
        assert space.decode(12) == (ActionType.CLOSE, 3)

    def test_decode_rejects_out_of_range(self):
        space = DiscreteActionSpace(max_runners=4)
        with pytest.raises(ValueError):
            space.decode(-1)
        with pytest.raises(ValueError):
            space.decode(space.n)

    def test_encode_rejects_runner_idx_for_noop(self):
        space = DiscreteActionSpace(max_runners=4)
        with pytest.raises(ValueError):
            space.encode(ActionType.NOOP, 0)

    def test_encode_requires_runner_idx_for_open_close(self):
        space = DiscreteActionSpace(max_runners=4)
        with pytest.raises(ValueError):
            space.encode(ActionType.OPEN_BACK, None)
        with pytest.raises(ValueError):
            space.encode(ActionType.CLOSE, None)

    def test_encode_rejects_out_of_range_runner_idx(self):
        space = DiscreteActionSpace(max_runners=4)
        with pytest.raises(ValueError):
            space.encode(ActionType.OPEN_BACK, 4)
        with pytest.raises(ValueError):
            space.encode(ActionType.CLOSE, -1)


# ── Mask helper fixtures ────────────────────────────────────────────────────


def _scalping_config() -> dict:
    return {
        "training": {
            "max_runners": 4,
            "starting_budget": 100.0,
            "max_bets_per_race": 20,
            "scalping_mode": True,
        },
        "actions": {"force_aggressive": True},
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "efficiency_penalty": 0.01,
        },
    }


def _make_day_with_first_runner(
    *,
    status: str = "ACTIVE",
    ltp: float = 4.0,
) -> Day:
    """Build a 1-race day where runner 101 (slot 0) has the given state.

    Other runners (102, 103) stay healthy ACTIVE/4.0 so they can be
    used as the "control" slots in mask tests.
    """
    market_id = "1.20000001"
    start_time = datetime(2026, 3, 26, 14, 0, 0)
    runners = [
        _make_runner_snap(101, ltp=ltp, status=status),
        _make_runner_snap(102),
        _make_runner_snap(103),
    ]
    ticks: list[Tick] = []
    for i in range(5):
        ts = start_time - timedelta(seconds=600 - i * 5)
        ticks.append(Tick(
            market_id=market_id,
            timestamp=ts,
            sequence_number=i,
            venue="Newmarket",
            market_start_time=start_time,
            number_of_active_runners=len(runners),
            traded_volume=10000.0,
            in_play=False,
            winner_selection_id=101,
            race_status=None,
            temperature=15.0,
            precipitation=0.0,
            wind_speed=5.0,
            wind_direction=180.0,
            humidity=60.0,
            weather_code=0,
            runners=runners,
        ))
    meta = {sid: _make_runner_meta(sid) for sid in (101, 102, 103)}
    race = Race(
        market_id=market_id,
        venue="Newmarket",
        market_start_time=start_time,
        winner_selection_id=101,
        ticks=ticks,
        runner_metadata=meta,
    )
    return Day(date="2026-03-26", races=[race])


# ── compute_mask ────────────────────────────────────────────────────────────


class TestComputeMask:
    def _reset_env(self, day: Day | None = None) -> BetfairEnv:
        env = BetfairEnv(day or _make_day(n_races=1), _scalping_config())
        env.reset()
        return env

    def test_noop_always_legal(self):
        space = DiscreteActionSpace(max_runners=4)
        env = self._reset_env()
        mask = compute_mask(space, env)
        assert mask[0]

    def test_open_unmasked_for_healthy_active_runner(self):
        """Sanity-check: an ACTIVE runner with LTP > 1 and zero bets is openable."""
        space = DiscreteActionSpace(max_runners=4)
        env = self._reset_env()
        mask = compute_mask(space, env)
        assert mask[space.encode(ActionType.OPEN_BACK, 0)]
        assert mask[space.encode(ActionType.OPEN_LAY, 0)]

    def test_open_masked_when_runner_inactive(self):
        space = DiscreteActionSpace(max_runners=4)
        day = _make_day_with_first_runner(status="REMOVED")
        env = self._reset_env(day)
        mask = compute_mask(space, env)
        assert not mask[space.encode(ActionType.OPEN_BACK, 0)]
        assert not mask[space.encode(ActionType.OPEN_LAY, 0)]
        # Healthy runner in slot 1 (sid 102) is still openable.
        assert mask[space.encode(ActionType.OPEN_BACK, 1)]
        assert mask[space.encode(ActionType.OPEN_LAY, 1)]

    def test_open_masked_when_no_ltp(self):
        space = DiscreteActionSpace(max_runners=4)
        # Sentinel for "unpriceable" — the matcher's junk filter rejects
        # anything with ltp <= 1.0 (or None).
        day = _make_day_with_first_runner(ltp=1.0)
        env = self._reset_env(day)
        mask = compute_mask(space, env)
        assert not mask[space.encode(ActionType.OPEN_BACK, 0)]
        assert not mask[space.encode(ActionType.OPEN_LAY, 0)]

    def test_open_masked_when_already_open_position(self):
        """An unsettled bet on runner 101 makes both opens illegal."""
        space = DiscreteActionSpace(max_runners=4)
        env = self._reset_env()
        race = env.day.races[env._race_idx]
        tick = race.ticks[env._tick_idx]
        runner_101 = next(r for r in tick.runners if r.selection_id == 101)
        env.bet_manager.place_back(
            runner_101, stake=10.0, market_id=race.market_id, pair_id="p1",
        )
        mask = compute_mask(space, env)
        assert not mask[space.encode(ActionType.OPEN_BACK, 0)]
        assert not mask[space.encode(ActionType.OPEN_LAY, 0)]
        # Slot 1 (sid 102) untouched.
        assert mask[space.encode(ActionType.OPEN_BACK, 1)]

    def test_close_masked_when_no_open_pair(self):
        space = DiscreteActionSpace(max_runners=4)
        env = self._reset_env()
        mask = compute_mask(space, env)
        for slot in range(4):
            assert not mask[space.encode(ActionType.CLOSE, slot)]

    def test_close_unmasked_after_open(self):
        """An aggressive leg with an unfilled passive partner ⇒ close legal."""
        space = DiscreteActionSpace(max_runners=4)
        env = self._reset_env()
        race = env.day.races[env._race_idx]
        tick = race.ticks[env._tick_idx]
        runner_102 = next(r for r in tick.runners if r.selection_id == 102)
        # Place an aggressive leg with a pair_id; passive never fills,
        # so the pair is INCOMPLETE → close is legal on slot 1.
        bet = env.bet_manager.place_back(
            runner_102, stake=10.0, market_id=race.market_id, pair_id="pp",
        )
        assert bet is not None  # sanity — the test runs against the matcher
        mask = compute_mask(space, env)
        assert mask[space.encode(ActionType.CLOSE, 1)]
        # Slots without a pair stay masked off for close.
        assert not mask[space.encode(ActionType.CLOSE, 0)]
        assert not mask[space.encode(ActionType.CLOSE, 2)]

    def test_open_masked_when_budget_below_min_stake(self):
        space = DiscreteActionSpace(max_runners=4)
        env = self._reset_env()
        # Manually deplete the budget below MIN_BET_STAKE.
        env.bet_manager.budget = MIN_BET_STAKE - 0.5
        mask = compute_mask(space, env)
        for slot in range(4):
            assert not mask[space.encode(ActionType.OPEN_BACK, slot)]
            assert not mask[space.encode(ActionType.OPEN_LAY, slot)]
        # No-op stays legal regardless of budget.
        assert mask[0]

    def test_no_op_legal_when_env_unreset(self):
        """Mask without a reset must not crash — no-op only."""
        space = DiscreteActionSpace(max_runners=4)
        env = BetfairEnv(_make_day(n_races=1), _scalping_config())
        mask = compute_mask(space, env)
        assert mask[0]
        assert not mask[1:].any()
