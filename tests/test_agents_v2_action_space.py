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


# ── Predictor p_win action gate (plans/scalping-pwin-gate/) ────────────────


class TestPredictorPWinGate:
    """`compute_mask` honours the champion-p_win action gate.

    When `_predictor_p_win_gate_active` is True on the env, the mask
    refuses OPEN_BACK on runners with `p_win < back_threshold` and
    OPEN_LAY on runners with `p_win > lay_threshold`. Defaults of 0.0
    / 1.0 are byte-identical to pre-gate behaviour.

    Tests inject known p_win values directly into the env's cache
    so we don't depend on a real PredictorBundle for unit tests.
    """

    def _reset_env(self) -> BetfairEnv:
        env = BetfairEnv(_make_day(n_races=1), _scalping_config())
        env.reset()
        return env

    def test_gate_disabled_by_default(self):
        """Default constructor → gate inactive → mask matches pre-gate."""
        space = DiscreteActionSpace(max_runners=4)
        env = self._reset_env()
        assert env._predictor_p_win_back_threshold == 0.0
        assert env._predictor_p_win_lay_threshold == 1.0
        assert env._predictor_p_win_gate_active is False
        mask = compute_mask(space, env)
        # Slot 0 (sid 101) is healthy → both opens legal.
        assert mask[space.encode(ActionType.OPEN_BACK, 0)]
        assert mask[space.encode(ActionType.OPEN_LAY, 0)]

    def test_back_threshold_blocks_low_pwin(self):
        """back_threshold=0.4, runner p_win=0.2 → OPEN_BACK masked."""
        space = DiscreteActionSpace(max_runners=4)
        env = self._reset_env()
        # Activate the gate manually — bypasses the
        # use_race_outcome_predictor requirement (tests are
        # PredictorBundle-free).
        env._predictor_p_win_back_threshold = 0.4
        env._predictor_p_win_lay_threshold = 1.0
        env._predictor_p_win_gate_active = True
        # Slot 0 (sid 101) p_win=0.2, slot 1 (sid 102) p_win=0.6.
        env._race_p_win_by_race[env._race_idx] = {101: 0.2, 102: 0.6}

        mask = compute_mask(space, env)
        assert not mask[space.encode(ActionType.OPEN_BACK, 0)], \
            "back on low-p_win runner should be masked"
        assert mask[space.encode(ActionType.OPEN_LAY, 0)], \
            "lay still legal — lay_threshold=1.0 allows all"
        assert mask[space.encode(ActionType.OPEN_BACK, 1)], \
            "back on high-p_win runner stays legal"

    def test_lay_threshold_blocks_high_pwin(self):
        """lay_threshold=0.3, runner p_win=0.6 → OPEN_LAY masked."""
        space = DiscreteActionSpace(max_runners=4)
        env = self._reset_env()
        env._predictor_p_win_back_threshold = 0.0
        env._predictor_p_win_lay_threshold = 0.3
        env._predictor_p_win_gate_active = True
        env._race_p_win_by_race[env._race_idx] = {101: 0.2, 102: 0.6}

        mask = compute_mask(space, env)
        assert mask[space.encode(ActionType.OPEN_BACK, 0)], \
            "back stays legal — back_threshold=0.0 allows all"
        assert mask[space.encode(ActionType.OPEN_LAY, 0)], \
            "lay on p_win=0.2 ≤ 0.3 still legal"
        assert not mask[space.encode(ActionType.OPEN_LAY, 1)], \
            "lay on p_win=0.6 > 0.3 should be masked"

    def test_both_thresholds_active(self):
        """Both gates active: only mid-p_win runners are unbackable AND unlayable."""
        space = DiscreteActionSpace(max_runners=4)
        env = self._reset_env()
        env._predictor_p_win_back_threshold = 0.5  # back if p_win >= 0.5
        env._predictor_p_win_lay_threshold = 0.5   # lay if p_win <= 0.5
        env._predictor_p_win_gate_active = True
        # Three runners at low, mid, high p_win.
        env._race_p_win_by_race[env._race_idx] = {101: 0.1, 102: 0.5, 103: 0.9}

        mask = compute_mask(space, env)
        # Slot 0 (p_win=0.1): can lay (0.1 ≤ 0.5), cannot back (0.1 < 0.5).
        assert not mask[space.encode(ActionType.OPEN_BACK, 0)]
        assert mask[space.encode(ActionType.OPEN_LAY, 0)]
        # Slot 1 (p_win=0.5): can both — boundary inclusive.
        assert mask[space.encode(ActionType.OPEN_BACK, 1)]
        assert mask[space.encode(ActionType.OPEN_LAY, 1)]
        # Slot 2 (p_win=0.9): can back (0.9 ≥ 0.5), cannot lay (0.9 > 0.5).
        assert mask[space.encode(ActionType.OPEN_BACK, 2)]
        assert not mask[space.encode(ActionType.OPEN_LAY, 2)]

    def test_missing_pwin_falls_back_to_zero(self):
        """Runner not in the p_win cache → treated as p_win=0.

        Backs on that runner get masked under any back_threshold > 0;
        lays stay legal because 0 ≤ any lay_threshold ≥ 0.
        """
        space = DiscreteActionSpace(max_runners=4)
        env = self._reset_env()
        env._predictor_p_win_back_threshold = 0.1
        env._predictor_p_win_lay_threshold = 1.0
        env._predictor_p_win_gate_active = True
        # Cache only has slot 1 (sid 102); slot 0 (sid 101) is missing.
        env._race_p_win_by_race[env._race_idx] = {102: 0.6}

        mask = compute_mask(space, env)
        assert not mask[space.encode(ActionType.OPEN_BACK, 0)], \
            "missing p_win defaults to 0 < 0.1 → back masked"
        assert mask[space.encode(ActionType.OPEN_LAY, 0)], \
            "p_win=0 ≤ 1.0 → lay legal"
        assert mask[space.encode(ActionType.OPEN_BACK, 1)], \
            "slot 1 has p_win=0.6 ≥ 0.1 → back legal"

    def test_invalid_threshold_raises(self):
        """Constructor rejects threshold outside [0, 1]."""
        with pytest.raises(ValueError, match="back_threshold"):
            BetfairEnv(
                _make_day(n_races=1),
                _scalping_config(),
                predictor_p_win_back_threshold=1.5,
            )
        with pytest.raises(ValueError, match="lay_threshold"):
            BetfairEnv(
                _make_day(n_races=1),
                _scalping_config(),
                predictor_p_win_lay_threshold=-0.1,
            )

    def test_gate_byte_identical_when_disabled(self):
        """back=0.0, lay=1.0 → gate inactive → mask matches no-gate behavior bit-for-bit."""
        space = DiscreteActionSpace(max_runners=4)
        env_a = self._reset_env()
        env_b = self._reset_env()
        # env_b explicitly sets the defaults, env_a leaves them implicit.
        env_b._predictor_p_win_back_threshold = 0.0
        env_b._predictor_p_win_lay_threshold = 1.0
        # Inject p_win values that WOULD trigger a gate if it were active.
        env_b._race_p_win_by_race[env_b._race_idx] = {101: 0.05, 102: 0.95}
        mask_a = compute_mask(space, env_a)
        mask_b = compute_mask(space, env_b)
        assert (mask_a == mask_b).all(), \
            "gate-disabled env must produce identical mask regardless of p_win values"


# ─── Each-way action space (predictor-integration Session 04) ────────────────


class TestEachWayActionSpace:
    """`DiscreteActionSpace(each_way=True)` extends the discrete head
    with `OPEN_BACK_EACH_WAY` + `OPEN_LAY_EACH_WAY` per runner. Used by
    `value_each_way` strategy_mode (plans/predictor-integration/
    session_prompts/04_each_way_action_surface.md).

    Default `each_way=False` keeps the action space byte-identical to
    pre-plan — existing call sites unchanged.
    """

    def test_default_each_way_false_keeps_n_unchanged(self):
        n_legacy = DiscreteActionSpace(max_runners=4).n
        n_explicit_off = DiscreteActionSpace(max_runners=4, each_way=False).n
        assert n_legacy == n_explicit_off == 1 + 3 * 4

    def test_each_way_true_extends_n(self):
        space = DiscreteActionSpace(max_runners=4, each_way=True)
        # 1 + 5 * max_runners (NOOP + OPEN_BACK + OPEN_LAY + CLOSE +
        # OPEN_BACK_EW + OPEN_LAY_EW per runner)
        assert space.n == 1 + 5 * 4
        assert space.each_way is True

    def test_encode_each_way_back(self):
        space = DiscreteActionSpace(max_runners=3, each_way=True)
        # CLOSE block ends at index 1 + 3 * 3 = 10. OPEN_BACK_EACH_WAY
        # starts there.
        assert space.encode(ActionType.OPEN_BACK_EACH_WAY, 0) == 10
        assert space.encode(ActionType.OPEN_BACK_EACH_WAY, 2) == 12

    def test_encode_each_way_lay(self):
        space = DiscreteActionSpace(max_runners=3, each_way=True)
        # OPEN_LAY_EACH_WAY follows immediately after the BACK_EW block.
        assert space.encode(ActionType.OPEN_LAY_EACH_WAY, 0) == 13
        assert space.encode(ActionType.OPEN_LAY_EACH_WAY, 2) == 15

    def test_decode_each_way_back(self):
        space = DiscreteActionSpace(max_runners=3, each_way=True)
        kind, slot = space.decode(10)
        assert kind is ActionType.OPEN_BACK_EACH_WAY
        assert slot == 0
        kind, slot = space.decode(12)
        assert kind is ActionType.OPEN_BACK_EACH_WAY
        assert slot == 2

    def test_decode_each_way_lay(self):
        space = DiscreteActionSpace(max_runners=3, each_way=True)
        kind, slot = space.decode(13)
        assert kind is ActionType.OPEN_LAY_EACH_WAY
        assert slot == 0
        kind, slot = space.decode(15)
        assert kind is ActionType.OPEN_LAY_EACH_WAY
        assert slot == 2

    def test_round_trip_all_each_way_actions(self):
        space = DiscreteActionSpace(max_runners=5, each_way=True)
        for kind in (
            ActionType.NOOP,
            ActionType.OPEN_BACK,
            ActionType.OPEN_LAY,
            ActionType.CLOSE,
            ActionType.OPEN_BACK_EACH_WAY,
            ActionType.OPEN_LAY_EACH_WAY,
        ):
            slots = [None] if kind is ActionType.NOOP else range(5)
            for slot in slots:
                idx = space.encode(kind, slot)
                k2, s2 = space.decode(idx)
                assert k2 is kind
                assert s2 == slot

    def test_encode_each_way_when_disabled_raises(self):
        """Hard_constraints §10: silent fallback forbidden. Encoding an
        each-way type against a non-EW space must raise loudly."""
        space = DiscreteActionSpace(max_runners=4, each_way=False)
        with pytest.raises(ValueError, match="each_way=False"):
            space.encode(ActionType.OPEN_BACK_EACH_WAY, 0)
        with pytest.raises(ValueError, match="each_way=False"):
            space.encode(ActionType.OPEN_LAY_EACH_WAY, 0)


# ── Direction-predictor action gate (plans/scalping-direction-gate/) ────────


class TestDirectionGate:
    """`compute_mask` honours the asymmetric direction-predictor gate.

    When `_direction_gate_active` is True on the env, the mask refuses
    OPEN_LAY on `(tick, sid)` pairs where `dir_fire_drift` did NOT fire.
    OPEN_BACK is NEVER direction-gated (the shorten signal is broken,
    per the 2026-05-12 audit). Default-off mode is byte-identical to
    the pwin-only path.

    Tests inject drift-fire booleans directly into the env's cache so
    we don't depend on a real PredictorBundle for unit tests.
    """

    def _reset_env(self) -> BetfairEnv:
        env = BetfairEnv(_make_day(n_races=1), _scalping_config())
        env.reset()
        return env

    def test_direction_gate_disabled_by_default(self):
        """Default constructor → gate inactive → mask matches pre-plan."""
        space = DiscreteActionSpace(max_runners=4)
        env = self._reset_env()
        assert env._direction_gate_enabled is False
        assert env._direction_gate_active is False
        mask = compute_mask(space, env)
        # Slot 0 (sid 101) is healthy → both opens legal regardless of
        # what's in the (empty) drift cache.
        assert mask[space.encode(ActionType.OPEN_BACK, 0)]
        assert mask[space.encode(ActionType.OPEN_LAY, 0)]

    def test_direction_gate_refuses_lay_when_drift_not_firing(self):
        """Gate active, drift cache empty → OPEN_LAY masked on every slot."""
        space = DiscreteActionSpace(max_runners=4)
        env = self._reset_env()
        env._direction_gate_active = True
        # No fires at all.
        env._tick_drift_fires_by_race[env._race_idx] = {}

        mask = compute_mask(space, env)
        # Active runners (slots 0, 1) — OPEN_LAY refused, OPEN_BACK legal.
        assert not mask[space.encode(ActionType.OPEN_LAY, 0)], \
            "drift not firing → OPEN_LAY must be masked"
        assert not mask[space.encode(ActionType.OPEN_LAY, 1)], \
            "drift not firing → OPEN_LAY must be masked"
        assert mask[space.encode(ActionType.OPEN_BACK, 0)], \
            "OPEN_BACK is NOT direction-gated"
        assert mask[space.encode(ActionType.OPEN_BACK, 1)], \
            "OPEN_BACK is NOT direction-gated"

    def test_direction_gate_allows_lay_when_drift_firing(self):
        """Drift firing on (tick, sid) → OPEN_LAY remains legal there."""
        space = DiscreteActionSpace(max_runners=4)
        env = self._reset_env()
        env._direction_gate_active = True
        # Drift fires on both runners at tick 0.
        env._tick_drift_fires_by_race[env._race_idx] = {
            (env._tick_idx, 101): True,
            (env._tick_idx, 102): True,
        }

        mask = compute_mask(space, env)
        assert mask[space.encode(ActionType.OPEN_LAY, 0)], \
            "drift firing on slot 0 → OPEN_LAY legal"
        assert mask[space.encode(ActionType.OPEN_LAY, 1)], \
            "drift firing on slot 1 → OPEN_LAY legal"

    def test_direction_gate_does_not_touch_back(self):
        """Asymmetry guard: OPEN_BACK legal even when drift absent everywhere."""
        space = DiscreteActionSpace(max_runners=4)
        env = self._reset_env()
        env._direction_gate_active = True
        env._tick_drift_fires_by_race[env._race_idx] = {}

        mask = compute_mask(space, env)
        # Slots 0 and 1 are the active runners in the fixture day.
        assert mask[space.encode(ActionType.OPEN_BACK, 0)]
        assert mask[space.encode(ActionType.OPEN_BACK, 1)]

    def test_direction_gate_byte_identical_when_disabled(self):
        """Gate off → mask matches reference even with a populated cache."""
        space = DiscreteActionSpace(max_runners=4)
        env_a = self._reset_env()
        env_b = self._reset_env()
        # env_b leaves the gate off but stuffs the cache with drift
        # bools that WOULD trigger refusal if the gate were active.
        assert env_b._direction_gate_active is False
        env_b._tick_drift_fires_by_race[env_b._race_idx] = {
            (env_b._tick_idx, 101): False,
            (env_b._tick_idx, 102): False,
        }
        mask_a = compute_mask(space, env_a)
        mask_b = compute_mask(space, env_b)
        assert (mask_a == mask_b).all(), \
            "gate-disabled env must produce identical mask regardless of cache"

    def test_direction_gate_raises_without_use_direction_predictor(self):
        """Loud-fail on incompatible flags (hard_constraints §2)."""
        with pytest.raises(ValueError, match="use_direction_predictor"):
            BetfairEnv(
                _make_day(n_races=1),
                _scalping_config(),
                direction_gate_enabled=True,
                # use_direction_predictor stays None → resolves False
                # without a predictor_bundle.
            )


# ── Race-confidence action gate (plans/scalping-race-confidence-gate/) ──────


class TestRaceConfidenceGate:
    """`compute_mask` honours the per-race confidence gate.

    When ``_race_confidence_gate_active`` is True on the env, the mask
    refuses ALL non-NOOP actions on races where
    ``max(champion p_win) < race_confidence_threshold``. Composes
    additively with the per-runner pwin gate. Default 0.0 is
    byte-identical to pre-plan behaviour.

    Tests inject ``_race_is_confident_by_race`` directly to avoid the
    need for a real PredictorBundle.
    """

    def _reset_env(self) -> BetfairEnv:
        env = BetfairEnv(_make_day(n_races=1), _scalping_config())
        env.reset()
        return env

    def test_gate_disabled_by_default(self):
        """Default constructor → gate inactive → mask matches pre-plan."""
        space = DiscreteActionSpace(max_runners=4)
        env = self._reset_env()
        assert env._race_confidence_threshold == 0.0
        assert env._race_confidence_gate_active is False
        mask = compute_mask(space, env)
        assert mask[space.encode(ActionType.OPEN_BACK, 0)]
        assert mask[space.encode(ActionType.OPEN_LAY, 0)]

    def test_confident_race_passes_through_unchanged(self):
        """Gate active, race confident → OPEN_BACK / OPEN_LAY stay legal."""
        space = DiscreteActionSpace(max_runners=4)
        env = self._reset_env()
        env._race_confidence_threshold = 0.3
        env._race_confidence_gate_active = True
        env._race_is_confident_by_race = [True]
        mask = compute_mask(space, env)
        assert mask[space.encode(ActionType.OPEN_BACK, 0)], \
            "confident race → OPEN_BACK legal on active slot"
        assert mask[space.encode(ActionType.OPEN_LAY, 0)], \
            "confident race → OPEN_LAY legal on active slot"

    def test_non_confident_race_masks_all_opens_and_closes(self):
        """Gate active, race not confident → only NOOP legal everywhere."""
        space = DiscreteActionSpace(max_runners=4)
        env = self._reset_env()
        env._race_confidence_threshold = 0.5
        env._race_confidence_gate_active = True
        env._race_is_confident_by_race = [False]
        mask = compute_mask(space, env)
        # NOOP is the ONLY legal action.
        assert mask[0] is True or mask[0] == True  # noqa: E712
        # Every non-NOOP slot for every action type is masked.
        for slot in range(space.max_runners):
            assert not mask[space.encode(ActionType.OPEN_BACK, slot)], \
                f"non-confident race must mask OPEN_BACK slot {slot}"
            assert not mask[space.encode(ActionType.OPEN_LAY, slot)], \
                f"non-confident race must mask OPEN_LAY slot {slot}"
            assert not mask[space.encode(ActionType.CLOSE, slot)], \
                f"non-confident race must mask CLOSE slot {slot}"

    def test_byte_identical_when_disabled(self):
        """Gate off → mask matches reference even with a populated cache."""
        space = DiscreteActionSpace(max_runners=4)
        env_a = self._reset_env()
        env_b = self._reset_env()
        assert env_b._race_confidence_gate_active is False
        # Stuff the cache with False — would trigger refusal if active.
        env_b._race_is_confident_by_race = [False]
        mask_a = compute_mask(space, env_a)
        mask_b = compute_mask(space, env_b)
        assert (mask_a == mask_b).all(), \
            "gate-disabled env must produce identical mask regardless of cache"

    def test_raises_without_use_race_outcome_predictor(self):
        """Loud-fail on incompatible flags (hard_constraints §2)."""
        with pytest.raises(ValueError, match="use_race_outcome_predictor"):
            BetfairEnv(
                _make_day(n_races=1),
                _scalping_config(),
                race_confidence_threshold=0.3,
                # use_race_outcome_predictor stays None → resolves False
                # without a predictor_bundle.
            )

    def test_invalid_threshold_raises(self):
        """Constructor rejects threshold outside [0, 1]."""
        with pytest.raises(ValueError, match="race_confidence_threshold"):
            BetfairEnv(
                _make_day(n_races=1),
                _scalping_config(),
                race_confidence_threshold=1.5,
            )
        with pytest.raises(ValueError, match="race_confidence_threshold"):
            BetfairEnv(
                _make_day(n_races=1),
                _scalping_config(),
                race_confidence_threshold=-0.1,
            )

    def test_composes_with_pwin_gate(self):
        """Race-confidence gate masks BEFORE the per-runner pwin gate runs.

        Non-confident race → OPEN_LAY masked on every slot regardless of
        pwin. Confident race + p_win > lay_threshold → OPEN_LAY masked by
        the pwin gate (gate-confidence didn't make it newly legal).
        """
        space = DiscreteActionSpace(max_runners=4)
        # Branch A: race not confident → all opens masked.
        env_a = self._reset_env()
        env_a._race_confidence_threshold = 0.3
        env_a._race_confidence_gate_active = True
        env_a._race_is_confident_by_race = [False]
        env_a._predictor_p_win_back_threshold = 0.0
        env_a._predictor_p_win_lay_threshold = 1.0
        env_a._predictor_p_win_gate_active = True
        env_a._race_p_win_by_race[env_a._race_idx] = {101: 0.2, 102: 0.6}
        mask_a = compute_mask(space, env_a)
        assert not mask_a[space.encode(ActionType.OPEN_LAY, 0)], \
            "non-confident race masks OPEN_LAY regardless of pwin"
        assert not mask_a[space.encode(ActionType.OPEN_LAY, 1)]

        # Branch B: race confident, pwin lay gate trips on high-pwin runner.
        env_b = self._reset_env()
        env_b._race_confidence_threshold = 0.3
        env_b._race_confidence_gate_active = True
        env_b._race_is_confident_by_race = [True]
        env_b._predictor_p_win_back_threshold = 0.0
        env_b._predictor_p_win_lay_threshold = 0.3
        env_b._predictor_p_win_gate_active = True
        env_b._race_p_win_by_race[env_b._race_idx] = {101: 0.2, 102: 0.6}
        mask_b = compute_mask(space, env_b)
        # Slot 0 (p_win=0.2 ≤ 0.3): lay legal (race confident, pwin OK).
        assert mask_b[space.encode(ActionType.OPEN_LAY, 0)]
        # Slot 1 (p_win=0.6 > 0.3): masked by pwin gate, NOT race-confidence.
        assert not mask_b[space.encode(ActionType.OPEN_LAY, 1)]
