"""Tests for ``agents_v2.env_shim.DiscreteActionShim``.

Phase 1, Session 01 deliverable. The shim translates the discrete
policy outputs to the env's 70-dim Box action vector and augments
observations with the Phase 0 scorer.

These tests skip cleanly when the ``models/scorer_v1/`` artefacts
aren't on disk — same convention as
``tests/test_scorer_v1_inference.py``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from data.episode_builder import Day, Race, RunnerSnap, Tick
from env.betfair_env import (
    MAX_ARB_TICKS,
    MIN_ARB_TICKS,
    SCALPING_ACTIONS_PER_RUNNER,
    BetfairEnv,
)

from agents_v2.action_space import ActionType, DiscreteActionSpace
from tests.test_betfair_env import (
    _make_day,
    _make_runner_meta,
    _make_runner_snap,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCORER_DIR = REPO_ROOT / "models" / "scorer_v1"


def _scorer_runtime_available() -> tuple[bool, str]:
    if not (SCORER_DIR / "model.lgb").exists():
        return False, (
            f"Scorer artefacts missing under {SCORER_DIR}; "
            "run `python -m training_v2.scorer.train_and_evaluate` first."
        )
    try:
        import lightgbm  # noqa: F401
    except Exception as exc:
        return False, f"lightgbm not importable in this environment: {exc!r}"
    try:
        import joblib  # noqa: F401
    except Exception as exc:
        return False, f"joblib not importable in this environment: {exc!r}"
    return True, ""


_runtime_ok, _runtime_reason = _scorer_runtime_available()
pytestmark = pytest.mark.skipif(not _runtime_ok, reason=_runtime_reason)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _scalping_config(max_runners: int = 4) -> dict:
    return {
        "training": {
            "max_runners": max_runners,
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
    market_id = "1.20000001"
    start_time = datetime(2026, 3, 26, 14, 0, 0)
    runners = [
        _make_runner_snap(101, ltp=ltp, status=status),
        _make_runner_snap(102),
        _make_runner_snap(103),
    ]
    ticks = []
    for i in range(5):
        ticks.append(Tick(
            market_id=market_id,
            timestamp=start_time - timedelta(seconds=600 - i * 5),
            sequence_number=i,
            venue="Newmarket",
            market_start_time=start_time,
            number_of_active_runners=len(runners),
            traded_volume=10000.0,
            in_play=False,
            winner_selection_id=101,
            race_status=None,
            temperature=15.0, precipitation=0.0, wind_speed=5.0,
            wind_direction=180.0, humidity=60.0, weather_code=0,
            runners=runners,
        ))
    meta = {sid: _make_runner_meta(sid) for sid in (101, 102, 103)}
    race = Race(
        market_id=market_id, venue="Newmarket",
        market_start_time=start_time,
        winner_selection_id=101, ticks=ticks, runner_metadata=meta,
    )
    return Day(date="2026-03-26", races=[race])


@pytest.fixture
def shim():
    """Construct a shim around a fresh BetfairEnv on a synthetic day."""
    from agents_v2.env_shim import DiscreteActionShim

    env = BetfairEnv(_make_day(n_races=1), _scalping_config())
    s = DiscreteActionShim(env)
    s.reset()
    return s


@pytest.fixture
def shim_first_inactive():
    from agents_v2.env_shim import DiscreteActionShim

    day = _make_day_with_first_runner(status="REMOVED")
    env = BetfairEnv(day, _scalping_config())
    s = DiscreteActionShim(env)
    s.reset()
    return s


# ── Action translation ─────────────────────────────────────────────────────


class TestObsDim:
    def test_obs_dim_is_base_plus_2x_runners(self):
        from agents_v2.env_shim import DiscreteActionShim

        env = BetfairEnv(_make_day(n_races=1), _scalping_config())
        s = DiscreteActionShim(env)
        base = int(env.observation_space.shape[0])
        assert s.obs_dim == base + 2 * env.max_runners

    def test_reset_returns_extended_obs_of_correct_shape(self):
        from agents_v2.env_shim import DiscreteActionShim

        env = BetfairEnv(_make_day(n_races=1), _scalping_config())
        s = DiscreteActionShim(env)
        obs, _info = s.reset()
        assert obs.shape == (s.obs_dim,)
        assert obs.dtype == np.float32
        assert np.isfinite(obs).all()


class TestActionEncoding:
    def test_encode_noop_writes_all_zeros(self, shim):
        action = shim.encode_action(0)  # NOOP
        assert action.shape == (
            shim.max_runners * SCALPING_ACTIONS_PER_RUNNER,
        )
        assert np.allclose(action, 0.0)

    def test_step_with_noop_writes_zeros_to_box_action(self, shim, monkeypatch):
        """Instrument the env to capture the action vector it received."""
        captured: list[np.ndarray] = []

        def _spy(action, tick, race):
            captured.append(action.copy())

        monkeypatch.setattr(shim.env, "_process_action", _spy)
        shim.step(0)  # NOOP
        assert len(captured) == 1
        assert np.allclose(captured[0], 0.0)
        assert captured[0].shape == (
            shim.max_runners * SCALPING_ACTIONS_PER_RUNNER,
        )

    def test_step_with_open_back_writes_correct_per_runner_slot(
        self, shim, monkeypatch,
    ):
        captured: list[np.ndarray] = []
        monkeypatch.setattr(
            shim.env, "_process_action",
            lambda action, tick, race: captured.append(action.copy()),
        )
        space = shim.action_space
        slot = 1
        idx = space.encode(ActionType.OPEN_BACK, slot)
        shim.step(idx)

        action = captured[0]
        N = shim.max_runners
        # The targeted runner's slot has the expected non-zero values.
        assert action[slot] > 0.33                       # signal → BACK
        assert action[N + slot] > -1.0                   # stake encoded
        assert action[2 * N + slot] > 0.0                # aggression
        assert action[3 * N + slot] == 0.0               # cancel
        assert action[4 * N + slot] != 0.0               # arb_spread
        assert action[5 * N + slot] == 0.0               # requote
        assert action[6 * N + slot] == 0.0               # close
        # Every OTHER runner's slot is all zeros across all 7 dims.
        for other_slot in range(N):
            if other_slot == slot:
                continue
            for d in range(SCALPING_ACTIONS_PER_RUNNER):
                assert action[d * N + other_slot] == 0.0, (
                    f"non-target slot {other_slot} dim {d} non-zero "
                    f"({action[d * N + other_slot]})"
                )

    def test_step_with_open_lay_writes_negative_signal(
        self, shim, monkeypatch,
    ):
        captured: list[np.ndarray] = []
        monkeypatch.setattr(
            shim.env, "_process_action",
            lambda action, tick, race: captured.append(action.copy()),
        )
        space = shim.action_space
        slot = 0
        shim.step(space.encode(ActionType.OPEN_LAY, slot))
        action = captured[0]
        N = shim.max_runners
        assert action[slot] < -0.33  # LAY threshold

    def test_step_with_close_writes_close_signal(self, shim, monkeypatch):
        captured: list[np.ndarray] = []
        monkeypatch.setattr(
            shim.env, "_process_action",
            lambda action, tick, race: captured.append(action.copy()),
        )
        space = shim.action_space
        slot = 2
        shim.step(space.encode(ActionType.CLOSE, slot))
        action = captured[0]
        N = shim.max_runners
        # Only the close dim of the target slot is set.
        assert action[6 * N + slot] > 0.5
        for d in range(SCALPING_ACTIONS_PER_RUNNER):
            for s in range(N):
                if d == 6 and s == slot:
                    continue
                assert action[d * N + s] == 0.0

    def test_arb_spread_round_trips_to_default_ticks(self, shim):
        """Encoded arb_raw decodes back to the locked default arb_ticks=20."""
        from env.betfair_env import MAX_ARB_TICKS, MIN_ARB_TICKS

        space = shim.action_space
        action = shim.encode_action(space.encode(ActionType.OPEN_BACK, 0))
        N = shim.max_runners
        arb_raw = float(action[4 * N + 0])
        # Replicate the env's decode (with arb_spread_scale=1.0).
        frac = (arb_raw + 1.0) / 2.0
        ticks = int(round(
            MIN_ARB_TICKS + frac * (MAX_ARB_TICKS - MIN_ARB_TICKS),
        ))
        assert ticks == 20

    def test_stake_round_trips_through_env_decode(self, shim):
        """Encoded stake_raw decodes back to ~default_stake on env-side."""
        space = shim.action_space
        action = shim.encode_action(space.encode(ActionType.OPEN_BACK, 0))
        N = shim.max_runners
        stake_raw = float(action[N + 0])
        # Env: stake = ((raw+1)/2) * bm.budget
        budget = shim.env.bet_manager.budget
        stake_decoded = (stake_raw + 1.0) / 2.0 * budget
        # default_stake=10, budget=100 → exact.
        assert stake_decoded == pytest.approx(10.0, abs=1e-5)

    def test_explicit_stake_override(self, shim):
        space = shim.action_space
        action = shim.encode_action(
            space.encode(ActionType.OPEN_BACK, 0), stake=25.0,
        )
        N = shim.max_runners
        stake_raw = float(action[N + 0])
        budget = shim.env.bet_manager.budget
        stake_decoded = (stake_raw + 1.0) / 2.0 * budget
        assert stake_decoded == pytest.approx(25.0, abs=1e-5)


# ── Scorer wiring ──────────────────────────────────────────────────────────


class TestScorerWiring:
    def test_scorer_predictions_packed_at_correct_indices(
        self, shim, monkeypatch,
    ):
        """Monkeypatch booster to a fixed value; assert layout in obs."""
        FIXED_RAW = np.array([0.7], dtype=np.float64)
        # Predict returns a 1-element array per call.
        monkeypatch.setattr(
            shim._booster, "predict", lambda x, *a, **k: FIXED_RAW.copy(),
        )
        # Calibrator: identity-like for predictability.
        monkeypatch.setattr(
            shim._calibrator, "predict",
            lambda x: np.asarray(x, dtype=np.float64),
        )
        race = shim.env.day.races[shim.env._race_idx]
        tick = race.ticks[shim.env._tick_idx]
        # Build a fake base obs to pass through.
        base = np.zeros(
            int(shim.env.observation_space.shape[0]), dtype=np.float32,
        )
        extended = shim.compute_extended_obs(base)
        base_dim = base.shape[0]
        # Expect 0.7 at every (slot, side) where slot maps to an
        # ACTIVE runner with LTP > 1.
        slot_map = shim.env._slot_maps[shim.env._race_idx]
        runner_by_sid = {r.selection_id: r for r in tick.runners}
        for slot in range(shim.max_runners):
            sid = slot_map.get(slot)
            if sid is None:
                continue
            runner = runner_by_sid.get(sid)
            if runner is None or runner.status != "ACTIVE":
                continue
            ltp = runner.last_traded_price
            if ltp is None or ltp <= 1.0:
                continue
            for side_idx in range(2):
                assert extended[base_dim + 2 * slot + side_idx] == (
                    pytest.approx(0.7, abs=1e-6)
                ), f"slot {slot} side {side_idx} did not receive booster value"

    def test_scorer_returns_zero_for_inactive_runner(
        self, shim_first_inactive, monkeypatch,
    ):
        """REMOVED runner → both scorer slots stay at 0 even though booster fires."""
        s = shim_first_inactive
        monkeypatch.setattr(
            s._booster, "predict", lambda x, *a, **k: np.array([0.9]),
        )
        monkeypatch.setattr(
            s._calibrator, "predict",
            lambda x: np.asarray(x, dtype=np.float64),
        )
        base = np.zeros(
            int(s.env.observation_space.shape[0]), dtype=np.float32,
        )
        extended = s.compute_extended_obs(base)
        base_dim = base.shape[0]
        # Slot 0 is the REMOVED runner (sid 101 sorts first).
        assert extended[base_dim + 0] == 0.0
        assert extended[base_dim + 1] == 0.0
        # An ACTIVE neighbour does receive the patched value.
        assert extended[base_dim + 2] == pytest.approx(0.9, abs=1e-6)


# ── Mask plumbing ──────────────────────────────────────────────────────────


class TestActionMaskForwarding:
    def test_action_mask_blocks_open_on_inactive_runner(
        self, shim_first_inactive,
    ):
        s = shim_first_inactive
        space = s.action_space
        mask = s.get_action_mask()
        assert mask[0]  # NOOP
        assert not mask[space.encode(ActionType.OPEN_BACK, 0)]
        assert not mask[space.encode(ActionType.OPEN_LAY, 0)]
        # Slot 1 is healthy.
        assert mask[space.encode(ActionType.OPEN_BACK, 1)]


# ── Constructor guards ─────────────────────────────────────────────────────


class TestConstructorGuards:
    def test_rejects_non_scalping_env(self):
        from agents_v2.env_shim import DiscreteActionShim

        non_scalping = {
            "training": {
                "max_runners": 4, "starting_budget": 100.0,
                "max_bets_per_race": 20,
            },
            "actions": {"force_aggressive": True},
            "reward": {
                "early_pick_bonus_min": 1.2, "early_pick_bonus_max": 1.5,
                "early_pick_min_seconds": 300, "efficiency_penalty": 0.01,
            },
        }
        env = BetfairEnv(_make_day(n_races=1), non_scalping)
        with pytest.raises(ValueError, match="scalping_mode"):
            DiscreteActionShim(env)

    def test_rejects_arb_ticks_out_of_range(self):
        from agents_v2.env_shim import DiscreteActionShim

        env = BetfairEnv(_make_day(n_races=1), _scalping_config())
        with pytest.raises(ValueError):
            DiscreteActionShim(env, arb_ticks=MAX_ARB_TICKS + 5)
        with pytest.raises(ValueError):
            DiscreteActionShim(env, arb_ticks=MIN_ARB_TICKS - 1)
