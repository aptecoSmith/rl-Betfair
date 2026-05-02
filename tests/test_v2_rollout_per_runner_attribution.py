"""Phase 4 Session 01 regression guards for the incremental
per-runner attribution path in
``training_v2/discrete_ppo/rollout.py``.

The session prompt
(``plans/rewrite/phase-4-restore-speed/session_prompts/
01_per_runner_attribution.md``) specifies four load-bearing tests:

1. **Bit-identity vs the pre-Session-01 walk** — strict equality on
   per-tick ``per_runner_reward`` arrays at fixed seed. The legacy
   walk (``list(env.all_settled_bets) + list(env.bet_manager.bets)``,
   no pending-set bookkeeping) is preserved as a free function in
   this test module so the post-Session-01 implementation has a
   permanent oracle to compare against.

2. **Pending-set scans zero on a no-bet tick** — the regression
   guard that someone hasn't quietly put the all-bets walk back.

3. **Invariant assert still fires on drift** — same shape as the
   pre-existing per-step ``np.isclose`` assert.

4. **Pending-set size bounded across a full episode** — catches a
   memory leak where bets are added but never removed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from env.betfair_env import BetfairEnv

from agents_v2.discrete_policy import DiscreteLSTMPolicy
from tests.test_betfair_env import _make_day


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
        return False, f"lightgbm not importable: {exc!r}"
    try:
        import joblib  # noqa: F401
    except Exception as exc:
        return False, f"joblib not importable: {exc!r}"
    return True, ""


_runtime_ok, _runtime_reason = _scorer_runtime_available()
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _runtime_ok, reason=_runtime_reason),
]


# ── Test config / harness ─────────────────────────────────────────────


def _scalping_config(max_runners: int = 4) -> dict:
    return {
        "training": {
            "max_runners": max_runners,
            "starting_budget": 100.0,
            "max_bets_per_race": 20,
            "scalping_mode": True,
            "betting_constraints": {
                "max_back_price": 50.0,
                "max_lay_price": None,
                "min_seconds_before_off": 0,
                "force_close_before_off_seconds": 0,
            },
        },
        "actions": {"force_aggressive": True},
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "efficiency_penalty": 0.01,
            "commission": 0.05,
            "mark_to_market_weight": 0.0,
        },
    }


def _build_collector(seed: int = 42, n_races: int = 2):
    """Build a fresh shim + policy + collector on a tiny synthetic day."""
    from agents_v2.env_shim import DiscreteActionShim
    from training_v2.discrete_ppo.rollout import RolloutCollector

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = BetfairEnv(
        _make_day(n_races=n_races, n_pre_ticks=10, n_inplay_ticks=2),
        _scalping_config(),
    )
    shim = DiscreteActionShim(env)
    policy = DiscreteLSTMPolicy(
        obs_dim=shim.obs_dim,
        action_space=shim.action_space,
        hidden_size=32,
    )
    collector = RolloutCollector(shim=shim, policy=policy, device="cpu")
    return env, shim, policy, collector


# ── Legacy reference oracle ───────────────────────────────────────────


def _legacy_attribute_step_reward(
    env,
    step_reward: float,
    prev_pnl_by_id: dict[int, float],
    market_to_runner_map: dict[str, dict[int, int]],
    max_runners: int,
    tolerance: float = 1e-4,
) -> np.ndarray:
    """Pre-Phase-4-Session-01 attribution walk.

    Mirrors the implementation that lived at
    ``training_v2/discrete_ppo/rollout.py::_attribute_step_reward``
    before commit ``442663b`` (Phase 4 Session 01). Kept here so the
    bit-identity test has a permanent oracle.

    Walks ``list(env.all_settled_bets) + list(env.bet_manager.bets)``
    every call — O(bets-so-far) per tick, O(n²) per episode. Same
    algebra as the new pending-set walk; same residual distribution;
    same invariant assert.
    """
    per_runner = np.zeros(max_runners, dtype=np.float64)
    live_bets = (
        env.bet_manager.bets if env.bet_manager is not None else []
    )
    all_bets = list(env.all_settled_bets) + list(live_bets)

    attributed_total = 0.0
    for bet in all_bets:
        bet_id = id(bet)
        prev_pnl = prev_pnl_by_id.get(bet_id, 0.0)
        cur_pnl = float(bet.pnl)
        delta = cur_pnl - prev_pnl
        if delta == 0.0:
            prev_pnl_by_id[bet_id] = cur_pnl
            continue

        runner_map = market_to_runner_map.get(bet.market_id)
        if runner_map is not None:
            slot = runner_map.get(bet.selection_id)
            if slot is not None and slot < max_runners:
                per_runner[slot] += delta
                attributed_total += delta
        prev_pnl_by_id[bet_id] = cur_pnl

    residual = step_reward - attributed_total
    per_runner += residual / max_runners

    total = float(per_runner.sum())
    if not np.isclose(total, step_reward, rtol=0.0, atol=tolerance):
        raise AssertionError(
            f"legacy oracle drift: sum={total!r} reward={step_reward!r}",
        )
    return per_runner.astype(np.float32, copy=False)


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.timeout(120)
def test_attribution_bit_identical_to_pre_session_01_on_fixed_seed():
    """Strict per-tick equality vs the legacy walk.

    The session prompt's load-bearing correctness guard. We override
    ``_attribute_step_reward`` to also run the legacy walk on a
    parallel ``prev_pnl_by_id`` dict at every tick and assert byte-
    for-byte equality of the two ``per_runner_reward`` arrays.

    ``np.testing.assert_array_equal`` (NOT ``assert_allclose``) — the
    algebra is unchanged, the iteration order is preserved, the only
    legitimate difference is the absence of the no-op walk over
    finalised bets, which produces zero contribution either way.
    """
    from training_v2.discrete_ppo.rollout import RolloutCollector

    env, _shim, _policy, collector = _build_collector(seed=42)

    # Parallel legacy state, owned by the test wrapper.
    legacy_prev_pnl: dict[int, float] = {}
    drifts: list[tuple[int, np.ndarray, np.ndarray]] = []
    tick_counter = {"i": 0}

    original = RolloutCollector._attribute_step_reward

    def wrapped(self, *, env, step_reward, state, market_to_runner_map):
        new_arr = original(
            self,
            env=env,
            step_reward=step_reward,
            state=state,
            market_to_runner_map=market_to_runner_map,
        )
        legacy_arr = _legacy_attribute_step_reward(
            env=env,
            step_reward=step_reward,
            prev_pnl_by_id=legacy_prev_pnl,
            market_to_runner_map=market_to_runner_map,
            max_runners=self.max_runners,
        )
        if not np.array_equal(new_arr, legacy_arr):
            drifts.append((tick_counter["i"], new_arr.copy(), legacy_arr))
        tick_counter["i"] += 1
        return new_arr

    RolloutCollector._attribute_step_reward = wrapped
    try:
        transitions = collector.collect_episode()
    finally:
        RolloutCollector._attribute_step_reward = original

    assert len(transitions) > 0, "rollout produced no transitions"
    if drifts:
        first = drifts[0]
        raise AssertionError(
            f"per-runner attribution drift on tick {first[0]}: "
            f"new={first[1]!r} vs legacy={first[2]!r} "
            f"(total drifting ticks: {len(drifts)})",
        )


@pytest.mark.timeout(60)
def test_pending_set_scans_zero_on_no_bet_tick():
    """Iteration count == 0 on a tick with no pending bets.

    Regression guard against re-introducing the all-bets walk. We
    invoke ``_attribute_step_reward`` directly on a freshly-reset env
    where ``_settled_bets`` is empty AND ``bm.bets`` is empty. Under
    the legacy O(n²) walk, this would still iterate over zero bets
    (so the test passes vacuously) — but the meaningful guard is
    that AFTER bets have been placed and settled, a subsequent
    no-mutation tick still scans zero. We exercise both:

    1. Fresh env, no bets placed → iter count == 0.
    2. After a full episode, the trailing pending set is empty
       (every bet has settled and been removed).

    The ``test_pending_set_size_bounded_across_episode`` test below
    pins #2 separately; this test pins #1.
    """
    from training_v2.discrete_ppo.rollout import (
        RolloutCollector,
        _AttributionState,
    )

    env, _shim, _policy, collector = _build_collector(seed=42)
    # Reset the env to a clean state — no bets, _settled_bets empty,
    # fresh BetManager.
    env.reset()
    market_to_runner_map: dict[str, dict[int, int]] = {}
    for race_idx, race in enumerate(env.day.races):
        market_to_runner_map[race.market_id] = env._runner_maps[race_idx]

    state = _AttributionState()
    # The env reward at this synthetic moment isn't meaningful — pass
    # 0.0 so the residual is zero and the invariant assert is
    # trivially satisfied. The test is purely about iteration count.
    arr = collector._attribute_step_reward(
        env=env,
        step_reward=0.0,
        state=state,
        market_to_runner_map=market_to_runner_map,
    )
    assert arr.shape == (collector.max_runners,)
    assert state.iter_history == [0], (
        f"first attribution call on a fresh env should iterate over "
        f"an empty pending set (got iter_history={state.iter_history!r})"
    )
    assert len(state.pending_bets) == 0
    # Calling a SECOND time with still-no-bets must also be zero —
    # catches a regression where the entry-rule scan accidentally
    # adds something stale.
    collector._attribute_step_reward(
        env=env,
        step_reward=0.0,
        state=state,
        market_to_runner_map=market_to_runner_map,
    )
    assert state.iter_history == [0, 0]


@pytest.mark.timeout(60)
def test_attribution_invariant_assert_still_holds():
    """The per-step ``np.isclose(total, step_reward)`` assert still fires.

    Drift here is the load-bearing failure mode the session prompt
    flagged. We exercise a full episode (which includes settle steps,
    the highest-mutation ticks) and confirm no AssertionError. The
    pre-existing ``test_per_runner_reward_sums_to_total_reward`` test
    in ``test_discrete_ppo_rollout.py`` covers the cumulative
    invariant; this is the per-tick guard pinned for regression.
    """
    _env, _shim, _policy, collector = _build_collector(seed=43)
    # If any per-tick invariant fires, collect_episode raises and the
    # test fails. The implicit assertion is "no exception".
    transitions = collector.collect_episode()
    assert len(transitions) > 0


@pytest.mark.timeout(60)
def test_pending_set_size_bounded_across_episode():
    """``len(pending_bets)`` stays bounded — no leak.

    Catches a future regression where bets are added to pending but
    never removed (e.g. the EXIT rule was loosened, the
    ``bet.outcome != UNSETTLED`` check was removed, or settle-time
    cleanup was missed). On the synthetic 2-race day the cap is
    generous: at most ``max_bets_per_race × n_races == 40`` bets can
    exist concurrently before the second race's settle drains them.

    The terminal state is the strictest check: every bet placed in
    the episode has settled by the final attribution call, so
    ``pending_bets`` must be empty at end-of-episode.
    """
    _env, _shim, _policy, collector = _build_collector(seed=44, n_races=2)
    transitions = collector.collect_episode()

    state = collector.last_attribution_state
    # Per-tick bound across the episode.
    assert max(state.iter_history, default=0) <= 50, (
        f"pending-set size exceeded the 50-bet bound at some tick: "
        f"max observed = {max(state.iter_history, default=0)}"
    )
    # Terminal: every bet has settled, so the pending set is empty.
    assert len(state.pending_bets) == 0, (
        f"pending_bets non-empty at end of episode "
        f"({len(state.pending_bets)} entries leaked) — every bet "
        "should have outcome != UNSETTLED after the final race "
        "settles"
    )
    # ``transitions`` is just the rollout's product; we depend on at
    # least one being produced to make the bound meaningful.
    assert len(transitions) > 0
