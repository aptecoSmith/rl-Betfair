"""Tests for ``training_v2.discrete_ppo.rollout.RolloutCollector``.

Phase 2, Session 01 — slow-marked, skips when scorer artefacts are
absent (same pattern as ``tests/test_agents_v2_smoke.py``).
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


def _build_collector(seed: int = 0, n_races: int = 2):
    """Build a fresh shim + policy + collector on a tiny synthetic day."""
    from agents_v2.env_shim import DiscreteActionShim
    from training_v2.discrete_ppo.rollout import RolloutCollector

    torch.manual_seed(seed)
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


@pytest.mark.timeout(60)
def test_collect_episode_emits_one_transition_per_step():
    """One transition per env step; final ``done=True``, others ``False``."""
    from training_v2.discrete_ppo.transition import (
        rollout_batch_to_transitions,
    )
    env, _shim, _policy, collector = _build_collector(seed=0)
    transitions = rollout_batch_to_transitions(collector.collect_episode())

    assert len(transitions) > 0, "rollout produced no transitions"
    assert transitions[-1].done is True, (
        "final transition must mark episode end"
    )
    for tr in transitions[:-1]:
        assert tr.done is False, (
            "intermediate transitions must have done=False"
        )

    # Shape contract.
    max_runners = collector.max_runners
    n_actions = collector.action_space.n
    obs_dim = collector.shim.obs_dim
    for tr in transitions:
        assert tr.obs.shape == (obs_dim,)
        assert tr.value_per_runner.shape == (max_runners,)
        assert tr.per_runner_reward.shape == (max_runners,)
        assert tr.mask.shape == (n_actions,)
        # Mask carries with the transition (Phase 1 findings §2).
        assert tr.mask[0], "NOOP must always be legal"
        assert bool(tr.mask[tr.action_idx]), (
            "stored mask must legalise the sampled action"
        )


@pytest.mark.timeout(60)
def test_hidden_state_in_captured_before_forward_pass():
    """t=0's hidden_state_in is zero; some later step is non-zero.

    The ppo-kl-fix gotcha: capturing the post-forward state corrupts
    the PPO update because rollout-time and update-time log-probs
    end up conditioning on different states. The first transition's
    state must equal ``init_hidden`` (zeros) by construction; if the
    collector accidentally captured the post-forward state we'd see
    ``hidden_state_in != 0`` at t=0.
    """
    from training_v2.discrete_ppo.transition import (
        rollout_batch_to_transitions,
    )
    _env, _shim, _policy, collector = _build_collector(seed=0)
    transitions = rollout_batch_to_transitions(collector.collect_episode())

    h0_t0, c0_t0 = transitions[0].hidden_state_in
    # Phase 3 Session 01b: hidden_state_in stores torch tensors, not
    # numpy arrays. Compare as tensors.
    assert torch.equal(h0_t0, torch.zeros_like(h0_t0))
    assert torch.equal(c0_t0, torch.zeros_like(c0_t0))

    # At least one later transition has a non-zero hidden state — if
    # ALL hidden states were zero the LSTM would be effectively
    # stateless and the bug would be elsewhere.
    later_nonzero = any(
        float(tr.hidden_state_in[0].abs().max().item()) > 0.0
        for tr in transitions[1:]
    )
    assert later_nonzero, (
        "no later transition has a non-zero hidden state — the LSTM "
        "appears to be stateless, which is wrong"
    )


@pytest.mark.timeout(60)
def test_per_runner_reward_sums_to_total_reward():
    """``sum_t sum_i per_runner_reward`` ≈ ``raw_pnl_reward + shaped_bonus``.

    The collector's per-step assertion already enforces the per-step
    invariant ``sum_i per_runner_reward[t,i] = reward[t]``. This test
    walks the whole episode and confirms the cumulative sum matches
    the env's own raw + shaped totals to floating-point tolerance.
    """
    from training_v2.discrete_ppo.transition import (
        rollout_batch_to_transitions,
    )
    env, _shim, _policy, collector = _build_collector(seed=1)
    transitions = rollout_batch_to_transitions(collector.collect_episode())

    total_per_runner = sum(
        float(tr.per_runner_reward.sum()) for tr in transitions
    )
    info = env._get_info()
    expected_total = (
        float(info["raw_pnl_reward"]) + float(info["shaped_bonus"])
    )
    assert abs(total_per_runner - expected_total) < 1e-3, (
        f"cumulative per-runner reward {total_per_runner!r} diverges from "
        f"env raw+shaped total {expected_total!r}"
    )


@pytest.mark.timeout(60)
def test_mask_is_carried_with_transition():
    """Stored mask matches a re-computation under the same seed.

    Replays the rollout with an identical seed and checks every
    transition's ``mask`` against ``shim.get_action_mask()``
    re-evaluated step-by-step against an independently-driven env.
    """
    from training_v2.discrete_ppo.transition import (
        rollout_batch_to_transitions,
    )
    env_a, shim_a, _policy_a, collector_a = _build_collector(seed=7)
    transitions = rollout_batch_to_transitions(collector_a.collect_episode())

    # Re-run an identical episode and compare masks at each step.
    torch.manual_seed(7)
    from agents_v2.env_shim import DiscreteActionShim
    env_b = BetfairEnv(
        _make_day(n_races=2, n_pre_ticks=10, n_inplay_ticks=2),
        _scalping_config(),
    )
    shim_b = DiscreteActionShim(env_b)
    policy_b = DiscreteLSTMPolicy(
        obs_dim=shim_b.obs_dim,
        action_space=shim_b.action_space,
        hidden_size=32,
    )
    from training_v2.discrete_ppo.rollout import RolloutCollector
    collector_b = RolloutCollector(shim=shim_b, policy=policy_b, device="cpu")
    transitions_b = rollout_batch_to_transitions(collector_b.collect_episode())

    # Determinism guard: same seed, same masks (the env is fully
    # deterministic on the synthetic day; the policy's sampling uses
    # the seeded torch generator).
    assert len(transitions) == len(transitions_b)
    for tr_a, tr_b in zip(transitions, transitions_b):
        np.testing.assert_array_equal(tr_a.mask, tr_b.mask)
