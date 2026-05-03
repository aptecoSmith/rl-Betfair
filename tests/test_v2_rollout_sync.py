"""Throughput-fix Session 01 — rollout sync removal regression guards.

Three tests:

1. ``test_cuda_self_parity_after_sync_removal`` (gpu+slow) — two
   CUDA runs at the same seed produce bit-identical
   ``total_reward`` and ``value_loss_mean``. The deferred end-of-
   episode CPU transfer must NOT introduce any numerical drift.
   Same shape as ``tests/test_v2_gpu_parity.py::test_cuda_self_
   parity_5_episodes``, scoped to a 1-episode run for speed.

2. ``test_action_idx_and_stake_unit_still_materialise_per_tick``
   — the two STRUCTURAL ``.item()`` calls (action_idx,
   stake_unit) must remain. The CPU env consumes ``int`` and
   ``float`` every tick; a refactor that batches the env step
   would break the env contract. Spies on ``env.step`` and
   asserts the action / stake types it receives.

3. ``test_transition_log_probs_byte_identical_across_two_cpu_runs``
   — the CPU code path is unchanged in this session. Two CPU
   runs at the same seed must produce bit-identical
   log_prob_action / log_prob_stake / value_per_runner values
   across all transitions. If the deferred-batched-cpu transfer
   accidentally re-orders the per-tick values, this catches it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from env.betfair_env import BetfairEnv

from agents_v2.discrete_policy import DiscreteLSTMPolicy
from tests.test_betfair_env import _make_day


REPO_ROOT = Path(__file__).resolve().parents[1]
SCORER_DIR = REPO_ROOT / "models" / "scorer_v1"
DATA_DIR = REPO_ROOT / "data" / "processed"


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


# ── Test 1: CUDA self-parity ───────────────────────────────────────────────


_data_ok = (DATA_DIR / "2026-04-23.parquet").exists()


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not _runtime_ok, reason=_runtime_reason)
@pytest.mark.skipif(
    not _data_ok,
    reason=f"data/processed/2026-04-23.parquet not present under {DATA_DIR}",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.timeout(1800)
def test_cuda_self_parity_after_sync_removal(
    tmp_path: Path,
) -> None:
    """Two CUDA runs at the same seed produce bit-identical results.

    1-episode scoped (vs the 5-episode gpu_parity test) so this
    runs in ~3-5 min instead of ~25-30 min. The end-of-episode
    batched ``.cpu()`` transfer must not introduce any numerical
    drift relative to the per-tick ``.item()`` it replaced — the
    values are the same tensor contents in both shapes; only the
    transfer batching changes.
    """
    from training_v2.discrete_ppo.train import main

    cuda_a_out = tmp_path / "cuda_a.jsonl"
    cuda_b_out = tmp_path / "cuda_b.jsonl"

    rc_a = main(
        day_str="2026-04-23",
        data_dir=DATA_DIR,
        n_episodes=1,
        seed=42,
        out_path=cuda_a_out,
        scorer_dir=SCORER_DIR,
        device="cuda",
    )
    assert rc_a == 0
    rc_b = main(
        day_str="2026-04-23",
        data_dir=DATA_DIR,
        n_episodes=1,
        seed=42,
        out_path=cuda_b_out,
        scorer_dir=SCORER_DIR,
        device="cuda",
    )
    assert rc_b == 0

    rows_a = [
        json.loads(line)
        for line in cuda_a_out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows_b = [
        json.loads(line)
        for line in cuda_b_out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows_a) == len(rows_b) == 1

    for ra, rb in zip(rows_a, rows_b):
        assert abs(ra["total_reward"] - rb["total_reward"]) < 1e-7, (
            f"CUDA↔CUDA total_reward not bit-identical: "
            f"a={ra['total_reward']!r} b={rb['total_reward']!r} "
            f"diff={ra['total_reward'] - rb['total_reward']!r}"
        )
        assert abs(ra["value_loss_mean"] - rb["value_loss_mean"]) < 1e-7, (
            f"CUDA↔CUDA value_loss_mean not bit-identical: "
            f"a={ra['value_loss_mean']!r} b={rb['value_loss_mean']!r} "
            f"diff={ra['value_loss_mean'] - rb['value_loss_mean']!r}"
        )


# ── Test 2: structural .item() calls remain ────────────────────────────────


@pytest.mark.slow
@pytest.mark.skipif(not _runtime_ok, reason=_runtime_reason)
@pytest.mark.timeout(60)
def test_action_idx_and_stake_unit_still_materialise_per_tick():
    """``int(action.item())`` and ``float(stake_unit.item())`` are
    structurally required — the CPU env's ``step`` consumes them
    on every tick. A regression where a refactor accidentally
    deferred these too would either error loudly (env rejects
    a tensor) or — worse — coerce silently into broken behaviour.

    Wrap ``shim.step`` (the rollout's interface to the env) with a
    type assertion that re-dispatches.  Asserts both the action_idx
    arg is ``int`` and the stake kwarg is ``float`` on every call —
    these are the values the rollout extracted via the two
    structural ``.item()`` calls.
    """
    _env, shim, _policy, collector = _build_collector(seed=0)

    real_step = shim.step
    type_witness = {"action_calls": 0, "stake_calls": 0}

    def asserting_step(action_idx, *args, **kwargs):
        # action_idx must be a built-in Python int (the
        # env/BetManager indexing into action-space lookups depends
        # on this; a tensor or numpy scalar would either error or
        # silently wrong-key).
        assert type(action_idx) is int, (
            f"shim.step received action_idx of exact type "
            f"{type(action_idx)!r} on call "
            f"#{type_witness['action_calls']}; expected exact ``int``. "
            f"A torch.Tensor / np.ndarray would mean the structural "
            f".item() call was accidentally deferred."
        )
        type_witness["action_calls"] += 1
        if "stake" in kwargs:
            stake = kwargs["stake"]
            # stake_pounds = max(stake_unit * budget, MIN_BET_STAKE);
            # depending on env.bet_manager.budget's type this can be
            # either ``float`` or ``np.float64``. Both are CPU scalars
            # the env accepts. The regression we're guarding against
            # is a torch.Tensor or numpy.ndarray landing here, which
            # would mean stake_unit_t.item() got deferred.
            assert not isinstance(stake, torch.Tensor), (
                f"shim.step received stake of type {type(stake)!r} "
                f"on call #{type_witness['stake_calls']}; tensor "
                f"would mean stake_unit_t.item() was deferred"
            )
            assert not isinstance(stake, np.ndarray), (
                f"shim.step received stake of type {type(stake)!r} "
                f"on call #{type_witness['stake_calls']}; ndarray "
                f"is not a scalar"
            )
            # Numpy scalars expose __float__; built-in float is fine.
            assert isinstance(stake, (float, np.floating)), (
                f"shim.step received stake of type {type(stake)!r} "
                f"on call #{type_witness['stake_calls']}; "
                f"expected float / np.floating"
            )
            type_witness["stake_calls"] += 1
        return real_step(action_idx, *args, **kwargs)

    from training_v2.discrete_ppo.transition import (
        rollout_batch_to_transitions,
    )
    shim.step = asserting_step  # type: ignore[method-assign]
    transitions = rollout_batch_to_transitions(
        collector.collect_episode()
    )

    assert type_witness["action_calls"] == len(transitions)
    assert type_witness["stake_calls"] == len(transitions)
    assert len(transitions) > 0


# ── Test 3: CPU code path is byte-identical across two runs ────────────────


@pytest.mark.slow
@pytest.mark.skipif(not _runtime_ok, reason=_runtime_reason)
@pytest.mark.timeout(120)
def test_transition_log_probs_byte_identical_across_two_cpu_runs():
    """CPU runs at the same seed produce bit-identical log probs and
    per-runner values across all transitions.

    The CPU code path is unchanged in this session — only the
    timing/order of CPU materialisation moves from per-tick to
    end-of-episode batched. The numerical values must therefore be
    bit-identical across two identically-seeded runs.

    If the end-of-episode ``torch.stack`` + ``.cpu()`` accidentally
    re-orders or duplicates per-tick values, this catches it: the
    two runs would either differ across runs (re-ordering bug)
    or differ from the bookkeeping fields that ARE kept per-tick
    (action_idx, stake_unit, mask).

    Strict equality (``==``), not ``np.isclose`` — CPU is fully
    deterministic with seeded RNG.
    """
    from training_v2.discrete_ppo.transition import (
        rollout_batch_to_transitions,
    )
    _env_a, _shim_a, _policy_a, collector_a = _build_collector(seed=7)
    transitions_a = rollout_batch_to_transitions(
        collector_a.collect_episode()
    )
    _env_b, _shim_b, _policy_b, collector_b = _build_collector(seed=7)
    transitions_b = rollout_batch_to_transitions(
        collector_b.collect_episode()
    )

    assert len(transitions_a) == len(transitions_b)
    assert len(transitions_a) > 0

    for i, (ta, tb) in enumerate(zip(transitions_a, transitions_b)):
        assert ta.action_idx == tb.action_idx, f"tick {i}: action drift"
        assert ta.stake_unit == tb.stake_unit, f"tick {i}: stake drift"
        assert ta.log_prob_action == tb.log_prob_action, (
            f"tick {i}: log_prob_action drift "
            f"a={ta.log_prob_action!r} b={tb.log_prob_action!r}"
        )
        assert ta.log_prob_stake == tb.log_prob_stake, (
            f"tick {i}: log_prob_stake drift "
            f"a={ta.log_prob_stake!r} b={tb.log_prob_stake!r}"
        )
        assert np.array_equal(ta.value_per_runner, tb.value_per_runner), (
            f"tick {i}: value_per_runner drift "
            f"a={ta.value_per_runner!r} b={tb.value_per_runner!r}"
        )
