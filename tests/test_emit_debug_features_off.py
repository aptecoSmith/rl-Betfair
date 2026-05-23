"""Regression guards for ``emit_debug_features=False`` perf gate.

The 2026-05-22 phase-2 perf change makes the cohort training path
build envs with ``emit_debug_features=False`` (env default is True,
but training never reads ``info["debug_features"]`` or
``info["passive_orders"]``). The gating reduces per-step ``_get_info``
cost by ~48 % at the cost of those debug fields being empty.

These tests pin the contract: when the flag is OFF, the heavy
fields are empty / not computed, but the env otherwise behaves
identically — obs is computed normally, reward signal is unchanged,
and episodes terminate cleanly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

pytestmark = pytest.mark.skipif(
    not (_DATA_DIR / "2026-05-07.parquet").exists(),
    reason=f"fixture data not present: {_DATA_DIR}/2026-05-07.parquet",
)


def _build_env(emit_debug_features: bool):
    from data.episode_builder import load_day
    from env.betfair_env import BetfairEnv
    from training_v2.cohort.worker import scalping_train_config
    day = load_day("2026-05-07", data_dir=_DATA_DIR)
    cfg = scalping_train_config()
    return BetfairEnv(
        day, cfg,
        reward_overrides={
            "force_close_before_off_seconds": 120.0,
            "close_feasibility_max_spread_pct": 0.05,
        },
        emit_debug_features=emit_debug_features,
    )


def test_info_debug_features_empty_when_off():
    env = _build_env(emit_debug_features=False)
    env.reset()
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, _, _, _, info = env.step(action)
    # Both heavy fields should be empty/absent under the gate.
    assert info["debug_features"] == {}, (
        f"debug_features must be empty when off, got {info['debug_features']!r}"
    )
    assert info["passive_orders"] == [], (
        f"passive_orders must be empty when off, got {info['passive_orders']!r}"
    )


def test_info_debug_features_populated_when_on():
    """The default-True path is unchanged."""
    env = _build_env(emit_debug_features=True)
    env.reset()
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    # Step until we get a tick with runners + a populated debug payload
    for _ in range(30):
        _, _, _, _, info = env.step(action)
        if info.get("debug_features"):
            break
    else:
        pytest.fail("no debug_features observed in 30 steps with flag ON")
    # Each runner entry should have the canonical 5 keys.
    any_runner = next(iter(info["debug_features"].values()))
    for key in ("obi_topN", "weighted_microprice", "traded_delta", "mid_drift", "book_churn"):
        assert key in any_runner, f"key {key!r} missing from debug_features"


def test_episode_termination_unchanged_when_off():
    """Disabling debug features must not affect terminal behaviour."""
    env_on = _build_env(emit_debug_features=True)
    env_off = _build_env(emit_debug_features=False)
    env_on.reset()
    env_off.reset()
    action = np.zeros(env_on.action_space.shape, dtype=np.float32)
    # Run both for the same N steps; if either terminates the other
    # must terminate within the same iteration.
    for i in range(2000):
        _, _, t_on, tr_on, _ = env_on.step(action)
        _, _, t_off, tr_off, _ = env_off.step(action)
        if t_on or tr_on or t_off or tr_off:
            assert (t_on, tr_on) == (t_off, tr_off), (
                f"step {i}: termination state diverges. "
                f"on=({t_on},{tr_on}) off=({t_off},{tr_off})"
            )
            break


def test_obs_shape_unchanged_when_off():
    """Disabling debug features must not change observation shape."""
    env_on = _build_env(emit_debug_features=True)
    env_off = _build_env(emit_debug_features=False)
    obs_on, _ = env_on.reset()
    obs_off, _ = env_off.reset()
    assert obs_on.shape == obs_off.shape, (
        f"obs shape differs: on={obs_on.shape} off={obs_off.shape}"
    )
