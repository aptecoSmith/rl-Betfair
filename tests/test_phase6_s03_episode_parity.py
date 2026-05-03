"""Phase 6 Session 03 episode-parity guard for the closed-form
``_spread_in_ticks``.

The unit-test file (``test_feature_extractor_spread_in_ticks.py``) covers
bit-identity on 10 k synthetic random pairs. This integration test runs
one full episode on ``--seed 42 --day 2026-04-23 --device cpu`` twice —
once with the production closed form and once with the pre-S03 iterative
walk monkey-patched in — and asserts byte-equality on every step's
``obs``, ``mask``, per-step ``info["raw_pnl_reward"]``, final
``info["day_pnl"]``, and the first 100 ``log_prob_action`` entries from
the rollout's collected transitions.

This guards against the case where the closed form is bit-identical on
synthetic random inputs but somehow diverges on the specific price
distributions seen in real data (unlikely but cheap to verify).

Skips cleanly when the scorer artefacts (``models/scorer_v1/``) or the
2026-04-23 parquet under ``data/processed_amber_v2_window/`` are absent —
same convention as ``tests/test_env_shim_batched_scorer.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from env.tick_ladder import tick_offset


REPO_ROOT = Path(__file__).resolve().parents[1]
SCORER_DIR = REPO_ROOT / "models" / "scorer_v1"
DATA_DIR = REPO_ROOT / "data" / "processed_amber_v2_window"
INTEGRATION_DAY = "2026-04-23"
INTEGRATION_PARQUET = DATA_DIR / f"{INTEGRATION_DAY}.parquet"


def _scorer_runtime_available() -> tuple[bool, str]:
    if not (SCORER_DIR / "model.lgb").exists():
        return False, f"Scorer artefacts missing under {SCORER_DIR}."
    try:
        import lightgbm  # noqa: F401
        import joblib  # noqa: F401
    except Exception as exc:
        return False, f"scorer deps unavailable: {exc!r}"
    return True, ""


_runtime_ok, _runtime_reason = _scorer_runtime_available()
pytestmark = pytest.mark.skipif(not _runtime_ok, reason=_runtime_reason)


def _walk_spread_in_ticks(best_back: float, best_lay: float) -> float:
    """Pre-S03 iterative walk — the parity oracle for this test."""
    if best_lay <= best_back:
        return 0.0
    for n in range(1, 50):
        p = tick_offset(best_back, n, +1)
        if p >= best_lay - 1e-9:
            return float(n)
    return math.nan


_integration_skip_reason = (
    None
    if INTEGRATION_PARQUET.exists()
    else f"Integration day data missing: {INTEGRATION_PARQUET}"
)


@pytest.mark.skipif(
    _integration_skip_reason is not None,
    reason=_integration_skip_reason or "",
)
@pytest.mark.slow
def test_full_episode_byte_identical_walk_vs_closed_form(monkeypatch):
    """Full-episode bit-identity guard for the S03 closed-form rewrite."""
    from data.episode_builder import load_day
    from env.betfair_env import BetfairEnv

    from agents_v2.discrete_policy import DiscreteLSTMPolicy
    from agents_v2.env_shim import DiscreteActionShim
    from training_v2.discrete_ppo.rollout import RolloutCollector
    from training_v2.discrete_ppo.train import _scalping_train_config

    seed = 42

    def _run_episode(use_walk: bool) -> dict:
        torch.manual_seed(seed)
        np.random.seed(seed)

        if use_walk:
            # Monkey-patch the production closed form back to the walk
            # oracle for the duration of this episode.
            monkeypatch.setattr(
                "training_v2.scorer.feature_extractor._spread_in_ticks",
                _walk_spread_in_ticks,
            )
        else:
            monkeypatch.undo()

        cfg = _scalping_train_config(max_runners=14)
        day = load_day(INTEGRATION_DAY, data_dir=DATA_DIR)
        env = BetfairEnv(day, cfg)
        shim = DiscreteActionShim(env, scorer_dir=SCORER_DIR)

        policy = DiscreteLSTMPolicy(
            obs_dim=shim.obs_dim,
            action_space=shim.action_space,
            hidden_size=128,
        )

        captured_infos: list[dict] = []
        orig_step = env.step

        def _capturing_step(action):
            obs, reward, term, trunc, info = orig_step(action)
            snap = {
                "raw_pnl_reward": float(info.get("raw_pnl_reward", 0.0)),
                "day_pnl": float(info.get("day_pnl", 0.0)),
            }
            captured_infos.append(snap)
            return obs, reward, term, trunc, info

        env.step = _capturing_step  # type: ignore[method-assign]

        collector = RolloutCollector(shim=shim, policy=policy, device="cpu")
        batch = collector.collect_episode()

        return {
            "obs": np.asarray(batch.obs).copy(),
            "mask": np.asarray(batch.mask).copy(),
            "log_prob_action": np.asarray(batch.log_prob_action).copy(),
            "n_steps": int(batch.n_steps),
            "infos": captured_infos,
        }

    walk = _run_episode(use_walk=True)
    closed = _run_episode(use_walk=False)

    assert walk["n_steps"] == closed["n_steps"], (
        f"step count diverged: walk={walk['n_steps']} "
        f"closed={closed['n_steps']}"
    )
    n = walk["n_steps"]
    assert n > 0, "rollout produced zero steps"

    for t in range(n):
        if not np.array_equal(walk["obs"][t], closed["obs"][t]):
            diff = np.where(walk["obs"][t] != closed["obs"][t])[0]
            raise AssertionError(
                f"obs diverged at step {t}; differing indices "
                f"{diff[:8].tolist()} walk[idx]={walk['obs'][t][diff[:4]]} "
                f"closed[idx]={closed['obs'][t][diff[:4]]}"
            )
        if not np.array_equal(walk["mask"][t], closed["mask"][t]):
            raise AssertionError(f"mask diverged at step {t}")

    assert len(walk["infos"]) == len(closed["infos"]) == n
    for t in range(n):
        if walk["infos"][t]["raw_pnl_reward"] != closed["infos"][t]["raw_pnl_reward"]:
            raise AssertionError(
                f"raw_pnl_reward diverged at step {t}: "
                f"walk={walk['infos'][t]['raw_pnl_reward']!r} "
                f"closed={closed['infos'][t]['raw_pnl_reward']!r}"
            )
    assert walk["infos"][-1]["day_pnl"] == closed["infos"][-1]["day_pnl"], (
        f"final day_pnl diverged: walk={walk['infos'][-1]['day_pnl']!r} "
        f"closed={closed['infos'][-1]['day_pnl']!r}"
    )

    n_check = min(100, n)
    walk_lp = walk["log_prob_action"][:n_check]
    closed_lp = closed["log_prob_action"][:n_check]
    if not np.array_equal(walk_lp, closed_lp):
        diff_idx = np.where(walk_lp != closed_lp)[0]
        first = int(diff_idx[0])
        raise AssertionError(
            f"log_prob_action diverged in first {n_check}: "
            f"first diff at idx {first} "
            f"walk={walk_lp[first]!r} closed={closed_lp[first]!r} "
            f"(ULP-strict equality required)"
        )
