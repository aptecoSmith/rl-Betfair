"""End-to-end smoke test for the v2 discrete policy + env shim.

Phase 1, Session 02 deliverable. Marked ``@pytest.mark.slow`` —
constructs a tiny synthetic day so the test stays well under the
project-wide 60s timeout. The full 1000-step smoke is the CLI run
(``python -m agents_v2.smoke_test``), not this test.

Skips cleanly if the Phase 0 scorer artefacts are missing or
``lightgbm`` / ``joblib`` aren't importable.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from env.betfair_env import BetfairEnv

from agents_v2.action_space import ActionType
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


@pytest.mark.timeout(45)
def test_smoke_random_policy_runs_through_synthetic_day():
    """Random masked policy → shim → env, no exceptions, sane shapes.

    Drives a synthetic 2-race day to exhaustion and asserts:

    - obs stays finite and ``shape == (shim.obs_dim,)`` every step
    - mask sampled action never lands on an illegal index
    - hidden state propagates (no None / shape break across steps)
    """
    from agents_v2.env_shim import DiscreteActionShim

    torch.manual_seed(0)
    env = BetfairEnv(
        _make_day(n_races=2, n_pre_ticks=10, n_inplay_ticks=2),
        _scalping_config(),
    )
    shim = DiscreteActionShim(env)
    obs, _info = shim.reset()

    policy = DiscreteLSTMPolicy(
        obs_dim=shim.obs_dim,
        action_space=shim.action_space,
        hidden_size=32,
    )
    policy.eval()
    hidden = policy.init_hidden(batch=1)

    histogram: Counter[ActionType] = Counter()
    refusal_count = 0
    n_steps = 0
    done = False
    while not done:
        assert obs.shape == (shim.obs_dim,)
        assert np.isfinite(obs).all()
        mask_np = shim.get_action_mask()
        assert mask_np[0], "NOOP must always be legal"

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(torch.float32)
            mask_t = torch.from_numpy(mask_np).unsqueeze(0)
            out = policy(obs_t, hidden_state=hidden, mask=mask_t)
            action_idx = int(out.action_dist.sample().item())
            hidden = out.new_hidden_state

        if not mask_np[action_idx]:
            refusal_count += 1
        kind, _runner = shim.action_space.decode(action_idx)
        histogram[kind] += 1

        obs, _r, terminated, truncated, _info = shim.step(action_idx)
        done = terminated or truncated
        n_steps += 1
        if n_steps > 200:
            pytest.fail(
                "synthetic day didn't terminate within 200 steps — "
                "fixture or env behaviour drift",
            )

    # Success bar #4: zero illegal actions through the masked categorical.
    assert refusal_count == 0, (
        f"masked categorical produced {refusal_count} illegal samples"
    )
    # NOOP is always available; expect at least some NOOPs in a random
    # policy run.
    assert histogram[ActionType.NOOP] >= 1


def test_smoke_value_head_shape_through_real_step():
    """The forward path on a real obs returns ``(1, max_runners)`` value."""
    from agents_v2.env_shim import DiscreteActionShim

    torch.manual_seed(0)
    env = BetfairEnv(_make_day(n_races=1), _scalping_config())
    shim = DiscreteActionShim(env)
    obs, _info = shim.reset()

    policy = DiscreteLSTMPolicy(
        obs_dim=shim.obs_dim,
        action_space=shim.action_space,
        hidden_size=32,
    )
    policy.eval()

    with torch.no_grad():
        out = policy(torch.from_numpy(obs).unsqueeze(0).to(torch.float32))
    assert out.value_per_runner.shape == (1, shim.action_space.max_runners)
