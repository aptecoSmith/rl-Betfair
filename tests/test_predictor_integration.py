"""Predictor-integration regression guards (plans/predictor-integration/).

Covers `hard_constraints.md §1` (flag-off byte-identical to pre-plan) and
the obs-schema bookkeeping deltas Session 02 lands.

The load-bearing test is
:func:`test_flag_off_is_byte_identical_to_pre_plan` — it re-runs the
deterministic zero-action rollout that captured
``tests/fixtures/predictor_integration_baseline.json`` and asserts the
SHA256 digest of the per-step (reward, raw_pnl_reward, shaped_bonus,
race_idx) tuples matches the captured baseline. Both predictor flags
default off; once Session 02 lands the new kwargs, this test will
explicitly pass `use_race_outcome_predictor=False` +
`use_direction_predictor=False`.

The fixture was captured via
``python tests/_capture_predictor_integration_baseline.py`` against the
pre-Session-02 commit (`81cd092` = "feat(predictor-integration):
Session 01 — predictor loader + segment router"). If the fixture ever
needs to be re-captured (e.g. after a deliberate env change), do it on
a clean checkout of that commit and bump the captured_at marker.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed_amber_v2_window"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "predictor_integration_baseline.json"


def _require_fixture_and_data() -> dict:
    if not FIXTURE_PATH.exists():
        pytest.skip(f"baseline fixture missing: {FIXTURE_PATH}")
    with FIXTURE_PATH.open("r", encoding="utf-8") as fh:
        baseline = json.load(fh)
    if not (DATA_DIR / f"{baseline['day']}.parquet").exists():
        pytest.skip(
            f"baseline day parquet missing: {DATA_DIR / (baseline['day'] + '.parquet')}"
        )
    return baseline


@pytest.mark.slow
def test_flag_off_is_byte_identical_to_pre_plan():
    """Hard_constraints §1: flag-off env is byte-identical to pre-plan.

    Replays the same deterministic zero-action rollout that produced the
    captured baseline and asserts the per-step digest matches. The
    digest fingerprints (reward, raw_pnl_reward, shaped_bonus, race_idx)
    on every step — any drift in env-side behaviour (matcher, reward
    shaping, settlement, scalping accounting, …) trips the digest.

    Post-Session-02 this test will pass `use_race_outcome_predictor=False`
    + `use_direction_predictor=False` to the env constructor. Today the
    env doesn't yet have those kwargs; the test runs against today's
    flag-defaulted-off behaviour, which IS the pre-Session-02 baseline.
    """
    baseline = _require_fixture_and_data()

    from data.episode_builder import load_day  # type: ignore[import-not-found]
    from env.betfair_env import BetfairEnv  # type: ignore[import-not-found]
    from training_v2.discrete_ppo.train import (  # type: ignore[import-not-found]
        _scalping_train_config,
    )

    cfg = _scalping_train_config(max_runners=baseline["max_runners"])
    day = load_day(baseline["day"], data_dir=DATA_DIR)
    env_kwargs = {}
    # Session 02 will add `use_race_outcome_predictor` /
    # `use_direction_predictor` kwargs; tolerate their absence pre-Session-02.
    try:
        env = BetfairEnv(day, cfg, **env_kwargs)
    except TypeError:
        env = BetfairEnv(day, cfg)

    obs, _info = env.reset(seed=baseline["seed"])
    action_dim = int(env.action_space.shape[0])
    assert action_dim == baseline["action_dim"], (
        f"action_dim drift: env={action_dim} baseline={baseline['action_dim']}"
    )

    zero_action = np.zeros(action_dim, dtype=np.float32)
    digest = hashlib.sha256()
    n_steps = 0
    info: dict = {}
    while True:
        obs, reward, terminated, truncated, info = env.step(zero_action)
        digest.update(np.float64(float(reward)).tobytes())
        digest.update(np.float64(float(info.get("raw_pnl_reward", 0.0))).tobytes())
        digest.update(np.float64(float(info.get("shaped_bonus", 0.0))).tobytes())
        digest.update(int(info.get("race_idx", -1)).to_bytes(4, "little", signed=True))
        n_steps += 1
        if terminated or truncated:
            break
        if n_steps >= 30_000:
            raise RuntimeError("regression run overshot 30k steps")

    assert n_steps == baseline["n_steps"], (
        f"step count drifted: env={n_steps} baseline={baseline['n_steps']}"
    )
    actual_digest = digest.hexdigest()
    if actual_digest != baseline["per_step_digest"]:
        # Diagnostic localisation: re-run with sample capture if the
        # digest diverges. For now, surface the mismatch with the sampled
        # steps from the baseline so the operator can replay locally.
        sample_summary = ", ".join(
            f"step={s['step']} reward={s['reward']:.6g} raw_pnl={s['raw_pnl_reward']:.6g}"
            for s in baseline["sampled_steps"][:3]
        )
        raise AssertionError(
            f"per-step digest mismatch:\n"
            f"  baseline  = {baseline['per_step_digest']}\n"
            f"  current   = {actual_digest}\n"
            f"  baseline samples (first 3): {sample_summary}\n"
            f"  Re-run tests/_capture_predictor_integration_baseline.py "
            f"on a clean pre-Session-02 commit to confirm the fixture is "
            f"current; otherwise investigate env-side drift."
        )

    # Final aggregates as a secondary check.
    final_day_pnl = float(info.get("day_pnl", 0.0))
    assert final_day_pnl == baseline["final_day_pnl"], (
        f"day_pnl drift: env={final_day_pnl} baseline={baseline['final_day_pnl']}"
    )
