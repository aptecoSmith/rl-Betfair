"""Capture pre-Session-02 baseline for the predictor-integration byte-identical regression.

Runs a deterministic single-race rollout against the v7 (pre-Session-02)
env with a hardcoded zero-action stream and writes the env's per-step
output (raw_pnl_reward, day_pnl, bet records) to a JSON fixture.

Why a fixed action stream rather than a real policy: when Session 02
extends RUNNER_KEYS (RUNNER_DIM 125 → 143), a fresh-init seeded policy's
first-layer weights necessarily differ between v7 and v8 so its actions
will differ. The byte-identical guarantee in `hard_constraints.md §1` is
about ENV behaviour — given the same actions, the env produces the same
observations / rewards / bets. Replaying a fixed action stream isolates
that.

Usage (run ONCE, before Session 02 touches env/betfair_env.py):

    python tests/_capture_predictor_integration_baseline.py

Writes `tests/fixtures/predictor_integration_baseline.json`. Commit the
fixture; Session 02's regression test reads it and compares against a
fresh re-run with both predictor flags off.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed_amber_v2_window"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "predictor_integration_baseline.json"


def _pick_baseline_day() -> str:
    """Pick a deterministic day for the baseline.

    Hard-coded rather than auto-discovered so the baseline doesn't drift
    if new parquets land between sessions.
    """
    candidates = ["2026-04-23", "2026-04-22", "2026-04-21", "2026-04-20"]
    for d in candidates:
        if (DATA_DIR / f"{d}.parquet").exists():
            return d
    raise FileNotFoundError(
        f"none of {candidates} present under {DATA_DIR}; check data dir"
    )


def _capture_baseline() -> dict:
    sys.path.insert(0, str(REPO_ROOT))
    from data.episode_builder import load_day  # type: ignore[import-not-found]
    from env.betfair_env import (  # type: ignore[import-not-found]
        OBS_SCHEMA_VERSION,
        RUNNER_DIM,
        BetfairEnv,
    )
    from training_v2.discrete_ppo.train import (  # type: ignore[import-not-found]
        _scalping_train_config,
    )

    day_str = _pick_baseline_day()
    cfg = _scalping_train_config(max_runners=14)
    day = load_day(day_str, data_dir=DATA_DIR)
    env = BetfairEnv(day, cfg)

    seed = 42
    obs, _info = env.reset(seed=seed)
    obs_dim = int(obs.shape[0])
    action_dim = int(env.action_space.shape[0])

    # Deterministic zero-action stream (no-op: agent does nothing).
    zero_action = np.zeros(action_dim, dtype=np.float32)

    rewards: list[float] = []
    raw_pnl_rewards: list[float] = []
    shaped_bonuses: list[float] = []
    race_idxs: list[int] = []
    n_steps = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(zero_action)
        rewards.append(float(reward))
        raw_pnl_rewards.append(float(info.get("raw_pnl_reward", 0.0)))
        shaped_bonuses.append(float(info.get("shaped_bonus", 0.0)))
        race_idxs.append(int(info.get("race_idx", -1)))
        n_steps += 1
        if terminated or truncated:
            break
        # Hard cap; one full day is typically ~10-15k ticks.
        if n_steps >= 30_000:
            raise RuntimeError(
                f"baseline-capture overshot 30k steps without terminating; "
                f"investigate before committing the fixture"
            )

    # Compact summary fixture: SHA256 over every per-step value the
    # regression test cares about, plus a few sampled steps for
    # diagnostic localisation when the hash diverges. Full per-step
    # arrays bloat the fixture to ~2 MB on a one-day capture; the hash
    # fingerprints the same data in 64 chars.
    digest = hashlib.sha256()
    for r, rp, sb, idx in zip(rewards, raw_pnl_rewards, shaped_bonuses, race_idxs):
        # Encode as exact float bytes so any floating-point drift trips
        # the digest regardless of repr-rounding.
        digest.update(np.float64(r).tobytes())
        digest.update(np.float64(rp).tobytes())
        digest.update(np.float64(sb).tobytes())
        digest.update(int(idx).to_bytes(4, "little", signed=True))

    # Sampled steps for diagnostic output on a digest mismatch.
    sample_indices = sorted(
        set(
            list(range(min(10, n_steps)))
            + list(range(max(0, n_steps - 10), n_steps))
            + list(range(0, n_steps, max(1, n_steps // 20)))
        )
    )
    sampled_steps = [
        {
            "step": i,
            "reward": rewards[i],
            "raw_pnl_reward": raw_pnl_rewards[i],
            "shaped_bonus": shaped_bonuses[i],
            "race_idx": race_idxs[i],
        }
        for i in sample_indices
    ]

    final_info = info or {}
    payload = {
        "captured_at": "pre-Session-02",
        "obs_schema_version": int(OBS_SCHEMA_VERSION),
        "runner_dim": int(RUNNER_DIM),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "day": day_str,
        "seed": seed,
        "max_runners": 14,
        "n_steps": n_steps,
        "final_day_pnl": float(final_info.get("day_pnl", 0.0)),
        "final_total_pnl": float(final_info.get("total_pnl", 0.0)),
        "races_completed": int(final_info.get("races_completed", 0)),
        # Aggregate stats over the per-step arrays — useful at a glance
        # without needing to recompute from a dropped digest.
        "sum_reward": float(np.sum(rewards)),
        "sum_raw_pnl_reward": float(np.sum(raw_pnl_rewards)),
        "sum_shaped_bonus": float(np.sum(shaped_bonuses)),
        # Load-bearing equality surface.
        "per_step_digest": digest.hexdigest(),
        "sampled_steps": sampled_steps,
    }
    return payload


def main() -> int:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = _capture_baseline()
    with FIXTURE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"baseline captured: {FIXTURE_PATH}")
    print(
        f"  schema={payload['obs_schema_version']} "
        f"runner_dim={payload['runner_dim']} "
        f"obs_dim={payload['obs_dim']} "
        f"action_dim={payload['action_dim']}"
    )
    print(
        f"  day={payload['day']} seed={payload['seed']} "
        f"n_steps={payload['n_steps']}"
    )
    print(
        f"  final_day_pnl={payload['final_day_pnl']:.6f} "
        f"final_total_pnl={payload['final_total_pnl']:.6f} "
        f"races_completed={payload['races_completed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
