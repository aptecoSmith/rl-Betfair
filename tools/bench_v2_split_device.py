"""Bench: split-device (CPU rollout + CUDA update) vs single-device.

Runs DiscretePPOTrainer.train_episode once per configuration on the
same day so the rollout has identical content (deterministic env). The
TRAINER measures wall_time_sec internally; we also wrap the call to
verify.

Per-config configs:
  1) single-device cuda    (baseline)
  2) split: rollout=cpu, update=cuda  (Option A)
  3) single-device cpu     (sanity bound)

For each: env_build_s (warm-up only counted once per config — env is
not shared across configs to keep state independent).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from agents_v2.discrete_policy import DiscreteLSTMPolicy
from training_v2.discrete_ppo.trainer import DiscretePPOTrainer

# Reuse setup
from tools.profile_v2_full_agent import build_env_and_shim


REPO_ROOT = Path(__file__).resolve().parents[1]


def bench_one(*, day: str, data_dir: Path, device: str,
              rollout_device: str | None, n_episodes: int) -> dict:
    print(
        f"[bench] device={device} rollout_device={rollout_device or device} "
        f"warming env...",
        flush=True,
    )
    env, shim = build_env_and_shim(day=day, data_dir=data_dir)
    policy = DiscreteLSTMPolicy(
        obs_dim=shim.obs_dim,
        action_space=shim.action_space,
        hidden_size=128,
    )
    trainer = DiscretePPOTrainer(
        policy=policy, shim=shim,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, entropy_coeff=0.01, value_coeff=0.5,
        ppo_epochs=4, mini_batch_size=64,
        device=device, rollout_device=rollout_device,
    )

    walls = []
    n_steps_seen = 0
    rewards = []
    for ep in range(n_episodes):
        if ep > 0:
            # rebuild env+shim for next episode (current day exhausted)
            _, shim2 = build_env_and_shim(day=day, data_dir=data_dir)
            trainer.shim = shim2
            trainer.action_space = shim2.action_space
            from training_v2.discrete_ppo.rollout import RolloutCollector
            trainer._collector = RolloutCollector(
                shim=shim2, policy=trainer.policy,
                device=str(trainer.rollout_device),
            )
        t0 = time.perf_counter()
        stats = trainer.train_episode()
        w = time.perf_counter() - t0
        walls.append(w)
        n_steps_seen = int(stats.n_steps)
        rewards.append(float(stats.total_reward))
        print(
            f"  ep{ep}: wall={w:.2f}s n_steps={stats.n_steps} "
            f"n_updates={stats.n_updates_run} reward={stats.total_reward:.2f}",
            flush=True,
        )
    return {
        "device": device,
        "rollout_device": rollout_device or device,
        "n_steps": n_steps_seen,
        "walls": walls,
        "rewards": rewards,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", default="2026-04-16")
    ap.add_argument("--data-dir", default="data/processed", type=Path)
    ap.add_argument("--n-episodes", type=int, default=2,
                    help="Per config (the first counts as warm-up).")
    ap.add_argument(
        "--configs", nargs="+",
        default=["cuda", "split", "cpu"],
        choices=["cuda", "split", "cpu"],
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA unavailable; only CPU configs will run.",
              file=sys.stderr)
        args.configs = [c for c in args.configs if c == "cpu"]

    results: list[dict] = []
    for cfg in args.configs:
        if cfg == "cuda":
            r = bench_one(day=args.day, data_dir=args.data_dir,
                          device="cuda", rollout_device=None,
                          n_episodes=args.n_episodes)
        elif cfg == "split":
            r = bench_one(day=args.day, data_dir=args.data_dir,
                          device="cuda", rollout_device="cpu",
                          n_episodes=args.n_episodes)
        else:  # cpu
            r = bench_one(day=args.day, data_dir=args.data_dir,
                          device="cpu", rollout_device=None,
                          n_episodes=args.n_episodes)
        results.append(r)

    print("\n=== summary (timed episode = last of each config) ===")
    print(f"{'config':<35} {'wall_s':>9} {'ms/step':>9} {'reward':>10}")
    for r in results:
        cfg_label = f"dev={r['device']} rollout={r['rollout_device']}"
        wall = r["walls"][-1]
        ms_step = wall / max(1, r["n_steps"]) * 1000.0
        print(f"{cfg_label:<35} {wall:>9.2f} {ms_step:>9.3f} "
              f"{r['rewards'][-1]:>10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
