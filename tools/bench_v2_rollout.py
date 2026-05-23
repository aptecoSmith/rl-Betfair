"""Quick wall-time bench of rollout (no profiler) for device A/B.

Reuses build helpers from ``tools.profile_v2_rollout``. Runs ONE
``collect_episode`` per requested device and prints wall + ms/step.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

from tools.profile_v2_rollout import (
    build_env_and_shim,
    build_policy,
)
from training_v2.discrete_ppo.rollout import RolloutCollector


def bench_once(*, day: str, data_dir: Path, device: str) -> tuple[int, float]:
    env, shim = build_env_and_shim(day=day, data_dir=data_dir)
    policy = build_policy(shim=shim, device=device)
    collector = RolloutCollector(shim=shim, policy=policy, device=device)
    t0 = time.perf_counter()
    batch = collector.collect_episode(deterministic=False)
    wall = time.perf_counter() - t0
    return int(batch.n_steps), wall


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", default="2026-04-16")
    ap.add_argument("--data-dir", default="data/processed", type=Path)
    ap.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    args = ap.parse_args()

    for dev in args.devices:
        print(f"[bench] device={dev} building+running...", flush=True)
        n, w = bench_once(day=args.day, data_dir=args.data_dir, device=dev)
        print(f"[bench] device={dev} steps={n} wall={w:.2f}s "
              f"ms_per_step={(w/n*1000):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
