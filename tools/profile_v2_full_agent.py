"""Profile the FULL per-agent inner loop: env build + rollout + PPO update.

Runs ``DiscretePPOTrainer.train_episode`` once on a single training day
with the same hyperparameters the running cohort used. Reports a
high-level wall breakdown (env build / rollout / PPO update) and writes
a cProfile dump.

Use this to size the PPO-update bucket — phase_3.md option F. The
rollout-only profile (``tools.profile_v2_rollout``) already covers the
rollout path.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import torch

from agents_v2.discrete_policy import DiscreteLSTMPolicy
from agents_v2.env_shim import DEFAULT_SCORER_DIR
from training_v2.cohort.worker import _build_env_for_day, scalping_train_config
from training_v2.discrete_ppo.trainer import DiscretePPOTrainer


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_predictor_bundle():
    from predictors import PredictorBundle
    base = REPO_ROOT.parent / "betfair-predictors" / "production"
    return PredictorBundle.from_manifests(
        champion_manifest=base / "race-outcome" / "manifest.json",
        ranker_manifest=base / "race-outcome-ranker" / "manifest.json",
        direction_manifest=base / "direction-predictor" / "manifest.json",
    )


def build_env_and_shim(*, day: str, data_dir: Path):
    cfg = scalping_train_config()
    cfg["training"]["strategy_mode"] = "arb"
    cfg.setdefault("observations", {})
    cfg["observations"]["use_race_outcome_predictor"] = True
    cfg["observations"]["use_direction_predictor"] = True
    bundle = build_predictor_bundle()
    return _build_env_for_day(
        day_str=day,
        data_dir=data_dir,
        cfg=cfg,
        scorer_dir=DEFAULT_SCORER_DIR,
        reward_overrides={
            "force_close_before_off_seconds": 120.0,
            "close_feasibility_max_spread_pct": 0.05,
        },
        predictor_bundle=bundle,
        use_race_outcome_predictor=True,
        use_direction_predictor=True,
        predictor_lean_obs=True,
        predictor_p_win_back_threshold=0.20,
        predictor_p_win_lay_threshold=0.40,
        race_confidence_threshold=0.50,
        lay_price_max=20.0,
        emit_debug_features=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", default="2026-04-16")
    ap.add_argument("--data-dir", default="data/processed", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "plans" / "cohort_training_speedup" / "phase_3_full_agent.txt",
    )
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    print(f"[setup] build env+shim {args.day} ...", flush=True)
    t0 = time.perf_counter()
    env, shim = build_env_and_shim(day=args.day, data_dir=args.data_dir)
    env_build_s = time.perf_counter() - t0
    print(
        f"[setup] env+shim built {env_build_s:.2f}s "
        f"obs_dim={shim.obs_dim} action_n={shim.action_space.n}",
        flush=True,
    )

    policy = DiscreteLSTMPolicy(
        obs_dim=shim.obs_dim,
        action_space=shim.action_space,
        hidden_size=128,
    )
    policy.to(args.device)

    trainer = DiscretePPOTrainer(
        policy=policy,
        shim=shim,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coeff=0.01,
        value_coeff=0.5,
        ppo_epochs=4,
        mini_batch_size=64,
        device=args.device,
    )

    # warm-up: untimed first episode to compile any cuda kernels and
    # pre-allocate buffers
    print("[warmup] running one untimed train_episode ...", flush=True)
    trainer.train_episode()

    # rebuild env+shim for second episode (collector exhausts the day)
    t0 = time.perf_counter()
    env2, shim2 = build_env_and_shim(day=args.day, data_dir=args.data_dir)
    env_build_s2 = time.perf_counter() - t0
    trainer.shim = shim2
    trainer.action_space = shim2.action_space
    from training_v2.discrete_ppo.rollout import RolloutCollector
    trainer._collector = RolloutCollector(
        shim=shim2, policy=trainer.policy, device=args.device,
    )

    print(f"[setup] env+shim rebuilt {env_build_s2:.2f}s", flush=True)

    # Profile the full train_episode (rollout + GAE + ppo_update).
    print(f"[profile] starting train_episode on device={args.device} ...", flush=True)
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    stats = trainer.train_episode()
    pr.disable()
    full_wall = time.perf_counter() - t0
    print(
        f"[profile] train_episode wall={full_wall:.2f}s "
        f"n_steps={stats.n_steps} n_updates={stats.n_updates_run} "
        f"reward={stats.total_reward:.2f}",
        flush=True,
    )

    out = io.StringIO()
    out.write(
        f"# phase_3 full-agent profile\n"
        f"# day={args.day} device={args.device}\n"
        f"# env_build_s={env_build_s:.3f} (warm-up) "
        f"env_build_s2={env_build_s2:.3f} (timed)\n"
        f"# train_episode_wall_s={full_wall:.3f} "
        f"n_steps={stats.n_steps} n_updates={stats.n_updates_run} "
        f"trainer_internal_wall_s={stats.wall_time_sec:.3f}\n"
        f"# python={sys.version.split()[0]} torch={torch.__version__}\n"
    )
    for sort_key in ("cumulative", "tottime"):
        out.write(f"\n=== sorted by {sort_key} (top {args.top}) ===\n")
        pstats.Stats(pr, stream=out).sort_stats(sort_key).print_stats(args.top)
    out.write("\n=== callers of _ppo_update ===\n")
    pstats.Stats(pr, stream=out).sort_stats("cumulative").print_callers(
        args.top, "_ppo_update"
    )
    out.write("\n=== callees of _ppo_update ===\n")
    pstats.Stats(pr, stream=out).sort_stats("cumulative").print_callees(
        args.top, "_ppo_update"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out.getvalue(), encoding="utf-8")
    pr.dump_stats(str(args.out.with_suffix(".prof")))
    print(f"[profile] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
