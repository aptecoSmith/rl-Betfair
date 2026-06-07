"""Cached-path full-agent profile — the REAL training shape.

Like tools/profile_v2_full_agent.py, but the env is built WITH the static_obs
cache injected (predictors + scorer features baked out, exactly as the
multiprocess worker runs). The live profile included the per-tick predictor +
feature-extraction cost that training bakes away; this isolates the true
training floor: env-sim (matching/settlement) + obs-read + forward + update.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from agents_v2.discrete_policy import DiscreteLSTMPolicy  # noqa: E402
from agents_v2.env_shim import DEFAULT_SCORER_DIR  # noqa: E402
from training_v2.cohort.multiproc_worker import (  # noqa: E402
    _worker_load_static_obs,
    prebuild_static_obs_cache,
)
from training_v2.cohort.worker import (  # noqa: E402
    _build_env_for_day,
    scalping_train_config,
)
from training_v2.discrete_ppo.rollout import RolloutCollector  # noqa: E402
from training_v2.discrete_ppo.trainer import DiscretePPOTrainer  # noqa: E402


def build_bundle():
    from predictors import PredictorBundle
    base = REPO_ROOT.parent / "betfair-predictors" / "production"
    return PredictorBundle.from_manifests(
        champion_manifest=base / "race-outcome" / "manifest.json",
        ranker_manifest=base / "race-outcome-ranker" / "manifest.json",
        direction_manifest=base / "direction-predictor" / "manifest.json",
    )


def build_cached_env(*, day, data_dir, cache_dir, lean):
    bundle = build_bundle()
    paths = prebuild_static_obs_cache(
        [day], data_dir=data_dir, cache_dir=cache_dir, predictor_bundle=bundle,
        use_race_outcome_predictor=True, use_direction_predictor=True,
        predictor_lean_obs=lean,
    )
    npy, side = paths[day]
    art = _worker_load_static_obs(day, npy, side)
    cfg = scalping_train_config()
    cfg["training"]["strategy_mode"] = "arb"
    cfg.setdefault("observations", {})
    cfg["observations"]["use_race_outcome_predictor"] = True
    cfg["observations"]["use_direction_predictor"] = True
    return _build_env_for_day(
        day_str=day, data_dir=data_dir, cfg=cfg, scorer_dir=DEFAULT_SCORER_DIR,
        reward_overrides={
            "force_close_before_off_seconds": 120.0,
            "close_feasibility_max_spread_pct": 0.05,
        },
        predictor_bundle=bundle, use_race_outcome_predictor=True,
        use_direction_predictor=True, predictor_lean_obs=lean,
        predictor_p_win_back_threshold=0.20, predictor_p_win_lay_threshold=0.40,
        race_confidence_threshold=0.50, lay_price_max=20.0,
        emit_debug_features=False, static_obs_cache={day: art},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2026-04-10")
    ap.add_argument("--data-dir", default="data/processed", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lean", type=int, default=1)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "plans" / "pbt-gpu-forward" /
                    "_measure" / "cached_agent_profile.txt")
    args = ap.parse_args()
    lean = bool(args.lean)
    cache_dir = (REPO_ROOT / "plans" / "pbt-gpu-forward" / "_measure"
                 / "_sobs_cache")

    print(f"[setup] bake + build CACHED env {args.day} (lean={lean}) ...",
          flush=True)
    t0 = time.perf_counter()
    env, shim = build_cached_env(day=args.day, data_dir=args.data_dir,
                                 cache_dir=cache_dir, lean=lean)
    print(f"[setup] cached env built {time.perf_counter() - t0:.2f}s "
          f"obs_dim={shim.obs_dim} action_n={shim.action_space.n}", flush=True)

    policy = DiscreteLSTMPolicy(
        obs_dim=shim.obs_dim, action_space=shim.action_space, hidden_size=128,
    ).to(args.device)
    trainer = DiscretePPOTrainer(
        policy=policy, shim=shim, learning_rate=3e-4, gamma=0.99,
        gae_lambda=0.95, clip_range=0.2, entropy_coeff=0.01, value_coeff=0.5,
        ppo_epochs=4, mini_batch_size=64, device=args.device,
    )

    print("[warmup] one untimed train_episode ...", flush=True)
    trainer.train_episode()

    env2, shim2 = build_cached_env(day=args.day, data_dir=args.data_dir,
                                   cache_dir=cache_dir, lean=lean)
    trainer.shim = shim2
    trainer.action_space = shim2.action_space
    trainer._collector = RolloutCollector(
        shim=shim2, policy=trainer.policy, device=args.device)

    print(f"[profile] train_episode on device={args.device} ...", flush=True)
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    stats = trainer.train_episode()
    pr.disable()
    wall = time.perf_counter() - t0
    print(f"[profile] train_episode wall={wall:.2f}s n_steps={stats.n_steps} "
          f"n_updates={stats.n_updates_run}", flush=True)

    out = io.StringIO()
    out.write(f"# CACHED full-agent profile\n# day={args.day} "
              f"device={args.device} lean={lean}\n# wall={wall:.3f} "
              f"n_steps={stats.n_steps} n_updates={stats.n_updates_run}\n")
    for sk in ("cumulative", "tottime"):
        ps = pstats.Stats(pr, stream=out)
        ps.sort_stats(sk)
        out.write(f"\n=== {sk} (top 40) ===\n")
        ps.print_stats(40)
    args.out.write_text(out.getvalue())
    print(f"[profile] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
