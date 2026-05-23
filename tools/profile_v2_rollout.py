"""cProfile a single v2 cohort training rollout.

Mirrors the build path of ``training_v2.cohort.worker.train_one_agent``
for the cohort currently running as
``_predictor_SCALPING_postfix_e3_cohort_*`` (PID 32292) — same predictor
bundle, same reward_overrides, same predictor thresholds, same
strategy_mode. Runs one ``RolloutCollector.collect_episode`` pass on a
training day under ``cProfile`` and dumps two views:

  - top-N by cumulative time (where the time spends including callees)
  - top-N by tottime    (where the work is actually done, ex-callees)

Writes plans/cohort_training_speedup/phase_3_profile.txt.

Usage::

    python -m tools.profile_v2_rollout --day 2026-04-16 --device cuda \
        --top 40

Note: this script BUILDS its own env/policy/collector — it does NOT
attach to the running cohort. Safe to run alongside the cohort (uses
its own python process; CUDA shares cleanly).
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
from training_v2.discrete_ppo.rollout import RolloutCollector


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

    reward_overrides = {
        "force_close_before_off_seconds": 120.0,
        "close_feasibility_max_spread_pct": 0.05,
    }
    return _build_env_for_day(
        day_str=day,
        data_dir=data_dir,
        cfg=cfg,
        scorer_dir=DEFAULT_SCORER_DIR,
        reward_overrides=reward_overrides,
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


def build_policy(*, shim, device: str) -> DiscreteLSTMPolicy:
    policy = DiscreteLSTMPolicy(
        obs_dim=shim.obs_dim,
        action_space=shim.action_space,
        hidden_size=128,
    )
    policy.to(device)
    policy.eval()
    return policy


def render_stats(pr: cProfile.Profile, top_n: int) -> str:
    out = io.StringIO()
    for sort_key in ("cumulative", "tottime"):
        out.write(f"\n=== sorted by {sort_key} (top {top_n}) ===\n")
        st = pstats.Stats(pr, stream=out).sort_stats(sort_key)
        st.print_stats(top_n)
    out.write("\n=== callers of env.step ===\n")
    try:
        pstats.Stats(pr, stream=out).sort_stats("cumulative").print_callers(
            top_n, "betfair_env.py:.*step"
        )
    except Exception as exc:
        out.write(f"(callers print failed: {exc})\n")
    return out.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", default="2026-04-16")
    ap.add_argument("--data-dir", default="data/processed", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "plans" / "cohort_training_speedup" / "phase_3_profile.txt",
    )
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU", file=sys.stderr)
        args.device = "cpu"

    print(f"[setup] building env + shim for {args.day} ...", flush=True)
    t0 = time.perf_counter()
    env, shim = build_env_and_shim(day=args.day, data_dir=args.data_dir)
    build_t = time.perf_counter() - t0
    print(
        f"[setup] env+shim built in {build_t:.2f}s "
        f"obs_dim={shim.obs_dim} action_n={shim.action_space.n} "
        f"max_runners={shim.max_runners} races={len(env.day.races)}",
        flush=True,
    )

    policy = build_policy(shim=shim, device=args.device)
    collector = RolloutCollector(shim=shim, policy=policy, device=args.device)

    # warm-up: run a short slice without profiling so cuda kernels are
    # compiled and the import / first-step costs don't pollute the
    # profile.  We do this by collecting one episode on the first race
    # only would be ideal, but the collector runs whole-day; instead
    # we accept that the warm-up day takes ~60 s but only profile the
    # second episode.  To keep the harness simple we ALWAYS profile a
    # single full-day episode (the very thing the cohort runs); the
    # first-step CUDA-init cost is part of the cohort's cost too.
    print(
        f"[profile] starting collect_episode on device={args.device} ...",
        flush=True,
    )
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    batch = collector.collect_episode(deterministic=False)
    pr.disable()
    rollout_t = time.perf_counter() - t0

    n_steps = int(batch.n_steps)
    ms_per_step = (rollout_t / n_steps * 1000.0) if n_steps else float("nan")
    print(
        f"[profile] done — {n_steps} steps in {rollout_t:.2f}s "
        f"= {ms_per_step:.3f} ms/step",
        flush=True,
    )

    header = (
        f"# phase_3 profile dump\n"
        f"# day={args.day} device={args.device} "
        f"n_steps={n_steps} rollout_wall_s={rollout_t:.3f} "
        f"ms_per_step={ms_per_step:.3f}\n"
        f"# obs_dim={shim.obs_dim} action_n={shim.action_space.n} "
        f"max_runners={shim.max_runners} races={len(env.day.races)}\n"
        f"# env+shim build_s={build_t:.2f}\n"
        f"# python={sys.version.split()[0]} torch={torch.__version__}\n"
    )
    body = render_stats(pr, args.top)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(header + body, encoding="utf-8")
    print(f"[profile] wrote {args.out}", flush=True)

    # Also dump raw .prof for ad-hoc snakeviz/grep later.
    raw = args.out.with_suffix(".prof")
    pr.dump_stats(str(raw))
    print(f"[profile] wrote raw stats to {raw}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
