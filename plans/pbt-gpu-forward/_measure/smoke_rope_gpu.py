"""End-to-end smoke for the RoPE transformer on the GPU policy lane
(pbt-gpu-forward task #8). Trains ONE ctx256 transformer agent for one day
with pos_encoding='rope' vs 'learned', both on the GPU lane, confirming:

  * a rope transformer trains an agent-day through the REAL worker path
    (env + rollout + PPO update + eval + registry) without error,
  * it runs on CUDA (the rope cos/sin buffers + rope_layers move to GPU),
  * the rope-vs-learned wall-clock overhead is small (rope adds only the
    Q/K rotation; the ctx^2 attention that dominates is unchanged).

The learned-vs-rope QUALITY comparison is the campaign's job (that is what
the structural gene is for) — this only proves rope is safe to sample.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from predictors import PredictorBundle  # noqa: E402
from training_v2.cohort.genes import CohortGenes  # noqa: E402
from training_v2.cohort.worker import train_one_agent  # noqa: E402


def bundle():
    base = REPO.parent / "betfair-predictors" / "production"
    return PredictorBundle.from_manifests(
        champion_manifest=base / "race-outcome" / "manifest.json",
        ranker_manifest=base / "race-outcome-ranker" / "manifest.json",
        direction_manifest=base / "direction-predictor" / "manifest.json",
    )


def tf_genes(pos: str):
    return CohortGenes(
        learning_rate=3e-4, entropy_coeff=0.01, clip_range=0.2, gae_lambda=0.95,
        value_coeff=0.5, mini_batch_size=64, hidden_size=256,
        architecture="transformer", transformer_depth=3, transformer_heads=8,
        transformer_ctx_ticks=256, transformer_pos_encoding=pos,
    )


def run(pos: str):
    b = bundle()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    res = train_one_agent(
        agent_id=f"smoke-rope-{pos}", genes=tf_genes(pos),
        days_to_train=["2026-04-10"], eval_days=["2026-04-10"],
        data_dir=REPO / "data" / "processed", device="cpu", seed=0,
        predictor_bundle=b, use_race_outcome_predictor=True,
        use_direction_predictor=True, predictor_lean_obs=True,
        strategy_mode="arb", gpu_policy_lane=True,
    )
    wall = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() / 1e6
            if torch.cuda.is_available() else 0.0)
    print(f"  pos={pos:8s} wall={wall:6.1f}s  peak_gpu_mem={peak:7.1f}MB  "
          f"arch={res.architecture_name}  eval_pnl={res.eval.day_pnl:+.2f}",
          flush=True)
    return wall


print(f"cuda={torch.cuda.is_available()} "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
print("ctx256/depth3 transformer on the GPU lane, one day (2026-04-10):")
learned = run("learned")
rope = run("rope")
print(f"\nrope/learned wall ratio: {rope / learned:.2f}x "
      f"(expect ~1.0-1.2x — rope adds only the Q/K rotation)")
