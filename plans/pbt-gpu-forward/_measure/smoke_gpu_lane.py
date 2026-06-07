"""End-to-end smoke for the GPU policy lane: train ONE ctx256 transformer
agent for one day with the lane ON vs OFF, confirm it completes and is faster
on GPU. Also samples GPU memory mid-run to prove the policy is actually on CUDA.
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


def tf_genes():
    return CohortGenes(
        learning_rate=3e-4, entropy_coeff=0.01, clip_range=0.2, gae_lambda=0.95,
        value_coeff=0.5, mini_batch_size=64, hidden_size=256,
        architecture="transformer", transformer_depth=3, transformer_heads=8,
        transformer_ctx_ticks=256,
    )


def run(lane: bool):
    b = bundle()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    res = train_one_agent(
        agent_id=f"smoke-tf-{'gpu' if lane else 'cpu'}", genes=tf_genes(),
        days_to_train=["2026-04-10"], eval_days=["2026-04-10"],
        data_dir=REPO / "data" / "processed", device="cpu", seed=0,
        predictor_bundle=b, use_race_outcome_predictor=True,
        use_direction_predictor=True, predictor_lean_obs=True,
        strategy_mode="arb", gpu_policy_lane=lane,
    )
    wall = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() / 1e6
            if torch.cuda.is_available() else 0.0)
    print(f"  lane={'ON ' if lane else 'OFF'} "
          f"wall={wall:6.1f}s  peak_gpu_mem={peak:7.1f}MB  "
          f"arch={res.architecture_name}", flush=True)
    return wall


print(f"cuda={torch.cuda.is_available()} "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
print("ctx256/depth3 transformer, one day (2026-04-10):")
gpu = run(True)
cpu = run(False)
print(f"\nspeedup (CPU/GPU agent-day): {cpu / gpu:.2f}x"
      if gpu else "GPU run failed")
