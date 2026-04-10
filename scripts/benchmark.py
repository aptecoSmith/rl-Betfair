"""
scripts/benchmark.py -- Performance profiling benchmark.

Runs a single-agent pipeline (data loading → feature engineering → rollout →
PPO update → evaluation) and reports wall-clock timings for each phase.

Designed to be re-run before and after optimisation to measure improvement.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --output logs/bench_before.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.architecture_registry import create_policy
from agents.population_manager import PopulationManager
from agents.ppo_trainer import PPOTrainer
from data.episode_builder import load_day, load_days
from env.betfair_env import BetfairEnv, MARKET_DIM, VELOCITY_DIM, RUNNER_DIM, AGENT_STATE_DIM, ACTIONS_PER_RUNNER
from training.evaluator import Evaluator
from training.perf_log import gpu_memory_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark")


def find_available_dates(data_dir: str = "data/processed") -> list[str]:
    processed = Path(data_dir)
    dates = set()
    for f in processed.glob("*_runners.parquet"):
        date_str = f.name.replace("_runners.parquet", "")
        if (processed / f"{date_str}.parquet").exists():
            dates.add(date_str)
    return sorted(dates)


def benchmark_data_loading(dates: list[str], data_dir: str) -> tuple[list, float]:
    """Benchmark: load all days from Parquet."""
    t0 = time.perf_counter()
    days = load_days(dates, data_dir=data_dir)
    elapsed = time.perf_counter() - t0
    return days, elapsed


def benchmark_feature_engineering(days, config) -> tuple[dict, float]:
    """Benchmark: feature engineering for all days (cold cache)."""
    from data.feature_engineer import engineer_day

    feature_cache: dict[str, list] = {}
    t0 = time.perf_counter()
    for day in days:
        feature_cache[day.date] = engineer_day(day)
    elapsed = time.perf_counter() - t0
    return feature_cache, elapsed


def benchmark_rollout(day, config, policy, device, feature_cache) -> tuple[object, object, float]:
    """Benchmark: rollout collection for one day."""
    trainer = PPOTrainer(
        policy=policy,
        config=config,
        device=device,
        feature_cache=feature_cache,
    )
    t0 = time.perf_counter()
    rollout, ep_stats = trainer._collect_rollout(day)
    elapsed = time.perf_counter() - t0
    return rollout, ep_stats, elapsed


def benchmark_ppo_update(trainer, rollout) -> tuple[dict, float]:
    """Benchmark: PPO update on collected rollout."""
    t0 = time.perf_counter()
    loss_info = trainer._ppo_update(rollout)
    elapsed = time.perf_counter() - t0
    return loss_info, elapsed


def benchmark_evaluation(day, config, policy, device, feature_cache) -> tuple[object, float]:
    """Benchmark: evaluation on one day."""
    evaluator = Evaluator(
        config=config,
        model_store=None,
        device=device,
        feature_cache=feature_cache,
    )
    t0 = time.perf_counter()
    _, day_records = evaluator.evaluate(
        model_id="bench",
        policy=policy,
        test_days=[day],
        train_cutoff_date="2000-01-01",
    )
    elapsed = time.perf_counter() - t0
    return day_records, elapsed


def compare(before_path: str, after_path: str) -> None:
    """Print a comparison table between two benchmark runs."""
    before = json.loads(Path(before_path).read_text(encoding="utf-8"))
    after = json.loads(Path(after_path).read_text(encoding="utf-8"))

    print("\n" + "=" * 65)
    print("BENCHMARK COMPARISON")
    print("=" * 65)
    print(f"  Before: {before_path}")
    print(f"  After:  {after_path}")
    print()

    for key in before["timings"]:
        b = before["timings"][key]
        a = after["timings"][key]
        pct = (1 - a / b) * 100 if b > 0 else 0
        label = key.replace("_s", "").replace("_", " ").title()
        direction = "faster" if pct > 0 else "slower"
        print(f"  {label:25s}  {b:7.3f}s -> {a:7.3f}s  ({abs(pct):.1f}% {direction})")

    b_total = before["timings"]["total_train_eval_s"]
    a_total = after["timings"]["total_train_eval_s"]
    print()
    print(f"  Rollout throughput:      {before['metrics']['rollout_steps_per_s']}"
          f" -> {after['metrics']['rollout_steps_per_s']} steps/s")
    print(f"  Train+eval speedup:      {b_total / a_total:.2f}x")
    print(f"  Data loading speedup:    "
          f"{before['timings']['data_loading_s'] / after['timings']['data_loading_s']:.2f}x")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="rl-betfair performance benchmark")
    parser.add_argument(
        "--output", "-o",
        default="logs/benchmark_latest.json",
        help="Path to write results JSON",
    )
    parser.add_argument(
        "--compare", "-c",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        help="Compare two benchmark result files instead of running",
    )
    args = parser.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1])
        return

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Single agent config
    config["population"]["size"] = 1
    config["population"]["n_elite"] = 1
    config["training"]["require_gpu"] = False

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    if device == "cuda":
        logger.info(
            "GPU: %s (%.1f GB VRAM)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # ---- Phase 1: Data Loading ----
    dates = find_available_dates(config["paths"]["processed_data"])
    logger.info("Found %d dates: %s", len(dates), dates)

    days, load_time = benchmark_data_loading(dates, config["paths"]["processed_data"])
    total_ticks = sum(sum(len(r.ticks) for r in d.races) for d in days)
    total_races = sum(len(d.races) for d in days)
    logger.info(
        "Data loading: %.3fs (%d days, %d races, %d ticks)",
        load_time, len(days), total_races, total_ticks,
    )

    # ---- Phase 2: Feature Engineering ----
    feature_cache, feat_time = benchmark_feature_engineering(days, config)
    logger.info("Feature engineering: %.3fs (cold cache, all days)", feat_time)

    # ---- Phase 3: Create policy ----
    max_runners = config["training"]["max_runners"]
    obs_dim = MARKET_DIM + VELOCITY_DIM + (RUNNER_DIM * max_runners) + AGENT_STATE_DIM
    action_dim = max_runners * ACTIONS_PER_RUNNER
    hyperparams = {
        "learning_rate": 3e-4,
        "lstm_hidden_size": 256,
        "mlp_hidden_size": 128,
        "mlp_layers": 2,
        "architecture_name": "ppo_lstm_v1",
    }
    policy = create_policy("ppo_lstm_v1", obs_dim, action_dim, max_runners, hyperparams)
    policy = policy.to(device)
    logger.info("Policy created: %s on %s", "ppo_lstm_v1", device)

    # Use the biggest day for benchmarking
    bench_day = max(days, key=lambda d: sum(len(r.ticks) for r in d.races))
    bench_ticks = sum(len(r.ticks) for r in bench_day.races)
    logger.info(
        "Benchmark day: %s (%d races, %d ticks)",
        bench_day.date, len(bench_day.races), bench_ticks,
    )

    # ---- Phase 4: Rollout Collection ----
    trainer = PPOTrainer(
        policy=policy,
        config=config,
        device=device,
        feature_cache=feature_cache,
    )

    # Warm up GPU
    if device == "cuda":
        dummy = torch.randn(1, obs_dim, device=device)
        hidden = policy.init_hidden(1)
        hidden = (hidden[0].to(device), hidden[1].to(device))
        _ = policy(dummy, hidden)
        torch.cuda.synchronize()

    rollout, ep_stats, rollout_time = benchmark_rollout(
        bench_day, config, policy, device, feature_cache,
    )
    logger.info(
        "Rollout collection: %.3fs (%d steps, %.0f steps/s)",
        rollout_time, ep_stats.n_steps,
        ep_stats.n_steps / rollout_time if rollout_time > 0 else 0,
    )

    # ---- Phase 5: PPO Update ----
    if len(rollout) > 0:
        loss_info, ppo_time = benchmark_ppo_update(trainer, rollout)
        n_updates = trainer.ppo_epochs * max(1, len(rollout) // trainer.mini_batch_size)
        logger.info(
            "PPO update: %.3fs (%d transitions, ~%d mini-batch updates)",
            ppo_time, len(rollout), n_updates,
        )
    else:
        loss_info = {}
        ppo_time = 0.0
        n_updates = 0

    # ---- Phase 6: Evaluation ----
    day_records, eval_time = benchmark_evaluation(
        bench_day, config, policy, device, feature_cache,
    )
    logger.info("Evaluation: %.3fs (1 day)", eval_time)

    # ---- GPU Memory ----
    gpu_mem = gpu_memory_summary()
    if gpu_mem:
        logger.info(gpu_mem)

    # ---- Summary ----
    total_train_eval = rollout_time + ppo_time + eval_time
    results = {
        "timestamp": time.time(),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "dates": dates,
        "benchmark_day": bench_day.date,
        "total_races": total_races,
        "total_ticks": total_ticks,
        "bench_ticks": bench_ticks,
        "bench_races": len(bench_day.races),
        "timings": {
            "data_loading_s": round(load_time, 3),
            "feature_engineering_s": round(feat_time, 3),
            "rollout_collection_s": round(rollout_time, 3),
            "ppo_update_s": round(ppo_time, 3),
            "evaluation_s": round(eval_time, 3),
            "total_train_eval_s": round(total_train_eval, 3),
        },
        "metrics": {
            "rollout_steps": ep_stats.n_steps,
            "rollout_steps_per_s": round(ep_stats.n_steps / rollout_time, 1) if rollout_time > 0 else 0,
            "ppo_updates": n_updates,
            "rollout_pct_of_train_eval": round(rollout_time / total_train_eval * 100, 1) if total_train_eval > 0 else 0,
            "ppo_pct_of_train_eval": round(ppo_time / total_train_eval * 100, 1) if total_train_eval > 0 else 0,
            "eval_pct_of_train_eval": round(eval_time / total_train_eval * 100, 1) if total_train_eval > 0 else 0,
        },
        "gpu_memory": gpu_mem,
        "training_stats": {
            "total_reward": round(ep_stats.total_reward, 4),
            "total_pnl": round(ep_stats.total_pnl, 2),
            "bet_count": ep_stats.bet_count,
        },
    }

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Device:              {device}" + (f" ({results['gpu_name']})" if results['gpu_name'] else ""))
    print(f"  Benchmark day:       {bench_day.date} ({len(bench_day.races)} races, {bench_ticks} ticks)")
    print()
    print("  --- Timings ---")
    print(f"  Data loading:        {load_time:>8.3f}s")
    print(f"  Feature engineering: {feat_time:>8.3f}s")
    print(f"  Rollout collection:  {rollout_time:>8.3f}s  ({results['metrics']['rollout_pct_of_train_eval']}% of train+eval)")
    print(f"  PPO update:          {ppo_time:>8.3f}s  ({results['metrics']['ppo_pct_of_train_eval']}% of train+eval)")
    print(f"  Evaluation:          {eval_time:>8.3f}s  ({results['metrics']['eval_pct_of_train_eval']}% of train+eval)")
    print(f"  Total train+eval:    {total_train_eval:>8.3f}s")
    print()
    print(f"  Rollout:             {ep_stats.n_steps} steps @ {results['metrics']['rollout_steps_per_s']} steps/s")
    if gpu_mem:
        print(f"  GPU:                 {gpu_mem}")
    print("=" * 70)

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Results written to %s", out_path)

    return results


if __name__ == "__main__":
    main()
