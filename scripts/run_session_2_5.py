"""
Session 2.5 — First multi-generation run.

Runs the full training pipeline on available real data to verify the entire
system works end-to-end: population init → train → evaluate → score →
select → breed → repeat for N generations.

With only 1 day of data, the same day is used for both training and
evaluation (scores will be optimistic — expected and documented).

Usage:
    python scripts/run_session_2_5.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.episode_builder import load_days
from registry.model_store import ModelStore
from training.run_training import TrainingOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def find_available_dates(data_dir: str = "data/processed") -> list[str]:
    """Find all extracted dates."""
    processed = Path(data_dir)
    dates = set()
    for f in processed.glob("*_runners.parquet"):
        date_str = f.name.replace("_runners.parquet", "")
        # Verify ticks file also exists
        if (processed / f"{date_str}.parquet").exists():
            dates.add(date_str)
    return sorted(dates)


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Override for this experiment: small population, fast
    config["population"]["size"] = 6
    config["population"]["n_elite"] = 2

    n_generations = 3
    n_epochs = 2
    seed = 42

    # Find data
    dates = find_available_dates(config["paths"]["processed_data"])
    if not dates:
        logger.error("No extracted data found in %s", config["paths"]["processed_data"])
        sys.exit(1)

    logger.info("Available dates: %s", dates)

    # Load days
    logger.info("Loading episodes...")
    days = load_days(dates, data_dir=config["paths"]["processed_data"])
    logger.info("Loaded %d day(s) with %d total races",
                len(days), sum(len(d.races) for d in days))

    # Chronological split
    if len(days) >= 2:
        split = len(days) // 2
        train_days = days[:split]
        test_days = days[split:]
    else:
        logger.warning(
            "Only %d day(s) available — using same data for train and test. "
            "Scores will be optimistic.", len(days),
        )
        train_days = days
        test_days = []  # orchestrator will handle this

    logger.info("Train days: %d, Test days: %d",
                len(train_days), len(test_days))

    # Set up registry
    db_path = config["paths"]["registry_db"]
    weights_dir = config["paths"]["model_weights"]

    # Use a session-specific DB to avoid polluting the main one
    session_db = "registry/session_2_5.db"
    session_weights = "registry/session_2_5_weights"
    store = ModelStore(db_path=session_db, weights_dir=session_weights)

    # Progress queue (drain events to stdout)
    queue = asyncio.Queue()

    # Run
    logger.info(
        "Starting %d generations, population=%d, epochs=%d",
        n_generations, config["population"]["size"], n_epochs,
    )
    start = time.time()

    orch = TrainingOrchestrator(
        config,
        model_store=store,
        progress_queue=queue,
        device="cpu",
    )

    result = orch.run(
        train_days=train_days,
        test_days=test_days,
        n_generations=n_generations,
        n_epochs=n_epochs,
        seed=seed,
    )

    elapsed = time.time() - start
    logger.info("Completed in %.1f seconds", elapsed)

    # Drain and count events
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())

    phase_starts = [e for e in events if e["event"] == "phase_start"]
    phase_completes = [e for e in events if e["event"] == "phase_complete"]
    progress_events = [e for e in events if e["event"] == "progress"]

    logger.info("Events: %d phase_start, %d phase_complete, %d progress",
                len(phase_starts), len(phase_completes), len(progress_events))

    # Print summary
    print("\n" + "=" * 70)
    print("SESSION 2.5 — MULTI-GENERATION RUN SUMMARY")
    print("=" * 70)
    print(f"Generations:    {n_generations}")
    print(f"Population:     {config['population']['size']}")
    print(f"Train epochs:   {n_epochs}")
    print(f"Train days:     {len(train_days)}")
    print(f"Test days:      {len(test_days)} {'(fallback to train)' if not test_days else ''}")
    print(f"Elapsed:        {elapsed:.1f}s")
    print(f"Total events:   {len(events)}")
    print()

    # Per-generation summary
    for gen_result in result.generations:
        gen = gen_result.generation
        n_agents = len(gen_result.training_stats)
        top_score = gen_result.scores[0] if gen_result.scores else None

        print(f"--- Generation {gen} ---")
        print(f"  Agents trained:  {n_agents}")
        print(f"  Models scored:   {len(gen_result.scores)}")

        if top_score:
            print(f"  Top score:       {top_score.composite_score:.4f} ({top_score.model_id[:12]})")
            print(f"    win_rate={top_score.win_rate:.2f}  sharpe={top_score.sharpe:.2f}  "
                  f"mean_pnl={top_score.mean_daily_pnl:.2f}  efficiency={top_score.efficiency:.2f}")

        if gen_result.selection:
            print(f"  Survivors:       {len(gen_result.selection.survivors)}")
            print(f"  Eliminated:      {len(gen_result.selection.eliminated)}")
            print(f"  Children bred:   {len(gen_result.children)}")

        if gen_result.discarded:
            print(f"  Discarded:       {len(gen_result.discarded)}")

        # Training stats summary
        mean_rewards = [s.mean_reward for s in gen_result.training_stats.values()]
        mean_pnls = [s.mean_pnl for s in gen_result.training_stats.values()]
        if mean_rewards:
            import numpy as np
            print(f"  Training reward: mean={np.mean(mean_rewards):.3f}  "
                  f"std={np.std(mean_rewards):.3f}")
            print(f"  Training P&L:    mean={np.mean(mean_pnls):.2f}  "
                  f"std={np.std(mean_pnls):.2f}")
        print()

    # Final rankings
    print("--- Final Scoreboard ---")
    for i, s in enumerate(result.final_rankings[:10]):
        print(f"  #{i+1:2d}  {s.model_id[:12]}  score={s.composite_score:.4f}  "
              f"wr={s.win_rate:.2f}  sharpe={s.sharpe:.2f}  "
              f"pnl={s.mean_daily_pnl:.2f}  eff={s.efficiency:.2f}")
    print()

    # Registry summary
    all_models = store.list_models()
    active = [m for m in all_models if m.status == "active"]
    discarded = [m for m in all_models if m.status == "discarded"]
    print(f"Registry: {len(all_models)} total models "
          f"({len(active)} active, {len(discarded)} discarded)")

    # Genetic events
    gen0_events = store.get_genetic_events(generation=0)
    print(f"Genetic events (gen 0): {len(gen0_events)}")

    # Check genetic log files
    genetics_dir = Path(config["paths"]["logs"]) / "genetics"
    if genetics_dir.exists():
        log_files = sorted(genetics_dir.glob("gen_*.log"))
        print(f"Genetic log files: {len(log_files)}")
        for f in log_files:
            print(f"  {f.name} ({f.stat().st_size} bytes)")

    # Evaluation coverage
    models_with_eval = 0
    for m in all_models:
        run = store.get_latest_evaluation_run(m.model_id)
        if run is not None:
            models_with_eval += 1
    print(f"Models with evaluations: {models_with_eval}/{len(all_models)}")

    # Bet log coverage
    total_bets = 0
    for m in active:
        run = store.get_latest_evaluation_run(m.model_id)
        if run:
            bets = store.get_evaluation_bets(run.run_id)
            total_bets += len(bets)
    print(f"Total evaluation bets (active models): {total_bets}")

    print("\n" + "=" * 70)

    # Save summary to file
    summary_path = Path("logs/session_2_5_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_generations": n_generations,
        "population_size": config["population"]["size"],
        "n_epochs": n_epochs,
        "train_dates": [d.date for d in train_days],
        "test_dates": [d.date for d in test_days] if test_days else ["(fallback to train)"],
        "elapsed_seconds": round(elapsed, 1),
        "total_events": len(events),
        "total_models": len(all_models),
        "active_models": len(active),
        "discarded_models": len(discarded),
        "models_with_evaluations": models_with_eval,
        "total_evaluation_bets": total_bets,
        "final_rankings": [
            {
                "model_id": s.model_id[:12],
                "composite_score": round(s.composite_score, 4),
                "win_rate": round(s.win_rate, 2),
                "sharpe": round(s.sharpe, 2),
                "mean_daily_pnl": round(s.mean_daily_pnl, 2),
                "efficiency": round(s.efficiency, 2),
            }
            for s in result.final_rankings
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
