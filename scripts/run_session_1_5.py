"""
Session 1.5 -- End-to-end single agent run.

Trains one agent (population=1, generation=1) on the chronological training
split (earliest ~50% of days) and evaluates on the test split (later ~50%).
Writes results to registry, prints scoreboard, bet log summary, and per-day
P&L.

Requires 2+ days of extracted data in data/processed/.

Usage:
    python scripts/run_session_1_5.py
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
from registry.scoreboard import Scoreboard
from training.run_training import TrainingOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def find_available_dates(data_dir: str = "data/processed") -> list[str]:
    """Find all extracted dates with both ticks and runner Parquet files."""
    processed = Path(data_dir)
    dates = set()
    for f in processed.glob("*_runners.parquet"):
        date_str = f.name.replace("_runners.parquet", "")
        if (processed / f"{date_str}.parquet").exists():
            dates.add(date_str)
    return sorted(dates)


def main():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Single agent: population=1, 1 generation, no selection/breeding
    config["population"]["size"] = 1
    config["population"]["n_elite"] = 1
    # Allow CPU fallback — this session is a functional test, not a perf run
    config["training"]["require_gpu"] = False

    n_generations = 1
    n_epochs = 3
    seed = 42

    # ---- Find and load data ------------------------------------------------

    dates = find_available_dates(config["paths"]["processed_data"])
    if len(dates) < 2:
        logger.error(
            "Need 2+ days for a train/test split. Found %d: %s",
            len(dates), dates,
        )
        sys.exit(1)

    logger.info("Available dates: %s", dates)

    logger.info("Loading episodes...")
    days = load_days(dates, data_dir=config["paths"]["processed_data"])
    logger.info(
        "Loaded %d day(s) with %d total races",
        len(days), sum(len(d.races) for d in days),
    )

    # Chronological train/test split (~50/50)
    split = len(days) // 2
    train_days = days[:split]
    test_days = days[split:]

    logger.info(
        "Train split: %d day(s) [%s]",
        len(train_days), ", ".join(d.date for d in train_days),
    )
    logger.info(
        "Test split:  %d day(s) [%s]",
        len(test_days), ", ".join(d.date for d in test_days),
    )

    # ---- Registry setup -----------------------------------------------------

    session_db = "registry/session_1_5.db"
    session_weights = "registry/session_1_5_weights"
    store = ModelStore(db_path=session_db, weights_dir=session_weights)
    queue: asyncio.Queue = asyncio.Queue()

    # ---- Train + Evaluate ---------------------------------------------------

    logger.info(
        "Starting: 1 agent, %d epoch(s), %d train day(s), %d test day(s)",
        n_epochs, len(train_days), len(test_days),
    )
    start = time.time()

    orch = TrainingOrchestrator(
        config,
        model_store=store,
        progress_queue=queue,
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

    # ---- Drain progress events ---------------------------------------------

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())

    phase_starts = [e for e in events if e.get("event") == "phase_start"]
    phase_completes = [e for e in events if e.get("event") == "phase_complete"]
    progress_events = [e for e in events if e.get("event") == "progress"]

    # ---- Print summary -----------------------------------------------------

    print("\n" + "=" * 70)
    print("SESSION 1.5 -- END-TO-END SINGLE AGENT RUN")
    print("=" * 70)
    print(f"Epochs:         {n_epochs}")
    print(f"Train days:     {len(train_days)} [{', '.join(d.date for d in train_days)}]")
    print(f"Test days:      {len(test_days)} [{', '.join(d.date for d in test_days)}]")
    print(f"Elapsed:        {elapsed:.1f}s")
    print(f"Events:         {len(events)} total "
          f"({len(phase_starts)} phase_start, "
          f"{len(phase_completes)} phase_complete, "
          f"{len(progress_events)} progress)")
    print()

    # ---- Agent training stats ----------------------------------------------

    gen_result = result.generations[0]
    model_id = list(gen_result.training_stats.keys())[0]
    stats = gen_result.training_stats[model_id]

    print("--- Training Stats ---")
    print(f"  Model ID:       {model_id[:12]}...")
    print(f"  Episodes:       {stats.episodes_completed}")
    print(f"  Total steps:    {stats.total_steps}")
    print(f"  Mean reward:    {stats.mean_reward:.4f}")
    print(f"  Mean P&L:       {stats.mean_pnl:.2f}")
    print(f"  Mean bet count: {stats.mean_bet_count:.1f}")
    print(f"  Final losses:   policy={stats.final_policy_loss:.6f}  "
          f"value={stats.final_value_loss:.6f}  "
          f"entropy={stats.final_entropy:.6f}")
    print()

    # ---- Scoreboard --------------------------------------------------------

    scoreboard = Scoreboard(store, config)
    rankings = scoreboard.rank_all()

    print("--- Scoreboard ---")
    if rankings:
        for i, s in enumerate(rankings):
            print(f"  #{i+1}  {s.model_id[:12]}  "
                  f"score={s.composite_score:.4f}  "
                  f"win_rate={s.win_rate:.2f}  "
                  f"sharpe={s.sharpe:.2f}  "
                  f"mean_pnl={s.mean_daily_pnl:.2f}  "
                  f"efficiency={s.efficiency:.2f}")
    else:
        print("  (empty -- no scored models)")
    print()

    # ---- Per-day P&L -------------------------------------------------------

    run = store.get_latest_evaluation_run(model_id)
    if run:
        day_records = store.get_evaluation_days(run.run_id)

        print("--- Per-Day P&L ---")
        for dr in day_records:
            flag = "+" if dr.profitable else "-"
            print(
                f"  [{flag}] {dr.date}  "
                f"pnl={dr.day_pnl:>10.2f}  "
                f"bets={dr.bet_count:>4d}  "
                f"winning={dr.winning_bets:>4d}  "
                f"precision={dr.bet_precision:.3f}  "
                f"pnl/bet={dr.pnl_per_bet:.3f}  "
                f"early={dr.early_picks}"
            )

        total_pnl = sum(dr.day_pnl for dr in day_records)
        total_bets = sum(dr.bet_count for dr in day_records)
        profitable_days = sum(1 for dr in day_records if dr.profitable)

        print()
        print(f"  Total P&L:       {total_pnl:.2f}")
        print(f"  Total bets:      {total_bets}")
        print(f"  Profitable days: {profitable_days}/{len(day_records)}")
        print()

        # ---- Bet log sample ------------------------------------------------

        bets = store.get_evaluation_bets(run.run_id)
        print(f"--- Bet Log ({len(bets)} total bets) ---")
        if bets:
            # Show first 10 and last 5
            sample = bets[:10]
            print("  First 10 bets:")
            for b in sample:
                print(
                    f"    {b.date} {b.market_id[:12]} "
                    f"{b.runner_name or b.runner_id}  "
                    f"{b.action:>4s} @{b.price:.2f}  "
                    f"stake={b.stake:.2f}  "
                    f"matched={b.matched_size:.2f}  "
                    f"{b.outcome:>4s}  "
                    f"pnl={b.pnl:>8.2f}"
                )

            if len(bets) > 15:
                print(f"  ... ({len(bets) - 15} more) ...")
                for b in bets[-5:]:
                    print(
                        f"    {b.date} {b.market_id[:12]} "
                        f"{b.runner_name or b.runner_id}  "
                        f"{b.action:>4s} @{b.price:.2f}  "
                        f"stake={b.stake:.2f}  "
                        f"matched={b.matched_size:.2f}  "
                        f"{b.outcome:>4s}  "
                        f"pnl={b.pnl:>8.2f}"
                    )
        print()

    else:
        print("  (no evaluation run found)")

    # ---- Sanity checks -----------------------------------------------------

    print("--- Sanity Checks ---")
    all_models = store.list_models()
    checks_passed = 0
    checks_total = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal checks_passed, checks_total
        checks_total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            checks_passed += 1
        suffix = f" ({detail})" if detail else ""
        print(f"  [{status}] {name}{suffix}")

    check("Model registered", len(all_models) == 1, f"{len(all_models)} model(s)")
    check("Model is active", all_models[0].status == "active" if all_models else False)
    check("Evaluation run exists", run is not None)

    if run:
        check(
            "Per-day metrics recorded",
            len(day_records) == len(test_days),
            f"{len(day_records)} days",
        )
        check("Bet log populated", len(bets) > 0, f"{len(bets)} bets")
        check(
            "Composite score computed",
            rankings and rankings[0].composite_score is not None,
            f"{rankings[0].composite_score:.4f}" if rankings else "N/A",
        )
        check("Scoreboard non-empty", len(rankings) > 0, f"{len(rankings)} model(s)")
        check(
            "P&L not exploding",
            all(abs(dr.day_pnl) < 10000 for dr in day_records),
            f"max abs pnl = {max(abs(dr.day_pnl) for dr in day_records):.2f}",
        )
        check(
            "Bets were matched",
            any(b.matched_size > 0 for b in bets) if bets else False,
        )
        check(
            "Reward signal sensible",
            abs(stats.mean_reward) < 1e6,
            f"mean_reward = {stats.mean_reward:.4f}",
        )

    print(f"\n  {checks_passed}/{checks_total} checks passed")
    print("=" * 70)

    # ---- Save summary ------------------------------------------------------

    summary_path = Path("logs/session_1_5_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_id": model_id,
        "n_epochs": n_epochs,
        "train_dates": [d.date for d in train_days],
        "test_dates": [d.date for d in test_days],
        "elapsed_seconds": round(elapsed, 1),
        "training": {
            "episodes": stats.episodes_completed,
            "total_steps": stats.total_steps,
            "mean_reward": round(stats.mean_reward, 4),
            "mean_pnl": round(stats.mean_pnl, 2),
            "mean_bet_count": round(stats.mean_bet_count, 1),
        },
        "evaluation": {
            "per_day": [
                {
                    "date": dr.date,
                    "day_pnl": round(dr.day_pnl, 2),
                    "bet_count": dr.bet_count,
                    "winning_bets": dr.winning_bets,
                    "bet_precision": round(dr.bet_precision, 3),
                    "profitable": dr.profitable,
                }
                for dr in day_records
            ] if run else [],
            "total_bets": len(bets) if run else 0,
        },
        "scoreboard": [
            {
                "model_id": s.model_id[:12],
                "composite_score": round(s.composite_score, 4),
                "win_rate": round(s.win_rate, 2),
                "sharpe": round(s.sharpe, 2),
                "mean_daily_pnl": round(s.mean_daily_pnl, 2),
                "efficiency": round(s.efficiency, 2),
            }
            for s in rankings
        ],
        "sanity_checks_passed": checks_passed,
        "sanity_checks_total": checks_total,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
