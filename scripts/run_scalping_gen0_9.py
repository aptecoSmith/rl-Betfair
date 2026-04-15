"""
scripts/run_scalping_gen0_9.py — 9-model, 1-generation scalping run.

Validates the corrected scalping reward path landed in commit
c218bfb (asymmetric hedge sizing + worst-case locked_pnl).

- population.size: 9
- n_generations: 1
- starting_budget: £100/race
- scalping_mode: True
- train/test split: 4 / 4 days (chronological)

Writes to the **production registry** (registry/models.db) so the
Bet Explorer picks up the new models alongside existing Gen 0s.

Usage::

    python scripts/run_scalping_gen0_9.py

Post-run: open Bet Explorer, pick one of the new models, check the
Locked / Neutral / Directional / Naked counter bar per
plans/scalping-asymmetric-hedging/progress.md.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

import yaml

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


def find_available_dates(data_dir: str) -> list[str]:
    processed = Path(data_dir)
    dates = set()
    for f in processed.glob("*_runners.parquet"):
        date_str = f.name.replace("_runners.parquet", "")
        if (processed / f"{date_str}.parquet").exists():
            dates.add(date_str)
    return sorted(dates)


def main() -> int:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ── Scalping validation run settings ────────────────────────────
    config["population"]["size"] = 9
    config["population"]["n_elite"] = 2
    config["training"]["starting_budget"] = 100.0
    config["training"]["scalping_mode"] = True
    # Give the new reward path actual weight — 0.0 defaults make the
    # scalping branch a no-op. Values mirror the hp search_ranges
    # midpoints so the initial population is in a reasonable basin.
    config["reward"]["naked_penalty_weight"] = 1.0
    config["reward"]["early_lock_bonus_weight"] = 0.5

    n_generations = 1
    n_epochs = 3
    seed = 42

    # ── Data ───────────────────────────────────────────────────────
    dates = find_available_dates(config["paths"]["processed_data"])
    if len(dates) < 2:
        logger.error("Need at least 2 extracted days, found %d", len(dates))
        return 1

    logger.info("Available dates: %s", dates)
    days = load_days(dates, data_dir=config["paths"]["processed_data"])
    logger.info("Loaded %d day(s), %d races total",
                len(days), sum(len(d.races) for d in days))

    split = len(days) // 2
    train_days = days[:split]
    test_days = days[split:]
    logger.info("Train: %d day(s) [%s]",
                len(train_days), ", ".join(d.date for d in train_days))
    logger.info("Test:  %d day(s) [%s]",
                len(test_days), ", ".join(d.date for d in test_days))

    # ── Registry — production DB so Bet Explorer picks up the models
    store = ModelStore(
        db_path=config["paths"]["registry_db"],
        weights_dir=config["paths"]["model_weights"],
    )
    queue: asyncio.Queue = asyncio.Queue()

    logger.info(
        "Starting scalping Gen 0 run: pop=%d, epochs=%d, budget=£%.0f",
        config["population"]["size"], n_epochs,
        config["training"]["starting_budget"],
    )
    start = time.time()

    orch = TrainingOrchestrator(
        config, model_store=store, progress_queue=queue,
    )
    result = orch.run(
        train_days=train_days,
        test_days=test_days,
        n_generations=n_generations,
        n_epochs=n_epochs,
        seed=seed,
    )

    elapsed = time.time() - start
    logger.info("Run completed in %.1f s", elapsed)

    # ── Summary ────────────────────────────────────────────────────
    gen = result.generations[0]
    print("\n" + "=" * 70)
    print("SCALPING GEN 0 × 9 AGENTS — RESULTS")
    print("=" * 70)
    print(f"Elapsed:      {elapsed:.1f} s")
    print(f"Train days:   {len(train_days)}")
    print(f"Test days:    {len(test_days)}")
    print(f"Models saved: {len(gen.training_stats)}")
    print()

    scoreboard = Scoreboard(store, config)
    rankings = scoreboard.rank_all()
    print("--- Scoreboard (new models only) ---")
    new_ids = set(gen.training_stats.keys())
    for i, entry in enumerate(r for r in rankings if r.model_id in new_ids):
        print(f"  {i+1:>2}. {entry.model_id[:8]}  "
              f"score={entry.composite_score:+.4f}  "
              f"pnl={entry.mean_daily_pnl:+.2f}  "
              f"bets={entry.total_bets}")
    print()
    print("Next: open the Bet Explorer, pick one of the new models, "
          "and follow the post-training checklist at the top of "
          "plans/scalping-asymmetric-hedging/progress.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
