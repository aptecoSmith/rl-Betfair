"""
scripts/run_scalping_gen1.py — Pop-16, 4-generation scalping run.

Builds on the Gen 0 validation run (commit 93e1a48) that confirmed
the asymmetric-sizing / worst-case-locked_pnl fix works:
top performer ab460eb9 locked +£43 on 131 bets.

This run gives the GA actual selection pressure:
- population.size: 16 (up from 9 — more diversity, 3 selection events)
- n_generations: 4 (0 → 1 → 2 → 3 selections)
- n_epochs: 3 (matches Gen 0 run for a clean comparison)
- starting_budget: £100/race
- scalping_mode: pinned True (not a gene — this run is scalping-only)
- naked_penalty_weight: 0.5 (compromise between the Gen 0 "too harsh"
  collapse (5/9 agents sat out) and "too soft" runaway naked losses)
- early_lock_bonus_weight: 1.0 (bumped from Gen 0's 0.5 — pull
  completions forward; Gen 0 ratio was 1037 naked : 142 completed
  on ef3ce98d, the worst case)

Estimated runtime: ~16 hours on the current box (Gen 0 baseline:
9 × 1 × 3 = 27 slots, ~45 min; this run = 16 × 4 × 3 = 192 slots
≈ 16 h at roughly 5 min / slot plus eval overhead).

Writes to the production registry so the Bet Explorer picks up
the new models alongside existing Gen 0s.

Usage::

    python scripts/run_scalping_gen1.py

Post-run: open the Scoreboard, check the new Gen 1–3 models.
Open Bet Explorer for the top model; Locked / Neutral /
Directional / Naked counters should show a higher locked ratio
than Gen 0's +£43 best. Full checklist in
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

    # ── Run spec ────────────────────────────────────────────────────
    config["population"]["size"] = 16
    config["population"]["n_elite"] = 3
    # Shrink mutation a touch from default 0.3 → 0.2 so the GA does
    # more exploitation of what worked in Gen 0 (selective bettors
    # like ab460eb9) and less wild exploration.
    config["population"]["mutation_rate"] = 0.2

    config["training"]["starting_budget"] = 100.0
    config["training"]["scalping_mode"] = True

    # Scalping reward weights — static across gens (see script
    # docstring for rationale). naked_penalty_weight = 0.5 is the
    # compromise between Gen 0's 1.0 (5/9 agents collapsed to
    # no-bet) and 0.0 (naked losses have no teeth, agent learns
    # to stake huge unhedged positions and get lucky).
    config["reward"]["naked_penalty_weight"] = 0.5
    config["reward"]["early_lock_bonus_weight"] = 1.0

    # Pin scalping_mode OUT of the gene pool for this run so the
    # GA can't silently evolve pure directional agents that beat
    # scalpers on raw P&L. We want to breed better scalpers, not
    # abandon the strategy.
    search_ranges = config["hyperparameters"]["search_ranges"]
    if "scalping_mode" in search_ranges:
        search_ranges.pop("scalping_mode")
        logger.info("Removed scalping_mode from gene pool — pinned True for all agents")

    n_generations = 4
    n_epochs = 3
    seed = 42

    # ── Data ───────────────────────────────────────────────────────
    dates = find_available_dates(config["paths"]["processed_data"])
    if len(dates) < 2:
        logger.error("Need ≥ 2 extracted days, found %d", len(dates))
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

    # ── Registry — production DB
    store = ModelStore(
        db_path=config["paths"]["registry_db"],
        weights_dir=config["paths"]["model_weights"],
    )
    queue: asyncio.Queue = asyncio.Queue()

    logger.info(
        "Scalping run: pop=%d, gens=%d, epochs=%d, budget=£%.0f, "
        "naked_penalty=%.2f, early_lock_bonus=%.2f",
        config["population"]["size"], n_generations, n_epochs,
        config["training"]["starting_budget"],
        config["reward"]["naked_penalty_weight"],
        config["reward"]["early_lock_bonus_weight"],
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
    logger.info("Run completed in %.1f s (%.1f h)", elapsed, elapsed / 3600)

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SCALPING GEN 1 — 16 AGENTS × 4 GENERATIONS × 3 EPOCHS")
    print("=" * 72)
    print(f"Elapsed:          {elapsed:.1f} s  ({elapsed / 3600:.1f} h)")
    print(f"Train days:       {len(train_days)}")
    print(f"Test days:        {len(test_days)}")
    print(f"Generations:      {len(result.generations)}")
    print()

    # Per-generation scoreboard digest
    all_new_ids: set[str] = set()
    for g in result.generations:
        ids = set(g.training_stats.keys())
        all_new_ids.update(ids)
        print(f"--- Generation {g.generation} — {len(ids)} agents trained ---")
        for mid, stats in list(g.training_stats.items())[:3]:
            print(f"  {mid[:8]}  reward={stats.mean_reward:+.3f}  "
                  f"pnl={stats.mean_pnl:+.2f}  bets={stats.mean_bet_count:.1f}")
        print()

    # Final scoreboard
    scoreboard = Scoreboard(store, config)
    rankings = scoreboard.rank_all()
    print("--- Final scoreboard (top 10, new models only) ---")
    for i, entry in enumerate(
        [r for r in rankings if r.model_id in all_new_ids][:10]
    ):
        print(f"  {i+1:>2}. {entry.model_id[:8]}  gen={entry.generation}  "
              f"score={entry.composite_score:+.4f}  "
              f"pnl={entry.mean_daily_pnl:+.2f}  "
              f"bets={entry.total_bets}")

    print()
    print("Next: open Bet Explorer, pick a top-ranked new model, and")
    print("check the Locked / Neutral / Directional / Naked counter bar.")
    print("Full post-run checklist: plans/scalping-asymmetric-hedging/progress.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
