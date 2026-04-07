"""
Session 9 — Full Gen-0 GPU shakeout.

Drives one end-to-end Gen-0 training run under the planner:

- Builds a :class:`training.training_plan.TrainingPlan` targeting
  population 21 (7 per architecture), every new gene from Sessions 1-7
  active via ``config.yaml`` defaults, arch-specific transformer LR
  override, and a fixed RNG seed so the run is reproducible.
- Persists the plan via :class:`PlanRegistry` so the UI can see it
  after the fact (and so ``_record_plan_outcome`` in the orchestrator
  has somewhere to land its ``GenerationOutcome``).
- Runs **one** generation via :class:`TrainingOrchestrator`. This is a
  shakeout, not an evolution run — we only want Gen-0 data to verify
  the exploration infrastructure produces diverse, honest agents.
- Writes a summary JSON under ``logs/session_9/`` that the companion
  analysis script (:mod:`scripts.session_9_analysis`) reads to verify
  the five post-run invariants listed in
  ``plans/arch-exploration/session_9_gpu_shakeout.md``.

This script is **not** a CPU-only unit test. It must only be run when
the task is to actually execute Session 9. Running it will:

- Read the processed data directory and do real PPO updates on real
  market data.
- Persist a plan file under ``registry/training_plans/`` (gitignored).
- Persist model weights under ``registry/session_9_weights/``
  (gitignored).
- Persist a bet log per agent/day under ``registry/session_9_bet_logs/``
  (gitignored).
- Write episode-level JSONL under ``logs/session_9/training/``
  (gitignored).

Usage::

    python scripts/session_9_shakeout.py                # run the shakeout
    python scripts/session_9_shakeout.py --train-days 4 --test-days 2
    python scripts/session_9_shakeout.py --dry-run      # build/validate plan only

The script expects to be launched from the repo root (or anywhere — it
resolves paths relative to its own location).
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import sys
import time
import uuid
from pathlib import Path

import yaml

# Allow "python scripts/session_9_shakeout.py" from the repo root.
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.episode_builder import load_days  # noqa: E402
from registry.model_store import ModelStore  # noqa: E402
from training.run_training import TrainingOrchestrator  # noqa: E402
from training.training_plan import (  # noqa: E402
    PlanRegistry,
    TrainingPlan,
    is_launchable,
    validate_plan,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("session_9")


# ── Plan constants ──────────────────────────────────────────────────────────

#: Architectures to exercise. All three must be present or the shakeout
#: doesn't prove anything about coverage.
ARCHITECTURES = [
    "ppo_lstm_v1",
    "ppo_time_lstm_v1",
    "ppo_transformer_v1",
]

#: 7 agents per arch × 3 arches == 21 (the session plan's minimum).
AGENTS_PER_ARCH = 7
POPULATION_SIZE = AGENTS_PER_ARCH * len(ARCHITECTURES)

#: Fixed seed so the run is reproducible and so two shakeouts can be
#: compared like-for-like.
SHAKEOUT_SEED = 20260407

#: Transformers prefer a lower learning-rate distribution than LSTMs.
#: Session 6 added ``TrainingPlan.arch_lr_ranges`` for exactly this
#: case. Range picked to sit entirely below the global float_log range
#: ``[1e-5, 5e-4]`` — a separate distribution, not just a narrower one.
TRANSFORMER_LR_RANGE = {"type": "float_log", "min": 5e-6, "max": 1e-4}


# ── Helpers ─────────────────────────────────────────────────────────────────


def find_available_dates(data_dir: Path) -> list[str]:
    """Return sorted list of dates with matched `.parquet` + `_runners.parquet`."""
    dates: set[str] = set()
    for f in data_dir.glob("*_runners.parquet"):
        date_str = f.name.replace("_runners.parquet", "")
        if (data_dir / f"{date_str}.parquet").exists():
            dates.add(date_str)
    return sorted(dates)


def build_plan(
    *,
    notes: str = "",
) -> TrainingPlan:
    """Construct the Session-9 training plan.

    ``hp_ranges`` is intentionally left empty so that
    :meth:`PopulationManager.initialise_population` falls back to the
    full ``config.yaml`` ``search_ranges`` — which already contains every
    gene added in Sessions 1-7 (reward shaping, PPO schema, LSTM
    structural, transformer, drawdown).
    """
    arch_mix = {arch: AGENTS_PER_ARCH for arch in ARCHITECTURES}
    return TrainingPlan.new(
        name="session_9_shakeout",
        population_size=POPULATION_SIZE,
        architectures=ARCHITECTURES,
        hp_ranges={},
        seed=SHAKEOUT_SEED,
        arch_mix=arch_mix,
        min_arch_samples=5,
        notes=notes or (
            "Session 9 full Gen-0 shakeout: 7 agents per architecture, "
            "all Sessions 1-7 genes active via config.yaml defaults, "
            "transformer LR override, fixed RNG seed, one generation."
        ),
        arch_lr_ranges={"ppo_transformer_v1": TRANSFORMER_LR_RANGE},
    )


def patch_config_for_shakeout(
    base_config: dict,
    *,
    session_tag: str,
) -> dict:
    """Return a deep-copy of ``base_config`` with session-scoped paths.

    Isolates the shakeout from any existing ``registry/models.db``,
    weights, bet logs or training episode logs so that the analysis
    script can read clean files without having to filter.
    """
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("paths", {})
    cfg["paths"]["logs"] = f"logs/{session_tag}"
    cfg["paths"]["registry_db"] = f"registry/{session_tag}.db"
    cfg["paths"]["model_weights"] = f"registry/{session_tag}_weights"
    cfg.setdefault("population", {})["size"] = POPULATION_SIZE
    cfg["population"]["n_elite"] = max(1, POPULATION_SIZE // 10)
    cfg.setdefault("training", {})
    # require_gpu=True in config.yaml is fine — the shakeout needs the
    # GPU — but we still let the orchestrator's own auto-detection raise
    # if CUDA isn't available.
    return cfg


def write_summary(
    *,
    summary_path: Path,
    plan: TrainingPlan,
    train_dates: list[str],
    test_dates: list[str],
    agents_snapshot: list[dict],
    training_stats: dict,
    elapsed_seconds: float,
    model_store_path: str,
    log_dir: str,
    errors: list[str],
) -> None:
    """Persist everything the analysis script needs into a single JSON file."""
    summary = {
        "plan_id": plan.plan_id,
        "plan_name": plan.name,
        "plan_seed": plan.seed,
        "population_size": plan.population_size,
        "architectures": plan.architectures,
        "arch_mix": plan.arch_mix,
        "arch_lr_ranges": plan.arch_lr_ranges,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "model_store_db": model_store_path,
        "log_dir": log_dir,
        "errors": errors,
        # Gen-0 agents as built by the planner: one entry per slot.
        "agents": agents_snapshot,
        # Per-agent aggregate training stats returned by the orchestrator.
        "training_stats": training_stats,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Summary written to %s", summary_path)


# ── Main ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 9 Gen-0 shakeout driver")
    p.add_argument(
        "--train-days", type=int, default=4,
        help="How many earliest dates to use as the training split.",
    )
    p.add_argument(
        "--test-days", type=int, default=2,
        help="How many latest dates to use as the evaluation split.",
    )
    p.add_argument(
        "--n-epochs", type=int, default=1,
        help="PPO training epochs per day (default 1 for a shakeout).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Build and validate the plan, then exit without running PPO.",
    )
    p.add_argument(
        "--session-tag", default="session_9",
        help="Directory-name suffix for isolation (default: session_9).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ── Config + data ────────────────────────────────────────────────────
    config_path = REPO_ROOT / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    run_config = patch_config_for_shakeout(base_config, session_tag=args.session_tag)

    data_dir = Path(run_config["paths"]["processed_data"])
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir

    all_dates = find_available_dates(data_dir)
    if len(all_dates) < args.train_days + args.test_days:
        logger.error(
            "Need %d+%d=%d dates, found %d: %s",
            args.train_days, args.test_days,
            args.train_days + args.test_days, len(all_dates), all_dates,
        )
        return 1

    train_dates = all_dates[: args.train_days]
    test_dates = all_dates[args.train_days : args.train_days + args.test_days]
    logger.info("Train dates: %s", train_dates)
    logger.info("Test dates:  %s", test_dates)

    # ── Plan ─────────────────────────────────────────────────────────────
    plan = build_plan()
    issues = validate_plan(plan)
    for issue in issues:
        logger.log(
            logging.ERROR if issue.severity == "error" else logging.WARNING,
            "Plan issue [%s/%s]: %s", issue.severity, issue.code, issue.message,
        )
    if not is_launchable(issues):
        logger.error("Plan failed validation; aborting.")
        return 2

    plan_registry = PlanRegistry(REPO_ROOT / "registry" / "training_plans")
    plan_registry.save(plan)
    logger.info("Plan persisted: %s (%s)", plan.plan_id, plan.name)

    # ── Registry + summary paths ─────────────────────────────────────────
    store = ModelStore(
        db_path=run_config["paths"]["registry_db"],
        weights_dir=run_config["paths"]["model_weights"],
        bet_logs_dir=f"registry/{args.session_tag}_bet_logs",
    )

    summary_path = REPO_ROOT / "logs" / args.session_tag / "shakeout_summary.json"

    errors: list[str] = []

    if args.dry_run:
        logger.info("--dry-run: skipping PPO. Plan validated.")
        write_summary(
            summary_path=summary_path,
            plan=plan,
            train_dates=train_dates,
            test_dates=test_dates,
            agents_snapshot=[],
            training_stats={},
            elapsed_seconds=0.0,
            model_store_path=run_config["paths"]["registry_db"],
            log_dir=run_config["paths"]["logs"],
            errors=errors,
        )
        return 0

    # ── Load data ────────────────────────────────────────────────────────
    logger.info("Loading %d train days...", len(train_dates))
    train_days = load_days(train_dates, data_dir=str(data_dir))
    logger.info("Loading %d test days...", len(test_dates))
    test_days = load_days(test_dates, data_dir=str(data_dir))
    logger.info(
        "Loaded %d train + %d test day(s), %d total races",
        len(train_days), len(test_days),
        sum(len(d.races) for d in train_days) + sum(len(d.races) for d in test_days),
    )

    # ── Launch ───────────────────────────────────────────────────────────
    queue: asyncio.Queue = asyncio.Queue()
    orch = TrainingOrchestrator(
        config=run_config,
        model_store=store,
        progress_queue=queue,
        training_plan=plan,
        plan_registry=plan_registry,
    )

    start = time.time()
    try:
        result = orch.run(
            train_days=train_days,
            test_days=test_days,
            n_generations=1,
            n_epochs=args.n_epochs,
            seed=SHAKEOUT_SEED,
        )
    except Exception as exc:  # noqa: BLE001
        # Session plan rule: log failures honestly rather than silently
        # retrying with different seeds. A shakeout that fails is a
        # successful shakeout — it tells you something the CPU tests
        # could not.
        logger.exception("Shakeout run raised an exception")
        errors.append(f"{type(exc).__name__}: {exc}")
        result = None
    elapsed = time.time() - start

    # ── Collect per-agent snapshots for analysis ─────────────────────────
    agents_snapshot: list[dict] = []
    training_stats_json: dict = {}
    if result is not None and result.generations:
        gen0 = result.generations[0]
        for model_id, stats in gen0.training_stats.items():
            record = store.get_model(model_id)
            arch_name = record.architecture_name if record else ""
            hp = dict(record.hyperparameters or {}) if record else {}
            agents_snapshot.append({
                "model_id": model_id,
                "architecture_name": arch_name,
                "hyperparameters": hp,
            })
            training_stats_json[model_id] = {
                "architecture_name": arch_name,
                "episodes_completed": stats.episodes_completed,
                "total_steps": stats.total_steps,
                "mean_reward": float(stats.mean_reward),
                "mean_pnl": float(stats.mean_pnl),
                "mean_bet_count": float(stats.mean_bet_count),
                "final_policy_loss": float(stats.final_policy_loss),
                "final_value_loss": float(stats.final_value_loss),
                "final_entropy": float(stats.final_entropy),
            }

    write_summary(
        summary_path=summary_path,
        plan=plan,
        train_dates=train_dates,
        test_dates=test_dates,
        agents_snapshot=agents_snapshot,
        training_stats=training_stats_json,
        elapsed_seconds=elapsed,
        model_store_path=run_config["paths"]["registry_db"],
        log_dir=run_config["paths"]["logs"],
        errors=errors,
    )

    logger.info(
        "Shakeout complete in %.1fs. Agents: %d. Errors: %d",
        elapsed, len(agents_snapshot), len(errors),
    )

    return 0 if not errors else 3


if __name__ == "__main__":
    sys.exit(main())
