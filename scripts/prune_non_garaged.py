"""Prune all non-garaged model artefacts from the registry.

Purpose
-------
After enough arch churn the registry accumulates models, evaluation
runs, bet-log parquets, training episodes, and genetic-event history
that can no longer be compared apples-to-apples with what we'd train
today. This script wipes everything **except** the hand-picked
garaged models and the empty training-plan drafts.

Design
------
- **Dry-run by default.** No touches unless ``--apply`` is passed.
- **Backup before delete.** The live ``registry/models.db`` + weights
  directory are copied to a timestamped archive dir first so an
  "oh-no" rollback is a manual copy rather than a restore from git.
- **Foreign-key-safe delete order**: evaluation_days → evaluation_runs
  → genetic_events → exploration_runs → models (weights paths collected
  before row deletion).
- **Filesystem cleanup** after SQL: weight files, bet-log dirs under
  ``logs/bet_logs/`` and ``registry/bet_logs/``, the global
  ``logs/training/episodes.jsonl``, and the scratch ``_eval_diag.log``.
- **Paused plan JSON deletion**: the ``Arb keep-betting GA`` plan in
  ``registry/training_plans/`` is removed; the four draft activation
  plans stay.

Usage
-----
    python scripts/prune_non_garaged.py           # dry-run
    python scripts/prune_non_garaged.py --apply   # actually delete
    python scripts/prune_non_garaged.py --apply --no-backup

The --no-backup flag exists for tests; production users should never
pass it.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# -- Constants ---------------------------------------------------------------

#: The paused plan that currently lives in the registry. Hard-coded
#: rather than detected by name so a future rename doesn't mis-delete
#: a different plan.
PAUSED_PLAN_ID = "179d3e9c-2ab3-40aa-b16c-147f3a339e02"


# -- Types -------------------------------------------------------------------


@dataclass
class PruneInventory:
    """The full set of things the script would touch, for reporting."""

    garaged_model_ids: list[str] = field(default_factory=list)
    keep_run_ids: list[str] = field(default_factory=list)
    delete_model_ids: list[str] = field(default_factory=list)
    delete_weight_paths: list[Path] = field(default_factory=list)
    delete_run_ids: list[str] = field(default_factory=list)
    delete_bet_log_dirs: list[Path] = field(default_factory=list)
    delete_training_plan_files: list[Path] = field(default_factory=list)
    delete_episodes_path: Path | None = None
    delete_eval_diag_path: Path | None = None
    legacy_bet_log_root: Path | None = None
    # counts carved from the db for the summary table
    pre_counts: dict[str, int] = field(default_factory=dict)


# -- Core --------------------------------------------------------------------


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"Registry DB not found: {db_path}")
    c = sqlite3.connect(db_path)
    c.execute("PRAGMA foreign_keys = OFF")  # we handle order ourselves
    return c


def _fetch_pre_counts(c: sqlite3.Connection) -> dict[str, int]:
    """Snapshot of every DB table's row count before we touch anything."""
    counts: dict[str, int] = {}
    for row in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall():
        t = row[0]
        counts[t] = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    return counts


def _gather(
    c: sqlite3.Connection,
    *,
    repo_root: Path,
    weights_root: Path,
    bet_logs_root: Path,
    legacy_bet_log_root: Path,
    training_plans_dir: Path,
    episodes_path: Path,
    eval_diag_path: Path,
) -> PruneInventory:
    """Collect every path / id / row to delete. Pure read — no side effects."""
    inv = PruneInventory()
    inv.pre_counts = _fetch_pre_counts(c)

    inv.garaged_model_ids = [
        r[0] for r in c.execute("SELECT model_id FROM models WHERE garaged=1").fetchall()
    ]
    if not inv.garaged_model_ids:
        raise RuntimeError(
            "Refusing to prune: zero garaged models found. A data issue (or a "
            "typo in the script) could be at fault — bailing so we don't "
            "nuke everything."
        )

    placeholders = ",".join("?" * len(inv.garaged_model_ids))
    inv.keep_run_ids = [
        r[0] for r in c.execute(
            f"SELECT run_id FROM evaluation_runs WHERE model_id IN ({placeholders})",
            inv.garaged_model_ids,
        ).fetchall()
    ]

    delete_rows = c.execute(
        "SELECT model_id, weights_path FROM models "
        "WHERE garaged IS NULL OR garaged=0"
    ).fetchall()
    inv.delete_model_ids = [r[0] for r in delete_rows]
    inv.delete_weight_paths = []
    for _, wp in delete_rows:
        if not wp:
            continue
        p = Path(wp)
        if not p.is_absolute():
            # Paths are stored relative to repo root in the DB.
            p = repo_root / p
        if p.exists():
            inv.delete_weight_paths.append(p)

    inv.delete_run_ids = [
        r[0] for r in c.execute(
            "SELECT run_id FROM evaluation_runs WHERE run_id NOT IN ({})".format(
                ",".join("?" * len(inv.keep_run_ids)) or "''"
            ),
            inv.keep_run_ids,
        ).fetchall()
    ]

    if bet_logs_root.exists():
        keep_set = set(inv.keep_run_ids)
        for child in bet_logs_root.iterdir():
            if child.is_dir() and child.name not in keep_set:
                inv.delete_bet_log_dirs.append(child)

    inv.legacy_bet_log_root = legacy_bet_log_root if legacy_bet_log_root.exists() else None

    # Training plan JSONs: only the paused plan ID. The four activation
    # drafts stay. Look for a file whose basename starts with that UUID.
    if training_plans_dir.exists():
        for p in training_plans_dir.iterdir():
            if p.is_file() and p.name.startswith(PAUSED_PLAN_ID):
                inv.delete_training_plan_files.append(p)

    inv.delete_episodes_path = episodes_path if episodes_path.exists() else None
    inv.delete_eval_diag_path = eval_diag_path if eval_diag_path.exists() else None
    return inv


def _backup(src_db: Path, src_weights: Path, src_plans: Path, archive_dir: Path) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_db, archive_dir / src_db.name)
    if src_weights.exists():
        shutil.copytree(src_weights, archive_dir / src_weights.name)
    if src_plans.exists():
        shutil.copytree(src_plans, archive_dir / src_plans.name)


def _apply_sql_deletes(c: sqlite3.Connection, keep_model_ids: list[str]) -> None:
    """Run the DB delete statements. Single transaction; on error roll back."""
    try:
        c.execute("BEGIN")
        # evaluation_days references evaluation_runs.run_id. Delete
        # days whose run_id is going away.
        placeholders = ",".join("?" * len(keep_model_ids))
        keep_run_ids = [
            r[0] for r in c.execute(
                f"SELECT run_id FROM evaluation_runs WHERE model_id IN ({placeholders})",
                keep_model_ids,
            ).fetchall()
        ]
        run_placeholders = ",".join("?" * len(keep_run_ids)) or "''"
        c.execute(
            f"DELETE FROM evaluation_days WHERE run_id NOT IN ({run_placeholders})",
            keep_run_ids,
        )
        c.execute(
            f"DELETE FROM evaluation_runs WHERE model_id NOT IN ({placeholders})",
            keep_model_ids,
        )
        # genetic_events references model_ids on both parent and child
        # sides. A garaged model's breeding history isn't needed for
        # inference — the weights + row are self-sufficient. Truncate
        # the whole table.
        c.execute("DELETE FROM genetic_events")
        # exploration_runs: user confirmed delete.
        c.execute("DELETE FROM exploration_runs")
        # Finally the models themselves.
        c.execute(
            f"DELETE FROM models WHERE model_id NOT IN ({placeholders})",
            keep_model_ids,
        )
        c.execute("COMMIT")
    except Exception:
        c.execute("ROLLBACK")
        raise
    c.execute("VACUUM")


def _apply_filesystem_deletes(inv: PruneInventory) -> dict[str, int]:
    stats = {
        "weight_files": 0,
        "bet_log_dirs": 0,
        "legacy_bet_log_tree": 0,
        "training_plan_files": 0,
        "episodes_truncated": 0,
        "eval_diag_deleted": 0,
    }
    for p in inv.delete_weight_paths:
        try:
            p.unlink()
            stats["weight_files"] += 1
        except OSError as exc:
            logger.warning("Could not delete weight file %s: %s", p, exc)
    for d in inv.delete_bet_log_dirs:
        try:
            shutil.rmtree(d)
            stats["bet_log_dirs"] += 1
        except OSError as exc:
            logger.warning("Could not delete bet-log dir %s: %s", d, exc)
    if inv.legacy_bet_log_root is not None:
        try:
            shutil.rmtree(inv.legacy_bet_log_root)
            stats["legacy_bet_log_tree"] = 1
        except OSError as exc:
            logger.warning(
                "Could not delete legacy bet-log root %s: %s",
                inv.legacy_bet_log_root, exc,
            )
    for p in inv.delete_training_plan_files:
        try:
            p.unlink()
            stats["training_plan_files"] += 1
        except OSError as exc:
            logger.warning("Could not delete plan file %s: %s", p, exc)
    if inv.delete_episodes_path is not None:
        inv.delete_episodes_path.write_text("", encoding="utf-8")
        stats["episodes_truncated"] = 1
    if inv.delete_eval_diag_path is not None:
        inv.delete_eval_diag_path.unlink()
        stats["eval_diag_deleted"] = 1
    return stats


# -- Reporting ---------------------------------------------------------------


def print_dry_run(inv: PruneInventory, archive_path: Path | None) -> None:
    print("\n=== PRUNE NON-GARAGED - dry run ===\n")
    print(f"DB tables before:")
    for t, n in sorted(inv.pre_counts.items()):
        print(f"  {t:25s}  {n}")
    print()
    print(f"Garaged models kept ({len(inv.garaged_model_ids)}):")
    for mid in inv.garaged_model_ids:
        print(f"  {mid}")
    print()
    print(f"Eval runs kept (belong to garaged): {len(inv.keep_run_ids)}")
    print(f"Models to delete: {len(inv.delete_model_ids)}")
    print(f"Weights files to delete: {len(inv.delete_weight_paths)}")
    print(f"Bet-log dirs under logs/bet_logs/ to delete: {len(inv.delete_bet_log_dirs)}")
    print(f"Legacy registry/bet_logs tree: "
          f"{'will delete' if inv.legacy_bet_log_root else 'absent'}")
    print(f"Training plans to delete (paused plan only): "
          f"{[p.name for p in inv.delete_training_plan_files]}")
    print(f"episodes.jsonl: "
          f"{'truncate' if inv.delete_episodes_path else 'absent'}")
    print(f"_eval_diag.log: "
          f"{'delete' if inv.delete_eval_diag_path else 'absent'}")
    print()
    if archive_path is not None:
        print(f"Backup target (on --apply): {archive_path}")
    else:
        print("Backup: DISABLED (--no-backup). Weights + DB will be unrecoverable.")
    print()
    print("Re-run with --apply to execute.")


def print_apply_summary(
    inv: PruneInventory,
    fs_stats: dict[str, int],
    archive_path: Path | None,
    after_counts: dict[str, int],
) -> None:
    print("\n=== PRUNE NON-GARAGED - applied ===\n")
    if archive_path is not None:
        print(f"Backup written to: {archive_path}")
    print()
    # ASCII-only so Windows consoles using cp1252 don't crash.
    print("DB counts (before -> after):")
    for t in sorted(set(inv.pre_counts) | set(after_counts)):
        before = inv.pre_counts.get(t, 0)
        after = after_counts.get(t, 0)
        marker = "" if before == after else f"  (-{before - after})"
        print(f"  {t:25s}  {before} -> {after}{marker}")
    print()
    print("Filesystem:")
    for k, n in fs_stats.items():
        print(f"  {k:25s}  {n}")
    print()


# -- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "",
    )
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete; default is dry-run.")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip the archive copy. Only for tests.")
    parser.add_argument("--repo-root", default=".",
                        help="Path to the repo (default: cwd).")
    args = parser.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    db_path = repo / "registry" / "models.db"
    weights_root = repo / "registry" / "weights"
    legacy_bet_log_root = repo / "registry" / "bet_logs"
    bet_logs_root = repo / "logs" / "bet_logs"
    training_plans_dir = repo / "registry" / "training_plans"
    episodes_path = repo / "logs" / "training" / "episodes.jsonl"
    eval_diag_path = repo / "_eval_diag.log"

    with _connect(db_path) as c:
        inv = _gather(
            c,
            repo_root=repo,
            weights_root=weights_root,
            bet_logs_root=bet_logs_root,
            legacy_bet_log_root=legacy_bet_log_root,
            training_plans_dir=training_plans_dir,
            episodes_path=episodes_path,
            eval_diag_path=eval_diag_path,
        )

    archive_path: Path | None = None
    if args.apply and not args.no_backup:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_path = repo / "registry" / f"archive_{stamp}"

    if not args.apply:
        print_dry_run(inv, archive_path or (repo / "registry" / "archive_YYYYMMDDTHHMMSSZ"))
        return 0

    if archive_path is not None:
        logger.info("Backing up DB + weights + plans to %s …", archive_path)
        _backup(db_path, weights_root, training_plans_dir, archive_path)

    # Re-open connection so VACUUM runs outside the context manager commit.
    c = _connect(db_path)
    try:
        _apply_sql_deletes(c, inv.garaged_model_ids)
    finally:
        c.close()

    fs_stats = _apply_filesystem_deletes(inv)

    with _connect(db_path) as c:
        after_counts = _fetch_pre_counts(c)
    print_apply_summary(inv, fs_stats, archive_path, after_counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
