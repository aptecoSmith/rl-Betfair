"""Unit tests for ``scripts/prune_non_garaged.py``.

Runs the script against a synthetic repo on ``tmp_path`` with a hand-
built SQLite DB mirroring the real schema. Verifies:
- Garaged models + their eval_runs + eval_days survive.
- Non-garaged models + their artefacts are deleted.
- Dry-run never mutates.
- Refuses to run when zero garaged models exist.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest


# Add scripts/ to path so we can import the module under test.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import prune_non_garaged as prune  # noqa: E402


SCHEMA = """
CREATE TABLE models (
    model_id TEXT PRIMARY KEY,
    generation INTEGER,
    parent_a_id TEXT,
    parent_b_id TEXT,
    architecture_name TEXT,
    architecture_description TEXT,
    hyperparameters TEXT,
    status TEXT,
    created_at TEXT,
    last_evaluated_at TEXT,
    weights_path TEXT,
    composite_score REAL,
    garaged INTEGER DEFAULT 0,
    garaged_at TEXT
);
CREATE TABLE evaluation_runs (
    run_id TEXT PRIMARY KEY,
    model_id TEXT,
    evaluated_at TEXT,
    train_cutoff_date TEXT,
    test_days TEXT
);
CREATE TABLE evaluation_days (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    date TEXT,
    day_pnl REAL,
    bet_count INTEGER,
    winning_bets INTEGER,
    bet_precision REAL,
    pnl_per_bet REAL,
    early_picks INTEGER,
    profitable INTEGER,
    mean_opportunity_window_s REAL,
    median_opportunity_window_s REAL,
    starting_budget REAL,
    arbs_completed INTEGER,
    arbs_naked INTEGER,
    locked_pnl REAL,
    naked_pnl REAL,
    paired_rejects_no_ltp INTEGER,
    paired_rejects_price_invalid INTEGER,
    paired_rejects_budget_back INTEGER,
    paired_rejects_budget_lay INTEGER,
    paired_fill_skips INTEGER
);
CREATE TABLE genetic_events (
    event_id TEXT PRIMARY KEY,
    generation INTEGER,
    event_type TEXT,
    child_model_id TEXT,
    parent_a_id TEXT,
    parent_b_id TEXT,
    hyperparameter TEXT,
    parent_a_value TEXT,
    parent_b_value TEXT,
    inherited_from TEXT,
    mutation_delta REAL,
    final_value TEXT,
    selection_reason TEXT,
    human_summary TEXT
);
CREATE TABLE exploration_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    created_at TEXT,
    seed_point TEXT,
    region_id TEXT,
    strategy TEXT,
    coverage_before TEXT,
    notes TEXT
);
"""


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """Build a minimal repo layout + DB at ``tmp_path`` and return the root."""
    (tmp_path / "registry").mkdir()
    (tmp_path / "registry" / "weights").mkdir()
    (tmp_path / "registry" / "bet_logs").mkdir()
    (tmp_path / "registry" / "training_plans").mkdir()
    (tmp_path / "logs" / "bet_logs").mkdir(parents=True)
    (tmp_path / "logs" / "training").mkdir(parents=True)

    # Build DB: 1 garaged + 2 non-garaged models.
    db_path = tmp_path / "registry" / "models.db"
    c = sqlite3.connect(db_path)
    c.executescript(SCHEMA)
    c.executemany(
        "INSERT INTO models (model_id, architecture_name, created_at, weights_path, garaged) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            ("keep-1", "ppo_lstm_v1", "2026-04-15T00:00:00Z",
             "registry/weights/keep-1.pt", 1),
            ("drop-1", "ppo_lstm_v1", "2026-04-17T00:00:00Z",
             "registry/weights/drop-1.pt", 0),
            ("drop-2", "ppo_time_lstm_v1", "2026-04-17T00:00:00Z",
             "registry/weights/drop-2.pt", None),
        ],
    )
    c.executemany(
        "INSERT INTO evaluation_runs (run_id, model_id, evaluated_at) VALUES (?, ?, ?)",
        [
            ("run-keep-1", "keep-1", "2026-04-15T01:00:00Z"),
            ("run-drop-1", "drop-1", "2026-04-17T01:00:00Z"),
            ("run-drop-2", "drop-2", "2026-04-17T02:00:00Z"),
        ],
    )
    c.executemany(
        "INSERT INTO evaluation_days (run_id, date, day_pnl) VALUES (?, ?, ?)",
        [
            ("run-keep-1", "2026-04-14", 5.0),
            ("run-drop-1", "2026-04-16", -10.0),
            ("run-drop-2", "2026-04-16", -20.0),
        ],
    )
    c.executemany(
        "INSERT INTO genetic_events (event_id, child_model_id, event_type) VALUES (?, ?, ?)",
        [("e1", "drop-1", "mutation"), ("e2", "drop-2", "crossover")],
    )
    c.execute("INSERT INTO exploration_runs (run_id, strategy) VALUES (?, ?)",
              ("expl-1", "sobol"))
    c.commit()
    c.close()

    # Weight files.
    for mid in ("keep-1", "drop-1", "drop-2"):
        (tmp_path / "registry" / "weights" / f"{mid}.pt").write_text("fake weights")

    # Bet-log dirs: one per run_id under logs/bet_logs; legacy dir
    # under registry/bet_logs.
    for rid in ("run-keep-1", "run-drop-1", "run-drop-2"):
        (tmp_path / "logs" / "bet_logs" / rid).mkdir()
        (tmp_path / "logs" / "bet_logs" / rid / "day.parquet").write_text("stub")
    (tmp_path / "registry" / "bet_logs" / "legacy-run").mkdir()
    (tmp_path / "registry" / "bet_logs" / "legacy-run" / "day.parquet").write_text("stub")

    # Training plans: paused + one draft.
    paused_file = tmp_path / "registry" / "training_plans" / f"{prune.PAUSED_PLAN_ID}.json"
    paused_file.write_text(json.dumps({"plan_id": prune.PAUSED_PLAN_ID, "name": "paused"}))
    draft_file = tmp_path / "registry" / "training_plans" / "abc-draft.json"
    draft_file.write_text(json.dumps({"plan_id": "abc-draft", "name": "draft"}))

    # Logs + scratch files.
    (tmp_path / "logs" / "training" / "episodes.jsonl").write_text("{}\n{}\n{}\n")
    (tmp_path / "_eval_diag.log").write_text("stale debug output\n")
    return tmp_path


def test_dry_run_touches_nothing(fake_repo, capsys):
    """Dry-run prints the plan but mutates no file or row."""
    rc = prune.main(["--repo-root", str(fake_repo)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry run" in out.lower()
    assert "keep-1" in out
    # DB unchanged
    c = sqlite3.connect(fake_repo / "registry" / "models.db")
    assert c.execute("SELECT COUNT(*) FROM models").fetchone()[0] == 3
    assert c.execute("SELECT COUNT(*) FROM evaluation_runs").fetchone()[0] == 3
    assert c.execute("SELECT COUNT(*) FROM exploration_runs").fetchone()[0] == 1
    c.close()
    # Weight files unchanged
    assert (fake_repo / "registry" / "weights" / "drop-1.pt").exists()
    # episodes.jsonl unchanged
    assert (fake_repo / "logs" / "training" / "episodes.jsonl").read_text() != ""


def test_apply_deletes_non_garaged_and_preserves_garaged(fake_repo, capsys):
    rc = prune.main(["--apply", "--no-backup", "--repo-root", str(fake_repo)])
    assert rc == 0

    # DB: only garaged model + its run + its eval day survive.
    c = sqlite3.connect(fake_repo / "registry" / "models.db")
    assert [r[0] for r in c.execute("SELECT model_id FROM models").fetchall()] == ["keep-1"]
    assert [r[0] for r in c.execute("SELECT run_id FROM evaluation_runs").fetchall()] == ["run-keep-1"]
    assert c.execute("SELECT COUNT(*) FROM evaluation_days").fetchone()[0] == 1
    assert c.execute("SELECT COUNT(*) FROM genetic_events").fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM exploration_runs").fetchone()[0] == 0
    c.close()

    # Weights: keep-1 survives; drops are gone.
    assert (fake_repo / "registry" / "weights" / "keep-1.pt").exists()
    assert not (fake_repo / "registry" / "weights" / "drop-1.pt").exists()
    assert not (fake_repo / "registry" / "weights" / "drop-2.pt").exists()

    # Bet-log dirs: run-keep-1 stays; drop runs + legacy tree gone.
    assert (fake_repo / "logs" / "bet_logs" / "run-keep-1").exists()
    assert not (fake_repo / "logs" / "bet_logs" / "run-drop-1").exists()
    assert not (fake_repo / "logs" / "bet_logs" / "run-drop-2").exists()
    assert not (fake_repo / "registry" / "bet_logs").exists()

    # Training plans: paused gone, draft stays.
    plan_dir = fake_repo / "registry" / "training_plans"
    assert not (plan_dir / f"{prune.PAUSED_PLAN_ID}.json").exists()
    assert (plan_dir / "abc-draft.json").exists()

    # episodes.jsonl truncated; eval_diag deleted.
    assert (fake_repo / "logs" / "training" / "episodes.jsonl").read_text() == ""
    assert not (fake_repo / "_eval_diag.log").exists()


def test_apply_writes_backup(fake_repo):
    rc = prune.main(["--apply", "--repo-root", str(fake_repo)])
    assert rc == 0
    archives = list((fake_repo / "registry").glob("archive_*"))
    assert len(archives) == 1
    archive = archives[0]
    assert (archive / "models.db").exists()
    assert (archive / "weights" / "drop-1.pt").exists()  # deleted from live, preserved in archive
    assert (archive / "training_plans" / f"{prune.PAUSED_PLAN_ID}.json").exists()


def test_refuses_when_no_garaged_models(tmp_path: Path):
    (tmp_path / "registry").mkdir()
    db = tmp_path / "registry" / "models.db"
    c = sqlite3.connect(db)
    c.executescript(SCHEMA)
    c.execute(
        "INSERT INTO models (model_id, architecture_name, garaged) VALUES (?, ?, ?)",
        ("m1", "ppo_lstm_v1", 0),
    )
    c.commit()
    c.close()
    with pytest.raises(RuntimeError, match="zero garaged models"):
        prune.main(["--apply", "--no-backup", "--repo-root", str(tmp_path)])


def test_missing_db_raises(tmp_path: Path):
    (tmp_path / "registry").mkdir()
    with pytest.raises(FileNotFoundError):
        prune.main(["--apply", "--no-backup", "--repo-root", str(tmp_path)])
