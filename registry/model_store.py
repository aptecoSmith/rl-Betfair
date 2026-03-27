"""
registry/model_store.py -- SQLite-backed model registry.

Stores model metadata, evaluation runs, per-day metrics, and individual bets
in a local SQLite database.  Supports save/load of PyTorch model weights.

Tables
------
- ``models``           -- one row per trained model
- ``evaluation_runs``  -- one row per evaluation run
- ``evaluation_days``  -- one row per test-day per run
- ``evaluation_bets``  -- one row per bet placed during evaluation

Usage::

    store = ModelStore("registry/models.db", weights_dir="registry/weights")
    model_id = store.create_model(...)
    store.save_weights(model_id, policy.state_dict())
    state_dict = store.load_weights(model_id)
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import torch


# -- Data classes for API results ---------------------------------------------


@dataclass
class ModelRecord:
    """A model record from the registry."""

    model_id: str
    generation: int
    parent_a_id: str | None
    parent_b_id: str | None
    architecture_name: str
    architecture_description: str
    hyperparameters: dict
    status: str  # "active" | "discarded"
    created_at: str
    last_evaluated_at: str | None
    weights_path: str | None
    composite_score: float | None


@dataclass
class EvaluationRunRecord:
    """An evaluation run record."""

    run_id: str
    model_id: str
    evaluated_at: str
    train_cutoff_date: str
    test_days: list[str]


@dataclass
class EvaluationDayRecord:
    """Per-day evaluation metrics."""

    run_id: str
    date: str
    day_pnl: float
    bet_count: int
    winning_bets: int
    bet_precision: float
    pnl_per_bet: float
    early_picks: int
    profitable: bool


@dataclass
class EvaluationBetRecord:
    """An individual bet from evaluation."""

    run_id: str
    date: str
    market_id: str
    tick_timestamp: str
    seconds_to_off: float
    runner_id: int
    runner_name: str
    action: str  # "back" | "lay"
    price: float
    stake: float
    matched_size: float
    outcome: str  # "won" | "lost" | "void"
    pnl: float


# -- ModelStore ----------------------------------------------------------------


class ModelStore:
    """SQLite-backed persistent model registry.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database file.
    weights_dir : str | Path
        Directory for saving model weight files (.pt).
    """

    def __init__(
        self,
        db_path: str | Path = "registry/models.db",
        weights_dir: str | Path = "registry/weights",
    ) -> None:
        self.db_path = Path(db_path)
        self.weights_dir = Path(weights_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_conn()
        try:
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    # -- Model CRUD -----------------------------------------------------------

    def create_model(
        self,
        generation: int,
        architecture_name: str,
        architecture_description: str,
        hyperparameters: dict,
        parent_a_id: str | None = None,
        parent_b_id: str | None = None,
        model_id: str | None = None,
    ) -> str:
        """Insert a new model record.  Returns the model_id (UUID)."""
        mid = model_id or str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()

        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO models
                    (model_id, generation, parent_a_id, parent_b_id,
                     architecture_name, architecture_description,
                     hyperparameters, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)
                """,
                (
                    mid,
                    generation,
                    parent_a_id,
                    parent_b_id,
                    architecture_name,
                    architecture_description,
                    json.dumps(hyperparameters),
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return mid

    def get_model(self, model_id: str) -> ModelRecord | None:
        """Fetch a single model by ID."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM models WHERE model_id = ?", (model_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_model(row)
        finally:
            conn.close()

    def list_models(self, status: str | None = None) -> list[ModelRecord]:
        """List all models, optionally filtered by status."""
        conn = self._get_conn()
        try:
            if status:
                rows = conn.execute(
                    "SELECT * FROM models WHERE status = ? ORDER BY created_at DESC",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM models ORDER BY created_at DESC",
                ).fetchall()
            return [self._row_to_model(r) for r in rows]
        finally:
            conn.close()

    def update_model_status(self, model_id: str, status: str) -> None:
        """Update a model's status (e.g. 'active' -> 'discarded')."""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE models SET status = ? WHERE model_id = ?",
                (status, model_id),
            )
            conn.commit()
        finally:
            conn.close()

    def update_composite_score(self, model_id: str, score: float) -> None:
        """Update a model's composite score and last_evaluated_at."""
        now = datetime.now(UTC).isoformat()
        conn = self._get_conn()
        try:
            conn.execute(
                """UPDATE models
                   SET composite_score = ?, last_evaluated_at = ?
                   WHERE model_id = ?""",
                (score, now, model_id),
            )
            conn.commit()
        finally:
            conn.close()

    # -- Weights I/O ----------------------------------------------------------

    def save_weights(self, model_id: str, state_dict: dict) -> str:
        """Save PyTorch state dict to disk.  Returns the file path."""
        path = self.weights_dir / f"{model_id}.pt"
        torch.save(state_dict, str(path))

        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE models SET weights_path = ? WHERE model_id = ?",
                (str(path), model_id),
            )
            conn.commit()
        finally:
            conn.close()
        return str(path)

    def load_weights(self, model_id: str) -> dict:
        """Load PyTorch state dict from disk."""
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found in registry")
        if model.weights_path is None:
            raise ValueError(f"Model {model_id} has no saved weights")
        return torch.load(model.weights_path, weights_only=True)

    # -- Evaluation runs ------------------------------------------------------

    def create_evaluation_run(
        self,
        model_id: str,
        train_cutoff_date: str,
        test_days: list[str],
        run_id: str | None = None,
    ) -> str:
        """Create an evaluation run.  Returns the run_id (UUID)."""
        rid = run_id or str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()

        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO evaluation_runs
                    (run_id, model_id, evaluated_at, train_cutoff_date, test_days)
                VALUES (?, ?, ?, ?, ?)
                """,
                (rid, model_id, now, train_cutoff_date, json.dumps(test_days)),
            )
            conn.commit()
        finally:
            conn.close()
        return rid

    def record_evaluation_day(self, record: EvaluationDayRecord) -> None:
        """Insert one per-day evaluation result."""
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO evaluation_days
                    (run_id, date, day_pnl, bet_count, winning_bets,
                     bet_precision, pnl_per_bet, early_picks, profitable)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.date,
                    record.day_pnl,
                    record.bet_count,
                    record.winning_bets,
                    record.bet_precision,
                    record.pnl_per_bet,
                    record.early_picks,
                    1 if record.profitable else 0,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def record_evaluation_bet(self, record: EvaluationBetRecord) -> None:
        """Insert one evaluation bet record."""
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO evaluation_bets
                    (run_id, date, market_id, tick_timestamp, seconds_to_off,
                     runner_id, runner_name, action, price, stake,
                     matched_size, outcome, pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.date,
                    record.market_id,
                    record.tick_timestamp,
                    record.seconds_to_off,
                    record.runner_id,
                    record.runner_name,
                    record.action,
                    record.price,
                    record.stake,
                    record.matched_size,
                    record.outcome,
                    record.pnl,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_evaluation_days(self, run_id: str) -> list[EvaluationDayRecord]:
        """Get all per-day metrics for an evaluation run."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM evaluation_days WHERE run_id = ? ORDER BY date",
                (run_id,),
            ).fetchall()
            return [
                EvaluationDayRecord(
                    run_id=r["run_id"],
                    date=r["date"],
                    day_pnl=r["day_pnl"],
                    bet_count=r["bet_count"],
                    winning_bets=r["winning_bets"],
                    bet_precision=r["bet_precision"],
                    pnl_per_bet=r["pnl_per_bet"],
                    early_picks=r["early_picks"],
                    profitable=bool(r["profitable"]),
                )
                for r in rows
            ]
        finally:
            conn.close()

    def get_evaluation_bets(self, run_id: str) -> list[EvaluationBetRecord]:
        """Get all bets from an evaluation run."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM evaluation_bets WHERE run_id = ? ORDER BY date, tick_timestamp",
                (run_id,),
            ).fetchall()
            return [
                EvaluationBetRecord(
                    run_id=r["run_id"],
                    date=r["date"],
                    market_id=r["market_id"],
                    tick_timestamp=r["tick_timestamp"],
                    seconds_to_off=r["seconds_to_off"],
                    runner_id=r["runner_id"],
                    runner_name=r["runner_name"],
                    action=r["action"],
                    price=r["price"],
                    stake=r["stake"],
                    matched_size=r["matched_size"],
                    outcome=r["outcome"],
                    pnl=r["pnl"],
                )
                for r in rows
            ]
        finally:
            conn.close()

    def get_latest_evaluation_run(self, model_id: str) -> EvaluationRunRecord | None:
        """Get the most recent evaluation run for a model."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT * FROM evaluation_runs
                   WHERE model_id = ?
                   ORDER BY evaluated_at DESC LIMIT 1""",
                (model_id,),
            ).fetchone()
            if row is None:
                return None
            return EvaluationRunRecord(
                run_id=row["run_id"],
                model_id=row["model_id"],
                evaluated_at=row["evaluated_at"],
                train_cutoff_date=row["train_cutoff_date"],
                test_days=json.loads(row["test_days"]),
            )
        finally:
            conn.close()

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _row_to_model(row: sqlite3.Row) -> ModelRecord:
        return ModelRecord(
            model_id=row["model_id"],
            generation=row["generation"],
            parent_a_id=row["parent_a_id"],
            parent_b_id=row["parent_b_id"],
            architecture_name=row["architecture_name"],
            architecture_description=row["architecture_description"],
            hyperparameters=json.loads(row["hyperparameters"]),
            status=row["status"],
            created_at=row["created_at"],
            last_evaluated_at=row["last_evaluated_at"],
            weights_path=row["weights_path"],
            composite_score=row["composite_score"],
        )


# -- SQL Schema ----------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS models (
    model_id                TEXT PRIMARY KEY,
    generation              INTEGER NOT NULL,
    parent_a_id             TEXT,
    parent_b_id             TEXT,
    architecture_name       TEXT NOT NULL,
    architecture_description TEXT NOT NULL DEFAULT '',
    hyperparameters         TEXT NOT NULL DEFAULT '{}',
    status                  TEXT NOT NULL DEFAULT 'active',
    created_at              TEXT NOT NULL,
    last_evaluated_at       TEXT,
    weights_path            TEXT,
    composite_score         REAL
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id              TEXT PRIMARY KEY,
    model_id            TEXT NOT NULL REFERENCES models(model_id),
    evaluated_at        TEXT NOT NULL,
    train_cutoff_date   TEXT NOT NULL,
    test_days           TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS evaluation_days (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES evaluation_runs(run_id),
    date            TEXT NOT NULL,
    day_pnl         REAL NOT NULL,
    bet_count       INTEGER NOT NULL,
    winning_bets    INTEGER NOT NULL DEFAULT 0,
    bet_precision   REAL NOT NULL DEFAULT 0.0,
    pnl_per_bet     REAL NOT NULL DEFAULT 0.0,
    early_picks     INTEGER NOT NULL DEFAULT 0,
    profitable      INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_eval_days_run ON evaluation_days(run_id);

CREATE TABLE IF NOT EXISTS evaluation_bets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES evaluation_runs(run_id),
    date            TEXT NOT NULL,
    market_id       TEXT NOT NULL,
    tick_timestamp  TEXT NOT NULL,
    seconds_to_off  REAL NOT NULL DEFAULT 0.0,
    runner_id       INTEGER NOT NULL,
    runner_name     TEXT NOT NULL DEFAULT '',
    action          TEXT NOT NULL,
    price           REAL NOT NULL,
    stake           REAL NOT NULL,
    matched_size    REAL NOT NULL DEFAULT 0.0,
    outcome         TEXT NOT NULL DEFAULT 'unsettled',
    pnl             REAL NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_eval_bets_run ON evaluation_bets(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_bets_market ON evaluation_bets(run_id, date, market_id);
"""
