"""
registry/model_store.py -- SQLite-backed model registry.

Stores model metadata, evaluation runs, per-day metrics, and genetic events
in a local SQLite database.  Evaluation bet logs are stored as Parquet files
under ``registry/bet_logs/{run_id}/{date}.parquet``.

Tables
------
- ``models``           -- one row per trained model
- ``evaluation_runs``  -- one row per evaluation run
- ``evaluation_days``  -- one row per test-day per run
- ``genetic_events``   -- one row per genetic event
- ``exploration_runs`` -- one row per HP-search exploration run

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

import pandas as pd


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
    garaged: bool = False
    garaged_at: str | None = None


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
    mean_opportunity_window_s: float = 0.0
    median_opportunity_window_s: float = 0.0
    starting_budget: float = 100.0
    # Forced-arbitrage (scalping) metrics — Issue 05. Always zero for
    # directional models; non-zero rows indicate the model was trained
    # with training.scalping_mode=true.
    arbs_completed: int = 0
    arbs_naked: int = 0
    locked_pnl: float = 0.0
    naked_pnl: float = 0.0
    # Paired-arb silent-failure diagnostics (added 2026-04-15). Lets the
    # operator distinguish "agent never tried to pair" (all zeros) from
    # "agent tried but every paired leg was refused at placement"
    # (paired_rejects_* > 0) from "paired leg placed but the LTP-distance
    # fill filter rejected every tick" (paired_fill_skips > 0).
    paired_rejects_no_ltp: int = 0
    paired_rejects_price_invalid: int = 0
    paired_rejects_budget_back: int = 0
    paired_rejects_budget_lay: int = 0
    paired_fill_skips: int = 0
    # Cohort-visibility S01b (2026-05-02). Per-pair lifecycle counters
    # added across `arb-signal-cleanup` (force_closed),
    # `force-close-architecture` S01 (target_pnl_refused, pairs_opened,
    # closed_pnl, force_closed_pnl), and `force-close-architecture`
    # S02 (stop_closed, stop_closed_pnl). Stored here so `models.db`
    # readers (peek tools, future v1 UI panels) see the full per-agent
    # breakdown live as each agent finishes its eval. Pre-plan rows
    # default to 0 / 0.0; post-plan rows carry honest values from
    # ``EvalSummary``. See ``plans/rewrite/phase-3-followups/
    # cohort-visibility/``.
    arbs_closed: int = 0
    arbs_force_closed: int = 0
    arbs_stop_closed: int = 0
    arbs_target_pnl_refused: int = 0
    pairs_opened: int = 0
    closed_pnl: float = 0.0
    force_closed_pnl: float = 0.0
    stop_closed_pnl: float = 0.0


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
    opportunity_window_s: float = 0.0
    # EW metadata
    is_each_way: bool = False
    each_way_divisor: float | None = None
    number_of_places: int | None = None
    settlement_type: str = "standard"
    effective_place_odds: float | None = None
    starting_budget: float = 100.0
    # Scalping: pair_id links the aggressive leg to its passive counter-order.
    # None for non-scalping bets. Back/lay bets sharing a pair_id form a
    # hedged pair whose locked P&L floor can be classified in the UI.
    pair_id: str | None = None
    # Scalping-active-management §02: decision-time fill-probability
    # prediction stamped on the ``Bet`` by the PPO rollout. Mirrored here
    # so the Parquet bet log can be read back for calibration plots
    # (predicted vs observed fill rate, bucketed). Optional — ``None``
    # for pre-Session-02 runs and for any bet the scalping head did not
    # produce a prediction for (directional bets, stub tests).
    fill_prob_at_placement: float | None = None
    # Scalping-active-management §03: decision-time risk-head outputs.
    # Mean + stddev (``stddev = exp(0.5 * clamped_log_var)`` computed at
    # capture time) of predicted locked P&L. Mirrors the Bet fields so
    # the Parquet log can drive risk badges in the UI and scatter plots
    # (predicted vs realised stddev) for calibration. Optional — ``None``
    # for pre-Session-03 runs.
    predicted_locked_pnl_at_placement: float | None = None
    predicted_locked_stddev_at_placement: float | None = None
    # Arb-signal-cleanup Session 03b (2026-04-22). Distinguishes closing
    # legs placed via agent-initiated ``close_signal`` from env-initiated
    # force-close at T−N. Without these, the bet-explorer UI treats both
    # as regular matched bets — a force-closed pair's legs look the same
    # as any other closed pair, and the operator can't tell why the
    # agent ended up in that trade. ``close_leg`` is True for any bet
    # placed via ``_attempt_close`` (either force-close or close_signal).
    # ``force_close`` is True only when ``_attempt_close`` was called
    # with ``force_close=True`` (env-initiated). Both default to False
    # so pre-fix parquet rows stay readable without schema migration.
    close_leg: bool = False
    force_close: bool = False


@dataclass
class GeneticEventRecord:
    """A genetic event (selection, crossover, mutation, discard)."""

    event_id: str
    generation: int
    event_type: str  # "selection" | "crossover" | "mutation" | "discard"
    child_model_id: str | None = None
    parent_a_id: str | None = None
    parent_b_id: str | None = None
    hyperparameter: str | None = None
    parent_a_value: str | None = None
    parent_b_value: str | None = None
    inherited_from: str | None = None  # "A" | "B"
    mutation_delta: float | None = None
    final_value: str | None = None
    selection_reason: str | None = None
    human_summary: str | None = None


@dataclass
class ExplorationRunRecord:
    """A hyperparameter-search exploration run."""

    id: int | None  # auto-increment PK (None before insert)
    run_id: str  # links to a training run / plan
    created_at: str
    seed_point: dict  # gene name → value
    region_id: str | None = None
    strategy: str = "random"  # "random" | "sobol" | "coverage" | "manual"
    coverage_before: dict | None = None  # snapshot of coverage at seed time
    notes: str | None = None


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
        bet_logs_dir: str | Path | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.weights_dir = Path(weights_dir)
        # Default bet_logs_dir sits next to the db file
        if bet_logs_dir is None:
            self.bet_logs_dir = Path("logs/bet_logs")
        else:
            self.bet_logs_dir = Path(bet_logs_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.bet_logs_dir.mkdir(parents=True, exist_ok=True)
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
            # Migration: add garaged column to existing databases
            try:
                conn.execute(
                    "ALTER TABLE models ADD COLUMN garaged INTEGER NOT NULL DEFAULT 0"
                )
            except Exception:
                pass  # column already exists
            try:
                conn.execute(
                    "ALTER TABLE models ADD COLUMN garaged_at TEXT"
                )
            except Exception:
                pass  # column already exists
            # Migration: add opportunity window columns to existing databases
            for col in ("mean_opportunity_window_s", "median_opportunity_window_s"):
                try:
                    conn.execute(
                        f"ALTER TABLE evaluation_days ADD COLUMN {col} REAL NOT NULL DEFAULT 0.0"
                    )
                except Exception:
                    pass  # column already exists
            # Migration: add starting_budget column (default 100.0 for existing rows)
            try:
                conn.execute(
                    "ALTER TABLE evaluation_days ADD COLUMN starting_budget REAL NOT NULL DEFAULT 100.0"
                )
            except Exception:
                pass  # column already exists
            # Migration: add forced-arbitrage (scalping) columns — Issue 05.
            # Plus cohort-visibility S01b (2026-05-02): widen with the
            # post-Session-01-and-02 per-pair lifecycle counters so the
            # SQLite readout is the canonical per-agent live source.
            for col, sql_type, default in (
                ("arbs_completed", "INTEGER", "0"),
                ("arbs_naked", "INTEGER", "0"),
                ("locked_pnl", "REAL", "0.0"),
                ("naked_pnl", "REAL", "0.0"),
                # Paired-arb silent-failure diagnostics (added 2026-04-15).
                ("paired_rejects_no_ltp", "INTEGER", "0"),
                ("paired_rejects_price_invalid", "INTEGER", "0"),
                ("paired_rejects_budget_back", "INTEGER", "0"),
                ("paired_rejects_budget_lay", "INTEGER", "0"),
                ("paired_fill_skips", "INTEGER", "0"),
                # Cohort-visibility S01b: per-pair lifecycle counters
                # introduced by arb-signal-cleanup (force_closed),
                # force-close-architecture S01 (target_pnl_refused,
                # pairs_opened, closed_pnl, force_closed_pnl), and
                # force-close-architecture S02 (stop_closed,
                # stop_closed_pnl). Migrations are idempotent so re-
                # running ModelStore.__init__ on a post-plan db is a
                # no-op. Pre-plan rows default to 0 / 0.0.
                ("arbs_closed", "INTEGER", "0"),
                ("arbs_force_closed", "INTEGER", "0"),
                ("arbs_stop_closed", "INTEGER", "0"),
                ("arbs_target_pnl_refused", "INTEGER", "0"),
                ("pairs_opened", "INTEGER", "0"),
                ("closed_pnl", "REAL", "0.0"),
                ("force_closed_pnl", "REAL", "0.0"),
                ("stop_closed_pnl", "REAL", "0.0"),
            ):
                try:
                    conn.execute(
                        f"ALTER TABLE evaluation_days ADD COLUMN {col} {sql_type} "
                        f"NOT NULL DEFAULT {default}"
                    )
                except Exception:
                    pass  # column already exists
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

    def update_hyperparameters(self, model_id: str, hyperparameters: dict) -> None:
        """Overwrite a model's stored hyperparameters JSON."""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE models SET hyperparameters = ? WHERE model_id = ?",
                (json.dumps(hyperparameters), model_id),
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

    def set_garaged(self, model_id: str, garaged: bool) -> None:
        """Set or clear the garaged flag on a model."""
        now = datetime.now(UTC).isoformat() if garaged else None
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE models SET garaged = ?, garaged_at = ? WHERE model_id = ?",
                (1 if garaged else 0, now, model_id),
            )
            conn.commit()
        finally:
            conn.close()

    def list_garaged_models(self) -> list[ModelRecord]:
        """List all garaged models regardless of active/discarded status."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM models WHERE garaged = 1 ORDER BY composite_score DESC",
            ).fetchall()
            return [self._row_to_model(r) for r in rows]
        finally:
            conn.close()

    def delete_model(self, model_id: str) -> bool:
        """Fully delete a model: weights, bet logs, eval data, genetic events, DB record.

        Returns True if the model existed and was deleted.
        """
        model = self.get_model(model_id)
        if model is None:
            return False

        # Delete weights file
        if model.weights_path:
            wp = Path(model.weights_path)
            if wp.exists():
                wp.unlink()

        # Delete bet log Parquets for all evaluation runs
        conn = self._get_conn()
        try:
            runs = conn.execute(
                "SELECT run_id FROM evaluation_runs WHERE model_id = ?",
                (model_id,),
            ).fetchall()
            for run in runs:
                run_dir = self.bet_logs_dir / run["run_id"]
                if run_dir.exists():
                    import shutil
                    shutil.rmtree(run_dir)

            # Delete DB records (order matters for foreign keys)
            for run in runs:
                conn.execute("DELETE FROM evaluation_days WHERE run_id = ?", (run["run_id"],))
            conn.execute("DELETE FROM evaluation_runs WHERE model_id = ?", (model_id,))
            conn.execute("DELETE FROM genetic_events WHERE child_model_id = ?", (model_id,))
            conn.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
            conn.commit()
        finally:
            conn.close()
        return True

    def purge_discarded(self) -> list[str]:
        """Delete all discarded, non-garaged models and their artefacts.

        Returns list of deleted model IDs.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT model_id FROM models WHERE status = 'discarded' AND garaged = 0",
            ).fetchall()
        finally:
            conn.close()

        purged = []
        for row in rows:
            mid = row["model_id"]
            if self.delete_model(mid):
                purged.append(mid)
        return purged

    def purge_incompatible(
        self,
        required_obs_version: int,
        *,
        dry_run: bool = False,
    ) -> list[str]:
        """Delete all active models whose weights don't match *required_obs_version*.

        Checks the ``obs_schema_version`` key in each checkpoint without fully
        loading the weights into a policy.  Models with missing weights, missing
        schema keys, or a version mismatch are considered incompatible.

        Garaged models are **skipped** (un-garage them first if you want them
        purged).

        Returns list of deleted (or would-be-deleted if *dry_run*) model IDs.
        """
        import torch

        models = self.list_models(status="active")
        incompatible: list[str] = []
        for m in models:
            if m.garaged:
                continue
            if not m.weights_path or not Path(m.weights_path).exists():
                incompatible.append(m.model_id)
                continue
            try:
                raw = torch.load(m.weights_path, weights_only=True)
                version = raw.get("obs_schema_version") if isinstance(raw, dict) else None
                if version != required_obs_version:
                    incompatible.append(m.model_id)
            except Exception:
                incompatible.append(m.model_id)

        if dry_run:
            return incompatible

        deleted: list[str] = []
        for mid in incompatible:
            if self.delete_model(mid):
                deleted.append(mid)
        return deleted

    def stamp_weights(
        self,
        model_id: str,
        obs_schema_version: int,
        action_schema_version: int | None = None,
    ) -> bool:
        """Add schema-version metadata to an existing checkpoint in place.

        If the checkpoint already has the correct version, this is a no-op.
        Returns True if the file was rewritten.
        """
        import torch

        model = self.get_model(model_id)
        if model is None or not model.weights_path:
            return False
        p = Path(model.weights_path)
        if not p.exists():
            return False

        raw = torch.load(str(p), weights_only=True)
        if isinstance(raw, dict) and raw.get("obs_schema_version") == obs_schema_version:
            return False  # already stamped

        # Extract bare weights
        weights = raw["weights"] if isinstance(raw, dict) and "weights" in raw else raw
        payload: dict = {"weights": weights, "obs_schema_version": obs_schema_version}
        if action_schema_version is not None:
            payload["action_schema_version"] = action_schema_version
        torch.save(payload, str(p))
        return True

    # -- Weights I/O ----------------------------------------------------------

    def save_weights(
        self,
        model_id: str,
        state_dict: dict,
        obs_schema_version: int | None = None,
        action_schema_version: int | None = None,
    ) -> str:
        """Save PyTorch state dict to disk.  Returns the file path.

        When *obs_schema_version* is provided the checkpoint is wrapped as
        ``{"obs_schema_version": N, "weights": state_dict}`` so that loaders
        can refuse mismatched versions loudly.  *action_schema_version* is
        included alongside it (session 28 / P3a).
        """
        import torch

        path = self.weights_dir / f"{model_id}.pt"
        if obs_schema_version is not None or action_schema_version is not None:
            payload: dict = {"weights": state_dict}
            if obs_schema_version is not None:
                payload["obs_schema_version"] = obs_schema_version
            if action_schema_version is not None:
                payload["action_schema_version"] = action_schema_version
        else:
            payload = state_dict
        torch.save(payload, str(path))

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

    def load_weights(
        self,
        model_id: str,
        expected_obs_schema_version: int | None = None,
        expected_action_schema_version: int | None = None,
    ) -> dict:
        """Load PyTorch state dict from disk.

        When *expected_obs_schema_version* or *expected_action_schema_version*
        is provided the checkpoint is validated before the state dict is
        returned.  A version mismatch raises ``ValueError``.
        """
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found in registry")
        if model.weights_path is None:
            raise ValueError(f"Model {model_id} has no saved weights")
        import torch

        raw = torch.load(model.weights_path, weights_only=True)
        if expected_obs_schema_version is not None:
            from env.betfair_env import validate_obs_schema
            validate_obs_schema(raw)
        if expected_action_schema_version is not None:
            from env.betfair_env import validate_action_schema
            validate_action_schema(raw)
        # Support both wrapped format ({"obs_schema_version": N, "weights": ...})
        # and bare state-dict format for forward compatibility.
        if isinstance(raw, dict) and "weights" in raw:
            return raw["weights"]
        return raw

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
                     bet_precision, pnl_per_bet, early_picks, profitable,
                     mean_opportunity_window_s, median_opportunity_window_s,
                     starting_budget,
                     arbs_completed, arbs_naked, locked_pnl, naked_pnl,
                     paired_rejects_no_ltp, paired_rejects_price_invalid,
                     paired_rejects_budget_back, paired_rejects_budget_lay,
                     paired_fill_skips,
                     arbs_closed, arbs_force_closed, arbs_stop_closed,
                     arbs_target_pnl_refused, pairs_opened,
                     closed_pnl, force_closed_pnl, stop_closed_pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?)
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
                    record.mean_opportunity_window_s,
                    record.median_opportunity_window_s,
                    record.starting_budget,
                    record.arbs_completed,
                    record.arbs_naked,
                    record.locked_pnl,
                    record.naked_pnl,
                    record.paired_rejects_no_ltp,
                    record.paired_rejects_price_invalid,
                    record.paired_rejects_budget_back,
                    record.paired_rejects_budget_lay,
                    record.paired_fill_skips,
                    record.arbs_closed,
                    record.arbs_force_closed,
                    record.arbs_stop_closed,
                    record.arbs_target_pnl_refused,
                    record.pairs_opened,
                    record.closed_pnl,
                    record.force_closed_pnl,
                    record.stop_closed_pnl,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def write_bet_logs_parquet(
        self, run_id: str, date: str, records: list[EvaluationBetRecord],
    ) -> Path | None:
        """Write evaluation bet records as a Parquet file.

        Path: ``bet_logs_dir/{run_id}/{date}.parquet``

        Returns the written path, or None if *records* is empty.
        """
        if not records:
            return None
        run_dir = self.bet_logs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"{date}.parquet"
        df = pd.DataFrame([
            {
                "run_id": r.run_id,
                "date": r.date,
                "market_id": r.market_id,
                "tick_timestamp": r.tick_timestamp,
                "seconds_to_off": r.seconds_to_off,
                "runner_id": r.runner_id,
                "runner_name": r.runner_name,
                "action": r.action,
                "price": r.price,
                "stake": r.stake,
                "matched_size": r.matched_size,
                "outcome": r.outcome,
                "pnl": r.pnl,
                "opportunity_window_s": r.opportunity_window_s,
                "is_each_way": r.is_each_way,
                "each_way_divisor": r.each_way_divisor,
                "number_of_places": r.number_of_places,
                "settlement_type": r.settlement_type,
                "effective_place_odds": r.effective_place_odds,
                "starting_budget": r.starting_budget,
                "pair_id": r.pair_id,
                # Scalping-active-management §02. Nullable float — pandas
                # stores ``None`` as NaN which the reader below maps back
                # to ``None`` via ``pd.notna`` check.
                "fill_prob_at_placement": r.fill_prob_at_placement,
                # Scalping-active-management §03. Nullable floats — same
                # NaN-as-None convention as the fill-prob column.
                "predicted_locked_pnl_at_placement": (
                    r.predicted_locked_pnl_at_placement
                ),
                "predicted_locked_stddev_at_placement": (
                    r.predicted_locked_stddev_at_placement
                ),
                # Arb-signal-cleanup Session 03b (2026-04-22). Surfaces
                # force-close metadata to the bet-explorer UI. Booleans
                # stored as-is; ``None`` isn't possible here because the
                # source ``Bet`` object always has these flags.
                "close_leg": r.close_leg,
                "force_close": r.force_close,
            }
            for r in records
        ])
        df.to_parquet(path, index=False)
        return path

    def get_evaluation_days(self, run_id: str) -> list[EvaluationDayRecord]:
        """Get all per-day metrics for an evaluation run."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM evaluation_days WHERE run_id = ? ORDER BY date",
                (run_id,),
            ).fetchall()
            col_names = set(rows[0].keys()) if rows else set()
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
                    mean_opportunity_window_s=(
                        r["mean_opportunity_window_s"]
                        if "mean_opportunity_window_s" in col_names else 0.0
                    ),
                    median_opportunity_window_s=(
                        r["median_opportunity_window_s"]
                        if "median_opportunity_window_s" in col_names else 0.0
                    ),
                    starting_budget=(
                        r["starting_budget"]
                        if "starting_budget" in col_names else 100.0
                    ),
                    arbs_completed=(
                        r["arbs_completed"]
                        if "arbs_completed" in col_names else 0
                    ),
                    arbs_naked=(
                        r["arbs_naked"]
                        if "arbs_naked" in col_names else 0
                    ),
                    locked_pnl=(
                        r["locked_pnl"]
                        if "locked_pnl" in col_names else 0.0
                    ),
                    naked_pnl=(
                        r["naked_pnl"]
                        if "naked_pnl" in col_names else 0.0
                    ),
                    paired_rejects_no_ltp=(
                        r["paired_rejects_no_ltp"]
                        if "paired_rejects_no_ltp" in col_names else 0
                    ),
                    paired_rejects_price_invalid=(
                        r["paired_rejects_price_invalid"]
                        if "paired_rejects_price_invalid" in col_names else 0
                    ),
                    paired_rejects_budget_back=(
                        r["paired_rejects_budget_back"]
                        if "paired_rejects_budget_back" in col_names else 0
                    ),
                    paired_rejects_budget_lay=(
                        r["paired_rejects_budget_lay"]
                        if "paired_rejects_budget_lay" in col_names else 0
                    ),
                    paired_fill_skips=(
                        r["paired_fill_skips"]
                        if "paired_fill_skips" in col_names else 0
                    ),
                    # Cohort-visibility S01b (2026-05-02). Default-tolerant
                    # reads on legacy rows (NULL / missing column → 0 or
                    # 0.0).
                    arbs_closed=(
                        (r["arbs_closed"] or 0)
                        if "arbs_closed" in col_names else 0
                    ),
                    arbs_force_closed=(
                        (r["arbs_force_closed"] or 0)
                        if "arbs_force_closed" in col_names else 0
                    ),
                    arbs_stop_closed=(
                        (r["arbs_stop_closed"] or 0)
                        if "arbs_stop_closed" in col_names else 0
                    ),
                    arbs_target_pnl_refused=(
                        (r["arbs_target_pnl_refused"] or 0)
                        if "arbs_target_pnl_refused" in col_names else 0
                    ),
                    pairs_opened=(
                        (r["pairs_opened"] or 0)
                        if "pairs_opened" in col_names else 0
                    ),
                    closed_pnl=(
                        (r["closed_pnl"] or 0.0)
                        if "closed_pnl" in col_names else 0.0
                    ),
                    force_closed_pnl=(
                        (r["force_closed_pnl"] or 0.0)
                        if "force_closed_pnl" in col_names else 0.0
                    ),
                    stop_closed_pnl=(
                        (r["stop_closed_pnl"] or 0.0)
                        if "stop_closed_pnl" in col_names else 0.0
                    ),
                )
                for r in rows
            ]
        finally:
            conn.close()

    def get_evaluation_bets(self, run_id: str) -> list[EvaluationBetRecord]:
        """Get all bets from an evaluation run (reads from Parquet files).

        Reads all ``{date}.parquet`` files under ``bet_logs_dir/{run_id}/``
        and returns them as a flat list sorted by date and tick_timestamp.
        """
        run_dir = self.bet_logs_dir / run_id
        if not run_dir.exists():
            return []
        parquet_files = sorted(run_dir.glob("*.parquet"))
        if not parquet_files:
            return []
        dfs = [pd.read_parquet(p) for p in parquet_files]
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values(["date", "tick_timestamp"]).reset_index(drop=True)
        has_opp_window = "opportunity_window_s" in df.columns
        has_ew = "is_each_way" in df.columns
        has_budget = "starting_budget" in df.columns
        has_pair_id = "pair_id" in df.columns
        # Scalping-active-management §02 — Parquet files written before
        # this column existed load as ``fill_prob_at_placement=None``
        # (hard_constraints §11: new columns are optional).
        has_fill_prob = "fill_prob_at_placement" in df.columns
        # Scalping-active-management §03 — same back-compat pattern:
        # pre-Session-03 parquet files load with the two risk fields as
        # ``None``.
        has_predicted_locked_pnl = (
            "predicted_locked_pnl_at_placement" in df.columns
        )
        has_predicted_locked_stddev = (
            "predicted_locked_stddev_at_placement" in df.columns
        )
        # Arb-signal-cleanup Session 03b (2026-04-22). Pre-fix parquet
        # files lack these columns — default to False on the read side.
        has_close_leg = "close_leg" in df.columns
        has_force_close = "force_close" in df.columns
        return [
            EvaluationBetRecord(
                run_id=str(row["run_id"]),
                date=str(row["date"]),
                market_id=str(row["market_id"]),
                tick_timestamp=str(row["tick_timestamp"]),
                seconds_to_off=float(row["seconds_to_off"]),
                runner_id=int(row["runner_id"]),
                runner_name=str(row["runner_name"]),
                action=str(row["action"]),
                price=float(row["price"]),
                stake=float(row["stake"]),
                matched_size=float(row["matched_size"]),
                outcome=str(row["outcome"]),
                pnl=float(row["pnl"]),
                opportunity_window_s=(
                    float(row["opportunity_window_s"]) if has_opp_window else 0.0
                ),
                is_each_way=bool(row["is_each_way"]) if has_ew else False,
                each_way_divisor=(
                    float(row["each_way_divisor"])
                    if has_ew and pd.notna(row.get("each_way_divisor"))
                    else None
                ),
                number_of_places=(
                    int(row["number_of_places"])
                    if has_ew and pd.notna(row.get("number_of_places"))
                    else None
                ),
                settlement_type=(
                    str(row["settlement_type"]) if has_ew else "standard"
                ),
                effective_place_odds=(
                    float(row["effective_place_odds"])
                    if has_ew and pd.notna(row.get("effective_place_odds"))
                    else None
                ),
                starting_budget=(
                    float(row["starting_budget"]) if has_budget else 100.0
                ),
                pair_id=(
                    str(row["pair_id"])
                    if has_pair_id and pd.notna(row.get("pair_id"))
                    else None
                ),
                fill_prob_at_placement=(
                    float(row["fill_prob_at_placement"])
                    if has_fill_prob
                    and pd.notna(row.get("fill_prob_at_placement"))
                    else None
                ),
                predicted_locked_pnl_at_placement=(
                    float(row["predicted_locked_pnl_at_placement"])
                    if has_predicted_locked_pnl
                    and pd.notna(row.get("predicted_locked_pnl_at_placement"))
                    else None
                ),
                predicted_locked_stddev_at_placement=(
                    float(row["predicted_locked_stddev_at_placement"])
                    if has_predicted_locked_stddev
                    and pd.notna(row.get("predicted_locked_stddev_at_placement"))
                    else None
                ),
                close_leg=(
                    bool(row["close_leg"]) if has_close_leg else False
                ),
                force_close=(
                    bool(row["force_close"]) if has_force_close else False
                ),
            )
            for _, row in df.iterrows()
        ]

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

    # -- Genetic events -------------------------------------------------------

    def record_genetic_event(self, record: GeneticEventRecord) -> None:
        """Insert one genetic event record."""
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO genetic_events
                    (event_id, generation, event_type, child_model_id,
                     parent_a_id, parent_b_id, hyperparameter,
                     parent_a_value, parent_b_value, inherited_from,
                     mutation_delta, final_value, selection_reason,
                     human_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.event_id,
                    record.generation,
                    record.event_type,
                    record.child_model_id,
                    record.parent_a_id,
                    record.parent_b_id,
                    record.hyperparameter,
                    record.parent_a_value,
                    record.parent_b_value,
                    record.inherited_from,
                    record.mutation_delta,
                    record.final_value,
                    record.selection_reason,
                    record.human_summary,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_genetic_events(
        self,
        generation: int | None = None,
        child_model_id: str | None = None,
    ) -> list[GeneticEventRecord]:
        """Query genetic events, optionally filtered by generation or child model."""
        conn = self._get_conn()
        try:
            conditions: list[str] = []
            params: list = []
            if generation is not None:
                conditions.append("generation = ?")
                params.append(generation)
            if child_model_id is not None:
                conditions.append("child_model_id = ?")
                params.append(child_model_id)

            where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
            rows = conn.execute(
                f"SELECT * FROM genetic_events{where} ORDER BY rowid",
                params,
            ).fetchall()
            return [
                GeneticEventRecord(
                    event_id=r["event_id"],
                    generation=r["generation"],
                    event_type=r["event_type"],
                    child_model_id=r["child_model_id"],
                    parent_a_id=r["parent_a_id"],
                    parent_b_id=r["parent_b_id"],
                    hyperparameter=r["hyperparameter"],
                    parent_a_value=r["parent_a_value"],
                    parent_b_value=r["parent_b_value"],
                    inherited_from=r["inherited_from"],
                    mutation_delta=r["mutation_delta"],
                    final_value=r["final_value"],
                    selection_reason=r["selection_reason"],
                    human_summary=r["human_summary"],
                )
                for r in rows
            ]
        finally:
            conn.close()

    # -- Exploration runs -----------------------------------------------------

    def record_exploration_run(
        self,
        run_id: str,
        seed_point: dict,
        strategy: str = "random",
        region_id: str | None = None,
        coverage_before: dict | None = None,
        notes: str | None = None,
    ) -> int:
        """Insert an exploration run and return its auto-increment id."""
        conn = self._get_conn()
        try:
            now = datetime.now(UTC).isoformat()
            cursor = conn.execute(
                """INSERT INTO exploration_runs
                   (run_id, created_at, seed_point, region_id,
                    strategy, coverage_before, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    now,
                    json.dumps(seed_point),
                    region_id,
                    strategy,
                    json.dumps(coverage_before) if coverage_before is not None else None,
                    notes,
                ),
            )
            conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]
        finally:
            conn.close()

    def get_exploration_history(
        self,
        strategy: str | None = None,
    ) -> list[ExplorationRunRecord]:
        """Return exploration runs, optionally filtered by strategy."""
        conn = self._get_conn()
        try:
            if strategy is not None:
                rows = conn.execute(
                    "SELECT * FROM exploration_runs WHERE strategy = ? ORDER BY created_at",
                    (strategy,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM exploration_runs ORDER BY created_at",
                ).fetchall()
            return [self._row_to_exploration_run(r) for r in rows]
        finally:
            conn.close()

    def get_exploration_run_count(self) -> int:
        """Return the total number of exploration runs (used for Sobol skip)."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM exploration_runs").fetchone()
            return row["cnt"]
        finally:
            conn.close()

    @staticmethod
    def _row_to_exploration_run(row: sqlite3.Row) -> ExplorationRunRecord:
        return ExplorationRunRecord(
            id=row["id"],
            run_id=row["run_id"],
            created_at=row["created_at"],
            seed_point=json.loads(row["seed_point"]),
            region_id=row["region_id"],
            strategy=row["strategy"],
            coverage_before=(
                json.loads(row["coverage_before"])
                if row["coverage_before"] is not None
                else None
            ),
            notes=row["notes"],
        )

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _row_to_model(row: sqlite3.Row) -> ModelRecord:
        keys = row.keys()
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
            garaged=bool(row["garaged"]) if "garaged" in keys else False,
            garaged_at=row["garaged_at"] if "garaged_at" in keys else None,
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
    profitable      INTEGER NOT NULL DEFAULT 0,
    mean_opportunity_window_s  REAL NOT NULL DEFAULT 0.0,
    median_opportunity_window_s REAL NOT NULL DEFAULT 0.0,
    starting_budget REAL NOT NULL DEFAULT 100.0,
    arbs_completed  INTEGER NOT NULL DEFAULT 0,
    arbs_naked      INTEGER NOT NULL DEFAULT 0,
    locked_pnl      REAL NOT NULL DEFAULT 0.0,
    naked_pnl       REAL NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_eval_days_run ON evaluation_days(run_id);

CREATE TABLE IF NOT EXISTS genetic_events (
    event_id            TEXT PRIMARY KEY,
    generation          INTEGER NOT NULL,
    event_type          TEXT NOT NULL,
    child_model_id      TEXT,
    parent_a_id         TEXT,
    parent_b_id         TEXT,
    hyperparameter      TEXT,
    parent_a_value      TEXT,
    parent_b_value      TEXT,
    inherited_from      TEXT,
    mutation_delta       REAL,
    final_value         TEXT,
    selection_reason     TEXT,
    human_summary       TEXT
);

CREATE INDEX IF NOT EXISTS idx_genetic_generation ON genetic_events(generation);
CREATE INDEX IF NOT EXISTS idx_genetic_child ON genetic_events(child_model_id);

CREATE TABLE IF NOT EXISTS exploration_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    seed_point          TEXT NOT NULL DEFAULT '{}',
    region_id           TEXT,
    strategy            TEXT NOT NULL DEFAULT 'random',
    coverage_before     TEXT,
    notes               TEXT
);

CREATE INDEX IF NOT EXISTS idx_exploration_run_id ON exploration_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_exploration_strategy ON exploration_runs(strategy);
"""
