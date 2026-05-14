"""Tests for tools/build_naked_variance_report.py."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.build_naked_variance_report import (
    DAILY_VOL_HARD_FILTER,
    N_NAKED_LEGS_MIN,
    PER_LEG_STD_HARD_FILTER,
    build_report,
)


def _write_scoreboard(cohort_dir: Path, agents: list[tuple[str, int]]) -> None:
    """Write a minimal scoreboard.jsonl. Each tuple = (agent_id, gen)."""
    rows = []
    for aid, gen in agents:
        rows.append({
            "agent_id": aid,
            "model_id": aid,  # treat agent_id == model_id in tests
            "generation": gen,
        })
    with (cohort_dir / "scoreboard.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_models_db(
    cohort_dir: Path,
    rows: list[tuple[str, str, float, float]],
) -> None:
    """Write a synthetic models.db with evaluation_days + evaluation_runs.

    rows: list of (model_id, date, locked_pnl, naked_pnl).
    """
    db_path = cohort_dir / "models.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE evaluation_runs (
            run_id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            evaluated_at TEXT NOT NULL,
            train_cutoff_date TEXT NOT NULL,
            test_days TEXT DEFAULT '[]'
        )
    """)
    cur.execute("""
        CREATE TABLE evaluation_days (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            date TEXT NOT NULL,
            day_pnl REAL NOT NULL,
            bet_count INTEGER NOT NULL DEFAULT 0,
            winning_bets INTEGER NOT NULL DEFAULT 0,
            bet_precision REAL NOT NULL DEFAULT 0,
            pnl_per_bet REAL NOT NULL DEFAULT 0,
            early_picks INTEGER NOT NULL DEFAULT 0,
            profitable INTEGER NOT NULL DEFAULT 0,
            mean_opportunity_window_s REAL NOT NULL DEFAULT 0,
            median_opportunity_window_s REAL NOT NULL DEFAULT 0,
            starting_budget REAL NOT NULL DEFAULT 100,
            arbs_completed INTEGER NOT NULL DEFAULT 0,
            arbs_naked INTEGER NOT NULL DEFAULT 0,
            locked_pnl REAL NOT NULL DEFAULT 0,
            naked_pnl REAL NOT NULL DEFAULT 0,
            paired_rejects_no_ltp INTEGER NOT NULL DEFAULT 0,
            paired_rejects_price_invalid INTEGER NOT NULL DEFAULT 0,
            paired_rejects_budget_back INTEGER NOT NULL DEFAULT 0,
            paired_rejects_budget_lay INTEGER NOT NULL DEFAULT 0,
            paired_fill_skips INTEGER NOT NULL DEFAULT 0,
            arbs_closed INTEGER NOT NULL DEFAULT 0,
            arbs_force_closed INTEGER NOT NULL DEFAULT 0,
            arbs_stop_closed INTEGER NOT NULL DEFAULT 0,
            arbs_target_pnl_refused INTEGER NOT NULL DEFAULT 0,
            pairs_opened INTEGER NOT NULL DEFAULT 0,
            closed_pnl REAL NOT NULL DEFAULT 0,
            force_closed_pnl REAL NOT NULL DEFAULT 0,
            stop_closed_pnl REAL NOT NULL DEFAULT 0
        )
    """)
    run_ids: dict[str, str] = {}
    for i, (mid, date, locked, naked) in enumerate(rows):
        if mid not in run_ids:
            run_id = f"run_{mid[:6]}"
            run_ids[mid] = run_id
            cur.execute(
                "INSERT INTO evaluation_runs (run_id, model_id, evaluated_at, train_cutoff_date) VALUES (?, ?, ?, ?)",
                (run_id, mid, "2026-01-01", "2026-01-01"),
            )
        cur.execute(
            "INSERT INTO evaluation_days (run_id, date, day_pnl, locked_pnl, naked_pnl) VALUES (?, ?, ?, ?, ?)",
            (run_ids[mid], date, locked + naked, locked, naked),
        )
    conn.commit()
    conn.close()


def test_recovers_known_values_on_synthetic_data(tmp_path: Path) -> None:
    """Per-leg array → recovered sigma_leg & daily_naked_vol match
    closed-form expectations."""
    aid = "agent-known-0001"
    _write_scoreboard(tmp_path, [(aid, 3)])

    # 9 naked legs across 3 days; pnls chosen so std (ddof=0) is exact
    leg_rows = []
    pnls = [-10.0, 0.0, +10.0, -10.0, 0.0, +10.0, -10.0, 0.0, +10.0]
    for i, p in enumerate(pnls):
        day = ["2026-05-04", "2026-05-05", "2026-05-06"][i // 3]
        leg_rows.append({"agent_id": aid, "day": day, "pnl": p})
    pd.DataFrame(leg_rows).to_csv(tmp_path / "naked_pnl_per_leg.csv", index=False)

    # 3 days of per-day rollups: locked_pnl average 50, naked sum equal across days
    _write_models_db(tmp_path, [
        (aid, "2026-05-04", 50.0, 0.0),
        (aid, "2026-05-05", 50.0, 0.0),
        (aid, "2026-05-06", 50.0, 0.0),
    ])

    df = build_report(tmp_path, csv_out=None)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["n_naked_legs"] == 9
    assert row["n_eval_days"] == 3
    # ddof=0 std of [-10, 0, +10, -10, 0, +10, -10, 0, +10]
    expected_sigma = float(np.std(pnls, ddof=0))
    assert row["sigma_leg"] == pytest.approx(expected_sigma, rel=1e-9)
    # daily_naked_vol = sigma_leg * sqrt(N/n_days) = sigma * sqrt(3)
    assert row["daily_naked_vol"] == pytest.approx(expected_sigma * np.sqrt(3.0), rel=1e-9)
    assert row["mean_locked"] == pytest.approx(50.0, rel=1e-9)


def test_score_e_boundary(tmp_path: Path) -> None:
    """An agent exactly at sigma_leg=30 AND daily=100 keeps its
    mean_locked (filter is <=, not <)."""
    aid = "agent-boundary-1"
    _write_scoreboard(tmp_path, [(aid, 0)])

    # We need sigma_leg == 30. With pnls = [-30, +30, -30, +30, ...]
    # population std = 30. Use 9 legs to make N=9, n_eval_days=9 →
    # daily_naked_vol = 30 * sqrt(9/9) = 30. That's NOT 100.
    # We want daily_naked_vol == 100, so sigma_leg * sqrt(N/days) = 100.
    # If sigma=30, sqrt(N/days) = 100/30 = 10/3 → N/days = 100/9 ≈ 11.11
    # → N=100, days=9: daily = 30 * sqrt(100/9) = 30 * (10/3) = 100. ✓
    pnls = []
    for _ in range(50):
        pnls.append(-30.0)
        pnls.append(+30.0)  # 100 legs total
    leg_rows = []
    for i, p in enumerate(pnls):
        day = f"2026-05-{(i % 9) + 1:02d}"
        leg_rows.append({"agent_id": aid, "day": day, "pnl": p})
    pd.DataFrame(leg_rows).to_csv(tmp_path / "naked_pnl_per_leg.csv", index=False)

    # 9 days of per-day rollups so n_eval_days == 9
    db_rows = []
    for d in range(1, 10):
        db_rows.append((aid, f"2026-05-{d:02d}", 75.0, 0.0))
    _write_models_db(tmp_path, db_rows)

    df = build_report(tmp_path, csv_out=None)
    row = df.iloc[0]
    assert row["sigma_leg"] == pytest.approx(PER_LEG_STD_HARD_FILTER, rel=1e-9)
    assert row["daily_naked_vol"] == pytest.approx(DAILY_VOL_HARD_FILTER, rel=1e-9)
    # KEEP the mean_locked (boundary is inclusive)
    assert row["score_e_combined_filter"] == pytest.approx(75.0, rel=1e-9)


def test_falls_back_to_db_when_no_per_leg_data(tmp_path: Path) -> None:
    """No per-leg sources → sigma_leg/n_naked_legs are NaN/0, but
    per-day stats still populate from models.db."""
    aid = "agent-db-only"
    _write_scoreboard(tmp_path, [(aid, 1)])
    # No naked_pnl_per_leg.csv, no bet_logs.
    _write_models_db(tmp_path, [
        (aid, "2026-05-04", 100.0, -50.0),
        (aid, "2026-05-05", 100.0, +50.0),
        (aid, "2026-05-06", 100.0, 0.0),
    ])

    df = build_report(tmp_path, csv_out=None)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["n_naked_legs"] == 0
    assert row["n_eval_days"] == 3
    assert np.isnan(row["sigma_leg"])
    assert np.isnan(row["daily_naked_vol"])
    assert row["mean_locked"] == pytest.approx(100.0, rel=1e-9)
    # naked_std_daily defined (n_eval_days >= 2)
    assert not np.isnan(row["naked_std_daily"])
    # naked_pnl values [-50, 50, 0] → sample std (ddof=1) = 50.0
    assert row["naked_std_daily"] == pytest.approx(50.0, rel=1e-9)
    # score_e: sigma_leg is NaN → keep mask False → 0.0
    assert row["score_e_combined_filter"] == 0.0


def test_nan_when_sample_too_small(tmp_path: Path) -> None:
    """n_naked_legs < N_NAKED_LEGS_MIN → sigma_leg == NaN."""
    aid = "agent-tiny"
    _write_scoreboard(tmp_path, [(aid, 0)])
    # 4 legs only (< 5)
    pnls = [-10.0, +10.0, -5.0, +5.0]
    leg_rows = [{"agent_id": aid, "day": "2026-05-04", "pnl": p} for p in pnls]
    pd.DataFrame(leg_rows).to_csv(tmp_path / "naked_pnl_per_leg.csv", index=False)
    _write_models_db(tmp_path, [(aid, "2026-05-04", 25.0, 0.0)])

    df = build_report(tmp_path, csv_out=None)
    row = df.iloc[0]
    assert row["n_naked_legs"] == 4
    assert row["n_naked_legs"] < N_NAKED_LEGS_MIN
    assert np.isnan(row["sigma_leg"])
    assert np.isnan(row["daily_naked_vol"])


def test_empty_cohort_produces_empty_csv(tmp_path: Path) -> None:
    """No scoreboard.jsonl → build_report returns empty + writes empty CSV."""
    csv_out = tmp_path / "naked_variance_report.csv"
    df = build_report(tmp_path, csv_out=csv_out)
    assert df.empty
    assert csv_out.exists()
    # File is on disk with no data rows — verifies the tool doesn't
    # crash on cohorts that haven't started yet.
    content = csv_out.read_text().strip()
    assert content == ""
