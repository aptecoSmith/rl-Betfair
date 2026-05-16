"""Per-cohort naked-variance report.

Generalises the day-1-only one-off in
``tools/compare_naked_variance_cohorts.py`` into a single-cohort,
re-runnable tool that reads BOTH per-leg pnl AND per-day rollups,
emits one CSV row per agent with the variance metrics required by
``plans/scalping-tight-naked-variance/hard_constraints.md``, and
prints the top-15 per score plus the union top-5 across the five
candidate selector scores.

Data sources, in order of preference (hard_constraints §4):
  (a) Per-leg pnl from ``<cohort>/naked_pnl_per_leg.csv`` (raceconf)
      OR ``<cohort>/bet_logs/adhoc_<agent>/<date>.parquet`` filtered
      to ``final_outcome == 'naked'`` (layq). Source of sigma_leg.
  (b) Per-day rollups from ``models.db.evaluation_days`` joined to
      ``evaluation_runs``. Source of naked_std_daily / naked_range /
      mean_locked.

Both sets land in the same per-agent row.

Usage:
    python -m tools.build_naked_variance_report \
        --cohort-dir registry/_predictor_SCALPING_layq_1778712871
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Selector constants (hard_constraints §5) ──────────────────────────────────
PER_LEG_STD_HARD_FILTER = 30.0   # £/leg — memory: feedback_naked_variance_primary_metric.md
DAILY_VOL_HARD_FILTER = 100.0    # £/day — daily naked vol cap
TIGHT_VARIANCE_VOL_COEF = 0.5    # weight on daily_naked_vol in score_d
N_NAKED_LEGS_MIN = 5             # below this, sigma_leg is noise


def _load_scoreboard(cohort_dir: Path) -> pd.DataFrame:
    """Load scoreboard.jsonl into a per-agent DataFrame."""
    rows: list[dict] = []
    p = cohort_dir / "scoreboard.jsonl"
    if not p.exists():
        return pd.DataFrame(columns=["agent_id", "model_id", "gen"])
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append({
                "agent_id": r.get("agent_id"),
                "model_id": r.get("model_id", r.get("agent_id")),
                "gen": int(r.get("generation", -1)),
            })
    return pd.DataFrame(rows)


def _load_per_leg(cohort_dir: Path) -> pd.DataFrame:
    """Return one DataFrame with columns [agent_id, day, pnl] across all
    available per-leg sources. Empty DataFrame if neither source exists.
    """
    frames: list[pd.DataFrame] = []

    # Primary: <cohort>/naked_pnl_per_leg.csv (raceconf format)
    csv_path = cohort_dir / "naked_pnl_per_leg.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if {"agent_id", "day", "pnl"}.issubset(df.columns):
            frames.append(df[["agent_id", "day", "pnl"]].copy())

    # Fallback: per-agent parquets, filtered to final_outcome == 'naked'
    bet_logs = cohort_dir / "bet_logs"
    if bet_logs.exists():
        for agent_dir in sorted(bet_logs.iterdir()):
            if not agent_dir.is_dir() or not agent_dir.name.startswith("adhoc_"):
                continue
            aid = agent_dir.name[len("adhoc_"):]
            for parquet in sorted(agent_dir.glob("*.parquet")):
                try:
                    df = pd.read_parquet(parquet)
                except Exception:
                    continue
                if "final_outcome" not in df.columns or "pnl" not in df.columns:
                    continue
                df = df[df["final_outcome"] == "naked"]
                if df.empty:
                    continue
                day = parquet.stem  # YYYY-MM-DD
                frames.append(pd.DataFrame({
                    "agent_id": aid,
                    "day": day,
                    "pnl": df["pnl"].to_numpy(),
                }))

    if not frames:
        return pd.DataFrame(columns=["agent_id", "day", "pnl"])
    return pd.concat(frames, ignore_index=True)


def _load_per_day(cohort_dir: Path) -> pd.DataFrame:
    """Per-(agent, day) rollups from models.db.

    Columns: model_id, date, locked_pnl, naked_pnl, day_pnl, closed_pnl.
    Empty DataFrame if the DB is missing.
    """
    db = cohort_dir / "models.db"
    if not db.exists():
        return pd.DataFrame(columns=[
            "model_id", "date", "locked_pnl", "naked_pnl",
            "day_pnl", "closed_pnl",
        ])
    try:
        conn = sqlite3.connect(str(db))
        df = pd.read_sql_query(
            """
            SELECT er.model_id, ed.date, ed.locked_pnl, ed.naked_pnl,
                   ed.day_pnl, ed.closed_pnl
            FROM evaluation_days ed
            JOIN evaluation_runs er ON ed.run_id = er.run_id
            """,
            conn,
        )
        conn.close()
        return df
    except sqlite3.Error:
        return pd.DataFrame(columns=[
            "model_id", "date", "locked_pnl", "naked_pnl",
            "day_pnl", "closed_pnl",
        ])


def _per_agent_stats(
    scoreboard: pd.DataFrame,
    per_leg: pd.DataFrame,
    per_day: pd.DataFrame,
) -> pd.DataFrame:
    """Build the per-agent stats DataFrame.

    Row per scoreboard agent. Missing per-leg or per-day data → NaN
    columns, not row drop (hard_constraints §6).
    """
    # Per-leg aggregates per agent
    if not per_leg.empty:
        leg_grp = per_leg.groupby("agent_id")["pnl"]
        leg_stats = pd.DataFrame({
            "n_naked_legs": leg_grp.count(),
            # ddof=0 (population std) matches numpy default and
            # compare_naked_variance_cohorts.py
            "sigma_leg_raw": leg_grp.std(ddof=0),
            "worst_leg_loss": leg_grp.min(),
        }).reset_index()
    else:
        leg_stats = pd.DataFrame(columns=[
            "agent_id", "n_naked_legs", "sigma_leg_raw", "worst_leg_loss",
        ])

    # Per-day aggregates per model_id
    if not per_day.empty:
        day_grp = per_day.groupby("model_id")
        # ddof=1 (sample std) for day-to-day std — small samples and
        # we want unbiased estimate
        day_stats = pd.DataFrame({
            "n_eval_days": day_grp["date"].count(),
            "mean_locked": day_grp["locked_pnl"].mean(),
            "mean_naked": day_grp["naked_pnl"].mean(),
            "naked_std_daily_raw": day_grp["naked_pnl"].std(ddof=1),
            "naked_min": day_grp["naked_pnl"].min(),
            "naked_max": day_grp["naked_pnl"].max(),
            "mean_pnl": day_grp["day_pnl"].mean(),
        }).reset_index()
        day_stats["naked_range"] = day_stats["naked_max"] - day_stats["naked_min"]
    else:
        day_stats = pd.DataFrame(columns=[
            "model_id", "n_eval_days", "mean_locked", "mean_naked",
            "naked_std_daily_raw", "naked_min", "naked_max", "mean_pnl",
            "naked_range",
        ])

    # Join — scoreboard is authoritative for the agent set
    out = scoreboard.merge(leg_stats, on="agent_id", how="left")
    out = out.merge(day_stats, on="model_id", how="left")

    # Fill missing sample-size columns with 0 so masks work cleanly
    out["n_naked_legs"] = out["n_naked_legs"].fillna(0).astype(int)
    out["n_eval_days"] = out["n_eval_days"].fillna(0).astype(int)

    # sigma_leg = NaN when n_naked_legs < N_NAKED_LEGS_MIN (§6)
    sigma_mask = out["n_naked_legs"] >= N_NAKED_LEGS_MIN
    out["sigma_leg"] = np.where(sigma_mask, out["sigma_leg_raw"], np.nan)

    # daily_naked_vol = sigma_leg * sqrt(n_naked_legs / n_eval_days)
    vol_mask = sigma_mask & (out["n_eval_days"] >= 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.sqrt(out["n_naked_legs"].astype(float) / out["n_eval_days"].replace(0, np.nan))
    out["daily_naked_vol"] = np.where(vol_mask, out["sigma_leg"] * scale, np.nan)

    # naked_std_daily = NaN when n_eval_days < 2 (§6)
    std_mask = out["n_eval_days"] >= 2
    out["naked_std_daily"] = np.where(std_mask, out["naked_std_daily_raw"], np.nan)

    # Drop the raw intermediates
    out = out.drop(columns=["sigma_leg_raw", "naked_std_daily_raw"])

    return out


def _selector_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Append the five selector scores to df. Agents missing the
    inputs for a given score get NaN for that score."""
    ml = df["mean_locked"]
    sl = df["sigma_leg"]
    dv = df["daily_naked_vol"]

    df["score_a_pure_locked"] = ml
    df["score_b_per_leg_sharpe"] = ml / (sl + 1.0)
    df["score_c_daily_sharpe"] = ml / (dv + 1.0)
    df["score_d_daily_vol_penalty"] = ml - TIGHT_VARIANCE_VOL_COEF * dv

    # score_e: combined hard filter — keep mean_locked only when
    # sigma_leg AND daily_naked_vol are both finite AND below cap.
    keep = (
        sl.notna() & dv.notna()
        & (sl <= PER_LEG_STD_HARD_FILTER)
        & (dv <= DAILY_VOL_HARD_FILTER)
    )
    df["score_e_combined_filter"] = np.where(keep, ml, 0.0)

    return df


SCORE_COLS = [
    "score_a_pure_locked",
    "score_b_per_leg_sharpe",
    "score_c_daily_sharpe",
    "score_d_daily_vol_penalty",
    "score_e_combined_filter",
]


def _print_top_per_score(df: pd.DataFrame, n: int = 15) -> None:
    """Print the top-n rows for each selector score, sorted desc."""
    display_cols = [
        "agent_id", "gen", "n_naked_legs", "n_eval_days",
        "sigma_leg", "daily_naked_vol", "mean_locked",
        "mean_naked", "naked_std_daily",
    ]
    for sc in SCORE_COLS:
        print()
        print("=" * 110)
        print(f"TOP-{n} BY {sc}")
        print("=" * 110)
        sub = df[df[sc].notna()].sort_values(sc, ascending=False).head(n)
        if sub.empty:
            print("  (no agents with finite score)")
            continue
        sub_short = sub.copy()
        sub_short["agent_id"] = sub_short["agent_id"].str[:12]
        cols = display_cols + [sc]
        print(sub_short[cols].to_string(index=False, float_format=lambda x: f"{x:+.2f}" if isinstance(x, float) and not np.isnan(x) else "nan"))


def _union_top5(df: pd.DataFrame) -> list[str]:
    """Union of the top-5 agent_ids across the five scores."""
    union: list[str] = []
    for sc in SCORE_COLS:
        sub = df[df[sc].notna()].sort_values(sc, ascending=False).head(5)
        for aid in sub["agent_id"]:
            if aid not in union:
                union.append(aid)
    return union


def _positive_pnl_smallest_span_top(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Agents with positive in-sample mean_pnl, sorted by smallest
    naked_range ascending (PnL-descending tiebreak).

    Added 2026-05-16 after Phase 2A operator observation: the union
    of the five selector scores can include negative-PnL agents whose
    span happens to be small (e.g. agent `85121e4b` in the
    tnv_raceconf cohort — span 50, mean_pnl −£54 in-sample). A
    deployment-shape filter wants BOTH a tight span AND positive PnL.
    This view drops the trivially-tight-but-losing agents.

    Returns the top-n rows of the filtered + sorted DataFrame.
    """
    sub = df[(df["mean_pnl"] > 0) & (df["naked_range"].notna())].copy()
    sub = sub.sort_values(["naked_range", "mean_pnl"], ascending=[True, False])
    return sub.head(n)


def build_report(cohort_dir: Path, csv_out: Path | None = None) -> pd.DataFrame:
    """Build the variance report DataFrame. If csv_out is supplied,
    also write it to disk. Returns the DataFrame for downstream use."""
    scoreboard = _load_scoreboard(cohort_dir)
    if scoreboard.empty:
        if csv_out is not None:
            pd.DataFrame().to_csv(csv_out, index=False)
        return pd.DataFrame()

    per_leg = _load_per_leg(cohort_dir)
    per_day = _load_per_day(cohort_dir)

    out = _per_agent_stats(scoreboard, per_leg, per_day)
    out = _selector_scores(out)

    # Order columns predictably
    col_order = [
        "agent_id", "model_id", "gen",
        "n_naked_legs", "n_eval_days",
        "sigma_leg", "daily_naked_vol",
        "mean_locked", "mean_naked",
        "naked_std_daily", "naked_range", "naked_min", "naked_max",
        "worst_leg_loss", "mean_pnl",
    ] + SCORE_COLS
    col_order = [c for c in col_order if c in out.columns]
    out = out[col_order]

    if csv_out is not None:
        out.to_csv(csv_out, index=False)

    return out


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort-dir", required=True, type=Path)
    p.add_argument(
        "--output", default=None, type=Path,
        help=(
            "Output CSV path. Default: <cohort_dir>/naked_variance_report.csv."
        ),
    )
    p.add_argument(
        "--top-n", default=15, type=int,
        help="Top-N rows per score to print. Default 15.",
    )
    p.add_argument(
        "--top5-union-out", default=None, type=Path,
        help=(
            "If set, write the union-of-top-5 agent_ids (one per line) "
            "to this path. Default: <cohort_dir>/phase1_top5_union.txt."
        ),
    )
    p.add_argument(
        "--positive-pnl-top-n", default=10, type=int,
        help=(
            "Number of rows for the positive-PnL + smallest-naked-span "
            "filtered ranking. Default 10."
        ),
    )
    p.add_argument(
        "--positive-pnl-out", default=None, type=Path,
        help=(
            "If set, write the positive-PnL + smallest-naked-span "
            "top-N agent_ids to this path. Default: "
            "<cohort_dir>/positive_pnl_smallest_span_top.txt."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    cohort_dir: Path = args.cohort_dir
    if not cohort_dir.is_dir():
        print(f"ERROR: cohort_dir does not exist: {cohort_dir}", file=sys.stderr)
        return 2

    csv_out = args.output or (cohort_dir / "naked_variance_report.csv")
    df = build_report(cohort_dir, csv_out=csv_out)

    if df.empty:
        print(f"No agents found in {cohort_dir}/scoreboard.jsonl — wrote empty CSV at {csv_out}")
        return 0

    print(f"Cohort: {cohort_dir.name}")
    print(f"Wrote {len(df)} agent rows to {csv_out}")
    n_with_sigma = int(df["sigma_leg"].notna().sum())
    n_with_std = int(df["naked_std_daily"].notna().sum())
    print(f"  agents with sigma_leg (n_naked_legs >= {N_NAKED_LEGS_MIN}): {n_with_sigma}/{len(df)}")
    print(f"  agents with naked_std_daily (n_eval_days >= 2): {n_with_std}/{len(df)}")

    _print_top_per_score(df, n=args.top_n)

    union = _union_top5(df)
    print()
    print("=" * 110)
    print(f"UNION OF TOP-5 ACROSS ALL {len(SCORE_COLS)} SELECTORS — {len(union)} unique agents")
    print("=" * 110)
    for aid in union:
        print(f"  {aid}")

    union_out = args.top5_union_out or (cohort_dir / "phase1_top5_union.txt")
    with union_out.open("w") as f:
        for aid in union:
            f.write(aid + "\n")
    print()
    print(f"Wrote {len(union)} agent_ids to {union_out}")

    # Positive-PnL + smallest-span ranking (2026-05-16). Surfaces
    # agents that combine tight in-sample variance with positive in-
    # sample EV — the deployment shape the variance penalty plan was
    # ultimately trying to find.
    pos_top = _positive_pnl_smallest_span_top(df, n=args.positive_pnl_top_n)
    print()
    print("=" * 130)
    print(
        f"TOP-{args.positive_pnl_top_n} BY POSITIVE in-sample PnL + smallest naked span "
        "(lex sort: span asc, mean_pnl desc)"
    )
    print("=" * 130)
    if pos_top.empty:
        print("  (no agents with positive mean_pnl AND finite naked_range)")
    else:
        display_cols = [
            "agent_id", "gen", "naked_range",
            "mean_pnl", "mean_locked", "mean_naked",
            "naked_min", "naked_max",
        ]
        pos_disp = pos_top.copy()
        pos_disp["agent_id"] = pos_disp["agent_id"].str[:12]
        pos_disp = pos_disp[[c for c in display_cols if c in pos_disp.columns]]
        print(pos_disp.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:+.2f}" if isinstance(x, float) and not np.isnan(x) else "nan"
            ),
        ))

    pos_out = args.positive_pnl_out or (
        cohort_dir / "positive_pnl_smallest_span_top.txt"
    )
    with pos_out.open("w") as f:
        for aid in pos_top["agent_id"]:
            f.write(aid + "\n")
    print()
    print(f"Wrote {len(pos_top)} agent_ids to {pos_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
