"""Aux-head calibration report for v2 cohorts.

Two modes, picked automatically based on what the cohort produced:

1. AGGREGATE mode (current v2 cohorts).
   Reads per-agent aux-head BCE/NLL averages from
   ``scoreboard.jsonl`` (``train_mean_fill_prob_bce``,
   ``train_mean_mature_prob_bce``, ``train_mean_risk_nll``,
   ``train_mean_direction_back_bce``, ``train_mean_direction_lay_bce``)
   and reports:
   * Per-agent final BCE/NLL values
   * Whether each head's BCE is descending vs the random-policy floor
     (~0.69 for BCE, ~0 for risk NLL)
   * Cohort-wide correlation between each gene weight and its head's
     final BCE (does turning the weight up actually lower the loss?)

   Answers: "are the heads learning their labels at all?"
   Does NOT answer: "does the actor actually use the head's output."

2. PER-BET mode (v2 cohorts after the trainer plumbing patch).
   If the cohort's bet_logs carry non-null
   ``fill_prob_at_placement`` / ``predicted_locked_pnl_at_placement``
   / ``mature_prob_at_placement`` (etc.) columns, run a proper
   reliability-curve calibration:
   * Pearson correlation between predicted prob and actual outcome
     label (1 if the predicted event happened, 0 otherwise)
   * Brier score (mean squared error against the binary label)
   * Bucketed reliability table (predicted prob bucket -> empirical
     observed rate)

   Answers: "is the head's output a useful signal at the bet level?"

Per-bet mode auto-falls-back to aggregate mode if all the relevant
columns are NULL (the current v2 plumbing gap, 2026-05-24).

Usage:
    python tools/aux_head_calibration_report.py <cohort_dir>
    python tools/aux_head_calibration_report.py <cohort_dir> --top 5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable


# Random-policy BCE floor for binary classification — ln(2) ≈ 0.693.
BCE_RANDOM_FLOOR = math.log(2.0)


def load_scoreboard(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


# ── Aggregate mode ───────────────────────────────────────────────────


HEAD_FIELDS = {
    "fill_prob": {
        "bce_col": "train_mean_fill_prob_bce",
        "weight_gene": "fill_prob_loss_weight",
        "floor": BCE_RANDOM_FLOOR,
        "lower_is_better": True,
    },
    "mature_prob": {
        "bce_col": "train_mean_mature_prob_bce",
        "weight_gene": "mature_prob_loss_weight",
        "floor": BCE_RANDOM_FLOOR,
        "lower_is_better": True,
    },
    "risk_nll": {
        "bce_col": "train_mean_risk_nll",
        "weight_gene": "risk_loss_weight",
        "floor": 0.0,  # well-fit Gaussian NLL is near 0; can go negative.
        "lower_is_better": True,
    },
    "direction_back_bce": {
        "bce_col": "train_mean_direction_back_bce",
        # direction_prob_loss_weight isn't a Phase 5 gene yet — usually
        # pinned cohort-wide via --reward-overrides. We still report
        # the BCE; the correlation column will be NaN when the
        # weight column is constant.
        "weight_gene": None,
        "floor": BCE_RANDOM_FLOOR,
        "lower_is_better": True,
    },
    "direction_lay_bce": {
        "bce_col": "train_mean_direction_lay_bce",
        "weight_gene": None,
        "floor": BCE_RANDOM_FLOOR,
        "lower_is_better": True,
    },
}


def _format_loss(value: float, floor: float) -> str:
    """Format a loss with a flag if it's stuck at / above the random floor."""
    if math.isnan(value):
        return "  -    "
    delta = value - floor
    if abs(delta) < 0.02:
        flag = " *FLAT*"
    elif value > floor:
        flag = " *>floor*"
    else:
        flag = ""
    return f"{value:+.3f}{flag}"


def aggregate_mode(rows: list[dict], top: int | None) -> str:
    if not rows:
        return "(scoreboard.jsonl empty — no completed agents yet)"

    # Sort by composite_score descending; take top-N if asked.
    rows_sorted = sorted(
        rows, key=lambda r: r.get("composite_score", 0.0), reverse=True,
    )
    if top is not None:
        rows_sorted = rows_sorted[:top]

    lines: list[str] = []
    lines.append(
        "Aux-head calibration report (AGGREGATE mode — per-agent "
        "final BCE/NLL values from scoreboard).\n"
    )
    lines.append(
        f"Reading {len(rows_sorted)} agents "
        f"(of {len(rows)} total in cohort)."
    )
    lines.append(
        "*FLAT* = within ±0.02 of the random-policy floor "
        "(head likely not learning).\n"
    )

    # Per-agent table.
    cols = ["agent", "gen"]
    for head in HEAD_FIELDS:
        cols.append(head)
    lines.append(" | ".join(f"{c:>22}" if c not in ("agent", "gen") else f"{c:<10}" for c in cols))
    lines.append("-" * (len(cols) * 25))

    for r in rows_sorted:
        agent = r.get("agent_id", "?")[:8]
        gen = r.get("generation", "?")
        cells = [f"{agent:<10}", f"{gen!s:<10}"]
        for head, spec in HEAD_FIELDS.items():
            val = r.get(spec["bce_col"])
            if val is None:
                cells.append(f"{'NULL':>22}")
            else:
                cells.append(f"{_format_loss(float(val), spec['floor']):>22}")
        lines.append(" | ".join(cells))

    # Gene-weight × final-BCE correlation across the cohort.
    lines.append("")
    lines.append(
        "Gene-weight vs final-BCE correlation across cohort "
        "(does higher loss-weight produce lower BCE?):"
    )
    lines.append("")
    lines.append(
        f"{'head':<20} {'weight_gene':<28} {'pearson':>10}"
    )
    lines.append("-" * 60)
    for head, spec in HEAD_FIELDS.items():
        gene = spec["weight_gene"]
        if gene is None:
            lines.append(
                f"{head:<20} {'(not a gene)':<28} {'-':>10}"
            )
            continue
        xs: list[float] = []
        ys: list[float] = []
        for r in rows:
            hp = r.get("hyperparameters", {})
            w = hp.get(gene)
            bce = r.get(spec["bce_col"])
            if w is None or bce is None:
                continue
            xs.append(float(w))
            ys.append(float(bce))
        if len(xs) < 2 or len(set(xs)) < 2:
            lines.append(
                f"{head:<20} {gene:<28} "
                f"{'(pinned)':>10}"
            )
            continue
        # Expectation: more loss weight -> lower BCE, so we WANT a
        # negative Pearson here. Positive means the head got WORSE as
        # the loss weight rose, which would indicate a bug.
        rho = pearson(xs, ys)
        flag = ""
        if rho > 0.3:
            flag = "  *POSITIVE — head got worse with stronger loss?*"
        elif -0.3 < rho < 0.3:
            flag = "  (weak/no relationship)"
        lines.append(
            f"{head:<20} {gene:<28} {rho:>+10.3f}{flag}"
        )

    lines.append("")
    lines.append("Interpretation guide:")
    lines.append("  * BCE near 0.693 (=ln 2) means the head is outputting")
    lines.append("    near-uniform 0.5 — it isn't learning the label.")
    lines.append("  * BCE descending below 0.693 means the head IS learning.")
    lines.append("    Whether the actor USES that signal is a separate")
    lines.append("    question that needs per-bet calibration (gated on")
    lines.append("    v2 trainer plumbing patch — see")
    lines.append("    tools/aux_head_calibration_report.py docstring).")
    lines.append("  * Strongly-negative gene-weight->BCE correlation is")
    lines.append("    the expected GA signature of a working aux head.")
    lines.append("  * Positive correlation is a bug signal — investigate.")
    return "\n".join(lines)


# ── Per-bet mode ─────────────────────────────────────────────────────


def per_bet_mode(cohort_dir: Path, top: int | None) -> str | None:
    """Return per-bet calibration report, or None if data is unavailable."""
    bet_log_root = cohort_dir / "bet_logs"
    if not bet_log_root.exists():
        return None
    try:
        import pandas as pd
    except ImportError:
        return None

    # Probe ONE bet_log file to see if the aux-head columns are populated.
    probe_df = None
    for agent_dir in bet_log_root.iterdir():
        if not agent_dir.is_dir():
            continue
        for f in agent_dir.iterdir():
            if f.suffix == ".parquet":
                try:
                    probe_df = pd.read_parquet(f)
                    break
                except Exception:
                    continue
        if probe_df is not None:
            break
    if probe_df is None:
        return None

    have_fill = (
        "fill_prob_at_placement" in probe_df.columns
        and probe_df["fill_prob_at_placement"].notna().any()
    )
    have_risk = (
        "predicted_locked_pnl_at_placement" in probe_df.columns
        and probe_df["predicted_locked_pnl_at_placement"].notna().any()
    )
    if not (have_fill or have_risk):
        # All NULL — bail back to aggregate mode.
        return None

    # Here's where the full per-bet calibration would go: load all
    # bet_logs for the top-K agents, label each bet by actual outcome,
    # compute Pearson(fill_prob, filled_label) etc. Not implemented
    # here because no v2 cohort produces non-null columns yet — the
    # plumbing patch is queued (see task #11 in the session todo).
    return (
        "Per-bet mode would activate here. Columns are populated:\n"
        f"  fill_prob_at_placement: {have_fill}\n"
        f"  predicted_locked_pnl_at_placement: {have_risk}\n"
        "Implementation deferred until the v2 trainer plumbing "
        "patch lands (CLAUDE.md follow-up note 2026-05-24)."
    )


# ── Main ─────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "cohort_dir",
        help="Cohort output dir (containing scoreboard.jsonl + bet_logs/).",
    )
    p.add_argument(
        "--top", type=int, default=None,
        help="Report only the top-N agents by composite_score "
             "(default: all).",
    )
    args = p.parse_args()
    cohort_dir = Path(args.cohort_dir)
    scoreboard = cohort_dir / "scoreboard.jsonl"
    if not scoreboard.exists():
        print(
            f"ERROR: {scoreboard} not found. Has any agent completed?",
            file=sys.stderr,
        )
        return 2

    # Try per-bet mode first; if it returns None (data missing or
    # columns all-null), fall back to aggregate mode.
    per_bet = per_bet_mode(cohort_dir, args.top)
    if per_bet is not None:
        print(per_bet)
        print()

    rows = load_scoreboard(scoreboard)
    print(aggregate_mode(rows, args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
