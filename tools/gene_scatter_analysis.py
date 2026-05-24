"""Gene ->> outcome scatter analysis for v2 cohort scoreboards.

Reads a cohort's `scoreboard.jsonl`, extracts every agent's evolving
gene values + key eval outcomes, and prints:

* Pearson + Spearman correlation between each EVOLVING gene and each
  outcome (locked_pnl, naked_pnl, bets/day, mat%, fc%, cls%,
  composite_score).
* ASCII scatter plot per (gene, outcome) pair, so you can see the
  shape — linear / U-shape / no-relationship — without leaving the
  terminal.

Run as soon as gen-1 scoreboard rows land (~12h into a typical
12-agent cohort). Re-run after each generation to see whether the
breeding step shifted the gene distribution toward higher-scoring
regions.

Usage:
    python tools/gene_scatter_analysis.py <cohort_dir>
    python tools/gene_scatter_analysis.py <cohort_dir> --gen 1
    python tools/gene_scatter_analysis.py <cohort_dir> --outcome mat_pct

`--gen` filters to a single generation (default: all gens combined).
`--outcome` restricts to one outcome metric (default: print all).

A gene is "evolving" iff at least 2 distinct values are seen across the
cohort's agents — pinned genes are auto-skipped so the report only
covers things the GA can actually attribute. No matplotlib dependency
(uses ASCII scatter so it works over SSH / inside Claude Code).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable


# ── Per-row extraction ──────────────────────────────────────────────

# The five outcome metrics we report on. Each is (column_label,
# extractor that takes a scoreboard row dict).
OUTCOME_EXTRACTORS: dict[str, callable] = {
    "locked_pnl": lambda r: r.get("eval_locked_pnl", 0.0),
    "naked_pnl": lambda r: r.get("eval_naked_pnl", 0.0),
    "bets_per_day": lambda r: r.get("eval_bet_count", 0)
    / max(len(r.get("eval_days", [])) or 1, 1),
    "mat_pct": lambda r: (
        (r.get("eval_arbs_completed", 0) /
         max(r.get("eval_pairs_opened", 1), 1)) * 100.0
    ),
    "cls_pct": lambda r: (
        (r.get("eval_arbs_closed", 0) /
         max(r.get("eval_pairs_opened", 1), 1)) * 100.0
    ),
    "fc_pct": lambda r: (
        (r.get("eval_arbs_force_closed", 0) /
         max(r.get("eval_pairs_opened", 1), 1)) * 100.0
    ),
    "composite_score": lambda r: r.get("composite_score", 0.0),
}


def load_rows(scoreboard_path: Path) -> list[dict]:
    rows: list[dict] = []
    with scoreboard_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def find_evolving_genes(rows: list[dict]) -> list[str]:
    """Return gene names that show variation across the cohort.

    Inspects the ``hyperparameters`` dict on every row, collects all
    keys with numeric values, and returns those where ≥2 distinct
    values are seen. Pinned cohort-wide genes (e.g. mark_to_market_weight
    at 0.05 for every agent) are auto-skipped.
    """
    if not rows:
        return []
    seen_values: dict[str, set] = {}
    for r in rows:
        hp = r.get("hyperparameters", {})
        for k, v in hp.items():
            if not isinstance(v, (int, float)):
                continue
            seen_values.setdefault(k, set()).add(round(float(v), 9))
    return sorted(k for k, vs in seen_values.items() if len(vs) >= 2)


# ── Statistics ──────────────────────────────────────────────────────


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


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank-based correlation; robust to outliers and nonlinearity."""
    n = len(xs)
    if n < 2:
        return float("nan")

    def rank(vals: list[float]) -> list[float]:
        # Average ranks for ties (standard fractional ranking).
        indexed = sorted(enumerate(vals), key=lambda p: p[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed average
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    return pearson(rank(xs), rank(ys))


# ── ASCII scatter ───────────────────────────────────────────────────


def ascii_scatter(
    xs: list[float],
    ys: list[float],
    *,
    width: int = 60,
    height: int = 12,
    x_label: str = "",
    y_label: str = "",
) -> str:
    """Render a tiny ASCII scatter, suitable for terminal/log dumps."""
    if not xs:
        return "(no data)"
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min:
        x_max = x_min + 1e-9
    if y_max == y_min:
        y_max = y_min + 1e-9
    grid = [[" "] * width for _ in range(height)]
    for x, y in zip(xs, ys):
        col = int((x - x_min) / (x_max - x_min) * (width - 1))
        row = height - 1 - int(
            (y - y_min) / (y_max - y_min) * (height - 1)
        )
        if grid[row][col] == " ":
            grid[row][col] = "."
        elif grid[row][col] == ".":
            grid[row][col] = "o"
        else:
            grid[row][col] = "O"
    lines = []
    for i, row in enumerate(grid):
        if i == 0:
            prefix = f"{y_max:>8.3f} | "
        elif i == height - 1:
            prefix = f"{y_min:>8.3f} | "
        else:
            prefix = " " * 8 + " | "
        lines.append(prefix + "".join(row))
    lines.append(" " * 8 + " +" + "-" * width)
    footer = " " * 10
    footer += f"{x_min:.4f}"
    pad = width - len(footer) + 10 - len(f"{x_max:.4f}")
    footer += " " * max(pad, 1) + f"{x_max:.4f}"
    lines.append(footer)
    if x_label or y_label:
        lines.append(
            " " * 8 + f"   x = {x_label}   y = {y_label}"
        )
    return "\n".join(lines)


# ── Reporting ────────────────────────────────────────────────────────


def report(
    rows: list[dict],
    *,
    outcomes: list[str] | None = None,
    gen_filter: int | None = None,
    no_scatter: bool = False,
) -> str:
    if gen_filter is not None:
        rows = [r for r in rows if r.get("generation") == gen_filter]
    if not rows:
        return (
            f"(no rows match filter gen={gen_filter})"
            if gen_filter is not None
            else "(scoreboard.jsonl empty — no completed agents yet)"
        )

    evolving = find_evolving_genes(rows)
    if not evolving:
        return (
            "(no evolving genes detected — every gene is pinned at "
            "the same value across the cohort. Check --enable-gene "
            "flags.)"
        )

    if outcomes is None:
        outcomes = list(OUTCOME_EXTRACTORS.keys())

    lines: list[str] = []
    gen_label = "all gens" if gen_filter is None else f"gen {gen_filter}"
    lines.append(
        f"Gene ->> outcome analysis ({len(rows)} agents, {gen_label})"
    )
    lines.append("=" * 70)
    lines.append(f"Evolving genes: {', '.join(evolving)}")
    lines.append("")

    # Correlation table — one row per (gene, outcome) pair.
    lines.append(
        f"{'gene':<32} {'outcome':<18} {'pearson':>8} {'spearman':>9}"
    )
    lines.append("-" * 70)
    for gene in evolving:
        xs = [float(r["hyperparameters"][gene]) for r in rows]
        for out_name in outcomes:
            ys = [float(OUTCOME_EXTRACTORS[out_name](r)) for r in rows]
            p = pearson(xs, ys)
            s = spearman(xs, ys)
            lines.append(
                f"{gene:<32} {out_name:<18} {p:>+8.3f} {s:>+9.3f}"
            )
        lines.append("-" * 70)

    if no_scatter:
        return "\n".join(lines)

    # Scatter plots for any (gene, outcome) pair where |spearman| ≥ 0.30
    # — heuristic for "worth a visual look".
    lines.append("")
    lines.append("Scatter plots for |spearman| ≥ 0.30:")
    lines.append("")
    plotted = 0
    for gene in evolving:
        xs = [float(r["hyperparameters"][gene]) for r in rows]
        for out_name in outcomes:
            ys = [float(OUTCOME_EXTRACTORS[out_name](r)) for r in rows]
            s = spearman(xs, ys)
            if math.isnan(s) or abs(s) < 0.30:
                continue
            lines.append(
                f">>> {gene} vs {out_name}  (spearman={s:+.3f})"
            )
            lines.append(
                ascii_scatter(
                    xs, ys,
                    x_label=gene,
                    y_label=out_name,
                )
            )
            lines.append("")
            plotted += 1
    if plotted == 0:
        lines.append(
            "(no gene-outcome pair crosses |spearman|=0.30 threshold "
            "— either the cohort isn't differentiated yet, or the "
            "evolving genes aren't the levers for these outcomes.)"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Gene scatter analysis for v2 cohorts. Reports per-gene "
            "Pearson + Spearman correlations against key outcomes and "
            "draws ASCII scatter plots for any strong relationships."
        ),
    )
    p.add_argument(
        "cohort_dir",
        help="Cohort output dir (containing scoreboard.jsonl).",
    )
    p.add_argument(
        "--gen", type=int, default=None,
        help="Restrict to a single generation (default: all gens).",
    )
    p.add_argument(
        "--outcome", default=None,
        choices=sorted(OUTCOME_EXTRACTORS.keys()),
        help="Restrict to one outcome metric (default: all).",
    )
    p.add_argument(
        "--no-scatter", action="store_true",
        help="Omit ASCII scatter plots, print correlation table only.",
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

    rows = load_rows(scoreboard)
    outcomes = [args.outcome] if args.outcome else None
    print(report(
        rows,
        outcomes=outcomes,
        gen_filter=args.gen,
        no_scatter=args.no_scatter,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
