"""Print a probe-shaped status table for a small (5-agent × 1-gen)
cohort run testing one reward-side lever.

Unlike ``show_cohort_status.py`` (designed for multi-gen GA cohorts),
this tool:

* Prints per-agent rows **chronologically** as agents finish — the
  trajectory across the small population is the signal.
* Diffs every metric against an operator-supplied baseline (typically
  tnv3 gen 0: pnl=-46, fc_n=54, fc_pnl=-86, cl_n=9, span=227, bets=178).
* Renders a BITE verdict per cohort-mean metric using operator-
  supplied thresholds (``--bite-fc-n``, ``--bite-cl-n`` etc.).
* No per-gen or naked-span-by-gen tables — there's only one gen.

Usage::

    python -m tools.show_probe_status \\
        registry/_predictor_SCALPING_probe_a_<ts> \\
        --probe-name "A: close_signal £1→£10" \\
        --lever "close_signal_bonus=10.0 (pinned cohort-wide)" \\
        --watch 60

Baseline + thresholds default to tnv3 gen-0 (fc=120, raceconf, 10
in-sample-eval days). Override via CLI for other comparisons.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path


def _per_agent_stats(db_path: Path) -> dict[str, dict]:
    """Pull per-model in-sample-eval stats from ``models.db``.

    Returns ``{model_id: {evaluated_at, span, naked_min/max/mean,
    fc_count_mean, fc_pnl_mean, closed_count_mean, closed_pnl_mean,
    n_days}}``. Lifted from show_cohort_status._per_agent_naked_range
    with the same schema.
    """
    if not db_path.exists():
        return {}
    out: dict[str, dict] = {}
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        sql = """
        SELECT er.model_id,
               MAX(er.evaluated_at) AS evaluated_at,
               MIN(ed.naked_pnl) AS naked_min,
               MAX(ed.naked_pnl) AS naked_max,
               AVG(ed.naked_pnl) AS naked_mean,
               (MAX(ed.naked_pnl) - MIN(ed.naked_pnl)) AS naked_span,
               AVG(ed.arbs_force_closed) AS fc_count_mean,
               AVG(ed.force_closed_pnl) AS fc_pnl_mean,
               AVG(ed.arbs_closed) AS closed_count_mean,
               AVG(ed.closed_pnl) AS closed_pnl_mean,
               COUNT(*)           AS n_days
        FROM evaluation_days ed
        JOIN evaluation_runs er ON ed.run_id = er.run_id
        GROUP BY er.model_id
        """
        for r in cur.execute(sql):
            out[r["model_id"]] = {
                "evaluated_at": str(r["evaluated_at"] or ""),
                "span": float(r["naked_span"]),
                "naked_mean": float(r["naked_mean"]),
                "fc_count_mean": float(r["fc_count_mean"] or 0.0),
                "fc_pnl_mean": float(r["fc_pnl_mean"] or 0.0),
                "closed_count_mean": float(r["closed_count_mean"] or 0.0),
                "closed_pnl_mean": float(r["closed_pnl_mean"] or 0.0),
                "n_days": int(r["n_days"]),
            }
        conn.close()
    except sqlite3.Error:
        pass
    return out


def _read_rows(scoreboard_path: Path) -> list[dict]:
    rows: list[dict] = []
    if not scoreboard_path.exists():
        return rows
    with scoreboard_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _format(
    rows: list[dict],
    per_agent: dict[str, dict],
    *,
    cohort_tag: str,
    probe_name: str,
    lever_desc: str,
    n_target: int,
    baseline: dict[str, float],
    bite_thresholds: dict[str, tuple[str, float]],
) -> str:
    """Render the probe panel.

    ``bite_thresholds`` maps metric name → (direction, target) where
    direction is one of ``"<="`` (lower is better) or ``">="`` (higher
    is better). The verdict line per metric tests ``mean_metric
    {direction} target``.
    """
    o: list[str] = []
    o.append(f"PROBE: {probe_name}")
    o.append(f"Lever: {lever_desc}")
    o.append(f"Cohort: {cohort_tag}")
    b = baseline
    o.append(
        f"Baseline (tnv3 gen 0, fc=120, raceconf, 10-day eval): "
        f"pnl={b.get('pnl', 0):+.0f}  "
        f"fc_n={b.get('fc_n', 0):.0f}  "
        f"fc_£={b.get('fc_pnl', 0):+.0f}  "
        f"cl_n={b.get('cl_n', 0):.0f}  "
        f"span={b.get('span', 0):.0f}  "
        f"bets={b.get('bets', 0):.0f}"
    )
    o.append(f"Progress: {len(rows)}/{n_target} agents")
    if not rows:
        o.append("")
        o.append("(no agents finished yet)")
        return "\n".join(o)

    # Build per-agent line data, sorted chronologically by evaluated_at.
    enriched: list[dict] = []
    for r in rows:
        mid = r.get("agent_id", "")
        st = per_agent.get(mid, {})
        evaluated_at = (st.get("evaluated_at", "") or "")[:19].replace("T", " ")
        opened = r.get("eval_pairs_opened", 0)
        comp = r.get("eval_arbs_completed", 0)
        clos = r.get("eval_arbs_closed", 0)
        mr = (comp + clos) / opened if opened > 0 else 0.0
        enriched.append({
            "agent": mid[:8],
            "done_at": evaluated_at,
            "pnl": float(r.get("eval_day_pnl", 0.0)),
            "locked": float(r.get("eval_locked_pnl", 0.0)),
            "naked": float(st.get("naked_mean", 0.0)),
            "span": float(st.get("span", 0.0)),
            "fc_n": float(st.get("fc_count_mean", 0.0)),
            "fc_pnl": float(st.get("fc_pnl_mean", 0.0)),
            "cl_n": float(st.get("closed_count_mean", 0.0)),
            "cl_pnl": float(st.get("closed_pnl_mean", 0.0)),
            "bets": float(r.get("eval_bet_count", 0)),
            "mr": mr,
        })
    enriched.sort(key=lambda d: d["done_at"])

    o.append("")
    o.append("Per-agent (chronological, top = first finished):")
    o.append(
        f"  {'done@':<20} {'agent':<10} "
        f"{'pnl':>7} {'locked':>7} {'naked':>7} {'span':>6} "
        f"{'fc_n':>5} {'fc_£':>7} {'cl_n':>5} {'cl_£':>7} "
        f"{'bets':>5} {'mr':>5}"
    )
    for d in enriched:
        o.append(
            f"  {d['done_at']:<20} {d['agent']:<10} "
            f"{d['pnl']:>+7.0f} {d['locked']:>+7.0f} "
            f"{d['naked']:>+7.0f} {d['span']:>6.0f} "
            f"{d['fc_n']:>5.0f} {d['fc_pnl']:>+7.0f} "
            f"{d['cl_n']:>5.0f} {d['cl_pnl']:>+7.0f} "
            f"{d['bets']:>5.0f} {d['mr']:>5.3f}"
        )

    # Cohort summary with deltas + bite verdict.
    means = {
        "pnl": _mean([d["pnl"] for d in enriched]),
        "locked": _mean([d["locked"] for d in enriched]),
        "fc_n": _mean([d["fc_n"] for d in enriched]),
        "fc_pnl": _mean([d["fc_pnl"] for d in enriched]),
        "cl_n": _mean([d["cl_n"] for d in enriched]),
        "cl_pnl": _mean([d["cl_pnl"] for d in enriched]),
        "span": _mean([d["span"] for d in enriched]),
        "bets": _mean([d["bets"] for d in enriched]),
    }
    o.append("")
    o.append(f"COHORT SO FAR ({len(enriched)}/{n_target}):")

    def _line(label: str, key: str, fmt: str = "{:+.1f}") -> str:
        v = means[key]
        d = v - baseline.get(key, 0.0)
        s = f"  {label:<12} mean={fmt.format(v):>9}  "
        s += f"vs baseline {fmt.format(d):>9}"
        if key in bite_thresholds:
            direction, target = bite_thresholds[key]
            ok = (v <= target) if direction == "<=" else (v >= target)
            tag = "✓ BITE" if ok else "✗ no"
            s += f"  threshold {direction} {target:+.1f}  {tag}"
        return s

    o.append(_line("pnl", "pnl"))
    o.append(_line("locked", "locked"))
    o.append(_line("fc_n", "fc_n", "{:.1f}"))
    o.append(_line("fc_£", "fc_pnl"))
    o.append(_line("cl_n", "cl_n", "{:.1f}"))
    o.append(_line("cl_£", "cl_pnl"))
    o.append(_line("naked_span", "span", "{:.1f}"))
    o.append(_line("bets", "bets", "{:.1f}"))

    # Overall verdict — all bite-thresholded metrics passed?
    if bite_thresholds:
        bites = []
        for key, (direction, target) in bite_thresholds.items():
            v = means[key]
            bites.append((v <= target) if direction == "<=" else (v >= target))
        passed = sum(bites)
        total = len(bites)
        o.append("")
        if passed == total:
            o.append(f"VERDICT (provisional): BITES on all {total}/{total} bite metrics.")
        elif passed > 0:
            o.append(
                f"VERDICT (provisional): MIXED — {passed}/{total} bite metrics passed."
            )
        else:
            o.append(f"VERDICT (provisional): NO bite ({passed}/{total} bite metrics).")

    o.append("")
    o.append(f"(Last refreshed: {time.strftime('%Y-%m-%d %H:%M:%S')})")
    return "\n".join(o)


def _parse_baseline(s: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k, _, v = tok.partition("=")
        out[k.strip()] = float(v)
    return out


def _parse_bites(s: str) -> dict[str, tuple[str, float]]:
    """Parse ``key<=val,key>=val,...`` into ``{key: (direction, val)}``."""
    out: dict[str, tuple[str, float]] = {}
    if not s:
        return out
    import re
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = re.match(r"^(\w+)\s*(<=|>=)\s*([+-]?\d+(?:\.\d+)?)$", tok)
        if not m:
            raise ValueError(f"bad bite spec {tok!r}; expected key<=val or key>=val")
        out[m.group(1)] = (m.group(2), float(m.group(3)))
    return out


def main(argv: list[str] | None = None) -> int:
    # Windows default console is cp1252; the panel contains £/✓/✗
    # characters. Force UTF-8 so launchers without PYTHONIOENCODING
    # set still render correctly.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cohort_dir", type=Path)
    p.add_argument("--probe-name", default="<unnamed>",
                   help="Display name for the probe.")
    p.add_argument("--lever", default="<unspecified>",
                   help="One-line lever description.")
    p.add_argument("--n-target", type=int, default=5,
                   help="Total agents expected (probe default 5).")
    p.add_argument(
        "--baseline",
        default="pnl=-46,fc_n=54,fc_pnl=-86,cl_n=9,span=227,bets=178,locked=88,cl_pnl=-13",
        help="Comma-separated key=value pairs. Default is tnv3 gen 0.",
    )
    p.add_argument(
        "--bites", default="",
        help="Comma-separated bite-criteria specs of form key<=val or "
             "key>=val. Example: fc_n<=45,cl_n>=14,pnl>=-50",
    )
    p.add_argument("--watch", type=int, default=0,
                   help="Refresh every N seconds. 0 = print once.")
    p.add_argument("--out", type=Path, default=None,
                   help="Write to this path (defaults <cohort_dir>/probe_status.txt).")
    args = p.parse_args(argv)

    cohort_dir: Path = args.cohort_dir
    # In --watch mode, the cohort dir might not exist yet (trainer
    # creates it on first write, ~30s after process start). Tolerate
    # absence and re-poll instead of bailing.
    if not cohort_dir.exists() and args.watch <= 0:
        print(f"ERROR: cohort dir not found: {cohort_dir}", file=sys.stderr)
        return 1
    scoreboard = cohort_dir / "scoreboard.jsonl"
    db_path = cohort_dir / "models.db"
    out_path = args.out or (cohort_dir / "probe_status.txt")
    baseline = _parse_baseline(args.baseline)
    bites = _parse_bites(args.bites)

    def render_once() -> None:
        if not cohort_dir.exists():
            print(
                f"(waiting for cohort dir {cohort_dir} — trainer not yet "
                "written first row)"
            )
            return
        rows = _read_rows(scoreboard)
        per_agent = _per_agent_stats(db_path)
        text = _format(
            rows, per_agent,
            cohort_tag=cohort_dir.name,
            probe_name=args.probe_name,
            lever_desc=args.lever,
            n_target=args.n_target,
            baseline=baseline,
            bite_thresholds=bites,
        )
        print(text)
        try:
            out_path.write_text(text, encoding="utf-8")
        except OSError as e:
            print(f"(warn: could not write {out_path}: {e})", file=sys.stderr)

    if args.watch <= 0:
        render_once()
        return 0
    print(f"Watching {scoreboard} every {args.watch}s; Ctrl+C to stop.\n")
    try:
        while True:
            render_once()
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
