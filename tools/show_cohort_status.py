"""Print a human-readable status table for a v2 cohort.

Reads <cohort_dir>/scoreboard.jsonl and prints a per-generation
summary plus per-agent top-10 + bottom-3 + ETA. Optionally writes
the same text to <cohort_dir>/status.txt for `cat`-ing without
having to re-run the script.

Usage:
    python -m tools.show_cohort_status registry/_predictor_SCALPING_pwingate_1778571007

    # With background refresh (writes to status.txt every 60s):
    python -m tools.show_cohort_status registry/.../safety_... --watch 60
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path


def _per_agent_naked_range(db_path: Path) -> dict[str, dict]:
    """Compute per-model naked stats across in-sample-eval days.

    Returns ``{model_id: {span, min, max, mean, n_days}}`` sourced
    from the ``evaluation_days`` table joined to ``evaluation_runs``.
    Empty dict when the DB is missing or has no rows yet.
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
               COUNT(*)           AS n_days
        FROM evaluation_days ed
        JOIN evaluation_runs er ON ed.run_id = er.run_id
        GROUP BY er.model_id
        """
        for r in cur.execute(sql):
            out[r["model_id"]] = {
                "evaluated_at": str(r["evaluated_at"] or ""),
                "span": float(r["naked_span"]),
                "min": float(r["naked_min"]),
                "max": float(r["naked_max"]),
                "mean": float(r["naked_mean"]),
                "n_days": int(r["n_days"]),
            }
        conn.close()
    except sqlite3.Error:
        # DB locked mid-write or schema mismatch — skip the section.
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


def _format(
    rows: list[dict], cohort_tag: str, total_target: int = 96,
    naked_range_by_model: dict[str, dict] | None = None,
) -> str:
    out: list[str] = []
    out.append(f"COHORT: {cohort_tag}")
    out.append(f"Progress: {len(rows)}/{total_target} agents")
    if not rows:
        out.append("(no rows yet)")
        return "\n".join(out)

    by_gen: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_gen[r.get("generation", 0)].append(r)
    out.append("")
    out.append("Per-generation summary:")
    out.append(f"  {'gen':<4} {'n':>4} {'mean':>8} {'median':>8} {'best':>8} {'profitable':>11} {'mean_locked':>12} {'mean_naked':>11} {'mean_bets':>10}")
    for g in sorted(by_gen):
        rs = by_gen[g]
        pnls = sorted([r.get("eval_day_pnl", 0) for r in rs])
        bets = [r.get("eval_bet_count", 0) for r in rs]
        locks = [r.get("eval_locked_pnl", 0) for r in rs]
        nakeds = [r.get("eval_naked_pnl", 0) for r in rs]
        nprof = sum(1 for p in pnls if p > 0)
        mn = sum(pnls) / len(pnls)
        med = pnls[len(pnls) // 2]
        best = pnls[-1]
        mb = sum(bets) / len(bets)
        ml = sum(locks) / len(locks)
        mnk = sum(nakeds) / len(nakeds)
        out.append(
            f"  {g:<4} {len(rs):>4} {mn:>+8.0f} {med:>+8.0f} {best:>+8.0f} "
            f"{nprof}/{len(rs):>9} {ml:>+12.0f} {mnk:>+11.0f} {mb:>10.0f}"
        )

    # Top-10 by eval_day_pnl
    ranked = sorted(rows, key=lambda r: -r.get("eval_day_pnl", 0))
    out.append("")
    out.append("Top-10 agents by eval_day_pnl:")
    out.append(f"  {'agent':<10} {'gen':>4} {'pnl':>8} {'locked':>8} {'naked':>8} {'bets':>5} {'mr':>5}")
    for r in ranked[:10]:
        opened = r.get("eval_pairs_opened", 0)
        comp = r.get("eval_arbs_completed", 0)
        clos = r.get("eval_arbs_closed", 0)
        mr = (comp + clos) / opened if opened > 0 else 0
        out.append(
            f"  {r['agent_id'][:8]:<10} {r.get('generation', 0):>4} "
            f"{r.get('eval_day_pnl', 0):>+8.0f} "
            f"{r.get('eval_locked_pnl', 0):>+8.0f} "
            f"{r.get('eval_naked_pnl', 0):>+8.0f} "
            f"{r.get('eval_bet_count', 0):>5} "
            f"{mr:>5.3f}"
        )
    out.append("")
    out.append("Bottom-3 agents by eval_day_pnl:")
    for r in ranked[-3:]:
        opened = r.get("eval_pairs_opened", 0)
        comp = r.get("eval_arbs_completed", 0)
        clos = r.get("eval_arbs_closed", 0)
        mr = (comp + clos) / opened if opened > 0 else 0
        out.append(
            f"  {r['agent_id'][:8]:<10} {r.get('generation', 0):>4} "
            f"{r.get('eval_day_pnl', 0):>+8.0f} "
            f"{r.get('eval_locked_pnl', 0):>+8.0f} "
            f"{r.get('eval_naked_pnl', 0):>+8.0f} "
            f"{r.get('eval_bet_count', 0):>5} "
            f"{mr:>5.3f}"
        )

    # Per-agent naked range across the in-sample-eval days. "Range"
    # = MAX(daily naked) - MIN(daily naked) per agent. Larger span =
    # more luck-dominated; smaller span at positive mean = the
    # structurally stable phenotype the locked-weighted selection
    # is meant to surface.
    if naked_range_by_model:
        spans = [v["span"] for v in naked_range_by_model.values()]
        out.append("")
        out.append(
            f"Per-agent naked range (n_agents={len(spans)}; "
            f"each agent spans 10 in-sample-eval days):"
        )
        out.append(
            f"  cohort-wide: min span {min(spans):.1f}, "
            f"max span {max(spans):.1f}, "
            f"mean span {sum(spans) / len(spans):.1f}"
        )
        out.append("")
        out.append("Top-10 agents by naked range (smallest span first):")
        out.append(
            f"  {'agent':<10} {'gen':>4} {'evaluated_at':<20} "
            f"{'span':>7} {'min':>8} {'max':>8} {'mean':>8} "
            f"{'locked':>8} {'pnl':>8}"
        )
        # Index scoreboard rows by model_id (== agent_id in v2).
        rows_by_id = {r.get("agent_id", ""): r for r in rows}
        # Smaller span = tighter naked variance = better. Sort ascending.
        # (Reversed 2026-05-15 for the scalping-tight-naked-variance
        # plan; previously largest-first, which surfaced the worst
        # cohort members at the top.)
        ranked = sorted(
            naked_range_by_model.items(), key=lambda kv: kv[1]["span"],
        )
        for model_id, st in ranked[:10]:
            sb = rows_by_id.get(model_id, {})
            # Trim ISO timestamp to "YYYY-MM-DD HH:MM:SS" (drop fractional
            # seconds + 'T' separator) for readability.
            ts = (st.get("evaluated_at", "") or "")[:19].replace("T", " ")
            out.append(
                f"  {model_id[:8]:<10} {sb.get('generation', 0):>4} "
                f"{ts:<20} "
                f"{st['span']:>7.1f} {st['min']:>+8.1f} "
                f"{st['max']:>+8.1f} {st['mean']:>+8.1f} "
                f"{sb.get('eval_locked_pnl', 0):>+8.0f} "
                f"{sb.get('eval_day_pnl', 0):>+8.0f}"
            )

    # ETA
    wall_times = [
        r.get("train_wall_time_sec", 0) + r.get("eval_wall_time_sec", 0)
        for r in rows
    ]
    if wall_times:
        avg_s = sum(wall_times) / len(wall_times)
        remaining = total_target - len(rows)
        eta_min = remaining * avg_s / 60.0
        out.append("")
        out.append(f"Avg wall per agent: {avg_s / 60:.1f} min")
        out.append(f"Remaining: {remaining} agents ~= {eta_min:.0f} min ({eta_min / 60:.1f} h)")

    out.append("")
    out.append(f"(Last refreshed: {time.strftime('%Y-%m-%d %H:%M:%S')})")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cohort_dir", type=Path)
    p.add_argument("--target-rows", type=int, default=96,
                   help="Total agents expected. Default 96 (12 × 8 gens).")
    p.add_argument("--watch", type=int, default=0,
                   help="Refresh every N seconds. 0 = print once and exit.")
    p.add_argument("--out", type=Path, default=None,
                   help="Also write the report to this path (defaults to "
                        "<cohort_dir>/status.txt).")
    args = p.parse_args(argv)

    cohort_dir: Path = args.cohort_dir
    if not cohort_dir.exists():
        print(f"ERROR: cohort dir not found: {cohort_dir}", file=sys.stderr)
        return 1
    scoreboard = cohort_dir / "scoreboard.jsonl"
    db_path = cohort_dir / "models.db"
    out_path = args.out or (cohort_dir / "status.txt")

    def render_once() -> None:
        rows = _read_rows(scoreboard)
        naked_range = _per_agent_naked_range(db_path)
        text = _format(
            rows,
            cohort_tag=cohort_dir.name,
            total_target=args.target_rows,
            naked_range_by_model=naked_range,
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
