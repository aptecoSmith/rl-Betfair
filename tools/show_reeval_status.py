"""Live reeval status panel.

Reads the four held-out reeval JSONLs (whichever pattern the operator
passes via --pattern), the matching .log files, and emits a single
``reeval_status.txt`` summarising:

- Which of the N jobs is currently running, and progress within it.
- For each completed agent in the current/recent jobs: a per-day
  PnL / locked / naked breakdown extracted from the JSONL's
  ``reeval_per_day`` field.
- Cumulative ETA estimate based on observed agent throughput.

Usage:
    python -m tools.show_reeval_status \
        registry/_predictor_SCALPING_tnv_raceconf_1778852093 \
        --pattern reeval_phase3b_*.jsonl \
        --watch 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except (PermissionError, OSError):
        # JSONL may be mid-write on Windows — ignore and retry next tick.
        pass
    return rows


def _read_log_progress(log_path: Path) -> dict:
    """Parse the in-progress reeval .log file. UTF-16-LE on Windows
    when written via Tee-Object; UTF-8 elsewhere. Returns the most
    recent ``[N/M] agent_id`` line and a few raw status hints.
    """
    out = {
        "current_idx": 0,
        "total": 0,
        "current_agent": "",
        "last_day": "",
        "last_line_ts": "",
    }
    if not log_path.exists():
        return out
    try:
        data = log_path.read_bytes()
    except (PermissionError, OSError):
        return out
    # UTF-16-LE with BOM is what PowerShell's Tee-Object writes.
    if data.startswith(b"\xff\xfe"):
        txt = data.decode("utf-16-le", errors="replace")
    else:
        txt = data.decode("utf-8", errors="replace")
    lines = txt.split("\n")
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        # Pattern: "2026-05-15 12:46:32,727 [N/M] agent_id_short: <msg>"
        # The reeval tool emits one such line per agent boundary.
        import re
        m = re.search(r"(\d{2}:\d{2}:\d{2}).*\[(\d+)/(\d+)\]\s+([0-9a-f-]+)", line)
        if m:
            out["last_line_ts"] = m.group(1)
            out["current_idx"] = int(m.group(2))
            out["total"] = int(m.group(3))
            out["current_agent"] = m.group(4)
            break
    # Find most recent "Loaded YYYY-MM-DD" to know which day is in flight.
    import re
    for line in reversed(lines[-200:]):
        m = re.search(r"Loaded (\d{4}-\d{2}-\d{2})", line)
        if m:
            out["last_day"] = m.group(1)
            break
    return out


def _per_day_summary(row: dict) -> list[dict]:
    return list(row.get("reeval_per_day", []) or [])


def _format_one_job(cohort_dir: Path, jsonl: Path) -> list[str]:
    rows = _read_jsonl(jsonl)
    log_path = Path(str(jsonl) + ".log")
    log = _read_log_progress(log_path)

    out: list[str] = []
    out.append("=" * 120)
    out.append(f"JOB: {jsonl.name}")
    out.append("=" * 120)
    if log["total"]:
        out.append(
            f"  Live: agent {log['current_idx']}/{log['total']}  "
            f"({log['current_agent'][:12]})  "
            f"day in flight: {log['last_day'] or '?'}  "
            f"last log: {log['last_line_ts'] or '?'}"
        )
    elif rows:
        out.append(f"  Complete: {len(rows)} rows written")
    else:
        out.append("  Not started")

    if not rows:
        return out

    out.append("")
    out.append(
        f"  {'agent':<14} {'days':>5} "
        f"{'pnl/d':>9} {'locked/d':>9} {'naked/d':>9} "
        f"{'min':>9} {'max':>9} {'prof':>6}"
    )
    for r in rows:
        per_day = _per_day_summary(r)
        n = len(per_day)
        if n == 0:
            continue
        day_pnls = [d.get("day_pnl", 0.0) for d in per_day]
        locked = [d.get("locked_pnl", 0.0) for d in per_day]
        nakeds = [d.get("naked_pnl", 0.0) for d in per_day]
        out.append(
            f"  {r['agent_id'][:12]:<14} {n:>5d} "
            f"{sum(day_pnls)/n:>+9.2f} {sum(locked)/n:>+9.2f} "
            f"{sum(nakeds)/n:>+9.2f} {min(day_pnls):>+9.2f} "
            f"{max(day_pnls):>+9.2f} "
            f"{sum(1 for p in day_pnls if p > 0)}/{n}"
        )
        # Per-day detail rows
        for d in per_day:
            out.append(
                f"      {d.get('date', '?'):<10}  "
                f"pnl={d.get('day_pnl', 0.0):>+8.2f}  "
                f"locked={d.get('locked_pnl', 0.0):>+8.2f}  "
                f"naked={d.get('naked_pnl', 0.0):>+8.2f}  "
                f"closed={d.get('closed_pnl', 0.0):>+8.2f}  "
                f"fc={d.get('force_closed_pnl', 0.0):>+7.2f}  "
                f"bets={d.get('bet_count', 0)}  "
                f"matured={d.get('arbs_completed', 0)}  "
                f"closed#={d.get('arbs_closed', 0)}  "
                f"forced#={d.get('arbs_force_closed', 0)}  "
                f"naked#={d.get('arbs_naked', 0)}"
            )
        out.append("")

    return out


def render(cohort_dir: Path, pattern: str) -> str:
    jsonls = sorted(cohort_dir.glob(pattern))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out: list[str] = []
    out.append(f"# Reeval status — {cohort_dir.name}")
    out.append(f"# Generated {now}")
    out.append(f"# Pattern: {pattern}  ({len(jsonls)} jobs)")
    out.append("")

    # Global per-day summary if jsonls present
    total_rows = 0
    completed_jobs = 0
    for j in jsonls:
        rows = _read_jsonl(j)
        n = len(rows)
        total_rows += n
        # Heuristic: job complete if at least 1 row AND launcher log says DONE.
        # Cheaper proxy: don't bother; the log line above tells progress.
    out.append(f"## Summary across {len(jsonls)} jobs")
    out.append(f"  Total rows written across all jobs: {total_rows}")
    out.append("")

    for j in jsonls:
        out += _format_one_job(cohort_dir, j)
        out.append("")

    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cohort_dir", type=Path)
    p.add_argument(
        "--pattern", default="reeval_phase3b_*.jsonl",
        help="Glob pattern for the reeval JSONLs (default reeval_phase3b_*.jsonl).",
    )
    p.add_argument(
        "--out", default=None, type=Path,
        help=(
            "Status file path. Default: <cohort_dir>/reeval_status.txt."
        ),
    )
    p.add_argument(
        "--watch", default=0, type=int,
        help=(
            "Refresh every N seconds. Default 0 = render once and exit."
        ),
    )
    args = p.parse_args(argv or sys.argv[1:])

    out_path = args.out or (args.cohort_dir / "reeval_status.txt")
    while True:
        try:
            txt = render(args.cohort_dir, args.pattern)
            out_path.write_text(txt, encoding="utf-8")
        except Exception as exc:
            # Don't crash on a transient file lock — just retry next tick.
            print(f"render error: {exc}", file=sys.stderr)
        if args.watch <= 0:
            break
        time.sleep(args.watch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
