"""Render per-tranche gauntlet leaderboards from a run's scoreboard.

The gauntlet stores raw per-agent-per-tranche rows in ``scoreboard.jsonl``
(keyed by ``tranche_K``) and per-lineage status (active/culled) in
``gauntlet_ledger.jsonl``. This joins them into a readable board PER tranche
depth, sorted by the selection metric (``composite_score`` = tnv2 by default),
marking which lineages the breeder kept vs culled.

Usage:
    python -m tools.gauntlet_leaderboard --dir registry/tt_tick_002
    python -m tools.gauntlet_leaderboard --dir registry/tt_tick_002 --tranche 1
    python -m tools.gauntlet_leaderboard --dir registry/tt_tick_002 --sort locked --top 20

Safe to run on a LIVE run (read-only) — re-run any time to watch a board fill.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

_SORTS = {
    "composite": "composite_score",
    "locked": "eval_locked_pnl",
    "naked": "eval_naked_pnl",
    "matured": "eval_arbs_completed",
    "day_pnl": "eval_day_pnl",
}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a half-written trailing line on a live run
    return out


def _ledger_status(rows: list[dict]) -> dict[str, str]:
    """lineage_id -> latest status (last snapshot wins)."""
    st: dict[str, str] = {}
    for r in rows:
        lid = r.get("lineage_id")
        if lid in (None, "__meta__", "__bred__"):
            continue
        st[lid] = r.get("status", "?")
    return st


def _fmt_board(rows: list[dict], status: dict[str, str], K: int,
               sort_key: str, top: int) -> str:
    rows = sorted(rows, key=lambda r: (r.get(sort_key) is None, -(r.get(sort_key) or 0)))
    lines = [f"\n===== TRANCHE {K} leaderboard ({len(rows)} entries; "
             f"sorted by {sort_key}) ====="]
    hdr = (f"{'#':>3} {'lineage':>8} {'orig':>6} {'status':>7} {'comp':>7} "
           f"{'locked':>7} {'naked':>8} {'mat':>4} {'opens':>5} {'fc':>3} "
           f"{'bets':>5}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    shown = rows if top <= 0 else rows[:top]
    for i, r in enumerate(shown, 1):
        lid = str(r.get("lineage_id") or r.get("agent_id") or "?")
        lines.append(
            f"{i:>3} {lid[:8]:>8} {str(r.get('origin'))[:6]:>6} "
            f"{str(status.get(lid, '?'))[:7]:>7} "
            f"{(r.get('composite_score') or 0):>7.2f} "
            f"{(r.get('eval_locked_pnl') or 0):>7.1f} "
            f"{(r.get('eval_naked_pnl') or 0):>8.1f} "
            f"{(r.get('eval_arbs_completed') or 0):>4.0f} "
            f"{(r.get('eval_pairs_opened') or 0):>5.0f} "
            f"{(r.get('eval_arbs_force_closed') or 0):>3.0f} "
            f"{(r.get('eval_bet_count') or 0):>5.0f}")
    if top > 0 and len(rows) > top:
        lines.append(f"    … {len(rows) - top} more")
    # kept-vs-culled summary (the selection sanity check)
    active = [r for r in rows if status.get(str(r.get("lineage_id"))) != "culled"]
    culled = [r for r in rows if status.get(str(r.get("lineage_id"))) == "culled"]

    def _avg(rs, k):
        return (sum((r.get(k) or 0) for r in rs) / len(rs)) if rs else 0.0
    lines.append(
        f"  KEPT  (n={len(active)}): mean locked £{_avg(active, 'eval_locked_pnl'):.1f}, "
        f"mean matured {_avg(active, 'eval_arbs_completed'):.1f}")
    lines.append(
        f"  CULLED(n={len(culled)}): mean locked £{_avg(culled, 'eval_locked_pnl'):.1f}, "
        f"mean matured {_avg(culled, 'eval_arbs_completed'):.1f}")
    return "\n".join(lines)


def _build_report(d: Path, tranche: int | None, sort_key: str, top: int) -> str:
    sb = _load_jsonl(d / "scoreboard.jsonl")
    if not sb:
        return (f"(no scoreboard rows yet in {d}/scoreboard.jsonl — "
                f"no tranche has completed)")
    status = _ledger_status(_load_jsonl(d / "gauntlet_ledger.jsonl"))
    by_t = collections.defaultdict(list)
    for r in sb:
        by_t[r.get("tranche_K")].append(r)
    tranches = [tranche] if tranche is not None else sorted(
        k for k in by_t if k is not None)
    parts = [f"run: {d}   tranches present: "
             f"{dict(sorted((k, len(v)) for k, v in by_t.items() if k is not None))}"]
    for K in tranches:
        parts.append(_fmt_board(by_t[K], status, K, sort_key, top)
                     if K in by_t else f"(no rows for tranche {K})")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="run output dir (e.g. registry/tt_tick_002)")
    ap.add_argument("--tranche", type=int, default=None, help="only this tranche depth")
    ap.add_argument("--sort", choices=list(_SORTS), default="composite")
    ap.add_argument("--top", type=int, default=20, help="rows to show (0 = all)")
    ap.add_argument("--watch", type=int, default=0, metavar="SECS",
                    help="refresh every SECS, writing <dir>/leaderboards.txt "
                         "(0 = print once to stdout). Keeps the file current "
                         "alongside a live run.")
    ap.add_argument("--out", default=None,
                    help="watch output file (default <dir>/leaderboards.txt)")
    args = ap.parse_args(argv)

    # Detached/Windows stdout is cp1252 and chokes on £/…/σ — make it tolerant
    # (the leaderboards.txt file is always written UTF-8 regardless).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    d = Path(args.dir)
    sort_key = _SORTS[args.sort]
    if args.watch <= 0:
        print(_build_report(d, args.tranche, sort_key, args.top))
        return 0

    import time
    out = Path(args.out) if args.out else d / "leaderboards.txt"
    print(f"watching {d} every {args.watch}s -> {out} (Ctrl-C to stop)")
    while True:
        report = _build_report(d, args.tranche, sort_key, args.top)
        try:
            tmp = out.with_suffix(out.suffix + ".tmp")
            tmp.write_text(report + "\n", encoding="utf-8")
            tmp.replace(out)  # atomic — a reader never sees a half-written file
        except OSError:
            pass
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
