"""Compare multiple phase-15 smoke runs side-by-side.

Usage::

    python -m tools.phase15_compare \\
        registry/_phase15_smoke_bcv2_1778278340.log \\
        registry/_phase15_smoke_md_1778279309.log
"""

from __future__ import annotations

import sys
from pathlib import Path

from tools.phase15_summary import parse_log, parse_scoreboard


def render(log_paths: list[Path]):
    print(f"\n=== Phase-15 smoke comparison ({len(log_paths)} runs) ===\n")
    print(f"{'run':<32} {'agent':<14} {'bc_b':>7} {'pp_b':>7} {'mat':>4} "
          f"{'mat%':>6} {'eval_pnl':>9} {'bets':>5}")
    for path in log_paths:
        summary = parse_log(path)
        scoreboard = path.parent / path.stem / "scoreboard.jsonl"
        sb_rows = parse_scoreboard(scoreboard)
        sb_by_agent = {r["agent_id"]: r for r in sb_rows}
        run_id = path.stem
        for agent_id, a in summary["agents"].items():
            sb = sb_by_agent.get(agent_id, {})
            pairs = sb.get("eval_pairs_opened", 0) or 0
            completed = sb.get("eval_arbs_completed", 0) or 0
            closed = sb.get("eval_arbs_closed", 0) or 0
            mat = completed + closed
            mat_rate = (mat / pairs * 100) if pairs > 0 else 0
            eval_pnl = sb.get("eval_day_pnl", a.get("eval_pnl", 0))
            bc_b = a.get("post_bc_dir_bce_back", float("nan"))
            pp_b = a.get("post_ppo_dir_bce_back", float("nan"))
            bets = a.get("eval_bets", 0)
            print(
                f"{run_id[:32]:<32} {agent_id[:14]:<14} "
                f"{bc_b:>7.4f} {pp_b:>7.4f} {mat:>4} {mat_rate:>5.1f}% "
                f"{eval_pnl:>+9.2f} {bets:>5}"
            )
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m tools.phase15_compare <log_path>...")
        sys.exit(1)
    render([Path(a) for a in sys.argv[1:]])
