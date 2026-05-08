"""Phase-15 smoke summarizer.

Extracts key metrics from a phase-15 smoke's log + scoreboard:
- Post-BC direction BCE (back, lay)
- Post-PPO direction BCE on BC oracle pool (if logged)
- Per-day PPO direction BCE
- Eval pnl, mature rate, bets per agent
- Aggregate mature rate, eval pnl

Usage::

    python -m tools.phase15_summary registry/_phase15_smoke_xxx.log
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def parse_log(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    summary: dict = {
        "agents": {},
        "log_path": str(log_path),
    }
    for line in text.splitlines():
        # BC pretrain done line includes the post-BC BCE diagnostic.
        m = re.search(
            r"Agent (\S+): BC pretrain done.*?steps=(\d+).*?"
            r"final_ce=([\d.-]+).*?post_entropy=([\d.-]+)"
            r"(?:.*?post_bc_dir_bce_back=([\d.-]+) lay=([\d.-]+))?",
            line,
        )
        if m:
            agent_id = m.group(1)
            agent = summary["agents"].setdefault(agent_id, {})
            agent["bc_steps"] = int(m.group(2))
            agent["bc_final_ce"] = float(m.group(3))
            agent["post_bc_entropy"] = float(m.group(4))
            if m.group(5):
                agent["post_bc_dir_bce_back"] = float(m.group(5))
                agent["post_bc_dir_bce_lay"] = float(m.group(6))

        # POST-PPO direction BCE on BC oracle pool diagnostic.
        m = re.search(
            r"Agent (\S+): POST-PPO direction BCE on BC oracle pool: "
            r"back=([\d.-]+) lay=([\d.-]+) \(n=(\d+)\)",
            line,
        )
        if m:
            agent_id = m.group(1)
            agent = summary["agents"].setdefault(agent_id, {})
            agent["post_ppo_dir_bce_back"] = float(m.group(2))
            agent["post_ppo_dir_bce_lay"] = float(m.group(3))
            agent["post_ppo_n"] = int(m.group(4))

        # Frozen indicator.
        if "direction_prob_head frozen post-BC" in line:
            m = re.search(r"Agent (\S+):", line)
            if m:
                agent_id = m.group(1)
                summary["agents"].setdefault(agent_id, {})["frozen"] = True

        # Per-day PPO BCE.
        m = re.search(
            r"Agent (\S+) day (\d+/\d+) \[(\S+)\] reward=([\d+.-]+) "
            r"pnl=([\d+.-]+).*?dir_bce_back=([\d.-]+) "
            r"dir_bce_lay=([\d.-]+)",
            line,
        )
        if m:
            agent_id = m.group(1)
            agent = summary["agents"].setdefault(agent_id, {})
            day = m.group(3)
            ppo_days = agent.setdefault("ppo_days", {})
            ppo_days[day] = {
                "reward": float(m.group(4)),
                "pnl": float(m.group(5)),
                "dir_bce_back": float(m.group(6)),
                "dir_bce_lay": float(m.group(7)),
            }

        # Eval per day.
        m = re.search(
            r"Agent (\S+) eval \[(\S+)\] reward=([\d+.-]+) "
            r"pnl=([\d+.-]+) bets=(\d+) precision=([\d.]+) "
            r"arbs=(\d+)/(\d+) locked=([\d+.-]+) naked=([\d+.-]+)",
            line,
        )
        if m:
            agent_id = m.group(1)
            agent = summary["agents"].setdefault(agent_id, {})
            agent["eval_day"] = m.group(2)
            agent["eval_reward"] = float(m.group(3))
            agent["eval_pnl"] = float(m.group(4))
            agent["eval_bets"] = int(m.group(5))
            agent["eval_precision"] = float(m.group(6))
            agent["eval_arbs_completed"] = int(m.group(7))
            agent["eval_arbs_naked"] = int(m.group(8))
            agent["eval_locked"] = float(m.group(9))
            agent["eval_naked"] = float(m.group(10))

    return summary


def parse_scoreboard(scoreboard_path: Path) -> list[dict]:
    if not scoreboard_path.exists():
        return []
    rows = []
    for line in scoreboard_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def render_summary(log_path: Path):
    summary = parse_log(log_path)
    # Try to find scoreboard nearby.
    log_stem = log_path.stem
    scoreboard = log_path.parent / log_stem / "scoreboard.jsonl"
    rows = parse_scoreboard(scoreboard)
    sb_by_agent = {r["agent_id"]: r for r in rows}

    print(f"\n=== Phase-15 smoke summary: {log_path.name} ===")
    if not summary["agents"]:
        print("  (no agent metrics found in log)")
        return
    print(f"\n{'agent':<14} {'bc_b':>7} {'bc_l':>7} {'pp_b':>7} {'pp_l':>7} "
          f"{'frzn':>4} {'bets':>5} {'mat':>4} {'cls':>4} {'eval_pnl':>9} "
          f"{'pairs_op':>8} {'mat%':>6}")
    for agent_id, a in summary["agents"].items():
        sb = sb_by_agent.get(agent_id, {})
        eval_pairs = sb.get("eval_pairs_opened", 0) or 0
        eval_completed = sb.get("eval_arbs_completed", 0) or 0
        eval_closed = sb.get("eval_arbs_closed", 0) or 0
        eval_force = sb.get("eval_arbs_force_closed", 0) or 0
        eval_pnl = sb.get("eval_day_pnl", a.get("eval_pnl", 0))
        n_matured = eval_completed + eval_closed
        mat_rate = (n_matured / eval_pairs * 100) if eval_pairs > 0 else 0
        bc_b = a.get("post_bc_dir_bce_back", float("nan"))
        bc_l = a.get("post_bc_dir_bce_lay", float("nan"))
        pp_b = a.get("post_ppo_dir_bce_back", float("nan"))
        pp_l = a.get("post_ppo_dir_bce_lay", float("nan"))
        frzn = "Y" if a.get("frozen") else "N"
        bets = a.get("eval_bets", 0)
        print(
            f"{agent_id[:14]:<14} {bc_b:>7.4f} {bc_l:>7.4f} "
            f"{pp_b:>7.4f} {pp_l:>7.4f} {frzn:>4} {bets:>5} "
            f"{eval_completed:>4} {eval_closed:>4} {eval_pnl:>+9.2f} "
            f"{eval_pairs:>8} {mat_rate:>5.1f}%"
        )
    # Aggregate
    all_pnl = [
        sb_by_agent.get(a, {}).get("eval_day_pnl", 0)
        for a in summary["agents"]
    ]
    if all_pnl:
        print(f"\n  mean eval pnl: {sum(all_pnl)/len(all_pnl):+.2f}")
        print(f"  positive pnl agents: "
              f"{sum(1 for p in all_pnl if p > 0)}/{len(all_pnl)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m tools.phase15_summary <log_path>")
        sys.exit(1)
    for arg in sys.argv[1:]:
        render_summary(Path(arg))
