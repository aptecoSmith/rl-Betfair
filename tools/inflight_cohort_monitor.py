"""In-flight cohort health monitor.

Tails a v2 cohort log every N seconds and emits ONE compact line that
tells the operator:

* which agent / day is currently training (so pace is visible)
* the latest per-update aux-head BCE values + KL + n_updates
* health flags (BCE stuck at ~0.69 = uncalibrated; KL exploding;
  policy/value loss out of range)

Designed for the case "12-agent × 3-gen × 16-day cohort, ~28h wall,
I don't want to wait 12h for the first scoreboard row before knowing
if it's working." Reads only the .log file + episodes.jsonl — no
runtime hooks, safe to run alongside the training job.

Usage:
    python tools/inflight_cohort_monitor.py <cohort_dir> [--every 600]

`<cohort_dir>` is the cohort output dir; the log path is
`<cohort_dir>.log` (sibling). `--every` defaults to 600s = 10 min.

Emits to stdout, one line per refresh. Pipe into Tee if you want
on-disk history.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# ── regexes over the cohort log ──────────────────────────────────────

# Generation header — "── Generation 2/3 ──" (with unicode dashes, often
# escaped to "──" in our log because the operator's PowerShell
# console doesn't render unicode). Match both.
RE_GEN = re.compile(
    r"(?:──|──|\\u2500\\u2500) Generation (\d+)/(\d+)"
)

# Agent gene-print line (fires for ALL agents up-front during gen
# init, BEFORE training starts) — "Generation 1 agent 3/12
# (id=10afb009-39c) genes=..."
RE_AGENT_GENE_PRINT = re.compile(
    r"Generation (\d+) agent (\d+)/(\d+) \(id=([0-9a-f-]+)\)"
)

# Actual training-start marker — fires when an agent starts loading
# data for its first day. "Agent <id>: loading first day YYYY-MM-DD"
RE_AGENT_TRAINING_START = re.compile(
    r"Agent ([0-9a-f-]+): loading first day (\d{4}-\d{2}-\d{2})"
)

# Per-day completion — "Agent <id> day 3/16 [2026-04-15] reward=-X pnl=Y ..."
# Phase-15 (2026-05-24): optional gate_refusals=N + arb_realised_lock=±X.XXX|nan
# captured when the cohort's worker emits them (newer runs).
RE_DAY_DONE = re.compile(
    r"Agent ([0-9a-f-]+) day (\d+)/(\d+) \[(\d{4}-\d{2}-\d{2})\] "
    r"reward=([+-]?\d+\.\d+) pnl=([+-]?\d+\.\d+).*?wall=([\d.]+)s"
)
RE_GATE_REFUSALS = re.compile(r"gate_refusals=(\d+)")
RE_ARB_REALISED = re.compile(r"arb_realised_lock=([+-]?\d+\.\d+|nan)")

# Per-update PPO log — "DiscretePPOTrainer episode: n_steps=N
# n_updates=N policy_loss=X value_loss=X entropy=X approx_kl=X
# fill_prob_bce_mean=X mature_prob_bce_mean=X risk_nll_mean=X
# total_reward=X n_mature_targets=N wall=Xs"
RE_PPO_UPDATE = re.compile(
    r"DiscretePPOTrainer episode: n_steps=(\d+) n_updates=(\d+) "
    r"policy_loss=([+-]?\d+\.\d+) value_loss=([+-]?\d+\.\d+) "
    r"entropy=([+-]?\d+\.\d+) approx_kl=([+-]?\d+\.\d+) "
    r"fill_prob_bce_mean=([+-]?\d+\.\d+) "
    r"mature_prob_bce_mean=([+-]?\d+\.\d+) "
    r"risk_nll_mean=([+-]?\d+\.\d+) "
    r"total_reward=([+-]?\d+\.\d+) "
    r"n_mature_targets=(\d+) wall=([\d.]+)s"
)

# BC pretrain completion — "Agent <id>: BC pretrain done ... post_entropy=X
# (warmup_eps=5, bc_lr=Y) post_bc_dir_bce_back=X lay=X (n=N)"
RE_BC_DONE = re.compile(
    r"Agent ([0-9a-f-]+): BC pretrain done . steps=(\d+) samples=(\d+) "
    r"final_ce=([+-]?\d+\.\d+) post_entropy=([+-]?\d+\.\d+)"
)

# Eval line — "Agent <id> eval AGGREGATE across N days: reward=X pnl=Y
# bets=N arbs=A/B locked=X naked=Y (wall_sum=Zs)"
RE_EVAL_AGGREGATE = re.compile(
    r"Agent ([0-9a-f-]+) eval AGGREGATE across (\d+) days: "
    r"reward=([+-]?\d+\.\d+) pnl=([+-]?\d+\.\d+) bets=(\d+) "
    r"arbs=(\d+)/(\d+) locked=([+-]?\d+\.\d+) naked=([+-]?\d+\.\d+)"
)

# Cohort completion — "Cohort complete in Ns. Wrote ..."
RE_COHORT_DONE = re.compile(r"Cohort complete in ([\d.]+)s")


# ── thresholds for health flags ──────────────────────────────────────

# An aux-head BCE stuck near ln(2) ≈ 0.693 means the head's output
# is near uniform = it's not learning the label. We flag if the LATEST
# value is within ±0.02 of 0.693.
BCE_UNCALIBRATED_LO = 0.673
BCE_UNCALIBRATED_HI = 0.713

# KL > 1.0 on a single update means the PPO step jumped the policy
# distribution by a factor of e; >5 is dangerous.
KL_WARN = 1.0
KL_DANGER = 5.0

# value_loss > 100 means the value head is chasing wildly mis-scaled
# rewards (typically a reward-centering units bug; see CLAUDE.md
# 2026-04-18 lesson).
VALUE_LOSS_WARN = 100.0


def parse_log(log_path: Path) -> dict:
    """Single pass over the log; return latest-state dict."""
    state: dict = {
        "current_gen": None,
        "current_gen_total": None,
        "current_agent_idx": None,
        "current_agent_total": None,
        "current_agent_id": None,
        "last_day_idx": None,
        "last_day_total": None,
        "last_day_date": None,
        "last_day_wall": None,
        "last_day_pnl": None,
        "last_day_reward": None,
        "last_gate_refusals": None,
        "last_arb_realised_lock": None,
        "last_update": None,  # tuple of PPO update values
        "last_bc_done": None,
        "agents_complete": 0,  # eval_aggregate count
        "cohort_complete_wall": None,
        "log_size": 0,
    }
    # Map agent_id-prefix → (gen, idx, total) from the gene-print
    # enumeration so we can resolve idx from training events. The
    # gene-print line truncates the id to ~11 chars for display
    # ("57458819-428") while training events carry the full UUID
    # ("57458819-4285-4b8b-b8ab-a577507532dc"). Key the dict by the
    # truncated form; on lookup, truncate the full id to the prefix
    # length we stored.
    id_to_idx: dict[str, tuple[int, int, int]] = {}

    def _lookup_idx(full_id: str) -> tuple[int, int, int] | None:
        # Try exact match first (cheap), then prefix scan.
        if full_id in id_to_idx:
            return id_to_idx[full_id]
        for prefix, val in id_to_idx.items():
            if full_id.startswith(prefix):
                return val
        return None
    if not log_path.exists():
        return state
    state["log_size"] = log_path.stat().st_size
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            # Strip leading timestamp like "2026-05-24 07:48:31,449 "
            line = raw
            m = RE_GEN.search(line)
            if m:
                state["current_gen"] = int(m.group(1))
                state["current_gen_total"] = int(m.group(2))
                continue
            m = RE_AGENT_GENE_PRINT.search(line)
            if m:
                # Record id→idx mapping only; do NOT update "current"
                # — every agent in the gen gets gene-printed up-front
                # before training starts.
                gen = int(m.group(1))
                idx = int(m.group(2))
                total = int(m.group(3))
                agent_id = m.group(4)
                id_to_idx[agent_id] = (gen, idx, total)
                if state["current_gen"] is None:
                    state["current_gen"] = gen
                continue
            m = RE_AGENT_TRAINING_START.search(line)
            if m:
                agent_id = m.group(1)
                lookup = _lookup_idx(agent_id)
                if lookup is not None:
                    gen, idx, total = lookup
                    state["current_gen"] = gen
                    state["current_agent_idx"] = idx
                    state["current_agent_total"] = total
                state["current_agent_id"] = agent_id
                # New agent — clear day cursor
                state["last_day_idx"] = None
                state["last_day_total"] = None
                state["last_day_date"] = None
                state["last_update"] = None
                continue
            m = RE_DAY_DONE.search(line)
            if m:
                agent_id = m.group(1)
                lookup = _lookup_idx(agent_id)
                if lookup is not None:
                    gen, idx, total = lookup
                    state["current_gen"] = gen
                    state["current_agent_idx"] = idx
                    state["current_agent_total"] = total
                state["current_agent_id"] = agent_id
                state["last_day_idx"] = int(m.group(2))
                state["last_day_total"] = int(m.group(3))
                state["last_day_date"] = m.group(4)
                state["last_day_reward"] = float(m.group(5))
                state["last_day_pnl"] = float(m.group(6))
                state["last_day_wall"] = float(m.group(7))
                gr = RE_GATE_REFUSALS.search(line)
                if gr:
                    state["last_gate_refusals"] = int(gr.group(1))
                ar = RE_ARB_REALISED.search(line)
                if ar:
                    raw = ar.group(1)
                    state["last_arb_realised_lock"] = (
                        float(raw) if raw != "nan" else float("nan")
                    )
                continue
            m = RE_PPO_UPDATE.search(line)
            if m:
                state["last_update"] = {
                    "n_steps": int(m.group(1)),
                    "n_updates": int(m.group(2)),
                    "policy_loss": float(m.group(3)),
                    "value_loss": float(m.group(4)),
                    "entropy": float(m.group(5)),
                    "approx_kl": float(m.group(6)),
                    "fill_bce": float(m.group(7)),
                    "mature_bce": float(m.group(8)),
                    "risk_nll": float(m.group(9)),
                    "total_reward": float(m.group(10)),
                    "n_mature_targets": int(m.group(11)),
                    "wall": float(m.group(12)),
                }
                continue
            m = RE_BC_DONE.search(line)
            if m:
                state["last_bc_done"] = {
                    "agent_id": m.group(1),
                    "steps": int(m.group(2)),
                    "samples": int(m.group(3)),
                    "final_ce": float(m.group(4)),
                    "post_entropy": float(m.group(5)),
                }
                continue
            if RE_EVAL_AGGREGATE.search(line):
                state["agents_complete"] += 1
                continue
            m = RE_COHORT_DONE.search(line)
            if m:
                state["cohort_complete_wall"] = float(m.group(1))
    return state


def fmt_bce_with_flag(value: float, label: str) -> str:
    flag = ""
    if BCE_UNCALIBRATED_LO <= value <= BCE_UNCALIBRATED_HI:
        flag = "*FLAT*"
    return f"{label}={value:.3f}{flag}"


def fmt_kl_with_flag(value: float) -> str:
    if value >= KL_DANGER:
        return f"kl={value:.3f} *DANGER*"
    if value >= KL_WARN:
        return f"kl={value:.3f} *HIGH*"
    return f"kl={value:.3f}"


def fmt_value_loss_with_flag(value: float) -> str:
    if value >= VALUE_LOSS_WARN:
        return f"v_loss={value:.1f} *EXPLODE*"
    return f"v_loss={value:.3f}"


def render(state: dict) -> str:
    if state["cohort_complete_wall"] is not None:
        return (
            f"[COHORT COMPLETE] wall={state['cohort_complete_wall']:.0f}s "
            f"({state['cohort_complete_wall']/3600:.1f}h) "
            f"agents_evaluated={state['agents_complete']}"
        )

    parts: list[str] = []
    if state["current_gen"] is not None:
        parts.append(
            f"gen={state['current_gen']}/{state['current_gen_total']}"
        )
    if state["current_agent_idx"] is not None:
        parts.append(
            f"agent={state['current_agent_idx']}/"
            f"{state['current_agent_total']}"
        )
    if state["last_day_idx"] is not None:
        parts.append(
            f"day={state['last_day_idx']}/{state['last_day_total']} "
            f"[{state['last_day_date']}]"
        )
    if state["last_day_wall"] is not None:
        parts.append(f"last_day_wall={state['last_day_wall']:.0f}s")
    if state["last_day_pnl"] is not None:
        parts.append(f"last_day_pnl={state['last_day_pnl']:+.1f}")
    if state.get("last_gate_refusals") is not None:
        # *HIGH* if more than ~half of total opens are gate-refused;
        # this is a rough heuristic — calibrate from data once we see
        # typical numbers.
        gr = state["last_gate_refusals"]
        parts.append(f"gate_ref={gr}")
    if state.get("last_arb_realised_lock") is not None:
        arl = state["last_arb_realised_lock"]
        if arl != arl:  # NaN
            parts.append("arb_lock=nan")
        else:
            parts.append(f"arb_lock={arl:+.3f}")
    if state["last_bc_done"] is not None and state["last_day_idx"] is None:
        bc = state["last_bc_done"]
        parts.append(
            f"BC done: steps={bc['steps']} "
            f"post_entropy={bc['post_entropy']:.2f} "
            f"post_ce={bc['final_ce']:.2f}"
        )
    if state["last_update"] is not None:
        u = state["last_update"]
        parts.append(fmt_kl_with_flag(u["approx_kl"]))
        parts.append(fmt_value_loss_with_flag(u["value_loss"]))
        parts.append(fmt_bce_with_flag(u["fill_bce"], "fill_bce"))
        parts.append(fmt_bce_with_flag(u["mature_bce"], "mat_bce"))
        parts.append(f"risk_nll={u['risk_nll']:+.2f}")
        parts.append(f"n_mat_tgt={u['n_mature_targets']}")
        parts.append(f"n_upd={u['n_updates']}")
    parts.append(f"completed_agents={state['agents_complete']}")
    return " | ".join(parts) if parts else "(no progress lines yet)"


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "In-flight cohort health monitor. Tails the cohort log "
            "and emits a compact health snapshot every N seconds."
        ),
    )
    p.add_argument(
        "cohort_dir",
        help="Cohort output directory (the .log is its sibling).",
    )
    p.add_argument(
        "--every", type=int, default=600,
        help="Refresh interval in seconds (default 600 = 10 min).",
    )
    p.add_argument(
        "--once", action="store_true",
        help="Emit one snapshot then exit (for ad-hoc checks).",
    )
    args = p.parse_args()

    cohort_dir = Path(args.cohort_dir)
    log_path = Path(str(cohort_dir).rstrip("\\/") + ".log")
    if not log_path.exists():
        # Maybe the operator passed the log path directly.
        log_path = cohort_dir
    if not log_path.exists():
        print(f"ERROR: log not found at {log_path}", file=sys.stderr)
        return 2

    while True:
        state = parse_log(log_path)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        snapshot = render(state)
        print(f"[{ts}] {snapshot}", flush=True)
        if args.once or state["cohort_complete_wall"] is not None:
            return 0
        time.sleep(args.every)


if __name__ == "__main__":
    raise SystemExit(main())
