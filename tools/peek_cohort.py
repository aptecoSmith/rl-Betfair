"""Live per-agent visibility into a running (or completed) cohort.

Reads ``models.db`` and ``scoreboard.jsonl`` from a cohort output dir
and surfaces the verdict-bar metrics every Phase-3-followups plan
needs:

- per-agent ``day_pnl`` and ``bet_count``
- per-pair lifecycle counts (``matured``, ``closed``, ``stop_closed``,
  ``force_closed``, ``naked``)
- ``policy-close fraction`` (pcf) and ``stop-close fraction`` (scf)
  with the **TRUE denominator** including stop-closed pairs (which
  the old ``naked / (matured + naked)`` proxy folded into naked)
- aggregate verdict-bar metrics (mean fc_rate, positive eval P&L count)

Sources picked per-agent:

- Prefer ``models.db`` ``evaluation_days`` row (post-cohort-visibility
  S01b carries the full breakdown). Falls back to ``scoreboard.jsonl``
  for runs whose db pre-dates the schema migration.
- Either source updates per-agent as the cohort runs (post-cohort-
  visibility S01a writes ``scoreboard.jsonl`` per-agent in sequential
  mode); pre-plan runs flushed the JSONL only at end-of-generation.

Usage:
    python -m tools.peek_cohort <run_dir>
    python -m tools.peek_cohort --json <run_dir>      # machine-readable

See plans/rewrite/phase-3-followups/cohort-visibility/.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentSnapshot:
    """One agent's live readout. Numeric fields default 0 / 0.0 so a
    pre-S01b row with NULLs on the new columns reads safely."""

    agent_id: str
    eval_day: str
    day_pnl: float = 0.0
    bet_count: int = 0
    bet_precision: float = 0.0
    arbs_completed: int = 0
    arbs_closed: int = 0
    arbs_stop_closed: int = 0
    arbs_force_closed: int = 0
    arbs_naked: int = 0
    pairs_opened: int = 0
    locked_pnl: float = 0.0
    closed_pnl: float = 0.0
    stop_closed_pnl: float = 0.0
    force_closed_pnl: float = 0.0
    naked_pnl: float = 0.0
    source: str = "?"  # "db" or "jsonl"

    @property
    def total_outcomes(self) -> int:
        """All pair outcomes (matured naturally + agent-closed +
        stop-closed + force-closed + naked). Denominator for the
        TRUE fc_rate, pcf, scf."""
        return (
            self.arbs_completed + self.arbs_closed
            + self.arbs_stop_closed + self.arbs_force_closed
            + self.arbs_naked
        )

    @property
    def fc_rate(self) -> float | None:
        """Naked-outcome fraction (the operator's "fc_rate" from
        AMBER v2 / Session 01 findings — pairs that never matured
        and weren't policy- or env-closed). The verdict-bar metric.
        Compare directly with AMBER v2's 0.809 / S01's 0.821."""
        d = self.total_outcomes
        return (self.arbs_naked / d) if d else None

    @property
    def policy_close_fraction(self) -> float | None:
        """Agent-initiated closes as a fraction of all pair outcomes.
        Session 01 metric — measures whether the policy is using
        ``close_signal`` instead of leaving pairs naked. AMBER v2 ~0,
        S01 0.255."""
        d = self.total_outcomes
        return (self.arbs_closed / d) if d else None

    @property
    def stop_close_fraction(self) -> float | None:
        """Env-initiated stop-closes as a fraction of all pair
        outcomes. Session 02 metric. AMBER v2 / S01 = 0; S02 useful
        range is 0.10–0.30 (per purpose.md §"Session 02"). > 0.50
        means the threshold is too tight (closing pairs that would
        have matured)."""
        d = self.total_outcomes
        return (self.arbs_stop_closed / d) if d else None


def _read_db(run_dir: Path) -> dict[str, AgentSnapshot]:
    """Build {model_id: AgentSnapshot} from ``models.db``. Returns
    {} if the db doesn't exist or is empty."""
    db = run_dir / "models.db"
    if not db.exists():
        return {}
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT m.model_id, ed.*
            FROM evaluation_days ed
            JOIN evaluation_runs er ON er.run_id = ed.run_id
            JOIN models m ON m.model_id = er.model_id
            ORDER BY m.created_at ASC
        """).fetchall()
    except sqlite3.OperationalError:
        return {}
    out: dict[str, AgentSnapshot] = {}
    for r in rows:
        keys = set(r.keys())

        def _g(name, default=0):
            return (r[name] if name in keys and r[name] is not None
                    else default)

        snap = AgentSnapshot(
            agent_id=r["model_id"],
            eval_day=r["date"],
            day_pnl=float(_g("day_pnl", 0.0)),
            bet_count=int(_g("bet_count", 0)),
            bet_precision=float(_g("bet_precision", 0.0)),
            arbs_completed=int(_g("arbs_completed", 0)),
            arbs_closed=int(_g("arbs_closed", 0)),
            arbs_stop_closed=int(_g("arbs_stop_closed", 0)),
            arbs_force_closed=int(_g("arbs_force_closed", 0)),
            arbs_naked=int(_g("arbs_naked", 0)),
            pairs_opened=int(_g("pairs_opened", 0)),
            locked_pnl=float(_g("locked_pnl", 0.0)),
            closed_pnl=float(_g("closed_pnl", 0.0)),
            stop_closed_pnl=float(_g("stop_closed_pnl", 0.0)),
            force_closed_pnl=float(_g("force_closed_pnl", 0.0)),
            naked_pnl=float(_g("naked_pnl", 0.0)),
            source="db",
        )
        out[snap.agent_id] = snap
    return out


def _read_jsonl(run_dir: Path) -> dict[str, AgentSnapshot]:
    """Build {agent_id: AgentSnapshot} from ``scoreboard.jsonl``.
    Returns {} if the file doesn't exist or is empty."""
    path = run_dir / "scoreboard.jsonl"
    if not path.exists():
        return {}
    out: dict[str, AgentSnapshot] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        agent_id = row.get("agent_id") or row.get("model_id")
        if not agent_id:
            continue
        out[agent_id] = AgentSnapshot(
            agent_id=agent_id,
            eval_day=row.get("eval_day", ""),
            day_pnl=float(row.get("eval_day_pnl", 0.0) or 0.0),
            bet_count=int(row.get("eval_bet_count", 0) or 0),
            bet_precision=float(row.get("eval_bet_precision", 0.0) or 0.0),
            arbs_completed=int(row.get("eval_arbs_completed", 0) or 0),
            arbs_closed=int(row.get("eval_arbs_closed", 0) or 0),
            arbs_stop_closed=int(row.get("eval_arbs_stop_closed", 0) or 0),
            arbs_force_closed=int(row.get("eval_arbs_force_closed", 0) or 0),
            arbs_naked=int(row.get("eval_arbs_naked", 0) or 0),
            pairs_opened=int(row.get("eval_pairs_opened", 0) or 0),
            locked_pnl=float(row.get("eval_locked_pnl", 0.0) or 0.0),
            closed_pnl=float(row.get("eval_closed_pnl", 0.0) or 0.0),
            stop_closed_pnl=float(row.get("eval_stop_closed_pnl", 0.0) or 0.0),
            force_closed_pnl=float(
                row.get("eval_force_closed_pnl", 0.0) or 0.0
            ),
            naked_pnl=float(row.get("eval_naked_pnl", 0.0) or 0.0),
            source="jsonl",
        )
    return out


def collect_snapshots(run_dir: Path) -> list[AgentSnapshot]:
    """Merge db + jsonl sources, preferring whichever has more
    information per agent. Returns insertion-order list (DB order
    preferred since it's chronological by created_at; falls back to
    JSONL order for agents missing from the db)."""
    db_snaps = _read_db(run_dir)
    jsonl_snaps = _read_jsonl(run_dir)
    seen: dict[str, AgentSnapshot] = {}
    for agent_id, snap in db_snaps.items():
        # Prefer the source with more lifecycle counts. On a post-S01b
        # cohort, db carries everything. On a pre-S01b cohort the db
        # has zeros for the new columns; jsonl may have them.
        if agent_id in jsonl_snaps:
            jsonl_snap = jsonl_snaps[agent_id]
            db_total = snap.total_outcomes
            jsonl_total = jsonl_snap.total_outcomes
            if jsonl_total > db_total:
                seen[agent_id] = jsonl_snap
                continue
        seen[agent_id] = snap
    # Pull in any JSONL-only agents (rare — only if db write failed).
    for agent_id, snap in jsonl_snaps.items():
        seen.setdefault(agent_id, snap)
    return list(seen.values())


def _fmt_frac(x: float | None) -> str:
    return "  -  " if x is None else f"{x:.3f}"


def _fmt_pnl(x: float) -> str:
    return f"{x:+10.2f}"


def render_table(snaps: list[AgentSnapshot]) -> str:
    """Pretty-printed verdict table for stdout. ASCII-safe (Windows
    CMD compatible)."""
    if not snaps:
        return "No agents complete yet (models.db / scoreboard.jsonl empty)."

    lines: list[str] = []
    header = (
        f"{'agent':14} {'eval':12} "
        f"{'day_pnl':>10} {'bets':>5} "
        f"{'matured':>8} {'closed':>7} {'stop':>5} {'forced':>7} {'naked':>6} "
        f"{'fc_rate':>8} {'pcf':>6} {'scf':>6} "
        f"{'locked':>10} {'naked_pnl':>11}"
    )
    sep = "-" * len(header)
    lines.append(header)
    lines.append(sep)

    pos_pnl = 0
    fc_rates: list[float] = []
    pcfs: list[float] = []
    scfs: list[float] = []
    pnls: list[float] = []
    for s in snaps:
        if s.day_pnl > 0:
            pos_pnl += 1
        if s.fc_rate is not None:
            fc_rates.append(s.fc_rate)
        if s.policy_close_fraction is not None:
            pcfs.append(s.policy_close_fraction)
        if s.stop_close_fraction is not None:
            scfs.append(s.stop_close_fraction)
        pnls.append(s.day_pnl)
        lines.append(
            f"{s.agent_id[:12]:14} {s.eval_day:12} "
            f"{_fmt_pnl(s.day_pnl)} {s.bet_count:>5} "
            f"{s.arbs_completed:>8} {s.arbs_closed:>7} "
            f"{s.arbs_stop_closed:>5} {s.arbs_force_closed:>7} "
            f"{s.arbs_naked:>6} "
            f"{_fmt_frac(s.fc_rate):>8} "
            f"{_fmt_frac(s.policy_close_fraction):>6} "
            f"{_fmt_frac(s.stop_close_fraction):>6} "
            f"{_fmt_pnl(s.locked_pnl)} {_fmt_pnl(s.naked_pnl)}"
        )
    lines.append(sep)

    n = len(snaps)
    if fc_rates:
        mean_fc = sum(fc_rates) / len(fc_rates)
        lines.append(
            f"# mean fc_rate (TRUE denominator): {mean_fc:.3f}  "
            f"(verdict bar: <= 0.30; AMBER v2: 0.809; S01: 0.821)"
        )
    if pcfs:
        med_pcf = sorted(pcfs)[len(pcfs) // 2]
        lines.append(
            f"# median policy-close fraction: {med_pcf:.3f}  "
            f"(AMBER v2: 0.000; S01: 0.255)"
        )
    if scfs:
        med_scf = sorted(scfs)[len(scfs) // 2]
        lines.append(
            f"# median stop-close fraction:   {med_scf:.3f}  "
            f"(AMBER v2: 0.000; S01: 0.000; useful range: 0.10-0.30)"
        )
    lines.append(
        f"# positive day_pnl: {pos_pnl}/{n}  "
        f"(verdict bar: >= 4/12; AMBER v2: 2/12; S01: 3/12)"
    )

    sources = {s.source for s in snaps}
    if sources != {"db"}:
        lines.append(
            f"# sources: {sorted(sources)}  (post-cohort-visibility S01b "
            f"runs are 'db'-only)"
        )

    return "\n".join(lines)


def render_json(snaps: list[AgentSnapshot]) -> str:
    """Machine-readable JSON output."""
    pos = sum(1 for s in snaps if s.day_pnl > 0)
    fc_rates = [s.fc_rate for s in snaps if s.fc_rate is not None]
    pcfs = [
        s.policy_close_fraction for s in snaps
        if s.policy_close_fraction is not None
    ]
    scfs = [
        s.stop_close_fraction for s in snaps
        if s.stop_close_fraction is not None
    ]
    payload = {
        "n_agents_complete": len(snaps),
        "positive_day_pnl_count": pos,
        "mean_fc_rate": (
            sum(fc_rates) / len(fc_rates) if fc_rates else None
        ),
        "median_policy_close_fraction": (
            sorted(pcfs)[len(pcfs) // 2] if pcfs else None
        ),
        "median_stop_close_fraction": (
            sorted(scfs)[len(scfs) // 2] if scfs else None
        ),
        "agents": [
            {
                "agent_id": s.agent_id,
                "eval_day": s.eval_day,
                "day_pnl": s.day_pnl,
                "bet_count": s.bet_count,
                "arbs_completed": s.arbs_completed,
                "arbs_closed": s.arbs_closed,
                "arbs_stop_closed": s.arbs_stop_closed,
                "arbs_force_closed": s.arbs_force_closed,
                "arbs_naked": s.arbs_naked,
                "pairs_opened": s.pairs_opened,
                "locked_pnl": s.locked_pnl,
                "closed_pnl": s.closed_pnl,
                "stop_closed_pnl": s.stop_closed_pnl,
                "force_closed_pnl": s.force_closed_pnl,
                "naked_pnl": s.naked_pnl,
                "fc_rate": s.fc_rate,
                "policy_close_fraction": s.policy_close_fraction,
                "stop_close_fraction": s.stop_close_fraction,
                "source": s.source,
            }
            for s in snaps
        ],
    }
    return json.dumps(payload, indent=2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "run_dir",
        help="Cohort output directory (registry/v2_<plan>_<ts>/).",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of the text table.",
    )
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"run_dir {run_dir} does not exist", file=sys.stderr)
        return 2

    snaps = collect_snapshots(run_dir)
    if args.json:
        print(render_json(snaps))
    else:
        n = len(snaps)
        print(f"# Cohort {run_dir.name}")
        print(f"# {n} agent{'s' if n != 1 else ''} complete")
        print()
        print(render_table(snaps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
