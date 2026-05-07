"""Empirical per-pair P&L summary across cohort scoreboards.

Phase-13 follow-up. The threshold-sweep tool used eyeball P&L numbers
(£2.50 locked / £3.00 loss). To turn the gate strategy's go/no-go
into a real decision we need empirical numbers from cohort runs.

Reads scoreboard.jsonl files and reports per-pair P&L for each
lifecycle category:

* matured  = (completed + closed) — pair is fully resolved into a
  locked spread (or agent-closed equivalent). Mean P&L per pair.
* force_closed — env force-closed at T-N seconds. Loss per pair.
* stop_closed — env stop-closed mid-race on MTM (zero on cohorts
  without that lever).
* naked — settled with one leg open. Loss per pair (variance high).

Aggregates:

* across-cohort mean per-pair (£) for each category
* implied break-even gate-mature-rate using realistic P_locked /
  P_loss

Usage::

    python -m tools.cohort_per_pair_pnl_summary \
        registry/_phase13_10h_arm_A_off_1778132072 \
        registry/_phase13_10h_arm_B_on_1778132077
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def _load(scoreboard_path: Path) -> list[dict]:
    return [json.loads(l) for l in scoreboard_path.read_text(
        encoding="utf-8",
    ).splitlines()]


def per_pair_pnl(rows):
    """Compute per-row per-pair P&L per category.

    Returns a list of dicts ready for aggregation.
    """
    out = []
    for r in rows:
        completed = int(r.get("eval_arbs_completed", 0) or 0)
        closed = int(r.get("eval_arbs_closed", 0) or 0)
        force_closed = int(r.get("eval_arbs_force_closed", 0) or 0)
        stop_closed = int(r.get("eval_arbs_stop_closed", 0) or 0)
        naked = int(r.get("eval_arbs_naked", 0) or 0)
        pairs_opened = int(r.get("eval_pairs_opened", 0) or 0)

        locked_pnl = float(r.get("eval_locked_pnl", 0.0) or 0.0)
        closed_pnl = float(r.get("eval_closed_pnl", 0.0) or 0.0)
        force_closed_pnl = float(r.get("eval_force_closed_pnl", 0.0) or 0.0)
        stop_closed_pnl = float(r.get("eval_stop_closed_pnl", 0.0) or 0.0)
        naked_pnl = float(r.get("eval_naked_pnl", 0.0) or 0.0)

        out.append({
            "agent": str(r.get("agent_id", ""))[:8],
            "gen": int(r.get("generation", 0)),
            "pairs_opened": pairs_opened,
            "n_completed": completed,
            "n_closed": closed,
            "n_matured": completed + closed,
            "n_force_closed": force_closed,
            "n_stop_closed": stop_closed,
            "n_naked": naked,
            "locked_pnl": locked_pnl,
            "closed_pnl": closed_pnl,
            "force_closed_pnl": force_closed_pnl,
            "stop_closed_pnl": stop_closed_pnl,
            "naked_pnl": naked_pnl,
            # Per-pair averages — guarded against /0.
            "locked_per_completed": (
                locked_pnl / completed if completed > 0 else None
            ),
            "closed_per_closed": (
                closed_pnl / closed if closed > 0 else None
            ),
            "matured_pnl_total": locked_pnl + closed_pnl,
            "matured_per_pair": (
                (locked_pnl + closed_pnl) / (completed + closed)
                if (completed + closed) > 0 else None
            ),
            "force_per_pair": (
                force_closed_pnl / force_closed
                if force_closed > 0 else None
            ),
            "stop_per_pair": (
                stop_closed_pnl / stop_closed
                if stop_closed > 0 else None
            ),
            "naked_per_pair": (
                naked_pnl / naked if naked > 0 else None
            ),
        })
    return out


def _agg_mean(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, 0
    return statistics.mean(vals), len(vals)


def _agg_median(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, 0
    return statistics.median(vals), len(vals)


def main():
    if len(sys.argv) < 2:
        print("usage: cohort_per_pair_pnl_summary.py "
              "<registry_dir> [<registry_dir> ...]")
        sys.exit(1)

    all_rows = []
    by_dir = {}
    for arg in sys.argv[1:]:
        p = Path(arg) / "scoreboard.jsonl"
        if not p.exists():
            print(f"WARN: {p} missing")
            continue
        rows = _load(p)
        per = per_pair_pnl(rows)
        by_dir[arg] = per
        all_rows.extend(per)

    if not all_rows:
        print("no data")
        return

    print(f"\n=== per-cohort summary (mean over agents) ===")
    print(f"{'cohort':<55} {'n_rows':>6} "
          f"{'pairs':>6} {'matured':>7} {'force':>6} {'naked':>6} "
          f"{'£/matured':>10} {'£/force':>10} {'£/naked':>10}")
    print('-' * 130)
    for d, rows in by_dir.items():
        n = len(rows)
        avg_pairs = statistics.mean([r["pairs_opened"] for r in rows])
        avg_matured = statistics.mean([r["n_matured"] for r in rows])
        avg_force = statistics.mean([r["n_force_closed"] for r in rows])
        avg_naked = statistics.mean([r["n_naked"] for r in rows])
        m_mat, n_mat = _agg_mean([r["matured_per_pair"] for r in rows])
        m_force, n_force = _agg_mean([r["force_per_pair"] for r in rows])
        m_naked, n_naked = _agg_mean([r["naked_per_pair"] for r in rows])
        m_mat_s = f"£{m_mat:+.3f}" if m_mat is not None else "—"
        m_force_s = f"£{m_force:+.3f}" if m_force is not None else "—"
        m_naked_s = f"£{m_naked:+.3f}" if m_naked is not None else "—"
        short_d = d.replace("registry/", "").replace("registry\\", "")[:54]
        print(f"{short_d:<55} {n:>6} {avg_pairs:>6.0f} {avg_matured:>7.1f} "
              f"{avg_force:>6.1f} {avg_naked:>6.1f} "
              f"{m_mat_s:>10} {m_force_s:>10} {m_naked_s:>10}")

    # Pooled across all cohorts.
    mat_pp = [r["matured_per_pair"] for r in all_rows if r["matured_per_pair"] is not None]
    force_pp = [r["force_per_pair"] for r in all_rows if r["force_per_pair"] is not None]
    naked_pp = [r["naked_per_pair"] for r in all_rows if r["naked_per_pair"] is not None]
    locked_pp = [r["locked_per_completed"] for r in all_rows if r["locked_per_completed"] is not None]
    closed_pp = [r["closed_per_closed"] for r in all_rows if r["closed_per_closed"] is not None]

    def _stats(vals, label):
        if not vals:
            print(f"  {label}: no data")
            return None
        m = statistics.mean(vals)
        med = statistics.median(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"  {label}: mean=£{m:+.3f}  median=£{med:+.3f}  "
              f"sd=£{sd:.3f}  n={len(vals)}")
        return m

    print(f"\n=== pooled per-pair P&L (across all cohorts, n={len(all_rows)}) ===")
    pp_mat = _stats(mat_pp, "matured (completed+closed)")
    pp_locked_only = _stats(locked_pp, "  └─ completed (natural fill)")
    pp_closed_only = _stats(closed_pp, "  └─ closed (close_signal)")
    pp_force = _stats(force_pp, "force_closed")
    pp_naked = _stats(naked_pp, "naked")

    # Lifecycle counts pooled.
    print(f"\n=== pooled lifecycle counts ===")
    avg_pairs = statistics.mean([r["pairs_opened"] for r in all_rows])
    avg_matured = statistics.mean([r["n_matured"] for r in all_rows])
    avg_force = statistics.mean([r["n_force_closed"] for r in all_rows])
    avg_naked = statistics.mean([r["n_naked"] for r in all_rows])
    print(f"  per agent eval rollout (mean):")
    print(f"    pairs_opened   = {avg_pairs:.1f}")
    print(f"    matured        = {avg_matured:.1f}  "
          f"({avg_matured/avg_pairs*100:.1f}%)")
    print(f"    force_closed   = {avg_force:.1f}  "
          f"({avg_force/avg_pairs*100:.1f}%)")
    print(f"    naked          = {avg_naked:.1f}  "
          f"({avg_naked/avg_pairs*100:.1f}%)")

    # Break-even analysis. The agent's per-open expected P&L is:
    #   E[pnl/open] = mature_rate * pp_mat
    #               + force_rate * pp_force
    #               + naked_rate * pp_naked
    # Under a gate that filters out low-confidence opens, naked
    # disappears (rare in fc=60 mode anyway) and force_rate moves
    # toward 1 - mature_rate. Approximating force_rate = 1 - mature_rate:
    if pp_mat is not None and pp_force is not None:
        # Loss per non-matured = pp_force (it's negative). Use the
        # weighted naked+force average if naked is non-trivial.
        # In fc=60 mode naked is rare so pp_force dominates the
        # "didn't mature" cost.
        loss_per_pair = -pp_force  # express as a positive loss number
        gain_per_pair = pp_mat
        if loss_per_pair > 0 and gain_per_pair > 0:
            be_mature = loss_per_pair / (gain_per_pair + loss_per_pair)
            print(f"\n=== break-even analysis ===")
            print(f"  P_locked (per matured pair)  = £{gain_per_pair:+.3f}")
            print(f"  P_loss   (per force-closed)  = £{loss_per_pair:+.3f}")
            print(f"  → break-even mature_rate     = {be_mature:.4f}")
        elif gain_per_pair < 0:
            print(f"\nWARN: pooled matured P&L is NEGATIVE (£{gain_per_pair:.3f})."
                  " Even matured pairs lose money on average — strategy is")
            print("structurally unprofitable at current commission/spread.")
        else:
            print(f"\nWARN: force-closed P&L is non-negative; can't compute "
                  "break-even.")


if __name__ == "__main__":
    main()
