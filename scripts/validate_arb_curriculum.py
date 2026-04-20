"""Post-run validator for arb-curriculum-probe.

Reads logs/training/episodes.jsonl (or a path you supply) and evaluates the
5 validation criteria from plans/arb-curriculum/purpose.md §What success
looks like.

Usage (from repo root):
    python scripts/validate_arb_curriculum.py
    python scripts/validate_arb_curriculum.py --log logs/training/episodes.jsonl
    python scripts/validate_arb_curriculum.py --spot-check-n 20  # more invariant rows

Criteria evaluated
------------------
1. ≥80% of agents remain active (bets>0) through episode 15.
2. arbs_closed/arbs_naked > 15% on ≥3 agents by episode 15.
3. policy_loss stays O(1)+ (≥0.1) through episode 15 on ≥50% of agents.
4. ≥3 agents reach total_reward > 0 at any point in the run.
5. raw_pnl_reward + shaped_bonus ≈ total_reward every episode
   (spot-check on --spot-check-n random rows; tolerance 0.01).

Exit codes:
    0: all criteria pass
    1: one or more criteria fail (or data missing)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path


INVARIANT_TOL = 0.01      # max acceptable |raw+shaped-total| per episode
TARGET_EPISODE = 15       # "ep15" from the success criteria (1-indexed)
MIN_ACTIVE_FRACTION = 0.80
MIN_CLOSE_RATIO = 0.15
MIN_POLICY_LOSS = 0.10    # O(1)+ => at least 0.1
MIN_POSITIVE_AGENTS = 3
MIN_CLOSE_RATIO_AGENTS = 3
DEFAULT_SPOT_CHECK_N = 10


def load_rows(log_path: Path) -> list[dict]:
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    # Exclude smoke-test rows (those are a 3-agent probe, not the full run)
    return [r for r in rows if not r.get("smoke_test")]


def group_by_agent(rows: list[dict]) -> dict[str, list[dict]]:
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        mid = r.get("model_id") or "unknown"
        by_agent[mid].append(r)
    for mid in by_agent:
        by_agent[mid].sort(key=lambda r: r.get("episode", 0))
    return dict(by_agent)


def episode_at(episodes: list[dict], n: int) -> dict | None:
    """Return the row at episode number n (1-indexed). None if missing."""
    for r in episodes:
        if r.get("episode") == n:
            return r
    return None


def _check(label: str, passed: bool, detail: str = "") -> bool:
    tag = "PASS" if passed else "FAIL"
    msg = f"  [{tag}] {label}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    return passed


def evaluate(rows: list[dict], spot_check_n: int = DEFAULT_SPOT_CHECK_N) -> dict[str, bool]:
    by_agent = group_by_agent(rows)
    n_agents = len(by_agent)
    print(f"\nAgents in log: {n_agents}")
    print(f"Total episodes: {len(rows)}")
    print()

    results: dict[str, bool] = {}

    # ── Criterion 1: ≥80% active at ep15 ──────────────────────────────────
    print("Criterion 1: ≥80% of agents active (bets>0) through episode 15")
    active_count = 0
    missing_ep15 = 0
    for mid, eps in by_agent.items():
        ep15 = episode_at(eps, TARGET_EPISODE)
        if ep15 is None:
            missing_ep15 += 1
            continue
        if ep15.get("bet_count", 0) > 0:
            active_count += 1
    measurable = n_agents - missing_ep15
    fraction = active_count / measurable if measurable > 0 else 0.0
    detail = (
        f"{active_count}/{measurable} agents active at ep{TARGET_EPISODE} "
        f"({fraction:.0%}). "
        + (f"{missing_ep15} agents didn't reach ep{TARGET_EPISODE}." if missing_ep15 else "")
    )
    results["c1_active"] = _check(
        f"≥{MIN_ACTIVE_FRACTION:.0%} active at ep{TARGET_EPISODE}",
        fraction >= MIN_ACTIVE_FRACTION,
        detail,
    )

    # ── Criterion 2: arbs_closed/arbs_naked > 15% on ≥3 agents at ep15 ────
    print("\nCriterion 2: arbs_closed/arbs_naked > 15% on ≥3 agents by ep15")
    qualifying_agents = 0
    for mid, eps in by_agent.items():
        ep15 = episode_at(eps, TARGET_EPISODE)
        if ep15 is None:
            continue
        closed = ep15.get("arbs_closed", 0)
        naked = ep15.get("arbs_naked", 0)
        total_arbs = closed + naked
        if total_arbs > 0 and (closed / total_arbs) > MIN_CLOSE_RATIO:
            qualifying_agents += 1
    results["c2_close_ratio"] = _check(
        f"≥{MIN_CLOSE_RATIO_AGENTS} agents with arbs_closed/arbs_naked > "
        f"{MIN_CLOSE_RATIO:.0%} at ep{TARGET_EPISODE}",
        qualifying_agents >= MIN_CLOSE_RATIO_AGENTS,
        f"{qualifying_agents} qualifying agent(s). "
        f"(Baseline: 5–7% population-wide; target: triple to >15%)",
    )

    # ── Criterion 3: policy_loss ≥0.1 through ep15 on ≥50% of agents ──────
    print("\nCriterion 3: policy_loss stays ≥0.1 through ep15 on ≥50% of agents")
    stable_count = 0
    for mid, eps in by_agent.items():
        ep15 = episode_at(eps, TARGET_EPISODE)
        if ep15 is None:
            continue
        if ep15.get("policy_loss", 0.0) >= MIN_POLICY_LOSS:
            stable_count += 1
    stable_frac = stable_count / measurable if measurable > 0 else 0.0
    results["c3_policy_loss"] = _check(
        f"≥50% of agents have policy_loss ≥{MIN_POLICY_LOSS} at ep{TARGET_EPISODE}",
        stable_frac >= 0.50,
        f"{stable_count}/{measurable} agents ({stable_frac:.0%}) still have "
        f"policy_loss ≥{MIN_POLICY_LOSS}.",
    )

    # ── Criterion 4: ≥3 agents reach total_reward > 0 ─────────────────────
    print("\nCriterion 4: ≥3 agents reach total_reward > 0 at any point")
    positive_agents = 0
    best_rewards: list[tuple[str, float]] = []
    for mid, eps in by_agent.items():
        peak = max((e.get("total_reward", float("-inf")) for e in eps), default=float("-inf"))
        if peak > 0.0:
            positive_agents += 1
        best_rewards.append((mid, peak))
    best_rewards.sort(key=lambda t: t[1], reverse=True)
    top3 = ", ".join(
        f"{mid[:8]}…={rw:+.2f}" for mid, rw in best_rewards[:3]
    )
    results["c4_positive_reward"] = _check(
        f"≥{MIN_POSITIVE_AGENTS} agents reach total_reward > 0",
        positive_agents >= MIN_POSITIVE_AGENTS,
        f"{positive_agents} agent(s) ever reached positive reward. "
        f"Top-3 peak: {top3}",
    )

    # ── Criterion 5: invariant raw+shaped≈total ────────────────────────────
    print(f"\nCriterion 5: raw+shaped ≈ total (spot-check {spot_check_n} random rows, "
          f"tol={INVARIANT_TOL})")
    sample = random.sample(rows, min(spot_check_n, len(rows))) if rows else []
    violations: list[tuple[int, float]] = []
    for r in sample:
        raw = r.get("raw_pnl_reward", 0.0)
        shaped = r.get("shaped_bonus", 0.0)
        total = r.get("total_reward", 0.0)
        diff = abs(raw + shaped - total)
        if diff > INVARIANT_TOL:
            ep = r.get("episode", "?")
            violations.append((ep, diff))
    if violations:
        print(f"         VIOLATIONS ({len(violations)}/{len(sample)}):")
        for ep, diff in violations[:5]:
            print(f"           ep={ep} |raw+shaped-total|={diff:.6f}")
    detail = (
        f"Checked {len(sample)} rows; {len(violations)} violations "
        f"(tol={INVARIANT_TOL}). "
        + ("✓ All pass." if not violations else
           "✗ Fix accounting in Sessions 02/03 before proceeding.")
    )
    results["c5_invariant"] = _check(
        "raw+shaped ≈ total (correctness gate — non-negotiable)",
        len(violations) == 0,
        detail,
    )

    return results


def print_bc_diagnostics(by_agent: dict[str, list[dict]]) -> None:
    """Print BC pretrain results from the first post-BC episode."""
    print("\nBC pretrain diagnostics (ep1 rows)")
    print(f"  {'agent':>10}  {'arch':>20}  {'bc_steps':>8}  "
          f"{'sig_loss':>9}  {'arb_loss':>9}  {'post_bc_ent':>11}")
    any_bc = False
    for mid, eps in sorted(by_agent.items()):
        ep1 = episode_at(eps, 1)
        if ep1 is None:
            ep1 = eps[0] if eps else None
        if ep1 and ep1.get("bc_pretrain_steps", 0) > 0:
            any_bc = True
            arch = ep1.get("architecture_name", "?")[-16:]
            print(
                f"  {mid[:10]}  {arch:>20}  "
                f"{ep1['bc_pretrain_steps']:>8d}  "
                f"{ep1.get('bc_final_signal_loss', float('nan')):>9.4f}  "
                f"{ep1.get('bc_final_arb_spread_loss', float('nan')):>9.4f}  "
                f"{ep1.get('target_entropy', float('nan')):>11.2f}"
            )
    if not any_bc:
        print("  (no BC pretrain rows found — all agents may have bc_pretrain_steps=0)")


def print_per_agent_summary(by_agent: dict[str, list[dict]]) -> None:
    """Print a compact per-agent trajectory table."""
    print("\nPer-agent summary (last 5 episodes shown)")
    print(f"  {'agent':>10}  {'arch':>20}  {'eps':>4}  "
          f"{'peak_rw':>8}  {'bets@ep15':>9}  {'cls/nkd@15':>10}  "
          f"{'curriculum':>12}")
    for mid, eps in sorted(by_agent.items()):
        peak = max((e.get("total_reward", float("-inf")) for e in eps), default=float("-inf"))
        ep15 = episode_at(eps, 15) or {}
        bets15 = ep15.get("bet_count", "?")
        closed15 = ep15.get("arbs_closed", 0)
        naked15 = ep15.get("arbs_naked", 0)
        ratio = f"{closed15}/{closed15+naked15}" if (closed15 + naked15) > 0 else "0/0"
        curriculum = (ep15.get("curriculum_day_order") or eps[0].get("curriculum_day_order", "?"))
        arch = (eps[0].get("architecture_name") or "?")[-16:]
        print(
            f"  {mid[:10]}  {arch:>20}  {len(eps):>4d}  "
            f"{peak:>+8.2f}  {str(bets15):>9}  {ratio:>10}  "
            f"{curriculum:>12}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate arb-curriculum-probe run")
    ap.add_argument(
        "--log",
        default="logs/training/episodes.jsonl",
        help="Path to episodes.jsonl (default: logs/training/episodes.jsonl)",
    )
    ap.add_argument(
        "--spot-check-n",
        type=int,
        default=DEFAULT_SPOT_CHECK_N,
        help=f"Number of random rows for invariant check (default: {DEFAULT_SPOT_CHECK_N})",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for spot-check sampling (default: 42)",
    )
    args = ap.parse_args(argv)

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"ERROR: Log file not found: {log_path}", file=sys.stderr)
        return 2

    random.seed(args.seed)

    print("=" * 60)
    print("arb-curriculum-probe validation")
    print("=" * 60)
    print(f"Log: {log_path}  ({log_path.stat().st_size:,} bytes)")

    rows = load_rows(log_path)
    if not rows:
        print("ERROR: No non-smoke-test rows found in log.", file=sys.stderr)
        return 2

    by_agent = group_by_agent(rows)
    print_bc_diagnostics(by_agent)
    print_per_agent_summary(by_agent)

    print("\n" + "─" * 60)
    print("Criteria evaluation")
    print("─" * 60)
    results = evaluate(rows, spot_check_n=args.spot_check_n)

    # Summary
    print("\n" + "=" * 60)
    passed = sum(results.values())
    total = len(results)
    print(f"SUMMARY: {passed}/{total} criteria passed")

    criterion_labels = {
        "c1_active":        "C1 active ≥80%",
        "c2_close_ratio":   "C2 close_ratio >15% (≥3 agents)",
        "c3_policy_loss":   "C3 policy_loss O(1)+ (≥50% agents)",
        "c4_positive_reward": "C4 positive reward (≥3 agents)",
        "c5_invariant":     "C5 invariant raw+shaped≈total",
    }
    for key, label in criterion_labels.items():
        tag = "PASS" if results.get(key) else "FAIL"
        print(f"  [{tag}] {label}")

    if not results.get("c5_invariant"):
        print("\n⚠ CORRECTNESS GATE FAILED (C5). Do NOT proceed.")
        print("  Roll back Sessions 02/03, fix accounting, retest, redo Session 06.")
    elif passed == total:
        print("\n✓ All criteria pass. Next: open arb-curriculum-scale plan.")
    elif passed >= 4:
        print("\n~ ≥4 criteria pass. Partial success — consider arb-curriculum-tune.")
    else:
        print("\n✗ <4 criteria pass. See master_todo.md §After Session 07 for decision tree.")

    # Decision guide
    print("\n" + "─" * 60)
    print("Decision guide (master_todo.md §After Session 07)")
    print("─" * 60)
    print("  ≥4 pass:    open arb-curriculum-scale (16-agent x 10-gen scale run)")
    print("  C5 fails:   STOP. Fix invariant. Do not ship.")
    print("  1-4 fail:   open observation-space-audit")
    print("  partial:    open arb-curriculum-tune (tighten gene ranges)")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
