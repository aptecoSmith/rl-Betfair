"""Post-run validator for arb-signal-cleanup-probe.

Reads logs/training/episodes.jsonl (or a path you supply) and evaluates
the 5 validation criteria from
``plans/arb-signal-cleanup/purpose.md`` "What success looks like"
(same 5 criteria as arb-curriculum). Adds cohort attribution so the
three-cohort ablation (A=all three mechanisms, B=entropy only,
C=warmup+force-close) produces a per-cohort pass/fail matrix and
force-close/BC diagnostic tables per cohort.

Usage (from repo root):
    python scripts/validate_arb_signal_cleanup.py
    python scripts/validate_arb_signal_cleanup.py --log path/to/episodes.jsonl
    python scripts/validate_arb_signal_cleanup.py --spot-check-n 20

Criteria
--------
1. >=80% of agents remain active (bets>0) through episode 15.
2. arbs_closed/(arbs_closed+arbs_naked) > 15% on >=3 agents by ep15.
3. policy_loss stays O(1)+ (>=0.1) through episode 15 on >=50% of agents.
4. >=3 agents reach total_reward > 0 at any point in the run.
5. raw_pnl_reward + shaped_bonus ~= total_reward every episode
   (spot-check on --spot-check-n random rows; tolerance 0.01).

Per-cohort pass/fail matrix is emitted for C1 and C4 (the two failures
attacked by this plan). C2 / C3 / C5 are population-wide only: C2 and
C3 are existence / fraction checks and C5 is an invariant over every
row regardless of cohort.

Logs without a ``cohort`` field (pre-plan rows, e.g. the
arb-curriculum-probe archive) are keyed as ``"ungrouped"`` and the
per-cohort tables degrade gracefully to ``N/A``.

Exit codes:
    0: all criteria pass
    1: one or more criteria fail
    2: no data (log missing / no non-smoke rows)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


INVARIANT_TOL = 0.01
TARGET_EPISODE = 15
MIN_ACTIVE_FRACTION = 0.80
MIN_CLOSE_RATIO = 0.15
MIN_POLICY_LOSS = 0.10
MIN_POSITIVE_AGENTS = 3
MIN_CLOSE_RATIO_AGENTS = 3
DEFAULT_SPOT_CHECK_N = 10

COHORTS_KNOWN = ("A", "B", "C")


def load_rows(log_path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return [r for r in rows if not r.get("smoke_test")]


def group_by_agent(rows: list[dict]) -> dict[str, list[dict]]:
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        mid = r.get("model_id") or "unknown"
        by_agent[mid].append(r)
    for mid in by_agent:
        by_agent[mid].sort(key=lambda r: r.get("episode", 0))
    return dict(by_agent)


def agent_cohort(episodes: list[dict]) -> str:
    """Return the cohort label for an agent (first non-empty cohort seen)."""
    for r in episodes:
        c = r.get("cohort")
        if c:
            return str(c)
    return "ungrouped"


def split_by_cohort(
    by_agent: dict[str, list[dict]]
) -> dict[str, dict[str, list[dict]]]:
    """Return ``cohort -> {model_id: episodes}`` partition."""
    out: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for mid, eps in by_agent.items():
        out[agent_cohort(eps)][mid] = eps
    return dict(out)


def episode_at(episodes: list[dict], n: int) -> dict | None:
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


# -- Criterion evaluators ----------------------------------------------------

def _eval_c1(by_agent: dict[str, list[dict]]) -> tuple[int, int, int]:
    """Return ``(active_count, measurable, missing_ep15)``."""
    active = 0
    missing = 0
    for eps in by_agent.values():
        ep15 = episode_at(eps, TARGET_EPISODE)
        if ep15 is None:
            missing += 1
            continue
        if ep15.get("bet_count", 0) > 0:
            active += 1
    measurable = len(by_agent) - missing
    return active, measurable, missing


def _eval_c2(by_agent: dict[str, list[dict]]) -> int:
    qualifying = 0
    for eps in by_agent.values():
        ep15 = episode_at(eps, TARGET_EPISODE)
        if ep15 is None:
            continue
        closed = ep15.get("arbs_closed", 0)
        naked = ep15.get("arbs_naked", 0)
        total_arbs = closed + naked
        if total_arbs > 0 and (closed / total_arbs) > MIN_CLOSE_RATIO:
            qualifying += 1
    return qualifying


def _eval_c3(by_agent: dict[str, list[dict]]) -> tuple[int, int]:
    stable = 0
    measurable = 0
    for eps in by_agent.values():
        ep15 = episode_at(eps, TARGET_EPISODE)
        if ep15 is None:
            continue
        measurable += 1
        if ep15.get("policy_loss", 0.0) >= MIN_POLICY_LOSS:
            stable += 1
    return stable, measurable


def _eval_c4(
    by_agent: dict[str, list[dict]]
) -> tuple[int, list[tuple[str, float]]]:
    positives = 0
    peaks: list[tuple[str, float]] = []
    for mid, eps in by_agent.items():
        peak = max(
            (e.get("total_reward", float("-inf")) for e in eps),
            default=float("-inf"),
        )
        if peak > 0.0:
            positives += 1
        peaks.append((mid, peak))
    peaks.sort(key=lambda t: t[1], reverse=True)
    return positives, peaks


def evaluate(
    rows: list[dict],
    spot_check_n: int = DEFAULT_SPOT_CHECK_N,
) -> tuple[dict[str, bool], dict[str, dict[str, bool]]]:
    """Return ``(population_results, per_cohort_results)``.

    ``per_cohort_results`` is keyed ``cohort -> {"c1_active": bool,
    "c4_positive_reward": bool}`` for each cohort observed (C1 and C4
    only, matching the per-cohort ablation target).
    """
    by_agent = group_by_agent(rows)
    n_agents = len(by_agent)
    print(f"\nAgents in log: {n_agents}")
    print(f"Total episodes: {len(rows)}")

    by_cohort = split_by_cohort(by_agent)
    cohort_counts = {c: len(by_cohort.get(c, {})) for c in COHORTS_KNOWN}
    ungrouped = len(by_cohort.get("ungrouped", {}))
    print(
        "Cohort split: "
        + ", ".join(f"{c}={cohort_counts[c]}" for c in COHORTS_KNOWN)
        + (f", ungrouped={ungrouped}" if ungrouped else "")
    )
    print()

    results: dict[str, bool] = {}
    per_cohort_c1_c4: dict[str, dict[str, bool]] = {}

    # -- C1 (population-wide + per-cohort) ---------------------------------
    print("Criterion 1: >=80% of agents active (bets>0) through episode 15")
    active, measurable, missing = _eval_c1(by_agent)
    fraction = active / measurable if measurable > 0 else 0.0
    detail = (
        f"{active}/{measurable} agents active at ep{TARGET_EPISODE} "
        f"({fraction:.0%})."
    )
    if missing:
        detail += f" {missing} agents didn't reach ep{TARGET_EPISODE}."
    results["c1_active"] = _check(
        f">={MIN_ACTIVE_FRACTION:.0%} active at ep{TARGET_EPISODE}",
        fraction >= MIN_ACTIVE_FRACTION,
        detail,
    )

    # -- C2 ----------------------------------------------------------------
    print("\nCriterion 2: arbs_closed/(closed+naked) > 15% on >=3 agents by ep15")
    qualifying = _eval_c2(by_agent)
    results["c2_close_ratio"] = _check(
        f">={MIN_CLOSE_RATIO_AGENTS} agents with close ratio > "
        f"{MIN_CLOSE_RATIO:.0%} at ep{TARGET_EPISODE}",
        qualifying >= MIN_CLOSE_RATIO_AGENTS,
        f"{qualifying} qualifying agent(s).",
    )

    # -- C3 ----------------------------------------------------------------
    print("\nCriterion 3: policy_loss stays >=0.1 through ep15 on >=50% of agents")
    stable, c3_measurable = _eval_c3(by_agent)
    stable_frac = stable / c3_measurable if c3_measurable > 0 else 0.0
    results["c3_policy_loss"] = _check(
        f">=50% of agents have policy_loss >={MIN_POLICY_LOSS} at ep"
        f"{TARGET_EPISODE}",
        stable_frac >= 0.50,
        f"{stable}/{c3_measurable} agents ({stable_frac:.0%}).",
    )

    # -- C4 (population-wide + per-cohort) ---------------------------------
    print("\nCriterion 4: >=3 agents reach total_reward > 0 at any point")
    positives, peaks = _eval_c4(by_agent)
    top3 = ", ".join(f"{mid[:8]}...={rw:+.2f}" for mid, rw in peaks[:3])
    results["c4_positive_reward"] = _check(
        f">={MIN_POSITIVE_AGENTS} agents reach total_reward > 0",
        positives >= MIN_POSITIVE_AGENTS,
        f"{positives} agent(s). Top-3 peak: {top3}",
    )

    # -- Per-cohort matrix for C1 and C4 ----------------------------------
    for cohort in list(COHORTS_KNOWN) + (
        ["ungrouped"] if ungrouped else []
    ):
        subset = by_cohort.get(cohort, {})
        if not subset:
            continue
        active, measurable, _ = _eval_c1(subset)
        c1_pass = (
            measurable > 0
            and (active / measurable) >= MIN_ACTIVE_FRACTION
        )
        positives, _ = _eval_c4(subset)
        c4_pass = positives >= MIN_POSITIVE_AGENTS
        per_cohort_c1_c4[cohort] = {
            "c1_active": c1_pass,
            "c4_positive_reward": c4_pass,
        }

    # -- C5 invariant (population-wide) ------------------------------------
    print(
        f"\nCriterion 5: raw+shaped ~= total (spot-check {spot_check_n} "
        f"random rows, tol={INVARIANT_TOL})"
    )
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
        f"(tol={INVARIANT_TOL})."
    )
    results["c5_invariant"] = _check(
        "raw+shaped ~= total (correctness gate -- non-negotiable)",
        len(violations) == 0,
        detail,
    )

    return results, per_cohort_c1_c4


# -- Diagnostic tables -------------------------------------------------------

def print_cohort_matrix(per_cohort: dict[str, dict[str, bool]]) -> None:
    print("\nPer-cohort pass/fail matrix (C1 and C4):")
    if not per_cohort:
        print("  (no cohort data -- all rows ungrouped)")
        return
    print(f"  {'Cohort':>10}  {'C1 (>=80% active)':>22}  "
          f"{'C4 (>=3 positive)':>22}")
    for cohort in sorted(per_cohort):
        row = per_cohort[cohort]
        c1 = "PASS" if row["c1_active"] else "FAIL"
        c4 = "PASS" if row["c4_positive_reward"] else "FAIL"
        print(f"  {cohort:>10}  {c1:>22}  {c4:>22}")


def print_force_close_diagnostic(
    by_cohort: dict[str, dict[str, list[dict]]],
) -> None:
    print("\nForce-close diagnostic (per-cohort means across all episodes):")
    print(
        f"  {'Cohort':>10}  {'agents':>6}  "
        f"{'force_closed/race':>18}  "
        f"{'naked/race':>11}  "
        f"{'force_pnl/race':>15}"
    )
    any_row = False
    for cohort in list(COHORTS_KNOWN) + ["ungrouped"]:
        subset = by_cohort.get(cohort, {})
        if not subset:
            continue
        any_row = True
        total_races = 0
        total_fc = 0
        total_naked = 0
        total_fc_pnl = 0.0
        for eps in subset.values():
            for r in eps:
                races = max(int(r.get("races_completed", 0) or 0), 1)
                total_races += races
                total_fc += int(r.get("arbs_force_closed", 0) or 0)
                total_naked += int(r.get("arbs_naked", 0) or 0)
                total_fc_pnl += float(
                    r.get("scalping_force_closed_pnl", 0.0) or 0.0
                )
        if total_races == 0:
            continue
        print(
            f"  {cohort:>10}  {len(subset):>6d}  "
            f"{total_fc/total_races:>18.3f}  "
            f"{total_naked/total_races:>11.3f}  "
            f"{total_fc_pnl/total_races:>+15.3f}"
        )
    if not any_row:
        print("  (no data)")


def print_bc_diagnostic(
    by_cohort: dict[str, dict[str, list[dict]]],
) -> None:
    print("\nBC diagnostics (per-cohort means over post-BC ep1 rows):")
    print(
        f"  {'Cohort':>10}  {'agents_bc':>9}  "
        f"{'sig_loss':>9}  {'arb_loss':>9}"
    )
    any_row = False
    for cohort in list(COHORTS_KNOWN) + ["ungrouped"]:
        subset = by_cohort.get(cohort, {})
        if not subset:
            continue
        sig_losses: list[float] = []
        arb_losses: list[float] = []
        for eps in subset.values():
            first = next(
                (r for r in eps if int(r.get("bc_pretrain_steps", 0) or 0) > 0),
                None,
            )
            if first is None:
                continue
            sig_losses.append(
                float(first.get("bc_final_signal_loss", 0.0) or 0.0)
            )
            arb_losses.append(
                float(first.get("bc_final_arb_spread_loss", 0.0) or 0.0)
            )
        if not sig_losses:
            continue
        any_row = True
        mean_sig = sum(sig_losses) / len(sig_losses)
        mean_arb = sum(arb_losses) / len(arb_losses)
        print(
            f"  {cohort:>10}  {len(sig_losses):>9d}  "
            f"{mean_sig:>9.4f}  {mean_arb:>9.4f}"
        )
    if not any_row:
        print("  (no rows with bc_pretrain_steps > 0)")


def print_per_agent_summary(by_agent: dict[str, list[dict]]) -> None:
    print("\nPer-agent summary:")
    print(
        f"  {'agent':>10}  {'arch':>20}  {'cohort':>7}  {'eps':>4}  "
        f"{'peak_rw':>8}  {'bets@15':>7}  {'cls/tot@15':>10}  "
        f"{'fc@15':>5}"
    )
    for mid, eps in sorted(by_agent.items()):
        peak = max(
            (e.get("total_reward", float("-inf")) for e in eps),
            default=float("-inf"),
        )
        ep15 = episode_at(eps, 15) or {}
        bets15 = ep15.get("bet_count", "?")
        closed15 = int(ep15.get("arbs_closed", 0) or 0)
        naked15 = int(ep15.get("arbs_naked", 0) or 0)
        fc15 = int(ep15.get("arbs_force_closed", 0) or 0)
        total15 = closed15 + naked15 + fc15
        ratio = f"{closed15}/{total15}" if total15 > 0 else "0/0"
        arch = (eps[0].get("architecture_name") or "?")[-16:]
        cohort = agent_cohort(eps)
        print(
            f"  {mid[:10]}  {arch:>20}  {cohort:>7}  {len(eps):>4d}  "
            f"{peak:>+8.2f}  {str(bets15):>7}  {ratio:>10}  {fc15:>5d}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Validate arb-signal-cleanup-probe run",
    )
    ap.add_argument(
        "--log",
        default="logs/training/episodes.jsonl",
        help="Path to episodes.jsonl (default: logs/training/episodes.jsonl)",
    )
    ap.add_argument(
        "--spot-check-n",
        type=int,
        default=DEFAULT_SPOT_CHECK_N,
        help=f"Number of random rows for C5 (default: {DEFAULT_SPOT_CHECK_N})",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for spot-check sampling (default: 42)",
    )
    args = ap.parse_args(argv)

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"ERROR: Log file not found: {log_path}", file=sys.stderr)
        return 2

    random.seed(args.seed)

    print("=" * 60)
    print("arb-signal-cleanup-probe validation")
    print("=" * 60)
    print(f"Log: {log_path}  ({log_path.stat().st_size:,} bytes)")

    rows = load_rows(log_path)
    if not rows:
        print("ERROR: No non-smoke-test rows found in log.", file=sys.stderr)
        return 2

    by_agent = group_by_agent(rows)
    by_cohort = split_by_cohort(by_agent)

    print_per_agent_summary(by_agent)

    print("\n" + "-" * 60)
    print("Criteria evaluation")
    print("-" * 60)
    results, per_cohort = evaluate(rows, spot_check_n=args.spot_check_n)

    print_cohort_matrix(per_cohort)
    print_force_close_diagnostic(by_cohort)
    print_bc_diagnostic(by_cohort)

    print("\n" + "=" * 60)
    passed = sum(results.values())
    total = len(results)
    print(f"SUMMARY: {passed}/{total} criteria passed")

    labels = {
        "c1_active":          "C1 active >=80%",
        "c2_close_ratio":     "C2 close_ratio >15% (>=3 agents)",
        "c3_policy_loss":     "C3 policy_loss O(1)+ (>=50% agents)",
        "c4_positive_reward": "C4 positive reward (>=3 agents)",
        "c5_invariant":       "C5 invariant raw+shaped~=total",
    }
    for key, label in labels.items():
        tag = "PASS" if results.get(key) else "FAIL"
        print(f"  [{tag}] {label}")

    if not results.get("c5_invariant"):
        print("\n[!] CORRECTNESS GATE FAILED (C5). Do NOT ship.")
        print("   Roll back, fix accounting, retest.")
    elif passed == total:
        print("\nOK All criteria pass. Next: scale run.")
    elif passed >= 4:
        print("\n~~ >=4 criteria pass. Partial success -- check cohort matrix.")
    else:
        print("\nFAIL <4 criteria pass. See decision tree below.")

    print("\n" + "-" * 60)
    print("Decision tree (purpose.md 'What happens next'):")
    print("-" * 60)
    print("  5/5 pass       -> scale-run plan")
    print("  C1 pass, C4 fail -> observation-space-audit")
    print("  C4 pass, C1 fail -> controller-arch plan (PI / Adam)")
    print("  1-4 all fail    -> observation-space-audit")
    print("  C5 fail         -> rollback; do NOT ship")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
