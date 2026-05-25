"""Analysis for the non-scalping-directional-probe Phase 6 verdict.

Reads per-bet JSONL logs from a probe output dir and computes the
pre-registered metrics (per-bet EV, Sharpe, days-profitable, bet
count, calibration) and applies the PASS/FAIL criteria from
``plans/non-scalping-directional-probe/README.md::Success bar``.

Usage:

    python -m tools.analyse_directional_probe \\
        --probe-dir registry/probe_A_back \\
        --label "Probe A (back)" --side back
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-dir", required=True, type=Path)
    p.add_argument("--label", required=True, help="Display label, e.g. 'Probe A (back)'.")
    p.add_argument(
        "--side", choices=["back", "lay", "both"], default="both",
        help="Filter logged bets to this side (smoke probe was 'both').",
    )
    return p.parse_args(argv)


def _load_bets(probe_dir: Path, side_filter: str) -> list[dict]:
    bets = []
    for path in sorted(probe_dir.glob("bets_*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if side_filter == "both" or rec["side"] == side_filter:
                    bets.append(rec)
    return bets


def _per_bet_stats(bets: list[dict]) -> dict:
    pnls = [b["final_pnl"] for b in bets]
    n = len(pnls)
    if n == 0:
        return {
            "n": 0, "mean": 0.0, "std": 0.0, "sharpe": 0.0,
            "n_wins": 0, "winrate": 0.0,
        }
    mean = statistics.mean(pnls)
    std = statistics.stdev(pnls) if n > 1 else 0.0
    sharpe = (mean / std) if std > 0 else 0.0
    n_wins = sum(1 for b in bets if b["final_outcome"] == "win")
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "sum": sum(pnls),
        "n_wins": n_wins,
        "winrate": n_wins / n,
    }


def _per_day_pnl(bets: list[dict]) -> dict[str, float]:
    by_day = defaultdict(float)
    by_day_n = defaultdict(int)
    for b in bets:
        by_day[b["day"]] += b["final_pnl"]
        by_day_n[b["day"]] += 1
    return {
        "by_day_pnl": dict(by_day),
        "by_day_count": dict(by_day_n),
    }


def _calibration_table(bets: list[dict]) -> list[dict]:
    """Bin by champion_p_win decile, compute predicted-vs-realised win rate.

    For BACK bets: predicted win rate = pwin; realised win rate =
    n_wins / n.
    For LAY bets:  predicted win rate = 1 - pwin; realised win rate
    = n_wins / n (note: a lay "wins" when the runner loses).
    """
    if not bets:
        return []
    deciles = []
    for b in bets:
        pwin = b["runner_champion_p_win"]
        if b["side"] == "back":
            predicted = pwin
        else:
            predicted = 1.0 - pwin
        deciles.append((predicted, b["final_outcome"] == "win"))
    # Sort by predicted, split into 10 equal-size bins.
    deciles.sort()
    n = len(deciles)
    bins = []
    bin_size = max(1, n // 10)
    for i in range(0, n, bin_size):
        chunk = deciles[i:i + bin_size]
        if not chunk:
            continue
        predicted_mean = sum(p for p, _ in chunk) / len(chunk)
        realised_mean = sum(1 for _, w in chunk if w) / len(chunk)
        bins.append({
            "decile_lo": chunk[0][0],
            "decile_hi": chunk[-1][0],
            "predicted_mean": predicted_mean,
            "realised_mean": realised_mean,
            "n": len(chunk),
            "delta": realised_mean - predicted_mean,
        })
    return bins


def _verdict(stats: dict, day_pnl: dict, calib: list[dict]) -> dict:
    """Apply pre-registered PASS/FAIL criteria from README.md."""
    n_days = len(day_pnl["by_day_pnl"])
    profitable_days = sum(1 for p in day_pnl["by_day_pnl"].values() if p > 0)
    avg_bets_per_day = (
        stats["n"] / n_days if n_days > 0 else 0
    )

    ev_pass = stats["mean"] > 0.50
    ev_fail = stats["mean"] < 0.10
    sharpe_pass = stats["sharpe"] > 0.10
    sharpe_fail = stats["sharpe"] < 0.05
    days_pass = profitable_days >= 2
    days_fail = profitable_days == 0
    count_pass = 20 <= avg_bets_per_day <= 300
    count_fail = avg_bets_per_day < 10 or avg_bets_per_day > 600

    # Calibration circuit-breaker: top admitted decile mismatch > 10pp.
    calib_break = False
    if calib:
        top_decile = calib[-1]
        if abs(top_decile["delta"]) > 0.10 and top_decile["n"] >= 20:
            calib_break = True

    band_pass = (
        ev_pass and sharpe_pass and days_pass and count_pass
        and not calib_break
    )
    band_fail = (
        ev_fail or sharpe_fail or days_fail or count_fail or calib_break
    )

    if band_pass:
        verdict = "PASS"
    elif band_fail:
        verdict = "FAIL"
    else:
        verdict = "BORDERLINE"

    return {
        "verdict": verdict,
        "checks": {
            "per_bet_ev": {
                "pass": ev_pass, "fail": ev_fail,
                "actual": stats["mean"],
                "criterion": "PASS: >+0.50, FAIL: <+0.10",
            },
            "sharpe": {
                "pass": sharpe_pass, "fail": sharpe_fail,
                "actual": stats["sharpe"],
                "criterion": "PASS: >0.10, FAIL: <0.05",
            },
            "days_profitable": {
                "pass": days_pass, "fail": days_fail,
                "actual": f"{profitable_days}/{n_days}",
                "criterion": "PASS: >=2/3, FAIL: 0/3",
            },
            "bet_count_per_day": {
                "pass": count_pass, "fail": count_fail,
                "actual": avg_bets_per_day,
                "criterion": "PASS: 20-300/day, FAIL: <10 or >600",
            },
            "calibration_circuit_breaker": {
                "tripped": calib_break,
                "criterion": "Top admitted decile delta > 10pp invalidates EV",
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    bets = _load_bets(args.probe_dir, args.side)
    if not bets:
        print(f"{args.label}: NO BETS FOUND in {args.probe_dir}")
        return 1

    stats = _per_bet_stats(bets)
    day_pnl = _per_day_pnl(bets)
    calib = _calibration_table(bets)
    verdict = _verdict(stats, day_pnl, calib)

    print(f"\n{'=' * 72}")
    print(f"{args.label}")
    print(f"{'=' * 72}\n")
    print(f"Bets: n={stats['n']}, wins={stats['n_wins']} ({stats['winrate']*100:.1f}%)")
    print(f"Per-bet P&L: mean={stats['mean']:+.4f} std={stats['std']:.4f}")
    print(f"Per-bet Sharpe: {stats['sharpe']:+.4f}")
    print(f"Cumulative P&L: £{stats['sum']:+.2f}")
    print()
    print(f"By day:")
    for day in sorted(day_pnl["by_day_pnl"]):
        pnl = day_pnl["by_day_pnl"][day]
        n = day_pnl["by_day_count"][day]
        print(f"  {day}: £{pnl:+.2f} (n={n})")
    print()
    print(f"Calibration table (predicted vs realised win rate by decile):")
    print(f"  {'decile':>8} {'n':>6} {'predicted':>10} {'realised':>10} {'delta':>10}")
    for b in calib:
        print(
            f"  {b['decile_lo']:.3f}-{b['decile_hi']:.3f} {b['n']:>6} "
            f"{b['predicted_mean']:>10.4f} {b['realised_mean']:>10.4f} "
            f"{b['delta']:>+10.4f}"
        )
    print()
    print(f"VERDICT vs pre-registered success bar:")
    for name, check in verdict["checks"].items():
        if "tripped" in check:
            status = "TRIPPED" if check["tripped"] else "ok"
        else:
            status = "PASS" if check["pass"] else ("FAIL" if check["fail"] else "borderline")
        print(f"  {name:30} {status:>10}  actual={check.get('actual', 'n/a')}")
    print()
    print(f"OVERALL: {verdict['verdict']}")
    print()

    # Persist analysis JSON next to the input dir.
    out_path = args.probe_dir / "_analysis.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({
            "label": args.label,
            "side_filter": args.side,
            "stats": stats,
            "day_pnl": day_pnl,
            "calibration": calib,
            "verdict": verdict,
        }, fh, indent=2)
    print(f"Analysis JSON: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
