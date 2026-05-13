"""Probe per-race max(champion p_win) distribution across multiple days.

Used to pick a sensible `race_confidence_threshold` for
`plans/scalping-race-confidence-gate/`. The smoke on 2026-05-04
showed threshold=0.30 admits 100% of races (gate inert). This probe
samples several days and prints the per-race max-p_win distribution
so the operator (or autonomous loop) can lock a threshold that
splits ~20-50% of races as non-confident.

Usage:

    python -m tools.probe_race_confidence_distribution \
        --days 2026-05-04 2026-05-05 2026-05-06 \
        --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("probe_race_confidence")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", nargs="+", required=True, metavar="YYYY-MM-DD")
    p.add_argument("--data-dir", default="data/processed", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--predictor-bundle-manifests", nargs=3, default=None,
        metavar=("CHAMPION", "RANKER", "DIRECTION"),
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _default_manifests() -> tuple[str, str, str]:
    root = Path(__file__).resolve().parents[1]
    sibling = root.parent / "betfair-predictors"
    return (
        str(sibling / "production" / "race-outcome" / "manifest.json"),
        str(sibling / "production" / "race-outcome-ranker" / "manifest.json"),
        str(sibling / "production" / "direction-predictor" / "manifest.json"),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(message)s",
    )

    import numpy as np

    from data.episode_builder import load_day
    from env.betfair_env import BetfairEnv
    from predictors import PredictorBundle
    from training_v2.cohort.worker import scalping_train_config

    manifests = args.predictor_bundle_manifests or _default_manifests()
    bundle = PredictorBundle.from_manifests(
        champion_manifest=manifests[0],
        ranker_manifest=manifests[1],
        direction_manifest=manifests[2],
    )
    logger.info("bundle: champion=%s", bundle.champion_experiment_id)

    cfg = scalping_train_config()
    cfg["training"]["strategy_mode"] = "arb"

    # Aggregate max-p_win across all races across all days.
    all_max_pwins: list[float] = []
    per_day: dict[str, list[float]] = {}

    for day_str in args.days:
        day = load_day(day_str, data_dir=args.data_dir)
        env = BetfairEnv(
            day, cfg,
            predictor_bundle=bundle,
            use_race_outcome_predictor=True,
            use_direction_predictor=True,
            predictor_lean_obs=True,
            # gate disabled; we just want the cache populated.
            race_confidence_threshold=0.0,
        )
        day_maxes = []
        for race_pwins in env._race_p_win_by_race:
            if race_pwins:
                day_maxes.append(max(race_pwins.values()))
        all_max_pwins.extend(day_maxes)
        per_day[day_str] = day_maxes
        logger.info(
            "day %s: %d races, max-p_win range [%.3f, %.3f]",
            day_str, len(day_maxes),
            min(day_maxes) if day_maxes else float("nan"),
            max(day_maxes) if day_maxes else float("nan"),
        )

    arr = np.array(all_max_pwins, dtype=float)
    n = arr.size

    rows = [
        "",
        "PER-RACE max(champion p_win) DISTRIBUTION",
        "=" * 66,
        f"  total races: {n}",
        f"  days: {len(args.days)}",
        "",
        "QUANTILES:",
    ]
    for q_pct in (5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95):
        rows.append(
            f"  p{q_pct:>2d}: {np.percentile(arr, q_pct):.4f}"
        )

    rows.append("")
    rows.append("THRESHOLD CANDIDATE TABLE (frac of races BELOW threshold):")
    rows.append("  threshold | races skipped | qualification rate")
    rows.append("  " + "-" * 50)
    for thr in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        below = int(np.sum(arr < thr))
        qual = 1.0 - below / n
        rows.append(
            f"  {thr:.2f}      | {below:5d} ({below/n*100:5.1f}%) "
            f"| {qual*100:5.1f}%"
        )

    rows.append("")
    rows.append("PER-DAY MEDIAN max-p_win:")
    for d, vals in per_day.items():
        rows.append(
            f"  {d}: n={len(vals):3d} median={np.median(vals):.4f} "
            f"min={min(vals):.4f} max={max(vals):.4f}"
        )

    rows.append("=" * 66)

    # Suggested threshold: aim for ~25-40% of races to be non-confident
    # (legal_ratio ~60-75%, comfortably under the 80% bar without
    # starving the agent).
    target_lo, target_hi = 0.25, 0.40
    candidates = []
    for thr_int in range(15, 75):
        thr = thr_int / 100
        below = float(np.mean(arr < thr))
        if target_lo <= below <= target_hi:
            candidates.append((thr, below))
    if candidates:
        # Pick the threshold whose skip-rate is closest to the midpoint
        # of the target band (lands smack in §3's PASS region with
        # margin on both sides).
        target_mid = (target_lo + target_hi) / 2
        candidates.sort(key=lambda x: abs(x[1] - target_mid))
        best_thr, best_skip = candidates[0]
        rows.append(
            f"SUGGESTED THRESHOLD: {best_thr:.2f} "
            f"(skips {best_skip*100:.1f}% of races; target band "
            f"{target_lo*100:.0f}-{target_hi*100:.0f}%)"
        )
    else:
        rows.append(
            f"NO CANDIDATE in [{target_lo*100:.0f}-{target_hi*100:.0f}%] skip band — "
            f"distribution too narrow."
        )
    rows.append("")

    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
