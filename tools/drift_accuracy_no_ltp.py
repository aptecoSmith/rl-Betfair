"""Direction-predictor accuracy bucketed by LTP-availability.

For each drift fire, compute price_open and price_close at the
fire-tick and horizon-tick. Use LTP when available, fall back to
mid-of-book `(best_back + best_lay) / 2` when LTP is missing.
Bucket events by `had_ltp_at_open` to separate the predictor's
edge on liquid (recent-trade) runners from illiquid (no recent
trade) runners.

Hypothesis: 59% of drift fires happen on no-LTP runners. If the
predictor STILL has +20pp+ edge on that subset, infrastructure
work to act on those fires is worthwhile. If hit rate ~ base
rate on no-LTP, the predictor is also weak on illiquid runners
and the LTP gate is incidentally filtering low-value fires.

Usage:
    python -m tools.drift_accuracy_no_ltp \\
        --days 2026-05-04 2026-05-05 2026-05-06 \\
        --data-dir data/processed
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

logger = logging.getLogger("drift_accuracy_no_ltp")


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", required=True, nargs="+")
    p.add_argument("--data-dir", default="data/processed", type=Path)
    p.add_argument("--predictor-bundle-manifests", nargs=3, default=None)
    p.add_argument("--horizon-minutes", type=int, default=7)
    p.add_argument("--stake", type=float, default=10.0)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _default_manifests():
    root = Path(__file__).resolve().parents[1]
    sibling = root.parent / "betfair-predictors"
    return (
        str(sibling / "production" / "race-outcome" / "manifest.json"),
        str(sibling / "production" / "race-outcome-ranker" / "manifest.json"),
        str(sibling / "production" / "direction-predictor" / "manifest.json"),
    )


def _ref_price(snap):
    """Effective reference price: LTP if available, else mid-of-book."""
    ltp = getattr(snap, "last_traded_price", None)
    if ltp is not None and ltp > 1.01:
        return ltp, True  # (price, had_ltp)
    if snap.available_to_back and snap.available_to_lay:
        bb = snap.available_to_back[0].price
        bl = snap.available_to_lay[0].price
        if bb > 1.01 and bl > 1.01:
            return (bb + bl) / 2.0, False
    return None, False


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(message)s")

    from data.episode_builder import load_day
    from data.predictor_features import build_direction_windows_for_race
    from predictors import PredictorBundle

    manifests = args.predictor_bundle_manifests or _default_manifests()
    bundle = PredictorBundle.from_manifests(
        champion_manifest=manifests[0],
        ranker_manifest=manifests[1],
        direction_manifest=manifests[2],
    )
    logger.info("bundle direction=%s", bundle.direction_experiment_id)

    horizon = timedelta(minutes=args.horizon_minutes)

    # Buckets: with_ltp, no_ltp_midbook
    stats = {
        "with_ltp": {"events": 0, "correct": 0, "locked": 0.0, "open_prices": []},
        "no_ltp_midbook": {"events": 0, "correct": 0, "locked": 0.0, "open_prices": []},
    }
    # Base rate (whole population)
    base = {"with_ltp": [0, 0, 0], "no_ltp_midbook": [0, 0, 0]}  # [up, down, flat]

    for day_str in args.days:
        day = load_day(day_str, data_dir=args.data_dir)
        logger.info("=== %s: %d races ===", day_str, len(day.races))

        for race in day.races:
            try:
                windows, indices = build_direction_windows_for_race(race)
            except Exception:
                continue
            if windows.shape[0] == 0:
                continue
            try:
                _, fires = bundle.predict_tick_batch(windows)
            except Exception:
                continue

            # Build (ti, sid) -> (price, had_ltp) lookup using effective price
            ref_at = {}
            tick_ts = {}
            for ti, tick in enumerate(race.ticks):
                tick_ts[ti] = tick.timestamp
                for r in tick.runners:
                    pr, hd = _ref_price(r)
                    if pr is not None:
                        ref_at[(ti, r.selection_id)] = (pr, hd)

            for fi, (ti, sid) in enumerate(indices):
                ref_open = ref_at.get((ti, sid))
                if ref_open is None:
                    continue
                p_open, had_ltp = ref_open
                ts_open = tick_ts.get(ti)
                if ts_open is None:
                    continue

                # Find target tick at horizon
                target_ts = ts_open + horizon
                target_ti = None
                for tj in range(ti + 1, len(race.ticks)):
                    if race.ticks[tj].in_play:
                        target_ti = tj
                        break
                    if race.ticks[tj].timestamp >= target_ts:
                        target_ti = tj
                        break
                if target_ti is None:
                    target_ti = len(race.ticks) - 1
                ref_close = ref_at.get((target_ti, sid))
                if ref_close is None:
                    continue
                p_close = ref_close[0]
                if p_close <= 1.01 or p_open <= 1.01:
                    continue

                # Base-rate tally
                bucket_name = "with_ltp" if had_ltp else "no_ltp_midbook"
                if p_close > p_open:
                    base[bucket_name][0] += 1
                elif p_close < p_open:
                    base[bucket_name][1] += 1
                else:
                    base[bucket_name][2] += 1

                # Drift-fire-only tally
                if not fires[fi, 0]:
                    continue
                stats[bucket_name]["events"] += 1
                stats[bucket_name]["open_prices"].append(p_open)
                locked = args.stake * (p_close - p_open) / p_close
                stats[bucket_name]["locked"] += locked
                if p_close > p_open:
                    stats[bucket_name]["correct"] += 1

    print()
    print("=" * 92)
    print(f"DIRECTION-PREDICTOR ACCURACY BY LTP AVAILABILITY")
    print(f"Days: {', '.join(args.days)}  Horizon: {args.horizon_minutes}m  Stake: £{args.stake}")
    print("=" * 92)
    print(f"{'bucket':<20} {'drift_n':>8} {'hit%':>7} {'locked_pnl':>12} {'per_evt':>9} "
          f"{'base_up%':>9} {'edge_pp':>9} {'med_open':>10}")
    print("-" * 92)
    for name in ("with_ltp", "no_ltp_midbook"):
        s = stats[name]
        n = s["events"]
        if n == 0:
            print(f"{name:<20} (no events)")
            continue
        b = base[name]
        b_total = sum(b)
        base_up = b[0] / b_total if b_total else 0
        hit_rate = s["correct"] / n
        edge = (hit_rate - base_up) * 100
        med_open = sorted(s["open_prices"])[len(s["open_prices"])//2]
        print(
            f"{name:<20} {n:>8d} {hit_rate*100:>6.1f}% £{s['locked']:>+10.2f} £{s['locked']/n:>+7.2f} "
            f"{base_up*100:>8.1f}% {edge:>+8.1f} {med_open:>10.2f}"
        )
    print()
    print("Interpretation:")
    print("  with_ltp edge > 0 AND no_ltp_midbook edge > 0  -> predictor works on both;")
    print("    LTP gate is over-restrictive; building proper infrastructure has upside")
    print("  with_ltp edge > 0 AND no_ltp_midbook edge ~ 0  -> predictor is weak on illiquid;")
    print("    LTP gate is incidentally filtering low-value fires; don't bother relaxing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
