"""Audit refusal reasons for the drift strategy.

For each drift fire, classify what the matcher would do WITHOUT
placing any bet. Buckets:

  - no_ltp: runner has no last_traded_price at the fire tick
  - no_book: no available_to_lay levels
  - junk_filter: best_lay outside [LTP*0.5, LTP*1.5] (max_price_deviation_pct=0.5)
  - hard_cap_lay: max_lay_price (None by default = never triggers)
  - hard_cap_back_close: best_back at close exceeds max_back_price=50 on close-out
  - ok: would match

Also reports per-bucket the median LTP and best_lay, so the user can
see whether the junk filter is rejecting ONLY illiquid runners or
also liquid drift candidates.

Usage:
    python -m tools.drift_refusal_audit \\
        --days 2026-05-04 2026-05-05 2026-05-06 \\
        --data-dir data/processed
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from pathlib import Path

logger = logging.getLogger("drift_refusal_audit")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", required=True, nargs="+")
    p.add_argument("--data-dir", default="data/processed", type=Path)
    p.add_argument("--predictor-bundle-manifests", nargs=3, default=None)
    p.add_argument("--max-price-deviation-pct", type=float, default=0.5,
                   help="Junk filter threshold (default 0.5 = ±50%% of LTP).")
    p.add_argument("--max-back-price", type=float, default=50.0,
                   help="Hard cap on back bets (matters for close-out leg).")
    p.add_argument("--max-lay-price", type=float, default=None,
                   help="Hard cap on lay bets (default None = no cap).")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def _default_manifests() -> tuple[str, str, str]:
    root = Path(__file__).resolve().parents[1]
    sibling = root.parent / "betfair-predictors"
    return (
        str(sibling / "production" / "race-outcome" / "manifest.json"),
        str(sibling / "production" / "race-outcome-ranker" / "manifest.json"),
        str(sibling / "production" / "direction-predictor" / "manifest.json"),
    )


def _med(xs):
    return statistics.median(xs) if xs else None


def main(argv: list[str] | None = None) -> int:
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

    # Buckets: reason -> {ltp: [...], best_lay: [...], best_back: [...]}
    buckets: dict[str, dict] = {
        "no_ltp": {"ltp": [], "best_lay": [], "best_back": []},
        "no_book_lay": {"ltp": [], "best_lay": [], "best_back": []},
        "junk_filter_lay": {"ltp": [], "best_lay": [], "best_back": []},
        "hard_cap_lay": {"ltp": [], "best_lay": [], "best_back": []},
        "ok_open_lay": {"ltp": [], "best_lay": [], "best_back": []},
    }

    for day_str in args.days:
        day = load_day(day_str, data_dir=args.data_dir)
        logger.info("=== %s: %d races ===", day_str, len(day.races))

        n_fires = 0
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

            for fi, (ti, sid) in enumerate(indices):
                if not fires[fi, 0]:
                    continue
                n_fires += 1
                tick = race.ticks[ti]
                snap = next((r for r in tick.runners if r.selection_id == sid), None)
                if snap is None:
                    buckets["no_ltp"]["ltp"].append(None)
                    continue

                ltp = getattr(snap, "last_traded_price", None)
                best_lay = (
                    snap.available_to_lay[0].price
                    if snap.available_to_lay else None
                )
                best_back = (
                    snap.available_to_back[0].price
                    if snap.available_to_back else None
                )

                if ltp is None or ltp <= 1.01:
                    bk = "no_ltp"
                elif best_lay is None:
                    bk = "no_book_lay"
                else:
                    lo = ltp * (1.0 - args.max_price_deviation_pct)
                    hi = ltp * (1.0 + args.max_price_deviation_pct)
                    if best_lay < lo or best_lay > hi:
                        bk = "junk_filter_lay"
                    elif args.max_lay_price is not None and best_lay > args.max_lay_price:
                        bk = "hard_cap_lay"
                    else:
                        bk = "ok_open_lay"

                if ltp is not None: buckets[bk]["ltp"].append(ltp)
                if best_lay is not None: buckets[bk]["best_lay"].append(best_lay)
                if best_back is not None: buckets[bk]["best_back"].append(best_back)
        logger.info("%s: %d drift fires processed", day_str, n_fires)

    print()
    print("=" * 92)
    print(f"DRIFT-FIRE REFUSAL AUDIT — junk filter ±{args.max_price_deviation_pct*100:.0f}%, "
          f"max_back_price={args.max_back_price}, max_lay_price={args.max_lay_price}")
    print("=" * 92)
    total = sum(len(b["ltp"]) for b in buckets.values()) + len(buckets["no_ltp"]["ltp"]) - len(buckets["no_ltp"]["ltp"])
    total = sum(
        max(len(b["ltp"]), len(b["best_lay"]), len(b["best_back"]))
        for b in buckets.values()
    )
    print(f"{'bucket':<22} {'N':>6} {'pct':>6} {'med_ltp':>9} {'med_best_lay':>13} {'med_best_back':>14}")
    print("-" * 92)
    for name, b in buckets.items():
        n = max(len(b["ltp"]), len(b["best_lay"]), len(b["best_back"]))
        pct = n / total * 100 if total else 0
        ml = _med(b["ltp"])
        mly = _med(b["best_lay"])
        mb = _med(b["best_back"])
        ml_s = f"{ml:.2f}" if ml is not None else "    -"
        mly_s = f"{mly:.2f}" if mly is not None else "    -"
        mb_s = f"{mb:.2f}" if mb is not None else "    -"
        print(f"{name:<22} {n:>6d} {pct:>5.1f}% {ml_s:>9} {mly_s:>13} {mb_s:>14}")
    print()
    # Detail on junk_filter_lay: how far over LTP is best_lay?
    jf = buckets["junk_filter_lay"]
    if jf["ltp"] and jf["best_lay"]:
        ratios = [bl / lt for bl, lt in zip(jf["best_lay"], jf["ltp"]) if lt > 0]
        if ratios:
            ratios.sort()
            print(f"junk_filter_lay deviation from LTP (best_lay / LTP):")
            print(f"  min={ratios[0]:.2f}x  p25={ratios[len(ratios)//4]:.2f}x  "
                  f"median={ratios[len(ratios)//2]:.2f}x  "
                  f"p75={ratios[3*len(ratios)//4]:.2f}x  max={ratios[-1]:.2f}x")
            print(f"  threshold for refusal: > {1+args.max_price_deviation_pct:.2f}x")
    print()
    print("Interpretation: if 'junk_filter_lay' bucket is large AND its median LTP")
    print("is comparable to 'ok_open_lay' (i.e. the runners are similar), the junk")
    print("filter is over-blocking liquid drift candidates. If the junk-bucket LTPs")
    print("are much higher (longshots only), the filter is doing its intended job.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
