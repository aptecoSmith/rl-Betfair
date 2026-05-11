"""Confidence-bucketed direction-predictor accuracy.

For each drift fire, the predictor outputs quantile forecasts at 1m/3m/7m
horizons. The 7m spread (`q90_7m - q10_7m`) is a confidence proxy:
narrow spread = predictor confident, wide = uncertain.

Bucket all drift fires by quantile-spread decile (1 = narrowest /
most confident, 10 = widest / least confident) and report per-bucket:
  - hit rate (price actually moved up)
  - LTP-locked PnL per event
  - base rate within the bucket's truncation profile

If the narrowest-confidence bucket has hit rate >> base AND LTP-locked
PnL >> 0, a confidence filter rescues the drift signal for trading.
If accuracy is flat across buckets, the signal is uniform noise.

Usage:
    python -m tools.direction_confidence_buckets \\
        --days 2026-05-04 2026-05-05 2026-05-06 \\
        --data-dir data/processed \\
        --stake 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

logger = logging.getLogger("direction_confidence_buckets")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", required=True, nargs="+")
    p.add_argument("--data-dir", default="data/processed", type=Path)
    p.add_argument("--predictor-bundle-manifests", nargs=3, default=None)
    p.add_argument("--horizon-minutes", type=int, default=7)
    p.add_argument("--stake", type=float, default=10.0)
    p.add_argument("--n-buckets", type=int, default=10)
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(message)s")

    import numpy as np

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

    horizon_delta = timedelta(minutes=args.horizon_minutes)

    # Collect every drift event with (confidence_spread, hit, locked_pnl, q50_7m)
    drift_events: list[dict] = []

    for day_str in args.days:
        day = load_day(day_str, data_dir=args.data_dir)
        logger.info("=== %s: %d races ===", day_str, len(day.races))

        n_added = 0
        for race in day.races:
            try:
                windows, indices = build_direction_windows_for_race(race)
            except Exception:
                continue
            if windows.shape[0] == 0:
                continue
            try:
                quantiles, fires = bundle.predict_tick_batch(windows)
            except Exception:
                continue

            # Map (tick_idx, sid) -> LTP and ts
            ltp_at: dict[tuple[int, int], float] = {}
            tick_ts: dict[int, object] = {}
            for ti, tick in enumerate(race.ticks):
                tick_ts[ti] = tick.timestamp
                for r in tick.runners:
                    if getattr(r, "last_traded_price", None) is not None:
                        ltp_at[(ti, r.selection_id)] = r.last_traded_price

            for fire_idx, (ti, sid) in enumerate(indices):
                if not fires[fire_idx, 0]:
                    continue
                ts_open = tick_ts.get(ti)
                if ts_open is None:
                    continue
                ltp_open = ltp_at.get((ti, sid))
                if ltp_open is None or ltp_open <= 1.01:
                    continue

                # Find target tick at horizon
                target_ts = ts_open + horizon_delta
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
                ltp_close = ltp_at.get((target_ti, sid))
                if ltp_close is None or ltp_close <= 1.01:
                    continue

                # 7m horizon is the 3rd horizon (index 2). quantiles shape
                # (N, n_h, n_q) = (N, 3, 3) per predict_tick_batch contract.
                # Quantiles ordered q10, q50, q90.
                q10_7m = float(quantiles[fire_idx, 2, 0])
                q50_7m = float(quantiles[fire_idx, 2, 1])
                q90_7m = float(quantiles[fire_idx, 2, 2])
                spread_7m = q90_7m - q10_7m

                hit = ltp_close > ltp_open
                locked = args.stake * (ltp_close - ltp_open) / ltp_close

                drift_events.append({
                    "spread_7m": spread_7m,
                    "q50_7m": q50_7m,
                    "ltp_open": ltp_open,
                    "ltp_close": ltp_close,
                    "hit": hit,
                    "locked_pnl": locked,
                })
                n_added += 1
        logger.info("%s: %d drift events collected", day_str, n_added)

    n = len(drift_events)
    if n == 0:
        print("No drift events.")
        return 0

    # Bucket by spread_7m (ascending = narrowest first = most confident)
    drift_events.sort(key=lambda e: e["spread_7m"])
    bucket_size = max(1, n // args.n_buckets)

    # Overall base rate (any (tick,runner) — but we only have drift-fire
    # events here; we'll just print the population stats for context).
    pop_hit = sum(1 for e in drift_events if e["hit"])
    pop_locked = sum(e["locked_pnl"] for e in drift_events)

    print()
    print("=" * 96)
    print(f"DRIFT-FIRE CONFIDENCE BUCKETS (sorted by q90_7m - q10_7m spread)")
    print(f"Days: {', '.join(args.days)}   N events: {n}   Horizon: {args.horizon_minutes}m   Stake: £{args.stake}")
    print("=" * 96)
    print(f"Overall: hit_rate={pop_hit/n*100:.1f}%  total_locked_pnl=£{pop_locked:+.2f}  per_event=£{pop_locked/n:+.2f}")
    print()
    print(f"{'bucket':>7} {'N':>5} {'spread_range':>22} {'q50_med':>9} {'hit%':>7} {'locked_sum':>11} {'per_evt':>9} {'pct_total':>10}")
    print("-" * 96)
    for b in range(args.n_buckets):
        start = b * bucket_size
        end = (b + 1) * bucket_size if b < args.n_buckets - 1 else n
        bucket = drift_events[start:end]
        if not bucket:
            continue
        n_b = len(bucket)
        n_hit = sum(1 for e in bucket if e["hit"])
        locked_sum = sum(e["locked_pnl"] for e in bucket)
        spr_lo = bucket[0]["spread_7m"]
        spr_hi = bucket[-1]["spread_7m"]
        q50_med = sorted([e["q50_7m"] for e in bucket])[len(bucket)//2]
        share = locked_sum / pop_locked * 100 if pop_locked != 0 else 0
        print(
            f"{b+1:>7d} {n_b:>5d} {f'{spr_lo:.4f}..{spr_hi:.4f}':>22} "
            f"{q50_med:>+9.4f} {n_hit/n_b*100:>6.1f}% £{locked_sum:>+9.2f} "
            f"£{locked_sum/n_b:>+7.2f} {share:>+8.1f}%"
        )
    print()
    print("Interpretation:")
    print("  - hit% decreasing top-to-bottom -> confidence filter works; use bucket 1-2 only")
    print("  - hit% flat across buckets     -> confidence isn't a useful filter")
    print("  - per_evt > 0 in top buckets   -> LTP-level edge survives in confident subset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
