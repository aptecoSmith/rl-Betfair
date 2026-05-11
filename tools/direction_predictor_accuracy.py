"""Direction-predictor accuracy test on held-out days.

For each (tick, runner) where the predictor fires `drift` or `shorten`,
record the LTP at trigger and at the prediction horizon. Compute
hit-rate per direction class. Also compute the simulated locked PnL
of a back-first (on shorten) / lay-first (on drift) trade closed at
horizon at the then-current opposite-side best price.

No simulator, no matcher, no RL — just the deterministic predictor
output measured against the actual price trajectory.

If hit rate is materially above 50% AND simulated locked PnL is
positive, the predictor IS useful as a trading signal; the failure
to help the RL agent in the scalping cohort is then about the
ACTION shape (open-and-hold over 7 min), not the predictor.

If hit rate is ~50% / locked PnL ≈ 0, the predictor doesn't give
edge at the 7-min timescale on these held-out days and the directional
betting idea is moot.

Usage:
    python -m tools.direction_predictor_accuracy \\
        --days 2026-05-04 2026-05-05 2026-05-06 \\
        --data-dir data/processed \\
        --horizon-minutes 7 \\
        --stake 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

logger = logging.getLogger("direction_predictor_accuracy")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", required=True, nargs="+", metavar="YYYY-MM-DD")
    p.add_argument("--data-dir", default="data/processed", type=Path)
    p.add_argument("--predictor-bundle-manifests", nargs=3, default=None)
    p.add_argument(
        "--horizon-minutes", type=int, default=7,
        help="Prediction horizon in minutes (predictor's classifier "
             "fires at 7m by default).",
    )
    p.add_argument(
        "--stake", type=float, default=10.0,
        help="Notional stake for simulated locked-PnL calculation.",
    )
    p.add_argument(
        "--max-races-per-day", type=int, default=None,
        help="Cap races per day for fast iteration.",
    )
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

    # Per-day aggregates
    overall = {"shorten_events": 0, "shorten_correct": 0, "shorten_locked_pnl": 0.0,
               "drift_events": 0, "drift_correct": 0, "drift_locked_pnl": 0.0,
               "horizon_truncated": 0, "no_ltp_at_open": 0, "no_ltp_at_close": 0,
               "base_total": 0, "base_up": 0, "base_down": 0, "base_flat": 0}
    per_day = []

    for day_str in args.days:
        day = load_day(day_str, data_dir=args.data_dir)
        races = day.races[:args.max_races_per_day] if args.max_races_per_day else day.races
        logger.info("=== %s: %d races ===", day_str, len(races))

        day_stats = {"shorten_events": 0, "shorten_correct": 0, "shorten_locked_pnl": 0.0,
                     "drift_events": 0, "drift_correct": 0, "drift_locked_pnl": 0.0,
                     "horizon_truncated": 0, "no_ltp_at_open": 0, "no_ltp_at_close": 0,
                     "base_total": 0, "base_up": 0, "base_down": 0, "base_flat": 0}

        for race in races:
            try:
                windows, indices = build_direction_windows_for_race(race)
            except Exception as exc:
                logger.warning("race %s window build failed: %s", race.market_id, exc)
                continue
            if windows.shape[0] == 0:
                continue
            try:
                _, fires = bundle.predict_tick_batch(windows)
            except Exception as exc:
                logger.warning("race %s predict failed: %s", race.market_id, exc)
                continue
            # fires shape (N, 3) = [drift, shorten, no_signal]
            n_drift = int(fires[:, 0].sum())
            n_shorten = int(fires[:, 1].sum())
            if n_drift + n_shorten == 0:
                continue

            # Map (tick_idx, sid) -> LTP. Use last_traded_price from
            # the RunnerSnap. Skip when missing.
            ltp_at = {}
            tick_ts = {}
            for ti, tick in enumerate(race.ticks):
                tick_ts[ti] = tick.timestamp
                for r in tick.runners:
                    if getattr(r, "last_traded_price", None) is not None:
                        ltp_at[(ti, r.selection_id)] = r.last_traded_price

            for fire_idx, (ti, sid) in enumerate(indices):
                ts_open = tick_ts.get(ti)
                if ts_open is None:
                    continue
                ltp_open = ltp_at.get((ti, sid))
                if ltp_open is None or ltp_open <= 1.01:
                    day_stats["no_ltp_at_open"] += 1
                    continue

                # Find target tick at ts_open + horizon.
                target_ts = ts_open + horizon_delta
                target_ti = None
                for tj in range(ti + 1, len(race.ticks)):
                    if race.ticks[tj].in_play:
                        target_ti = tj
                        day_stats["horizon_truncated"] += 1
                        break
                    if race.ticks[tj].timestamp >= target_ts:
                        target_ti = tj
                        break
                if target_ti is None:
                    day_stats["horizon_truncated"] += 1
                    target_ti = len(race.ticks) - 1
                ltp_close = ltp_at.get((target_ti, sid))
                if ltp_close is None or ltp_close <= 1.01:
                    day_stats["no_ltp_at_close"] += 1
                    continue

                # Base-rate tally (counts EVERY (tick,runner) regardless of fires)
                day_stats["base_total"] += 1
                if ltp_close > ltp_open:
                    day_stats["base_up"] += 1
                elif ltp_close < ltp_open:
                    day_stats["base_down"] += 1
                else:
                    day_stats["base_flat"] += 1

                # Predictor-firing-only tally
                if not fires[fire_idx, 0] and not fires[fire_idx, 1]:
                    continue
                if fires[fire_idx, 1]:  # shorten
                    day_stats["shorten_events"] += 1
                    locked = args.stake * (ltp_open - ltp_close) / ltp_close
                    day_stats["shorten_locked_pnl"] += locked
                    if ltp_close < ltp_open:
                        day_stats["shorten_correct"] += 1
                else:  # drift
                    day_stats["drift_events"] += 1
                    locked = args.stake * (ltp_close - ltp_open) / ltp_close
                    day_stats["drift_locked_pnl"] += locked
                    if ltp_close > ltp_open:
                        day_stats["drift_correct"] += 1

        for k, v in day_stats.items():
            overall[k] += v
        per_day.append((day_str, day_stats))

        s_n = day_stats["shorten_events"]
        s_c = day_stats["shorten_correct"]
        s_pnl = day_stats["shorten_locked_pnl"]
        d_n = day_stats["drift_events"]
        d_c = day_stats["drift_correct"]
        d_pnl = day_stats["drift_locked_pnl"]
        logger.info(
            "%s: shorten %d events, %.1f%% correct, locked_pnl £%+.2f | "
            "drift %d events, %.1f%% correct, locked_pnl £%+.2f",
            day_str,
            s_n, (s_c / s_n * 100) if s_n else 0, s_pnl,
            d_n, (d_c / d_n * 100) if d_n else 0, d_pnl,
        )

    # Aggregate
    print()
    print("=" * 78)
    print("DIRECTION-PREDICTOR ACCURACY — HELD-OUT")
    print("=" * 78)
    print(f"Days: {', '.join(args.days)}  Horizon: {args.horizon_minutes}m  Stake: £{args.stake}")
    print()
    print(f"{'day':>12} {'sh_n':>5} {'sh_pct':>7} {'sh_pnl':>10} {'dr_n':>5} {'dr_pct':>7} {'dr_pnl':>10} {'trunc':>6}")
    for day_str, d in per_day:
        s_n, s_c, s_pnl = d["shorten_events"], d["shorten_correct"], d["shorten_locked_pnl"]
        dr_n, dr_c, dr_pnl = d["drift_events"], d["drift_correct"], d["drift_locked_pnl"]
        tr = d["horizon_truncated"]
        s_pct = (s_c / s_n * 100) if s_n else 0
        d_pct = (dr_c / dr_n * 100) if dr_n else 0
        print(f"{day_str:>12} {s_n:>5d} {s_pct:>6.1f}% £{s_pnl:>+8.2f} {dr_n:>5d} {d_pct:>6.1f}% £{dr_pnl:>+8.2f} {tr:>6d}")
    print("-" * 78)
    tot_s_n, tot_s_c, tot_s_pnl = overall["shorten_events"], overall["shorten_correct"], overall["shorten_locked_pnl"]
    tot_d_n, tot_d_c, tot_d_pnl = overall["drift_events"], overall["drift_correct"], overall["drift_locked_pnl"]
    print(f"{'TOTAL':>12} {tot_s_n:>5d} {(tot_s_c/tot_s_n*100) if tot_s_n else 0:>6.1f}% £{tot_s_pnl:>+8.2f} {tot_d_n:>5d} {(tot_d_c/tot_d_n*100) if tot_d_n else 0:>6.1f}% £{tot_d_pnl:>+8.2f} {overall['horizon_truncated']:>6d}")
    print()
    print(f"horizon_truncated (race went off or no tick beyond horizon): {overall['horizon_truncated']}")
    print(f"no_ltp_at_open: {overall['no_ltp_at_open']}  no_ltp_at_close: {overall['no_ltp_at_close']}")
    print()
    print("Interpretation:")
    print("  hit% > base-rate AND locked_pnl > 0 -> predictor has edge -> directional bet idea is viable")
    print("  hit% ~ base-rate AND locked_pnl ~ 0 -> predictor has no edge on held-out -> idea is moot")
    print()
    bt = overall["base_total"]
    if bt > 0:
        print(f"BASE RATE (all (tick,runner) pairs across cohort, regardless of predictor firing):")
        print(f"  total samples: {bt}")
        print(f"  price went UP:   {overall['base_up']} ({overall['base_up']/bt*100:.1f}%)")
        print(f"  price went DOWN: {overall['base_down']} ({overall['base_down']/bt*100:.1f}%)")
        print(f"  price stayed:    {overall['base_flat']} ({overall['base_flat']/bt*100:.1f}%)")
        print()
        # Compare predictor accuracy vs base-rate
        if tot_d_n > 0:
            base_up_rate = overall['base_up']/bt
            pred_up_rate = tot_d_c/tot_d_n
            print(f"  Drift predictor: {pred_up_rate*100:.1f}% accurate vs {base_up_rate*100:.1f}% base-rate -> "
                  f"edge {(pred_up_rate - base_up_rate)*100:+.1f}pp")
        if tot_s_n > 0:
            base_down_rate = overall['base_down']/bt
            pred_down_rate = tot_s_c/tot_s_n
            print(f"  Shorten predictor: {pred_down_rate*100:.1f}% accurate vs {base_down_rate*100:.1f}% base-rate -> "
                  f"edge {(pred_down_rate - base_down_rate)*100:+.1f}pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
