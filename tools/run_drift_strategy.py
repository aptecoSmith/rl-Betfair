"""Deterministic LAY-on-drift strategy with real Betfair matcher.

When the direction predictor fires `drift` on a runner at tick T:
  1. Lay the runner at current best lay price (stake S).
  2. Wait until tick where (timestamp >= T + 7min) OR race goes off
     OR (time_to_off <= --close-before-off-seconds).
  3. Place a hedging BACK at then-current best back, stake S * P_lay / P_back
     so the trade is equal-profit-locked.
  4. If the hedge fails (matcher refuses, no book), let the lay ride to settle.

PnL flows through `BetManager.settle_race` with 5% commission.

Hypothesis (from `tools/direction_predictor_accuracy.py` on 2026-05-04/05/06):
  Drift predictor has +31.5pp edge over base rate at 7m horizon. Locked
  PnL of +£2.72/event on LTP-vs-LTP simulator. Real matcher with spreads
  and commission should give a lower-but-still-positive ROI.

If this nets > 0% ROI on 3 days, directional betting is viable; build an
RL agent that decides WHEN NOT to trade.

Usage:
    python -m tools.run_drift_strategy \\
        --days 2026-05-04 2026-05-05 2026-05-06 \\
        --data-dir data/processed \\
        --stake 10 \\
        --horizon-minutes 7 \\
        --close-before-off-seconds 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path

logger = logging.getLogger("run_drift_strategy")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", required=True, nargs="+")
    p.add_argument("--data-dir", default="data/processed", type=Path)
    p.add_argument("--predictor-bundle-manifests", nargs=3, default=None)
    p.add_argument("--stake", type=float, default=10.0)
    p.add_argument("--horizon-minutes", type=float, default=7.0)
    p.add_argument("--close-before-off-seconds", type=float, default=30.0)
    p.add_argument(
        "--min-time-to-off-seconds", type=float, default=420.0,
        help="Skip opens where time_to_off < this (default 7min); "
             "ensures the horizon has room to play out.",
    )
    p.add_argument(
        "--max-spread-7m", type=float, default=None,
        help="Confidence filter: skip fires where (q90_7m - q10_7m) > "
             "this. From confidence_buckets analysis, ~26 keeps the "
             "75-88%% hit-rate buckets and drops the 46-49%% buckets.",
    )
    p.add_argument(
        "--max-q50-7m", type=float, default=None,
        help="Price filter: skip fires where q50_7m > this. From "
             "confidence_buckets analysis, ~12 selects mid-priced "
             "runners with deeper order books.",
    )
    p.add_argument("--commission", type=float, default=0.05)
    p.add_argument(
        "--starting-budget", type=float, default=100_000.0,
        help="Generous budget so liability clamps never bind.",
    )
    p.add_argument("--output", default=None, type=Path)
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
    from env.bet_manager import BetManager, MIN_BET_STAKE
    from predictors import PredictorBundle

    manifests = args.predictor_bundle_manifests or _default_manifests()
    bundle = PredictorBundle.from_manifests(
        champion_manifest=manifests[0],
        ranker_manifest=manifests[1],
        direction_manifest=manifests[2],
    )
    logger.info("bundle direction=%s", bundle.direction_experiment_id)

    horizon_delta = timedelta(minutes=args.horizon_minutes)
    close_before_off = timedelta(seconds=args.close_before_off_seconds)

    per_day_results = []
    all_trades = []

    for day_str in args.days:
        day = load_day(day_str, data_dir=args.data_dir)
        logger.info("=== %s: %d races ===", day_str, len(day.races))

        bm = BetManager(
            starting_budget=args.starting_budget,
            fill_mode=day.fill_mode,
        )

        day_trades = []
        day_opens_refused = 0
        day_closes_refused = 0
        day_naked_unintended = 0

        for race in day.races:
            try:
                windows, indices = build_direction_windows_for_race(race)
            except Exception as exc:
                logger.debug("race %s window build failed: %s", race.market_id, exc)
                continue
            if windows.shape[0] == 0:
                continue
            try:
                quantiles, fires = bundle.predict_tick_batch(windows)
            except Exception as exc:
                logger.debug("race %s predict failed: %s", race.market_id, exc)
                continue
            if int(fires[:, 0].sum()) == 0:
                continue  # no drift fires this race

            # fires[fire_idx, 0] = drift
            # quantiles shape (N, 3 horizons, 3 quantiles=q10/q50/q90); 7m = horizon idx 2
            # indices[fire_idx] = (tick_idx, sid)
            # Group fires by tick_idx; apply optional confidence filter.
            fires_by_tick: dict[int, list[tuple[int, int]]] = {}
            n_filtered = 0
            for fi, (ti, sid) in enumerate(indices):
                if not fires[fi, 0]:
                    continue
                if args.max_spread_7m is not None:
                    spread = float(quantiles[fi, 2, 2] - quantiles[fi, 2, 0])
                    if spread > args.max_spread_7m:
                        n_filtered += 1
                        continue
                if args.max_q50_7m is not None:
                    q50 = float(quantiles[fi, 2, 1])
                    if q50 > args.max_q50_7m:
                        n_filtered += 1
                        continue
                fires_by_tick.setdefault(ti, []).append((fi, sid))

            # Open positions: list of dicts {sid, open_tick_idx, open_ts,
            # close_ts, lay_price, lay_stake, market_id}
            open_positions: list[dict] = []

            for ti, tick in enumerate(race.ticks):
                if tick.in_play:
                    # Race went off — close any remaining naked positions
                    # by attempting back-out at this tick.
                    for pos in list(open_positions):
                        snap = next(
                            (r for r in tick.runners if r.selection_id == pos["sid"]),
                            None,
                        )
                        if snap is not None and snap.available_to_back:
                            back_price = snap.available_to_back[0].price
                            hedge_stake = pos["lay_stake"] * pos["lay_price"] / back_price
                            bet = bm.place_back(snap, hedge_stake, market_id=race.market_id)
                            if bet is not None:
                                pos["close_back_price"] = bet.average_price
                                pos["close_back_stake"] = bet.matched_stake
                                pos["close_reason"] = "in_play"
                            else:
                                pos["close_reason"] = "in_play_naked"
                                day_naked_unintended += 1
                                day_closes_refused += 1
                        else:
                            pos["close_reason"] = "in_play_no_book"
                            day_naked_unintended += 1
                        day_trades.append(pos)
                    open_positions.clear()
                    break

                tick_ts = tick.timestamp
                ts_to_off = race.market_start_time - tick_ts

                # Close-side: any open positions whose horizon expired OR
                # we're within close_before_off seconds of the off.
                for pos in list(open_positions):
                    should_close = False
                    if tick_ts >= pos["close_ts"]:
                        should_close = True
                        close_reason = "horizon"
                    elif ts_to_off <= close_before_off:
                        should_close = True
                        close_reason = "near_off"
                    if not should_close:
                        continue
                    snap = next(
                        (r for r in tick.runners if r.selection_id == pos["sid"]),
                        None,
                    )
                    if snap is None or not snap.available_to_back:
                        pos["close_reason"] = close_reason + "_no_book"
                        day_naked_unintended += 1
                        day_trades.append(pos)
                        open_positions.remove(pos)
                        continue
                    back_price = snap.available_to_back[0].price
                    if back_price <= 1.01:
                        pos["close_reason"] = close_reason + "_bad_price"
                        day_naked_unintended += 1
                        day_trades.append(pos)
                        open_positions.remove(pos)
                        continue
                    hedge_stake = pos["lay_stake"] * pos["lay_price"] / back_price
                    if hedge_stake < MIN_BET_STAKE:
                        hedge_stake = MIN_BET_STAKE
                    bet = bm.place_back(snap, hedge_stake, market_id=race.market_id)
                    if bet is None:
                        pos["close_reason"] = close_reason + "_refused"
                        day_closes_refused += 1
                        day_naked_unintended += 1
                    else:
                        pos["close_back_price"] = bet.average_price
                        pos["close_back_stake"] = bet.matched_stake
                        pos["close_tick_idx"] = ti
                        pos["close_reason"] = close_reason
                    day_trades.append(pos)
                    open_positions.remove(pos)

                # Open-side: drift fires at this tick.
                if ti not in fires_by_tick:
                    continue
                if ts_to_off < timedelta(seconds=args.min_time_to_off_seconds):
                    # Not enough room for horizon to play out
                    continue
                # Skip sids already in open_positions (no doubling)
                open_sids = {p["sid"] for p in open_positions}
                for _, sid in fires_by_tick[ti]:
                    if sid in open_sids:
                        continue
                    snap = next(
                        (r for r in tick.runners if r.selection_id == sid),
                        None,
                    )
                    if snap is None or not snap.available_to_lay:
                        continue
                    lay_price = snap.available_to_lay[0].price
                    if lay_price <= 1.01:
                        continue
                    bet = bm.place_lay(snap, args.stake, market_id=race.market_id)
                    if bet is None:
                        day_opens_refused += 1
                        continue
                    open_positions.append({
                        "market_id": race.market_id,
                        "sid": sid,
                        "open_tick_idx": ti,
                        "open_ts": tick_ts,
                        "close_ts": tick_ts + horizon_delta,
                        "lay_price": bet.average_price,
                        "lay_stake": bet.matched_stake,
                    })

            # Race ended without going off-tick observed (data ends).
            # Any remaining open positions are naked — leave for settle.
            for pos in open_positions:
                pos["close_reason"] = "end_of_data"
                day_naked_unintended += 1
                day_trades.append(pos)

        # Settle the day.
        day_pnl = 0.0
        for race in day.races:
            if race.winner_selection_id is not None or race.winning_selection_ids:
                winners = race.winning_selection_ids or {race.winner_selection_id}
                race_pnl = bm.settle_race(
                    winning_selection_ids=set(int(s) for s in winners if s is not None),
                    market_id=race.market_id,
                    commission=args.commission,
                    each_way_divisor=race.each_way_divisor,
                    winner_selection_id=race.winner_selection_id,
                    number_of_places=race.number_of_each_way_places,
                )
                day_pnl += race_pnl

        # Map settled bets back to trades for outcomes. Bet.side is BetSide.LAY/BACK.
        from env.bet_manager import BetSide
        bet_lookup_lay: dict[tuple[str, int], list] = {}
        bet_lookup_back: dict[tuple[str, int], list] = {}
        for b in bm.bets:
            key = (b.market_id, b.selection_id)
            if b.side is BetSide.LAY:
                bet_lookup_lay.setdefault(key, []).append(b)
            else:
                bet_lookup_back.setdefault(key, []).append(b)

        for tr in day_trades:
            mid = tr["market_id"]
            sid = tr["sid"]
            lay_bets = bet_lookup_lay.get((mid, sid), [])
            back_bets = bet_lookup_back.get((mid, sid), [])
            tr["lay_pnl"] = sum(float(b.pnl) for b in lay_bets)
            tr["back_pnl"] = sum(float(b.pnl) for b in back_bets)
            tr["combined_pnl"] = tr["lay_pnl"] + tr["back_pnl"]

        total_stake = sum(t.get("lay_stake", 0) for t in day_trades)
        n_trades = len(day_trades)
        n_closed = sum(
            1 for t in day_trades
            if t.get("close_reason") in {"horizon", "near_off", "in_play"}
        )
        n_naked = n_trades - n_closed
        per_day_results.append({
            "day": day_str,
            "n_races": len(day.races),
            "n_trades_opened": n_trades,
            "n_trades_closed": n_closed,
            "n_trades_naked": n_naked,
            "n_opens_refused": day_opens_refused,
            "n_closes_refused": day_closes_refused,
            "n_naked_unintended": day_naked_unintended,
            "total_lay_stake": total_stake,
            "day_pnl": day_pnl,
            "roi": day_pnl / total_stake if total_stake > 0 else 0.0,
        })
        all_trades.extend(day_trades)

        logger.info(
            "%s: races=%d trades=%d closed=%d naked=%d opens_refused=%d "
            "closes_refused=%d total_stake=£%.0f pnl=£%+.2f roi=%+.2f%%",
            day_str, len(day.races), n_trades, n_closed, n_naked,
            day_opens_refused, day_closes_refused, total_stake, day_pnl,
            (day_pnl / total_stake * 100) if total_stake > 0 else 0.0,
        )

    # Summary
    total_trades = sum(d["n_trades_opened"] for d in per_day_results)
    total_closed = sum(d["n_trades_closed"] for d in per_day_results)
    total_naked = sum(d["n_trades_naked"] for d in per_day_results)
    total_stake = sum(d["total_lay_stake"] for d in per_day_results)
    total_pnl = sum(d["day_pnl"] for d in per_day_results)

    print()
    print("=" * 80)
    print("LAY-ON-DRIFT STRATEGY — RESULTS (real Betfair matcher + commission)")
    print("=" * 80)
    print(
        f"Stake: £{args.stake}  Horizon: {args.horizon_minutes}m  "
        f"Close-before-off: {args.close_before_off_seconds}s  "
        f"Commission: {args.commission*100:.1f}%"
    )
    print()
    print(f"{'day':>12} {'races':>6} {'trades':>7} {'closed':>7} {'naked':>6} "
          f"{'stake':>10} {'pnl':>11} {'roi':>8}")
    for d in per_day_results:
        print(
            f"{d['day']:>12} {d['n_races']:>6} {d['n_trades_opened']:>7} "
            f"{d['n_trades_closed']:>7} {d['n_trades_naked']:>6} "
            f"£{d['total_lay_stake']:>8.0f} £{d['day_pnl']:>+9.2f} "
            f"{d['roi']*100:>+6.2f}%"
        )
    print("-" * 80)
    print(
        f"{'TOTAL':>12} {sum(d['n_races'] for d in per_day_results):>6} "
        f"{total_trades:>7} {total_closed:>7} {total_naked:>6} "
        f"£{total_stake:>8.0f} £{total_pnl:>+9.2f} "
        f"{(total_pnl/total_stake*100) if total_stake>0 else 0:>+6.2f}%"
    )
    print()
    closed_rate = total_closed / total_trades * 100 if total_trades else 0
    print(f"Close success rate: {closed_rate:.1f}% ({total_closed}/{total_trades})")
    print(f"Naked (unintended): {total_naked} trades rode to settle")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump({
                "args": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                "per_day": per_day_results,
                "total": {
                    "n_trades": total_trades,
                    "n_closed": total_closed,
                    "n_naked": total_naked,
                    "total_stake": total_stake,
                    "total_pnl": total_pnl,
                    "roi_pct": (total_pnl/total_stake*100) if total_stake>0 else 0,
                },
                "trades": [
                    {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in tr.items()}
                    for tr in all_trades
                ],
            }, f, indent=2, default=str)
        logger.info("results written to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
