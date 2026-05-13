"""Deterministic "lay whenever the combined gate permits" strategy.

Answers the operator's question: of the ~5-10 lay opportunities per
race that survive (pwin gate AND direction-drift gate), what would
the PnL be if we ONLY laid on those, nothing else?

No RL. No training. Just walk every (tick, runner) in the eval days;
if BOTH gates permit OPEN_LAY at that slot AND the runner is otherwise
legal (active, has LTP, no existing position on that sid), place a
flat-stake LAY through BetManager. The env's auto-passive mechanism
posts the passive back leg as normal. Settle each race; report:

  - bets matched per day
  - win rate (pairs that matured or closed in our favour)
  - per-bet PnL distribution
  - aggregate PnL

Compares directly to the operator's standard: "5 bets a day is fine
if we win them all."

Usage:
    python -m tools.run_gated_lay_strategy \\
        --days 2026-05-04 2026-05-05 2026-05-06 \\
        --data-dir data/processed \\
        --stake 10 \\
        --pwin-lay-threshold 0.40
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("run_gated_lay_strategy")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", required=True, nargs="+", metavar="YYYY-MM-DD")
    p.add_argument("--data-dir", default="data/processed", type=Path)
    p.add_argument("--predictor-bundle-manifests", nargs=3, default=None)
    p.add_argument("--stake", type=float, default=10.0)
    p.add_argument(
        "--pwin-lay-threshold", type=float, default=0.40,
        help="Only lay runners with p_win <= this. Matches the "
             "pwin-gate cohort default.",
    )
    p.add_argument(
        "--min-time-to-off-seconds", type=float, default=60.0,
        help="Skip opens where time_to_off < this. Default 60s.",
    )
    p.add_argument(
        "--commission", type=float, default=0.05,
    )
    p.add_argument(
        "--starting-budget", type=float, default=100_000.0,
        help="Generous so liability clamps don't bind.",
    )
    p.add_argument("--output", default=None, type=Path)
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
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(message)s")

    from datetime import timedelta

    from data.episode_builder import load_day
    from data.predictor_features import build_predict_race_dataframe
    from data.predictor_features import build_direction_windows_for_race
    from env.bet_manager import BetManager, BetSide, MIN_BET_STAKE
    from predictors import PredictorBundle

    manifests = args.predictor_bundle_manifests or _default_manifests()
    bundle = PredictorBundle.from_manifests(
        champion_manifest=manifests[0],
        ranker_manifest=manifests[1],
        direction_manifest=manifests[2],
    )
    logger.info("bundle direction=%s champion=%s",
                bundle.direction_experiment_id,
                bundle.champion_experiment_id)

    per_day: list[dict] = []
    all_pairs: list[dict] = []
    min_t = timedelta(seconds=args.min_time_to_off_seconds)

    for day_str in args.days:
        day = load_day(day_str, data_dir=args.data_dir)
        from datetime import datetime as _dt
        as_of = _dt.strptime(day_str, "%Y-%m-%d").date()
        logger.info("=== %s: %d races ===", day_str, len(day.races))

        bm = BetManager(
            starting_budget=args.starting_budget,
            fill_mode=day.fill_mode,
        )

        day_bets: list[dict] = []
        opens_attempted = 0
        opens_matched = 0
        opens_refused_matcher = 0
        opens_refused_min_t = 0

        for race in day.races:
            # Champion p_win per runner (once per race)
            try:
                df = build_predict_race_dataframe(race, as_of_date=as_of)
                race_outputs = bundle.predict_race(df)
                p_win_by_sid = dict(race_outputs.p_win)
            except Exception as exc:
                logger.debug("race %s predict_race failed: %s",
                             race.market_id, exc)
                continue

            # Direction predictor outputs per (tick, runner)
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
            drift_by_key: dict[tuple[int, int], bool] = {
                (ti, sid): bool(fires[fi, 0])
                for fi, (ti, sid) in enumerate(indices)
            }

            # Track sids we've already opened a pair on (one open per sid per race)
            opened_sids: set[int] = set()
            pair_counter = 0

            for ti, tick in enumerate(race.ticks):
                if tick.in_play:
                    break
                ts_to_off = race.market_start_time - tick.timestamp
                if ts_to_off < min_t:
                    # Too close to off; would have insufficient room
                    # for passive back to fill / settle window
                    continue

                # For each active runner: check both gates and runner state
                for runner in tick.runners:
                    sid = runner.selection_id
                    if sid in opened_sids:
                        continue
                    if runner.status != "ACTIVE":
                        continue
                    ltp = getattr(runner, "last_traded_price", None)
                    if ltp is None or ltp <= 1.0:
                        continue
                    if not runner.available_to_back:
                        # Need book to lay aggressively (we hit best back)
                        continue
                    # Champion gate: only lay low-p_win runners
                    p_win = p_win_by_sid.get(sid, 0.0)
                    if p_win > args.pwin_lay_threshold:
                        continue
                    # Direction gate: drift must fire on this (tick, sid)
                    if not drift_by_key.get((ti, sid), False):
                        continue

                    # Both gates pass. Place an aggressive LAY (scalping
                    # mechanic — BetManager auto-posts the passive back).
                    pair_counter += 1
                    pair_id = f"{race.market_id}-{sid}-{pair_counter}"
                    opens_attempted += 1
                    bet = bm.place_lay(
                        runner, stake=args.stake,
                        market_id=race.market_id,
                        pair_id=pair_id,
                    )
                    if bet is None:
                        opens_refused_matcher += 1
                        continue
                    opens_matched += 1
                    opened_sids.add(sid)
                    day_bets.append({
                        "market_id": race.market_id,
                        "selection_id": sid,
                        "pair_id": pair_id,
                        "open_tick_idx": ti,
                        "open_time_to_off_s": ts_to_off.total_seconds(),
                        "open_lay_price": bet.average_price,
                        "open_lay_stake": bet.matched_stake,
                        "p_win": p_win,
                    })

        # Settle every race for this day
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

        # Per-pair outcome lookup from settled bets
        bets_by_sid: dict[tuple[str, int], list] = {}
        for b in bm.bets:
            bets_by_sid.setdefault((b.market_id, b.selection_id), []).append(b)
        for tr in day_bets:
            key = (tr["market_id"], tr["selection_id"])
            legs = bets_by_sid.get(key, [])
            lay_legs = [b for b in legs if b.side is BetSide.LAY]
            back_legs = [b for b in legs if b.side is BetSide.BACK]
            lay_pnl = sum(float(b.pnl) for b in lay_legs)
            back_pnl = sum(float(b.pnl) for b in back_legs)
            tr["combined_pnl"] = lay_pnl + back_pnl
            tr["lay_pnl"] = lay_pnl
            tr["back_pnl"] = back_pnl
            tr["passive_filled"] = bool(back_legs)

        all_pairs.extend(day_bets)
        n_bets = len(day_bets)
        n_passive_filled = sum(1 for t in day_bets if t["passive_filled"])
        n_winning = sum(1 for t in day_bets if t["combined_pnl"] > 0)
        total_stake = sum(t["open_lay_stake"] for t in day_bets)
        per_day.append({
            "day": day_str,
            "n_races": len(day.races),
            "opens_attempted": opens_attempted,
            "opens_matched": opens_matched,
            "opens_refused_matcher": opens_refused_matcher,
            "n_passive_filled_pairs": n_passive_filled,
            "n_winning_bets": n_winning,
            "win_rate": n_winning / n_bets if n_bets > 0 else 0.0,
            "total_stake": total_stake,
            "day_pnl": day_pnl,
            "roi": day_pnl / total_stake if total_stake > 0 else 0.0,
        })
        logger.info(
            "%s: races=%d attempted=%d matched=%d passive_filled=%d "
            "winning=%d/%d (%.0f%%) stake=GBP%.0f pnl=GBP%+.2f roi=%+.1f%%",
            day_str, len(day.races), opens_attempted, opens_matched,
            n_passive_filled, n_winning, n_bets,
            (n_winning / n_bets * 100) if n_bets > 0 else 0,
            total_stake, day_pnl,
            (day_pnl / total_stake * 100) if total_stake > 0 else 0,
        )

    # Aggregate
    total_attempted = sum(d["opens_attempted"] for d in per_day)
    total_matched = sum(d["opens_matched"] for d in per_day)
    total_winning = sum(d["n_winning_bets"] for d in per_day)
    total_stake = sum(d["total_stake"] for d in per_day)
    total_pnl = sum(d["day_pnl"] for d in per_day)

    print()
    print("=" * 84)
    print("GATED LAY STRATEGY (pwin <= {:.2f} AND dir_fire_drift) — RESULTS"
          .format(args.pwin_lay_threshold))
    print("=" * 84)
    print(
        f"Stake: GBP{args.stake}  Commission: {args.commission*100:.1f}%  "
        f"Min t-to-off: {args.min_time_to_off_seconds}s"
    )
    print()
    print(f"{'day':>12} {'races':>6} {'tried':>6} {'matched':>8} "
          f"{'pasv_fill':>9} {'winning':>8} {'win%':>6} {'pnl':>10} {'roi':>7}")
    for d in per_day:
        n = d["opens_matched"]
        print(
            f"{d['day']:>12} {d['n_races']:>6} {d['opens_attempted']:>6} "
            f"{n:>8} {d['n_passive_filled_pairs']:>9} {d['n_winning_bets']:>8} "
            f"{d['win_rate']*100:>5.1f}% GBP{d['day_pnl']:>+8.2f} "
            f"{d['roi']*100:>+5.1f}%"
        )
    print("-" * 84)
    print(
        f"{'TOTAL':>12} {sum(d['n_races'] for d in per_day):>6} "
        f"{total_attempted:>6} {total_matched:>8} "
        f"{sum(d['n_passive_filled_pairs'] for d in per_day):>9} "
        f"{total_winning:>8} "
        f"{(total_winning / total_matched * 100) if total_matched else 0:>5.1f}% "
        f"GBP{total_pnl:>+8.2f} "
        f"{(total_pnl / total_stake * 100) if total_stake else 0:>+5.1f}%"
    )
    print()
    if total_matched > 0:
        mean_per_bet = total_pnl / total_matched
        print(f"Mean PnL per matched lay: GBP{mean_per_bet:+.2f}")
        # Win/loss distribution
        wins = [t["combined_pnl"] for t in all_pairs if t["combined_pnl"] > 0]
        losses = [t["combined_pnl"] for t in all_pairs if t["combined_pnl"] <= 0]
        if wins:
            print(f"  winners (n={len(wins)}): mean GBP{sum(wins)/len(wins):+.2f}  "
                  f"sum GBP{sum(wins):+.2f}")
        if losses:
            print(f"  losers  (n={len(losses)}): mean GBP{sum(losses)/len(losses):+.2f}  "
                  f"sum GBP{sum(losses):+.2f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump({
                "args": {k: str(v) if isinstance(v, Path) else v
                         for k, v in vars(args).items()},
                "per_day": per_day,
                "total": {
                    "opens_attempted": total_attempted,
                    "opens_matched": total_matched,
                    "n_winning": total_winning,
                    "total_stake": total_stake,
                    "total_pnl": total_pnl,
                },
                "pairs": [
                    {k: (v.isoformat() if hasattr(v, "isoformat") else v)
                     for k, v in p.items()}
                    for p in all_pairs
                ],
            }, f, indent=2, default=str)
        logger.info("results written to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
