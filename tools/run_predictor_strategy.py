"""Deterministic predictor strategy — substrate-validation tool.

Runs the betfair-predictors champion's value-spotting strategy through
rl-betfair's `BetfairEnv` matching simulator. NO RL, NO training —
just the deterministic logic from the manifest's
`value_spotting_at_inference_time` block:

    1. predict_race(market) -> per-runner p_win
    2. pick argmax(p_win)
    3. compute edge = p_win - implied_p_win  (implied from current best back price)
    4. if edge > THRESHOLD AND segment_strong: place a flat-£STAKE back bet
    5. hold to settle

Compare aggregate PnL to the predictor repo's own flat-£10 backtest.
If they roughly match, my data-bridging is correct; if they diverge,
I have a bug or the matcher behaves materially differently from the
predictor's flat-stake assumption.

Usage:
    python -m tools.run_predictor_strategy \\
        --days 2026-05-04 2026-05-05 2026-05-06 \\
        --data-dir data/processed \\
        --stake 10 \\
        --edge-threshold 0.05 \\
        --bet-tick-frac 0.5

Defaults match the manifest's recommended consumer logic + the
predictor's flat-£10 backtest convention. ``--bet-tick-frac`` is
where in the pre-off tick stream to place the bet (0.5 = halfway
through the pre-off window, 1.0 = last pre-off tick). The predictor's
backtest uses BSP, which corresponds to the very last pre-off tick;
0.5 gives more time for the matcher to find liquidity at the chosen
price.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("run_predictor_strategy")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--days", required=True, nargs="+", metavar="YYYY-MM-DD",
        help="One or more days to run the strategy on.",
    )
    p.add_argument(
        "--data-dir", default="data/processed", type=Path,
        help="Directory containing YYYY-MM-DD.parquet day files.",
    )
    p.add_argument(
        "--predictor-bundle-manifests", nargs=3, default=None,
        metavar=("CHAMPION", "RANKER", "DIRECTION"),
        help=(
            "Three paths to manifest.json files. Defaults to the "
            "production manifests under sibling betfair-predictors."
        ),
    )
    p.add_argument(
        "--stake", type=float, default=10.0,
        help="Flat stake per bet (matches predictor backtest = £10).",
    )
    p.add_argument(
        "--edge-threshold", type=float, default=0.05,
        help=(
            "Minimum (champion_p_win - implied_p_win) to fire. "
            "Manifest's recommended threshold = 0.05."
        ),
    )
    p.add_argument(
        "--bet-tick-frac", type=float, default=0.5,
        help=(
            "Fraction through the pre-off tick stream to place the bet. "
            "0.0 = first pre-off tick; 1.0 = last pre-off tick (closest "
            "to BSP). Default 0.5."
        ),
    )
    p.add_argument(
        "--require-segment-strong", action="store_true",
        help=(
            "Only fire when champion's segment_strong_flag is True. "
            "Matches the manifest's strict consumer logic."
        ),
    )
    p.add_argument(
        "--output", default=None, type=Path,
        help="JSON output path. Default: stdout summary only.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    return p.parse_args(argv)


def _default_manifests() -> tuple[str, str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    sibling = repo_root.parent / "betfair-predictors"
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

    # Heavy imports deferred so --help is fast.
    from data.episode_builder import load_day
    from data.predictor_features import build_predict_race_dataframe
    from env.bet_manager import BetManager, MIN_BET_STAKE
    from predictors import PredictorBundle

    manifests = args.predictor_bundle_manifests or _default_manifests()
    bundle = PredictorBundle.from_manifests(
        champion_manifest=manifests[0],
        ranker_manifest=manifests[1],
        direction_manifest=manifests[2],
    )
    logger.info(
        "bundle loaded: champion=%s ranker=%s direction=%s",
        bundle.champion_experiment_id,
        bundle.ranker_experiment_id,
        bundle.direction_experiment_id,
    )

    per_day_results: list[dict] = []
    all_bets: list[dict] = []

    for day_str in args.days:
        day = load_day(day_str, data_dir=args.data_dir)
        as_of = datetime.strptime(day_str, "%Y-%m-%d").date()
        logger.info("=== %s: %d races ===", day_str, len(day.races))

        # Fresh BetManager per day — flat-£10 strategy doesn't carry
        # P&L between days; matches the predictor's per-market backtest.
        # Generous starting budget so liability gates never bind.
        bm = BetManager(starting_budget=100_000.0, fill_mode=day.fill_mode)

        day_bets: list[dict] = []
        for race in day.races:
            try:
                df = build_predict_race_dataframe(race, as_of_date=as_of)
                outs = bundle.predict_race(df)
            except Exception as exc:
                logger.warning(
                    "race %s skip — predict_race failed: %s",
                    race.market_id, exc,
                )
                continue

            # Pick argmax(p_win) runner.
            if not outs.p_win:
                continue
            top_sid = max(outs.p_win, key=outs.p_win.get)
            top_p_win = outs.p_win[top_sid]
            segment_strong = outs.segment_strong_flag.get(top_sid, False)

            # Walk pre-off ticks; place bet at the configured fraction.
            pre_off_ticks = [t for t in race.ticks if not t.in_play]
            if not pre_off_ticks:
                continue
            bet_idx = max(0, int(len(pre_off_ticks) * args.bet_tick_frac) - 1)
            bet_tick = pre_off_ticks[bet_idx]

            # Find the runner snapshot at the bet tick.
            runner_snap = next(
                (r for r in bet_tick.runners if r.selection_id == top_sid),
                None,
            )
            if runner_snap is None:
                continue
            if not runner_snap.available_to_back:
                continue
            best_back_price = runner_snap.available_to_back[0].price
            if best_back_price <= 1.01:
                continue

            implied_p_win = 1.0 / best_back_price
            edge = top_p_win - implied_p_win

            decided_to_bet = edge > args.edge_threshold
            if args.require_segment_strong:
                decided_to_bet = decided_to_bet and segment_strong

            if not decided_to_bet:
                continue

            stake = max(args.stake, MIN_BET_STAKE)
            bet = bm.place_back(
                runner_snap, stake, market_id=race.market_id,
            )
            if bet is None:
                logger.debug(
                    "race %s sid=%d refused — matcher refused (price gate or liquidity)",
                    race.market_id, top_sid,
                )
                continue

            day_bets.append({
                "market_id": race.market_id,
                "selection_id": top_sid,
                "p_win": top_p_win,
                "implied_p_win": implied_p_win,
                "edge": edge,
                "matched_price": bet.average_price,
                "matched_stake": bet.matched_stake,
                "segment_strong": segment_strong,
            })

        # Settle every race for this day.
        day_pnl = 0.0
        for race in day.races:
            if race.winner_selection_id is not None or race.winning_selection_ids:
                winners = race.winning_selection_ids or {race.winner_selection_id}
                race_pnl = bm.settle_race(
                    winning_selection_ids=set(int(s) for s in winners if s is not None),
                    market_id=race.market_id,
                    commission=0.05,
                    each_way_divisor=race.each_way_divisor,
                    winner_selection_id=race.winner_selection_id,
                    number_of_places=race.number_of_each_way_places,
                )
                day_pnl += race_pnl

        # Augment day_bets with per-bet outcomes from settled bets.
        sid_to_bet_meta = {b["market_id"] + "_" + str(b["selection_id"]): b for b in day_bets}
        for bet in bm.bets:
            key = bet.market_id + "_" + str(bet.selection_id)
            if key in sid_to_bet_meta:
                sid_to_bet_meta[key]["outcome"] = bet.outcome.name
                sid_to_bet_meta[key]["pnl"] = float(bet.pnl)

        all_bets.extend(day_bets)
        n = len(day_bets)
        wins = sum(1 for b in day_bets if b.get("outcome") == "WON")
        total_stake = sum(b["matched_stake"] for b in day_bets)
        per_day_results.append({
            "day": day_str,
            "n_races": len(day.races),
            "n_bets": n,
            "n_wins": wins,
            "win_rate": wins / n if n > 0 else 0.0,
            "total_stake": total_stake,
            "day_pnl": day_pnl,
            "roi": day_pnl / total_stake if total_stake > 0 else 0.0,
        })
        logger.info(
            "%s: races=%d bets=%d wins=%d win_rate=%.1f%% stake=£%.0f pnl=£%+.2f roi=%+.1f%%",
            day_str, len(day.races), n, wins,
            (wins / n * 100) if n > 0 else 0.0,
            total_stake, day_pnl,
            (day_pnl / total_stake * 100) if total_stake > 0 else 0.0,
        )

    # Aggregate.
    total_n = sum(d["n_bets"] for d in per_day_results)
    total_wins = sum(d["n_wins"] for d in per_day_results)
    total_stake = sum(d["total_stake"] for d in per_day_results)
    total_pnl = sum(d["day_pnl"] for d in per_day_results)

    print()
    print("=" * 70)
    print("PREDICTOR DETERMINISTIC STRATEGY — RESULTS")
    print("=" * 70)
    print(f"Days: {', '.join(args.days)}")
    print(f"Edge threshold: {args.edge_threshold}  Stake: £{args.stake}")
    print(f"Require segment_strong: {args.require_segment_strong}")
    print()
    print(f"{'day':>12} {'races':>6} {'bets':>5} {'wins':>5} {'win%':>6} "
          f"{'stake':>8} {'pnl':>10} {'roi':>7}")
    for d in per_day_results:
        print(
            f"{d['day']:>12} {d['n_races']:>6} {d['n_bets']:>5} {d['n_wins']:>5} "
            f"{d['win_rate']*100:>5.1f}% £{d['total_stake']:>6.0f} "
            f"£{d['day_pnl']:>+8.2f} {d['roi']*100:>+5.1f}%"
        )
    print("-" * 70)
    print(
        f"{'TOTAL':>12} {sum(d['n_races'] for d in per_day_results):>6} "
        f"{total_n:>5} {total_wins:>5} "
        f"{(total_wins/total_n*100) if total_n>0 else 0:>5.1f}% "
        f"£{total_stake:>6.0f} £{total_pnl:>+8.2f} "
        f"{(total_pnl/total_stake*100) if total_stake>0 else 0:>+5.1f}%"
    )
    print()
    print("Predictor's reported test-set ROI (manifest):")
    print("  Calibrated champion: +18.6% on 114 markets (2026-05-04 to 06)")
    print("  Ranker:             +390% on same 114 markets")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump({
                "args": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                "bundle": {
                    "champion_id": bundle.champion_experiment_id,
                    "ranker_id": bundle.ranker_experiment_id,
                    "direction_id": bundle.direction_experiment_id,
                },
                "per_day": per_day_results,
                "total": {
                    "n_bets": total_n,
                    "n_wins": total_wins,
                    "total_stake": total_stake,
                    "total_pnl": total_pnl,
                    "roi": total_pnl / total_stake if total_stake > 0 else 0.0,
                },
                "bets": all_bets,
            }, f, indent=2, default=str)
        logger.info("results written to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
