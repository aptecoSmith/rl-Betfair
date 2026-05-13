"""Probe lay-side structural EV across the held-out eval window.

Given the pwin + race-confidence gate's lay rule:

    runner is lay-eligible iff
        race_max_pwin >= race_confidence_threshold (default 0.50) AND
        runner_pwin   <= predictor_p_win_lay_threshold (default 0.40)

For each such (race, runner) tuple across the eval days, this probe:

  1. Records the gate-eligible flag from the predictor.
  2. Looks up the actual winner_selection_id from the parquet to score
     win/loss.
  3. Approximates the lay price the agent would have hit by taking the
     LTP at a configurable distance from the off (default 30s before).
  4. Aggregates: lay-side win rate (lay wins iff the runner did NOT
     finish 1st), per-pound lay-side EV, and the distribution of lay
     prices.

Output table:

    DAY  N_eligible  lay_winrate  avg_lay_price  lay_EV_per_£stake
    ...                                          loss_distribution

The point is to tell us whether the gate-admitted lay set is
structurally +EV (the predictor's edge survives at the population
level) or -EV (the gate admits negative-EV bets and agents lose
money regardless of skill).

Usage:

    python -m tools.probe_lay_outcome_distribution \
        --days 2026-04-28 2026-04-29 2026-04-30 \
        --race-confidence-threshold 0.50 \
        --lay-threshold 0.40 \
        --secs-before-off 30 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("probe_lay_outcomes")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", nargs="+", required=True, metavar="YYYY-MM-DD")
    p.add_argument("--data-dir", default="data/processed", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--predictor-bundle-manifests", nargs=3, default=None,
        metavar=("CHAMPION", "RANKER", "DIRECTION"),
    )
    p.add_argument(
        "--race-confidence-threshold", type=float, default=0.50,
        help="Race is admitted iff max(p_win) >= this.",
    )
    p.add_argument(
        "--lay-threshold", type=float, default=0.40,
        help="Runner is lay-eligible iff p_win <= this.",
    )
    p.add_argument(
        "--secs-before-off", type=float, default=30.0,
        help="LTP proxy: use the tick this many seconds before the off.",
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


def _ltp_at_seconds_before_off(race, secs_before_off: float) -> dict[int, float]:
    """Return per-runner LTP at the tick closest to (off - secs_before_off).

    Computes time-to-off as (market_start_time - tick.timestamp).total_seconds().
    Positive = before the off, negative = in-play. Falls back to the LAST
    tick if no tick is found that early.
    """
    target = secs_before_off  # positive seconds before the off
    best_tick = None
    best_dist = float("inf")
    for tick in race.ticks:
        tto = (tick.market_start_time - tick.timestamp).total_seconds()
        dist = abs(tto - target)
        if dist < best_dist:
            best_dist = dist
            best_tick = tick
    if best_tick is None:
        best_tick = race.ticks[-1]
    return {
        r.selection_id: float(r.last_traded_price)
        for r in best_tick.runners
        if r.last_traded_price is not None and r.last_traded_price > 1.0
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(message)s",
    )

    import pandas as pd

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

    all_rows: list[dict] = []  # one row per gate-eligible (race, runner)

    for day_str in args.days:
        day = load_day(day_str, data_dir=args.data_dir)
        env = BetfairEnv(
            day, cfg,
            predictor_bundle=bundle,
            use_race_outcome_predictor=True,
            use_direction_predictor=True,
            predictor_lean_obs=True,
            race_confidence_threshold=0.0,  # don't gate; we filter here
        )

        # Winners lookup from the day's tick parquet.
        raw = pd.read_parquet(args.data_dir / f"{day_str}.parquet")
        winner_by_market = (
            raw.dropna(subset=["winner_selection_id"])
               .groupby("market_id")["winner_selection_id"]
               .first()
               .to_dict()
        )

        for race_idx, race in enumerate(day.races):
            p_wins = env._race_p_win_by_race[race_idx]
            if not p_wins:
                continue
            max_pwin = max(p_wins.values())
            if max_pwin < args.race_confidence_threshold:
                continue

            ltps = _ltp_at_seconds_before_off(race, args.secs_before_off)
            winner_sid = winner_by_market.get(race.market_id)
            if winner_sid is None:
                continue  # void / abandoned race

            for sid, pwin in p_wins.items():
                if pwin > args.lay_threshold:
                    continue
                lay_price = ltps.get(sid)
                if lay_price is None or lay_price <= 1.0:
                    continue
                runner_won = int(sid == int(winner_sid))
                all_rows.append({
                    "day": day_str,
                    "market_id": race.market_id,
                    "selection_id": sid,
                    "champion_p_win": float(pwin),
                    "race_max_pwin": float(max_pwin),
                    "lay_price_proxy": lay_price,
                    "winner_sid": int(winner_sid),
                    "runner_won": runner_won,
                })

        logger.info(
            "day %s: %d gate-eligible (race, runner) tuples so far",
            day_str, len(all_rows),
        )

    if not all_rows:
        print("No eligible rows found — check thresholds + days.")
        return 1

    df = pd.DataFrame(all_rows)
    df["lay_pnl_per_unit_stake"] = df.apply(
        lambda r: 1.0 if r["runner_won"] == 0 else -(r["lay_price_proxy"] - 1.0),
        axis=1,
    )

    print()
    print(f"PROBE: lay-side structural EV — pwin_lay_thr={args.lay_threshold}, "
          f"race_conf_thr={args.race_confidence_threshold}, "
          f"lay_price = LTP at off-{args.secs_before_off:.0f}s")
    print("=" * 78)
    print()
    print("OVERALL")
    n = len(df)
    win_runner = df["runner_won"].sum()
    lay_wins = n - win_runner
    lay_winrate = lay_wins / n
    avg_lay_price = df["lay_price_proxy"].mean()
    median_lay_price = df["lay_price_proxy"].median()
    ev_per_stake = df["lay_pnl_per_unit_stake"].mean()
    print(f"  gate-eligible (race, runner) tuples : {n}")
    print(f"  lay-side wins (runner did not win) .. {lay_wins} "
          f"({lay_winrate * 100:.1f}%)")
    print(f"  laid-runner wins (lay LOST) ......... {win_runner} "
          f"({(1 - lay_winrate) * 100:.1f}%)")
    print(f"  avg lay-price proxy ................. {avg_lay_price:.2f}")
    print(f"  median lay-price proxy .............. {median_lay_price:.2f}")
    print(f"  EV per £1 lay stake ................. £{ev_per_stake:+.4f}")
    print()
    print("PER-DAY")
    print(f"  {'day':>10}  {'n':>5}  {'lay_winrate':>11}  {'avg_lay_price':>13}  "
          f"{'EV/£stake':>10}")
    for day, sub in df.groupby("day"):
        sub_n = len(sub)
        sub_wins = sub_n - sub["runner_won"].sum()
        sub_wr = sub_wins / sub_n
        sub_lp = sub["lay_price_proxy"].mean()
        sub_ev = sub["lay_pnl_per_unit_stake"].mean()
        print(f"  {day:>10}  {sub_n:>5}  {sub_wr * 100:>10.1f}%  "
              f"{sub_lp:>13.2f}  £{sub_ev:>+8.4f}")
    print()
    print("LAY-PRICE BUCKETS (Betfair lay psychology — leverage on losses):")
    buckets = [
        ("price <= 2",   df[df["lay_price_proxy"] <= 2]),
        ("price 2-5",    df[(df["lay_price_proxy"] > 2) & (df["lay_price_proxy"] <= 5)]),
        ("price 5-10",   df[(df["lay_price_proxy"] > 5) & (df["lay_price_proxy"] <= 10)]),
        ("price 10-20",  df[(df["lay_price_proxy"] > 10) & (df["lay_price_proxy"] <= 20)]),
        ("price 20-50",  df[(df["lay_price_proxy"] > 20) & (df["lay_price_proxy"] <= 50)]),
        ("price > 50",   df[df["lay_price_proxy"] > 50]),
    ]
    print(f"  {'bucket':>14}  {'n':>5}  {'lay_winrate':>11}  {'EV/£stake':>10}  "
          f"{'avg_loss_when_lost':>20}")
    for name, sub in buckets:
        sub_n = len(sub)
        if sub_n == 0:
            continue
        sub_wins = sub_n - sub["runner_won"].sum()
        sub_wr = sub_wins / sub_n
        sub_ev = sub["lay_pnl_per_unit_stake"].mean()
        # Average leverage on losing-lay races
        losers = sub[sub["runner_won"] == 1]
        avg_loss = (
            -(losers["lay_price_proxy"] - 1.0).mean()
            if not losers.empty else float("nan")
        )
        print(f"  {name:>14}  {sub_n:>5}  {sub_wr * 100:>10.1f}%  "
              f"£{sub_ev:>+8.4f}  £{avg_loss:>+18.2f}")

    print()
    print("PREDICTOR-CALIBRATION CHECK (pwin band -> realised win rate):")
    print(f"  {'pwin band':>14}  {'n':>5}  {'realised_pwin':>13}  "
          f"{'avg_lay_price':>13}  {'EV/£stake':>10}")
    bands = [
        ("0.00-0.10", df[df["champion_p_win"] <= 0.10]),
        ("0.10-0.20", df[(df["champion_p_win"] > 0.10) & (df["champion_p_win"] <= 0.20)]),
        ("0.20-0.30", df[(df["champion_p_win"] > 0.20) & (df["champion_p_win"] <= 0.30)]),
        ("0.30-0.40", df[(df["champion_p_win"] > 0.30) & (df["champion_p_win"] <= 0.40)]),
    ]
    for name, sub in bands:
        sub_n = len(sub)
        if sub_n == 0:
            continue
        realised = sub["runner_won"].mean()
        sub_lp = sub["lay_price_proxy"].mean()
        sub_ev = sub["lay_pnl_per_unit_stake"].mean()
        print(f"  {name:>14}  {sub_n:>5}  {realised * 100:>12.1f}%  "
              f"{sub_lp:>13.2f}  £{sub_ev:>+8.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
