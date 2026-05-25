"""Probe runner for the non-scalping-directional-probe plan.

Runs uniform-random rollouts with the env-side value-bet gate active.
Writes per-bet logs per (seed, day) per
``plans/non-scalping-directional-probe/hard_constraints.md`` §8.

Used for BOTH Phase 3 (1-day smoke) and Phases 4/5 (5 seeds × 3 days
real probe). Phase distinction is just the args.

Per-bet log schema:
    seed, day, bet_idx, market_id, selection_id, side,
    price_matched, stake_matched, liability,
    runner_champion_p_win, race_max_pwin, tick_time_to_off_s,
    value_edge_at_placement, final_outcome, final_pnl

Usage:

    # Phase 3 smoke (1 seed × 1 day, both sides):
    python -m tools.probe_directional --days 2026-05-20 --n-seeds 1 \\
        --side both --output-dir registry/probe_smoke

    # Phase 4 Probe A (back-only, 5 seeds × 3 days):
    python -m tools.probe_directional \\
        --days 2026-04-28 2026-04-29 2026-04-30 \\
        --n-seeds 5 --side back --back-stake 10 \\
        --edge-threshold 0.05 --output-dir registry/probe_A_back

    # Phase 5 Probe B (lay-only, 5 seeds × 3 days):
    python -m tools.probe_directional \\
        --days 2026-04-28 2026-04-29 2026-04-30 \\
        --n-seeds 5 --side lay --lay-liability 20 \\
        --lay-price-max 20 --edge-threshold 0.05 \\
        --output-dir registry/probe_B_lay
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("probe_directional")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--days", nargs="+", required=True, metavar="YYYY-MM-DD",
        help="One or more day dates to rollout on.",
    )
    p.add_argument("--data-dir", default="data/processed", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--predictor-bundle-manifests", nargs=3, default=None,
        metavar=("CHAMPION", "RANKER", "DIRECTION"),
    )
    p.add_argument(
        "--n-seeds", type=int, default=5,
        help="Number of distinct seeds (\"agents\") per day.",
    )
    p.add_argument(
        "--side", choices=["back", "lay", "both"], default="both",
        help="Filter the policy's action signals to one side.",
    )
    p.add_argument(
        "--edge-threshold", type=float, default=0.05,
        help="value_edge_threshold gate.",
    )
    p.add_argument(
        "--back-stake", type=float, default=None,
        help="Override BACK stake (£). None = use policy action dim.",
    )
    p.add_argument(
        "--lay-liability", type=float, default=None,
        help="Override LAY liability (£). None = use policy action dim.",
    )
    p.add_argument(
        "--lay-price-max", type=float, default=0.0,
        help="Composes with the gate. 0 = disabled. Probe B uses 20.",
    )
    p.add_argument(
        "--commission", type=float, default=0.05,
        help="Commission used for edge computation in the log (the env "
             "reads commission from cfg['reward']['commission'] = 0.05; "
             "this is just for the value_edge_at_placement log column).",
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output dir for per-(seed, day) JSONL files.",
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


def _build_env(*, day, bundle, args):
    """Build a BetfairEnv configured for the directional probe."""
    from env.betfair_env import BetfairEnv
    from training_v2.cohort.worker import scalping_train_config
    cfg = scalping_train_config()
    cfg["training"]["strategy_mode"] = "value_win"
    cfg["training"]["scalping_mode"] = False
    return BetfairEnv(
        day, cfg,
        predictor_bundle=bundle,
        use_race_outcome_predictor=True,
        use_direction_predictor=True,
        predictor_lean_obs=True,
        value_edge_threshold=args.edge_threshold,
        directional_back_stake=args.back_stake,
        directional_lay_liability=args.lay_liability,
        lay_price_max=args.lay_price_max,
        # No pwin gate here — the value-edge gate is the binding
        # constraint. Setting pwin_back=0 and pwin_lay=1 = disabled.
        predictor_p_win_back_threshold=0.0,
        predictor_p_win_lay_threshold=1.0,
    )


def _rollout_one_day(*, env, seed: int, args, day_label: str, out_path: Path):
    """Run uniform-random rollout, write per-bet log."""
    import numpy as np
    from agents_v2.action_space import (
        ActionType, DiscreteActionSpace, compute_mask,
    )
    from env.bet_manager import BetSide
    from env.scalping_math import value_bet_edge

    space = DiscreteActionSpace(max_runners=env.max_runners)
    rng = np.random.default_rng(seed)
    env.reset()

    side_filter = args.side  # "back" / "lay" / "both"

    safety = 500_000
    last_info: dict = {}
    while safety > 0:
        safety -= 1
        mask = compute_mask(space, env)
        legal = np.where(mask)[0]
        if side_filter != "both":
            want = (
                ActionType.OPEN_BACK if side_filter == "back"
                else ActionType.OPEN_LAY
            )
            filtered = [
                i for i in legal
                if space.decode(int(i))[0] is want
            ]
            if filtered:
                legal = np.array(filtered, dtype=np.int64)
            else:
                legal = np.array([0], dtype=np.int64)
        action_idx = int(rng.choice(legal)) if legal.size > 0 else 0
        kind, slot = space.decode(int(action_idx))
        action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        if slot is not None and kind in (
            ActionType.OPEN_BACK, ActionType.OPEN_LAY,
        ):
            ap = env._actions_per_runner
            action[slot * ap] = (
                +1.0 if kind is ActionType.OPEN_BACK else -1.0
            )
        _, _, terminated, truncated, info = env.step(action)
        last_info = info
        if terminated:
            if env._race_idx + 1 >= len(env.day.races):
                break

    # ── Build per-bet log from env._settled_bets at end of day.
    # bm.bets is last-race-only (per CLAUDE.md "info[realised_pnl] is
    # last-race-only"); _settled_bets accumulates across the whole day.
    settled = list(getattr(env, "_settled_bets", []) or [])
    # Build market_id → race_idx map so we can look up the race's
    # tick array (for time_to_off) and per-race champion p_win cache.
    market_to_race_idx = {
        race.market_id: i for i, race in enumerate(env.day.races)
    }

    matched_records: list[dict] = []
    for bet in settled:
        if bet.matched_stake <= 0:
            continue
        market_id = bet.market_id
        race_idx = market_to_race_idx.get(market_id)
        if race_idx is None:
            continue
        race = env.day.races[race_idx]
        race_pwins = env._race_p_win_by_race[race_idx]
        race_max_pwin = max(race_pwins.values()) if race_pwins else 0.0
        sid = bet.selection_id
        pwin = float(race_pwins.get(sid, 0.0))
        price = float(bet.average_price)
        side_str = "back" if bet.side == BetSide.BACK else "lay"
        edge = value_bet_edge(
            pwin, price, side_str, args.commission,
        )
        liability = (
            bet.matched_stake * (price - 1.0)
            if bet.side == BetSide.LAY
            else bet.matched_stake
        )
        tick_idx = getattr(bet, "tick_index", None)
        if tick_idx is not None and 0 <= tick_idx < len(race.ticks):
            ttoff = float(
                (race.market_start_time - race.ticks[tick_idx].timestamp)
                .total_seconds()
            )
        else:
            ttoff = float("nan")
        matched_records.append({
            "seed": seed,
            "day": day_label,
            "market_id": market_id,
            "selection_id": sid,
            "side": side_str,
            "price_matched": price,
            "stake_matched": float(bet.matched_stake),
            "liability": float(liability),
            "runner_champion_p_win": pwin,
            "race_max_pwin": float(race_max_pwin),
            "tick_time_to_off_s": ttoff,
            "value_edge_at_placement": float(edge),
            "final_pnl": float(bet.pnl),
            "final_outcome": "win" if bet.pnl > 0 else "lose",
        })

    # Write JSONL.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in matched_records:
            fh.write(json.dumps(rec) + "\n")

    n_placed = len(matched_records)
    n_wins = sum(1 for r in matched_records if r["final_outcome"] == "win")
    sum_pnl = sum(r["final_pnl"] for r in matched_records)
    mean_pnl = sum_pnl / n_placed if n_placed > 0 else 0.0
    refusals = int(last_info.get("value_gate_refusals", 0)) if last_info else 0
    force_close_count = sum(
        1 for b in settled if getattr(b, "force_close", False)
    )
    return {
        "seed": seed,
        "day": day_label,
        "n_bets": n_placed,
        "n_wins": n_wins,
        "sum_pnl": sum_pnl,
        "mean_pnl_per_bet": mean_pnl,
        "value_gate_refusals": refusals,
        "force_close_bets": force_close_count,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(message)s",
    )

    from data.episode_builder import load_day
    from predictors import PredictorBundle

    manifests = args.predictor_bundle_manifests or _default_manifests()
    bundle = PredictorBundle.from_manifests(
        champion_manifest=manifests[0],
        ranker_manifest=manifests[1],
        direction_manifest=manifests[2],
    )
    logger.info("bundle loaded")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    t0 = time.time()
    for day_str in args.days:
        day = load_day(day_str, data_dir=args.data_dir)
        logger.info("day %s: %d races", day_str, len(day.races))
        for seed in range(args.n_seeds):
            env = _build_env(day=day, bundle=bundle, args=args)
            out_path = args.output_dir / f"bets_{day_str}_seed{seed}.jsonl"
            summary = _rollout_one_day(
                env=env, seed=seed, args=args,
                day_label=day_str, out_path=out_path,
            )
            summaries.append(summary)
            logger.info(
                "  seed=%d bets=%d wins=%d pnl=£%+.2f mean=£%+.4f/bet "
                "refusals=%d force_close=%d",
                seed, summary["n_bets"], summary["n_wins"],
                summary["sum_pnl"], summary["mean_pnl_per_bet"],
                summary["value_gate_refusals"], summary["force_close_bets"],
            )

    # Write run summary.
    summary_path = args.output_dir / "_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump({
            "args": {k: str(v) if isinstance(v, Path) else v
                     for k, v in vars(args).items()},
            "elapsed_s": time.time() - t0,
            "per_day_seed": summaries,
        }, fh, indent=2)

    # Top-level pass/fail (Phase 3 gates only — Phases 4/5 do their
    # own analysis post-hoc).
    total_bets = sum(s["n_bets"] for s in summaries)
    total_refusals = sum(s["value_gate_refusals"] for s in summaries)
    total_force_close = sum(s["force_close_bets"] for s in summaries)
    print("\n" + "=" * 72)
    print(f"PROBE SUMMARY")
    print(f"  output dir ............ {args.output_dir}")
    print(f"  seeds × days .......... {args.n_seeds} × {len(args.days)}")
    print(f"  total bets matched .... {total_bets}")
    print(f"  total gate refusals ... {total_refusals}")
    print(f"  total force_close bets  {total_force_close}")
    print(f"  elapsed ............... {time.time() - t0:.1f}s")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
