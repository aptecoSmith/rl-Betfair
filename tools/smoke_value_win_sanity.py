"""Phase 1 sanity smoke for the non-scalping-directional-probe plan.

Builds a BetfairEnv with ``strategy_mode="value_win"`` and runs a
uniform-random rollout across a configurable number of races on one
day. Reports four diagnostics gated by
``plans/non-scalping-directional-probe/hard_constraints.md`` §10.1:

    env constructed without exception
    >= 1 bet matched in the rollout (action pathway reaches matcher)
    no bet carries ``force_close=True`` (value_win is hold-to-settle)
    day_pnl is finite (no NaN / Inf in reward stream)

ALL FOUR must pass. ANY failure -> exit code 1 and a written diagnostic.

This proves the existing ``value_win`` codepath is functional BEFORE
we invest in Phase 2 CLI plumbing.

Usage:

    python -m tools.smoke_value_win_sanity --day 2026-05-20 \\
        --policy-rollout-races 3 --device cuda
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

logger = logging.getLogger("smoke_value_win_sanity")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--day", default="2026-05-20", metavar="YYYY-MM-DD")
    p.add_argument("--data-dir", default="data/processed", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--predictor-bundle-manifests", nargs=3, default=None,
        metavar=("CHAMPION", "RANKER", "DIRECTION"),
    )
    p.add_argument(
        "--policy-rollout-races", type=int, default=3,
        help="Limit to N races (full day takes ~10-20 min).",
    )
    p.add_argument("--seed", type=int, default=42)
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

    from agents_v2.action_space import (
        ActionType, DiscreteActionSpace, compute_mask,
    )
    from data.episode_builder import load_day
    from env.betfair_env import BetfairEnv
    from predictors import PredictorBundle
    from training_v2.cohort.worker import scalping_train_config

    # ── Construct env in value_win mode.
    manifests = args.predictor_bundle_manifests or _default_manifests()
    bundle = PredictorBundle.from_manifests(
        champion_manifest=manifests[0],
        ranker_manifest=manifests[1],
        direction_manifest=manifests[2],
    )
    logger.info("loaded predictor bundle")

    day = load_day(args.day, data_dir=args.data_dir)
    logger.info("loaded day %s: %d races", args.day, len(day.races))

    cfg = scalping_train_config()
    # The cross-rule in env line 1282-1286 forces scalping_mode=False
    # when strategy_mode in {value_win, value_each_way}. We set it
    # here too for clarity / belt-and-braces.
    cfg["training"]["strategy_mode"] = "value_win"
    cfg["training"]["scalping_mode"] = False

    construct_ok = False
    env: BetfairEnv | None = None
    try:
        env = BetfairEnv(
            day, cfg,
            predictor_bundle=bundle,
            use_race_outcome_predictor=True,
            use_direction_predictor=True,
            predictor_lean_obs=True,
        )
        construct_ok = True
        logger.info("env constructed OK: strategy_mode=%s scalping_mode=%s",
                    getattr(env, "_strategy_mode", "?"),
                    getattr(env, "scalping_mode", "?"))
    except Exception as exc:  # noqa: BLE001
        logger.exception("env construction FAILED: %s", exc)

    matched_bets = 0
    force_closed_bets = 0
    day_pnl_finite = False
    races_completed = 0
    last_info: dict = {}

    if construct_ok and env is not None:
        space = DiscreteActionSpace(max_runners=env.max_runners)
        rng = np.random.default_rng(args.seed)
        env.reset()
        races_to_run = min(args.policy_rollout_races, len(env.day.races))
        safety_steps = 100_000
        try:
            while races_completed < races_to_run and safety_steps > 0:
                safety_steps -= 1
                mask = compute_mask(space, env)
                legal = np.where(mask)[0]
                if legal.size == 0:
                    action_idx = 0
                else:
                    action_idx = int(rng.choice(legal))
                kind, slot = space.decode(int(action_idx))

                action = np.zeros(env.action_space.shape[0], dtype=np.float32)
                if slot is not None and kind in (
                    ActionType.OPEN_BACK, ActionType.OPEN_LAY,
                ):
                    ap = env._actions_per_runner
                    base = slot * ap
                    action[base] = (
                        +1.0 if kind is ActionType.OPEN_BACK else -1.0
                    )

                obs, reward, terminated, truncated, info = env.step(action)
                last_info = info
                if terminated or truncated:
                    races_completed += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("rollout FAILED at step: %s", exc)

        all_bets = getattr(env, "_settled_bets", []) or []
        matched_bets = sum(
            1 for b in all_bets if getattr(b, "matched_stake", 0) > 0
        )
        force_closed_bets = sum(
            1 for b in all_bets if getattr(b, "force_close", False)
        )
        day_pnl = last_info.get("day_pnl") if last_info else None
        if day_pnl is not None and not (
            math.isnan(float(day_pnl)) or math.isinf(float(day_pnl))
        ):
            day_pnl_finite = True

    # ── Verdict table
    bets_ok = matched_bets >= 1
    no_force_close_ok = force_closed_bets == 0
    finite_pnl_ok = day_pnl_finite

    rows = [
        "",
        f"VALUE_WIN SANITY SMOKE — {args.day}",
        "=" * 72,
        "",
        "ROLLOUT:",
        f"  races completed ....................... "
        f"{races_completed} / {args.policy_rollout_races}",
        f"  matched bets .......................... {matched_bets}",
        f"  bets with force_close=True ............ {force_closed_bets}",
        f"  last-info day_pnl ..................... "
        f"{last_info.get('day_pnl') if last_info else 'n/a'}",
        "",
        "VERDICT vs hard_constraints §10.1:",
        f"  env constructed                      "
        f"{'PASS' if construct_ok else 'FAIL'}",
        f"  matched_bets >= 1                    "
        f"{'PASS' if bets_ok else 'FAIL'} (actual {matched_bets})",
        f"  force_close bets == 0                "
        f"{'PASS' if no_force_close_ok else 'FAIL'} "
        f"(actual {force_closed_bets})",
        f"  day_pnl finite                       "
        f"{'PASS' if finite_pnl_ok else 'FAIL'}",
    ]

    all_pass = (
        construct_ok and bets_ok and no_force_close_ok and finite_pnl_ok
    )
    rows.append("")
    rows.append(
        f"OVERALL: "
        f"{'PASS — proceed to Phase 2' if all_pass else 'FAIL — STOP loop'}"
    )
    rows.append("")
    out = "\n".join(rows)
    print(out)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
