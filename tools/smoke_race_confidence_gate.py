"""Pre-flight smoke test for the race-confidence action-mask gate.

Builds two BetfairEnvs on a single day — one with
`race_confidence_threshold=0.30` and one without — and reports
three diagnostic numbers gated by `plans/scalping-race-confidence-
gate/hard_constraints.md` §3:

    race_qualification_rate              ≥ 30%
    legal_with_race_gate / no_race_gate  ≤ 80%
    bets_matched (full-day estimate)     ≥ 50

ALL THREE must pass for the loop to commit 12h to a cohort. ANY
failure → exit code 1 and a written diagnostic for the loop log.

No PPO, no GA, no real training — uniform-random policy on legal
actions only. The smoke is about "does the gate refuse meaningfully
without starving the agent."

Usage:

    python -m tools.smoke_race_confidence_gate --day 2026-05-04 --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("smoke_race_confidence_gate")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--day", default="2026-05-04", metavar="YYYY-MM-DD")
    p.add_argument("--data-dir", default="data/processed", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--predictor-bundle-manifests", nargs=3, default=None,
        metavar=("CHAMPION", "RANKER", "DIRECTION"),
    )
    p.add_argument(
        "--predictor-p-win-back-threshold", type=float, default=0.20,
    )
    p.add_argument(
        "--predictor-p-win-lay-threshold", type=float, default=0.40,
    )
    p.add_argument(
        "--race-confidence-threshold", type=float, default=0.30,
        help="Per-race max(p_win) threshold (locked default 0.30).",
    )
    p.add_argument(
        "--policy-rollout-races", type=int, default=3,
        help=(
            "How many full races to run the uniform-random policy on. "
            "Used to extrapolate matched-bets/day for the verdict."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for the uniform-random policy rollout.",
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


def _build_env(
    *, day, cfg, bundle,
    p_win_back: float, p_win_lay: float,
    race_confidence_threshold: float,
):
    from env.betfair_env import BetfairEnv
    return BetfairEnv(
        day, cfg,
        predictor_bundle=bundle,
        use_race_outcome_predictor=True,
        use_direction_predictor=True,
        predictor_lean_obs=True,
        predictor_p_win_back_threshold=p_win_back,
        predictor_p_win_lay_threshold=p_win_lay,
        race_confidence_threshold=race_confidence_threshold,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(message)s",
    )

    import numpy as np

    from agents_v2.action_space import ActionType, DiscreteActionSpace, compute_mask
    from data.episode_builder import load_day
    from predictors import PredictorBundle
    from training_v2.cohort.worker import scalping_train_config

    manifests = args.predictor_bundle_manifests or _default_manifests()
    bundle = PredictorBundle.from_manifests(
        champion_manifest=manifests[0],
        ranker_manifest=manifests[1],
        direction_manifest=manifests[2],
    )
    logger.info(
        "bundle: champion=%s ranker=%s direction=%s",
        bundle.champion_experiment_id,
        bundle.ranker_experiment_id,
        bundle.direction_experiment_id,
    )

    day = load_day(args.day, data_dir=args.data_dir)
    logger.info("loaded day %s: %d races", args.day, len(day.races))

    cfg = scalping_train_config()
    cfg["training"]["strategy_mode"] = "arb"

    env_with = _build_env(
        day=day, cfg=cfg, bundle=bundle,
        p_win_back=args.predictor_p_win_back_threshold,
        p_win_lay=args.predictor_p_win_lay_threshold,
        race_confidence_threshold=args.race_confidence_threshold,
    )
    env_no = _build_env(
        day=day, cfg=cfg, bundle=bundle,
        p_win_back=args.predictor_p_win_back_threshold,
        p_win_lay=args.predictor_p_win_lay_threshold,
        race_confidence_threshold=0.0,
    )

    space = DiscreteActionSpace(max_runners=env_with.max_runners)

    # ── Population stats: race qualification rate (per-race scalar).
    total_races = len(env_with.day.races)
    confident_flags = list(env_with._race_is_confident_by_race)
    confident_races = sum(1 for b in confident_flags if b)
    race_qualification_rate = confident_races / max(total_races, 1)

    # ── Walk every (race, tick) and compute the mask under both configs.
    # Count post-mask legal action surface for OPEN_BACK and OPEN_LAY.
    legal_back_with = legal_lay_with = 0
    legal_back_no = legal_lay_no = 0

    for env, counters in (
        (env_no, "no"),
        (env_with, "with"),
    ):
        env.reset()
        n_races_iter = len(env.day.races)
        for race_idx in range(n_races_iter):
            env._race_idx = race_idx
            n_ticks = len(env._static_obs[race_idx])
            for tick_idx in range(n_ticks):
                env._tick_idx = tick_idx
                mask = compute_mask(space, env)
                back_legal = sum(
                    1 for s in range(env.max_runners)
                    if mask[space.encode(ActionType.OPEN_BACK, s)]
                )
                lay_legal = sum(
                    1 for s in range(env.max_runners)
                    if mask[space.encode(ActionType.OPEN_LAY, s)]
                )
                if counters == "with":
                    legal_back_with += back_legal
                    legal_lay_with += lay_legal
                else:
                    legal_back_no += back_legal
                    legal_lay_no += lay_legal

    legal_total_with = legal_back_with + legal_lay_with
    legal_total_no = legal_back_no + legal_lay_no
    legal_ratio = legal_total_with / max(legal_total_no, 1)

    # ── Policy rollout: uniform-random over legal actions, with-gate
    # env, capped to N races. Measure matched bets.
    env_with.reset()
    rng = np.random.default_rng(args.seed)
    attempts_back = attempts_lay = 0

    races_to_run = min(args.policy_rollout_races, len(env_with.day.races))
    obs, info = env_with.reset(), {}
    races_seen = 0
    safety_steps = 50_000
    while races_seen < races_to_run and safety_steps > 0:
        safety_steps -= 1
        mask = compute_mask(space, env_with)
        legal = np.where(mask)[0]
        if legal.size == 0:
            action_idx = 0  # NOOP
        else:
            action_idx = int(rng.choice(legal))
        kind, slot = space.decode(int(action_idx))
        if kind is ActionType.OPEN_BACK:
            attempts_back += 1
        elif kind is ActionType.OPEN_LAY:
            attempts_lay += 1

        action = np.zeros(env_with.action_space.shape[0], dtype=np.float32)
        if slot is not None:
            ap = env_with._actions_per_runner
            base = slot * ap
            action[base] = +1.0 if kind is ActionType.OPEN_BACK else -1.0

        obs, reward, terminated, truncated, info = env_with.step(action)
        if terminated or truncated:
            races_seen += 1
            if races_seen >= races_to_run:
                break

    all_bets = getattr(env_with, "_settled_bets", []) or []
    matched_bets = sum(
        1 for b in all_bets if getattr(b, "matched_stake", 0) > 0
    )
    races_actually_seen = max(races_seen, 1)
    bets_per_race = matched_bets / races_actually_seen
    full_day_bets_est = int(bets_per_race * len(env_with.day.races))

    # ── Diagnostic table
    rows = [
        "",
        f"RACE-CONFIDENCE-GATE SMOKE — {args.day}",
        "=" * 66,
        "",
        "POPULATION (regardless of policy):",
        f"  total races ........................... {total_races}",
        f"  races confident (max p_win >= "
        f"{args.race_confidence_threshold:.2f}) "
        f".. {confident_races} ({race_qualification_rate * 100:.2f}%)",
        f"  races skipped ......................... "
        f"{total_races - confident_races}",
        "",
        "LEGAL ACTIONS (post-mask) by gate config:",
        "  baseline (pwin only):",
        f"    OPEN_BACK legal-slot-tick count ..... {legal_back_no}",
        f"    OPEN_LAY  legal-slot-tick count ..... {legal_lay_no}",
        "  with race-confidence gate:",
        f"    OPEN_BACK legal-slot-tick count ..... {legal_back_with} "
        f"(delta: {legal_back_with - legal_back_no:+d})",
        f"    OPEN_LAY  legal-slot-tick count ..... {legal_lay_with} "
        f"(delta: {legal_lay_with - legal_lay_no:+d})",
        f"    legal-tick ratio (with/no race gate)  {legal_ratio * 100:.2f}%",
        "",
        f"POLICY ROLLOUT (uniform-random over legal, "
        f"{races_actually_seen} race(s)):",
        f"  attempted opens BACK / LAY ............ "
        f"{attempts_back} / {attempts_lay}",
        f"  matched bets .......................... {matched_bets}",
        f"  → bets/race ........................... {bets_per_race:.2f}",
        f"  → full-day estimate ({len(env_with.day.races)} races) "
        f"... {full_day_bets_est}",
        "=" * 66,
    ]

    qual_ok = race_qualification_rate >= 0.30
    ratio_ok = legal_ratio <= 0.80
    bets_ok = full_day_bets_est >= 50

    rows.append("")
    rows.append("VERDICT vs hard_constraints §3:")
    rows.append(
        f"  race_qualification_rate >= 30%        "
        f"{'PASS' if qual_ok else 'FAIL'} "
        f"(actual {race_qualification_rate * 100:.2f}%)"
    )
    rows.append(
        f"  legal_ratio <= 80% (material work)    "
        f"{'PASS' if ratio_ok else 'FAIL'} "
        f"(actual {legal_ratio * 100:.2f}%)"
    )
    rows.append(
        f"  bets_matched >= 50 (full day est.)    "
        f"{'PASS' if bets_ok else 'FAIL'} "
        f"(estimate {full_day_bets_est})"
    )

    all_pass = qual_ok and ratio_ok and bets_ok
    rows.append("")
    rows.append(
        f"OVERALL: {'PASS — proceed to Session 03' if all_pass else 'FAIL — STOP loop'}"
    )
    rows.append("")

    out = "\n".join(rows)
    print(out)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
