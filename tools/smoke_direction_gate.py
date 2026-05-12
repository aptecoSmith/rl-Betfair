"""Pre-flight smoke test for the direction-gate action mask.

Builds two BetfairEnvs on a single day — one with
`direction_gate_enabled=True` and one without — and reports three
diagnostic numbers gated by `plans/scalping-direction-gate/
hard_constraints.md` §3:

    drift_fire_rate                  ≥ 5%
    lay_legal_with_gate / no_gate    ≤ 60%
    bets_matched (full-day estimate) ≥ 50

ALL THREE must pass for the loop to commit 12h to a cohort. ANY
failure → exit code 1 and a written diagnostic for the loop log.

No PPO, no GA, no real training — uniform-random policy on legal
actions only. The smoke is about "does the gate refuse meaningfully
without starving the agent."

Usage:

    python -m tools.smoke_direction_gate --day 2026-05-04 --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("smoke_direction_gate")


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
    *, day, cfg, bundle, gate_enabled: bool,
    p_win_back: float, p_win_lay: float,
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
        direction_gate_enabled=gate_enabled,
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
        day=day, cfg=cfg, bundle=bundle, gate_enabled=True,
        p_win_back=args.predictor_p_win_back_threshold,
        p_win_lay=args.predictor_p_win_lay_threshold,
    )
    env_no = _build_env(
        day=day, cfg=cfg, bundle=bundle, gate_enabled=False,
        p_win_back=args.predictor_p_win_back_threshold,
        p_win_lay=args.predictor_p_win_lay_threshold,
    )

    space = DiscreteActionSpace(max_runners=env_with.max_runners)

    # ── Population stats: drift-fire rate across all (tick, sid) pairs.
    total_pairs = 0
    drift_fires = 0
    for race_dict in env_with._tick_drift_fires_by_race:
        for _, fired in race_dict.items():
            total_pairs += 1
            if fired:
                drift_fires += 1
    drift_fire_rate = drift_fires / max(total_pairs, 1)

    # ── Walk every (race, tick) and compute the mask under both
    # configs. Count post-mask legal action surface for OPEN_BACK and
    # OPEN_LAY. This is the "gate impact on legality" measurement —
    # independent of any policy.
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

    lay_ratio = legal_lay_with / max(legal_lay_no, 1)

    # ── Policy rollout: uniform-random over legal actions, with-gate
    # env, capped to N races. We measure matched bets only; refusal
    # detail comes from env counters. Then extrapolate matched bets
    # to a full day.
    env_with.reset()
    rng = np.random.default_rng(args.seed)
    attempts_back = attempts_lay = 0
    matched_bets = 0

    races_to_run = min(args.policy_rollout_races, len(env_with.day.races))
    obs, info = None, {}
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

        # Encode as continuous action vector (the env's action_space
        # is Box(max_runners * actions_per_runner)). Action is
        # interpreted by the env's existing step path; the legal mask
        # came from compute_mask so this is the same surface the GA
        # actually fires at training time. For the smoke we just need
        # an action that hits the right slot — use a strong positive
        # signal in the chosen slot's signal-dim and zero elsewhere.
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
            # Env auto-advances to next race in step(); only reset
            # at episode boundary.

    # Drain matched-bet counter from the env's all-settled list.
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
        f"DIRECTION-GATE SMOKE — {args.day}",
        "=" * 66,
        "",
        "POPULATION (regardless of policy):",
        f"  total (tick, runner) pairs ............ {total_pairs}",
        f"  drift fired ........................... {drift_fires} "
        f"({drift_fire_rate * 100:.2f}%)",
        "",
        "LEGAL ACTIONS (post-mask) by gate config:",
        "  baseline (pwin only):",
        f"    OPEN_BACK legal-tick-slot-count ..... {legal_back_no}",
        f"    OPEN_LAY  legal-tick-slot-count ..... {legal_lay_no}",
        "  with direction-gate:",
        f"    OPEN_BACK legal-tick-slot-count ..... {legal_back_with} "
        f"(delta: {legal_back_with - legal_back_no:+d})",
        f"    OPEN_LAY  legal-tick-slot-count ..... {legal_lay_with} "
        f"(delta: {legal_lay_with - legal_lay_no:+d})",
        f"    lay-legal ratio (with-gate / no-gate) {lay_ratio * 100:.2f}%",
        "",
        f"POLICY ROLLOUT (uniform-random over legal actions, "
        f"{races_actually_seen} race(s)):",
        f"  attempted opens BACK / LAY ............ "
        f"{attempts_back} / {attempts_lay}",
        f"  matched bets .......................... {matched_bets}",
        f"  → bets/race ........................... {bets_per_race:.2f}",
        f"  → full-day estimate ({len(env_with.day.races)} races) "
        f"... {full_day_bets_est}",
        "=" * 66,
    ]

    drift_ok = drift_fire_rate >= 0.05
    ratio_ok = lay_ratio <= 0.60
    bets_ok = full_day_bets_est >= 50

    rows.append("")
    rows.append("VERDICT vs hard_constraints §3:")
    rows.append(
        f"  drift_fire_rate ≥ 5%%        ........... "
        f"{'PASS' if drift_ok else 'FAIL'} "
        f"(actual {drift_fire_rate * 100:.2f}%%)"
    )
    rows.append(
        f"  lay_legal_with_gate / no_gate ≤ 60%%  ... "
        f"{'PASS' if ratio_ok else 'FAIL'} "
        f"(actual {lay_ratio * 100:.2f}%%)"
    )
    rows.append(
        f"  bets_matched (full-day est.) ≥ 50 ..... "
        f"{'PASS' if bets_ok else 'FAIL'} "
        f"(estimate {full_day_bets_est})"
    )

    all_pass = drift_ok and ratio_ok and bets_ok
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
