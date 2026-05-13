"""Pre-flight smoke test for the lay-quality gate.

Builds two BetfairEnvs on a single day — one with the FULL gate
config (race_confidence + pwin_lay + lay_price_max) and one with
ONLY the race-confidence gate as a baseline — and reports four
diagnostic numbers gated by
``plans/scalping-lay-quality-gate/hard_constraints.md`` §3:

    race_qualification_rate               >= 30%
    legal_with_lay_gate / race_gate_only  <= 80%
    expected_per_pound_lay_EV (admitted)  >= -GBP 0.05
    bets_matched (full-day estimate)      >= 50

ALL FOUR must pass for the loop to commit 12h to a cohort. ANY
failure -> exit code 1 and a written diagnostic for the loop log.

The new EV threshold is the load-bearing one: it measures
whether the gate-admitted lay set is structurally +EV (or near-
zero) using the same arithmetic as
``tools/probe_lay_outcome_distribution.py``, applied to the
specific (race, runner) tuples the cohort would trade on this
smoke day under the new caps.

Usage:

    python -m tools.smoke_lay_quality_gate --day 2026-05-04 \\
        --predictor-p-win-lay-threshold 0.20 \\
        --lay-price-max 20 --race-confidence-threshold 0.50 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("smoke_lay_quality_gate")


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
        "--predictor-p-win-lay-threshold", type=float, default=0.20,
        help="Tightened from 0.40 per Phase 1 probe (calibration hole).",
    )
    p.add_argument(
        "--race-confidence-threshold", type=float, default=0.50,
        help="Inherited from race-confidence-gate (locked).",
    )
    p.add_argument(
        "--lay-price-max", type=float, default=20.0,
        help="New cap per Phase 1 probe (leverage trap).",
    )
    p.add_argument(
        "--secs-before-off", type=float, default=30.0,
        help=(
            "Lay-price proxy: use the LTP this many seconds before "
            "the off as the price at which the agent would lay."
        ),
    )
    p.add_argument(
        "--policy-rollout-races", type=int, default=3,
        help="How many races for the uniform-random matched-bets probe.",
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


def _build_env(
    *, day, cfg, bundle,
    p_win_back: float, p_win_lay: float,
    race_confidence_threshold: float,
    lay_price_max: float,
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
        lay_price_max=lay_price_max,
    )


def _ltp_at_secs_before_off(race, secs: float) -> float:
    """Return the LTP at the tick whose time_to_off is closest to
    ``secs`` from the off. Returns 0.0 if no priceable tick exists."""
    target = race.market_start_time
    best_tick = None
    best_dt = None
    for tick in race.ticks:
        ttoff = (target - tick.timestamp).total_seconds()
        if ttoff < 0:
            continue
        dt = abs(ttoff - secs)
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best_tick = tick
    if best_tick is None:
        return 0.0
    return best_tick


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

    # Full gate: race-confidence + pwin lay (tightened) + lay-price cap.
    env_full = _build_env(
        day=day, cfg=cfg, bundle=bundle,
        p_win_back=args.predictor_p_win_back_threshold,
        p_win_lay=args.predictor_p_win_lay_threshold,
        race_confidence_threshold=args.race_confidence_threshold,
        lay_price_max=args.lay_price_max,
    )
    # Baseline: race-confidence-only (predecessor cohort's gate).
    env_race_only = _build_env(
        day=day, cfg=cfg, bundle=bundle,
        p_win_back=args.predictor_p_win_back_threshold,
        # Loosen pwin lay to predecessor default so the diff isolates
        # the new caps' marginal restriction.
        p_win_lay=0.40,
        race_confidence_threshold=args.race_confidence_threshold,
        lay_price_max=0.0,
    )

    space = DiscreteActionSpace(max_runners=env_full.max_runners)

    # ── Race qualification (race-confidence gate's pass rate).
    total_races = len(env_full.day.races)
    confident_flags = list(env_full._race_is_confident_by_race)
    confident_races = sum(1 for b in confident_flags if b)
    race_qualification_rate = confident_races / max(total_races, 1)

    # ── Walk every (race, tick) under both gate configs. Count post-
    # mask legal OPEN_BACK / OPEN_LAY slot-ticks.
    legal_back_full = legal_lay_full = 0
    legal_back_race = legal_lay_race = 0
    # ── EV-on-admitted-set: collect (race_idx, sid, lay_price_proxy)
    # tuples once for the FULL gate config. Lay-price proxy is the LTP
    # at off-N seconds (probe convention). Score lay-win iff sid !=
    # winner_selection_id; EV per pound = winrate - (1-winrate)*(P-1).
    admitted_tuples: list[tuple[int, int, float]] = []
    seen_admitted: set[tuple[int, int]] = set()

    for env, tag in ((env_race_only, "race"), (env_full, "full")):
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
                if tag == "full":
                    legal_back_full += back_legal
                    legal_lay_full += lay_legal
                    # Record (race, sid) admitted by the FULL gate at
                    # ANY tick. The EV proxy is one shot per (race,
                    # runner) using the off-N LTP, so we dedupe.
                    if lay_legal > 0:
                        slot_map = env._slot_maps[race_idx]
                        for s in range(env.max_runners):
                            if not mask[space.encode(ActionType.OPEN_LAY, s)]:
                                continue
                            sid = slot_map.get(s)
                            if sid is None:
                                continue
                            key = (race_idx, sid)
                            if key in seen_admitted:
                                continue
                            seen_admitted.add(key)
                else:
                    legal_back_race += back_legal
                    legal_lay_race += lay_legal

    # ── Compute EV per pound on admitted set using off-N LTP.
    races = env_full.day.races
    n_admitted = 0
    n_wins = 0  # lay wins (sid != winner)
    sum_loss_amount = 0.0  # sum of -(P-1) on lay losses
    lay_prices: list[float] = []
    for race_idx, sid in seen_admitted:
        race = races[race_idx]
        winner = race.winner_selection_id
        # find off-N tick
        target = race.market_start_time
        best_tick = None
        best_dt = None
        for tick in race.ticks:
            ttoff = (target - tick.timestamp).total_seconds()
            if ttoff < 0:
                continue
            dt = abs(ttoff - args.secs_before_off)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_tick = tick
        if best_tick is None:
            continue
        lay_price = 0.0
        for r in best_tick.runners:
            if r.selection_id == sid:
                lay_price = float(r.last_traded_price or 0.0)
                break
        if lay_price <= 1.0:
            continue
        n_admitted += 1
        lay_prices.append(lay_price)
        if winner is None or sid != winner:
            n_wins += 1  # lay won (£1 gain)
        else:
            sum_loss_amount += (lay_price - 1.0)  # lay lost
    if n_admitted > 0:
        winrate = n_wins / n_admitted
        ev_per_pound = (
            winrate * 1.0
            - (sum_loss_amount / n_admitted)
        )
        avg_lay_price = float(np.mean(lay_prices)) if lay_prices else 0.0
    else:
        winrate = 0.0
        ev_per_pound = 0.0
        avg_lay_price = 0.0

    # 2026-05-14 — second pass: apply the ``lay_price_max`` cap to the
    # off-30s LTP AT EV-COMPUTATION TIME so the admitted set is
    # internally consistent (admit-tick LTP ≤ cap AND off-30s LTP ≤
    # cap). Without this, drifter runners (LTP ≤ cap at some tick,
    # > cap at off-30s) inflate the admitted set and contaminate the
    # EV with high-price losses that the gate would never actually
    # take. The probe tool applies the cap at probe-time on the same
    # LTP it uses for EV; this matches that methodology.
    n_admitted_consistent = 0
    n_wins_consistent = 0
    sum_loss_consistent = 0.0
    lay_prices_consistent: list[float] = []
    for race_idx, sid in seen_admitted:
        race = races[race_idx]
        winner = race.winner_selection_id
        target = race.market_start_time
        best_tick = None
        best_dt = None
        for tick in race.ticks:
            ttoff = (target - tick.timestamp).total_seconds()
            if ttoff < 0:
                continue
            dt = abs(ttoff - args.secs_before_off)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_tick = tick
        if best_tick is None:
            continue
        lay_price = 0.0
        for r in best_tick.runners:
            if r.selection_id == sid:
                lay_price = float(r.last_traded_price or 0.0)
                break
        if lay_price <= 1.0:
            continue
        if (
            args.lay_price_max > 0.0
            and lay_price > args.lay_price_max
        ):
            continue
        n_admitted_consistent += 1
        lay_prices_consistent.append(lay_price)
        if winner is None or sid != winner:
            n_wins_consistent += 1
        else:
            sum_loss_consistent += (lay_price - 1.0)
    if n_admitted_consistent > 0:
        winrate_c = n_wins_consistent / n_admitted_consistent
        ev_consistent = (
            winrate_c * 1.0
            - (sum_loss_consistent / n_admitted_consistent)
        )
        avg_lay_price_c = float(np.mean(lay_prices_consistent))
    else:
        winrate_c = 0.0
        ev_consistent = 0.0
        avg_lay_price_c = 0.0

    legal_total_full = legal_back_full + legal_lay_full
    legal_total_race = legal_back_race + legal_lay_race
    legal_ratio = legal_total_full / max(legal_total_race, 1)

    # ── Policy rollout: matched-bets sanity check (same shape as
    # smoke_race_confidence_gate.py).
    env_full.reset()
    rng = np.random.default_rng(args.seed)
    attempts_back = attempts_lay = 0
    races_to_run = min(args.policy_rollout_races, len(env_full.day.races))
    obs, info = env_full.reset(), {}
    races_seen = 0
    safety_steps = 50_000
    while races_seen < races_to_run and safety_steps > 0:
        safety_steps -= 1
        mask = compute_mask(space, env_full)
        legal = np.where(mask)[0]
        if legal.size == 0:
            action_idx = 0
        else:
            action_idx = int(rng.choice(legal))
        kind, slot = space.decode(int(action_idx))
        if kind is ActionType.OPEN_BACK:
            attempts_back += 1
        elif kind is ActionType.OPEN_LAY:
            attempts_lay += 1

        action = np.zeros(env_full.action_space.shape[0], dtype=np.float32)
        if slot is not None:
            ap = env_full._actions_per_runner
            base = slot * ap
            action[base] = (
                +1.0 if kind is ActionType.OPEN_BACK else -1.0
            )

        obs, reward, terminated, truncated, info = env_full.step(action)
        if terminated or truncated:
            races_seen += 1
            if races_seen >= races_to_run:
                break

    all_bets = getattr(env_full, "_settled_bets", []) or []
    matched_bets = sum(
        1 for b in all_bets if getattr(b, "matched_stake", 0) > 0
    )
    races_actually_seen = max(races_seen, 1)
    bets_per_race = matched_bets / races_actually_seen
    full_day_bets_est = int(bets_per_race * len(env_full.day.races))

    # ── Diagnostic table
    rows = [
        "",
        f"LAY-QUALITY-GATE SMOKE — {args.day}",
        "=" * 72,
        "",
        "CONFIG:",
        f"  race_confidence_threshold ............. {args.race_confidence_threshold}",
        f"  predictor_p_win_lay_threshold ......... {args.predictor_p_win_lay_threshold}",
        f"  lay_price_max ......................... {args.lay_price_max}",
        f"  secs_before_off (LTP proxy) ........... {args.secs_before_off}",
        "",
        "POPULATION (regardless of policy):",
        f"  total races ........................... {total_races}",
        f"  races confident (max p_win >= "
        f"{args.race_confidence_threshold:.2f}) "
        f".. {confident_races} ({race_qualification_rate * 100:.2f}%)",
        "",
        "LEGAL ACTIONS (post-mask) by gate config:",
        "  race-confidence only (baseline):",
        f"    OPEN_BACK legal-slot-tick count ..... {legal_back_race}",
        f"    OPEN_LAY  legal-slot-tick count ..... {legal_lay_race}",
        "  full lay-quality gate:",
        f"    OPEN_BACK legal-slot-tick count ..... {legal_back_full} "
        f"(delta: {legal_back_full - legal_back_race:+d})",
        f"    OPEN_LAY  legal-slot-tick count ..... {legal_lay_full} "
        f"(delta: {legal_lay_full - legal_lay_race:+d})",
        f"    legal-tick ratio (full/race-only) ... {legal_ratio * 100:.2f}%",
        "",
        f"EV ON ADMITTED LAY SET (LTP at off-{args.secs_before_off:.0f}s):",
        f"  raw admitted (mask at any tick) ....... {n_admitted}",
        f"  raw lay win rate ...................... {winrate * 100:.2f}%",
        f"  raw avg lay price ..................... {avg_lay_price:.2f}",
        f"  raw EV per GBP 1 stake ................ "
        f"GBP {ev_per_pound:+.4f} "
        f"(contaminated by drifters; see below)",
        f"  CONSISTENT admitted (off-30s LTP <= cap){n_admitted_consistent}",
        f"  CONSISTENT lay win rate ............... "
        f"{winrate_c * 100:.2f}%",
        f"  CONSISTENT avg lay price .............. {avg_lay_price_c:.2f}",
        f"  CONSISTENT EV per GBP 1 stake ......... "
        f"GBP {ev_consistent:+.4f}",
        "",
        f"POLICY ROLLOUT (uniform-random over legal, "
        f"{races_actually_seen} race(s)):",
        f"  attempted opens BACK / LAY ............ "
        f"{attempts_back} / {attempts_lay}",
        f"  matched bets .......................... {matched_bets}",
        f"  -> bets/race .......................... {bets_per_race:.2f}",
        f"  -> full-day estimate ({len(env_full.day.races)} races) "
        f"... {full_day_bets_est}",
        "=" * 72,
    ]

    qual_ok = race_qualification_rate >= 0.30
    ratio_ok = legal_ratio <= 0.80
    # Use the CONSISTENT EV (cap applied at both admission and pricing).
    # The raw figure is reported for diagnostic comparison but doesn't
    # represent what the gate actually admits.
    ev_ok = ev_consistent >= -0.05
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
        f"  expected_per_GBP_lay_EV >= -GBP 0.05   "
        f"{'PASS' if ev_ok else 'FAIL'} "
        f"(actual GBP {ev_consistent:+.4f}, "
        f"n={n_admitted_consistent})"
    )
    rows.append(
        f"  bets_matched >= 50 (full day est.)    "
        f"{'PASS' if bets_ok else 'FAIL'} "
        f"(estimate {full_day_bets_est})"
    )

    all_pass = qual_ok and ratio_ok and ev_ok and bets_ok
    rows.append("")
    rows.append(
        f"OVERALL: "
        f"{'PASS - proceed to Phase 5' if all_pass else 'FAIL - STOP loop'}"
    )
    rows.append("")

    out = "\n".join(rows)
    print(out)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
