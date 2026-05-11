"""Audit which agents are using the champion predictor's p_win.

Two diagnostics on one or more agents from a cohort:

  Test (1) "A/B sensitivity":
    Run the agent twice deterministically:
      - champion_obs ON  (--use-race-outcome-predictor at eval)
      - champion_obs OFF (champion + ranker obs cols zeroed)
    Identical pnl  -> policy puts zero weight on those columns.
    Different pnl  -> policy IS conditioning on champion obs.

  Test (2) "Strategic use":
    Run the agent once deterministically (champion ON). For every
    matched bet, look up the runner's champion p_win at race time.
    Aggregate:
      - mean p_win of runners BACKED  vs population mean p_win
      - mean p_win of runners LAYED   vs population mean p_win
    BACK p_win > population  AND  LAY p_win < population
        => agent is using p_win the right way (back winners, lay losers)
    Random vs population => agent is ignoring p_win.

Usage:
    python -m tools.audit_agent_predictor_use \\
        --cohort-dir registry/_predictor_SCALPING_safety_1778518690 \\
        --agent-ids acf9084a-... 56acc8e8-... 004e8534-... \\
        --eval-days 2026-05-04 2026-05-05 2026-05-06 \\
        --data-dir data/processed --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("audit_agent_predictor_use")


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort-dir", required=True, type=Path)
    p.add_argument("--agent-ids", nargs="+", required=True)
    p.add_argument("--eval-days", required=True, nargs="+")
    p.add_argument("--data-dir", default="data/processed", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--predictor-bundle-manifests", nargs=3, default=None)
    p.add_argument("--predictor-lean-obs", action="store_true", default=True)
    p.add_argument("--strategy-mode", default="arb")
    return p.parse_args(argv)


def _default_manifests():
    root = Path(__file__).resolve().parents[1]
    sibling = root.parent / "betfair-predictors"
    return (
        str(sibling / "production" / "race-outcome" / "manifest.json"),
        str(sibling / "production" / "race-outcome-ranker" / "manifest.json"),
        str(sibling / "production" / "direction-predictor" / "manifest.json"),
    )


def _load_agent_row(cohort_dir: Path, agent_id_prefix: str) -> dict:
    """Find scoreboard row for the agent."""
    with (cohort_dir / "scoreboard.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r["agent_id"].startswith(agent_id_prefix):
                return r
    raise SystemExit(f"agent {agent_id_prefix!r} not found")


def _build_env(day_str, args, cfg, bundle, use_race_outcome):
    from training_v2.cohort.worker import _build_env_for_day
    from agents_v2.env_shim import DEFAULT_SCORER_DIR
    return _build_env_for_day(
        day_str=day_str, data_dir=args.data_dir, cfg=cfg,
        scorer_dir=DEFAULT_SCORER_DIR,
        reward_overrides=None, scalping_overrides=None,
        predictor_bundle=bundle,
        use_race_outcome_predictor=use_race_outcome,
        use_direction_predictor=True,  # keep direction obs constant
        predictor_lean_obs=True,
    )


def _run_rollout(agent_row, args, cfg, bundle, use_race_outcome, day_str):
    """Run one deterministic rollout. Returns (env, batch, info)."""
    import torch
    from agents_v2.discrete_policy import DiscreteLSTMPolicy
    from training_v2.discrete_ppo.rollout import RolloutCollector

    env, shim = _build_env(day_str, args, cfg, bundle, use_race_outcome)
    obs_dim = env.observation_space.shape[0]
    hp = agent_row["hyperparameters"]
    hidden = int(hp.get("hidden_size", 128))

    policy = DiscreteLSTMPolicy(
        obs_dim=obs_dim, action_space=shim.action_space, hidden_size=hidden,
    )
    weights_path = agent_row.get("weights_path")
    if not weights_path:
        weights_path = str(args.cohort_dir / "weights" / f"{agent_row['agent_id']}.pt")
    sd = torch.load(weights_path, weights_only=False, map_location="cpu")
    policy.load_state_dict(sd["weights"])
    policy.to(args.device)
    policy.eval()

    collector = RolloutCollector(shim=shim, policy=policy, device=str(args.device))
    batch = collector.collect_episode(deterministic=True)
    return env, batch, collector.last_info


def _compute_pwin_for_race(bundle, race, as_of_date):
    """Run champion's predict_race for one race. Returns {sid: p_win}."""
    from data.predictor_features import build_predict_race_dataframe
    try:
        df = build_predict_race_dataframe(race, as_of_date=as_of_date)
        outputs = bundle.predict_race(df)
        return dict(outputs.p_win)
    except Exception as exc:
        logger.warning("predict_race failed for %s: %s", race.market_id, exc)
        return {}


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from env.bet_manager import BetSide
    from predictors import PredictorBundle
    from training_v2.cohort.worker import scalping_train_config

    manifests = args.predictor_bundle_manifests or _default_manifests()
    bundle = PredictorBundle.from_manifests(
        champion_manifest=manifests[0],
        ranker_manifest=manifests[1],
        direction_manifest=manifests[2],
    )
    logger.info("bundle direction=%s champion=%s",
                bundle.direction_experiment_id, bundle.champion_experiment_id)

    cfg = scalping_train_config()
    cfg["training"]["strategy_mode"] = args.strategy_mode

    for agent_id_prefix in args.agent_ids:
        agent_row = _load_agent_row(args.cohort_dir, agent_id_prefix)
        full_id = agent_row["agent_id"]
        aid_short = full_id[:8]
        print()
        print("#" * 78)
        print(f"# AGENT {full_id}")
        print("#" * 78)

        # Test (1): A/B
        print()
        print("### Test 1: A/B champion-obs ON vs OFF ###")
        ab = {}
        for use_co in (True, False):
            label = "ON" if use_co else "OFF"
            per_day_pnl = []
            per_day_bets = []
            for day_str in args.eval_days:
                env, batch, info = _run_rollout(
                    agent_row, args, cfg, bundle, use_co, day_str,
                )
                per_day_pnl.append(float(info.get("day_pnl", 0)))
                per_day_bets.append(len(env.all_settled_bets))
            ab[label] = {
                "total_pnl": sum(per_day_pnl),
                "per_day_pnl": per_day_pnl,
                "total_bets": sum(per_day_bets),
                "per_day_bets": per_day_bets,
            }
            print(f"  champion={label}: total_pnl=£{ab[label]['total_pnl']:+.2f} "
                  f"bets={ab[label]['total_bets']} per_day_pnl={['%.0f'%x for x in per_day_pnl]}")

        delta_pnl = ab["ON"]["total_pnl"] - ab["OFF"]["total_pnl"]
        delta_bets = ab["ON"]["total_bets"] - ab["OFF"]["total_bets"]
        print(f"  delta (ON - OFF): pnl=£{delta_pnl:+.2f}  bets={delta_bets:+d}")
        if abs(delta_pnl) < 1.0 and abs(delta_bets) <= 2:
            print("  -> policy is NOT meaningfully conditioning on champion obs")
        elif delta_pnl > 0:
            print("  -> policy IS using champion obs positively (champion ON helps)")
        else:
            print("  -> policy uses champion obs but it HURTS (champion ON makes it worse)")

        # Test (2): Strategic use — bet selection vs p_win population
        print()
        print("### Test 2: Strategic use — runner p_win at bet open ###")
        # Use the ON rollout's bets and race objects to look up p_win
        # Rebuild ON rollout to access env.day.races
        back_pwins = []
        lay_pwins = []
        pop_pwins_total = []

        for day_str in args.eval_days:
            env, _, _ = _run_rollout(
                agent_row, args, cfg, bundle, True, day_str,
            )
            as_of = datetime.strptime(day_str, "%Y-%m-%d").date()
            race_map = {r.market_id: r for r in env.day.races}

            # Population p_win across all (market, runner) pairs the agent SAW
            pwin_lookup: dict[tuple[str, int], float] = {}
            for race in env.day.races:
                pmap = _compute_pwin_for_race(bundle, race, as_of)
                for sid, pw in pmap.items():
                    pwin_lookup[(race.market_id, sid)] = pw
                    pop_pwins_total.append(pw)

            for bet in env.all_settled_bets:
                key = (bet.market_id, bet.selection_id)
                pw = pwin_lookup.get(key)
                if pw is None:
                    continue
                if bet.side is BetSide.BACK:
                    back_pwins.append(pw)
                else:
                    lay_pwins.append(pw)

        pop_mean = sum(pop_pwins_total) / len(pop_pwins_total) if pop_pwins_total else 0
        if back_pwins:
            back_mean = sum(back_pwins) / len(back_pwins)
            print(f"  back bets (n={len(back_pwins)}): mean p_win = {back_mean:.4f}  "
                  f"vs population {pop_mean:.4f}  delta={back_mean-pop_mean:+.4f}")
        if lay_pwins:
            lay_mean = sum(lay_pwins) / len(lay_pwins)
            print(f"  lay  bets (n={len(lay_pwins)}): mean p_win = {lay_mean:.4f}  "
                  f"vs population {pop_mean:.4f}  delta={lay_mean-pop_mean:+.4f}")

        # Distribution buckets
        if back_pwins:
            sorted_b = sorted(back_pwins)
            print(f"  back p_win quantiles: p10={sorted_b[len(sorted_b)//10]:.4f} "
                  f"p50={sorted_b[len(sorted_b)//2]:.4f} "
                  f"p90={sorted_b[9*len(sorted_b)//10]:.4f}")
        if lay_pwins:
            sorted_l = sorted(lay_pwins)
            print(f"  lay  p_win quantiles: p10={sorted_l[len(sorted_l)//10]:.4f} "
                  f"p50={sorted_l[len(sorted_l)//2]:.4f} "
                  f"p90={sorted_l[9*len(sorted_l)//10]:.4f}")

        # Verdict
        if back_pwins and lay_pwins:
            back_delta = (sum(back_pwins)/len(back_pwins)) - pop_mean
            lay_delta = (sum(lay_pwins)/len(lay_pwins)) - pop_mean
            if back_delta > 0.02 and lay_delta < -0.02:
                print("  -> aligned: backs higher p_win than population, lays lower")
            elif abs(back_delta) < 0.01 and abs(lay_delta) < 0.01:
                print("  -> uniform: bet selection is independent of p_win")
            elif back_delta < -0.02 or lay_delta > 0.02:
                print("  -> INVERTED: agent backing low-p_win / laying high-p_win!")
            else:
                print("  -> mild signal; not strongly conditioning on p_win")
    return 0


if __name__ == "__main__":
    sys.exit(main())
