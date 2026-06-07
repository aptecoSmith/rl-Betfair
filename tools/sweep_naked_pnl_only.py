"""Slim bet capture: writes only per-naked-leg pnl for variance analysis.

Bypasses worker.py's _build_eval_bet_records helper (which uses new
kwargs the raceconf-era env doesn't support). Captures the minimum:
per-agent x per-day list of naked-leg pnls.

Output: one CSV row per (agent, naked-bet) with pnl, day, side, price.
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("sweep_naked")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort-tag", required=True)
    p.add_argument("--cohort-dir-absolute", required=True, type=Path)
    p.add_argument("--days", nargs="+", default=[
        "2026-05-04", "2026-05-05", "2026-05-06",
    ])
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--scorer-dir", required=True, type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--predictor-manifests", nargs=3, required=True)
    p.add_argument("--predictor-back-thr", type=float, default=0.20)
    p.add_argument("--predictor-lay-thr", type=float, default=0.40)
    p.add_argument("--race-confidence-thr", type=float, default=0.50)
    p.add_argument("--out-csv", required=True, type=Path)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
    )

    import torch
    from agents_v2.discrete_policy import DiscreteLSTMPolicy
    from agents_v2.env_shim import DiscreteActionShim
    from data.episode_builder import load_day
    from env.betfair_env import BetfairEnv
    from predictors import PredictorBundle
    from registry.model_store import ModelStore
    from training_v2.discrete_ppo.rollout import RolloutCollector

    store = ModelStore(
        db_path=args.cohort_dir_absolute / "models.db",
        weights_dir=args.cohort_dir_absolute / "weights",
        bet_logs_dir=args.cohort_dir_absolute / "bet_logs",
    )

    # Agent list from scoreboard
    agents = []
    seen = set()
    with open(args.cohort_dir_absolute / "scoreboard.jsonl") as f:
        for line in f:
            r = json.loads(line)
            aid = r.get("agent_id")
            if aid and aid not in seen:
                seen.add(aid)
                agents.append((aid, int(r.get("generation", 0))))
    logger.info("Loaded %d unique agents from scoreboard", len(agents))

    bundle = PredictorBundle.from_manifests(
        champion_manifest=args.predictor_manifests[0],
        ranker_manifest=args.predictor_manifests[1],
        direction_manifest=args.predictor_manifests[2],
    )

    # Open output CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow([
            "agent_id", "gen", "day", "pair_id", "side", "price",
            "stake", "selection_id", "pnl", "tick_index",
        ])

        # Iterate days; build env once per day; loop agents inside
        for day_str in args.days:
            logger.info("=== day %s ===", day_str)
            day = load_day(day_str, data_dir=args.data_dir)
            cfg = {
                "training": {
                    "max_runners": 14,
                    "starting_budget": 100.0,
                    "max_bets_per_race": 20,
                    "scalping_mode": True,
                    "strategy_mode": "arb",
                },
                "actions": {"force_aggressive": True},
                "reward": {
                    "early_pick_bonus_min": 1.2,
                    "early_pick_bonus_max": 1.5,
                    "early_pick_min_seconds": 300,
                    "efficiency_penalty": 0.01,
                    "commission": 0.05,
                },
            }
            # Build env once with the raceconf gate config (no lay_price_max)
            env = BetfairEnv(
                day, cfg,
                predictor_bundle=bundle,
                use_race_outcome_predictor=True,
                use_direction_predictor=True,
                predictor_lean_obs=True,
                predictor_p_win_back_threshold=args.predictor_back_thr,
                predictor_p_win_lay_threshold=args.predictor_lay_thr,
                race_confidence_threshold=args.race_confidence_thr,
            )
            shim = DiscreteActionShim(env, scorer_dir=args.scorer_dir)

            for i, (aid, gen) in enumerate(agents):
                rec = store.get_model(aid)
                if rec is None:
                    continue
                hidden_size = int(rec.hyperparameters["hidden_size"])
                policy = DiscreteLSTMPolicy(
                    obs_dim=shim.obs_dim,
                    action_space=shim.action_space,
                    hidden_size=hidden_size,
                )
                weights_path = args.cohort_dir_absolute / "weights" / f"{aid}.pt"
                try:
                    state = torch.load(
                        weights_path, weights_only=True, map_location="cpu",
                    )
                    if isinstance(state, dict) and "weights" in state:
                        state = state["weights"]
                    policy.load_state_dict(state, strict=True)
                except Exception as e:
                    logger.warning("agent %s: weights fail (%s)", aid[:12], e)
                    continue
                policy.to(args.device)
                policy.eval()

                t0 = time.perf_counter()
                collector = RolloutCollector(
                    shim=shim, policy=policy, device=args.device,
                )
                try:
                    collector.collect_episode(deterministic=True)
                except Exception as e:
                    logger.warning("agent %s day %s rollout fail (%s)",
                                   aid[:12], day_str, e)
                    continue

                # Identify nakeds: count legs per pair_id in bm.bets.
                # A pair with only 1 matched leg = naked.
                all_bets = list(env.all_settled_bets)
                from collections import defaultdict
                pair_legs = defaultdict(list)
                for b in all_bets:
                    pid = getattr(b, "pair_id", None)
                    if pid is not None:
                        pair_legs[pid].append(b)
                for pid, legs in pair_legs.items():
                    if len(legs) >= 2:
                        continue  # matured — not naked
                    leg = legs[0]
                    writer.writerow([
                        aid, gen, day_str, pid,
                        leg.side.value, float(leg.average_price),
                        float(leg.matched_stake), int(leg.selection_id),
                        float(leg.pnl), int(leg.tick_index),
                    ])
                csv_f.flush()
                dt = time.perf_counter() - t0
                logger.info("[%d/%d] gen%d %s %s: %d naked legs in %.1fs",
                            i + 1, len(agents), gen, aid[:12], day_str,
                            sum(1 for legs in pair_legs.values() if len(legs) < 2),
                            dt)
    logger.info("DONE — wrote %s", args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
