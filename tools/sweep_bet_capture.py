"""Sweep bet-log capture over every completed agent in a cohort.

Works around the 2026-05-14 bet_logs/ wiring bug by re-creating the
eval rollouts ad-hoc. Builds the env ONCE per day (predictor warmup
is expensive) and loops agents inside that env scope, resetting
between agents.

Env match/gate logic doesn't depend on reward overrides (only the
policy obs does, and obs has no reward terms). So we can use a
single default-reward env per day and still get faithful trade
behaviour for every agent.

Output: one parquet per (agent, day) under
``registry/<TAG>/bet_logs/<agent_id>_adhoc_<date>.parquet``.

Usage:

    python -m tools.sweep_bet_capture \\
        --cohort-tag _predictor_SCALPING_layq_1778712871 \\
        --days 2026-05-04 2026-05-05 2026-05-06 \\
        --device cpu
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("sweep_bet_capture")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort-tag", required=True)
    p.add_argument("--days", nargs="+", default=[
        "2026-05-04", "2026-05-05", "2026-05-06",
    ])
    p.add_argument("--data-dir", default="data/processed", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--limit-agents", type=int, default=0,
        help="Cap on agents (0 = no cap, for smoke testing).",
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip (agent, day) pairs whose parquet already exists.",
    )
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )

    import torch

    from agents_v2.discrete_policy import DiscreteLSTMPolicy
    from predictors import PredictorBundle
    from registry.model_store import ModelStore
    from training_v2.cohort.worker import (
        _build_env_for_day,
        _build_eval_bet_records,
        scalping_train_config,
    )
    from training_v2.discrete_ppo.rollout import RolloutCollector

    cohort_dir = Path("registry") / args.cohort_tag
    if not cohort_dir.exists():
        logger.error("cohort dir %s missing", cohort_dir)
        return 1

    store = ModelStore(
        db_path=cohort_dir / "models.db",
        weights_dir=cohort_dir / "weights",
        bet_logs_dir=cohort_dir / "bet_logs",
    )

    # ── Read scoreboard for completed agents (agent_id + generation).
    scoreboard = cohort_dir / "scoreboard.jsonl"
    agents: list[tuple[str, int]] = []  # (agent_id, generation)
    seen: set[str] = set()
    with open(scoreboard) as f:
        for line in f:
            row = json.loads(line)
            aid = row.get("agent_id")
            gen = int(row.get("generation", 0))
            if aid and aid not in seen:
                seen.add(aid)
                agents.append((aid, gen))
    if args.limit_agents > 0:
        agents = agents[: args.limit_agents]
    logger.info("found %d completed agents in scoreboard", len(agents))

    # ── Predictor bundle (load once).
    root = Path(__file__).resolve().parents[1]
    sibling = root.parent / "betfair-predictors"
    bundle = PredictorBundle.from_manifests(
        champion_manifest=str(
            sibling / "production" / "race-outcome" / "manifest.json",
        ),
        ranker_manifest=str(
            sibling / "production" / "race-outcome-ranker" / "manifest.json",
        ),
        direction_manifest=str(
            sibling / "production" / "direction-predictor" / "manifest.json",
        ),
    )

    cfg = scalping_train_config()
    cfg["training"]["strategy_mode"] = "arb"

    # ── Iterate days; build env once per day; loop agents inside.
    total_captures = 0
    total_skipped = 0
    total_t0 = time.perf_counter()
    for day in args.days:
        logger.info("=== day %s — building env (default rewards) ===", day)
        env_t0 = time.perf_counter()
        _, shim = _build_env_for_day(
            day_str=day, data_dir=args.data_dir, cfg=cfg,
            scorer_dir=Path("models") / "scorer_v1",
            predictor_bundle=bundle,
            use_race_outcome_predictor=True,
            use_direction_predictor=True,
            predictor_lean_obs=True,
            predictor_p_win_back_threshold=0.20,
            predictor_p_win_lay_threshold=0.20,
            race_confidence_threshold=0.50,
            lay_price_max=20.0,
        )
        logger.info(
            "env for %s ready in %.1fs", day, time.perf_counter() - env_t0,
        )

        # Loop agents for this day.
        for i, (agent_id, gen) in enumerate(agents):
            out_path = (
                cohort_dir / "bet_logs"
                / f"adhoc_{agent_id}" / f"{day}.parquet"
            )
            if args.skip_existing and out_path.exists():
                total_skipped += 1
                continue

            record = store.get_model(agent_id)
            if record is None:
                logger.warning("agent %s missing model row; skipping", agent_id[:12])
                continue
            hidden_size = int(record.hyperparameters["hidden_size"])

            # Reset env to fresh state (new BetManager, new rollout).
            # collect_episode runs reset internally so an explicit reset
            # would only get clobbered; rely on the collector.

            policy = DiscreteLSTMPolicy(
                obs_dim=shim.obs_dim,
                action_space=shim.action_space,
                hidden_size=hidden_size,
            )
            weights_path = cohort_dir / "weights" / f"{agent_id}.pt"
            try:
                state = torch.load(
                    weights_path, weights_only=True, map_location="cpu",
                )
                if isinstance(state, dict) and "weights" in state:
                    state = state["weights"]
                policy.load_state_dict(state, strict=True)
            except Exception as e:
                logger.warning(
                    "agent %s: failed to load weights (%s); skipping",
                    agent_id[:12], e,
                )
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
                logger.warning(
                    "agent %s day %s: rollout failed (%s); skipping",
                    agent_id[:12], day, e,
                )
                continue

            try:
                records = _build_eval_bet_records(
                    env=shim.env,
                    day=shim.env.day,
                    starting_budget=float(shim.env.starting_budget),
                )
            except Exception as e:
                logger.warning(
                    "agent %s day %s: bet-records build failed (%s)",
                    agent_id[:12], day, e,
                )
                continue

            run_id = f"adhoc_{agent_id}"
            for r in records:
                r.run_id = run_id
            store.write_bet_logs_parquet(
                run_id=run_id, date=day, records=records,
            )
            total_captures += 1
            dt = time.perf_counter() - t0
            logger.info(
                "[%d/%d] gen%d %s %s: %d bets in %.1fs",
                i + 1, len(agents), gen, agent_id[:12], day,
                len(records), dt,
            )

    total_dt = time.perf_counter() - total_t0
    logger.info(
        "DONE: %d captures, %d skipped, total wall %.1fs",
        total_captures, total_skipped, total_dt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
