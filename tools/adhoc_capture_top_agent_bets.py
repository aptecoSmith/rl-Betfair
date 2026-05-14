"""One-off bet-log capture for a single trained agent.

Workaround for the 2026-05-14 bet_logs/ wiring bug in
training_v2/cohort/worker.py: we passed the env as ``day`` to
``_build_eval_bet_records``, so every agent in the mid-flight
cohort failed bet-log capture. The worker fix landed but the
already-launched cohort process won't pick it up. This script
re-creates the eval rollout for ONE saved agent and dumps the
bets as parquet so we can analyse how the agent bets.

Run on CPU to avoid contending GPU with the still-running
cohort. Slower (~10-15 min/day) but doesn't disturb training.

Usage:

    python -m tools.adhoc_capture_top_agent_bets \\
        --cohort-tag _predictor_SCALPING_layq_1778712871 \\
        --agent-id 747d3d62-9350-4282-a5a3-1aee07f2230f \\
        --day 2026-05-04
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("adhoc_capture")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort-tag", required=True)
    p.add_argument("--agent-id", required=True)
    p.add_argument("--day", default="2026-05-04")
    p.add_argument("--data-dir", default="data/processed", type=Path)
    p.add_argument("--device", default="cpu")
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
    store = ModelStore(
        db_path=cohort_dir / "models.db",
        weights_dir=cohort_dir / "weights",
        bet_logs_dir=cohort_dir / "bet_logs",
    )

    record = store.get_model(args.agent_id)
    if record is None:
        logger.error("agent %s not in registry", args.agent_id)
        return 1
    hp = record.hyperparameters
    hidden_size = int(hp["hidden_size"])
    logger.info(
        "agent %s gen=%d arch=%s hidden_size=%d",
        args.agent_id[:12], record.generation,
        record.architecture_name, hidden_size,
    )

    root = Path(__file__).resolve().parents[1]
    sibling = root.parent / "betfair-predictors"
    bundle = PredictorBundle.from_manifests(
        champion_manifest=str(sibling / "production" / "race-outcome" / "manifest.json"),
        ranker_manifest=str(sibling / "production" / "race-outcome-ranker" / "manifest.json"),
        direction_manifest=str(sibling / "production" / "direction-predictor" / "manifest.json"),
    )

    cfg = scalping_train_config()
    cfg["training"]["strategy_mode"] = "arb"

    _, shim = _build_env_for_day(
        day_str=args.day, data_dir=args.data_dir, cfg=cfg,
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

    policy = DiscreteLSTMPolicy(
        obs_dim=shim.obs_dim,
        action_space=shim.action_space,
        hidden_size=hidden_size,
    )
    weights_path = cohort_dir / "weights" / f"{args.agent_id}.pt"
    state = torch.load(weights_path, weights_only=True, map_location="cpu")
    if isinstance(state, dict) and "weights" in state:
        state = state["weights"]
    policy.load_state_dict(state, strict=True)
    policy.to(args.device)
    policy.eval()
    logger.info("policy loaded from %s", weights_path)

    collector = RolloutCollector(shim=shim, policy=policy, device=args.device)
    collector.collect_episode(deterministic=True)
    logger.info("rollout complete")

    records = _build_eval_bet_records(
        env=shim.env,
        day=shim.env.day,
        starting_budget=float(shim.env.starting_budget),
    )
    logger.info("built %d bet records", len(records))

    out_path = cohort_dir / "bet_logs" / f"{args.agent_id}_adhoc_{args.day}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for r in records:
        r.run_id = f"adhoc_{args.agent_id}"
    store.write_bet_logs_parquet(
        run_id=f"adhoc_{args.agent_id}", date=args.day, records=records,
    )
    logger.info("written bet logs under %s", cohort_dir / "bet_logs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
