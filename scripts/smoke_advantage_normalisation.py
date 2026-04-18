"""Smoke test for plans/policy-startup-stability Session 01.

Launches a 1-agent, 5-episode scalping run with the advantage-
normalisation fix in place, then inspects episodes.jsonl for the
resulting per-episode policy_loss series. Exit criterion: episode 1
policy_loss < 100 (and ideally < 5).

Usage::

    python scripts/smoke_advantage_normalisation.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ppo_trainer import PPOTrainer
from agents.architecture_registry import create_policy
from data.episode_builder import load_days
from env.betfair_env import (
    ACTIONS_PER_RUNNER, AGENT_STATE_DIM, MARKET_DIM, POSITION_DIM,
    RUNNER_DIM, SCALPING_ACTIONS_PER_RUNNER, VELOCITY_DIM,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["training"]["starting_budget"] = 100.0
    config["training"]["scalping_mode"] = True
    config["reward"]["naked_penalty_weight"] = 0.5
    config["reward"]["early_lock_bonus_weight"] = 1.0

    # Unique model_id so the log entries can be filtered out afterwards.
    model_id = f"smoke-advnorm-{int(time.time())}"
    logger.info("Smoke run model_id=%s", model_id)

    processed = Path(config["paths"]["processed_data"])
    dates = sorted({
        f.name.replace("_runners.parquet", "")
        for f in processed.glob("*_runners.parquet")
        if (processed / f.name.replace("_runners.parquet", ".parquet")).exists()
    })
    if not dates:
        logger.error("No processed days available.")
        return 1
    # Take 5 days for 5 episodes.
    dates = dates[:5]
    logger.info("Using dates: %s", dates)
    days = load_days(dates, data_dir=config["paths"]["processed_data"])

    max_runners = config["training"]["max_runners"]
    obs_dim = (
        MARKET_DIM + VELOCITY_DIM
        + RUNNER_DIM * max_runners
        + AGENT_STATE_DIM
        + POSITION_DIM * max_runners
    )
    action_dim = max_runners * SCALPING_ACTIONS_PER_RUNNER
    policy = create_policy(
        "ppo_lstm_v1", obs_dim, action_dim, max_runners,
        hyperparams={
            "lstm_hidden_size": 128, "mlp_hidden_size": 128, "mlp_layers": 2,
        },
    )

    trainer = PPOTrainer(
        policy, config,
        hyperparams={
            "learning_rate": 3e-4,
            "ppo_epochs": 4,
            "mini_batch_size": 64,
        },
        model_id=model_id,
        architecture_name="ppo_lstm_v1",
    )

    start = time.time()
    trainer.train(days, n_epochs=1)
    elapsed = time.time() - start
    logger.info("Smoke train completed in %.1f s", elapsed)

    # Inspect episodes.jsonl for this model_id's policy_loss series.
    log_path = Path(config["paths"]["logs"]) / "training" / "episodes.jsonl"
    entries = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("model_id") == model_id:
                entries.append(rec)

    print()
    print("=" * 72)
    print(f"SMOKE RUN — model_id={model_id}")
    print("=" * 72)
    print(f"Episodes logged: {len(entries)}")
    for e in entries:
        print(
            f"  ep {e.get('episode'):>2}  "
            f"date={e.get('day_date', '?'):<12}  "
            f"reward={e.get('total_reward', 0.0):+.4f}  "
            f"policy_loss={e.get('policy_loss', 0.0):+.6f}"
        )
    print()

    if not entries:
        print("NO entries found — something went wrong.")
        return 2

    ep1_loss = abs(float(entries[0].get("policy_loss", 0.0)))
    print(f"Episode-1 |policy_loss| = {ep1_loss:.6f}")
    if ep1_loss < 5:
        print("PASS  — bounded <5, normalisation is sufficient.")
        return 0
    if ep1_loss < 100:
        print("PASS  — bounded <100 (normalisation sufficient; LR warmup not needed).")
        return 0
    print("FAIL  — >=100, LR warmup should ship as defence-in-depth.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
