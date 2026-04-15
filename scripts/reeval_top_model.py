"""One-off re-eval of a single model with the new paired-arb diagnostic.

Loads the model's saved policy + its 4 historical test days from the
registry, re-runs the Evaluator (which now persists the new
paired_rejects_* / paired_fill_skips columns), and prints a summary
broken down per day. Intended to be deleted after the diagnostic value
of the run has been captured.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.architecture_registry import create_policy  # noqa: E402
from data.episode_builder import load_days  # noqa: E402
from env.betfair_env import (  # noqa: E402
    ACTIONS_PER_RUNNER,
    AGENT_STATE_DIM,
    MARKET_DIM,
    OBS_SCHEMA_VERSION,
    POSITION_DIM,
    RUNNER_DIM,
    SCALPING_ACTIONS_PER_RUNNER,
    SCALPING_AGENT_STATE_DIM,
    SCALPING_POSITION_DIM,
    VELOCITY_DIM,
)
from registry.model_store import ModelStore  # noqa: E402
from training.evaluator import Evaluator  # noqa: E402

MODEL_ID = "66ccb2ef-0ffc-49a4-a138-7b6db29dedb0"
TEST_DATES = ["2026-04-10", "2026-04-11", "2026-04-12", "2026-04-13"]


def _shapes_for(hp: dict, max_runners: int) -> tuple[int, int]:
    is_scalp = bool(hp.get("scalping_mode", False))
    extra_pos = SCALPING_POSITION_DIM if is_scalp else 0
    extra_ag = SCALPING_AGENT_STATE_DIM if is_scalp else 0
    obs_dim = (
        MARKET_DIM
        + VELOCITY_DIM
        + (RUNNER_DIM * max_runners)
        + AGENT_STATE_DIM + extra_ag
        + ((POSITION_DIM + extra_pos) * max_runners)
    )
    apr = SCALPING_ACTIONS_PER_RUNNER if is_scalp else ACTIONS_PER_RUNNER
    return obs_dim, max_runners * apr


def main() -> int:
    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    store = ModelStore(
        db_path=str(ROOT / config["paths"]["registry_db"]),
        weights_dir=str(ROOT / config["paths"]["model_weights"]),
        bet_logs_dir=str(ROOT / config["paths"]["bet_logs"]),
    )

    rec = store.get_model(MODEL_ID)
    if rec is None:
        print(f"Model {MODEL_ID} not found")
        return 1

    hp = rec.hyperparameters or {}
    print(f"Model: {MODEL_ID}")
    print(f"  Architecture: {rec.architecture_name}")
    print(f"  scalping_mode: {hp.get('scalping_mode')}")

    max_runners = config["training"]["max_runners"]
    obs_dim, action_dim = _shapes_for(hp, max_runners)

    policy = create_policy(
        name=rec.architecture_name,
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_runners=max_runners,
        hyperparams=hp,
    )
    state_dict = store.load_weights(
        MODEL_ID, expected_obs_schema_version=OBS_SCHEMA_VERSION,
    )
    policy.load_state_dict(state_dict)

    print(f"\nLoading {len(TEST_DATES)} test days...")
    test_days = load_days(
        TEST_DATES, data_dir=str(ROOT / config["paths"]["processed_data"]),
    )
    print(f"  Loaded {len(test_days)} days\n")

    evaluator = Evaluator(config=config, model_store=store)
    market_type_filter = hp.get("market_type_filter", "BOTH")
    run_id, day_records = evaluator.evaluate(
        model_id=MODEL_ID,
        policy=policy,
        test_days=test_days,
        train_cutoff_date=TEST_DATES[0],
        market_type_filter=market_type_filter,
        hyperparameters=hp or None,
    )

    print(f"\nNew evaluation run: {run_id}")
    print()
    print(
        f"{'date':<12} {'pnl':>8} {'bets':>5} "
        f"{'arbs_c':>6} {'arbs_n':>6} "
        f"{'rej_noltp':>9} {'rej_pinv':>8} "
        f"{'rej_bback':>9} {'rej_blay':>8} "
        f"{'fill_skip':>10}"
    )
    for d in day_records:
        print(
            f"{d.date:<12} {d.day_pnl:>+8.2f} {d.bet_count:>5d} "
            f"{d.arbs_completed:>6d} {d.arbs_naked:>6d} "
            f"{d.paired_rejects_no_ltp:>9d} {d.paired_rejects_price_invalid:>8d} "
            f"{d.paired_rejects_budget_back:>9d} {d.paired_rejects_budget_lay:>8d} "
            f"{d.paired_fill_skips:>10d}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
