"""Train a fresh model with the same hyperparameters as 66ccb2 (the
'top performer') under the new arb mechanics + reward shaping.

Used to validate the changes from the 2026-04-15 session:
- MAX_ARB_TICKS=25 (realistic)
- Relaxed paired-placement junk filter
- Joint-affordability pre-flight (Betfair freed-budget rule, both
  LAY-after-BACK and BACK-after-LAY paths)
- Asymmetric naked-loss reward (naked losses count as cash; windfalls
  excluded)

Trains on 04-06..04-09 (same as the original gen-0 model), evaluates
on 04-10..04-13 (same test days), and prints a side-by-side comparison
with the most recent re-eval of the original model.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.architecture_registry import create_policy  # noqa: E402
from agents.ppo_trainer import PPOTrainer  # noqa: E402
from data.episode_builder import load_days  # noqa: E402
from env.betfair_env import (  # noqa: E402
    ACTIONS_PER_RUNNER,
    ACTION_SCHEMA_VERSION,
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

SOURCE_MODEL_ID = "66ccb2ef-0ffc-49a4-a138-7b6db29dedb0"
TRAIN_DATES = ["2026-04-06", "2026-04-07", "2026-04-08", "2026-04-09"]
TEST_DATES = ["2026-04-10", "2026-04-11", "2026-04-12", "2026-04-13"]
N_EPOCHS = 10


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

    src = store.get_model(SOURCE_MODEL_ID)
    if src is None:
        print(f"Source model {SOURCE_MODEL_ID} not found")
        return 1
    hp = dict(src.hyperparameters or {})
    # Override: the original model's naked_penalty_weight=3.97 was chosen
    # to fight the OLD broken shaping cap. Under the new asymmetric raw
    # cash-loss reward, stacking 3.97× shaping on top double-punished
    # nakeds and the trained policy collapsed to zero bets. Drop the
    # shaping weight low so raw cash drives learning.
    hp["naked_penalty_weight"] = 0.5
    arch = src.architecture_name
    max_runners = config["training"]["max_runners"]
    obs_dim, action_dim = _shapes_for(hp, max_runners)

    print(f"Source model: {SOURCE_MODEL_ID}")
    print(f"  arch: {arch}  scalping_mode: {hp.get('scalping_mode')}")
    print(f"  naked_penalty_weight: {hp.get('naked_penalty_weight'):.4f}")
    print(f"  arb_spread_scale:     {hp.get('arb_spread_scale'):.4f}")
    print(f"  early_lock_bonus_w:   {hp.get('early_lock_bonus_weight'):.4f}")
    print()

    new_model_id = store.create_model(
        generation=0,
        architecture_name=arch,
        architecture_description=(
            "Retrain clone of 66ccb2 under new arb mechanics + reward (2026-04-15)"
        ),
        hyperparameters=hp,
    )
    print(f"New model id: {new_model_id}")

    print(f"\nLoading {len(TRAIN_DATES)} train days...")
    train_days = load_days(
        TRAIN_DATES, data_dir=str(ROOT / config["paths"]["processed_data"]),
    )
    print(f"  Loaded {len(train_days)} days")

    policy = create_policy(
        name=arch,
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_runners=max_runners,
        hyperparams=hp,
    )

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    trainer = PPOTrainer(
        policy=policy,
        config=config,
        hyperparams=hp,
        device=device,
        model_id=new_model_id,
        architecture_name=arch,
    )

    t0 = time.time()
    print(f"\nTraining {len(train_days)} days × {N_EPOCHS} epochs = {len(train_days)*N_EPOCHS} episodes...")
    stats = trainer.train(train_days, n_epochs=N_EPOCHS)
    train_s = time.time() - t0
    print(f"\nTraining complete in {train_s/60:.1f} min")
    print(f"  episodes: {stats.episodes_completed}")
    print(f"  mean_reward: {stats.mean_reward:+.3f}")
    print(f"  mean_pnl:    {stats.mean_pnl:+.3f}")
    print(f"  mean_bets:   {stats.mean_bet_count:.1f}")
    print("\nPer-episode trajectory (epoch×day):")
    for i, ep in enumerate(stats.episode_stats):
        ac = getattr(ep, "arbs_completed", 0)
        an = getattr(ep, "arbs_naked", 0)
        print(
            f"  {i+1:>3d}  reward={ep.total_reward:+8.2f}  "
            f"pnl={ep.total_pnl:+7.2f}  bets={ep.bet_count:>4d}  "
            f"arbs={ac}/{ac+an}"
        )

    store.save_weights(
        new_model_id, policy.state_dict(),
        obs_schema_version=OBS_SCHEMA_VERSION,
        action_schema_version=ACTION_SCHEMA_VERSION,
    )

    print(f"\nLoading {len(TEST_DATES)} eval days...")
    test_days = load_days(
        TEST_DATES, data_dir=str(ROOT / config["paths"]["processed_data"]),
    )

    evaluator = Evaluator(config=config, model_store=store, device=device)
    market_type_filter = hp.get("market_type_filter", "BOTH")

    t0 = time.time()
    run_id, day_records = evaluator.evaluate(
        model_id=new_model_id,
        policy=policy,
        test_days=test_days,
        train_cutoff_date=TRAIN_DATES[-1],
        market_type_filter=market_type_filter,
        hyperparameters=hp or None,
    )
    eval_s = time.time() - t0
    print(f"Evaluation complete in {eval_s/60:.1f} min  run_id={run_id}\n")

    print(
        f"{'date':<12} {'pnl':>8} {'locked':>8} {'naked':>8} "
        f"{'bets':>5} {'arbs_c':>6} {'arbs_n':>6} "
        f"{'rej_blay':>8} {'fill_skip':>10}"
    )
    tot = {"pnl": 0.0, "locked": 0.0, "naked": 0.0, "bets": 0, "c": 0, "n": 0}
    for d in day_records:
        print(
            f"{d.date:<12} {d.day_pnl:>+8.2f} "
            f"{d.locked_pnl:>+8.2f} {d.naked_pnl:>+8.2f} "
            f"{d.bet_count:>5d} "
            f"{d.arbs_completed:>6d} {d.arbs_naked:>6d} "
            f"{d.paired_rejects_budget_lay:>8d} {d.paired_fill_skips:>10d}"
        )
        tot["pnl"] += d.day_pnl; tot["locked"] += d.locked_pnl
        tot["naked"] += d.naked_pnl; tot["bets"] += d.bet_count
        tot["c"] += d.arbs_completed; tot["n"] += d.arbs_naked
    print(
        f"{'TOTAL':<12} {tot['pnl']:>+8.2f} "
        f"{tot['locked']:>+8.2f} {tot['naked']:>+8.2f} "
        f"{tot['bets']:>5d} {tot['c']:>6d} {tot['n']:>6d}"
    )
    arb_rate = tot["c"] / max(tot["c"] + tot["n"], 1)
    print(f"\narb completion rate: {100*arb_rate:.1f}%")
    print(f"new model id: {new_model_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
