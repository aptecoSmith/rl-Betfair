"""Step 1 (imitation-first): BC-to-convergence learnability test.

Trains a fresh v2 DiscreteLSTMPolicy by behavioural cloning on the
42-day MATURATION-CONDITIONED oracle labels (full obs + predictors),
holding out 2 train days for BC early-stop, then evaluates the BC'd
policy on the 7 reserved HOLDOUT days via real env rollout (fc=120,
close_walk=10). NO PPO — dense supervised gradient only.

The question: is the oracle's maturing-open decision LEARNABLE from
decision-time features? If the BC'd policy lifts holdout mat% materially
above the ~5% base rate (toward Step 0's 28-67% maturation ceiling),
maturation is predictable from obs -> proceed to Step 2. If flat -> not
learnable from current features -> richer data is the unlock (STOP).

Reuses the proven cohort components verbatim (no re-implementation):
``_build_env_for_day`` (env+shim, predictor-injected, full obs),
``DiscreteLSTMPolicy`` (h256, gates off, no frozen head),
``DiscreteBCPretrainer`` (actor_head CE toward OPEN_BACK + NOOP
negatives), ``RolloutCollector`` (deterministic holdout rollout).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agents_v2.action_space import ActionType  # noqa: E402
from agents_v2.discrete_policy import DiscreteLSTMPolicy  # noqa: E402
from agents_v2.env_shim import DEFAULT_SCORER_DIR  # noqa: E402
from training_v2.arb_oracle import _load_config  # noqa: E402
from training_v2.cohort.worker import _build_env_for_day  # noqa: E402
from training_v2.discrete_ppo.bc_pretrain import (  # noqa: E402
    DiscreteBCPretrainer,
    load_negative_samples_for_dates,
    load_oracle_samples_for_dates,
)
from training_v2.discrete_ppo.rollout import RolloutCollector  # noqa: E402

DATA_DIR = Path("data/processed")
PRED_MANIFESTS = (
    "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json",
    "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json",
    "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json",
)
HOLDOUT_DAYS = [
    "2026-05-20", "2026-05-21", "2026-05-22", "2026-05-25",
    "2026-05-27", "2026-05-28", "2026-05-29",
]
REWARD_OVERRIDES = {
    "force_close_before_off_seconds": 120.0,
    "close_walk_ticks": 10.0,
}


def _train_days() -> list[str]:
    import re
    pat = re.compile(r"^2026-\d{2}-\d{2}$")
    days = sorted(
        p.stem for p in DATA_DIR.glob("2026-*.parquet")
        if pat.match(p.stem)
    )
    return [d for d in days if "2026-04-06" <= d <= "2026-05-19"]


def _make_bundle():
    from predictors import PredictorBundle
    champ, rank, dirm = PRED_MANIFESTS
    return PredictorBundle.from_manifests(
        champion_manifest=Path(champ),
        ranker_manifest=Path(rank),
        direction_manifest=Path(dirm),
    )


def _build_env(date, cfg, bundle):
    return _build_env_for_day(
        day_str=date,
        data_dir=DATA_DIR,
        cfg=cfg,
        scorer_dir=DEFAULT_SCORER_DIR,
        reward_overrides=REWARD_OVERRIDES,
        predictor_bundle=bundle,
        use_race_outcome_predictor=True,
        use_direction_predictor=True,
        predictor_lean_obs=False,
        emit_debug_features=False,
    )


def _measure_val(policy, val_samples, action_space, device, max_n=8192):
    if not val_samples:
        return float("nan"), float("nan")
    batch = val_samples[:max_n]
    obs = torch.tensor(
        np.stack([s.obs for s in batch], axis=0),
        dtype=torch.float32, device=device,
    )
    tgt = torch.tensor(
        [action_space.encode(ActionType.OPEN_BACK, int(s.runner_idx))
         for s in batch],
        dtype=torch.long, device=device,
    )
    was_training = policy.training
    policy.eval()
    with torch.no_grad():
        out = policy(obs)
        ce = float(F.cross_entropy(out.logits, tgt).item())
        acc = float((out.logits.argmax(-1) == tgt).float().mean().item())
    policy.train(was_training)
    return ce, acc


def _eval_holdout(policy, cfg, bundle, device):
    rows = []
    for date in HOLDOUT_DAYS:
        env, shim = _build_env(date, cfg, bundle)
        collector = RolloutCollector(shim, policy, device=str(device))
        collector.collect_episode(deterministic=True)
        info = collector.last_info or {}
        po = int(info.get("pairs_opened", 0))
        comp = int(info.get("arbs_completed", 0))
        fc = int(info.get("arbs_force_closed", 0))
        naked = int(info.get("arbs_naked", 0))
        rows.append({
            "date": date,
            "pairs_opened": po,
            "arbs_completed": comp,
            "arbs_force_closed": fc,
            "arbs_naked": naked,
            "day_pnl": round(float(info.get("day_pnl", 0.0)), 2),
            "locked_pnl": round(float(info.get("locked_pnl", 0.0)), 2),
            "naked_pnl": round(float(info.get("naked_pnl", 0.0)), 2),
            "mat_pct": round(100.0 * comp / po, 2) if po else 0.0,
            "fc_pct": round(100.0 * fc / po, 2) if po else 0.0,
        })
        print(
            f"[holdout {date}] opened={po:>4} locked={rows[-1]['locked_pnl']:>9.2f} "
            f"day_pnl={rows[-1]['day_pnl']:>9.2f} mat%={rows[-1]['mat_pct']:>5.1f} "
            f"fc%={rows[-1]['fc_pct']:>5.1f}", flush=True,
        )
    agg = {k: sum(r[k] for r in rows) for k in (
        "pairs_opened", "arbs_completed", "arbs_force_closed",
        "arbs_naked", "day_pnl", "locked_pnl", "naked_pnl",
    )}
    po = agg["pairs_opened"]
    agg["mat_pct"] = round(100.0 * agg["arbs_completed"] / po, 2) if po else 0.0
    agg["fc_pct"] = round(100.0 * agg["arbs_force_closed"] / po, 2) if po else 0.0
    return rows, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--bc-lr", type=float, default=3e-4)
    ap.add_argument("--bc-batch-size", type=int, default=128)
    ap.add_argument("--chunk-steps", type=int, default=1000)
    ap.add_argument("--max-chunks", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="plans/imitation-first/_step1/bc_probe_results.json")
    ap.add_argument(
        "--eval-baseline", action="store_true",
        help="Also eval the UNTRAINED policy on holdout (control).",
    )
    args = ap.parse_args()

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available())
        else "cpu"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = _load_config()
    train_days = _train_days()
    bc_val_days = train_days[-2:]
    bc_train_days = train_days[:-2]
    print(f"train={len(train_days)} bc_train={len(bc_train_days)} "
          f"bc_val={bc_val_days} holdout={len(HOLDOUT_DAYS)}", flush=True)

    print("loading predictor bundle...", flush=True)
    bundle = _make_bundle()

    # Build one env to get obs_dim / action_space / runner_dim.
    probe_env, probe_shim = _build_env(HOLDOUT_DAYS[0], cfg, bundle)
    obs_dim = int(probe_shim.obs_dim)
    runner_dim = int(probe_env.active_runner_dim)
    print(f"obs_dim={obs_dim} runner_dim={runner_dim} "
          f"max_runners={probe_shim.max_runners}", flush=True)

    policy = DiscreteLSTMPolicy(
        obs_dim=obs_dim,
        action_space=probe_shim.action_space,
        hidden_size=int(args.hidden_size),
        runner_dim=runner_dim,
    ).to(device)

    action_space = probe_shim.action_space

    print("loading BC samples...", flush=True)
    train_pos = load_oracle_samples_for_dates(bc_train_days, DATA_DIR, obs_dim)
    train_neg = load_negative_samples_for_dates(bc_train_days, DATA_DIR, obs_dim)
    val_pos = load_oracle_samples_for_dates(bc_val_days, DATA_DIR, obs_dim)
    print(f"BC pool: train_pos={len(train_pos)} train_neg={len(train_neg)} "
          f"val_pos={len(val_pos)}", flush=True)

    baseline_agg = None
    if args.eval_baseline:
        print("\n=== BASELINE (untrained policy) holdout eval ===", flush=True)
        _, baseline_agg = _eval_holdout(policy, cfg, bundle, device)
        print(f"baseline agg: {baseline_agg}", flush=True)

    pretrainer = DiscreteBCPretrainer(
        lr=args.bc_lr, batch_size=args.bc_batch_size, seed=args.seed,
    )

    best_val = float("inf")
    best_state = None
    patience = args.patience
    bc_trace = []
    print("\n=== BC training (chunked, early-stop on val CE) ===", flush=True)
    for chunk in range(args.max_chunks):
        hist = pretrainer.pretrain(
            policy, train_pos, n_steps=args.chunk_steps,
            negative_samples=train_neg,
        )
        val_ce, val_acc = _measure_val(policy, val_pos, action_space, device)
        train_ce = hist.final_ce_loss
        bc_trace.append({
            "chunk": chunk, "steps": (chunk + 1) * args.chunk_steps,
            "train_ce": round(train_ce, 4),
            "val_ce": round(val_ce, 4), "val_acc": round(val_acc, 4),
        })
        print(f"  chunk {chunk:>2} steps={ (chunk+1)*args.chunk_steps:>6} "
              f"train_ce={train_ce:.4f} val_ce={val_ce:.4f} "
              f"val_acc={val_acc:.4f}", flush=True)
        if val_ce < best_val - 1e-4:
            best_val = val_ce
            best_state = {k: v.detach().cpu().clone()
                          for k, v in policy.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                print(f"  early stop at chunk {chunk} (val CE plateaued)",
                      flush=True)
                break

    if best_state is not None:
        policy.load_state_dict(best_state)

    print("\n=== BC'd policy holdout eval ===", flush=True)
    rows, agg = _eval_holdout(policy, cfg, bundle, device)
    print(f"\n==== STEP 1 BC HOLDOUT AGGREGATE ====", flush=True)
    print(f"opened={agg['pairs_opened']} locked={agg['locked_pnl']:.2f} "
          f"day_pnl={agg['day_pnl']:.2f} mat%={agg['mat_pct']} "
          f"fc%={agg['fc_pct']}", flush=True)

    out = {
        "bc_train_days": bc_train_days, "bc_val_days": bc_val_days,
        "holdout_days": HOLDOUT_DAYS, "obs_dim": obs_dim,
        "bc_pool": {"train_pos": len(train_pos), "train_neg": len(train_neg),
                    "val_pos": len(val_pos)},
        "bc_trace": bc_trace, "best_val_ce": round(best_val, 4),
        "baseline_holdout": baseline_agg,
        "bc_holdout_per_day": rows, "bc_holdout_agg": agg,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
