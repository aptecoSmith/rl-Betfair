"""Step 1c/1d (imitation-first): full-network BC canary, input-norm ON.

Now that the input-norm unblocker (Step 1b) is landed, run the literal
master_todo Step 1c/1d: train a v2 DiscreteLSTMPolicy by behavioural
cloning on the 42-day MATURATION-CONDITIONED oracle labels (full obs +
predictors), then eval on the 7 reserved HOLDOUT days via real env
rollout. Confirms a POLICY (not just LightGBM) can act on the learnable
maturation signal — does it lift holdout mat% off the ~12% base toward
the oracle's selectivity?

Differs from the cohort BC (which FREEZES the backbone — useless here
without input-norm + a trained backbone): this trains the WHOLE network
(input_proj + LSTM + actor_head) with per-dim input standardization ON,
so the backbone learns to use the normalized full obs.

BC-only imitates the oracle (~breakeven ceiling); profitability needs
the reward-aware PPO step (Step 2 proper, maturation_reward_mode now
wired). This canary's job is the mat%-LIFT learnability confirmation.
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
from env.bet_manager import MIN_BET_STAKE  # noqa: E402
from training_v2.arb_oracle import _load_config  # noqa: E402
from training_v2.cohort.worker import _build_env_for_day  # noqa: E402
from training_v2.discrete_ppo.bc_pretrain import (  # noqa: E402
    load_negative_samples_for_dates,
    load_oracle_samples_for_dates,
)

DATA_DIR = Path("data/processed")
PRED = (
    "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json",
    "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json",
    "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json",
)
HOLDOUT = [
    "2026-05-20", "2026-05-21", "2026-05-22", "2026-05-25",
    "2026-05-27", "2026-05-28", "2026-05-29",
]
RO = {"force_close_before_off_seconds": 120.0, "close_walk_ticks": 10.0}


def _train_days():
    import re
    pat = re.compile(r"^2026-\d{2}-\d{2}$")
    days = sorted(p.stem for p in DATA_DIR.glob("2026-*.parquet")
                  if pat.match(p.stem) and "2026-04-06" <= p.stem <= "2026-05-19")
    return days


def _bundle():
    from predictors import PredictorBundle
    c, r, d = PRED
    return PredictorBundle.from_manifests(
        champion_manifest=Path(c), ranker_manifest=Path(r),
        direction_manifest=Path(d),
    )


def _env(date, cfg, bundle):
    return _build_env_for_day(
        day_str=date, data_dir=DATA_DIR, cfg=cfg, scorer_dir=DEFAULT_SCORER_DIR,
        reward_overrides=RO, predictor_bundle=bundle,
        use_race_outcome_predictor=True, use_direction_predictor=True,
        predictor_lean_obs=False, emit_debug_features=False,
    )


def _val_metrics(policy, samples, space, device, n=8192):
    if not samples:
        return float("nan"), float("nan")
    b = samples[:n]
    obs = torch.tensor(np.stack([s.obs for s in b]), dtype=torch.float32, device=device)
    tgt = torch.tensor([space.encode(ActionType.OPEN_BACK, int(s.runner_idx)) for s in b],
                       dtype=torch.long, device=device)
    tr = policy.training
    policy.eval()
    with torch.no_grad():
        out = policy(obs)
        ce = float(F.cross_entropy(out.logits, tgt).item())
        acc = float((out.logits.argmax(-1) == tgt).float().mean().item())
    policy.train(tr)
    return ce, acc


@torch.no_grad()
def _eval_day(policy, shim, env, device):
    policy.eval()
    obs, _ = shim.reset()
    hidden = policy.init_hidden(batch=1)
    hidden = tuple(t.to(device) for t in hidden)
    done = False
    info = {}
    while not done:
        mask = shim.get_action_mask()
        obs_t = torch.tensor(np.asarray(obs), dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        out = policy(obs_t, hidden_state=hidden, mask=mask_t)
        hidden = out.new_hidden_state
        action_idx = int(out.action_dist.logits.argmax(-1).item())
        stake_unit = float((out.stake_alpha / (out.stake_alpha + out.stake_beta)).item())
        bm = env.bet_manager
        budget = bm.budget if bm is not None else 0.0
        stake = max(stake_unit * budget, MIN_BET_STAKE)
        obs, _r, term, trunc, info = shim.step(action_idx, stake=stake)
        done = bool(term or trunc)
    return info


def _eval_holdout(policy, cfg, bundle, device, tag):
    rows = []
    for date in HOLDOUT:
        env, shim = _env(date, cfg, bundle)
        info = _eval_day(policy, shim, env, device)
        po = int(info.get("pairs_opened", 0))
        comp = int(info.get("arbs_completed", 0))
        fc = int(info.get("arbs_force_closed", 0))
        rows.append({
            "date": date, "pairs_opened": po, "arbs_completed": comp,
            "arbs_force_closed": fc,
            "day_pnl": round(float(info.get("day_pnl", 0.0)), 2),
            "locked_pnl": round(float(info.get("locked_pnl", 0.0)), 2),
            "mat_pct": round(100.0 * comp / po, 2) if po else 0.0,
            "fc_pct": round(100.0 * fc / po, 2) if po else 0.0,
        })
        print(f"  [{tag} {date}] opened={po:>4} locked={rows[-1]['locked_pnl']:>8.2f} "
              f"day_pnl={rows[-1]['day_pnl']:>8.2f} mat%={rows[-1]['mat_pct']:>5.1f} "
              f"fc%={rows[-1]['fc_pct']:>5.1f}", flush=True)
    agg = {k: sum(r[k] for r in rows) for k in
           ("pairs_opened", "arbs_completed", "arbs_force_closed", "day_pnl", "locked_pnl")}
    po = agg["pairs_opened"]
    agg["mat_pct"] = round(100.0 * agg["arbs_completed"] / po, 2) if po else 0.0
    agg["fc_pct"] = round(100.0 * agg["arbs_force_closed"] / po, 2) if po else 0.0
    return rows, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--chunk-steps", type=int, default=500)
    ap.add_argument("--max-chunks", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neg-weight", type=float, default=0.1,
                    help="Weight on the NOOP-negative CE. The overfit diag "
                         "showed NOOP gradient is ~14x more concentrated than "
                         "the per-runner OPEN gradient; equal weight (1.0) "
                         "collapses the policy to NOOP. 0.0 = positives only.")
    ap.add_argument("--n-train-days", type=int, default=0,
                    help="If >0, evenly subsample this many BC train days "
                         "(faster iteration). 0 = all 40.")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="Skip the untrained-baseline holdout eval (known: "
                         "~115 opens/day, ~1%% mat).")
    ap.add_argument("--out", default="plans/imitation-first/_step1/bc_fullnet_results.json")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    cfg = _load_config()
    tdays = _train_days()
    val_days = tdays[-2:]
    bc_days = tdays[:-2]
    if args.n_train_days and args.n_train_days < len(bc_days):
        sel = np.linspace(0, len(bc_days) - 1, args.n_train_days).round().astype(int)
        bc_days = [bc_days[i] for i in sorted(set(sel.tolist()))]
    print(f"device={device} bc_train={len(bc_days)} bc_val={val_days} "
          f"holdout={len(HOLDOUT)} neg_weight={args.neg_weight}", flush=True)
    bundle = _bundle()

    probe_env, probe_shim = _env(HOLDOUT[0], cfg, bundle)
    obs_dim = int(probe_shim.obs_dim)
    runner_dim = int(probe_env.active_runner_dim)
    space = probe_shim.action_space
    print(f"obs_dim={obs_dim} runner_dim={runner_dim}", flush=True)

    print("loading BC samples...", flush=True)
    pos = load_oracle_samples_for_dates(bc_days, DATA_DIR, obs_dim)
    neg = load_negative_samples_for_dates(bc_days, DATA_DIR, obs_dim)
    val = load_oracle_samples_for_dates(val_days, DATA_DIR, obs_dim)
    print(f"pos={len(pos)} neg={len(neg)} val={len(val)}", flush=True)

    # Input-norm stats from the BC train obs (pos + a neg sample).
    stat_src = np.stack([s.obs for s in pos[:50000]] +
                        [s.obs for s in neg[:50000]], axis=0).astype(np.float64)
    obs_mean = stat_src.mean(axis=0)
    obs_std = stat_src.std(axis=0)
    print(f"obs stats: mean|.|={np.abs(obs_mean).mean():.3f} "
          f"std range [{obs_std.min():.3g}, {obs_std.max():.3g}]", flush=True)

    policy = DiscreteLSTMPolicy(
        obs_dim=obs_dim, action_space=space, hidden_size=args.hidden_size,
        runner_dim=runner_dim, input_norm=True,
    ).to(device)
    policy.set_input_norm_stats(obs_mean, obs_std)

    noop_idx = int(space.encode(ActionType.NOOP, None))
    pos_obs = np.stack([s.obs for s in pos], axis=0)
    pos_tgt = np.array([space.encode(ActionType.OPEN_BACK, int(s.runner_idx)) for s in pos], dtype=np.int64)
    neg_obs = np.stack([s.obs for s in neg], axis=0) if neg else None

    base_agg = None
    if not args.skip_baseline:
        print("\n=== BASELINE (untrained, input-norm) holdout eval ===", flush=True)
        _, base_agg = _eval_holdout(policy, cfg, bundle, device, "base")
        print(f"baseline: {base_agg}", flush=True)

    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    best_val, best_state, patience = float("inf"), None, args.patience
    trace = []
    print("\n=== full-network BC (input-norm ON) ===", flush=True)
    for chunk in range(args.max_chunks):
        policy.train()
        last = 0.0
        for _ in range(args.chunk_steps):
            pi = rng.integers(0, len(pos), size=args.batch_size)
            ob = torch.tensor(pos_obs[pi], dtype=torch.float32, device=device)
            tg = torch.tensor(pos_tgt[pi], dtype=torch.long, device=device)
            loss = F.cross_entropy(policy(ob).logits, tg)
            if neg_obs is not None and args.neg_weight > 0.0:
                ni = rng.integers(0, len(neg_obs), size=args.batch_size)
                nob = torch.tensor(neg_obs[ni], dtype=torch.float32, device=device)
                ntg = torch.full((args.batch_size,), noop_idx, dtype=torch.long, device=device)
                loss = loss + args.neg_weight * F.cross_entropy(policy(nob).logits, ntg)
            opt.zero_grad(); loss.backward(); opt.step()
            last = float(loss.item())
        vce, vacc = _val_metrics(policy, val, space, device)
        trace.append({"chunk": chunk, "steps": (chunk + 1) * args.chunk_steps,
                      "train_loss": round(last, 4), "val_ce": round(vce, 4),
                      "val_acc": round(vacc, 4)})
        print(f"  chunk {chunk:>2} steps={(chunk+1)*args.chunk_steps:>6} "
              f"train_loss={last:.4f} val_ce={vce:.4f} val_acc={vacc:.4f}", flush=True)
        if vce < best_val - 1e-4:
            best_val = vce
            best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                print(f"  early stop at chunk {chunk}", flush=True)
                break
    if best_state is not None:
        policy.load_state_dict(best_state)

    print("\n=== BC'd policy holdout eval ===", flush=True)
    rows, agg = _eval_holdout(policy, cfg, bundle, device, "bc")
    print(f"\n==== STEP 1c/1d BC HOLDOUT (input-norm ON) ====", flush=True)
    if base_agg is not None:
        print(f"BASELINE: opened={base_agg['pairs_opened']} mat%={base_agg['mat_pct']} "
              f"fc%={base_agg['fc_pct']} locked={base_agg['locked_pnl']:.2f} "
              f"day_pnl={base_agg['day_pnl']:.2f}", flush=True)
    else:
        print("BASELINE: (skipped; known ~115 opens/day, ~1% mat%)", flush=True)
    print(f"BC'd:     opened={agg['pairs_opened']} mat%={agg['mat_pct']} "
          f"fc%={agg['fc_pct']} locked={agg['locked_pnl']:.2f} "
          f"day_pnl={agg['day_pnl']:.2f}", flush=True)

    Path(args.out).write_text(json.dumps({
        "bc_train_days": bc_days, "val_days": val_days, "holdout": HOLDOUT,
        "obs_dim": obs_dim, "pos": len(pos), "neg": len(neg),
        "bc_trace": trace, "best_val_ce": round(best_val, 4),
        "baseline_holdout_agg": base_agg, "bc_holdout_per_day": rows,
        "bc_holdout_agg": agg,
    }, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
