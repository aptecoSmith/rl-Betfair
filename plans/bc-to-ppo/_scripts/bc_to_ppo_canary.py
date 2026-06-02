"""BC -> PPO canary (plans/bc-to-ppo Step 2+3). The profit step.

Warm-starts a DiscreteLSTMPolicy from the bc-getting-it-right Step B weights
(mature_prob_head holdout AUC 0.745), BC-trains the ACTOR to open on oracle
opportunities (mature head stays frozen at 0.745 — DiscreteBCPretrainer only
touches actor_head + direction_prob_head), then PPO-fine-tunes with the
MATURATION reward + open_cost toll so the policy learns to open SELECTIVELY
(fewer, higher-fill-prob pairs -> lower force-close rate -> locked clears the
toll). Fully hedged: fc=120, close_walk=10, the force-close safety barrier
(0.50, config default), input_norm ON. SELECT ON LOCKED.

This is the single-config standalone canary (hard_constraints §8). If it
drives holdout LOCKED cleanly positive with mat% above the BC's 14-15% via
LOWER fc%, Step 4 (GA cohort) is warranted — OPERATOR-GATED.

Usage (smoke): --train-days 2026-05-15,2026-05-16 --epochs 1 --eval-days 2026-05-20
Real canary:   --train-days alltrain_bc --epochs 3 --eval-days holdout
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agents_v2.action_space import ActionType  # noqa: E402
from agents_v2.discrete_policy import DiscreteLSTMPolicy  # noqa: E402
from agents_v2.env_shim import DEFAULT_SCORER_DIR  # noqa: E402
from env.bet_manager import MIN_BET_STAKE  # noqa: E402
from training_v2.arb_oracle import _load_config  # noqa: E402
from training_v2.cohort.worker import _build_env_for_day  # noqa: E402
from training_v2.discrete_ppo.trainer import DiscretePPOTrainer  # noqa: E402
from training_v2.discrete_ppo.rollout import RolloutCollector  # noqa: E402
from training_v2.discrete_ppo.bc_pretrain import (  # noqa: E402
    DiscreteBCPretrainer,
    load_negative_samples_for_dates,
    load_oracle_samples_for_dates,
    measure_post_bc_entropy,
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


def _all_train_days():
    import re
    pat = re.compile(r"^2026-\d{2}-\d{2}$")
    return sorted(p.stem for p in DATA_DIR.glob("2026-*.parquet")
                  if pat.match(p.stem) and "2026-04-06" <= p.stem <= "2026-05-19")


def _resolve_days(spec):
    if spec == "holdout":
        return list(HOLDOUT)
    if spec == "alltrain_bc":
        return [d for d in _all_train_days() if d not in ("2026-05-18", "2026-05-19")]
    return [d.strip() for d in spec.split(",") if d.strip()]


def _bundle():
    from predictors import PredictorBundle
    c, r, d = PRED
    return PredictorBundle.from_manifests(
        champion_manifest=Path(c), ranker_manifest=Path(r),
        direction_manifest=Path(d),
    )


def _env(date, cfg, bundle, reward_overrides):
    return _build_env_for_day(
        day_str=date, data_dir=DATA_DIR, cfg=cfg, scorer_dir=DEFAULT_SCORER_DIR,
        reward_overrides=reward_overrides, predictor_bundle=bundle,
        use_race_outcome_predictor=True, use_direction_predictor=True,
        predictor_lean_obs=False, emit_debug_features=False,
    )


@torch.no_grad()
def _eval_day(policy, shim, env, device=None):
    """Actor-driven deterministic eval (argmax of the masked action dist,
    with the mature gate). This is the policy's ACTUAL behaviour — distinct
    from the greedy-by-mature_prob rollout (which bypassed the actor).

    Device is derived from the policy itself — the PPO trainer parks the
    policy on its ``rollout_device`` (cpu in split-device mode) after
    training, so a fixed ``device`` arg would mismatch the input-norm
    buffers. Always use where the policy actually lives."""
    device = next(policy.parameters()).device
    policy.eval()
    obs, _ = shim.reset()
    hidden = policy.init_hidden(batch=1)
    hidden = tuple(t.to(device) for t in hidden)
    done = False
    info = {}
    while not done:
        mask = shim.get_action_mask()
        obs_t = torch.tensor(np.asarray(obs), dtype=torch.float32,
                             device=device).unsqueeze(0)
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


def _eval_holdout(policy, cfg, bundle, reward_overrides, device, tag):
    rows = []
    for date in HOLDOUT:
        env, shim = _env(date, cfg, bundle, reward_overrides)
        info = _eval_day(policy, shim, env, device)
        po = int(info.get("pairs_opened", 0))
        comp = int(info.get("arbs_completed", 0))
        fc = int(info.get("arbs_force_closed", 0))
        rows.append({
            "date": date, "pairs_opened": po, "arbs_completed": comp,
            "arbs_force_closed": fc,
            "day_pnl": round(float(info.get("day_pnl", 0.0)), 2),
            "locked_pnl": round(float(info.get("locked_pnl", 0.0)), 2),
            "peak_open_liability": round(float(info.get("peak_open_liability", 0.0)), 2),
            "mat_pct": round(100.0 * comp / po, 2) if po else 0.0,
            "fc_pct": round(100.0 * fc / po, 2) if po else 0.0,
        })
        print(f"  [{tag} {date}] opened={po:>4} locked={rows[-1]['locked_pnl']:>8.2f} "
              f"day_pnl={rows[-1]['day_pnl']:>8.2f} mat%={rows[-1]['mat_pct']:>5.1f} "
              f"fc%={rows[-1]['fc_pct']:>5.1f}", flush=True)
    agg = {k: sum(r[k] for r in rows) for k in
           ("pairs_opened", "arbs_completed", "arbs_force_closed",
            "day_pnl", "locked_pnl")}
    po = agg["pairs_opened"]
    agg["mat_pct"] = round(100.0 * agg["arbs_completed"] / po, 2) if po else 0.0
    agg["fc_pct"] = round(100.0 * agg["arbs_force_closed"] / po, 2) if po else 0.0
    return rows, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-start", default="plans/bc-getting-it-right/_scripts/stepB_alltrain_wd3e-3.pt")
    ap.add_argument("--train-days", default="2026-05-15,2026-05-16")
    ap.add_argument("--eval-days", default="2026-05-20")
    ap.add_argument("--epochs", type=int, default=1, help="passes over train days")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--starting-budget", type=float, default=200.0)
    ap.add_argument("--maturation-reward-mode", default="off",
                    help="'full' = raw reward is matured-locked only (hides the "
                         "toll); 'off' (default) = REAL cash P&L (the clean "
                         "reward — agent feels the toll, learns to cut losers "
                         "via MTM). The fresh-reward reframe uses 'off'.")
    ap.add_argument("--stop-loss", type=float, default=0.0,
                    help="stop_loss_pnl_threshold (FRACTION of stake). >0 = cut "
                         "a pair mid-race when per-pair MTM crosses -frac*stake.")
    ap.add_argument("--locked-weight", type=float, default=0.0,
                    help="locked_pnl_reward_weight (operator idea): amplify the "
                         "POSITIVE locked (matured) reward by this in the shaped "
                         "channel so it isn't drowned by loss variance and the "
                         "net value of opening flips positive (no NOOP collapse). "
                         "9.0 = '10x locked' (1x raw + 9x shaped). Losses stay "
                         "1x. Needs --per-pair-resolution 1.")
    ap.add_argument("--per-pair-resolution", type=int, default=1,
                    help="per_pair_reward_at_resolution. 1 (default) = credit "
                         "each pair's realised P&L (incl. force-close LOSSES) at "
                         "the tick it RESOLVES, per-pair — so the model can "
                         "attribute outcomes to specific opens (the credit-"
                         "assignment fix). 0 = lump all P&L at settle (the old, "
                         "blindfolded behaviour).")
    ap.add_argument("--open-cost", type=float, default=0.0)
    ap.add_argument("--mature-prob-loss-weight", type=float, default=1.0)
    ap.add_argument("--mature-open-threshold", type=float, default=0.0,
                    help="mature_prob open-gate (0=off, rely on reward). >0 masks "
                         "OPEN where mature_prob<thr (warmup-annealed by trainer).")
    ap.add_argument("--fixed-stake-unit", type=float, default=0.0,
                    help="If >0, set the Beta stake head to a CONSTANT stake_unit "
                         "and FREEZE it. The env decodes stake=stake_unit*budget, "
                         "so 0.1 @ £100 budget = £10/open. The warm-start's stake "
                         "head is untrained (~0.5 -> oversized stakes that can't "
                         "post the passive). Pin it so the canary isolates the "
                         "OPEN-selectivity question, not stake-head cold-start.")
    ap.add_argument("--bc-actor-steps", type=int, default=2000,
                    help="DiscreteBCPretrainer steps on the actor before PPO (0=skip)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy-coeff", type=float, default=0.01,
                    help="Entropy bonus. Default 0.01 is too weak — the policy "
                         "collapses to deterministic NOOP (absorbing state) in "
                         "~2 episodes. Crank to ~0.1-0.3 to keep it stochastic "
                         "(always some open prob) so it keeps trading + learning "
                         "which opens are the RIGHT ones.")
    ap.add_argument("--mtm-weight", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-baseline-eval", action="store_true")
    ap.add_argument("--out", default="plans/bc-to-ppo/_scripts/canary_result.json")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    train_days = _resolve_days(args.train_days)
    eval_is_holdout = (args.eval_days == "holdout")
    cfg = _load_config()
    import copy
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("training", {})["starting_budget"] = float(args.starting_budget)

    # Reward overrides: deployment pins + the maturation reward + open_cost +
    # MTM. The force-close barrier (0.50) comes from config.yaml by default.
    reward_overrides = {
        "force_close_before_off_seconds": 120.0,
        "close_walk_ticks": 10.0,
        "open_cost": args.open_cost,
        "mark_to_market_weight": args.mtm_weight,
    }
    # maturation_reward_mode is a BOOL (default False = REAL cash). Only add
    # it (True) when explicitly 'full'; 'off'/'none'/'' → omit → raw cash.
    mrm_on = str(args.maturation_reward_mode).lower() in ("full", "true", "on", "1")
    if mrm_on:
        reward_overrides["maturation_reward_mode"] = True
    if args.stop_loss > 0.0:
        reward_overrides["stop_loss_pnl_threshold"] = float(args.stop_loss)
    if args.per_pair_resolution:
        reward_overrides["per_pair_reward_at_resolution"] = True
    if args.locked_weight > 0.0:
        reward_overrides["locked_pnl_reward_weight"] = float(args.locked_weight)
    print(f"device={device} budget=£{args.starting_budget} "
          f"reward={'matured-only' if mrm_on else 'REAL-cash'} "
          f"stop_loss=£{args.stop_loss} open_cost={args.open_cost} "
          f"mtm={args.mtm_weight} mature_gate={args.mature_open_threshold} "
          f"mature_prob_loss_weight={args.mature_prob_loss_weight}", flush=True)
    print(f"train_days={train_days} eval={args.eval_days} epochs={args.epochs}", flush=True)
    bundle = _bundle()

    # Day-0 env to derive shapes + the warm-start policy.
    env0, shim0 = _env(train_days[0], cfg, bundle, reward_overrides)
    obs_dim = int(shim0.obs_dim)
    runner_dim = int(env0.active_runner_dim)
    action_space = shim0.action_space
    print(f"obs_dim={obs_dim} runner_dim={runner_dim}", flush=True)

    policy = DiscreteLSTMPolicy(
        obs_dim=obs_dim, action_space=action_space, hidden_size=256,
        runner_dim=runner_dim, input_norm=True,
        mature_prob_open_threshold=args.mature_open_threshold,
    ).to(device)

    # ── Warm-start: load Step B weights + input-norm stats ──────────────
    ck = torch.load(args.warm_start, map_location="cpu", weights_only=False)
    policy.load_state_dict(ck["state_dict"])
    policy.set_input_norm_stats(ck["norm_mean"], ck["norm_std"])
    print(f"warm-started from {args.warm_start} (mature_prob_head AUC 0.745)", flush=True)

    # Pin the stake head to a constant + freeze (avoids the untrained-stake-
    # head cold-start: ~0.5 * budget = oversized stakes that can't post the
    # passive lay -> everything force-closes). The env decodes
    # stake = stake_unit * budget, so target_su * budget = the £ stake.
    if args.fixed_stake_unit > 0.0:
        import math
        su = float(args.fixed_stake_unit)
        alpha_t = 2.0
        beta_t = alpha_t * (1.0 / su - 1.0)  # so alpha/(alpha+beta) = su
        a_bias = math.log(math.expm1(alpha_t - 1.0))  # softplus^-1(alpha-1)
        b_bias = math.log(math.expm1(beta_t - 1.0))
        with torch.no_grad():
            policy.stake_alpha_head.weight.zero_(); policy.stake_alpha_head.bias.fill_(a_bias)
            policy.stake_beta_head.weight.zero_(); policy.stake_beta_head.bias.fill_(b_bias)
        for p in policy.stake_alpha_head.parameters():
            p.requires_grad_(False)
        for p in policy.stake_beta_head.parameters():
            p.requires_grad_(False)
        print(f"stake head PINNED: stake_unit={su} -> £{su*args.starting_budget:.0f}/open "
              f"@ £{args.starting_budget:.0f} budget (frozen)", flush=True)

    # Baseline holdout eval (warm-start, pre-PPO) for the delta.
    base_agg = None
    if eval_is_holdout and not args.skip_baseline_eval:
        print("\n=== BASELINE (warm-start, pre-PPO) holdout eval ===", flush=True)
        _, base_agg = _eval_holdout(policy, cfg, bundle, reward_overrides, device, "base")
        print(f"baseline: {base_agg}", flush=True)

    # ── BC the actor on oracle opportunities (mature head frozen) ───────
    if args.bc_actor_steps > 0:
        pos = load_oracle_samples_for_dates(train_days, DATA_DIR, obs_dim)
        neg = load_negative_samples_for_dates(train_days, DATA_DIR, obs_dim)
        print(f"\n=== BC actor: {len(pos)} oracle + {len(neg)} negative samples, "
              f"{args.bc_actor_steps} steps ===", flush=True)
        if pos:
            hist = DiscreteBCPretrainer(lr=args.lr, batch_size=64, seed=args.seed).pretrain(
                policy=policy, samples=pos, n_steps=args.bc_actor_steps,
                negative_samples=neg if neg else None, positive_weight=1.0,
            )
            print(f"BC actor done. final_ce={hist.final_ce_loss:.4f}", flush=True)
        else:
            print("no oracle samples — skipping actor BC", flush=True)

    # ── PPO trainer ─────────────────────────────────────────────────────
    trainer_hp = {
        "mature_prob_loss_weight": args.mature_prob_loss_weight,
        "fill_prob_loss_weight": 0.0,
        "risk_loss_weight": 0.0,
        "per_transition_credit": True,
        "mature_gate_warmup_eps": 5,
    }
    trainer = DiscretePPOTrainer(
        policy=policy, shim=shim0, learning_rate=args.lr,
        entropy_coeff=args.entropy_coeff,
        device=str(device),
        rollout_device="cpu" if device.type == "cuda" else str(device),
        hp=trainer_hp,
    )
    if args.bc_actor_steps > 0:
        try:
            trainer.set_post_bc_entropy(measure_post_bc_entropy(policy, pos[:256]))
        except Exception as e:  # noqa: BLE001
            print(f"(post-bc entropy skip: {e})", flush=True)

    # ── PPO training loop ───────────────────────────────────────────────
    print("\n=== PPO fine-tune ===", flush=True)
    trace = []
    t0 = time.time()
    ep = 0
    for epoch in range(args.epochs):
        for di, day in enumerate(train_days):
            if not (epoch == 0 and di == 0):
                env_n, shim_n = _env(day, cfg, bundle, reward_overrides)
                trainer.shim = shim_n
                trainer.action_space = shim_n.action_space
                trainer.max_runners = shim_n.max_runners
                trainer._collector = RolloutCollector(
                    shim=shim_n, policy=trainer.policy,
                    device=str(trainer.rollout_device))
                if hasattr(trainer, "_direction_label_cache"):
                    trainer._direction_label_cache.clear()
            stats = trainer.train_episode()
            li = getattr(trainer._collector, "last_info", {}) or {}
            po = int(li.get("pairs_opened", 0))
            comp = int(li.get("arbs_completed", 0))
            row = {
                "epoch": epoch, "day": day,
                "reward": round(float(stats.total_reward), 3),
                "day_pnl": round(float(stats.day_pnl), 2),
                "locked_pnl": round(float(li.get("locked_pnl", 0.0)), 2),
                "opened": po, "matured": comp,
                "mat_pct": round(100.0 * comp / po, 1) if po else 0.0,
                "policy_loss": round(float(stats.policy_loss_mean), 4),
                "mature_bce": round(float(getattr(stats, "mature_prob_bce_mean", 0.0)), 4),
                "kl": round(float(stats.approx_kl_mean), 4),
                "n_updates": int(getattr(stats, "n_updates_run", 0) or 0),
            }
            trace.append(row)
            print(f"  ep{ep:>2} [{day}] reward={row['reward']:+.1f} "
                  f"pnl={row['day_pnl']:+.1f} locked={row['locked_pnl']:+.1f} "
                  f"opened={po:>4} mat%={row['mat_pct']:>4.1f} "
                  f"kl={row['kl']:.3f} nupd={row['n_updates']}", flush=True)
            ep += 1
    print(f"PPO wall {time.time()-t0:.0f}s", flush=True)

    # ── Final holdout eval ──────────────────────────────────────────────
    if eval_is_holdout:
        print("\n=== POST-PPO holdout eval ===", flush=True)
        rows, agg = _eval_holdout(policy, cfg, bundle, reward_overrides, device, "ppo")
        print("\n==== BC->PPO CANARY HOLDOUT (fc=120, close_walk=10, barrier 0.50) ====", flush=True)
        if base_agg is not None:
            print(f"WARM-START: opened={base_agg['pairs_opened']} mat%={base_agg['mat_pct']} "
                  f"fc%={base_agg['fc_pct']} locked={base_agg['locked_pnl']:.2f} "
                  f"day_pnl={base_agg['day_pnl']:.2f}", flush=True)
        print(f"POST-PPO:   opened={agg['pairs_opened']} mat%={agg['mat_pct']} "
              f"fc%={agg['fc_pct']} locked={agg['locked_pnl']:.2f} "
              f"day_pnl={agg['day_pnl']:.2f}", flush=True)
        print("  (BC ref: 14-15% mat%, locked+; bar = LOCKED positive via LOWER fc%)", flush=True)
    else:
        # smoke: eval the given days with the same actor-driven rollout
        rows, agg = [], {}
        for day in _resolve_days(args.eval_days):
            env_e, shim_e = _env(day, cfg, bundle, reward_overrides)
            info = _eval_day(policy, shim_e, env_e, device)
            po = int(info.get("pairs_opened", 0)); comp = int(info.get("arbs_completed", 0))
            print(f"  [smoke-eval {day}] opened={po} matured={comp} "
                  f"locked={float(info.get('locked_pnl',0)):.2f} "
                  f"day_pnl={float(info.get('day_pnl',0)):.2f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "warm_start": args.warm_start, "train_days": train_days,
        "eval_days": args.eval_days, "epochs": args.epochs,
        "budget": args.starting_budget, "open_cost": args.open_cost,
        "maturation_reward_mode": args.maturation_reward_mode,
        "mature_prob_loss_weight": args.mature_prob_loss_weight,
        "mature_open_threshold": args.mature_open_threshold,
        "baseline_holdout": base_agg, "ppo_trace": trace,
        "holdout_agg": agg if eval_is_holdout else None,
        "holdout_rows": rows if eval_is_holdout else None,
    }, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
