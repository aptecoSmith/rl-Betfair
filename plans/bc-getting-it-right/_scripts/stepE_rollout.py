"""Step B3 / E (bc-getting-it-right): fully-hedged holdout rollout driven by
the SUPERVISED mature_prob_head, gated by a maturation threshold.

The §1-tertiary / §8.2 test. Selectivity comes from the mature_prob GATE,
not an actor knob (plan §4): at each tick the policy opens OPEN_BACK on the
highest-``mature_prob`` LEGAL runner iff its mature_prob >= threshold; else
NOOP. One open per tick (oracle-style); the env auto-pairs the passive lay,
force-closes unfilled legs at T-120, and close_walk=10 hedges. Maturation =
the passive fills naturally before the off.

This isolates "does the 0.72-AUC mature_prob ranking translate into a
selective, locked-positive rollout?" with no random-actor confound — and it
IS a deployable policy (the gate + greedy-best-gated-open is exactly what a
threshold-gated actor converges to).

Loads a policy saved by ``mature_head_bc.py --save-policy`` (state_dict +
norm stats + obs_dim). Sweeps one or more thresholds; reports per-day +
aggregate mat% / locked / day_pnl / fc% / opens, fc=120 + close_walk=10.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mat_metric as M  # noqa: E402
from agents_v2.action_space import ActionType  # noqa: E402
from agents_v2.env_shim import DEFAULT_SCORER_DIR  # noqa: E402
from env.bet_manager import MIN_BET_STAKE, BetSide  # noqa: E402
from training_v2.arb_oracle import _load_config  # noqa: E402
from training_v2.cohort.worker import _build_env_for_day  # noqa: E402

DATA_DIR = Path("data/processed")
PRED = (
    "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json",
    "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json",
    "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json",
)
RO = {"force_close_before_off_seconds": 120.0, "close_walk_ticks": 10.0}


def _bundle():
    from predictors import PredictorBundle
    c, r, d = PRED
    return PredictorBundle.from_manifests(
        champion_manifest=Path(c), ranker_manifest=Path(r),
        direction_manifest=Path(d),
    )


def _env(date, cfg, bundle, reward_overrides=None):
    return _build_env_for_day(
        day_str=date, data_dir=DATA_DIR, cfg=cfg, scorer_dir=DEFAULT_SCORER_DIR,
        reward_overrides=reward_overrides if reward_overrides is not None else RO,
        predictor_bundle=bundle,
        use_race_outcome_predictor=True, use_direction_predictor=True,
        predictor_lean_obs=False, emit_debug_features=False,
    )


@torch.no_grad()
def _eval_day_greedy(policy, shim, env, space, device, threshold, stake_gbp):
    """Greedy-by-mature_prob open policy. One open/tick on the best legal
    runner clearing ``threshold``; else NOOP. No agent close (env hedges)."""
    policy.eval()
    obs, _ = shim.reset()
    hidden = policy.init_hidden(batch=1)
    hidden = tuple(t.to(device) for t in hidden)
    R = space.max_runners
    noop_idx = int(space.encode(ActionType.NOOP, None))
    done = False
    info = {}
    while not done:
        mask = shim.get_action_mask()
        obs_t = torch.tensor(np.asarray(obs), dtype=torch.float32,
                             device=device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        out = policy(obs_t, hidden_state=hidden, mask=mask_t)
        hidden = out.new_hidden_state
        mature = out.mature_prob_per_runner[0].detach().cpu().numpy()  # (R,)
        # Among runners whose OPEN_BACK is legal AND mature_prob >= T,
        # pick the highest mature_prob.
        best_i, best_p = -1, -1.0
        for i in range(R):
            ob_idx = int(space.encode(ActionType.OPEN_BACK, i))
            if not mask[ob_idx]:
                continue
            p = float(mature[i])
            if p >= threshold and p > best_p:
                best_p, best_i = p, i
        if best_i >= 0:
            action_idx = int(space.encode(ActionType.OPEN_BACK, best_i))
        else:
            action_idx = noop_idx
        obs, _r, term, trunc, info = shim.step(action_idx, stake=stake_gbp)
        done = bool(term or trunc)
    return info


def _sum_back_stake(env) -> float:
    """Total matched BACK stake over the day (capital put up on back legs)."""
    bets = getattr(env, "all_settled_bets", None) or []
    return float(sum(b.matched_stake for b in bets if b.side is BetSide.BACK))


def _rollout_threshold(policy, cfg, bundle, space, device, holdout, threshold,
                       stake_gbp, reward_overrides=None):
    rows = []
    for date in holdout:
        env, shim = _env(date, cfg, bundle, reward_overrides)
        info = _eval_day_greedy(policy, shim, env, space, device, threshold,
                                stake_gbp)
        po = int(info.get("pairs_opened", 0))
        comp = int(info.get("arbs_completed", 0))
        fc = int(info.get("arbs_force_closed", 0))
        sc = int(info.get("arbs_stop_closed", info.get("scalping_arbs_stop_closed", 0)))
        # Deployment-economics telemetry (what the model actually SPENT):
        # peak_open_liability = max simultaneous capital-at-risk in a single
        # race; budget_lay = # passives the per-race budget BLOCKED from
        # posting (the confound size). total back staked ~= matched back legs.
        rejects = info.get("paired_place_rejects", {}) or {}
        back_staked = round(float(_sum_back_stake(env)), 2)
        rows.append({
            "date": date, "pairs_opened": po, "arbs_completed": comp,
            "arbs_force_closed": fc,
            "day_pnl": round(float(info.get("day_pnl", 0.0)), 2),
            "locked_pnl": round(float(info.get("locked_pnl", 0.0)), 2),
            "arbs_stop_closed": sc,
            "mat_pct": round(100.0 * comp / po, 2) if po else 0.0,
            "fc_pct": round(100.0 * fc / po, 2) if po else 0.0,
            "sc_pct": round(100.0 * sc / po, 2) if po else 0.0,
            "peak_open_liability": round(float(info.get("peak_open_liability", 0.0)), 2),
            "peak_drawdown": round(float(info.get("peak_drawdown_from_high", 0.0)), 2),
            "budget_lay_rejects": int(rejects.get("budget_lay", 0)),
            "back_staked": back_staked,
        })
        print(f"  [T={threshold:.2f} {date}] opened={po:>4} "
              f"locked={rows[-1]['locked_pnl']:>7.2f} "
              f"day_pnl={rows[-1]['day_pnl']:>8.2f} mat%={rows[-1]['mat_pct']:>5.1f} "
              f"fc%={rows[-1]['fc_pct']:>5.1f} sc%={rows[-1]['sc_pct']:>5.1f} | "
              f"peak_liab={rows[-1]['peak_open_liability']:>7.2f} "
              f"blocks={rows[-1]['budget_lay_rejects']:>4}", flush=True)
    agg = {k: sum(r[k] for r in rows) for k in
           ("pairs_opened", "arbs_completed", "arbs_force_closed",
            "arbs_stop_closed", "day_pnl", "locked_pnl", "budget_lay_rejects",
            "back_staked")}
    po = agg["pairs_opened"]
    agg["mat_pct"] = round(100.0 * agg["arbs_completed"] / po, 2) if po else 0.0
    agg["fc_pct"] = round(100.0 * agg["arbs_force_closed"] / po, 2) if po else 0.0
    agg["sc_pct"] = round(100.0 * agg["arbs_stop_closed"] / po, 2) if po else 0.0
    # Peak liability is a MAX, not a sum (it's the worst single-race
    # capital-at-risk over the week).
    agg["peak_open_liability_max"] = round(
        max((r["peak_open_liability"] for r in rows), default=0.0), 2)
    agg["peak_open_liability_mean"] = round(
        sum(r["peak_open_liability"] for r in rows) / max(len(rows), 1), 2)
    agg["threshold"] = threshold
    return rows, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True,
                    help="path to a .pt saved by mature_head_bc.py --save-policy")
    ap.add_argument("--thresholds", default="0.3,0.4,0.5",
                    help="comma-separated mature_prob open thresholds to sweep")
    ap.add_argument("--holdout-days", default="holdout")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stake", type=float, default=10.0)
    ap.add_argument("--starting-budget", type=float, default=0.0,
                    help="If >0, override cfg training.starting_budget. Use a "
                         "large value (e.g. 100000) to ISOLATE the mature_prob "
                         "signal from the per-race budget confound (passives "
                         "cannot post under budget, go naked then force-close, "
                         "inflating force-close rate independent of the "
                         "ranking). Matches imitation-first Step 0.5 control.")
    ap.add_argument("--fc-max-dev", default="config",
                    help="force_close_max_deviation_pct for the run. 'config' "
                         "= use config.yaml default (0.50, the barrier). A "
                         "float = pin that value via reward_overrides. 'off' "
                         "= disable the barrier (None) to reproduce the legacy "
                         "unguarded force-close (the pre-fix rollout).")
    ap.add_argument("--stop-loss", type=float, default=0.0,
                    help="stop_loss_pnl_threshold (£). >0 = cut a pair mid-race "
                         "via the strict matcher when its per-pair MTM crosses "
                         "-this, instead of riding it to the T-120 force-close. "
                         "0 = off (ride to force-close, the default behaviour).")
    ap.add_argument("--out", default="plans/bc-getting-it-right/_scripts/stepE_rollout.json")
    args = ap.parse_args()

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available())
        else "cpu")
    holdout = M.resolve_days(args.holdout_days)
    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]

    ckpt = torch.load(args.policy, map_location="cpu", weights_only=False)
    obs_dim = int(ckpt["obs_dim"])
    hidden_size = int(ckpt["hidden_size"])
    policy, space = M.build_policy(
        obs_dim, hidden_size, device,
        norm_mean=ckpt["norm_mean"], norm_std=ckpt["norm_std"])
    policy.load_state_dict(ckpt["state_dict"])
    print(f"loaded policy obs_dim={obs_dim} hidden={hidden_size} "
          f"thresholds={thresholds} stake=£{args.stake}", flush=True)

    cfg = _load_config()
    if args.starting_budget > 0.0:
        import copy
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("training", {})["starting_budget"] = float(
            args.starting_budget)
        print(f"OVERRIDE starting_budget = £{args.starting_budget:.0f} "
              f"(signal-isolation control)", flush=True)
    bundle = _bundle()

    # Build reward_overrides: start from RO (fc=120, close_walk=10), then
    # set the force-close deviation barrier per --fc-max-dev.
    reward_overrides = dict(RO)
    if args.fc_max_dev == "config":
        fc_desc = "config default (0.50 barrier)"
        # leave out of overrides → env reads betting_constraints (0.50)
    elif args.fc_max_dev == "off":
        reward_overrides["force_close_max_deviation_pct"] = None
        fc_desc = "OFF (legacy unguarded force-close)"
    else:
        reward_overrides["force_close_max_deviation_pct"] = float(args.fc_max_dev)
        fc_desc = f"{float(args.fc_max_dev)} (pinned)"
    print(f"force_close_max_deviation_pct = {fc_desc}", flush=True)
    if args.stop_loss > 0.0:
        reward_overrides["stop_loss_pnl_threshold"] = float(args.stop_loss)
        print(f"stop_loss_pnl_threshold = £{args.stop_loss} (cut losers mid-race)",
              flush=True)

    all_aggs = []
    for t in thresholds:
        print(f"\n=== rollout threshold T={t:.2f} ===", flush=True)
        rows, agg = _rollout_threshold(policy, cfg, bundle, space, device,
                                       holdout, t, args.stake, reward_overrides)
        print(f"  AGG T={t:.2f}: opened={agg['pairs_opened']} "
              f"mat%={agg['mat_pct']} fc%={agg['fc_pct']} "
              f"locked={agg['locked_pnl']:.2f} day_pnl={agg['day_pnl']:.2f}",
              flush=True)
        all_aggs.append({"agg": agg, "per_day": rows})

    print("\n==== STEP E HOLDOUT ROLLOUT SWEEP (fc=120, close_walk=10) ====",
          flush=True)
    print("  vs imitation-first BC: 4% mat%, -£1513/7d ; random ~1% mat%",
          flush=True)
    for a in all_aggs:
        g = a["agg"]
        print(f"  T={g['threshold']:.2f}: opened={g['pairs_opened']:>5} "
              f"mat%={g['mat_pct']:>5.1f} fc%={g['fc_pct']:>5.1f} "
              f"locked={g['locked_pnl']:>8.2f} day_pnl={g['day_pnl']:>9.2f} "
              f"| back_staked={g['back_staked']:>9.2f} "
              f"peak_liab(max/mean per-race)={g['peak_open_liability_max']:.0f}/"
              f"{g['peak_open_liability_mean']:.0f} "
              f"budget_blocks={g['budget_lay_rejects']:>5}", flush=True)

    Path(args.out).write_text(json.dumps({
        "policy": args.policy, "thresholds": thresholds,
        "holdout": holdout, "stake": args.stake, "results": all_aggs,
    }, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
