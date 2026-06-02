"""Force-close fill-price forensic (bc-getting-it-right, operator ask).

Worry: force-closes use the RELAXED matcher (LTP requirement + ±50% junk
filter SKIPPED) and the config has ``max_lay_price: null`` — so a
force-close LAY (the leg that flattens a back-first scalp) has NO
upper-price protection. Are we crossing the near-off book at ridiculous
prices?

This runs the greedy-by-mature_prob rollout on a few holdout days, pulls
every ``force_close=True`` leg from ``env.all_settled_bets``, and tabulates
its fill ``average_price`` against ``ltp_at_placement`` (the fair reference
the env recorded at fill time). Reports the ratio distribution, the worst
fills with full context, and the £ "excess cost" of crossing far from LTP.

No env changes — pure post-hoc read of the settled bets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mat_metric as M  # noqa: E402
import stepE_rollout as SE  # noqa: E402
from env.bet_manager import BetSide  # noqa: E402
from training_v2.arb_oracle import _load_config  # noqa: E402


def _fc_rows(env):
    """Extract force-close legs with fill-vs-fair context."""
    bets = getattr(env, "all_settled_bets", None) or []
    rows = []
    for b in bets:
        if not getattr(b, "force_close", False):
            continue
        side = "LAY" if b.side is BetSide.LAY else "BACK"
        ltp = float(getattr(b, "ltp_at_placement", 0.0) or 0.0)
        price = float(b.average_price)
        stake = float(b.matched_stake)
        liab = stake * (price - 1.0) if b.side is BetSide.LAY else stake
        # ratio of fill price to fair LTP. >1 for a LAY = crossed UP (bad,
        # paying excess liability); <1 for a BACK close = backed too low.
        ratio = (price / ltp) if ltp > 0 else float("nan")
        # £ excess vs filling AT ltp (the spread we crossed):
        #   LAY: extra liability  = stake*(price - ltp)
        #   BACK: lost stake-value= stake*(ltp - price)/ltp (approx)
        if b.side is BetSide.LAY:
            excess = stake * (price - ltp) if ltp > 0 else float("nan")
        else:
            excess = stake * (ltp - price) if ltp > 0 else float("nan")
        rows.append({
            "side": side, "stake": stake, "price": price, "ltp": ltp,
            "ratio": ratio, "liability": liab, "pnl": float(b.pnl),
            "excess": excess, "pair_id": getattr(b, "pair_id", None),
            "tick_index": int(getattr(b, "tick_index", -1)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--days", default="2026-05-20,2026-05-29")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stake", type=float, default=10.0)
    ap.add_argument("--starting-budget", type=float, default=0.0)
    ap.add_argument("--ridiculous-ratio", type=float, default=1.5,
                    help="flag LAY closes filling at >this multiple of LTP")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available())
        else "cpu")
    days = [d.strip() for d in args.days.split(",") if d.strip()]

    ck = torch.load(args.policy, map_location="cpu", weights_only=False)
    policy, space = M.build_policy(
        int(ck["obs_dim"]), int(ck["hidden_size"]), device,
        norm_mean=ck["norm_mean"], norm_std=ck["norm_std"])
    policy.load_state_dict(ck["state_dict"])

    cfg = _load_config()
    if args.starting_budget > 0.0:
        import copy
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("training", {})["starting_budget"] = float(args.starting_budget)
    betting = cfg.get("training", {}).get("betting_constraints", {})
    print(f"config max_back_price={betting.get('max_back_price')} "
          f"max_lay_price={betting.get('max_lay_price')}", flush=True)
    print(f"budget={args.starting_budget or 'default(£100)'} "
          f"threshold={args.threshold} days={days}", flush=True)
    bundle = SE._bundle()

    all_rows = []
    for date in days:
        env, shim = SE._env(date, cfg, bundle)
        info = SE._eval_day_greedy(policy, shim, env, space, device,
                                   args.threshold, args.stake)
        rows = _fc_rows(env)
        all_rows.extend(rows)
        po = int(info.get("pairs_opened", 0))
        fc = int(info.get("arbs_force_closed", 0))
        print(f"  [{date}] pairs_opened={po} arbs_force_closed={fc} "
              f"force_close_legs={len(rows)}", flush=True)

    if not all_rows:
        print("NO force-close legs found.", flush=True)
        return

    lay = [r for r in all_rows if r["side"] == "LAY" and r["ltp"] > 0]
    back = [r for r in all_rows if r["side"] == "BACK" and r["ltp"] > 0]
    print(f"\n=== {len(all_rows)} force-close legs "
          f"({len(lay)} LAY, {len(back)} BACK, "
          f"{len(all_rows)-len(lay)-len(back)} missing-LTP) ===", flush=True)

    def _dist(rows, label):
        if not rows:
            return
        ratios = np.array([r["ratio"] for r in rows])
        prices = np.array([r["price"] for r in rows])
        ltps = np.array([r["ltp"] for r in rows])
        excess = np.array([r["excess"] for r in rows])
        print(f"\n-- {label} closes (n={len(rows)}) fill-price / LTP ratio --",
              flush=True)
        for q in (50, 75, 90, 95, 99, 100):
            print(f"   p{q:>3}: ratio={np.percentile(ratios, q):.3f}  "
                  f"price={np.percentile(prices, q):.2f}  "
                  f"ltp={np.percentile(ltps, q):.2f}", flush=True)
        rid = [r for r in rows if r["side"] == "LAY" and r["ratio"] > args.ridiculous_ratio]
        print(f"   LAY closes > {args.ridiculous_ratio}x LTP: {len(rid)} "
              f"({100.0*len(rid)/max(len(rows),1):.1f}%)  "
              f"excess £ from crossing = {excess[excess>0].sum():.2f} "
              f"(total over all {label} closes)", flush=True)

    _dist(lay, "LAY")
    _dist(back, "BACK")

    # Worst fills by ratio (LAY) / inverse-ratio (BACK).
    worst = sorted(
        [r for r in all_rows if r["ltp"] > 0],
        key=lambda r: (r["ratio"] if r["side"] == "LAY" else 1.0 / max(r["ratio"], 1e-6)),
        reverse=True)[: args.top]
    print(f"\n=== top {args.top} most-extreme fills ===", flush=True)
    print(f"  {'side':4} {'stake':>6} {'fill_px':>8} {'ltp':>7} {'ratio':>6} "
          f"{'liab':>8} {'pnl':>8} {'excess£':>8}", flush=True)
    for r in worst:
        print(f"  {r['side']:4} {r['stake']:>6.2f} {r['price']:>8.2f} "
              f"{r['ltp']:>7.2f} {r['ratio']:>6.2f} {r['liability']:>8.2f} "
              f"{r['pnl']:>8.2f} {r['excess']:>8.2f}", flush=True)

    tot_excess = sum(r["excess"] for r in all_rows
                     if r["ltp"] > 0 and np.isfinite(r["excess"]))
    tot_fc_pnl = sum(r["pnl"] for r in all_rows)
    print(f"\nTOTAL force-close legs: {len(all_rows)}", flush=True)
    print(f"TOTAL excess-vs-LTP £ (spread crossed): {tot_excess:.2f}", flush=True)
    print(f"TOTAL force-close settled pnl: {tot_fc_pnl:.2f}", flush=True)


if __name__ == "__main__":
    main()
