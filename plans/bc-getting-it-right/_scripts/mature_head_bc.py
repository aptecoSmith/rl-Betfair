"""Step B (bc-getting-it-right): hard-negative mature_prob BCE supervision.

THE LEAD LEVER (plan §3/§4). Train the policy's ``mature_prob_head`` with
BCE on the HARD split — positive = matured candidate, hard-negative =
placeable-but-force-close — directly porting the LightGBM-0.76 signal INTO
the policy. Then measure held-out maturation AUC (the §1 metric) vs 0.76.

Mirrors the trainer's clamped post-sigmoid BCE on
``mature_prob_per_runner`` (training_v2/discrete_ppo/trainer.py
``_compute_per_transition_mature_loss``) so the recipe transfers cleanly
to the cohort path later.

Two regimes (``--trainable``):
  full : train the whole network on the mature BCE objective — the
         decisive "can this architecture represent the signal" upper bound.
  head : freeze everything except mature_prob_head (cheap first cut over a
         random frozen backbone; expected weaker — the head reads a random
         256-d compression of the 2254-d obs).

Input-norm ON (plan §5); stats from the TRAIN cache only. Early-stop on
val-day AUC (2 held-out TRAIN days — never the holdout). Holdout AUC +
PR curve reported at the end; trained policy state_dict saved for Step E.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))      # _scripts dir

import mat_metric as M  # noqa: E402


def _train_target_names(trainable: str) -> tuple[str, ...] | None:
    """Return the parameter-name substrings BC may update, or None for all."""
    if trainable == "full":
        return None
    if trainable == "head":
        return ("mature_prob_head",)
    if trainable == "head+backbone":
        return ("mature_prob_head", "input_proj", "lstm")
    raise ValueError(f"unknown --trainable {trainable!r}")


def _bce(pred, label, pos_weight: float):
    """Clamped post-sigmoid BCE (matches trainer). Optional pos_weight on the
    positive class to counter the ~12% base rate."""
    eps = 1e-7
    pred_c = pred.clamp(eps, 1.0 - eps)
    bce = -(
        label * torch.log(pred_c)
        + (1.0 - label) * torch.log(1.0 - pred_c)
    )
    if pos_weight != 1.0:
        w = torch.where(label > 0.5, pos_weight, 1.0)
        return (w * bce).mean()
    return bce.mean()


@torch.no_grad()
def _holdout_auc(policy, obs, y, ri, device, batch):
    s = M.policy_mature_scores(policy, obs, ri, device, batch=batch)
    return M.evaluate(s, y), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(M.DEFAULT_CACHE))
    ap.add_argument("--train-days", default="train6")
    ap.add_argument("--val-days", default="val2")
    ap.add_argument("--holdout-days", default="holdout")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--trainable", default="full",
                    choices=["full", "head", "head+backbone"])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--chunk-steps", type=int, default=200)
    ap.add_argument("--max-chunks", type=int, default=60)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-batch", type=int, default=16384)
    ap.add_argument("--save-policy", default="")
    ap.add_argument("--out", default="plans/bc-getting-it-right/_scripts/stepB_result.json")
    args = ap.parse_args()

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available())
        else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_days = M.resolve_days(args.train_days)
    val_days = M.resolve_days(args.val_days)
    holdout_days = M.resolve_days(args.holdout_days)
    cache = Path(args.cache_dir)
    print(f"device={device} trainable={args.trainable} "
          f"pos_weight={args.pos_weight}", flush=True)
    print(f"train_days={train_days}", flush=True)
    print(f"val_days={val_days}", flush=True)
    print(f"holdout_days={holdout_days}", flush=True)

    print("loading splits...", flush=True)
    Xtr, ytr, ritr = M.load_split(train_days, cache)
    Xva, yva, riva = M.load_split(val_days, cache)
    Xte, yte, rite = M.load_split(holdout_days, cache)
    print(f"train n={len(ytr)} ({ytr.mean():.4f})  "
          f"val n={len(yva)} ({yva.mean():.4f})  "
          f"holdout n={len(yte)} ({yte.mean():.4f})  obs_dim={Xtr.shape[1]}",
          flush=True)

    mean, std = M.compute_norm_stats(Xtr)
    M.save_norm_stats(mean, std, cache / "norm_stats.npz")
    policy, space = M.build_policy(
        Xtr.shape[1], args.hidden_size, device,
        norm_mean=mean, norm_std=std, seed=args.seed)

    # Freeze per the trainable regime.
    targets = _train_target_names(args.trainable)
    if targets is None:
        train_params = list(policy.parameters())
    else:
        train_params = []
        for name, p in policy.named_parameters():
            if any(t in name for t in targets):
                train_params.append(p)
            else:
                p.requires_grad_(False)
    n_train_params = sum(p.numel() for p in train_params)
    print(f"trainable params: {n_train_params:,}", flush=True)

    # Baseline (untrained head) holdout AUC for the delta.
    m0, _ = _holdout_auc(policy, Xte, yte, rite, device, args.eval_batch)
    print(f"UNTRAINED holdout: AUC={m0['auc']} (base {m0['base_rate']})",
          flush=True)

    opt = torch.optim.Adam(train_params, lr=args.lr,
                           weight_decay=args.weight_decay)
    ytr_t_full = ytr.astype(np.float32)

    best_auc, best_state, patience = -1.0, None, args.patience
    trace = []
    n = len(ytr)
    print("\n=== mature_prob BCE supervision ===", flush=True)
    t0 = time.time()
    for chunk in range(args.max_chunks):
        policy.train()
        last = 0.0
        for _ in range(args.chunk_steps):
            bi = rng.integers(0, n, size=args.batch_size)
            ob = torch.from_numpy(Xtr[bi]).to(device=device, dtype=torch.float32)
            rib = torch.from_numpy(ritr[bi].astype(np.int64)).to(device)
            lab = torch.from_numpy(ytr_t_full[bi]).to(device)
            out = policy(ob)
            mp = out.mature_prob_per_runner
            rows = torch.arange(len(bi), device=device)
            pred = mp[rows, rib]
            loss = _bce(pred, lab, args.pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.item())
        mva, _ = _holdout_auc(policy, Xva, yva, riva, device, args.eval_batch)
        trace.append({"chunk": chunk, "steps": (chunk + 1) * args.chunk_steps,
                      "train_loss": round(last, 4), "val_auc": mva["auc"]})
        print(f"  chunk {chunk:>2} steps={(chunk+1)*args.chunk_steps:>6} "
              f"loss={last:.4f} val_auc={mva['auc']:.4f}", flush=True)
        if mva["auc"] > best_auc + 1e-4:
            best_auc = mva["auc"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in policy.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                print(f"  early stop at chunk {chunk}", flush=True)
                break
    if best_state is not None:
        policy.load_state_dict(best_state)
    print(f"train wall {time.time()-t0:.0f}s  best_val_auc={best_auc:.4f}",
          flush=True)

    # Final holdout metric (the §1 north star).
    mte, s_te = _holdout_auc(policy, Xte, yte, rite, device, args.eval_batch)
    pr = M.pr_curve(s_te, yte)
    print("\n==== STEP B HOLDOUT MATURATION AUC ====", flush=True)
    M._print_eval("BC-supervised head", mte)
    print(f"  (LightGBM reference = 0.7592;  untrained = {m0['auc']})",
          flush=True)
    print("  PR curve (threshold -> opens, mat%, recall, lift):", flush=True)
    for r in pr:
        print(f"    t={r['threshold']:.2f} opens={r['opens']:>6} "
              f"mat%={r['mat_pct']:>5.1f} recall={r['recall']:.3f} "
              f"lift={r['lift']:.2f}", flush=True)

    if args.save_policy:
        Path(args.save_policy).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": policy.state_dict(),
                    "obs_dim": int(Xtr.shape[1]),
                    "hidden_size": args.hidden_size,
                    "norm_mean": mean, "norm_std": std}, args.save_policy)
        print(f"saved policy -> {args.save_policy}", flush=True)

    Path(args.out).write_text(json.dumps({
        "trainable": args.trainable, "pos_weight": args.pos_weight,
        "lr": args.lr, "hidden_size": args.hidden_size,
        "train_days": train_days, "val_days": val_days,
        "holdout_days": holdout_days,
        "untrained_holdout": m0, "best_val_auc": best_auc,
        "holdout": mte, "pr_curve": pr, "trace": trace,
        "lightgbm_reference_auc": 0.7592,
    }, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
