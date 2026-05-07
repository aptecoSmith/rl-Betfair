"""Cross-validation threshold sweep — train on N days, eval on M held-out
days, sweep thresholds up to 0.95.

Phase-13 follow-up to ``direction_threshold_sweep_oos.py``. Two
extensions:

1. Multi-day TRAIN + multi-day EVAL. Concatenate features across N
   training days for a richer fit, so the OOS calibration isn't
   dominated by a single day's idiosyncrasies. Then evaluate
   independently on each held-out day.

2. Threshold sweep up to 0.95 in 0.05 steps. The in-sample top bin
   reached P=0.83; the OOS sweep capped at 0.70 may have missed
   genuine high-confidence signal at the upper extreme.

Usage::

    python -m tools.direction_threshold_sweep_xval \\
        --train-dates 2026-05-01,2026-05-02,2026-05-03 \\
        --eval-dates 2026-04-30,2026-05-04 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from training_v2.direction_label_scan import load_labels
from tools.direction_features_probe import (
    _MLP,
    _per_row_label_lookup,
    build_features,
)


def _build_filtered(date: str, data_dir: Path, args):
    base_arr, aug_arr, tick_arr, runner_arr, _sid = build_features(
        date, data_dir,
    )
    labels = load_labels(
        date, data_dir,
        direction_horizon_ticks=args.horizon_ticks,
        direction_threshold_ticks=args.threshold_ticks,
        force_close_before_off_seconds=args.fc_secs,
        strict=True,
    )
    label_back, label_lay, mask = _per_row_label_lookup(
        labels, tick_arr, runner_arr,
    )
    sel = np.where(mask)[0]
    base_s = base_arr[sel]
    aug_s = aug_arr[sel]
    feats = np.concatenate([base_s, aug_s], axis=1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats, label_back[sel], label_lay[sel]


def _calibration_table(p, y, n_bins=10):
    sidx = np.argsort(p)
    chunks = np.array_split(sidx, n_bins)
    return [
        (float(p[c].mean()), float(y[c].mean()), len(c))
        for c in chunks
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dates", required=True,
                    help="Comma-separated list, e.g. 2026-05-01,2026-05-02,2026-05-03.")
    ap.add_argument("--eval-dates", required=True,
                    help="Comma-separated list, must NOT overlap with train.")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--horizon-ticks", type=int, default=60)
    ap.add_argument("--threshold-ticks", type=int, default=5)
    ap.add_argument("--fc-secs", type=float, default=60.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p-locked", type=float, default=2.50)
    ap.add_argument("--p-loss", type=float, default=3.00)
    args = ap.parse_args()

    train_dates = [d.strip() for d in args.train_dates.split(",")]
    eval_dates = [d.strip() for d in args.eval_dates.split(",")]
    overlap = set(train_dates) & set(eval_dates)
    if overlap:
        raise SystemExit(
            f"train and eval dates overlap: {sorted(overlap)} — refusing"
        )

    data_dir = Path(args.data_dir)
    device = args.device

    print(f"=== TRAIN: {train_dates} ===")
    train_feats_all = []
    train_lback_all = []
    train_llay_all = []
    for d in train_dates:
        f, lb, ll = _build_filtered(d, data_dir, args)
        print(f"  {d}: rows={f.shape[0]}  pos_back={lb.mean():.4f}  "
              f"pos_lay={ll.mean():.4f}")
        train_feats_all.append(f)
        train_lback_all.append(lb)
        train_llay_all.append(ll)

    t_feats = np.concatenate(train_feats_all, axis=0)
    t_lback = np.concatenate(train_lback_all)
    t_llay = np.concatenate(train_llay_all)
    n_train = t_feats.shape[0]
    pos_back = float(t_lback.mean())
    pos_lay = float(t_llay.mean())
    print(f"  pooled N_train={n_train}  pos_back={pos_back:.4f}  "
          f"pos_lay={pos_lay:.4f}")

    mu = t_feats.mean(axis=0, keepdims=True)
    sigma = t_feats.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    t_feats_z = (t_feats - mu) / sigma
    t_feats_z = np.nan_to_num(t_feats_z, 0.0, 0.0, 0.0)

    print(f"\n=== training MLP on pooled {len(train_dates)} day(s), "
          f"steps={args.steps} ===")
    torch.manual_seed(int(args.seed))
    in_dim = t_feats_z.shape[1]
    model = _MLP(in_dim=in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    pw_back = torch.tensor(
        (1 - pos_back) / max(pos_back, 1e-6),
        dtype=torch.float32, device=device,
    )
    pw_lay = torch.tensor(
        (1 - pos_lay) / max(pos_lay, 1e-6),
        dtype=torch.float32, device=device,
    )
    feats_t = torch.from_numpy(t_feats_z.astype(np.float32)).to(device)
    lback_t = torch.from_numpy(t_lback).to(device)
    llay_t = torch.from_numpy(t_llay).to(device)
    rng = np.random.default_rng(int(args.seed))
    t0 = time.monotonic()
    for step in range(args.steps):
        idx = rng.integers(0, n_train, size=args.batch_size)
        idx_t = torch.from_numpy(idx).to(device)
        x = feats_t.index_select(0, idx_t)
        out = model(x)
        bback = nn.functional.binary_cross_entropy_with_logits(
            out[:, 0], lback_t.index_select(0, idx_t),
            pos_weight=pw_back,
        )
        blay = nn.functional.binary_cross_entropy_with_logits(
            out[:, 1], llay_t.index_select(0, idx_t),
            pos_weight=pw_lay,
        )
        (bback + blay).backward()
        opt.step()
        opt.zero_grad()
        if step % 500 == 0:
            print(f"  step={step}  bce_back={bback.item():.4f}  "
                  f"bce_lay={blay.item():.4f}")
    print(f"  train wall = {time.monotonic() - t0:.1f}s, "
          f"final bce_back={bback.item():.4f} bce_lay={blay.item():.4f}")

    # ── Per-eval-day analysis ──────────────────────────────────────────
    thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70,
                  0.75, 0.80, 0.85, 0.90, 0.95]
    summary_rows = []

    for ed in eval_dates:
        print(f"\n=== EVAL day {ed} ===")
        e_feats, e_lback, e_llay = _build_filtered(ed, data_dir, args)
        n_eval = e_feats.shape[0]
        e_pb = float(e_lback.mean())
        e_pl = float(e_llay.mean())
        print(f"  N_eval={n_eval}  pos_back={e_pb:.4f}  pos_lay={e_pl:.4f}")
        e_feats_z = (e_feats - mu) / sigma
        e_feats_z = np.nan_to_num(e_feats_z, 0.0, 0.0, 0.0)
        e_t = torch.from_numpy(e_feats_z.astype(np.float32)).to(device)
        with torch.no_grad():
            out = model(e_t)
            p_back = torch.sigmoid(out[:, 0]).cpu().numpy()
            p_lay = torch.sigmoid(out[:, 1]).cpu().numpy()

        # Calibration on each side independently — 10 bins so the high-
        # confidence tail is visible.
        cb = _calibration_table(p_back, e_lback, n_bins=10)
        cl = _calibration_table(p_lay, e_llay, n_bins=10)
        print(f"  back-side calibration (10 bins):")
        for i, (pb, rb, n) in enumerate(cb):
            print(f"    decile {i}: P_pred={pb:.3f} P_real={rb:.3f}  n={n}")
        print(f"  lay-side  calibration (10 bins):")
        for i, (pl, rl, n) in enumerate(cl):
            print(f"    decile {i}: P_pred={pl:.3f} P_real={rl:.3f}  n={n}")
        lift_back = cb[-1][1] / max(cb[0][1], 1e-6)
        lift_lay = cl[-1][1] / max(cl[0][1], 1e-6)
        print(f"  Top-vs-bottom decile lift: back {lift_back:.2f}×  "
              f"lay {lift_lay:.2f}×")

        # Per-row decision = argmax side; threshold sweep.
        chosen_back = p_back >= p_lay
        chosen_conf = np.where(chosen_back, p_back, p_lay)
        chosen_label = np.where(chosen_back, e_lback, e_llay)
        print(f"\n  threshold sweep — P_locked=£{args.p_locked}, "
              f"P_loss=£{args.p_loss}")
        print(f"  {'thresh':>7} {'n_opens':>9} {'%rows':>7} "
              f"{'mature':>8} {'force':>7} "
              f"{'pnl/open':>10} {'day_pnl':>10}")
        print('  ' + '-' * 72)
        best_pnl = -1e18
        best_T = None
        best_mature = None
        best_n = None
        for T in thresholds:
            gate = chosen_conf >= T
            n_opens = int(gate.sum())
            if n_opens < 50:
                # Too few rows for a meaningful read.
                print(f"  {T:>7.2f} {n_opens:>9d} {n_opens/n_eval*100:>5.2f}%"
                      f"  (n<50, skipping)")
                continue
            gated_label = chosen_label[gate]
            mature = float(gated_label.mean())
            force = 1.0 - mature
            pnl_per_open = mature * args.p_locked - force * args.p_loss
            day_pnl = pnl_per_open * n_opens
            print(f"  {T:>7.2f} {n_opens:>9d} {n_opens/n_eval*100:>5.2f}% "
                  f"{mature:>8.4f} {force:>7.4f} "
                  f"{pnl_per_open:>+10.3f} {day_pnl:>+10.1f}")
            if pnl_per_open > best_pnl:
                best_pnl = pnl_per_open
                best_T = T
                best_mature = mature
                best_n = n_opens
        summary_rows.append({
            "eval_day": ed,
            "n_eval": n_eval,
            "lift_back": lift_back,
            "lift_lay": lift_lay,
            "best_T": best_T,
            "best_mature": best_mature,
            "best_pnl_per_open": best_pnl,
            "best_n_opens": best_n,
        })

    # ── Cross-day summary ──────────────────────────────────────────────
    print(f"\n\n=== cross-day summary (train: {train_dates}) ===")
    print(f"{'eval_day':<12} {'lift_back':>9} {'lift_lay':>9} "
          f"{'best_T':>7} {'best_mature':>11} "
          f"{'best_pnl/open':>14} {'best_n':>7}")
    for r in summary_rows:
        T_str = f"{r['best_T']:.2f}" if r['best_T'] is not None else "—"
        mat_str = f"{r['best_mature']:.4f}" if r['best_mature'] is not None else "—"
        pnl_str = f"£{r['best_pnl_per_open']:+.3f}" if r['best_pnl_per_open'] > -1e17 else "—"
        n_str = f"{r['best_n_opens']}" if r['best_n_opens'] is not None else "—"
        print(f"{r['eval_day']:<12} {r['lift_back']:>9.2f}× "
              f"{r['lift_lay']:>8.2f}× {T_str:>7} {mat_str:>11} "
              f"{pnl_str:>14} {n_str:>7}")

    # Final read.
    profitable = [r for r in summary_rows if r["best_pnl_per_open"] > 0]
    print(f"\nDays where SOME threshold is profitable at the assumed cost "
          f"ratio: {len(profitable)} / {len(summary_rows)}")
    print(f"Break-even mature rate at "
          f"P_locked=£{args.p_locked}, P_loss=£{args.p_loss}: "
          f"{args.p_loss / (args.p_locked + args.p_loss):.4f}")


if __name__ == "__main__":
    main()
