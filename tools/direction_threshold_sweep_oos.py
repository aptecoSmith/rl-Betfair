"""Out-of-sample threshold sweep — train on one day, gate-test on another.

Phase-13 follow-up to ``direction_threshold_sweep.py``. The in-sample
sweep showed threshold ≥ 0.6 turns the gated strategy profitable on
2026-05-03. To check whether that calibration generalises, train the
augmented MLP on a TRAIN day, score every priceable (tick, runner)
on a separate EVAL day, and re-run the same threshold sweep against
the eval day's labels.

If the OOS calibration holds (top-bin realised positive rate ≈ 55%+
at threshold 0.65), the selectivity-gate strategy is real signal,
not in-sample overfit.

Usage::

    python -m tools.direction_threshold_sweep_oos \\
        --train-date 2026-05-03 --eval-date 2026-05-04 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from training_v2.arb_oracle import _load_config
from training_v2.direction_label_scan import load_labels
from tools.direction_features_probe import (
    _MLP,
    _per_row_label_lookup,
    build_features,
)


def _build_filtered(date: str, data_dir: Path, args):
    base_arr, aug_arr, tick_arr, runner_arr, sid_arr = build_features(
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
    lback_s = label_back[sel]
    llay_s = label_lay[sel]
    return feats, lback_s, llay_s


def _calibration_table(p, y, n_bins=5):
    sidx = np.argsort(p)
    chunks = np.array_split(sidx, n_bins)
    return [
        (float(p[c].mean()), float(y[c].mean()))
        for c in chunks
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-date", default="2026-05-03")
    ap.add_argument("--eval-date", default="2026-05-04")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--horizon-ticks", type=int, default=60)
    ap.add_argument("--threshold-ticks", type=int, default=5)
    ap.add_argument("--fc-secs", type=float, default=60.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p-locked", type=float, default=2.50)
    ap.add_argument("--p-loss", type=float, default=3.00)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)

    print(f"=== TRAIN day {args.train_date} ===")
    t_feats, t_lback, t_llay = _build_filtered(
        args.train_date, data_dir, args,
    )
    n_train = t_feats.shape[0]
    pos_back = float(t_lback.mean())
    pos_lay = float(t_llay.mean())
    print(f"  N_train={n_train}  pos_back={pos_back:.4f}  "
          f"pos_lay={pos_lay:.4f}")

    # Standardise on training stats; apply same to eval.
    mu = t_feats.mean(axis=0, keepdims=True)
    sigma = t_feats.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    t_feats_z = (t_feats - mu) / sigma
    t_feats_z = np.nan_to_num(t_feats_z, 0.0, 0.0, 0.0)

    print(f"\n=== training MLP on {args.train_date} ===")
    torch.manual_seed(int(args.seed))
    device = args.device
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
    print(f"  train wall = {time.monotonic() - t0:.1f}s, "
          f"final bce_back={bback.item():.4f} bce_lay={blay.item():.4f}")

    print(f"\n=== EVAL day {args.eval_date} ===")
    e_feats, e_lback, e_llay = _build_filtered(
        args.eval_date, data_dir, args,
    )
    n_eval = e_feats.shape[0]
    e_pos_back = float(e_lback.mean())
    e_pos_lay = float(e_llay.mean())
    print(f"  N_eval={n_eval}  pos_back={e_pos_back:.4f}  "
          f"pos_lay={e_pos_lay:.4f}")

    # Apply training-day standardisation.
    e_feats_z = (e_feats - mu) / sigma
    e_feats_z = np.nan_to_num(e_feats_z, 0.0, 0.0, 0.0)
    e_feats_t = torch.from_numpy(e_feats_z.astype(np.float32)).to(device)
    with torch.no_grad():
        out = model(e_feats_t)
        p_back = torch.sigmoid(out[:, 0]).cpu().numpy()
        p_lay = torch.sigmoid(out[:, 1]).cpu().numpy()

    # Calibration check first — does the predictor's ranking still hold OOS?
    print(f"\n=== eval-day calibration (5 quantile bins) ===")
    cb = _calibration_table(p_back, e_lback)
    cl = _calibration_table(p_lay, e_llay)
    for i, ((pb, rb), (pl, rl)) in enumerate(zip(cb, cl)):
        print(f"  bin {i}: back P_pred={pb:.3f} P_real={rb:.3f}   "
              f"lay P_pred={pl:.3f} P_real={rl:.3f}")
    lift_back = cb[-1][1] / max(cb[0][1], 1e-6)
    lift_lay = cl[-1][1] / max(cl[0][1], 1e-6)
    print(f"  Top-vs-bottom-quintile lift (OOS): "
          f"back {lift_back:.2f}×  lay {lift_lay:.2f}×")

    # Threshold sweep on eval day.
    chosen_back = p_back >= p_lay
    chosen_conf = np.where(chosen_back, p_back, p_lay)
    chosen_label = np.where(chosen_back, e_lback, e_llay)
    print(f"\n=== threshold sweep on EVAL day "
          f"(P_locked = £{args.p_locked:.2f}, P_loss = £{args.p_loss:.2f}) ===")
    print(f"{'thresh':>7} {'n_opens':>9} {'%rows':>7} "
          f"{'mature_rate':>11} {'force_rate':>10} "
          f"{'pnl/open £':>10} {'day_pnl £':>10}")
    print('-' * 78)
    for T in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        gate = chosen_conf >= T
        n_opens = int(gate.sum())
        if n_opens == 0:
            print(f"{T:>7.2f} {0:>9d} {0.0:>6.2f}% "
                  f"{'—':>11} {'—':>10} {'—':>10} {'—':>10}")
            continue
        gated_label = chosen_label[gate]
        mature = float(gated_label.mean())
        force = 1.0 - mature
        pnl_per_open = mature * args.p_locked - force * args.p_loss
        day_pnl = pnl_per_open * n_opens
        print(f"{T:>7.2f} {n_opens:>9d} {n_opens/n_eval*100:>5.2f}% "
              f"{mature:>10.4f} {force:>10.4f} "
              f"{pnl_per_open:>+10.3f} {day_pnl:>+10.1f}")

    # Open-everything reference.
    print()
    print(f"=== reference: 'open every priceable row' (eval day) ===")
    pnl_all = (
        chosen_label.mean() * args.p_locked
        - (1 - chosen_label.mean()) * args.p_loss
    )
    print(f"  mature_rate (chosen-side, no gate): "
          f"{float(chosen_label.mean()):.4f}")
    print(f"  pnl/open: £{pnl_all:+.3f}")
    print(f"  break-even mature rate at "
          f"P_locked=£{args.p_locked:.2f}, P_loss=£{args.p_loss:.2f}: "
          f"{args.p_loss / (args.p_locked + args.p_loss):.4f}")


if __name__ == "__main__":
    main()
