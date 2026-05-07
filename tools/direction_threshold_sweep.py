"""Threshold sweep — does using the direction predictor as a SELECTIVITY
GATE turn the per-pair-trade economics positive?

Phase-13 follow-up to ``direction_features_probe.py``. The probe
showed the augmented predictor lifts the top-quintile realised
positive rate to ~57% (back) / ~58% (lay) at horizon=60. The
strategic question this tool answers: if we ONLY open when the
predictor's confidence exceeds threshold T, how does the trade-off
between volume and quality play out across T?

For each (tick, runner) the predictor emits two probabilities:
P(label_back=1) and P(label_lay=1). The agent's natural decision is
to open the side with higher predicted probability — but only if
that probability exceeds T. So per-row the action is::

    side = argmax(P_back, P_lay)
    open = max(P_back, P_lay) >= T

For each T in {0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}::

* n_opens — count of (tick, runner) rows above T (= trade volume).
* mature_rate — fraction of opens where the chosen side's label was 1.
* implied_pnl — n_opens × (mature_rate × P_locked − (1 − mature_rate) × P_loss).

Default P_locked = £2.50 and P_loss = £3.00 reflect the rough
per-pair magnitudes the cohort runs surfaced. Operator can override
via CLI to taste.

Usage::

    python -m tools.direction_threshold_sweep \\
        --date 2026-05-03 --device cuda
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-05-03")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--horizon-ticks", type=int, default=60)
    ap.add_argument("--threshold-ticks", type=int, default=5)
    ap.add_argument("--fc-secs", type=float, default=60.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p-locked", type=float, default=2.50,
                    help="Implied locked-P&L per matured pair (£).")
    ap.add_argument("--p-loss", type=float, default=3.00,
                    help="Implied loss per non-matured pair (£). "
                         "Represents the AVERAGE across naked + force-"
                         "closed outcomes. Operator can sweep.")
    args = ap.parse_args()

    config = _load_config()
    data_dir = Path(args.data_dir)

    print(f"=== building features for {args.date} ===")
    base_arr, aug_arr, tick_arr, runner_arr, sid_arr = build_features(
        args.date, data_dir,
    )
    print(f"  rows = {base_arr.shape[0]}")

    print(f"\n=== loading labels (horizon={args.horizon_ticks} thresh={args.threshold_ticks}) ===")
    labels = load_labels(
        args.date, data_dir,
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
    lback_s = label_back[sel]
    llay_s = label_lay[sel]
    feats = np.concatenate([base_s, aug_s], axis=1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    n = feats.shape[0]
    pos_back = float(lback_s.mean())
    pos_lay = float(llay_s.mean())
    print(f"  N={n}  pos_back={pos_back:.4f}  pos_lay={pos_lay:.4f}")

    # Standardise.
    mu = feats.mean(axis=0, keepdims=True)
    sigma = feats.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    feats_z = (feats - mu) / sigma
    feats_z = np.nan_to_num(feats_z, nan=0.0, posinf=0.0, neginf=0.0)

    # Train base+augmented MLP.
    print(f"\n=== training augmented MLP (in_dim={feats_z.shape[1]}) ===")
    torch.manual_seed(int(args.seed))
    device = args.device
    model = _MLP(in_dim=feats_z.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    pw_back = torch.tensor(
        (1 - pos_back) / max(pos_back, 1e-6),
        dtype=torch.float32, device=device,
    )
    pw_lay = torch.tensor(
        (1 - pos_lay) / max(pos_lay, 1e-6),
        dtype=torch.float32, device=device,
    )
    feats_t = torch.from_numpy(feats_z.astype(np.float32)).to(device)
    lback_t = torch.from_numpy(lback_s).to(device)
    llay_t = torch.from_numpy(llay_s).to(device)
    rng = np.random.default_rng(int(args.seed))
    t0 = time.monotonic()
    for step in range(args.steps):
        idx = rng.integers(0, n, size=args.batch_size)
        idx_t = torch.from_numpy(idx).to(device)
        x = feats_t.index_select(0, idx_t)
        yb = lback_t.index_select(0, idx_t)
        yl = llay_t.index_select(0, idx_t)
        out = model(x)
        bback = nn.functional.binary_cross_entropy_with_logits(
            out[:, 0], yb, pos_weight=pw_back,
        )
        blay = nn.functional.binary_cross_entropy_with_logits(
            out[:, 1], yl, pos_weight=pw_lay,
        )
        loss = bback + blay
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"  train wall = {time.monotonic() - t0:.1f}s, "
          f"final loss = {loss.item():.4f}")

    # Score every (tick, runner) row.
    with torch.no_grad():
        out = model(feats_t)
        p_back = torch.sigmoid(out[:, 0]).cpu().numpy()
        p_lay = torch.sigmoid(out[:, 1]).cpu().numpy()

    # Per-row choice = argmax side; chosen confidence = max(P_back, P_lay).
    chosen_side_is_back = p_back >= p_lay
    chosen_conf = np.where(chosen_side_is_back, p_back, p_lay)
    chosen_label = np.where(chosen_side_is_back, lback_s, llay_s)

    # Estimate the WHOLE-DAY agent volume from the cohort runs as a
    # reference. The eval-day rollouts on the 12x4 cohort had ~410
    # pairs_opened per day across ~7000 priceable rows; current
    # all-rows count is 55190 (10x more). To compare apples to apples
    # we report the threshold-gated count both as raw and as a fraction
    # of the unfiltered priceable-row count — the cohort would only
    # actually open on a small subset of these even at threshold=0.0
    # because its per-tick action space allows one open / tick.
    print()
    print(f"=== threshold sweep (P_locked = £{args.p_locked:.2f}, "
          f"P_loss = £{args.p_loss:.2f}) ===")
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
        # Scale to a "cohort-comparable" volume: the eval-day cohort
        # opens ~410 pairs in 7k priceable rows, i.e. it samples ~6%
        # of priceable rows. Apply that scaling to project a
        # cohort-realistic per-day open count.
        cohort_open_frac = 410.0 / 7000.0  # ≈ 0.059
        cohort_n_opens = int(n_opens * cohort_open_frac)
        cohort_day_pnl = pnl_per_open * cohort_n_opens
        print(f"{T:>7.2f} {n_opens:>9d} {n_opens/n*100:>5.2f}% "
              f"{mature:>10.4f} {force:>10.4f} "
              f"{pnl_per_open:>+10.3f} {day_pnl:>+10.1f}")

    # Also report a "do nothing baseline" — open EVERY row, baseline
    # mature rate is just `pos_back` or `pos_lay`. Compare against the
    # selective-gate numbers above.
    print()
    print(f"=== reference: 'open everything' baseline ===")
    open_all_mature_back = pos_back
    open_all_mature_lay = pos_lay
    pnl_all_back = (
        pos_back * args.p_locked - (1 - pos_back) * args.p_loss
    )
    pnl_all_lay = (
        pos_lay * args.p_locked - (1 - pos_lay) * args.p_loss
    )
    print(f"  open every priceable BACK: mature={open_all_mature_back:.4f} "
          f"pnl_per_open=£{pnl_all_back:+.3f}")
    print(f"  open every priceable LAY : mature={open_all_mature_lay:.4f} "
          f"pnl_per_open=£{pnl_all_lay:+.3f}")
    print(f"  break-even mature_rate at "
          f"P_locked=£{args.p_locked:.2f}, P_loss=£{args.p_loss:.2f}: "
          f"{args.p_loss / (args.p_locked + args.p_loss):.4f}")

    # Confidence histogram so the operator can see WHERE the predictor
    # places the row mass — high-density at low confidence means the
    # gate prunes aggressively.
    print()
    print(f"=== confidence histogram (per-row max(P_back, P_lay)) ===")
    bins = np.linspace(0.0, 1.0, 11)
    hist, edges = np.histogram(chosen_conf, bins=bins)
    for i in range(len(hist)):
        bar = "#" * int(hist[i] / max(hist) * 50)
        print(f"  [{edges[i]:.2f}, {edges[i+1]:.2f}): {hist[i]:>6d}  {bar}")


if __name__ == "__main__":
    main()
