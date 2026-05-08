"""Phase-15 loss-variant benchmark for direction_prob_head.

Trains a fresh per-runner direction predictor (LayerNorm + 2-layer
MLP, mirroring agents_v2/discrete_policy.py's direction_prob_head)
on multi-day pooled features + cached direction labels with
different loss formulations. Reports calibration metrics on a
held-out day so we can pick the best variant before plumbing it
into the cohort BC pretrainer.

The benchmark is OFFLINE — no PPO, no env, no rollouts. It runs in
~30s per loss on GPU. Compare {BCE, BCE+pos_weight, focal, MSE}
side-by-side, no compute waste.

Usage::

    python -m tools.direction_loss_benchmark \\
        --train-dates 2026-05-03,2026-05-04,2026-05-05 \\
        --eval-date 2026-05-02 \\
        --losses bce bce_pos focal mse
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.episode_builder import load_day
from data.feature_engineer import engineer_day
from env.betfair_env import (
    MARKET_DIM,
    RUNNER_DIM,
    RUNNER_KEYS,
    VELOCITY_DIM,
    BetfairEnv,
)
from training_v2.direction_label_scan import load_labels


_DATA_DIR = Path("data/processed")
_LABEL_HORIZON = 60
_LABEL_THRESHOLD = 5
_LABEL_FORCE_CLOSE = 60.0
_BATCH_SIZE = 64
_HIDDEN = 64
_DEFAULT_LR = 3e-4


def _build_predictor(hidden: int = _HIDDEN, depth: int = 1) -> nn.Sequential:
    """Build a per-runner direction predictor.

    depth=1: LayerNorm + Linear(D, H) + ReLU + Linear(H, 2)  (default,
             mirrors agents_v2/discrete_policy.py:direction_prob_head)
    depth=2: LayerNorm + Linear(D, H) + ReLU + Linear(H, H) + ReLU
             + Linear(H, 2)
    depth=0: LayerNorm + Linear(D, 2)  (probe-style direct linear)
    """
    layers: list[nn.Module] = [nn.LayerNorm(RUNNER_DIM)]
    if depth == 0:
        layers.append(nn.Linear(RUNNER_DIM, 2))
    else:
        layers.append(nn.Linear(RUNNER_DIM, hidden))
        for _ in range(depth - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden, hidden))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, 2))
    return nn.Sequential(*layers)


def _build_pool(date: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y_back, y_lay) for all (tick, runner) cells with
    a direction label on this date.

    X: (N, RUNNER_DIM) per-runner feature slices.
    y_back, y_lay: (N,) binary labels.
    """
    print(f"  loading {date} ...")
    day = load_day(date, _DATA_DIR)
    feats_per_race = engineer_day(day)

    # Build static obs per (race, tick) — replicates env's
    # _features_to_array function. We need MARKET_DIM + VELOCITY_DIM
    # + max_runners*RUNNER_DIM floats. Per-race runner_map.
    X_list: list[np.ndarray] = []
    yb_list: list[float] = []
    yl_list: list[float] = []

    # Load direction labels
    labels = load_labels(
        date, _DATA_DIR,
        direction_horizon_ticks=_LABEL_HORIZON,
        direction_threshold_ticks=_LABEL_THRESHOLD,
        force_close_before_off_seconds=_LABEL_FORCE_CLOSE,
    )
    # Index: {(global_tick, runner_idx): (label_back, label_lay)}
    label_idx: dict = {}
    for r in labels:
        label_idx[(int(r.tick_index), int(r.runner_idx))] = (
            float(r.label_back), float(r.label_lay),
        )

    # Walk races/ticks. Need a global_tick index that matches the
    # direction-label scanner's iteration.
    global_tick = 0
    for race_idx, race in enumerate(day.races):
        # Build runner_map (sid -> slot) the same way env does.
        sids = set()
        for tick in race.ticks:
            for r in tick.runners:
                sids.add(r.selection_id)
        runner_map = {sid: i for i, sid in enumerate(sorted(sids))}

        race_feats = feats_per_race[race_idx]
        for tick_idx, tick in enumerate(race.ticks):
            tick_feats = race_feats[tick_idx]
            runners_dict = tick_feats["runners"]
            for sid, slot in runner_map.items():
                key = (global_tick, slot)
                if key not in label_idx:
                    global_tick_unused = 1  # placeholder for line readability
                    continue
                lb, ll = label_idx[key]
                # Build per-runner feature slice (RUNNER_DIM floats).
                slice_arr = np.zeros(RUNNER_DIM, dtype=np.float32)
                feats = runners_dict.get(sid)
                if feats is None:
                    # Sparse — treat as zero slice. Matches env's
                    # _features_to_array.
                    pass
                else:
                    for i, k in enumerate(RUNNER_KEYS):
                        slice_arr[i] = float(feats.get(k, 0.0))
                X_list.append(slice_arr)
                yb_list.append(lb)
                yl_list.append(ll)
            global_tick += 1
        # Note: scanner's global_tick uses ALL ticks (not race-local).
        # The +1 above happens once per tick of THIS race, accumulating
        # across the day. Same iteration order as direction_label_scan.

    X = np.stack(X_list, axis=0).astype(np.float32) if X_list else np.zeros(
        (0, RUNNER_DIM), dtype=np.float32,
    )
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    yb = np.array(yb_list, dtype=np.float32)
    yl = np.array(yl_list, dtype=np.float32)
    print(
        f"    -> {len(X)} cells | label_back: pos={int(yb.sum())} "
        f"({100*yb.mean():.1f}%) | label_lay: pos={int(yl.sum())} "
        f"({100*yl.mean():.1f}%)"
    )
    return X, yb, yl


def _eval(
    model: nn.Module,
    X: torch.Tensor,
    y_back: torch.Tensor,
    y_lay: torch.Tensor,
) -> dict:
    """Calibration metrics on a held-out pool."""
    model.eval()
    with torch.no_grad():
        out = model(X)  # (N, 2)
        logit_back = out[:, 0]
        logit_lay = out[:, 1]
        # BCE
        bce_back = F.binary_cross_entropy_with_logits(
            logit_back, y_back, reduction="mean",
        ).item()
        bce_lay = F.binary_cross_entropy_with_logits(
            logit_lay, y_lay, reduction="mean",
        ).item()
        # AUC (rough, via numpy)
        p_back = torch.sigmoid(logit_back).cpu().numpy()
        p_lay = torch.sigmoid(logit_lay).cpu().numpy()
        yb_np = y_back.cpu().numpy()
        yl_np = y_lay.cpu().numpy()
        auc_back = _auc(p_back, yb_np)
        auc_lay = _auc(p_lay, yl_np)
        # Top-decile precision (probe metric)
        td_back = _top_decile_precision(p_back, yb_np)
        td_lay = _top_decile_precision(p_lay, yl_np)
    model.train()
    return {
        "bce_back": bce_back, "bce_lay": bce_lay,
        "auc_back": auc_back, "auc_lay": auc_lay,
        "td_prec_back": td_back, "td_prec_lay": td_lay,
    }


def _auc(p: np.ndarray, y: np.ndarray) -> float:
    """Two-class AUC. Returns 0.5 when one class is empty."""
    pos = p[y > 0.5]
    neg = p[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Mann-Whitney U
    n_pos = len(pos)
    n_neg = len(neg)
    # Rank-sum approach: count concordant pairs.
    order = np.argsort(p)
    rank = np.empty_like(order, dtype=np.float64)
    rank[order] = np.arange(1, len(p) + 1)
    sum_pos_ranks = float(rank[y > 0.5].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _top_decile_precision(p: np.ndarray, y: np.ndarray) -> float:
    """Of the top-decile predicted, what fraction are positive?"""
    if len(p) == 0:
        return 0.0
    n_top = max(1, len(p) // 10)
    idx = np.argsort(-p)[:n_top]
    return float(y[idx].mean())


def _focal_loss(
    logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0,
) -> torch.Tensor:
    """Focal BCE: down-weight easy examples."""
    p = torch.sigmoid(logits)
    p_t = target * p + (1.0 - target) * (1.0 - p)
    bce = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none",
    )
    return ((1.0 - p_t) ** gamma * bce).mean()


def train_one(
    loss_name: str,
    X_train: np.ndarray,
    yb_train: np.ndarray,
    yl_train: np.ndarray,
    X_eval: np.ndarray,
    yb_eval: np.ndarray,
    yl_eval: np.ndarray,
    n_steps: int,
    lr: float,
    device: torch.device,
    seed: int = 42,
    hidden: int = _HIDDEN,
    depth: int = 1,
) -> dict:
    """Train a fresh predictor with the named loss; report metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _build_predictor(hidden=hidden, depth=depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    Yb = torch.tensor(yb_train, dtype=torch.float32, device=device)
    Yl = torch.tensor(yl_train, dtype=torch.float32, device=device)
    Xe = torch.tensor(X_eval, dtype=torch.float32, device=device)
    YbE = torch.tensor(yb_eval, dtype=torch.float32, device=device)
    YlE = torch.tensor(yl_eval, dtype=torch.float32, device=device)

    # pos_weight prepared for BCE+pos_weight variant
    n_pos_b = float(Yb.sum().item())
    n_neg_b = float(len(Yb) - n_pos_b)
    n_pos_l = float(Yl.sum().item())
    n_neg_l = float(len(Yl) - n_pos_l)
    pw_b = torch.tensor(
        min(10.0, n_neg_b / max(n_pos_b, 1.0)),
        dtype=torch.float32, device=device,
    )
    pw_l = torch.tensor(
        min(10.0, n_neg_l / max(n_pos_l, 1.0)),
        dtype=torch.float32, device=device,
    )

    n = Xt.shape[0]
    rng = np.random.default_rng(seed)
    t0 = time.monotonic()
    for step in range(n_steps):
        idx = rng.choice(n, _BATCH_SIZE, replace=True)
        xb = Xt[idx]
        yb = Yb[idx]
        yl = Yl[idx]
        out = model(xb)
        lo_back = out[:, 0]
        lo_lay = out[:, 1]

        if loss_name == "bce":
            lb = F.binary_cross_entropy_with_logits(lo_back, yb)
            ll = F.binary_cross_entropy_with_logits(lo_lay, yl)
            loss = lb + ll
        elif loss_name == "bce_pos":
            lb = F.binary_cross_entropy_with_logits(
                lo_back, yb, pos_weight=pw_b,
            )
            ll = F.binary_cross_entropy_with_logits(
                lo_lay, yl, pos_weight=pw_l,
            )
            loss = lb + ll
        elif loss_name == "focal":
            lb = _focal_loss(lo_back, yb, gamma=2.0)
            ll = _focal_loss(lo_lay, yl, gamma=2.0)
            loss = lb + ll
        elif loss_name == "mse":
            pb = torch.sigmoid(lo_back)
            pl = torch.sigmoid(lo_lay)
            loss = F.mse_loss(pb, yb) + F.mse_loss(pl, yl)
        else:
            raise ValueError(f"unknown loss {loss_name}")

        opt.zero_grad()
        loss.backward()
        opt.step()

    wall = time.monotonic() - t0
    metrics = _eval(model, Xe, YbE, YlE)
    metrics["wall_s"] = wall
    metrics["loss"] = loss_name
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dates", required=True,
                        help="Comma-separated YYYY-MM-DD list.")
    parser.add_argument("--eval-date", required=True,
                        help="Held-out date.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=_DEFAULT_LR)
    parser.add_argument("--losses", nargs="+",
                        default=["bce", "bce_pos", "focal", "mse"])
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[64],
                        help="Hidden sizes to sweep. Default [64].")
    parser.add_argument("--depths", nargs="+", type=int, default=[1],
                        help=("MLP depths to sweep. 0=direct Linear, "
                              "1=H-MLP (default), 2=H-H-MLP."))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    train_dates = [d.strip() for d in args.train_dates.split(",") if d.strip()]
    print(f"[bench] device={device} train_dates={train_dates} eval={args.eval_date}")

    print("Loading training pools (per-day)...")
    Xs, Ybs, Yls = [], [], []
    for d in train_dates:
        X, yb, yl = _build_pool(d)
        Xs.append(X); Ybs.append(yb); Yls.append(yl)
    X_train = np.concatenate(Xs, axis=0) if Xs else np.zeros((0, RUNNER_DIM), dtype=np.float32)
    yb_train = np.concatenate(Ybs, axis=0)
    yl_train = np.concatenate(Yls, axis=0)
    print(f"  pooled train: {len(X_train)} cells")

    print("Loading eval pool...")
    X_eval, yb_eval, yl_eval = _build_pool(args.eval_date)

    n_combos = len(args.losses) * len(args.hidden_sizes) * len(args.depths)
    print(
        f"\nRunning {n_combos} combination(s) "
        f"({len(args.losses)} loss × {len(args.hidden_sizes)} hidden × "
        f"{len(args.depths)} depth) for {args.steps} steps each:"
    )
    print(
        f"{'loss':<10} {'h':>4} {'d':>2} "
        f"{'bce_b':>8} {'bce_l':>8} {'auc_b':>7} {'auc_l':>7} "
        f"{'td_b':>6} {'td_l':>6} {'wall':>6}"
    )
    for hidden in args.hidden_sizes:
        for depth in args.depths:
            for loss_name in args.losses:
                m = train_one(
                    loss_name, X_train, yb_train, yl_train,
                    X_eval, yb_eval, yl_eval,
                    n_steps=args.steps, lr=args.lr, device=device,
                    seed=args.seed, hidden=hidden, depth=depth,
                )
                print(
                    f"{loss_name:<10} {hidden:>4} {depth:>2} "
                    f"{m['bce_back']:>8.4f} {m['bce_lay']:>8.4f} "
                    f"{m['auc_back']:>7.4f} {m['auc_lay']:>7.4f} "
                    f"{m['td_prec_back']:>6.3f} {m['td_prec_lay']:>6.3f} "
                    f"{m['wall_s']:>5.1f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
