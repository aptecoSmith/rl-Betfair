"""Linear probe: is the direction signal in the policy's obs vector?

The 2026-05-24 calibration discovery: the v2 trainer reports
`train_mean_direction_back/lay_bce` ≈ 1.14 across all probe2 agents.
That number is the POS-WEIGHTED per-cell BCE, so the "random uniform
0.5 output" baseline is ~1.14-1.17 (not 0.69) at the observed
positive-class rate (~15-20%). This means the direction_prob_head is
not getting worse than random — but it's also NOT LEARNING.

Two possible explanations:

(a) The signal IS in the obs vector but the head is undertrained
    (loss_weight=0.05 was pinned cohort-wide — small compared to PPO
    surrogate + value + 3 other aux losses). Fix: raise loss weight or
    promote to a GA gene with range like [0.0, 1.0].

(b) The signal is NOT in the obs vector (structural bug — predictor
    output not making it into the obs slice, time misalignment, sign
    convention flipped). Fix: investigate the predictor → obs plumbing.

This probe distinguishes (a) from (b) by training a 1-layer logistic
regression on (obs → direction_label) using the cached oracle samples
(same obs dim as the cohort, lean-obs=574) joined with direction
labels. If the linear classifier descends BCE significantly below the
uniform-0.5 floor, the signal IS in obs ⇒ explanation (a). If BCE
stays at the floor, the signal is NOT in obs ⇒ explanation (b).

Usage:
    python tools/direction_signal_probe.py 2026-04-11
    python tools/direction_signal_probe.py 2026-04-11 2026-04-15
    python tools/direction_signal_probe.py 2026-04-11 --c 0.1

`--c` is the inverse L2 regularisation strength for sklearn's
LogisticRegression (default 1.0). Lower = more regularised.

NOTE: the oracle cache subset biases toward ticks where the arb
oracle considered opening — so the trained classifier's BCE is
relevant for the question "is the signal IN the obs" but not directly
comparable to the agent's actual training distribution. For the
structural question this is the right test.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def _pos_weighted_floor(positive_rate: float) -> float:
    """Expected pos-weighted per-cell BCE for a uniform-0.5 classifier.

    Matches the trainer's loss formula:
        BCE = pos_weight * y * (-log p) + (1 - y) * (-log(1 - p))
    with p = 0.5 (logit 0) and pos_weight = (1 - d) / d.

    For d = positive_rate:
        per-cell loss = d * pos_weight * ln(2) + (1 - d) * ln(2)
                      = (1 - d) * ln(2) + (1 - d) * ln(2)
                      = 2 * (1 - d) * ln(2)
    """
    return 2.0 * (1.0 - positive_rate) * math.log(2.0)


def _pos_weighted_marginal_floor(positive_rate: float) -> float:
    """Expected pos-weighted BCE for the 'predict marginal rate' baseline.

    A classifier that always outputs the global positive rate `d`. This
    is what a model trained on label only (no features) would learn.
    """
    d = max(min(positive_rate, 1.0 - 1e-9), 1e-9)
    pos_weight = (1.0 - d) / d
    logp = math.log(d)
    log1mp = math.log(1.0 - d)
    return d * pos_weight * (-logp) + (1.0 - d) * (-log1mp)


def _pos_weighted_bce(
    probs: np.ndarray,
    labels: np.ndarray,
    positive_rate: float,
) -> float:
    """Compute the trainer's pos-weighted per-cell BCE for the eval set."""
    d = max(min(positive_rate, 1.0 - 1e-9), 1e-9)
    pos_weight = (1.0 - d) / d
    eps = 1e-9
    p = np.clip(probs, eps, 1.0 - eps)
    bce = -(
        pos_weight * labels * np.log(p)
        + (1.0 - labels) * np.log(1.0 - p)
    )
    return float(bce.mean())


def _load_day(date: str, oracle_root: Path, label_root: Path) -> dict | None:
    oracle_npz = oracle_root / date / "oracle_samples.npz"
    label_npz = label_root / date / "horizon60_thresh5_fc60.npz"
    if not oracle_npz.exists() or not label_npz.exists():
        print(
            f"  SKIP {date}: oracle={oracle_npz.exists()} "
            f"label={label_npz.exists()}",
        )
        return None

    o = np.load(oracle_npz)
    l = np.load(label_npz)
    oracle_keys = (o["tick_index"].astype(np.int64) << 16) | o["runner_idx"]
    label_keys = (l["tick_index"].astype(np.int64) << 16) | l["runner_idx"]

    # Map (tick, runner) -> row index in label.
    label_row = {int(k): i for i, k in enumerate(label_keys)}
    keep = []
    for i, k in enumerate(oracle_keys):
        j = label_row.get(int(k))
        if j is not None:
            keep.append((i, j))
    if not keep:
        print(f"  SKIP {date}: no oracle/label intersection")
        return None
    oi = np.array([p[0] for p in keep], dtype=np.int64)
    li = np.array([p[1] for p in keep], dtype=np.int64)

    return {
        "obs": o["obs"][oi].astype(np.float32),
        "label_back": l["label_back"][li].astype(np.float32),
        "label_lay": l["label_lay"][li].astype(np.float32),
        "n_oracle_rows": int(o["obs"].shape[0]),
        "n_label_rows": int(l["label_back"].shape[0]),
        "n_joined": int(len(keep)),
    }


def _train_and_eval_side(
    obs: np.ndarray,
    labels: np.ndarray,
    side: str,
    c: float,
) -> dict:
    """Train logistic regression on (obs -> labels), report BCE numbers.

    Pipeline: StandardScaler -> LogisticRegression. Scaling matters a
    lot for lbfgs convergence on mixed-magnitude features like ours
    (price, log-price, time-to-off, deltas...).
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"error": "sklearn not available"}

    pos_rate = float(labels.mean())
    uniform_floor = _pos_weighted_floor(pos_rate)
    marginal_floor = _pos_weighted_marginal_floor(pos_rate)

    if pos_rate < 1e-4 or pos_rate > 1.0 - 1e-4:
        return {
            "side": side,
            "pos_rate": pos_rate,
            "uniform_05_floor": uniform_floor,
            "marginal_floor": marginal_floor,
            "trained_train_bce": float("nan"),
            "trained_val_bce": float("nan"),
            "note": "degenerate class balance — skipping classifier",
        }

    # 80/20 split. random_state for determinism.
    obs_tr, obs_va, lbl_tr, lbl_va = train_test_split(
        obs, labels, test_size=0.2, random_state=42, stratify=labels,
    )

    # Standardise features — lbfgs converges much faster when input
    # columns have comparable magnitudes. Fit on TRAIN only to avoid
    # peeking at the held-out distribution.
    scaler = StandardScaler()
    obs_tr_s = scaler.fit_transform(obs_tr)
    obs_va_s = scaler.transform(obs_va)

    # class_weight='balanced' gives the same balance treatment as
    # pos_weight = (n_neg / n_pos) in the trainer's BCE.
    clf = LogisticRegression(
        C=c,
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        tol=1e-5,
    )
    clf.fit(obs_tr_s, lbl_tr)
    n_iter = int(clf.n_iter_[0]) if hasattr(clf, "n_iter_") else -1
    converged = n_iter < clf.max_iter

    probs_tr = clf.predict_proba(obs_tr_s)[:, 1]
    probs_va = clf.predict_proba(obs_va_s)[:, 1]
    trained_train_bce = _pos_weighted_bce(probs_tr, lbl_tr, pos_rate)
    trained_val_bce = _pos_weighted_bce(probs_va, lbl_va, pos_rate)

    return {
        "side": side,
        "pos_rate": pos_rate,
        "n_train": int(obs_tr.shape[0]),
        "n_val": int(obs_va.shape[0]),
        "uniform_05_floor": uniform_floor,
        "marginal_floor": marginal_floor,
        "trained_train_bce": trained_train_bce,
        "trained_val_bce": trained_val_bce,
        "n_iter": n_iter,
        "converged": converged,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "dates", nargs="+", metavar="YYYY-MM-DD",
        help="One or more dates to probe.",
    )
    p.add_argument(
        "--oracle-root", default="data/oracle_cache_v2",
        help="Oracle cache root (default data/oracle_cache_v2).",
    )
    p.add_argument(
        "--label-root", default="data/direction_labels",
        help="Direction-label cache root (default data/direction_labels).",
    )
    p.add_argument(
        "--c", type=float, default=1.0,
        help="LogisticRegression C param (inverse L2; default 1.0).",
    )
    args = p.parse_args()

    oracle_root = Path(args.oracle_root)
    label_root = Path(args.label_root)

    # Load + concat all days.
    obs_chunks: list[np.ndarray] = []
    back_chunks: list[np.ndarray] = []
    lay_chunks: list[np.ndarray] = []
    print(f"Loading {len(args.dates)} day(s)...")
    for d in args.dates:
        data = _load_day(d, oracle_root, label_root)
        if data is None:
            continue
        print(
            f"  {d}: oracle={data['n_oracle_rows']} label={data['n_label_rows']} "
            f"joined={data['n_joined']}"
        )
        obs_chunks.append(data["obs"])
        back_chunks.append(data["label_back"])
        lay_chunks.append(data["label_lay"])
    if not obs_chunks:
        print("ERROR: no usable days. Are caches present?", file=sys.stderr)
        return 2

    obs = np.concatenate(obs_chunks, axis=0)
    label_back = np.concatenate(back_chunks, axis=0)
    label_lay = np.concatenate(lay_chunks, axis=0)
    print(
        f"\nTotal: {obs.shape[0]} (tick, runner) pairs, "
        f"obs_dim={obs.shape[1]}"
    )

    # Run probe per side.
    print("\nTraining linear classifier per side...")
    print(
        "  pos_weighted BCE — baselines and trained values are comparable "
        "to the trainer's `train_mean_direction_back/lay_bce`."
    )
    print()

    for label_arr, side_name in [
        (label_back, "back"),
        (label_lay, "lay"),
    ]:
        result = _train_and_eval_side(obs, label_arr, side_name, args.c)
        print(f"--- direction_{side_name} ---")
        if "error" in result:
            print(f"  {result['error']}")
            continue
        print(
            f"  positive_rate          = {result['pos_rate']:.4f}"
        )
        print(
            f"  train rows / val rows  = "
            f"{result.get('n_train', '?'):>6d} / "
            f"{result.get('n_val', '?'):>6d}"
        )
        print(
            f"  uniform-0.5 floor      = {result['uniform_05_floor']:.4f}  "
            f"(naive head, no learning)"
        )
        print(
            f"  marginal-rate floor    = {result['marginal_floor']:.4f}  "
            f"(predicts pos_rate, no features)"
        )
        print(
            f"  TRAINED train BCE      = {result['trained_train_bce']:.4f}  "
            f"(in-sample, can overfit)"
        )
        print(
            f"  TRAINED val   BCE      = {result['trained_val_bce']:.4f}  "
            f"(held-out, the real signal)"
        )
        print(
            f"  classifier n_iter      = {result.get('n_iter', '?')} "
            f"(converged={result.get('converged', '?')})"
        )
        improvement = (
            result["uniform_05_floor"] - result["trained_val_bce"]
        )
        rel = improvement / result["uniform_05_floor"]
        verdict = (
            "SIGNAL IS IN OBS"
            if improvement > 0.05
            else "no meaningful signal in obs"
        )
        print(
            f"  val BCE - uniform_floor = {-improvement:+.4f}  "
            f"({rel*100:+.1f}% relative) -> {verdict}"
        )
        print()

    print("Interpretation:")
    print("  * If val BCE < uniform-0.5 floor by >0.05: the signal IS in")
    print("    the obs vector. The policy's head should be able to learn")
    print("    it — likely undertrained at loss_weight=0.05.")
    print("    Fix = raise loss_weight or promote to GA gene.")
    print("  * If val BCE ~= uniform-0.5 floor: signal NOT in obs.")
    print("    Structural bug in predictor->obs plumbing or labels.")
    print("    Fix = investigate before another cohort.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
