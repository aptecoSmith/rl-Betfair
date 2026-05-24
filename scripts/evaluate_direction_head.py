"""Multi-metric evaluation of a trained direction head.

Reports, per side (back / lay):

  * BCE (unweighted and pos-weighted) on the eval data
  * BCE relative to two baselines:
      - uniform 0.5 floor (= ln 2 unweighted, ~1.13 pos-weighted)
      - marginal-rate floor (constant prob = positive class rate)
  * Pearson correlation between head output and binary label
  * ROC AUC
  * Brier score (mean (prob - label)^2)
  * 10-bucket reliability table (predicted prob bucket -> empirical
    observed positive rate; ideal head has empirical rate ≈ bucket
    midpoint)

Usage::

    python -m scripts.evaluate_direction_head \\
        --manifest models/direction_head/v1_2026-05-24 \\
        --eval-dates 2026-04-10 \\
        --label-stem horizon60_thresh5_fc60

The eval dates can include HELD-OUT cohort eval days (the head was
NOT trained on those) — the held-out-day invariant from training
applies to the head's TRAINING corpus, not to evaluation. Reporting
metrics on held-out eval days is exactly the cleanest test.

Pre-flight:

  * Refuses if any eval-date is in the head's training-set
    (manifest's `training.training_dates`) — that's the head's own
    train set, would inflate metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn

from env.betfair_env import (
    LEAN_RUNNER_DIM,
    LEAN_RUNNER_KEYS,
    MARKET_DIM,
    OBS_SCHEMA_VERSION,
    VELOCITY_DIM,
)


def _load_day_obs_and_labels(
    date: str,
    label_stem: str,
    oracle_root: Path,
    label_root: Path,
) -> dict:
    """Same join as the train script."""
    oracle_path = oracle_root / date / "oracle_samples.npz"
    label_path = label_root / date / f"{label_stem}.npz"
    if not oracle_path.exists():
        raise FileNotFoundError(
            f"oracle cache missing for {date}: {oracle_path}",
        )
    if not label_path.exists():
        raise FileNotFoundError(
            f"direction-labels cache missing for {date}: {label_path}",
        )
    o = np.load(oracle_path)
    if int(o["obs_schema_version"]) != OBS_SCHEMA_VERSION:
        raise ValueError(
            f"{date}: oracle cache obs_schema_version="
            f"{int(o['obs_schema_version'])} but env expects "
            f"{OBS_SCHEMA_VERSION}. Re-scan with "
            f"`python -m training_v2.oracle_cli scan --date {date} "
            f"--predictor-lean-obs --use-direction-predictor "
            f"--use-race-outcome-predictor "
            f"--predictor-bundle-manifests ...`",
        )
    obs_all = o["obs"]
    tick_index = o["tick_index"].astype(np.int64)
    runner_idx = o["runner_idx"].astype(np.int64)
    l = np.load(label_path)
    lk = (l["tick_index"].astype(np.int64) << 16) | l["runner_idx"]
    ok = (tick_index << 16) | runner_idx
    lookup = {int(k): i for i, k in enumerate(lk)}
    keep = [(i, lookup[int(k)]) for i, k in enumerate(ok)
            if int(k) in lookup]
    if not keep:
        return {"n": 0}
    oi = np.array([p[0] for p in keep], dtype=np.int64)
    li = np.array([p[1] for p in keep], dtype=np.int64)
    obs_j = obs_all[oi]
    ri_j = runner_idx[oi]
    n = obs_j.shape[0]
    # Sanity: predictor populated
    q50_7m_idx = LEAN_RUNNER_KEYS.index("dir_q50_7m")
    sample = np.empty(min(n, 256), dtype=np.float32)
    for i in range(min(n, 256)):
        s = MARKET_DIM + VELOCITY_DIM + int(ri_j[i]) * LEAN_RUNNER_DIM
        sample[i] = obs_j[i, s + q50_7m_idx]
    if float(sample.std()) < 1e-6:
        raise ValueError(
            f"{date}: dir_q50_7m std ~ 0 — oracle cache wasn't "
            f"scanned with --use-direction-predictor. Re-scan first.",
        )
    # Extract per-runner block
    per_runner = np.empty((n, LEAN_RUNNER_DIM), dtype=np.float32)
    for i in range(n):
        s = MARKET_DIM + VELOCITY_DIM + int(ri_j[i]) * LEAN_RUNNER_DIM
        per_runner[i] = obs_j[i, s:s + LEAN_RUNNER_DIM]
    return {
        "n": n,
        "per_runner_obs": per_runner,
        "label_back": l["label_back"][li].astype(np.float32),
        "label_lay": l["label_lay"][li].astype(np.float32),
    }


class DirectionHead(nn.Module):
    """Match the architecture saved by train_direction_head.py."""
    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_head_from_manifest(
    manifest_dir: Path,
) -> tuple[DirectionHead, dict]:
    manifest = json.loads(
        (manifest_dir / "manifest.json").read_text(encoding="utf-8"),
    )
    arch = manifest["architecture"]
    head = DirectionHead(
        input_dim=int(arch["input_dim"]),
        hidden=int(arch["hidden_dims"][0]),
    )
    state = torch.load(
        manifest_dir / manifest["weights_path"],
        map_location="cpu",
        weights_only=True,
    )
    # The saved state has keys like "0.weight" (from flattening the
    # Sequential). The model's net.* expects "net.0.weight" — re-add
    # the prefix.
    prefixed = {f"net.{k}": v for k, v in state.items()}
    head.load_state_dict(prefixed, strict=True)
    head.eval()
    return head, manifest


# ── metrics ──────────────────────────────────────────────────────────


def _bce_unweighted(probs: np.ndarray, labels: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(probs, eps, 1 - eps)
    return float((-(labels * np.log(p)
                    + (1 - labels) * np.log(1 - p))).mean())


def _bce_pos_weighted(
    probs: np.ndarray, labels: np.ndarray, pos_rate: float,
) -> float:
    d = max(min(pos_rate, 1 - 1e-9), 1e-9)
    pw = (1 - d) / d
    eps = 1e-9
    p = np.clip(probs, eps, 1 - eps)
    return float((-(pw * labels * np.log(p)
                    + (1 - labels) * np.log(1 - p))).mean())


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    mx, my = x.mean(), y.mean()
    num = ((x - mx) * (y - my)).sum()
    dx = np.sqrt(((x - mx) ** 2).sum())
    dy = np.sqrt(((y - my) ** 2).sum())
    return float(num / (dx * dy)) if dx > 0 and dy > 0 else float("nan")


def _roc_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Manual AUC via Mann-Whitney-U / probability-of-correct-ranking."""
    pos = probs[labels > 0.5]
    neg = probs[labels < 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Mann-Whitney: prob(random pos > random neg)
    pos_sorted = np.sort(pos)
    # For each neg, count positives < it (for ties, +0.5)
    ranks_strict = np.searchsorted(pos_sorted, neg, side="left")
    ranks_inclusive = np.searchsorted(pos_sorted, neg, side="right")
    ties = ranks_inclusive - ranks_strict
    pos_above_neg = (pos.size - ranks_inclusive) + 0.5 * ties
    auc = pos_above_neg.sum() / (pos.size * neg.size)
    return float(auc)


def _brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(((probs - labels) ** 2).mean())


def _reliability_table(
    probs: np.ndarray, labels: np.ndarray, n_buckets: int = 10,
) -> list[dict]:
    bins = np.linspace(0.0, 1.0, n_buckets + 1)
    out: list[dict] = []
    for i in range(n_buckets):
        lo, hi = bins[i], bins[i + 1]
        if i == n_buckets - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        if n == 0:
            out.append({
                "bucket": f"[{lo:.2f}, {hi:.2f})",
                "n": 0,
                "mean_pred": float("nan"),
                "emp_rate": float("nan"),
            })
            continue
        out.append({
            "bucket": f"[{lo:.2f}, {hi:.2f})",
            "n": n,
            "mean_pred": float(probs[mask].mean()),
            "emp_rate": float(labels[mask].mean()),
        })
    return out


# ── main ─────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True,
                   help="Path to direction head manifest dir")
    p.add_argument("--eval-dates", required=True,
                   help="Comma-separated YYYY-MM-DD list")
    p.add_argument("--label-stem", default="horizon60_thresh5_fc60")
    p.add_argument("--oracle-root", default="data/oracle_cache_v2")
    p.add_argument("--label-root", default="data/direction_labels")
    args = p.parse_args()

    manifest_dir = Path(args.manifest)
    head, manifest = _load_head_from_manifest(manifest_dir)
    print(f"Loaded head: {manifest['experiment_id']}")
    print(
        f"  arch: input_dim={manifest['architecture']['input_dim']} "
        f"hidden={manifest['architecture']['hidden_dims']}"
    )

    training_dates = set(manifest["training"]["training_dates"])
    eval_dates = [d.strip() for d in args.eval_dates.split(",")]
    overlap = [d for d in eval_dates if d in training_dates]
    if overlap:
        raise ValueError(
            f"Eval dates {overlap} were in the head's TRAINING set "
            f"(per manifest). Pick eval dates the head has not seen "
            f"to get a clean held-out metric."
        )
    print(f"  eval dates (all held out): {eval_dates}")

    # Load + join across all eval dates.
    print()
    chunks = []
    for d in eval_dates:
        data = _load_day_obs_and_labels(
            d, args.label_stem,
            Path(args.oracle_root), Path(args.label_root),
        )
        if data["n"] == 0:
            print(f"  {d}: 0 joinable rows — skipping")
            continue
        print(
            f"  {d}: n={data['n']:>6d}  "
            f"back+={data['label_back'].mean()*100:.1f}%  "
            f"lay+={data['label_lay'].mean()*100:.1f}%"
        )
        chunks.append(data)
    if not chunks:
        print("ERROR: no eval data joined", file=sys.stderr)
        return 2
    X = np.concatenate([c["per_runner_obs"] for c in chunks], axis=0)
    Y_back = np.concatenate([c["label_back"] for c in chunks], axis=0)
    Y_lay = np.concatenate([c["label_lay"] for c in chunks], axis=0)
    n = X.shape[0]
    print(f"\nTotal joined: {n}")

    with torch.no_grad():
        Xt = torch.from_numpy(X)
        probs = torch.sigmoid(head(Xt)).numpy()
    p_back = probs[:, 0]
    p_lay = probs[:, 1]

    def report_side(side: str, p_pred: np.ndarray, y: np.ndarray) -> None:
        d = float(y.mean())
        const_05 = np.full_like(p_pred, 0.5)
        const_marg = np.full_like(p_pred, d)
        floor_05 = _bce_unweighted(const_05, y)
        floor_marg = _bce_unweighted(const_marg, y)
        bce_un = _bce_unweighted(p_pred, y)
        bce_pw = _bce_pos_weighted(p_pred, y, d)
        floor_05_pw = _bce_pos_weighted(const_05, y, d)
        rho = _pearson(p_pred, y)
        auc = _roc_auc(p_pred, y)
        brier = _brier(p_pred, y)
        brier_marg = _brier(const_marg, y)
        print()
        print(f"=== direction_{side} ===")
        print(f"  positive rate          = {d:.4f}")
        print(f"  output mean / std      = "
              f"{p_pred.mean():.4f} / {p_pred.std():.4f}")
        print(f"  output min / max       = "
              f"{p_pred.min():.4f} / {p_pred.max():.4f}")
        print(f"  BCE unweighted         = {bce_un:.4f}")
        print(f"   vs uniform-0.5 floor  = "
              f"{floor_05:.4f}  ({(floor_05-bce_un)/floor_05*100:+.1f}%)")
        print(f"   vs marginal-rate base = "
              f"{floor_marg:.4f}  ({(floor_marg-bce_un)/floor_marg*100:+.1f}%)")
        print(f"  BCE pos-weighted       = {bce_pw:.4f}  "
              f"(uniform-0.5 floor {floor_05_pw:.4f}, "
              f"{(floor_05_pw-bce_pw)/floor_05_pw*100:+.1f}%)")
        print(f"  Pearson(pred, label)   = {rho:+.4f}")
        print(f"  ROC AUC                = {auc:.4f}  "
              f"(0.5 = random, 1.0 = perfect)")
        print(f"  Brier score            = {brier:.4f}  "
              f"(marginal-base {brier_marg:.4f})")
        print()
        print("  Reliability (predicted prob bucket -> empirical rate):")
        print(f"  {'bucket':<14}{'n':>7}  {'mean_pred':>10}  "
              f"{'emp_rate':>9}  diagram")
        for row in _reliability_table(p_pred, y, n_buckets=10):
            if row["n"] == 0:
                print(f"  {row['bucket']:<14}{row['n']:>7}  "
                      f"{'-':>10}  {'-':>9}")
                continue
            # Tiny ASCII diagram comparing mean_pred (P) vs emp (E).
            scale = 30
            p_pos = int(round(row['mean_pred'] * scale))
            e_pos = int(round(row['emp_rate'] * scale))
            line = [' '] * (scale + 1)
            line[min(p_pos, scale)] = 'P'
            line[min(e_pos, scale)] = 'E'
            if line[min(p_pos, scale)] == line[min(e_pos, scale)] \
                    and p_pos == e_pos:
                line[min(p_pos, scale)] = '*'
            print(f"  {row['bucket']:<14}{row['n']:>7}  "
                  f"{row['mean_pred']:>10.3f}  "
                  f"{row['emp_rate']:>9.3f}  |{''.join(line)}|")

    report_side("back", p_back, Y_back)
    report_side("lay", p_lay, Y_lay)

    print()
    print("Verdict guide:")
    print("  * BCE below uniform-0.5 floor by >5%  → head has signal")
    print("  * BCE below marginal-rate baseline    → head is using")
    print("    features, not just predicting the marginal")
    print("  * Pearson > 0.10                      → meaningful correlation")
    print("  * ROC AUC > 0.55                       → discrimination > random")
    print("  * Reliability: P and E close per bucket → calibration OK")
    print("    P and E diverging in high-prob buckets → over- or under-")
    print("    confident; need calibration step or different loss.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
