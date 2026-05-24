"""Train ONE shared direction-prediction head offline, save weights
+ manifest, then it can be loaded frozen by every cohort agent.

Replaces the per-agent ``direction_prob_head`` that was being
re-trained inside each cohort's PPO loop (slow, gradient-interfered,
12× redundant). See ``plans/shared-direction-head/``.

Usage::

    python -m scripts.train_direction_head \
        --training-dates 2026-04-06,2026-04-08,... \
        --output-dir models/direction_head/v1_2026-05-24 \
        --hidden 64 --epochs 50 --lr 1e-3

Pre-flight: rejects any training-date that overlaps the cohort's
hardcoded eval-day or monitor-day lists (the held-out invariant).
The cohort's eval / monitor lists are pinned in this file so
edits to the cohort config force a re-think of this list.

Outputs land at the directory passed via ``--output-dir`` with:

* ``weights.pt`` — the head's ``state_dict()`` (NOT the full policy)
* ``manifest.json`` — see ``plans/shared-direction-head/
  hard_constraints.md §5`` for the schema
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env.betfair_env import (
    LEAN_RUNNER_DIM,
    LEAN_RUNNER_KEYS,
    MARKET_DIM,
    OBS_SCHEMA_VERSION,
    VELOCITY_DIM,
)


# Held-out-day invariant (purpose.md §"Hold-out invariants",
# hard_constraints.md §1). Edit these IFF the cohort's eval /
# monitor pool changes, and re-train the head.
COHORT_EVAL_DAYS: frozenset[str] = frozenset({
    "2026-04-07", "2026-04-10", "2026-04-14", "2026-04-17",
    "2026-04-21", "2026-04-23", "2026-04-25",
    "2026-05-01", "2026-05-03", "2026-05-06",
})
COHORT_MONITOR_DAYS: frozenset[str] = frozenset({
    "2026-05-07", "2026-05-08", "2026-05-09", "2026-05-10",
    "2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14",
    "2026-05-15", "2026-05-16", "2026-05-17", "2026-05-18",
    "2026-05-19", "2026-05-20",
})

LABEL_VERSION = "v1_threshold_crossing"
DEFAULT_HORIZON_TICKS = 60
DEFAULT_THRESHOLD_TICKS = 5
DEFAULT_FC_BEFORE_OFF_SECONDS = 60.0


def _git_sha() -> str:
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii").strip()
    except Exception:
        return "unknown"


def _verify_no_holdout_leak(training_dates: list[str]) -> None:
    """Hard guard: training_dates MUST NOT overlap cohort eval /
    monitor pools. Aborts with a clear error if any leak is found.
    """
    eval_leak = [d for d in training_dates if d in COHORT_EVAL_DAYS]
    monitor_leak = [d for d in training_dates if d in COHORT_MONITOR_DAYS]
    if eval_leak or monitor_leak:
        msg = (
            "Held-out-day leak detected — refusing to train the "
            "shared direction head on data the cohort will evaluate on."
            f"\n  Eval-day leaks: {eval_leak}"
            f"\n  Monitor-day leaks: {monitor_leak}"
            "\nUpdate --training-dates to exclude these, or update "
            "COHORT_EVAL_DAYS / COHORT_MONITOR_DAYS in this script "
            "if the cohort config has changed."
        )
        raise ValueError(msg)


def _load_day_obs_and_labels(
    date: str,
    oracle_root: Path,
    label_root: Path,
) -> dict:
    """Join oracle obs cache with direction labels for a single date,
    returning per-runner obs slices + labels.

    Returns dict with keys:
      - per_runner_obs: (N, 23) float32
      - label_back:     (N,) float32
      - label_lay:      (N,) float32
    """
    oracle_path = oracle_root / date / "oracle_samples.npz"
    label_path = (
        label_root / date / "horizon60_thresh5_fc60.npz"
    )
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
            f"{OBS_SCHEMA_VERSION}. Re-scan the oracle cache.",
        )
    obs_all = o["obs"]  # (N, obs_dim)
    tick_index = o["tick_index"].astype(np.int64)
    runner_idx = o["runner_idx"].astype(np.int64)

    l = np.load(label_path)
    lbl_tick = l["tick_index"].astype(np.int64)
    lbl_runner = l["runner_idx"].astype(np.int64)
    label_back = l["label_back"].astype(np.float32)
    label_lay = l["label_lay"].astype(np.float32)

    # Inner-join on (tick_index, runner_idx).
    ok = (tick_index << 16) | runner_idx
    lk = (lbl_tick << 16) | lbl_runner
    lookup = {int(k): i for i, k in enumerate(lk)}
    keep = [(i, lookup[int(k)]) for i, k in enumerate(ok)
            if int(k) in lookup]
    if not keep:
        raise ValueError(
            f"{date}: no joinable (tick, runner) pairs between "
            "oracle cache and direction labels.",
        )
    oi = np.array([p[0] for p in keep], dtype=np.int64)
    li = np.array([p[1] for p in keep], dtype=np.int64)

    obs_joined = obs_all[oi]
    runner_joined = runner_idx[oi]
    lb = label_back[li]
    ll = label_lay[li]

    # Sanity: predictor cols must NOT be all-zero (the 2026-05-24
    # bug class — scanning the oracle without --use-direction-
    # predictor leaves these columns zero-filled).
    q50_7m_in_runner = LEAN_RUNNER_KEYS.index("dir_q50_7m")
    n = obs_joined.shape[0]
    max_runners = (obs_joined.shape[1] - 56) // 37
    sample_q50 = np.empty(min(n, 1024), dtype=np.float32)
    for i in range(min(n, 1024)):
        s = (
            MARKET_DIM + VELOCITY_DIM
            + int(runner_joined[i]) * LEAN_RUNNER_DIM
        )
        sample_q50[i] = obs_joined[i, s + q50_7m_in_runner]
    if float(sample_q50.std()) < 1e-6:
        raise ValueError(
            f"{date}: dir_q50_7m std ~ 0 across 1024 sample obs — "
            "oracle cache was scanned WITHOUT "
            "--use-direction-predictor. Re-scan first."
        )

    # Extract per-runner block for each (tick, runner) sample.
    per_runner = np.empty((n, LEAN_RUNNER_DIM), dtype=np.float32)
    for i in range(n):
        s = (
            MARKET_DIM + VELOCITY_DIM
            + int(runner_joined[i]) * LEAN_RUNNER_DIM
        )
        per_runner[i] = obs_joined[i, s:s + LEAN_RUNNER_DIM]

    return {
        "per_runner_obs": per_runner,
        "label_back": lb,
        "label_lay": ll,
        "n_samples": n,
        "max_runners": int(max_runners),
    }


class _SkipHead(nn.Module):
    """C10 architecture: skip connection from raw input to penultimate
    layer so the output linear sees BOTH the hidden activation and the
    raw 23-d input. Tests whether the head needs direct access to
    linear features that a single hidden layer might be approximating.
    """

    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden)
        # Final layer sees concat([hidden activation, normalised raw input]).
        self.fc_out = nn.Linear(hidden + input_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = self.layer_norm(x)
        h = torch.relu(self.fc1(x_n))
        cat = torch.cat([h, x_n], dim=-1)
        return self.fc_out(cat)


class _PairwiseHead(nn.Module):
    """C15 architecture: expands the 23-d input to
    concat([x, flatten(outer_product(x, x))]) = 23 + 23*23 = 552
    features before a standard 2-hidden-layer MLP. Tests whether
    feature expressiveness — not architecture capacity — is the
    binding ceiling, without changing the policy's per-runner
    call-site contract.

    The outer-product matrix is symmetric (x_i*x_j == x_j*x_i) so
    half its entries are redundant; the post-expansion Linear
    layer learns to ignore the redundancy. Carrying the full square
    keeps the forward pass branchless / vectorised.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
    ) -> None:
        super().__init__()
        assert len(hidden_dims) == 2, \
            "C15 currently hardwires a 2-hidden-layer MLP after expansion"
        expanded_dim = input_dim + input_dim * input_dim
        self.input_dim = input_dim
        self.expanded_dim = expanded_dim
        self.net = nn.Sequential(
            nn.LayerNorm(expanded_dim),
            nn.Linear(expanded_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., input_dim). Pairwise outer product: (..., input_dim,
        # input_dim), flattened to (..., input_dim*input_dim).
        outer = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(-2)
        cat = torch.cat([x, outer], dim=-1)
        return self.net(cat)


def _act_for_variant(variant: str) -> type[nn.Module]:
    """Pick the activation module class for a variant. Only C19
    differs from ReLU (uses GELU); every other variant uses ReLU
    so the round-4 ablations are minimal-delta against C11.
    """
    if variant == "c19":
        return nn.GELU
    return nn.ReLU


class DirectionHead(nn.Module):
    """Variant-aware direction head.

    Variants (architecture-sweep, 2026-05-24, round 1):

    * ``c0`` — LayerNorm -> Linear(input, 64) -> ReLU -> Linear(64, 2).
      Original v1 architecture. ``hidden_dims=[64]``.
    * ``c1`` — LayerNorm -> Linear(input, 256) -> ReLU -> Linear(256, 2).
      4× width. ``hidden_dims=[256]``.
    * ``c2`` — LayerNorm -> Linear(input, 64) -> ReLU -> Linear(64, 32)
      -> ReLU -> Linear(32, 2). Deeper, narrower second layer.
      ``hidden_dims=[64, 32]``.
    * ``c3`` — Same arch as ``c0``; trained with ``pos_weight=1`` instead
      of class-balanced. ``hidden_dims=[64]``.
    * ``c4`` — LayerNorm -> Linear(input, 128) -> BatchNorm1d -> ReLU
      -> Dropout(p) -> Linear(128, 2). ``hidden_dims=[128]``;
      ``dropout=p``.

    Round 2 (asked after round 1 showed C1 marginally beat C0):

    * ``c6`` — Same as ``c1`` but ``hidden_dims=[512]``. Width keep
      helping past 256?
    * ``c7`` — Same as ``c1`` but ``hidden_dims=[1024]``. Where's the
      overfit point?
    * ``c8`` — Same architecture as ``c1`` (``hidden_dims=[256]``) but
      trained with ``pos_weight=1`` (combines C1's wider lift + C3's
      calibrated outputs).
    * ``c9`` — LayerNorm -> Linear(input, 256) -> ReLU -> Linear(256,
      128) -> ReLU -> Linear(128, 2). ``hidden_dims=[256, 128]``.
      Wider+deeper.
    * ``c10`` — LayerNorm + Linear(input, 256) + ReLU, then concat the
      hidden activation with the normalised raw input before the output
      linear (``Linear(256+23, 2)``). Skip from input to penultimate
      layer. ``hidden_dims=[256]``.

    Output: ``(batch, 2)`` raw logits — [direction_back_logit,
    direction_lay_logit]. Caller applies sigmoid for probabilities.
    """

    def __init__(
        self,
        input_dim: int,
        variant: str = "c0",
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.variant = variant
        if variant in ("c0", "c3"):
            hd = hidden_dims or [64]
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hd[0]),
                nn.ReLU(),
                nn.Linear(hd[0], 2),
            )
            self.hidden_dims = list(hd)
        elif variant in ("c1", "c6", "c7", "c8"):
            if variant == "c1":
                hd = hidden_dims or [256]
            elif variant == "c6":
                hd = hidden_dims or [512]
            elif variant == "c7":
                hd = hidden_dims or [1024]
            else:  # c8
                hd = hidden_dims or [256]
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hd[0]),
                nn.ReLU(),
                nn.Linear(hd[0], 2),
            )
            self.hidden_dims = list(hd)
        elif variant == "c2":
            hd = hidden_dims or [64, 32]
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hd[0]),
                nn.ReLU(),
                nn.Linear(hd[0], hd[1]),
                nn.ReLU(),
                nn.Linear(hd[1], 2),
            )
            self.hidden_dims = list(hd)
        elif variant == "c4":
            hd = hidden_dims or [128]
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hd[0]),
                nn.BatchNorm1d(hd[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hd[0], 2),
            )
            self.hidden_dims = list(hd)
        elif variant == "c9":
            hd = hidden_dims or [256, 128]
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hd[0]),
                nn.ReLU(),
                nn.Linear(hd[0], hd[1]),
                nn.ReLU(),
                nn.Linear(hd[1], 2),
            )
            self.hidden_dims = list(hd)
        elif variant == "c10":
            hd = hidden_dims or [256]
            # Skip from input to penultimate; not expressible as a
            # flat Sequential.
            self.net = _SkipHead(input_dim, hd[0])
            self.hidden_dims = list(hd)
        elif variant == "c11":
            # Same architecture as c9 but trained with pos_weight=1.
            hd = hidden_dims or [256, 128]
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hd[0]),
                nn.ReLU(),
                nn.Linear(hd[0], hd[1]),
                nn.ReLU(),
                nn.Linear(hd[1], 2),
            )
            self.hidden_dims = list(hd)
        elif variant in ("c12", "c14"):
            hd = hidden_dims or [256, 128, 64]
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hd[0]),
                nn.ReLU(),
                nn.Linear(hd[0], hd[1]),
                nn.ReLU(),
                nn.Linear(hd[1], hd[2]),
                nn.ReLU(),
                nn.Linear(hd[2], 2),
            )
            self.hidden_dims = list(hd)
        elif variant == "c13":
            hd = hidden_dims or [512, 256]
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hd[0]),
                nn.ReLU(),
                nn.Linear(hd[0], hd[1]),
                nn.ReLU(),
                nn.Linear(hd[1], 2),
            )
            self.hidden_dims = list(hd)
        elif variant == "c15":
            hd = hidden_dims or [256, 128]
            self.net = _PairwiseHead(input_dim, hd)
            self.hidden_dims = list(hd)
        elif variant in ("c16", "c17", "c18", "c19", "c20"):
            # Round-4 ablations on C11: same [256, 128] arch, only
            # the training recipe / activation differs per variant.
            # c19 swaps ReLU for GELU; all others use ReLU.
            hd = hidden_dims or [256, 128]
            Act = _act_for_variant(variant)
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hd[0]),
                Act(),
                nn.Linear(hd[0], hd[1]),
                Act(),
                nn.Linear(hd[1], 2),
            )
            self.hidden_dims = list(hd)
        else:
            raise ValueError(
                f"unknown variant {variant!r}; "
                f"expected one of c0,c1,c2,c3,c4,c6,c7,c8,c9,c10,"
                f"c11,c12,c13,c14,c15,c16,c17,c18,c19,c20"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--training-dates", required=True,
        help="Comma-separated YYYY-MM-DD list. Must NOT include any "
             "cohort eval or monitor day.",
    )
    p.add_argument(
        "--oracle-root", default="data/oracle_cache_v2",
    )
    p.add_argument(
        "--label-root", default="data/direction_labels",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Where to write weights.pt + manifest.json. Will be "
             "created if missing.",
    )
    p.add_argument(
        "--experiment-id", default=None,
        help="Optional id string. Defaults to a timestamp.",
    )
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument(
        "--variant", default="c0",
        choices=[
            "c0", "c1", "c2", "c3", "c4",
            "c6", "c7", "c8", "c9", "c10",
            "c11", "c12", "c13", "c14", "c15",
            "c16", "c17", "c18", "c19", "c20",
        ],
        help=(
            "Architecture variant for the head. c0=baseline 2-layer "
            "MLP (default). c1=wider single layer (256). c2=deeper "
            "(64->32). c3=baseline arch trained with pos_weight=1 "
            "(unweighted BCE). c4=128 + BatchNorm + Dropout(p). "
            "c6=hidden 512. c7=hidden 1024. c8=hidden 256 with "
            "pos_weight=1. c9=[256,128] deeper+wider. c10=hidden 256 "
            "with skip from input to penultimate layer. c11=c9 arch "
            "with pos_weight=1. c12=[256,128,64] depth-3. "
            "c13=[512,256] wider+deeper. c14=c12 arch with "
            "pos_weight=1. c15=pairwise feature expansion (23->552) "
            "then [256,128] MLP. c16=c11+AdamW(wd=1e-3). c17=c11+focal "
            "loss(gamma=2). c18=c11+epochs=200+patience=20 (longer "
            "training). c19=c11+GELU activation. c20=c11+label "
            "smoothing(0.05)."
        ),
    )
    p.add_argument(
        "--dropout", type=float, default=0.2,
        help="Dropout probability (used by variant c4 only).",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--val-frac", type=float, default=0.20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    # Round-4 training-recipe knobs. Each round-4 variant pins exactly
    # one of these to its non-default value; all other variants leave
    # them at the defaults so behaviour is byte-identical to round 3.
    p.add_argument(
        "--optimizer", default="adam", choices=["adam", "adamw"],
        help="Optimizer choice. c16 forces adamw with --weight-decay 1e-3.",
    )
    p.add_argument(
        "--weight-decay", type=float, default=0.0,
        help="AdamW weight_decay coefficient. c16 forces 1e-3.",
    )
    p.add_argument(
        "--loss", default="bce", choices=["bce", "focal"],
        help=(
            "Per-sample loss. 'bce' is the default pos-weighted (or "
            "unweighted) binary cross-entropy. 'focal' uses focal "
            "loss with --focal-gamma; pos_weight is dropped under "
            "focal since focal handles imbalance via the focusing "
            "term. c17 forces focal."
        ),
    )
    p.add_argument(
        "--focal-gamma", type=float, default=2.0,
        help="Focal loss gamma parameter (used with --loss focal).",
    )
    p.add_argument(
        "--label-smoothing", type=float, default=0.0,
        help=(
            "Label smoothing factor alpha in [0, 0.5). Hard targets "
            "y in {0,1} are mapped to y*(1-2a)+a, equivalent to soft "
            "targets that never reach exactly 0 or 1. c20 forces "
            "alpha=0.05."
        ),
    )
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    # Round-4 variant-driven recipe overrides. Each round-4 variant
    # pins exactly one knob away from the C11 baseline so the ablation
    # is clean. The override is applied AFTER argparse so CLI flags
    # alone still produce a non-overridden run (useful for sanity).
    if args.variant == "c16":
        args.optimizer = "adamw"
        if args.weight_decay == 0.0:
            args.weight_decay = 1e-3
    elif args.variant == "c17":
        args.loss = "focal"
    elif args.variant == "c18":
        # c18's question: is C11 under-trained at 50 epochs? Let it run
        # up to 200 with a wider patience window.
        if args.epochs == 50:
            args.epochs = 200
        if args.patience == 5:
            args.patience = 20
    elif args.variant == "c19":
        # GELU is wired in via DirectionHead at construction; nothing
        # to override on args.
        pass
    elif args.variant == "c20":
        if args.label_smoothing == 0.0:
            args.label_smoothing = 0.05

    training_dates = [d.strip() for d in args.training_dates.split(",")]
    if not training_dates:
        print("ERROR: --training-dates is empty", file=sys.stderr)
        return 2

    _verify_no_holdout_leak(training_dates)

    print(f"Loading {len(training_dates)} training days...")
    chunks = []
    for d in sorted(training_dates):
        data = _load_day_obs_and_labels(
            d,
            oracle_root=Path(args.oracle_root),
            label_root=Path(args.label_root),
        )
        print(
            f"  {d}: n={data['n_samples']:>6d}  "
            f"max_runners={data['max_runners']}  "
            f"back+={data['label_back'].mean()*100:.1f}%  "
            f"lay+={data['label_lay'].mean()*100:.1f}%"
        )
        chunks.append(data)
    X = np.concatenate([c["per_runner_obs"] for c in chunks], axis=0)
    Y_back = np.concatenate([c["label_back"] for c in chunks], axis=0)
    Y_lay = np.concatenate([c["label_lay"] for c in chunks], axis=0)
    Y = np.stack([Y_back, Y_lay], axis=1)  # (N, 2)
    n_total = X.shape[0]
    print(f"\nTotal joined: {n_total} (per_runner_obs, label) pairs")

    # Shuffle + train/val split
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n_total)
    n_val = int(n_total * args.val_frac)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val")

    Xtr = torch.from_numpy(X[train_idx]).to(args.device)
    Ytr = torch.from_numpy(Y[train_idx]).to(args.device)
    Xva = torch.from_numpy(X[val_idx]).to(args.device)
    Yva = torch.from_numpy(Y[val_idx]).to(args.device)

    # Class-balance pos_weight per side. C3, C8, C11, C14 and all
    # round-4 ablations (c16-c20) train unweighted — round 4 isolates
    # ONE knob change from C11, so they all share its pos_weight=1.
    pos_rate_back = float(Y_back[train_idx].mean())
    pos_rate_lay = float(Y_lay[train_idx].mean())
    unweighted_variants = {
        "c3", "c8", "c11", "c14",
        "c16", "c17", "c18", "c19", "c20",
    }
    if args.variant in unweighted_variants:
        pw_back = 1.0
        pw_lay = 1.0
    else:
        pw_back = (1 - pos_rate_back) / max(pos_rate_back, 1e-9)
        pw_lay = (1 - pos_rate_lay) / max(pos_rate_lay, 1e-9)
    pw = torch.tensor(
        [pw_back, pw_lay], dtype=torch.float32, device=args.device,
    )
    print(
        f"pos rates: back={pos_rate_back:.4f} lay={pos_rate_lay:.4f}  "
        f"pos_weight: back={pw_back:.2f} lay={pw_lay:.2f}  "
        f"variant={args.variant}"
    )

    torch.manual_seed(args.seed)
    # For c0/c3, --hidden still controls the hidden dim (default 64).
    # For c1, c2, c4: variant supplies a fixed hidden_dims list.
    if args.variant in ("c0", "c3"):
        hidden_dims = [args.hidden]
    elif args.variant == "c1":
        hidden_dims = [256]
    elif args.variant == "c2":
        hidden_dims = [64, 32]
    elif args.variant == "c4":
        hidden_dims = [128]
    elif args.variant == "c6":
        hidden_dims = [512]
    elif args.variant == "c7":
        hidden_dims = [1024]
    elif args.variant == "c8":
        hidden_dims = [256]
    elif args.variant == "c9":
        hidden_dims = [256, 128]
    elif args.variant == "c10":
        hidden_dims = [256]
    elif args.variant == "c11":
        hidden_dims = [256, 128]
    elif args.variant in ("c12", "c14"):
        hidden_dims = [256, 128, 64]
    elif args.variant == "c13":
        hidden_dims = [512, 256]
    elif args.variant == "c15":
        hidden_dims = [256, 128]
    elif args.variant in ("c16", "c17", "c18", "c19", "c20"):
        hidden_dims = [256, 128]
    else:
        raise ValueError(f"unknown variant {args.variant!r}")
    head = DirectionHead(
        LEAN_RUNNER_DIM,
        variant=args.variant,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    ).to(args.device)
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(
            head.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
    else:
        opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    print(
        f"optimizer={args.optimizer} weight_decay={args.weight_decay}  "
        f"loss={args.loss} focal_gamma={args.focal_gamma}  "
        f"label_smoothing={args.label_smoothing}  "
        f"epochs={args.epochs} patience={args.patience}"
    )

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    def _compute_loss(
        logits: torch.Tensor, y: torch.Tensor,
    ) -> torch.Tensor:
        """Per-batch loss honouring the round-4 recipe knobs.

        * Label smoothing: maps hard targets ``y in {0, 1}`` to
          ``y*(1-2a) + a`` so neither extreme is ever a "correct"
          target. Compatible with both BCE and focal.
        * Focal loss: ``alpha=1``, ``gamma=args.focal_gamma``. Drops
          ``pos_weight`` (focal handles imbalance via the focusing
          term — combining them double-weights minority examples).
        * Plain BCE: pos-weighted by ``pw`` (which is [1,1] for the
          unweighted variants).
        """
        a = args.label_smoothing
        y_eff = y * (1.0 - 2.0 * a) + a if a > 0 else y
        if args.loss == "focal":
            # Numerically stable focal: compute p_t per-element from
            # logits and the (smoothed) target, then
            # focal = -((1 - p_t) ** gamma) * log(p_t).
            p = torch.sigmoid(logits)
            p_t = p * y_eff + (1.0 - p) * (1.0 - y_eff)
            # log_p_t in a stable way via per-element BCE
            bce_elem = F.binary_cross_entropy_with_logits(
                logits, y_eff, reduction="none",
            )
            focal = ((1.0 - p_t) ** args.focal_gamma) * bce_elem
            return focal.mean()
        return F.binary_cross_entropy_with_logits(
            logits, y_eff, pos_weight=pw,
        )

    n_train = len(train_idx)
    for epoch in range(args.epochs):
        head.train()
        # Shuffle each epoch.
        perm = torch.randperm(n_train, device=args.device)
        train_loss_sum = 0.0
        n_batches = 0
        for start in range(0, n_train, args.batch_size):
            end = min(start + args.batch_size, n_train)
            bi = perm[start:end]
            xb = Xtr[bi]
            yb = Ytr[bi]
            logits = head(xb)
            loss = _compute_loss(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += float(loss.item())
            n_batches += 1
        train_loss = train_loss_sum / max(n_batches, 1)

        head.eval()
        with torch.no_grad():
            va_logits = head(Xva)
            va_loss = F.binary_cross_entropy_with_logits(
                va_logits, Yva, pos_weight=pw,
            ).item()
            va_probs = torch.sigmoid(va_logits).cpu().numpy()
            va_y = Yva.cpu().numpy()

        # Unweighted per-side BCE for reporting (simple, comparable
        # to other parts of the project that quote unweighted BCE).
        eps = 1e-9
        va_probs_c = np.clip(va_probs, eps, 1 - eps)
        bce_back_unw = float(np.mean(
            -(va_y[:, 0] * np.log(va_probs_c[:, 0])
              + (1 - va_y[:, 0]) * np.log(1 - va_probs_c[:, 0]))
        ))
        bce_lay_unw = float(np.mean(
            -(va_y[:, 1] * np.log(va_probs_c[:, 1])
              + (1 - va_y[:, 1]) * np.log(1 - va_probs_c[:, 1]))
        ))
        print(
            f"epoch {epoch + 1:>2d}  train_loss={train_loss:.4f}  "
            f"val_loss={va_loss:.4f}  "
            f"val_bce_back={bce_back_unw:.4f}  "
            f"val_bce_lay={bce_lay_unw:.4f}"
        )

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    f"Early stop at epoch {epoch+1} "
                    f"(no improvement for {args.patience} epochs)"
                )
                break

    # Restore best
    if best_state is not None:
        head.load_state_dict(best_state)

    # Final val metrics (with the best-state head)
    head.eval()
    with torch.no_grad():
        va_logits = head(Xva)
        va_probs = torch.sigmoid(va_logits).cpu().numpy()
        va_y = Yva.cpu().numpy()
    eps = 1e-9
    va_probs_c = np.clip(va_probs, eps, 1 - eps)
    final_bce_back = float(np.mean(
        -(va_y[:, 0] * np.log(va_probs_c[:, 0])
          + (1 - va_y[:, 0]) * np.log(1 - va_probs_c[:, 0]))
    ))
    final_bce_lay = float(np.mean(
        -(va_y[:, 1] * np.log(va_probs_c[:, 1])
          + (1 - va_y[:, 1]) * np.log(1 - va_probs_c[:, 1]))
    ))

    # Save outputs.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "weights.pt"
    manifest_path = out_dir / "manifest.json"

    # Save the state_dict relative to the policy's
    # direction_prob_head module (so the cohort policy can call
    # ``policy.direction_prob_head.load_state_dict(...)`` directly).
    head_state = head.state_dict()
    # Strip the OUTER "net." prefix only — NOT a global replace.
    # Inner submodules (e.g. c15's `_PairwiseHead.net = Sequential`)
    # use the same name and would otherwise be over-stripped, leaving
    # the eval loader unable to map keys back to the model.
    _PREFIX = "net."
    flat_state = {
        (k[len(_PREFIX):] if k.startswith(_PREFIX) else k): v
        for k, v in head_state.items()
    }
    torch.save(flat_state, weights_path)

    experiment_id = (
        args.experiment_id
        or f"directionhead_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    family_by_variant = {
        "c0": "linear_mlp",
        "c1": "linear_mlp",
        "c2": "linear_mlp",
        "c3": "linear_mlp",
        "c4": "linear_mlp_bn_dropout",
        "c6": "linear_mlp",
        "c7": "linear_mlp",
        "c8": "linear_mlp",
        "c9": "linear_mlp",
        "c10": "linear_mlp_skip",
        "c11": "linear_mlp",
        "c12": "linear_mlp",
        "c13": "linear_mlp",
        "c14": "linear_mlp",
        "c15": "linear_mlp_pairwise",
        # Round-4 ablations all share C11's [256, 128] arch, only the
        # training recipe / activation differs.
        "c16": "linear_mlp",
        "c17": "linear_mlp",
        "c18": "linear_mlp",
        "c19": "linear_mlp",
        "c20": "linear_mlp",
    }
    manifest = {
        "experiment_id": experiment_id,
        "weights_path": "weights.pt",
        "architecture": {
            "family": family_by_variant[args.variant],
            "variant": args.variant,
            "input_dim": LEAN_RUNNER_DIM,
            "output_dim": 2,
            "hidden_dims": head.hidden_dims,
            "dropout": (
                args.dropout if args.variant == "c4" else 0.0
            ),
            "pos_weight_mode": (
                "unweighted"
                if args.variant in unweighted_variants
                else "balanced"
            ),
            "activation": "gelu" if args.variant == "c19" else "relu",
        },
        "training_recipe": {
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "loss": args.loss,
            "focal_gamma": (
                args.focal_gamma if args.loss == "focal" else None
            ),
            "label_smoothing": args.label_smoothing,
        },
        "training": {
            "training_dates": sorted(training_dates),
            "label_version": LABEL_VERSION,
            "direction_horizon_ticks": DEFAULT_HORIZON_TICKS,
            "direction_threshold_ticks": DEFAULT_THRESHOLD_TICKS,
            "force_close_before_off_seconds": DEFAULT_FC_BEFORE_OFF_SECONDS,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "val_frac": args.val_frac,
            "seed": args.seed,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
        },
        "val_metrics": {
            "val_bce_back": final_bce_back,
            "val_bce_lay": final_bce_lay,
            "pos_rate_back": pos_rate_back,
            "pos_rate_lay": pos_rate_lay,
        },
        "obs_schema_version": OBS_SCHEMA_VERSION,
        "active_runner_dim": LEAN_RUNNER_DIM,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": _git_sha(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print()
    print("=== DONE ===")
    print(f"Weights: {weights_path}")
    print(f"Manifest: {manifest_path}")
    print(
        f"Final val_bce: back={final_bce_back:.4f}  "
        f"lay={final_bce_lay:.4f}"
    )
    print(
        f"Acceptance (purpose.md): both <= 1.05 -> "
        f"{'PASS' if (final_bce_back <= 1.05 and final_bce_lay <= 1.05) else 'FAIL'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
