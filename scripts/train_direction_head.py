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


class DirectionHead(nn.Module):
    """Same architecture as the per-agent direction_prob_head in
    DiscreteLSTMPolicy: LayerNorm(input) -> Linear(input, hidden)
    -> ReLU -> Linear(hidden, 2).

    Output: (batch, 2) raw logits — [direction_back_logit,
    direction_lay_logit]. Caller applies sigmoid for probabilities.
    """

    def __init__(self, input_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
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
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--val-frac", type=float, default=0.20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

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

    # Class-balance pos_weight per side
    pos_rate_back = float(Y_back[train_idx].mean())
    pos_rate_lay = float(Y_lay[train_idx].mean())
    pw_back = (1 - pos_rate_back) / max(pos_rate_back, 1e-9)
    pw_lay = (1 - pos_rate_lay) / max(pos_rate_lay, 1e-9)
    pw = torch.tensor(
        [pw_back, pw_lay], dtype=torch.float32, device=args.device,
    )
    print(
        f"pos rates: back={pos_rate_back:.4f} lay={pos_rate_lay:.4f}  "
        f"pos_weight: back={pw_back:.2f} lay={pw_lay:.2f}"
    )

    torch.manual_seed(args.seed)
    head = DirectionHead(LEAN_RUNNER_DIM, hidden=args.hidden).to(args.device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None
    patience_counter = 0

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
            # pos-weighted BCE; broadcast pw across batch.
            loss = F.binary_cross_entropy_with_logits(
                logits, yb, pos_weight=pw,
            )
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
    # Strip the "net." prefix that comes from `self.net = nn.Sequential(...)`.
    flat_state = {
        k.replace("net.", ""): v for k, v in head_state.items()
    }
    torch.save(flat_state, weights_path)

    experiment_id = (
        args.experiment_id
        or f"directionhead_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    manifest = {
        "experiment_id": experiment_id,
        "weights_path": "weights.pt",
        "architecture": {
            "family": "linear_mlp",
            "input_dim": LEAN_RUNNER_DIM,
            "output_dim": 2,
            "hidden_dims": [args.hidden],
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
        f"Acceptance (purpose.md): both ≤ 1.05 → "
        f"{'PASS' if (final_bce_back <= 1.05 and final_bce_lay <= 1.05) else 'FAIL'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
