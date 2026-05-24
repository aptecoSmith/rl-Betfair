"""Inspect what a trained agent's direction_prob_head is actually predicting.

Loads a saved agent's policy weights, runs a forward pass on a sample
of obs vectors from the v2 oracle cache, and reports:

1. Distribution of `direction_back_prob` / `direction_lay_prob` outputs
   across (tick, runner) samples — is the head outputting a flat 0.5
   or producing varied predictions?

2. Pearson correlation between head outputs and the ground-truth
   direction labels (from data/direction_labels/). Tests: "does the
   head's prediction actually track the label it's supervised on?"

3. Pearson correlation between head outputs and the betfair-predictors
   direction model's `dir_fire_drift` / `dir_fire_shorten` columns
   that are ALREADY in obs. Tests: "is the head learning to ECHO the
   offline predictor (the easy fit) or doing something else?"

Together these distinguish three failure modes:

  - Head stuck at 0.5: low std on outputs; corr with label and obs
    columns both near 0. Head is doing nothing.
  - Head echoing the predictor: high corr with `dir_fire_*` obs
    columns; corr with label tracks however well the offline predictor
    matches our offline labels.
  - Head learning task-specific features: high corr with label but
    NOT with `dir_fire_*` obs columns. Head is doing its own thing
    beyond echo.

Usage:
    python tools/direction_head_inspection.py <cohort_dir> [--agent <id>] [--date <YYYY-MM-DD>]

`--agent` defaults to the first agent_id with weights saved in the
cohort dir. `--date` defaults to the first training day with both an
oracle cache (obs vectors) and a direction-labels cache.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Script lives in tools/, but needs to import agents_v2/, env/,
# training_v2/ from the repo root. Insert repo root into sys.path
# unconditionally so `python tools/direction_head_inspection.py ...`
# works as well as `python -m tools.direction_head_inspection ...`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np


# ── obs-column resolution ────────────────────────────────────────────


# The 23-column lean per-runner block (LEAN_RUNNER_KEYS in
# env/betfair_env.py). We need the indices of dir_fire_drift +
# dir_fire_shorten within this block so we can pull them out of the
# global obs vector.
LEAN_RUNNER_KEYS = [
    "ltp", "available_to_back_price_0", "available_to_lay_price_0",
    "spread_ticks", "ltp_velocity_5tick", "ltp_velocity_30tick",
    "ltp_velocity_120tick",
    "champion_p_win", "champion_p_placed", "champion_segment_strong",
    "ranker_softmax_share", "ranker_top1_flag",
    "ranker_top1_high_conf_flag",
    "dir_q10_1m", "dir_q50_1m", "dir_q90_1m",
    "dir_q10_3m", "dir_q50_3m", "dir_q90_3m",
    "dir_q10_7m", "dir_q50_7m", "dir_q90_7m",
    "dir_fire_drift", "dir_fire_shorten", "dir_fire_no_signal",
]
DIR_FIRE_DRIFT_IN_RUNNER = LEAN_RUNNER_KEYS.index("dir_fire_drift")
DIR_FIRE_SHORTEN_IN_RUNNER = LEAN_RUNNER_KEYS.index("dir_fire_shorten")

# All 12 direction-predictor columns inside the per-runner block.
# Used for brute-force "which obs column best predicts the label"
# scan — sidesteps the question of which exact column is what when
# obs is normalised.
DIR_PRED_COLS_IN_RUNNER = [
    (LEAN_RUNNER_KEYS.index(k), k)
    for k in LEAN_RUNNER_KEYS
    if k.startswith("dir_")
]  # 12 entries: (col_idx_in_runner_block, name)


# The global obs vector layout (lean-obs + scalping mode), keyed by
# env constants from env/betfair_env.py:
#   [0:37]                       = MARKET_DIM
#   [37:48]                      = VELOCITY_DIM (= 11)
#   [48:48+R*23]                 = per-runner LEAN_RUNNER block × R
#   [48+23R : 48+23R+8]          = AGENT_STATE_DIM (6) + SCALPING (2)
#   [56+23R : 56+23R+12*R]       = per-runner POSITION (3) + SCALPING (9) blocks
# Plus shim appends 2*R scorer features → total shim.obs_dim = 56 + 37*R.
# For shim.obs_dim=574 the cohort's R=14.
MARKET_DIM = 37
VELOCITY_DIM = 11
PER_RUNNER_BLOCK_OFFSET = MARKET_DIM + VELOCITY_DIM  # 48


def runner_obs_slice(runner_idx: int, max_runners: int) -> slice:
    """Return the slice into the global obs vector that holds the
    per-runner LEAN_RUNNER block for ``runner_idx``.
    """
    per_runner_len = len(LEAN_RUNNER_KEYS)
    start = PER_RUNNER_BLOCK_OFFSET + runner_idx * per_runner_len
    return slice(start, start + per_runner_len)


def infer_max_runners(obs_dim: int) -> int:
    """Recover max_runners from shim.obs_dim using `obs_dim = 56 + 37*R`.

    The formula derives from env/betfair_env.py:1815 +
    agents_v2/env_shim.py:222 (obs_dim = env.obs_dim + 2*N).
    """
    # 56 = MARKET_DIM(37) + VELOCITY_DIM(11) +
    #      AGENT_STATE_DIM(6) + SCALPING_AGENT_STATE_DIM(2)
    # 37 per runner = LEAN_RUNNER_DIM(23) + POSITION_DIM(3) +
    #                 SCALPING_POSITION_DIM(9) + shim scorer(2)
    base, per_runner = 56, 37
    n = (obs_dim - base) / per_runner
    if not n.is_integer() or n <= 0:
        raise ValueError(
            f"Cannot recover max_runners from obs_dim={obs_dim} "
            f"(formula 56 + 37*R doesn't fit)."
        )
    return int(n)


# ── Data loading ─────────────────────────────────────────────────────


def load_oracle_obs(date: str, oracle_root: Path) -> dict | None:
    p = oracle_root / date / "oracle_samples.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return {
        "tick_index": d["tick_index"].astype(np.int64),
        "runner_idx": d["runner_idx"].astype(np.int64),
        "obs": d["obs"].astype(np.float32),  # (N, 574)
    }


def load_direction_labels(date: str, label_root: Path) -> dict | None:
    p = label_root / date / "horizon60_thresh5_fc60.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return {
        "tick_index": d["tick_index"].astype(np.int64),
        "runner_idx": d["runner_idx"].astype(np.int64),
        "label_back": d["label_back"].astype(np.float32),
        "label_lay": d["label_lay"].astype(np.float32),
    }


def inner_join(oracle: dict, labels: dict) -> dict:
    """Inner-join oracle samples and labels on (tick_index, runner_idx)."""
    okeys = (oracle["tick_index"] << 16) | oracle["runner_idx"]
    lkeys = (labels["tick_index"] << 16) | labels["runner_idx"]
    label_row = {int(k): i for i, k in enumerate(lkeys)}
    keep_o: list[int] = []
    keep_l: list[int] = []
    for i, k in enumerate(okeys):
        j = label_row.get(int(k))
        if j is not None:
            keep_o.append(i)
            keep_l.append(j)
    if not keep_o:
        return {"n": 0}
    oi = np.array(keep_o, dtype=np.int64)
    li = np.array(keep_l, dtype=np.int64)
    return {
        "n": len(keep_o),
        "obs": oracle["obs"][oi],
        "runner_idx": oracle["runner_idx"][oi],
        "label_back": labels["label_back"][li],
        "label_lay": labels["label_lay"][li],
    }


# ── Stats helpers ────────────────────────────────────────────────────


def pearson(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2:
        return float("nan")
    mx, my = float(xs.mean()), float(ys.mean())
    num = float(((xs - mx) * (ys - my)).sum())
    dx = float(np.sqrt(((xs - mx) ** 2).sum()))
    dy = float(np.sqrt(((ys - my) ** 2).sum()))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def fmt_hist(vals: np.ndarray, bins: int = 10) -> str:
    """Text histogram, returns a tiny ASCII bar chart."""
    if vals.size == 0:
        return "(empty)"
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        hi = lo + 1e-9
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(vals, bins=edges)
    peak = counts.max() if counts.size else 1
    lines = []
    for i in range(bins):
        bar_len = int(round(counts[i] / peak * 40)) if peak else 0
        lines.append(
            f"  [{edges[i]:.3f}, {edges[i+1]:.3f})  "
            f"{counts[i]:>8d}  "
            + "#" * bar_len
        )
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────


def find_default_agent(cohort_dir: Path) -> str | None:
    weights_dir = cohort_dir / "weights"
    if not weights_dir.exists():
        return None
    pts = sorted(weights_dir.glob("*.pt"))
    if not pts:
        return None
    return pts[0].stem


def find_default_date(
    oracle_root: Path, label_root: Path,
) -> str | None:
    for d in sorted(oracle_root.iterdir()):
        if not d.is_dir():
            continue
        date = d.name
        if (label_root / date / "horizon60_thresh5_fc60.npz").exists():
            if (d / "oracle_samples.npz").exists():
                return date
    return None


def load_agent_genes(cohort_dir: Path, agent_id: str) -> dict | None:
    """Load the agent's gene dict from scoreboard.jsonl."""
    sb = cohort_dir / "scoreboard.jsonl"
    if not sb.exists():
        return None
    with sb.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("agent_id", "").startswith(agent_id):
                return row.get("hyperparameters")
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cohort_dir")
    p.add_argument("--agent", default=None,
                   help="agent_id (UUID); defaults to first weights file")
    p.add_argument("--date", default=None,
                   help="YYYY-MM-DD; defaults to first available cache day")
    p.add_argument("--oracle-root", default="data/oracle_cache_v2")
    p.add_argument("--label-root", default="data/direction_labels")
    p.add_argument("--max-runners", type=int, default=None,
                   help="Override max-runners. Default: inferred from "
                        "the oracle cache obs_dim (typically 14 for "
                        "shim.obs_dim=574).")
    p.add_argument("--samples", type=int, default=20000,
                   help="Cap on rows used (random subsample if exceeded).")
    args = p.parse_args()

    cohort_dir = Path(args.cohort_dir)
    if not cohort_dir.exists():
        print(f"ERROR: cohort dir {cohort_dir} not found", file=sys.stderr)
        return 2

    agent_id = args.agent or find_default_agent(cohort_dir)
    if agent_id is None:
        print(
            f"ERROR: no .pt weights found under {cohort_dir}/weights/ — "
            "agent may not have finished yet.",
            file=sys.stderr,
        )
        return 2
    weights_path = cohort_dir / "weights" / f"{agent_id}.pt"
    if not weights_path.exists():
        # Tolerate prefix-only id (matches first agent whose id starts with it).
        cands = sorted((cohort_dir / "weights").glob(f"{agent_id}*.pt"))
        if not cands:
            print(
                f"ERROR: weights file {weights_path} not found",
                file=sys.stderr,
            )
            return 2
        weights_path = cands[0]
        agent_id = weights_path.stem
    print(f"agent_id: {agent_id}")
    print(f"weights:  {weights_path}")

    date = args.date or find_default_date(
        Path(args.oracle_root), Path(args.label_root),
    )
    if date is None:
        print("ERROR: no usable date found in caches", file=sys.stderr)
        return 2
    print(f"date:     {date}")

    # ── Genes ──────────────────────────────────────────────────────
    genes = load_agent_genes(cohort_dir, agent_id[:8])
    if genes:
        print()
        print("Agent genes (relevant):")
        for k in [
            "hidden_size", "direction_prob_loss_weight",
            "bc_direction_target_weight",
            "direction_gate_threshold",
        ]:
            print(f"  {k:>32}: {genes.get(k)}")

    # ── Data ───────────────────────────────────────────────────────
    print()
    print("Loading obs + labels...")
    oracle = load_oracle_obs(date, Path(args.oracle_root))
    if oracle is None:
        print(f"ERROR: oracle cache missing for {date}", file=sys.stderr)
        return 2
    labels = load_direction_labels(date, Path(args.label_root))
    if labels is None:
        print(f"ERROR: direction labels missing for {date}", file=sys.stderr)
        return 2
    joined = inner_join(oracle, labels)
    if joined["n"] == 0:
        print(f"ERROR: no overlap between oracle + labels on {date}",
              file=sys.stderr)
        return 2

    # Subsample.
    n_all = joined["n"]
    if n_all > args.samples:
        rng = np.random.default_rng(42)
        sel = rng.choice(n_all, size=args.samples, replace=False)
        obs = joined["obs"][sel]
        runner_idx = joined["runner_idx"][sel]
        label_back = joined["label_back"][sel]
        label_lay = joined["label_lay"][sel]
    else:
        obs = joined["obs"]
        runner_idx = joined["runner_idx"]
        label_back = joined["label_back"]
        label_lay = joined["label_lay"]
    n = obs.shape[0]
    print(f"using {n} joined samples (of {n_all} on {date})")

    # ── Pull obs columns ───────────────────────────────────────────
    max_runners = (
        int(args.max_runners) if args.max_runners is not None
        else infer_max_runners(obs.shape[1])
    )
    print(f"max_runners (inferred from obs_dim): {max_runners}")
    dir_drift_obs = np.empty(n, dtype=np.float32)
    dir_shorten_obs = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = runner_obs_slice(int(runner_idx[i]), max_runners)
        per_runner_block = obs[i, s]
        dir_drift_obs[i] = per_runner_block[DIR_FIRE_DRIFT_IN_RUNNER]
        dir_shorten_obs[i] = per_runner_block[DIR_FIRE_SHORTEN_IN_RUNNER]

    # ── Policy forward ─────────────────────────────────────────────
    print()
    print("Loading policy + running forward pass...")
    try:
        import torch
        from agents_v2.discrete_policy import DiscreteLSTMPolicy
        from agents_v2.env_shim import DEFAULT_SCORER_DIR
        from training_v2.cohort.worker import (
            _build_env_for_day, scalping_train_config,
        )
    except ImportError as e:
        print(f"ERROR: import failed ({e}) — run inside the repo venv",
              file=sys.stderr)
        return 2

    hidden_size = int(genes.get("hidden_size", 256)) if genes else 256

    # Build an env for the date to recover obs_dim + action_space.
    cfg = scalping_train_config()
    env, shim = _build_env_for_day(
        day_str=date,
        data_dir=Path("data/processed"),
        cfg=cfg,
        scorer_dir=DEFAULT_SCORER_DIR,
        predictor_lean_obs=True,
    )
    policy = DiscreteLSTMPolicy(
        obs_dim=shim.obs_dim,
        action_space=shim.action_space,
        hidden_size=hidden_size,
    )
    state = torch.load(
        weights_path, weights_only=True, map_location="cpu",
    )
    if isinstance(state, dict) and "weights" in state:
        state = state["weights"]
    policy.load_state_dict(state, strict=True)
    policy.eval()

    obs_t = torch.from_numpy(obs)
    # init_hidden() returns architecture-specific tuple. For LSTM that's
    # (h, c) shape (num_layers, batch, hidden). We want batch=N here.
    hidden = policy.init_hidden(batch=n)
    with torch.no_grad():
        out = policy(obs_t, hidden_state=hidden)
    # Both shape (N, max_runners). Pull each row's prediction at the
    # corresponding runner_idx.
    back_pred_all = out.direction_back_prob_per_runner.cpu().numpy()
    lay_pred_all = out.direction_lay_prob_per_runner.cpu().numpy()
    rows = np.arange(n)
    back_pred = back_pred_all[rows, runner_idx]
    lay_pred = lay_pred_all[rows, runner_idx]

    # ── Report ─────────────────────────────────────────────────────
    def report_side(
        side: str,
        head_out: np.ndarray,
        label: np.ndarray,
        obs_col: np.ndarray,
        obs_col_name: str,
    ) -> None:
        print(f"\n=== direction_{side}_prob_head ===")
        print(
            f"  output mean={head_out.mean():.4f}  "
            f"std={head_out.std():.4f}  "
            f"min={head_out.min():.4f}  max={head_out.max():.4f}"
        )
        print(f"  label rate={label.mean():.4f}")
        print(f"  obs[{obs_col_name}] rate={obs_col.mean():.4f}")
        rho_label = pearson(head_out, label)
        rho_obs = pearson(head_out, obs_col)
        rho_obs_label = pearson(obs_col, label)
        print(f"  rho(head_out, label)            = {rho_label:+.4f}")
        print(
            f"  rho(head_out, obs[{obs_col_name}]) = {rho_obs:+.4f}  "
            "(is head echoing the predictor?)"
        )
        print(
            f"  rho(obs[{obs_col_name}], label)    = {rho_obs_label:+.4f}  "
            "(does predictor match label?)"
        )
        print()
        print("  Histogram of head outputs:")
        print(fmt_hist(head_out, bins=12))

    report_side(
        "back", back_pred, label_back, dir_shorten_obs, "dir_fire_shorten",
    )
    report_side(
        "lay", lay_pred, label_lay, dir_drift_obs, "dir_fire_drift",
    )

    # ── Brute-force: correlation of each direction obs col vs label ──
    # Sidesteps the "is my column index right?" question. Pulls all 12
    # direction-predictor obs columns for each (sample, runner) and
    # reports which column best correlates with each label. The best
    # one is the ceiling the internal head could hit by perfect echo.
    print()
    print("Brute-force scan: correlation of each of the 12 direction")
    print("obs columns vs label_back / label_lay. The 'best' column is")
    print("the upper bound an echo-only head could achieve.\n")
    print(f"  {'obs column':<20} {'vs label_back':>15} {'vs label_lay':>15}")
    print("-" * 56)
    obs_cols = np.empty((n, len(DIR_PRED_COLS_IN_RUNNER)), dtype=np.float32)
    for i in range(n):
        s = runner_obs_slice(int(runner_idx[i]), max_runners)
        block = obs[i, s]
        for j, (col_idx, _) in enumerate(DIR_PRED_COLS_IN_RUNNER):
            obs_cols[i, j] = block[col_idx]
    for j, (_, name) in enumerate(DIR_PRED_COLS_IN_RUNNER):
        col = obs_cols[:, j]
        r_b = pearson(col, label_back)
        r_l = pearson(col, label_lay)
        print(f"  {name:<20} {r_b:>+15.4f} {r_l:>+15.4f}")

    # Also report correlation of head outputs vs label using the BEST
    # of the 12 columns (the column the head "should" learn to echo).
    best_back_col_idx = int(
        np.argmax(np.abs([
            pearson(obs_cols[:, j], label_back)
            for j in range(len(DIR_PRED_COLS_IN_RUNNER))
        ]))
    )
    best_lay_col_idx = int(
        np.argmax(np.abs([
            pearson(obs_cols[:, j], label_lay)
            for j in range(len(DIR_PRED_COLS_IN_RUNNER))
        ]))
    )
    print()
    print(
        f"  Best for label_back: {DIR_PRED_COLS_IN_RUNNER[best_back_col_idx][1]} "
        f"(rho={pearson(obs_cols[:, best_back_col_idx], label_back):+.4f})"
    )
    print(
        f"  Best for label_lay:  {DIR_PRED_COLS_IN_RUNNER[best_lay_col_idx][1]} "
        f"(rho={pearson(obs_cols[:, best_lay_col_idx], label_lay):+.4f})"
    )
    print(
        f"  rho(head_back, best_back_col) = "
        f"{pearson(back_pred, obs_cols[:, best_back_col_idx]):+.4f}"
    )
    print(
        f"  rho(head_lay,  best_lay_col)  = "
        f"{pearson(lay_pred, obs_cols[:, best_lay_col_idx]):+.4f}"
    )

    # ── Verdict ────────────────────────────────────────────────────
    print()
    print("Verdict guide:")
    print("  * std(head) < 0.05: head is stuck near a constant (~0.5)")
    print("  * |rho(head,obs_dir_fire)| > 0.3: head IS echoing the")
    print("    pretrained predictor that's already in obs")
    print("  * |rho(head,label)| > 0.1 but |rho(head,obs)| < 0.1: head")
    print("    is doing something beyond just echoing — task-specific")
    print("  * rho(obs,label) tells you what the offline predictor's")
    print("    correlation with our offline labels is — i.e. the")
    print("    ceiling the internal head could hit by perfect echo.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
