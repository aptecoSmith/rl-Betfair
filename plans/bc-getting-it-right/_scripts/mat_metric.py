"""Step A2/A3/A4 (bc-getting-it-right): the maturation-AUC metric harness.

THE METRIC (plan §1): held-out per-(tick, runner) maturation AUC — how
well a policy's ``mature_prob_head`` ranks "this open will mature" on the
7 reserved holdout days, vs the LightGBM 0.76 ceiling on the same data.

Library (imported by the Step B trainer):
  load_split(days, cache_dir)          -> obs, matured, runner_idx
  compute_norm_stats(obs, max_rows)    -> (mean, std) float64 per dim
  build_policy(obs_dim, hidden, ...)   -> DiscreteLSTMPolicy(input_norm=True)
  policy_mature_scores(policy, ...)    -> per-sample mature_prob @ its runner
  evaluate(scores, y)                  -> {auc, ap, top_decile_*, base_rate}
  pr_curve(scores, y, thresholds)      -> list of {threshold, opens, mat%, ...}

CLI baselines (plan A4 + GATE):
  --mode lgbm       reproduce the 0.76 LightGBM reference on the SAME cache
                    (sanity-checks the metric harness itself).
  --mode untrained  untrained policy mature_prob_head -> AUC ~0.5 (the head
                    is unsupervised at init; this is also the imitation-first
                    BC head's AUC since that BC never trained mature_prob).

GATE (plan A): harness reproduces ~0.76 from the LightGBM path AND cleanly
separates the untrained head (~0.5) from it -> the metric is trustworthy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agents_v2.action_space import DiscreteActionSpace  # noqa: E402
from agents_v2.discrete_policy import DiscreteLSTMPolicy  # noqa: E402
from env.betfair_env import RUNNER_DIM  # noqa: E402
from training_v2.arb_oracle import _load_config  # noqa: E402

DEFAULT_CACHE = Path("plans/bc-getting-it-right/_cache")
HOLDOUT_DAYS = [
    "2026-05-20", "2026-05-21", "2026-05-22", "2026-05-25",
    "2026-05-27", "2026-05-28", "2026-05-29",
]


def existing_train_days(n: int = 8) -> list[str]:
    import re
    pat = re.compile(r"^2026-\d{2}-\d{2}$")
    data_dir = Path("data/processed")
    days = sorted(
        p.stem for p in data_dir.glob("2026-*.parquet")
        if pat.match(p.stem) and "2026-04-06" <= p.stem <= "2026-05-19"
    )
    if len(days) <= n:
        return days
    idx = np.linspace(0, len(days) - 1, n).round().astype(int)
    return [days[i] for i in sorted(set(idx.tolist()))]


def all_train_days() -> list[str]:
    """Every EXISTING processed day in [04-06, 05-19] (the §2 train range)."""
    import re
    pat = re.compile(r"^2026-\d{2}-\d{2}$")
    data_dir = Path("data/processed")
    return sorted(
        p.stem for p in data_dir.glob("2026-*.parquet")
        if pat.match(p.stem) and "2026-04-06" <= p.stem <= "2026-05-19"
    )


# The 2 reserved BC-val days (§2: ~2 train days held out for early-stop).
# Latest two train days — closest to holdout, never overlapping it.
VAL_DAYS = ["2026-05-18", "2026-05-19"]


def resolve_days(spec: str) -> list[str]:
    if spec == "train8":
        return existing_train_days(8)
    if spec == "train6":
        return existing_train_days(8)[:6]
    if spec == "val2":
        return existing_train_days(8)[6:]
    if spec == "holdout":
        return list(HOLDOUT_DAYS)
    if spec == "alltrain":
        return all_train_days()
    if spec == "alltrain_bc":
        # All 42 train days minus the 2 BC-val days.
        return [d for d in all_train_days() if d not in VAL_DAYS]
    if spec == "alltrain_val":
        return list(VAL_DAYS)
    return [d.strip() for d in spec.split(",") if d.strip()]


# ── Data ────────────────────────────────────────────────────────────────────

def load_split(days: list[str], cache_dir: Path = DEFAULT_CACHE):
    """Concatenate cached (obs, matured, runner_idx) across days."""
    obs_l, mat_l, ri_l = [], [], []
    for d in days:
        p = Path(cache_dir) / f"{d}.npz"
        if not p.exists():
            raise FileNotFoundError(
                f"cache missing for {d}: {p} — run mat_dataset.py --days {d}")
        z = np.load(p)
        obs_l.append(z["obs"])
        mat_l.append(z["matured"].astype(np.int8))
        ri_l.append(z["runner_idx"].astype(np.int64))
    obs = np.concatenate(obs_l, axis=0)
    matured = np.concatenate(mat_l, axis=0)
    runner_idx = np.concatenate(ri_l, axis=0)
    return obs, matured, runner_idx


def compute_norm_stats(obs: np.ndarray, max_rows: int = 200_000, seed: int = 0):
    """Per-dim (mean, std) float64 from up to ``max_rows`` sampled train rows."""
    n = obs.shape[0]
    if n > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_rows, replace=False)
        src = obs[idx].astype(np.float64)
    else:
        src = obs.astype(np.float64)
    mean = src.mean(axis=0)
    std = src.std(axis=0)
    return mean, std


def save_norm_stats(mean, std, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=np.asarray(mean), std=np.asarray(std))


def load_norm_stats(path: Path):
    z = np.load(path)
    return z["mean"], z["std"]


# ── Policy scaffold ─────────────────────────────────────────────────────────

def max_runners_from_config() -> int:
    cfg = _load_config()
    return int(cfg["training"]["max_runners"])


def build_policy(obs_dim: int, hidden_size: int, device,
                 norm_mean=None, norm_std=None,
                 max_runners: int | None = None,
                 runner_dim: int = RUNNER_DIM,
                 mature_prob_open_threshold: float = 0.0,
                 seed: int | None = None):
    """DiscreteLSTMPolicy with input_norm ON (plan §5). Full-obs runner_dim."""
    if seed is not None:
        torch.manual_seed(seed)
    if max_runners is None:
        max_runners = max_runners_from_config()
    space = DiscreteActionSpace(max_runners=max_runners)
    policy = DiscreteLSTMPolicy(
        obs_dim=obs_dim, action_space=space, hidden_size=hidden_size,
        runner_dim=runner_dim, input_norm=True,
        mature_prob_open_threshold=mature_prob_open_threshold,
    ).to(device)
    if norm_mean is not None and norm_std is not None:
        policy.set_input_norm_stats(norm_mean, norm_std)
    return policy, space


@torch.no_grad()
def policy_mature_scores(policy, obs: np.ndarray, runner_idx: np.ndarray,
                         device, batch: int = 8192) -> np.ndarray:
    """Per-sample ``mature_prob`` at the candidate's runner column.

    Runs the policy at ctx=1 / zero-init hidden (the single-tick regime BC
    trains on and the LightGBM probe measured), gathering
    ``mature_prob_per_runner[i, runner_idx[i]]``.
    """
    was_training = policy.training
    policy.eval()
    out_scores = np.empty(obs.shape[0], dtype=np.float64)
    ri = np.asarray(runner_idx, dtype=np.int64)
    for s in range(0, obs.shape[0], batch):
        e = min(s + batch, obs.shape[0])
        ob = torch.from_numpy(obs[s:e]).to(device=device, dtype=torch.float32)
        rib = torch.from_numpy(ri[s:e]).to(device=device)
        po = policy(ob)
        mp = po.mature_prob_per_runner  # (b, R)
        rows = torch.arange(e - s, device=device)
        out_scores[s:e] = mp[rows, rib].detach().cpu().numpy()
    policy.train(was_training)
    return out_scores


# ── Metrics ─────────────────────────────────────────────────────────────────

def evaluate(scores: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.asarray(y).astype(np.int32)
    base = float(y.mean())
    auc = float(roc_auc_score(y, scores))
    ap = float(average_precision_score(y, scores))
    order = np.argsort(scores)[::-1]
    top10 = order[: max(1, len(order) // 10)]
    prec_top10 = float(y[top10].mean())
    return {
        "n": int(len(y)),
        "base_rate": round(base, 4),
        "auc": round(auc, 4),
        "ap": round(ap, 4),
        "top_decile_precision": round(prec_top10, 4),
        "top_decile_lift": round(prec_top10 / max(base, 1e-9), 3),
    }


def pr_curve(scores: np.ndarray, y: np.ndarray, thresholds=None) -> list[dict]:
    """Sweep the open threshold; report opens / precision(mat%) / recall.

    precision = fraction of opened candidates that mature (== rollout mat%
    on the labelled dataset). recall = matured-opened / all-matured.
    """
    y = np.asarray(y).astype(np.int32)
    total_pos = int(y.sum())
    n = len(y)
    if thresholds is None:
        thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                      0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]
    rows = []
    for t in thresholds:
        sel = scores >= t
        opens = int(sel.sum())
        if opens == 0:
            rows.append({"threshold": round(float(t), 3), "opens": 0,
                         "mat_pct": 0.0, "recall": 0.0, "lift": 0.0,
                         "frac_opened": 0.0})
            continue
        prec = float(y[sel].mean())
        rec = float(y[sel].sum() / max(total_pos, 1))
        base = float(y.mean())
        rows.append({
            "threshold": round(float(t), 3),
            "opens": opens,
            "mat_pct": round(100.0 * prec, 2),
            "recall": round(rec, 4),
            "lift": round(prec / max(base, 1e-9), 3),
            "frac_opened": round(opens / n, 4),
        })
    return rows


# ── CLI baselines ───────────────────────────────────────────────────────────

def _print_eval(tag, m):
    print(f"{tag}: n={m['n']} base={m['base_rate']} AUC={m['auc']} "
          f"AP={m['ap']} top10%prec={m['top_decile_precision']} "
          f"(lift {m['top_decile_lift']}x)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lgbm", "untrained"], required=True)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--train-days", default="train8")
    ap.add_argument("--holdout-days", default="holdout")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    train_days = resolve_days(args.train_days)
    holdout_days = resolve_days(args.holdout_days)
    print(f"train_days={train_days}", flush=True)
    print(f"holdout_days={holdout_days}", flush=True)

    print("loading holdout split...", flush=True)
    Xte, yte, rite = load_split(holdout_days, cache)
    print(f"holdout n={len(yte)} matured_rate={yte.mean():.4f} "
          f"obs_dim={Xte.shape[1]}", flush=True)

    result = {"mode": args.mode, "train_days": train_days,
              "holdout_days": holdout_days, "obs_dim": int(Xte.shape[1])}

    if args.mode == "lgbm":
        import lightgbm as lgb
        print("loading train split...", flush=True)
        Xtr, ytr, ritr = load_split(train_days, cache)
        print(f"train n={len(ytr)} matured_rate={ytr.mean():.4f}", flush=True)
        clf = lgb.LGBMClassifier(
            num_leaves=64, n_estimators=300, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
            n_jobs=-1, random_state=args.seed,
        )
        print("training LightGBM...", flush=True)
        clf.fit(Xtr, ytr)
        s_te = clf.predict_proba(Xte)[:, 1]
        m = evaluate(s_te, yte)
        _print_eval("LGBM holdout", m)
        result["holdout"] = m
        result["pr_curve"] = pr_curve(s_te, yte)

    elif args.mode == "untrained":
        device = torch.device(
            args.device if (args.device != "cuda" or torch.cuda.is_available())
            else "cpu")
        # Norm stats from train cache (deterministic; same as Step B will use).
        print("loading train split for norm stats...", flush=True)
        Xtr, _, _ = load_split(train_days, cache)
        mean, std = compute_norm_stats(Xtr)
        del Xtr
        policy, _ = build_policy(
            Xte.shape[1], args.hidden_size, device,
            norm_mean=mean, norm_std=std, seed=args.seed)
        s_te = policy_mature_scores(policy, Xte, rite, device)
        m = evaluate(s_te, yte)
        _print_eval("UNTRAINED head holdout", m)
        result["holdout"] = m
        result["pr_curve"] = pr_curve(s_te, yte)

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
