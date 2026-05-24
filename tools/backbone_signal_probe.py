"""Probe whether a trained policy's BACKBONE preserves direction signal.

Compares two logistic regressions:

  (A) full obs (574-dim) -> v1 direction label  -- the upper bound the
      direction_prob_head could achieve from raw obs
  (B) backbone hidden state (hidden_size-dim) -> v1 direction label --
      what the head actually has access to after PPO has shaped the
      backbone

If (B) descent ≈ (A) descent: backbone preserves the direction signal
→ the head's failure to descend BCE during cohort training is due to
the head/loss-weight combination, NOT the backbone. Fix is to bump
direction_prob_loss_weight gene range or widen the head.

If (B) descent ≪ (A) descent: backbone destroys the signal → PPO is
reshaping the LSTM toward policy/value features at the cost of
direction features. Fix requires architecture change (e.g. pass
obs[dir_*] directly to actor_head, bypassing the shared backbone).

Usage:
    python tools/backbone_signal_probe.py <agent_weights.pt> [--date 2026-04-11]

The script loads the agent's weights, runs the LSTM backbone forward
on a sample of obs vectors (from the oracle cache; init hidden state
= zero, single timestep per sample), captures lstm_last, then trains
sklearn LogisticRegression(balanced) on both (full obs) and
(lstm_last) → label.

Requires the oracle cache to have been scanned with
--use-direction-predictor (otherwise the predictor obs columns are
zero and the comparison is uninformative).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np


def _pos_weighted_floor(pos_rate: float) -> float:
    return 2.0 * (1.0 - pos_rate) * math.log(2.0)


def _pos_weighted_bce(
    probs: np.ndarray, labels: np.ndarray, pos_rate: float,
) -> float:
    d = max(min(pos_rate, 1.0 - 1e-9), 1e-9)
    pos_weight = (1.0 - d) / d
    eps = 1e-9
    p = np.clip(probs, eps, 1.0 - eps)
    bce = -(
        pos_weight * labels * np.log(p)
        + (1.0 - labels) * np.log(1.0 - p)
    )
    return float(bce.mean())


def _train_logreg(
    X: np.ndarray, y: np.ndarray, label_name: str,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    pos_rate = float(y.mean())
    floor = _pos_weighted_floor(pos_rate)
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    clf = LogisticRegression(
        C=1.0, max_iter=5000, class_weight="balanced",
        solver="lbfgs", tol=1e-5,
    )
    clf.fit(Xtr_s, ytr)
    probs_va = clf.predict_proba(Xva_s)[:, 1]
    val_bce = _pos_weighted_bce(probs_va, yva, pos_rate)
    descent = floor - val_bce
    rel_descent = descent / floor
    return {
        "label": label_name,
        "n_train": Xtr.shape[0],
        "n_val": Xva.shape[0],
        "pos_rate": pos_rate,
        "floor": floor,
        "val_bce": val_bce,
        "descent": descent,
        "rel_descent_pct": rel_descent * 100.0,
        "n_iter": int(clf.n_iter_[0]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "weights_path",
        help="Path to agent's .pt weights file",
    )
    p.add_argument("--date", default="2026-04-11")
    p.add_argument("--samples", type=int, default=20000)
    p.add_argument(
        "--hidden-size", type=int, default=None,
        help="Override hidden_size; default reads from scoreboard if "
             "weights are inside a cohort dir.",
    )
    args = p.parse_args()

    import torch
    from agents_v2.discrete_policy import DiscreteLSTMPolicy
    from agents_v2.env_shim import DEFAULT_SCORER_DIR
    from training_v2.cohort.worker import (
        _build_env_for_day, scalping_train_config,
    )
    from predictors import PredictorBundle

    weights_path = Path(args.weights_path)
    if not weights_path.exists():
        print(f"ERROR: {weights_path} not found", file=sys.stderr)
        return 2

    # Discover hidden_size from scoreboard.jsonl if available.
    hidden_size = args.hidden_size
    if hidden_size is None:
        cohort_dir = weights_path.parent.parent
        sb = cohort_dir / "scoreboard.jsonl"
        agent_id = weights_path.stem
        if sb.exists():
            import json
            with sb.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("agent_id", "").startswith(agent_id[:8]):
                        hidden_size = int(
                            row.get("hyperparameters", {})
                            .get("hidden_size", 256)
                        )
                        break
    if hidden_size is None:
        hidden_size = 256
    print(f"hidden_size = {hidden_size}")

    # Load labels + oracle obs.
    oracle = np.load(
        f"data/oracle_cache_v2/{args.date}/oracle_samples.npz"
    )
    labels = np.load(
        f"data/direction_labels/{args.date}/horizon60_thresh5_fc60.npz"
    )

    ok = (oracle["tick_index"].astype(np.int64) << 16) | oracle["runner_idx"]
    lk = (labels["tick_index"].astype(np.int64) << 16) | labels["runner_idx"]
    lookup = {int(k): i for i, k in enumerate(lk)}
    ji = [(i, lookup[int(k)]) for i, k in enumerate(ok) if int(k) in lookup]
    oi = np.array([p[0] for p in ji], dtype=np.int64)
    li = np.array([p[1] for p in ji], dtype=np.int64)

    obs_all = oracle["obs"][oi].astype(np.float32)
    label_back_all = labels["label_back"][li].astype(np.float32)
    label_lay_all = labels["label_lay"][li].astype(np.float32)
    n_all = obs_all.shape[0]
    if n_all > args.samples:
        rng = np.random.default_rng(42)
        sel = rng.choice(n_all, size=args.samples, replace=False)
        obs = obs_all[sel]
        label_back = label_back_all[sel]
        label_lay = label_lay_all[sel]
    else:
        obs = obs_all
        label_back = label_back_all
        label_lay = label_lay_all
    print(f"Using {obs.shape[0]} joined (obs, label) pairs on {args.date}")

    # Verify the obs cache has populated predictor columns. If they're
    # zero, the comparison is meaningless.
    from env.betfair_env import LEAN_RUNNER_KEYS, MARKET_DIM, VELOCITY_DIM
    N_KEY = len(LEAN_RUNNER_KEYS)
    max_runners = (obs.shape[1] - 56) // 37
    # Read dir_q50_7m on runner 0 across samples
    q50_7m_idx = LEAN_RUNNER_KEYS.index("dir_q50_7m")
    runner0_q50_7m = obs[:, MARKET_DIM + VELOCITY_DIM + 0 * N_KEY + q50_7m_idx]
    if abs(runner0_q50_7m.std()) < 1e-6:
        print(
            "\nWARN: dir_q50_7m std≈0 — oracle cache wasn't scanned with "
            "--use-direction-predictor. Re-scan first:",
            file=sys.stderr,
        )
        print(
            "  python -m training_v2.oracle_cli scan --date {} "
            "--predictor-lean-obs --predictor-bundle-manifests <C> <R> <D> "
            "--use-race-outcome-predictor --use-direction-predictor".format(
                args.date,
            ),
            file=sys.stderr,
        )

    # Build env + load policy weights.
    cfg = scalping_train_config()
    bundle = PredictorBundle.from_manifests(
        champion_manifest=Path(
            "C:/Users/jsmit/source/repos/betfair-predictors/"
            "production/race-outcome/manifest.json"
        ),
        ranker_manifest=Path(
            "C:/Users/jsmit/source/repos/betfair-predictors/"
            "production/race-outcome-ranker/manifest.json"
        ),
        direction_manifest=Path(
            "C:/Users/jsmit/source/repos/betfair-predictors/"
            "production/direction-predictor/manifest.json"
        ),
    )
    env, shim = _build_env_for_day(
        day_str=args.date,
        data_dir=Path("data/processed"),
        cfg=cfg,
        scorer_dir=DEFAULT_SCORER_DIR,
        predictor_bundle=bundle,
        use_race_outcome_predictor=True,
        use_direction_predictor=True,
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

    # Walk obs through the LSTM backbone in chunks; capture lstm_last.
    print("\nRunning obs through LSTM backbone...")
    obs_t = torch.from_numpy(obs)
    n = obs_t.shape[0]
    hidden_states = np.empty((n, hidden_size), dtype=np.float32)
    chunk = 1024
    with torch.no_grad():
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            batch_obs = obs_t[start:end].unsqueeze(1)  # (b, 1, obs_dim)
            b = end - start
            h0, c0 = policy.init_hidden(b)
            flat = batch_obs.reshape(b * 1, obs_t.shape[1])
            proj = policy.input_proj(flat).reshape(b, 1, -1)
            lstm_out, _ = policy.lstm(proj, (h0, c0))
            lstm_last = lstm_out[:, -1, :]  # (b, hidden)
            hidden_states[start:end] = lstm_last.cpu().numpy()
    print(
        f"backbone hidden_states shape: {hidden_states.shape}  "
        f"mean={hidden_states.mean():.4f} "
        f"std={hidden_states.std():.4f}"
    )

    # Run both logregs.
    print("\n=== A: full obs (574-dim) -> label ===")
    for side, y in [("back", label_back), ("lay", label_lay)]:
        r = _train_logreg(obs, y, side)
        print(
            f"  {side}: floor={r['floor']:.4f}  val_bce={r['val_bce']:.4f}  "
            f"descent={r['descent']:+.4f}  "
            f"rel={r['rel_descent_pct']:+.1f}%  "
            f"n_iter={r['n_iter']}"
        )

    print("\n=== B: backbone hidden_state (hidden_size-dim) -> label ===")
    for side, y in [("back", label_back), ("lay", label_lay)]:
        r = _train_logreg(hidden_states, y, side)
        print(
            f"  {side}: floor={r['floor']:.4f}  val_bce={r['val_bce']:.4f}  "
            f"descent={r['descent']:+.4f}  "
            f"rel={r['rel_descent_pct']:+.1f}%  "
            f"n_iter={r['n_iter']}"
        )

    print(
        "\nInterpretation:"
        "\n  * B descent matches A descent => backbone preserves signal."
        "\n    Fix = bump direction_prob_loss_weight gene range upward."
        "\n  * B descent << A descent => backbone destroys signal."
        "\n    Fix = architecture change (e.g. residual obs path into "
        "actor_head)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
