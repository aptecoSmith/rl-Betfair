"""Step 1 (imitation-first): maturation-predictability probe.

The decisive Step 1 question is "is the oracle's maturing-open decision
LEARNABLE from decision-time features?" Step 1b found the full obs is
unnormalized and the v2 policy has no input norm on the actor path, so a
BC-only probe (frozen random backbone) would be a confound. A
scale-invariant LightGBM classifier answers the SAME question cleanly:
among every profitable-spread candidate (the spread-placeable oracle's
opens), can we predict from the obs which ones will MATURE (passive fills
before T-fc) vs force-close?

Method: for each day, ONE labelled scan (``scan_day(..., predictors,
maturation_label_out=labels)``) emits every spread-placeable candidate
with its full predictor-injected obs AND a matured/not-matured flag (the
env-faithful forward walk). Train LightGBM on TRAIN days, evaluate ROC-AUC
on the 7 reserved HOLDOUT days.

Verdict:
  - holdout AUC >> 0.5 (say > 0.60) -> maturation IS predictable from
    features -> a policy could learn selectivity -> proceed to Step 1c/2
    (which need the input-norm fix, Step 1b).
  - holdout AUC ~ 0.5 -> NOT predictable from current features -> STOP;
    the unlock is richer data (deeper book), not more training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from training_v2.arb_oracle import _load_config, scan_day  # noqa: E402
from agents_v2.env_shim import DEFAULT_SCORER_DIR  # noqa: E402

DATA_DIR = Path("data/processed")
PRED_MANIFESTS = (
    "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json",
    "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json",
    "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json",
)
def _existing_train_days(n: int = 8) -> list[str]:
    """``n`` evenly-spaced EXISTING processed days in [04-06, 05-19]."""
    import re
    pat = re.compile(r"^2026-\d{2}-\d{2}$")
    days = sorted(
        p.stem for p in DATA_DIR.glob("2026-*.parquet")
        if pat.match(p.stem) and "2026-04-06" <= p.stem <= "2026-05-19"
    )
    if len(days) <= n:
        return days
    idx = np.linspace(0, len(days) - 1, n).round().astype(int)
    return [days[i] for i in sorted(set(idx.tolist()))]


TRAIN_DAYS = _existing_train_days(8)
HOLDOUT_DAYS = [
    "2026-05-20", "2026-05-21", "2026-05-22", "2026-05-25",
    "2026-05-27", "2026-05-28", "2026-05-29",
]


def _make_bundle():
    from predictors import PredictorBundle
    champ, rank, dirm = PRED_MANIFESTS
    return PredictorBundle.from_manifests(
        champion_manifest=Path(champ),
        ranker_manifest=Path(rank),
        direction_manifest=Path(dirm),
    )


def _day_dataset(date, cfg, bundle):
    """One labelled scan -> (X obs matrix, y matured flags)."""
    labels: dict = {}
    samples = scan_day(
        date, DATA_DIR, cfg, scorer_dir=DEFAULT_SCORER_DIR,
        predictor_bundle=bundle,
        use_race_outcome_predictor=True,
        use_direction_predictor=True,
        predictor_lean_obs=False,
        maturation_conditioned=False,        # emit ALL candidates
        force_close_before_off_seconds=120.0,
        maturation_label_out=labels,         # ...but label each one
    )
    if not samples:
        return None, None
    X = np.stack([s.obs for s in samples], axis=0).astype(np.float32)
    y = np.array(
        [1 if labels.get((s.tick_index, s.runner_idx), False) else 0
         for s in samples],
        dtype=np.int32,
    )
    return X, y


def _collect(days, cfg, bundle, tag):
    Xs, ys = [], []
    for d in days:
        X, y = _day_dataset(d, cfg, bundle)
        if X is None:
            print(f"[{tag} {d}] no samples", flush=True)
            continue
        Xs.append(X)
        ys.append(y)
        print(f"[{tag} {d}] n={len(y):>6} matured={int(y.sum()):>6} "
              f"rate={y.mean():.3f}", flush=True)
    return np.concatenate(Xs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plans/imitation-first/_step1/predictability_results.json")
    ap.add_argument("--num-leaves", type=int, default=64)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    args = ap.parse_args()

    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, average_precision_score

    cfg = _load_config()
    print("loading predictor bundle...", flush=True)
    bundle = _make_bundle()

    print("\n=== TRAIN days ===", flush=True)
    X_tr, y_tr = _collect(TRAIN_DAYS, cfg, bundle, "train")
    print("\n=== HOLDOUT days ===", flush=True)
    X_te, y_te = _collect(HOLDOUT_DAYS, cfg, bundle, "holdout")

    print(f"\ntrain: n={len(y_tr)} matured_rate={y_tr.mean():.4f}", flush=True)
    print(f"holdout: n={len(y_te)} matured_rate={y_te.mean():.4f}", flush=True)

    clf = lgb.LGBMClassifier(
        num_leaves=args.num_leaves,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=0.8, colsample_bytree=0.6,
        reg_lambda=1.0, n_jobs=-1, random_state=0,
    )
    print("\ntraining LightGBM...", flush=True)
    clf.fit(X_tr, y_tr)

    p_tr = clf.predict_proba(X_tr)[:, 1]
    p_te = clf.predict_proba(X_te)[:, 1]
    auc_tr = float(roc_auc_score(y_tr, p_tr))
    auc_te = float(roc_auc_score(y_te, p_te))
    ap_te = float(average_precision_score(y_te, p_te))
    base_te = float(y_te.mean())

    # Lift @ top-decile: of the candidates the model is most confident
    # will mature, what fraction actually mature (vs base rate)?
    order = np.argsort(p_te)[::-1]
    top10 = order[: max(1, len(order) // 10)]
    prec_top10 = float(y_te[top10].mean())

    imp = clf.feature_importances_
    top_feats = np.argsort(imp)[::-1][:20].tolist()

    print("\n==== STEP 1 MATURATION PREDICTABILITY ====", flush=True)
    print(f"train AUC = {auc_tr:.4f}", flush=True)
    print(f"HOLDOUT AUC = {auc_te:.4f}  (0.5 = no signal)", flush=True)
    print(f"holdout AP = {ap_te:.4f}  (base rate {base_te:.4f})", flush=True)
    print(f"holdout top-decile precision = {prec_top10:.4f}  "
          f"(base {base_te:.4f}, lift {prec_top10/max(base_te,1e-9):.2f}x)",
          flush=True)
    print(f"top-20 feature dims by importance: {top_feats}", flush=True)

    out = {
        "train_days": TRAIN_DAYS, "holdout_days": HOLDOUT_DAYS,
        "n_train": int(len(y_tr)), "n_holdout": int(len(y_te)),
        "train_matured_rate": round(float(y_tr.mean()), 4),
        "holdout_matured_rate": round(base_te, 4),
        "train_auc": round(auc_tr, 4),
        "holdout_auc": round(auc_te, 4),
        "holdout_ap": round(ap_te, 4),
        "holdout_top_decile_precision": round(prec_top10, 4),
        "holdout_top_decile_lift": round(prec_top10 / max(base_te, 1e-9), 3),
        "top_feature_dims": top_feats,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
