"""Sweep-specific evaluator: run every variant head against every
held-out eval day and emit one JSON line per (variant, day) plus a
final aggregate JSON per variant.

Usage::

    python -m scripts.sweep_eval_direction_heads \
        --manifests models/direction_head/v1_2026-05-24 \
                    models/direction_head/sweep_c1 \
                    ... \
        --eval-dates 2026-04-07,2026-04-10,...

The plan's acceptance ranking (per
``plans/direction-head-architecture-sweep/session_prompt.md``):

1. Primary: mean Pearson averaged across both sides and all eval days.
2. Tie-break: mean ROC AUC.
3. Sanity: mean Brier must not regress by > 10 % vs the baseline.

The script computes those numbers and ranks the variants. The
load-from-manifest path reuses ``evaluate_direction_head._load_head_
from_manifest`` so the sweep stays consistent with the existing
single-variant evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from scripts.evaluate_direction_head import (
    _bce_unweighted,
    _brier,
    _load_day_obs_and_labels,
    _load_head_from_manifest,
    _pearson,
    _roc_auc,
)


def _eval_one(
    manifest_dir: Path,
    eval_dates: list[str],
    oracle_root: Path,
    label_root: Path,
    label_stem: str,
) -> dict:
    head, manifest = _load_head_from_manifest(manifest_dir)
    variant_id = manifest.get("architecture", {}).get("variant", "c0")
    exp_id = manifest["experiment_id"]
    training_dates = set(manifest["training"]["training_dates"])
    leaks = [d for d in eval_dates if d in training_dates]
    if leaks:
        raise ValueError(
            f"{exp_id}: eval dates {leaks} overlap training set"
        )

    per_day = []
    for d in eval_dates:
        data = _load_day_obs_and_labels(
            d, label_stem, oracle_root, label_root,
        )
        if data["n"] == 0:
            continue
        X = data["per_runner_obs"]
        Y_back = data["label_back"]
        Y_lay = data["label_lay"]
        with torch.no_grad():
            probs = torch.sigmoid(head(torch.from_numpy(X))).numpy()
        p_back = probs[:, 0]
        p_lay = probs[:, 1]
        row = {
            "date": d,
            "n": int(data["n"]),
            "pos_rate_back": float(Y_back.mean()),
            "pos_rate_lay": float(Y_lay.mean()),
            "pearson_back": _pearson(p_back, Y_back),
            "pearson_lay": _pearson(p_lay, Y_lay),
            "auc_back": _roc_auc(p_back, Y_back),
            "auc_lay": _roc_auc(p_lay, Y_lay),
            "bce_back": _bce_unweighted(p_back, Y_back),
            "bce_lay": _bce_unweighted(p_lay, Y_lay),
            "brier_back": _brier(p_back, Y_back),
            "brier_lay": _brier(p_lay, Y_lay),
            "pred_mean_back": float(p_back.mean()),
            "pred_mean_lay": float(p_lay.mean()),
            "pred_std_back": float(p_back.std()),
            "pred_std_lay": float(p_lay.std()),
        }
        per_day.append(row)

    if not per_day:
        raise RuntimeError(f"{exp_id}: no eval data found")

    # Average across days.
    def avg(key: str) -> float:
        vals = [r[key] for r in per_day]
        # NaN-tolerant
        vals = [v for v in vals if v == v]  # noqa: PLR0124
        return float(sum(vals) / len(vals)) if vals else float("nan")

    agg = {
        "manifest_dir": str(manifest_dir),
        "experiment_id": exp_id,
        "variant": variant_id,
        "hidden_dims": manifest["architecture"]["hidden_dims"],
        "pos_weight_mode": manifest["architecture"].get(
            "pos_weight_mode", "balanced",
        ),
        "val_bce_back_in_sample": manifest["val_metrics"]["val_bce_back"],
        "val_bce_lay_in_sample": manifest["val_metrics"]["val_bce_lay"],
        "n_eval_days": len(per_day),
        "mean_pearson_back": avg("pearson_back"),
        "mean_pearson_lay": avg("pearson_lay"),
        "mean_pearson": (avg("pearson_back") + avg("pearson_lay")) / 2,
        "mean_auc_back": avg("auc_back"),
        "mean_auc_lay": avg("auc_lay"),
        "mean_auc": (avg("auc_back") + avg("auc_lay")) / 2,
        "mean_bce_back": avg("bce_back"),
        "mean_bce_lay": avg("bce_lay"),
        "mean_brier_back": avg("brier_back"),
        "mean_brier_lay": avg("brier_lay"),
        "mean_brier": (avg("brier_back") + avg("brier_lay")) / 2,
        "per_day": per_day,
    }
    return agg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifests", required=True, nargs="+",
        help="One or more manifest dirs to evaluate.",
    )
    ap.add_argument(
        "--eval-dates", required=True,
        help="Comma-separated YYYY-MM-DD list (all held-out).",
    )
    ap.add_argument("--label-stem", default="horizon60_thresh5_fc60")
    ap.add_argument("--oracle-root", default="data/oracle_cache_v2")
    ap.add_argument("--label-root", default="data/direction_labels")
    ap.add_argument(
        "--output-json", default=None,
        help="If set, write aggregate JSON to this file.",
    )
    args = ap.parse_args()
    eval_dates = [d.strip() for d in args.eval_dates.split(",")]

    all_results = []
    for m in args.manifests:
        print(f"\n=== Evaluating: {m} ===")
        try:
            agg = _eval_one(
                Path(m),
                eval_dates,
                Path(args.oracle_root),
                Path(args.label_root),
                args.label_stem,
            )
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            continue
        all_results.append(agg)
        print(
            f"  variant={agg['variant']}  "
            f"hidden_dims={agg['hidden_dims']}  "
            f"pos_weight={agg['pos_weight_mode']}"
        )
        print(
            f"  mean Pearson: back={agg['mean_pearson_back']:+.4f}  "
            f"lay={agg['mean_pearson_lay']:+.4f}  "
            f"avg={agg['mean_pearson']:+.4f}"
        )
        print(
            f"  mean ROC AUC: back={agg['mean_auc_back']:.4f}  "
            f"lay={agg['mean_auc_lay']:.4f}  avg={agg['mean_auc']:.4f}"
        )
        print(
            f"  mean BCE:     back={agg['mean_bce_back']:.4f}  "
            f"lay={agg['mean_bce_lay']:.4f}"
        )
        print(
            f"  mean Brier:   back={agg['mean_brier_back']:.4f}  "
            f"lay={agg['mean_brier_lay']:.4f}  avg={agg['mean_brier']:.4f}"
        )

    # Rank
    if not all_results:
        print("ERROR: no manifests evaluated", file=sys.stderr)
        return 2
    ranked = sorted(
        all_results,
        key=lambda r: (-r["mean_pearson"], -r["mean_auc"]),
    )
    baseline = next(
        (r for r in all_results if r["variant"] == "c0"),
        all_results[0],
    )
    baseline_brier = baseline["mean_brier"]
    brier_cap = 1.10 * baseline_brier
    print()
    print("=" * 72)
    print("RANKING (primary = mean Pearson, tie-break = mean AUC)")
    print(
        f"  Brier cap (no >10% regress vs baseline {baseline['variant']}): "
        f"{brier_cap:.4f}"
    )
    print(
        f"  {'rank':>4}  {'variant':<6}  {'mean_rho':>9}  "
        f"{'mean_auc':>8}  {'mean_brier':>10}  {'verdict':<10}"
    )
    for i, r in enumerate(ranked, 1):
        verdict = "OK"
        if r["mean_brier"] > brier_cap and r["variant"] != baseline["variant"]:
            verdict = "BRIER>cap"
        print(
            f"  {i:>4}  {r['variant']:<6}  {r['mean_pearson']:>+9.4f}  "
            f"{r['mean_auc']:>8.4f}  {r['mean_brier']:>10.4f}  {verdict:<10}"
        )
    # Pick winner (first eligible)
    winner = next(
        (
            r for r in ranked
            if r["mean_brier"] <= brier_cap
            or r["variant"] == baseline["variant"]
        ),
        ranked[0],
    )
    print()
    print(f"WINNER: {winner['variant']} ({winner['experiment_id']})")
    print(f"        mean_pearson={winner['mean_pearson']:+.4f}  "
          f"mean_auc={winner['mean_auc']:.4f}  "
          f"mean_brier={winner['mean_brier']:.4f}")

    payload = {
        "eval_dates": eval_dates,
        "baseline_variant": baseline["variant"],
        "brier_cap": brier_cap,
        "winner": {
            "variant": winner["variant"],
            "experiment_id": winner["experiment_id"],
            "manifest_dir": winner["manifest_dir"],
        },
        "results": all_results,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
