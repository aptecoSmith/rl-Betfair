"""scripts/predictor/run_s09.py — S09 sealed test-set evaluation.

ONE-SHOT. Runs inference on May 4–6 test parquets for a chosen set of
experiment IDs. Results written to registry/s09_test_results.csv.

Usage:
    python scripts/predictor/run_s09.py \\
        conv1d_k3_s1_3952e600e57b \\
        conv1d_k3_s1_3fc8e2c22c9c \\
        conv1d_k3_s1_9659e9e9c3fb

The experiment IDs must exist in registry/predictor_scoreboard_neural.csv
and their .pt weights must be present in registry/predictor/.

No training is performed. The test set is never touched by any other script
(splits.py hard-codes it as sealed until S09).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.predictor.datasets import (  # noqa: E402
    SequenceDataset,
    TabularDataset, TabularExamples,
    build_sequence_examples,
    feature_columns,
    load_split,
    to_feature_tensor,
    to_label_tensors,
)
from scripts.predictor.eval_metrics import (  # noqa: E402
    naive_backtest_pnl,
)
from scripts.predictor.models import (  # noqa: E402
    build_model,
    is_sequence_family,
)
from scripts.predictor.train_one import (  # noqa: E402
    evaluate_predictions,
    predict_torch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_SCOREBOARD = REPO_ROOT / "registry" / "predictor_scoreboard_neural.csv"
DEFAULT_MODEL_DIR = REPO_ROOT / "registry" / "predictor"
S09_RESULTS_PATH = REPO_ROOT / "registry" / "s09_test_results.csv"


def load_candidate(experiment_id: str, scoreboard_path: Path) -> dict:
    df = pd.read_csv(scoreboard_path)
    matches = df[df["experiment_id"] == experiment_id]
    if matches.empty:
        raise ValueError(
            f"experiment_id {experiment_id!r} not found in {scoreboard_path}"
        )
    return matches.iloc[0].to_dict()


def evaluate_one(
    experiment_id: str,
    scoreboard_path: Path,
    model_dir: Path,
    device: str,
) -> dict:
    logger.info("=== %s ===", experiment_id)

    row = load_candidate(experiment_id, scoreboard_path)
    family = str(row["architecture"])
    arch_kwargs = json.loads(str(row["arch_kwargs"]))
    feature_variant = str(row["feature_variant"])
    train_corpus = str(row["train_corpus"])
    horizons = [h.strip() for h in str(row["horizons"]).split(",")]
    quantiles = [float(q) for q in str(row["quantiles"]).split(",")]

    logger.info(
        "family=%s variant=%s features=%s corpus=%s horizons=%s quantiles=%s",
        family, row.get("variant_label"), feature_variant,
        train_corpus, horizons, quantiles,
    )

    # Load test data (the sealed split).
    t0 = time.time()
    df_test = load_split("test", feature_variant, horizons, train_corpus)
    logger.info("loaded test=%s rows in %.1fs", f"{len(df_test):,}", time.time() - t0)

    n_features = len(feature_columns(feature_variant))
    n_horizons = len(horizons)

    # Build dataset.
    if is_sequence_family(family):
        time_window = int(arch_kwargs.get("time_window", arch_kwargs.get("ctx_ticks", 32)))
        seq_test = build_sequence_examples(df_test, feature_variant, horizons, time_window)
        test_ds = SequenceDataset(seq_test)
        ltp_test = df_test.sort_values(
            ["market_id", "selection_id", "tick_idx"]
        )["ltp"].to_numpy(dtype=np.float32)
        df_test_for_groups = df_test.sort_values(
            ["market_id", "selection_id", "tick_idx"]
        ).reset_index(drop=True)
        y_eval = seq_test.y
        mask_eval = seq_test.mask
    else:
        X_test, _ = to_feature_tensor(df_test, feature_variant)
        y_test, mask_test = to_label_tensors(df_test, horizons)
        test_ds = TabularDataset(TabularExamples(
            X=X_test, y=y_test, mask=mask_test,
            feature_names=feature_columns(feature_variant),
            horizons=horizons,
        ))
        ltp_test = df_test["ltp"].to_numpy(dtype=np.float32)
        df_test_for_groups = df_test.reset_index(drop=True)
        y_eval = y_test
        mask_eval = mask_test

    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False,
                             num_workers=0, pin_memory=(device == "cuda"))

    # Build model and load weights.
    model = build_model(
        family=family,
        n_features=n_features,
        n_horizons=n_horizons,
        n_quantiles=len(quantiles),
        arch_kwargs=arch_kwargs,
    )
    weights_path = model_dir / f"{experiment_id}.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    logger.info("loaded weights from %s", weights_path)

    # Inference.
    t0 = time.time()
    pred_test = predict_torch(model, test_loader, device)
    logger.info("inference done in %.1fs", time.time() - t0)

    # Evaluate.
    metrics = evaluate_predictions(
        pred=pred_test,
        y=y_eval,
        mask=mask_eval,
        ltp_now=ltp_test,
        horizons=horizons,
        quantiles=quantiles,
        df_for_groups=df_test_for_groups,
    )

    result = {
        "experiment_id": experiment_id,
        "session": "S09",
        "timestamp": datetime.utcnow().isoformat(),
        "architecture": family,
        "variant_label": row.get("variant_label"),
        "feature_variant": feature_variant,
        "train_corpus": train_corpus,
        "horizons": row["horizons"],
        "output_formulation": row.get("output_formulation"),
        "quantiles": row["quantiles"],
        "seed": row.get("seed"),
        # Val-set numbers for side-by-side comparison.
        "val_dir_acc_k5_7m": row.get("dir_acc_k5_7m"),
        "val_dir_fires_k5_7m": row.get("dir_fires_k5_7m"),
        "val_backtest_pnl_k5_7m": row.get("backtest_pnl_k5_7m"),
    }
    result.update({f"test_{k}": v for k, v in metrics.items()})
    return result


def write_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    logger.info("S09 results written to %s", path)


def print_summary(results: list[dict]) -> None:
    print()
    print("=" * 80)
    print("S09 — SEALED TEST SET RESULTS (May 4–6)")
    print("=" * 80)
    header = f"{'experiment_id':32} {'horizons':20} {'variant':10} "
    header += f"{'test_dir_acc':12} {'test_fires':10} {'test_pnl':10} "
    header += f"{'val_dir_acc':12} {'val_pnl':10}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: float(x.get("test_backtest_pnl_k5_7m") or 0), reverse=True):
        exp = str(r["experiment_id"])
        hz = str(r["horizons"])
        var = f"{r.get('feature_variant')}/{r.get('output_formulation')}"
        test_acc = r.get("test_dir_acc_k5_7m")
        test_fires = r.get("test_dir_fires_k5_7m")
        test_pnl = r.get("test_backtest_pnl_k5_7m")
        val_acc = r.get("val_dir_acc_k5_7m")
        val_pnl = r.get("val_backtest_pnl_k5_7m")
        print(
            f"{exp:32} {hz:20} {var:10} "
            f"{float(test_acc or 0):.4f}       {int(float(test_fires or 0)):6}     "
            f"£{float(test_pnl or 0):8.2f}   "
            f"{float(val_acc or 0):.4f}       £{float(val_pnl or 0):8.2f}"
        )
    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(description="S09 sealed test-set evaluation")
    parser.add_argument(
        "experiment_ids",
        nargs="+",
        help="Experiment IDs to evaluate (must exist in scoreboard + registry/predictor/)",
    )
    parser.add_argument(
        "--scoreboard",
        default=str(DEFAULT_SCOREBOARD),
        help="Path to neural scoreboard CSV",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory containing .pt weight files",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available)",
    )
    args = parser.parse_args()

    scoreboard_path = Path(args.scoreboard)
    model_dir = Path(args.model_dir)

    logger.info("=== S09: sealed test-set evaluation ===")
    logger.info("candidates: %s", args.experiment_ids)
    logger.info("device: %s", args.device)

    results = []
    for exp_id in args.experiment_ids:
        try:
            result = evaluate_one(exp_id, scoreboard_path, model_dir, args.device)
            results.append(result)
            logger.info(
                "%s test_dir_acc_7m=%s test_fires_7m=%s test_pnl_7m=£%s",
                exp_id,
                result.get("test_dir_acc_k5_7m"),
                result.get("test_dir_fires_k5_7m"),
                f"{float(result.get('test_backtest_pnl_k5_7m') or 0):.2f}",
            )
        except Exception as e:  # noqa: BLE001
            logger.error("FAILED %s: %s", exp_id, e)

    if not results:
        logger.error("no results produced — aborting")
        return 1

    write_results(results, S09_RESULTS_PATH)
    print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
