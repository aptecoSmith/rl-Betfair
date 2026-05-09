"""scripts/predictor/train_one.py - train + evaluate ONE candidate.

Reads a YAML config, produces:
  - one row in registry/predictor_scoreboard.csv
  - one model weights file in registry/predictor/{experiment_id}.{ext}
  - one model card in plans/price-direction-predictor/models/{experiment_id}.md

Idempotent: if experiment_id already exists in the scoreboard the
script exits 0 without retraining unless --rebuild is passed.

Run:
    python scripts/predictor/train_one.py --config path/to/config.yaml [--rebuild]

Config schema:
    session: S03
    seed: 0
    dataset:
      feature_variant: V3
      train_corpus: tvl_required_10d
      horizons: [3m, 7m, 15m]
    architecture:
      family: lstm
      kwargs: {time_window: 32, hidden: 64, layers: 2}
      variant_label: tw32   # short string for filename
    output:
      formulation: pinball3
      quantiles: [0.1, 0.5, 0.9]
    training:
      batch_size: 1024
      learning_rate: 0.001
      max_epochs: 20
      early_stopping_patience: 3
    device: cuda
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.predictor.datasets import (  # noqa: E402
    TabularDataset, TabularExamples,
    SequenceDataset,
    build_sequence_examples,
    feature_columns,
    load_split,
    to_feature_tensor,
    to_label_tensors,
)
from scripts.predictor.eval_metrics import (  # noqa: E402
    calibration_gap, coverage,
    directional_accuracy,
    lag1_autocorr_per_group,
    mae,
    naive_backtest_pnl,
    pinball_loss,
)
from scripts.predictor.models import (  # noqa: E402
    build_model,
    count_parameters,
    is_sequence_family,
    is_torch_family,
)

logger = logging.getLogger(__name__)

DEFAULT_SCOREBOARD = REPO_ROOT / "registry" / "predictor_scoreboard.csv"
DEFAULT_MODEL_DIR = REPO_ROOT / "registry" / "predictor"
DEFAULT_CARD_DIR = (
    REPO_ROOT / "plans" / "price-direction-predictor" / "models"
)


# ----------------------------------------------------------------- config


def hash_config(cfg: dict) -> str:
    """Deterministic 12-char hash of the config (excluding `experiment_id`)."""
    cfg_copy = {k: v for k, v in cfg.items() if k != "experiment_id"}
    canon = json.dumps(cfg_copy, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


def experiment_id_for(cfg: dict) -> str:
    h = hash_config(cfg)
    family = cfg["architecture"]["family"]
    variant = cfg["architecture"].get("variant_label", "v")
    seed = cfg.get("seed", 0)
    return f"{family}_{variant}_s{seed}_{h}"


# ----------------------------------------------------------------- training


def quantile_pinball_loss_torch(
    pred: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    quantiles: list[float],
) -> torch.Tensor:
    """Mean pinball loss over masked entries.

    pred:     (B, H, Q)
    y, mask:  (B, H)

    NaN labels are pre-zeroed BEFORE the loss computation -- otherwise
    `0 * NaN = NaN` and the mean propagates NaN through training.
    """
    qs = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype)
    qs = qs.view(1, 1, -1)  # broadcast over (B, H)
    # Replace NaN labels with 0 (their contribution will be zeroed by the
    # mask multiply anyway -- but we must avoid 0 * NaN = NaN on the way
    # through the maximum() expression).
    y_clean = torch.where(mask, y, torch.zeros_like(y))
    err = y_clean.unsqueeze(-1) - pred  # (B, H, Q)
    loss = torch.maximum(qs * err, (qs - 1) * err)  # (B, H, Q)
    m = mask.unsqueeze(-1).to(pred.dtype)
    masked = loss * m
    denom = m.sum() * len(quantiles) + 1e-9
    return masked.sum() / denom


def train_torch_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    quantiles: list[float],
    learning_rate: float,
    max_epochs: int,
    early_stopping_patience: int,
    device: str,
) -> tuple[torch.nn.Module, dict]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val = float("inf")
    best_state: dict | None = None
    epochs_no_improve = 0
    history: list[dict] = []

    for epoch in range(max_epochs):
        # Train.
        model.train()
        t0 = time.time()
        train_losses: list[float] = []
        for x, y, mask in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            pred = model(x)
            loss = quantile_pinball_loss_torch(pred, y, mask, quantiles)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.item()))
        train_loss = float(np.mean(train_losses))

        # Eval.
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for x, y, mask in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                pred = model(x)
                val_losses.append(float(
                    quantile_pinball_loss_torch(pred, y, mask, quantiles).item()
                ))
        val_loss = float(np.mean(val_losses))
        elapsed = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "seconds": elapsed,
        })
        logger.info(
            "ep%2d | train %.4f | val %.4f | %.1fs",
            epoch, train_loss, val_loss, elapsed,
        )

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                logger.info("early stop at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_loss": best_val, "history": history}


def predict_torch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _y, _m in loader:
            x = x.to(device)
            preds.append(model(x).cpu().numpy())
    return np.concatenate(preds, axis=0)  # (N, H, Q)


# ----------------------------------------------------------------- GBM (lightgbm)


def train_gbm(
    X_train: np.ndarray, y_train: np.ndarray, mask_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray, mask_val: np.ndarray,
    horizons: list[str],
    quantiles: list[float],
    n_trees: int, max_depth: int, learning_rate: float,
) -> dict[tuple[int, float], Any]:
    """Train one lightgbm regressor per (horizon, quantile)."""
    import lightgbm as lgb
    boosters: dict[tuple[int, float], Any] = {}
    for h_idx in range(len(horizons)):
        for q in quantiles:
            mt = mask_train[:, h_idx]
            mv = mask_val[:, h_idx]
            params = {
                "objective": "quantile",
                "alpha": float(q),
                "learning_rate": float(learning_rate),
                "num_leaves": 2 ** max_depth,
                "max_depth": int(max_depth),
                "min_data_in_leaf": 50,
                "verbosity": -1,
                "deterministic": True,
                "force_col_wise": True,
            }
            dtr = lgb.Dataset(X_train[mt], label=y_train[mt, h_idx])
            dval = lgb.Dataset(X_val[mv], label=y_val[mv, h_idx])
            booster = lgb.train(
                params,
                dtr,
                num_boost_round=n_trees,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
            )
            boosters[(h_idx, q)] = booster
    return boosters


def predict_gbm(
    boosters: dict, X: np.ndarray, n_horizons: int, quantiles: list[float],
) -> np.ndarray:
    out = np.zeros((X.shape[0], n_horizons, len(quantiles)), dtype=np.float32)
    for h_idx in range(n_horizons):
        for q_idx, q in enumerate(quantiles):
            out[:, h_idx, q_idx] = boosters[(h_idx, q)].predict(X)
    return out


# ----------------------------------------------------------------- evaluation


def evaluate_predictions(
    pred: np.ndarray,            # (N, H, Q)
    y: np.ndarray,               # (N, H)
    mask: np.ndarray,            # (N, H)
    ltp_now: np.ndarray,         # (N,)
    horizons: list[str],
    quantiles: list[float],
    df_for_groups: pd.DataFrame, # for per-(market, runner) trajectories
) -> dict[str, Any]:
    """Compute all scoreboard metrics on the val (or test) set."""
    out: dict[str, Any] = {}
    q_idx = {q: i for i, q in enumerate(quantiles)}
    q10_i = q_idx.get(0.1)
    q50_i = q_idx.get(0.5)
    q90_i = q_idx.get(0.9)

    for h_idx, h in enumerate(horizons):
        m = mask[:, h_idx]
        if m.sum() == 0:
            continue
        y_h = y[m, h_idx]
        pred_h = pred[m, h_idx, :]
        # Pinball per quantile.
        for q_idx_i, q in enumerate(quantiles):
            pl = pinball_loss(y_h, pred_h[:, q_idx_i], q)
            out[f"pinball_{h}_q{int(q*100)}"] = round(pl, 6)
        # MAE on q50.
        if q50_i is not None:
            out[f"mae_{h}"] = round(mae(y_h, pred_h[:, q50_i]), 6)
        # Calibration on q10/q90.
        if q10_i is not None and q90_i is not None:
            out[f"coverage_{h}"] = round(coverage(y_h, pred_h[:, q10_i], pred_h[:, q90_i]), 4)
            out[f"calibration_gap_{h}"] = round(
                calibration_gap(y_h, pred_h[:, q10_i], pred_h[:, q90_i], 0.8), 4,
            )
        # Directional accuracy at k=5.
        if q10_i is not None and q50_i is not None and q90_i is not None:
            da = directional_accuracy(
                y_h, pred_h[:, q10_i], pred_h[:, q50_i], pred_h[:, q90_i], k_ticks=5,
            )
            out[f"dir_acc_k5_{h}"] = (
                round(da["total_acc"], 4) if not np.isnan(da["total_acc"]) else None
            )
            out[f"dir_fires_k5_{h}"] = da["n_total_fires"]
            out[f"dir_fire_rate_k5_{h}"] = round(da["fire_rate"], 4)

            # Backtest sanity (on h only).
            ltp_for_h = ltp_now[m]
            bt = naive_backtest_pnl(
                y_h, pred_h[:, q10_i], pred_h[:, q50_i], pred_h[:, q90_i],
                ltp_for_h, k_ticks=5,
            )
            out[f"backtest_pnl_k5_{h}"] = round(bt["total_pnl"], 4)
            out[f"backtest_winrate_k5_{h}"] = (
                round(bt["win_rate"], 4) if not np.isnan(bt["win_rate"]) else None
            )

        # Stability (lag-1 autocorr of q50 across each (market, runner) trajectory).
        if q50_i is not None:
            df_pred = df_for_groups.copy()
            df_pred["_q50"] = pred[:, h_idx, q50_i]
            df_pred = df_pred[m]
            groups = (
                df_pred.sort_values(["market_id", "selection_id", "tick_idx"])
                       .groupby(["market_id", "selection_id"])["_q50"]
                       .apply(np.asarray)
                       .tolist()
            )
            ac = lag1_autocorr_per_group(groups)
            out[f"lag1_autocorr_q50_{h}"] = round(ac, 4) if not np.isnan(ac) else None

    return out


# ----------------------------------------------------------------- scoreboard


def append_scoreboard_row(
    scoreboard_path: Path,
    row: dict[str, Any],
    *,
    overwrite_existing: bool = False,
) -> None:
    """Append a row, or replace it if `overwrite_existing` is set
    (only set this from a `--rebuild` code path; never silently).
    """
    scoreboard_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = scoreboard_path.exists()
    if file_exists:
        existing = pd.read_csv(scoreboard_path)
        if (
            "experiment_id" in existing.columns
            and row["experiment_id"] in set(existing["experiment_id"])
        ):
            if not overwrite_existing:
                # hard_constraints sec 12.
                raise RuntimeError(
                    f"experiment_id {row['experiment_id']} already in "
                    f"scoreboard -- refuse to overwrite (sec 12). Use "
                    f"--rebuild to force.",
                )
            existing = existing[
                existing["experiment_id"] != row["experiment_id"]
            ].copy()
        all_cols = list(dict.fromkeys(list(existing.columns) + list(row.keys())))
        df_new = pd.DataFrame([row])
        existing = existing.reindex(columns=all_cols)
        df_new = df_new.reindex(columns=all_cols)
        out = pd.concat([existing, df_new], ignore_index=True)
        out.to_csv(scoreboard_path, index=False)
    else:
        pd.DataFrame([row]).to_csv(scoreboard_path, index=False)


def scoreboard_has(scoreboard_path: Path, experiment_id: str) -> bool:
    if not scoreboard_path.exists():
        return False
    df = pd.read_csv(scoreboard_path)
    if "experiment_id" not in df.columns:
        return False
    return experiment_id in set(df["experiment_id"])


# ----------------------------------------------------------------- model card


def write_model_card(
    card_path: Path,
    cfg: dict,
    experiment_id: str,
    metrics: dict,
    extra: dict,
) -> None:
    card_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Predictor model card: {experiment_id}",
        "",
        f"- session: {cfg.get('session')}",
        f"- seed: {cfg.get('seed')}",
        f"- architecture: {cfg['architecture']['family']} "
        f"({cfg['architecture'].get('variant_label')})",
        f"- arch_kwargs: `{json.dumps(cfg['architecture'].get('kwargs', {}))}`",
        f"- feature variant: {cfg['dataset']['feature_variant']}",
        f"- train corpus: {cfg['dataset']['train_corpus']}",
        f"- horizons: {cfg['dataset']['horizons']}",
        f"- output: {cfg['output']['formulation']} "
        f"(quantiles {cfg['output']['quantiles']})",
        f"- training: lr={cfg['training']['learning_rate']}, "
        f"batch={cfg['training']['batch_size']}, "
        f"max_epochs={cfg['training']['max_epochs']}",
        "",
        "## Run extras",
        "```json",
        json.dumps(extra, indent=2, default=str),
        "```",
        "",
        "## Val metrics",
        "```json",
        json.dumps(metrics, indent=2, default=str),
        "```",
    ]
    card_path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--scoreboard", default=str(DEFAULT_SCOREBOARD))
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--card-dir", default=str(DEFAULT_CARD_DIR))
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    experiment_id = experiment_id_for(cfg)
    logger.info("experiment_id: %s", experiment_id)

    scoreboard_path = Path(args.scoreboard)
    if not args.rebuild and scoreboard_has(scoreboard_path, experiment_id):
        logger.info("already in scoreboard, skipping (use --rebuild to force)")
        return 0

    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    family = cfg["architecture"]["family"]
    arch_kwargs = cfg["architecture"].get("kwargs", {})
    quantiles = list(cfg["output"]["quantiles"])
    horizons = list(cfg["dataset"]["horizons"])
    feature_variant = cfg["dataset"]["feature_variant"]
    train_corpus = cfg["dataset"]["train_corpus"]

    # Load train / val.
    t0 = time.time()
    df_train = load_split("train", feature_variant, horizons, train_corpus)
    df_val = load_split("val", feature_variant, horizons, train_corpus)
    logger.info(
        "loaded train=%s val=%s in %.1fs",
        f"{len(df_train):,}", f"{len(df_val):,}", time.time() - t0,
    )

    # Build features + labels.
    n_features = len(feature_columns(feature_variant))
    n_horizons = len(horizons)
    train_kwargs = cfg.get("training", {})

    if family.lower() == "gbm":
        # Tabular path, lightgbm.
        X_train, _ = to_feature_tensor(df_train, feature_variant)
        y_train, mask_train = to_label_tensors(df_train, horizons)
        X_val, _ = to_feature_tensor(df_val, feature_variant)
        y_val, mask_val = to_label_tensors(df_val, horizons)
        ltp_val = df_val["ltp"].to_numpy(dtype=np.float32)

        t0 = time.time()
        boosters = train_gbm(
            X_train, y_train, mask_train,
            X_val, y_val, mask_val,
            horizons=horizons,
            quantiles=quantiles,
            n_trees=int(arch_kwargs.get("n_trees", 300)),
            max_depth=int(arch_kwargs.get("max_depth", 5)),
            learning_rate=float(arch_kwargs.get("learning_rate", 0.05)),
        )
        train_seconds = time.time() - t0
        pred_val = predict_gbm(boosters, X_val, n_horizons, quantiles)
        param_count = sum(
            booster.num_trees() for booster in boosters.values()
        )
    elif is_torch_family(family):
        # Decide tabular vs sequence.
        if is_sequence_family(family):
            time_window = int(arch_kwargs.get(
                "time_window",
                arch_kwargs.get("ctx_ticks", 32),
            ))
            seq_train = build_sequence_examples(
                df_train, feature_variant, horizons, time_window,
            )
            seq_val = build_sequence_examples(
                df_val, feature_variant, horizons, time_window,
            )
            train_ds = SequenceDataset(seq_train)
            val_ds = SequenceDataset(seq_val)
            ltp_val = df_val.sort_values(
                ["market_id", "selection_id", "tick_idx"]
            )["ltp"].to_numpy(dtype=np.float32)
            df_val_for_groups = df_val.sort_values(
                ["market_id", "selection_id", "tick_idx"]
            ).reset_index(drop=True)
        else:
            X_train, _ = to_feature_tensor(df_train, feature_variant)
            y_train, mask_train = to_label_tensors(df_train, horizons)
            X_val, _ = to_feature_tensor(df_val, feature_variant)
            y_val, mask_val = to_label_tensors(df_val, horizons)
            train_ds = TabularDataset(TabularExamples(
                X=X_train, y=y_train, mask=mask_train,
                feature_names=feature_columns(feature_variant),
                horizons=horizons,
            ))
            val_ds = TabularDataset(TabularExamples(
                X=X_val, y=y_val, mask=mask_val,
                feature_names=feature_columns(feature_variant),
                horizons=horizons,
            ))
            ltp_val = df_val["ltp"].to_numpy(dtype=np.float32)
            df_val_for_groups = df_val.reset_index(drop=True)

        train_loader = DataLoader(
            train_ds,
            batch_size=int(train_kwargs.get("batch_size", 1024)),
            shuffle=True, num_workers=0, pin_memory=(device == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(train_kwargs.get("batch_size", 1024)),
            shuffle=False, num_workers=0, pin_memory=(device == "cuda"),
        )

        model = build_model(
            family=family,
            n_features=n_features,
            n_horizons=n_horizons,
            n_quantiles=len(quantiles),
            arch_kwargs=arch_kwargs,
        )
        param_count = count_parameters(model)
        logger.info("param_count: %s", f"{param_count:,}")

        t0 = time.time()
        model, train_info = train_torch_model(
            model, train_loader, val_loader,
            quantiles=quantiles,
            learning_rate=float(train_kwargs.get("learning_rate", 1e-3)),
            max_epochs=int(train_kwargs.get("max_epochs", 20)),
            early_stopping_patience=int(train_kwargs.get("early_stopping_patience", 3)),
            device=device,
        )
        train_seconds = time.time() - t0

        pred_val = predict_torch(model, val_loader, device)
    else:
        raise ValueError(f"unknown architecture family: {family!r}")

    # Build the y/mask aligned with pred_val ordering. For sequence
    # path the dataset was built sorted; flat path uses df_val ordering.
    if is_torch_family(family) and is_sequence_family(family):
        y_eval = seq_val.y
        mask_eval = seq_val.mask
    else:
        # X_val/y_val/mask_val already in df_val order.
        y_eval = y_val if family.lower() == "gbm" else y_val
        mask_eval = mask_val if family.lower() == "gbm" else mask_val

    metrics = evaluate_predictions(
        pred=pred_val,
        y=y_eval,
        mask=mask_eval,
        ltp_now=ltp_val,
        horizons=horizons,
        quantiles=quantiles,
        df_for_groups=df_val_for_groups if (is_torch_family(family) and is_sequence_family(family)) else df_val.reset_index(drop=True),
    )

    # Save weights.
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if family.lower() == "gbm":
        # Save each booster.
        import joblib
        wpath = model_dir / f"{experiment_id}.joblib"
        joblib.dump(boosters, wpath)
    else:
        wpath = model_dir / f"{experiment_id}.pt"
        torch.save(model.state_dict(), wpath)
    logger.info("saved weights to %s", wpath)

    # Inference timing on a 1024-row batch (best-effort sanity).
    infer_us = None
    try:
        if family.lower() == "gbm":
            X_sample = X_val[:1024]
            t0 = time.time()
            _ = predict_gbm(boosters, X_sample, n_horizons, quantiles)
            infer_us = (time.time() - t0) * 1e6 / max(1, X_sample.shape[0])
        else:
            model.eval()
            with torch.no_grad():
                # First batch from val_loader.
                for x, _, _ in val_loader:
                    x = x.to(device)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.time()
                    _ = model(x)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    infer_us = (time.time() - t0) * 1e6 / max(1, x.shape[0])
                    break
    except Exception as e:  # noqa: BLE001
        logger.warning("inference timing failed: %s", e)

    # Build scoreboard row.
    row = {
        "experiment_id": experiment_id,
        "session": cfg.get("session"),
        "timestamp": datetime.utcnow().isoformat(),
        "architecture": family,
        "variant_label": cfg["architecture"].get("variant_label"),
        "arch_kwargs": json.dumps(arch_kwargs, default=str),
        "feature_variant": feature_variant,
        "train_corpus": train_corpus,
        "horizons": ",".join(horizons),
        "output_formulation": cfg["output"]["formulation"],
        "quantiles": ",".join(str(q) for q in quantiles),
        "seed": seed,
        "param_count": param_count,
        "train_seconds": round(train_seconds, 1),
        "infer_us_per_row": round(infer_us, 2) if infer_us is not None else None,
        "weights_path": str(wpath.relative_to(REPO_ROOT)),
        "config_hash": hash_config(cfg),
        **metrics,
    }
    append_scoreboard_row(
        scoreboard_path, row, overwrite_existing=bool(args.rebuild),
    )
    logger.info("scoreboard row written")

    # Model card.
    write_model_card(
        card_path=Path(args.card_dir) / f"{experiment_id}.md",
        cfg=cfg,
        experiment_id=experiment_id,
        metrics=metrics,
        extra={
            "param_count": param_count,
            "train_seconds": round(train_seconds, 1),
            "infer_us_per_row": infer_us,
            "device": device,
            "weights_path": str(wpath),
        },
    )
    logger.info("model card written")

    # Defensive teardown to avoid Windows STATUS_STACK_BUFFER_OVERRUN
    # (0xC0000409) on LSTM cuDNN cleanup. All 9 S03 LSTM runs landed
    # their scoreboard rows successfully then crashed on process exit;
    # without this the matrix runner counts those as exit-1 failures
    # even though the data is good. See S03_findings.md.
    try:
        import gc
        del_targets = [name for name in (
            "model", "boosters", "train_loader", "val_loader", "train_ds",
            "val_ds", "seq_train", "seq_val", "X_train", "X_val", "y_train",
            "y_val", "mask_train", "mask_val", "df_train", "df_val",
        ) if name in locals()]
        for name in del_targets:
            del locals()[name]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
