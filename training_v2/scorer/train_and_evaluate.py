"""Phase 0 — Session 02: train, calibrate, evaluate the scorer.

Reads ``data/scorer_v1/dataset/*.parquet`` (output of Session 01),
applies the chronological split locked in
``plans/rewrite/phase-0-supervised-scorer/session_01_findings.md``,
trains LightGBM with the hyperparameters fixed in
``session_prompts/02_train_and_evaluate.md``, calibrates with isotonic
regression on val, and writes the artefacts under
``models/scorer_v1/``:

- ``model.lgb`` (LightGBM Booster save_model text format)
- ``calibrator.joblib`` (sklearn IsotonicRegression)
- ``feature_spec.json`` (copied from data/scorer_v1/)
- ``calibration_curve.png``
- ``feature_importance.png``
- ``eval_summary.json``
- ``training_log.txt``

The success bars are evaluated and reported but never gated on — the
operator decides GREEN/AMBER/RED via the writeup. **No feature
engineering, no model-class change, no hyperparameter search.**

Run with::

    python -m training_v2.scorer.train_and_evaluate
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

from env.scalping_math import (
    equal_profit_back_stake,
    equal_profit_lay_stake,
)
from env.tick_ladder import tick_offset


logger = logging.getLogger("train_and_evaluate")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "data" / "scorer_v1" / "dataset"
FEATURE_SPEC_SRC = REPO_ROOT / "data" / "scorer_v1" / "feature_spec.json"
OUTPUT_DIR = REPO_ROOT / "models" / "scorer_v1"

# Chronological split — locked in session_01_findings.md.
TRAIN_DATE_END = "2026-04-16"   # inclusive
VAL_DATE_END = "2026-04-21"     # inclusive
TEST_DATE_START = "2026-04-22"  # inclusive

# Sizing constants — must match label_generator.py's defaults.
BACK_STAKE = 10.0
COMMISSION = 0.05
ARB_TICKS = 20

# Hyperparameters — locked by 02_train_and_evaluate.md §2.
LGB_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 1000,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbose": -1,
    "force_col_wise": True,
}
N_ESTIMATORS = 5000
EARLY_STOPPING_ROUNDS = 100
N_CALIBRATION_BINS = 10


@dataclass(slots=True)
class Splits:
    """Container for the chronological train/val/test splits."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_names: list[str]


def _load_feature_spec() -> dict[str, Any]:
    with FEATURE_SPEC_SRC.open() as fh:
        return json.load(fh)


def load_dataset(spec: dict[str, Any]) -> pd.DataFrame:
    """Load all per-day parquet shards, drop NaN-label rows."""
    files = sorted(DATASET_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards under {DATASET_DIR}")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    n_total = len(df)
    df = df.dropna(subset=[spec["label_column"]]).reset_index(drop=True)
    logger.info(
        "Loaded %d shards: %d rows total, %d feasible (label not NaN), %d dropped",
        len(files), n_total, len(df), n_total - len(df),
    )
    return df


def chronological_split(df: pd.DataFrame, feature_names: list[str]) -> Splits:
    """Apply the train/val/test split locked in session_01_findings.md."""
    train = df[df["date"] <= TRAIN_DATE_END].reset_index(drop=True)
    val = df[(df["date"] > TRAIN_DATE_END) & (df["date"] <= VAL_DATE_END)].reset_index(drop=True)
    test = df[df["date"] >= TEST_DATE_START].reset_index(drop=True)

    if len(train) + len(val) + len(test) != len(df):
        raise RuntimeError(
            "Split row counts do not sum to total — date boundaries leak. "
            f"train+val+test={len(train) + len(val) + len(test)} vs total={len(df)}",
        )
    return Splits(train=train, val=val, test=test, feature_names=feature_names)


def log_split_balance(splits: Splits) -> None:
    for name, frame in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        n = len(frame)
        pos = float(frame["label"].mean()) if n else 0.0
        logger.info(
            "  %-5s: %7d rows, label.mean=%.4f (positive class share)",
            name, n, pos,
        )
        if name == "test" and n < 100_000:
            logger.warning(
                "Test split has only %d rows — below the 100k threshold "
                "the prompt suggests for AUC bar reliability. Reporting "
                "anyway; flag in findings.",
                n,
            )
        if name == "test" and (pos < 0.05 or pos > 0.95):
            logger.warning(
                "Test split label balance %.3f is outside [0.05, 0.95]; "
                "AUC threshold may not be meaningful.",
                pos,
            )


def train_lightgbm(
    splits: Splits,
    feature_names: list[str],
) -> tuple[lgb.Booster, dict[str, list[float]], float]:
    """Train with early stopping. Returns (booster, eval_history, wall_time)."""
    train_set = lgb.Dataset(
        splits.train[feature_names].to_numpy(),
        label=splits.train["label"].to_numpy(),
        feature_name=feature_names,
        free_raw_data=False,
    )
    val_set = lgb.Dataset(
        splits.val[feature_names].to_numpy(),
        label=splits.val["label"].to_numpy(),
        feature_name=feature_names,
        reference=train_set,
        free_raw_data=False,
    )

    eval_history: dict[str, dict[str, list[float]]] = {}
    t0 = time.perf_counter()
    booster = lgb.train(
        LGB_PARAMS,
        train_set,
        num_boost_round=N_ESTIMATORS,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=100),
            lgb.record_evaluation(eval_history),
        ],
    )
    wall = time.perf_counter() - t0
    logger.info(
        "Training finished: best_iteration=%d, wall=%.1fs",
        booster.best_iteration, wall,
    )
    flat = {
        f"{split}/{metric}": values
        for split, metrics in eval_history.items()
        for metric, values in metrics.items()
    }
    return booster, flat, wall


def predict(booster: lgb.Booster, frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    return booster.predict(
        frame[feature_names].to_numpy(),
        num_iteration=booster.best_iteration,
    )


def calibrate(
    val_raw: np.ndarray, val_labels: np.ndarray,
) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(val_raw, val_labels)
    return iso


def calibration_bin_errors(
    p_calibrated: np.ndarray, labels: np.ndarray, n_bins: int = N_CALIBRATION_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_midpoints, predicted_means, observed_means, bin_counts).

    Bins are equal-width on [0, 1].
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    pred_means = np.full(n_bins, np.nan)
    obs_means = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p_calibrated >= lo) & (p_calibrated <= hi)
        else:
            mask = (p_calibrated >= lo) & (p_calibrated < hi)
        n = int(mask.sum())
        counts[i] = n
        if n:
            pred_means[i] = float(p_calibrated[mask].mean())
            obs_means[i] = float(labels[mask].mean())
    return mids, pred_means, obs_means, counts


def estimate_per_row_pnl(
    df: pd.DataFrame,
    naked_loss_estimate: float,
    force_close_loss_estimate: float,
) -> np.ndarray:
    """Deterministic per-row P&L estimate keyed off `outcome`.

    The greedy-threshold sanity check assumes a stake of ``BACK_STAKE``
    per opportunity. We don't have realised cash P&L per row in the
    dataset (the label simulator throws it away); the prompt tells us
    "if it's hard to wire up the env-replay machinery, file the
    simplification as a finding and ship Bar 3 with whatever you have".

    Rules:
        matured      → +stake × locked_pnl_per_unit_stake(agg, passive)
                       computed from the row's prices and the locked
                       arb_ticks=20 spread.
        force_closed → −force_close_loss_estimate (small ~£spread cost).
        naked        → −naked_loss_estimate (estimated from data; see
                       caller).
        infeasible   → 0 (never opened).

    Returns ``pnl`` aligned with ``df.index``.
    """
    pnl = np.zeros(len(df), dtype=np.float64)
    outcome = df["outcome"].to_numpy()
    side = df["side"].to_numpy()
    best_back = df["best_back"].to_numpy()
    best_lay = df["best_lay"].to_numpy()

    matured_mask = (outcome == "matured")
    if matured_mask.any():
        for i in np.flatnonzero(matured_mask):
            # Aggressive at top-of-book: back side crosses to lay (best_lay
            # is the agg price); lay side crosses to back (best_back).
            # Use equal-profit sizing — locked PnL is bounded and
            # symmetric across win/lose by construction. The lose side
            # is `-S_agg + S_pass × (1 − c)` for an aggressive back, or
            # `-S_pass + S_agg × (1 − c) × P_agg / 1` for an aggressive
            # lay (with appropriate sign conventions). We compute via
            # the lose side which is the simpler closed-form.
            try:
                if side[i] == "back":
                    agg_price = float(best_lay[i])
                    if not (agg_price > 1.0):
                        continue
                    pass_price = tick_offset(agg_price, ARB_TICKS, -1)
                    if pass_price <= 1.0:
                        continue
                    s_lay = equal_profit_lay_stake(
                        back_stake=BACK_STAKE,
                        back_price=agg_price,
                        lay_price=pass_price,
                        commission=COMMISSION,
                    )
                    # Lose-side (race-loss) PnL = -S_back + S_lay×(1-c).
                    locked = -BACK_STAKE + s_lay * (1.0 - COMMISSION)
                else:  # lay aggressive
                    agg_price = float(best_back[i])
                    if not (agg_price > 1.0):
                        continue
                    pass_price = tick_offset(agg_price, ARB_TICKS, +1)
                    if pass_price <= 1.0:
                        continue
                    s_back = equal_profit_back_stake(
                        lay_stake=BACK_STAKE,
                        lay_price=agg_price,
                        back_price=pass_price,
                        commission=COMMISSION,
                    )
                    # Lose-side (race-win) PnL for an agg lay paired
                    # against a back at higher price:
                    #   = -S_lay×(P_lay - 1) + S_back×(P_back - 1)×(1-c)
                    locked = (
                        -BACK_STAKE * (agg_price - 1.0)
                        + s_back * (pass_price - 1.0) * (1.0 - COMMISSION)
                    )
            except (ValueError, ZeroDivisionError):
                continue
            pnl[i] = float(locked)

    pnl[outcome == "force_closed"] = -force_close_loss_estimate
    pnl[outcome == "naked"] = -naked_loss_estimate
    return pnl


def greedy_threshold_pnl(
    df: pd.DataFrame, p_calibrated: np.ndarray, threshold: float,
    naked_loss_estimate: float, force_close_loss_estimate: float,
) -> tuple[float, dict[str, float], np.ndarray]:
    """Open every opportunity where p_calibrated > threshold; sum estimated P&L.

    Returns ``(total, per_day, opened_mask)``.
    """
    pnl = estimate_per_row_pnl(df, naked_loss_estimate, force_close_loss_estimate)
    opened = p_calibrated > threshold
    realised = np.where(opened, pnl, 0.0)
    per_day: dict[str, float] = {}
    for d, g in df.groupby("date"):
        idx = g.index.to_numpy()
        per_day[str(d)] = float(realised[idx].sum())
    return float(realised.sum()), per_day, opened


def estimate_loss_priors(train: pd.DataFrame) -> tuple[float, float]:
    """Heuristic naked + force_close loss magnitudes from train-set spreads.

    We don't simulate to settlement, so naked P&L is a rough proxy:
    take the matured locked P&L magnitude as a per-row spread cost
    estimate, and use:

        force_close_loss ≈ 1.0 × |matured_locked_pnl|  (close-leg crosses
                                                       a thin book at
                                                       roughly the same
                                                       spread cost)
        naked_loss        ≈ 5.0 × |matured_locked_pnl|  (race-outcome
                                                        variance is order
                                                        of magnitude
                                                        bigger than the
                                                        spread)

    These are calibrated against the prompt's "~£1" naked guidance —
    on £10 stake at typical spreads the locked P&L is ~£0.20-£0.40, so
    naked at ~5× is ~£1-2. Documented as a simplification in findings.
    """
    matured = train[train["outcome"] == "matured"]
    if matured.empty:
        return 5.0, 1.0
    pnl = estimate_per_row_pnl(matured, naked_loss_estimate=0.0, force_close_loss_estimate=0.0)
    pnl_pos = pnl[pnl > 0]
    if not pnl_pos.size:
        return 5.0, 1.0
    typical = float(np.median(pnl_pos))
    # naked: 5× the median locked spread is a rough race-outcome
    # variance proxy; force_close: 1× is roughly the cost of crossing
    # a thin opposite-side book once. Both are heuristics — the prompt
    # explicitly allows shipping Bar 3 with a simplified P&L estimate.
    return 5.0 * typical, 1.0 * typical


def tune_threshold_on_val(
    val: pd.DataFrame, p_val_calibrated: np.ndarray,
    naked_loss_estimate: float, force_close_loss_estimate: float,
    candidates: np.ndarray | None = None,
) -> tuple[float, float]:
    """Sweep threshold on val; return (best_threshold, best_pnl)."""
    if candidates is None:
        candidates = np.arange(0.0, 1.0001, 0.01)
    best_thr = 0.5
    best_pnl = -np.inf
    for thr in candidates:
        pnl_total, _, _ = greedy_threshold_pnl(
            val, p_val_calibrated, float(thr),
            naked_loss_estimate, force_close_loss_estimate,
        )
        if pnl_total > best_pnl:
            best_pnl = pnl_total
            best_thr = float(thr)
    return best_thr, best_pnl


def plot_calibration_curve(
    raw_test: np.ndarray, cal_test: np.ndarray, labels_test: np.ndarray,
    out_path: Path,
) -> None:
    raw_mids, raw_pred, raw_obs, raw_counts = calibration_bin_errors(raw_test, labels_test)
    cal_mids, cal_pred, cal_obs, cal_counts = calibration_bin_errors(cal_test, labels_test)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(raw_pred, raw_obs, "o-", label="Raw LightGBM", color="tab:orange")
    ax.plot(cal_pred, cal_obs, "s-", label="Isotonic-calibrated", color="tab:blue")
    ax.set_xlabel("Predicted P(mature)")
    ax.set_ylabel("Observed mature rate")
    ax.set_title("Test-set calibration curve (10 equal-width bins)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_feature_importance(booster: lgb.Booster, out_path: Path, top_n: int = 20) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    lgb.plot_importance(
        booster, ax=ax, max_num_features=top_n, importance_type="gain",
        title="LightGBM feature importance (gain) — top 20",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_artefacts(
    booster: lgb.Booster,
    calibrator: IsotonicRegression,
    feature_spec: dict[str, Any],
    eval_summary: dict[str, Any],
    eval_history: dict[str, list[float]],
    wall_time: float,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    booster.save_model(str(OUTPUT_DIR / "model.lgb"), num_iteration=booster.best_iteration)
    joblib.dump(calibrator, OUTPUT_DIR / "calibrator.joblib")

    with (OUTPUT_DIR / "feature_spec.json").open("w") as fh:
        json.dump(feature_spec, fh, indent=2)

    with (OUTPUT_DIR / "eval_summary.json").open("w") as fh:
        json.dump(eval_summary, fh, indent=2)

    log_lines = [
        "Phase 0 Session 02 training log",
        "================================",
        f"Run start (UTC): {pd.Timestamp.utcnow().isoformat()}",
        "",
        "Hyperparameters:",
        *(f"  {k}: {v}" for k, v in LGB_PARAMS.items()),
        f"  n_estimators (cap): {N_ESTIMATORS}",
        f"  early_stopping_rounds: {EARLY_STOPPING_ROUNDS}",
        f"  best_iteration: {booster.best_iteration}",
        "",
        f"Training wall time: {wall_time:.1f}s",
        "",
        "Environment:",
        f"  python: {sys.version.split()[0]}",
        f"  lightgbm: {lgb.__version__}",
        f"  sklearn: {sklearn.__version__}",
        f"  numpy: {np.__version__}",
        f"  pandas: {pd.__version__}",
        "",
        "Split (locked in session_01_findings.md):",
        f"  train_date_end: {TRAIN_DATE_END}",
        f"  val_date_end: {VAL_DATE_END}",
        f"  test_date_start: {TEST_DATE_START}",
        "",
        "Sizing constants (must match label_generator.py):",
        f"  back_stake: {BACK_STAKE}",
        f"  commission: {COMMISSION}",
        f"  arb_ticks: {ARB_TICKS}",
        "",
        "Train log-loss tail:",
        *(
            f"  iter {i:4d}  train={t:.6f}  val={v:.6f}"
            for i, (t, v) in enumerate(
                zip(
                    eval_history.get("train/binary_logloss", [])[-20:],
                    eval_history.get("val/binary_logloss", [])[-20:],
                ),
                start=max(0, len(eval_history.get("train/binary_logloss", [])) - 20),
            )
        ),
    ]
    (OUTPUT_DIR / "training_log.txt").write_text("\n".join(log_lines) + "\n")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    spec = _load_feature_spec()
    feature_names: list[str] = list(spec["feature_names"])
    logger.info("Loaded feature_spec.json (%d features)", len(feature_names))

    df = load_dataset(spec)
    splits = chronological_split(df, feature_names)
    log_split_balance(splits)

    if splits.test.empty:
        logger.error("Test split is empty — aborting.")
        return 1

    booster, eval_history, wall = train_lightgbm(splits, feature_names)

    raw_train = predict(booster, splits.train, feature_names)
    raw_val = predict(booster, splits.val, feature_names)
    raw_test = predict(booster, splits.test, feature_names)

    train_auc = float(roc_auc_score(splits.train["label"].to_numpy(), raw_train))
    val_auc = float(roc_auc_score(splits.val["label"].to_numpy(), raw_val))
    test_auc_raw = float(roc_auc_score(splits.test["label"].to_numpy(), raw_test))
    logger.info(
        "AUC: train=%.4f val=%.4f test_raw=%.4f", train_auc, val_auc, test_auc_raw,
    )

    # Per-side AUC — Session 01 flagged that side_back dominates the
    # asymmetry; per-side AUCs quantify how much of the headline AUC
    # the model owes to per-tick discrimination versus side prior.
    per_side_auc: dict[str, float] = {}
    for sd in ("back", "lay"):
        mask = (splits.test["side"] == sd).to_numpy()
        n = int(mask.sum())
        if n < 100:
            continue
        labs = splits.test["label"].to_numpy()[mask]
        if labs.min() == labs.max():
            continue  # AUC undefined when all labels identical
        per_side_auc[sd] = float(roc_auc_score(labs, raw_test[mask]))
    for sd, au in per_side_auc.items():
        logger.info("  per-side test AUC: side=%s auc=%.4f", sd, au)

    if val_auc < 0.65 or train_auc < 0.70:
        logger.warning(
            "Underfit / low-signal warning: train_auc=%.4f val_auc=%.4f. "
            "Per session_prompt step 2: stop and discuss before calibrating "
            "if these are persistent. Continuing — operator decides via "
            "findings.md verdict.",
            train_auc, val_auc,
        )

    # ── Calibrate ──
    calibrator = calibrate(raw_val, splits.val["label"].to_numpy())
    cal_test = calibrator.predict(raw_test)
    cal_val = calibrator.predict(raw_val)
    test_auc_cal = float(roc_auc_score(splits.test["label"].to_numpy(), cal_test))
    test_logloss_cal = float(
        log_loss(splits.test["label"].to_numpy(), np.clip(cal_test, 1e-7, 1 - 1e-7)),
    )
    logger.info(
        "Calibrated test: auc=%.4f log_loss=%.4f", test_auc_cal, test_logloss_cal,
    )

    # ── Bar 2: per-bin calibration error on test ──
    mids, pred_means, obs_means, counts = calibration_bin_errors(
        cal_test, splits.test["label"].to_numpy(),
    )
    bin_errors = np.abs(pred_means - obs_means)
    big_bin_mask = counts >= 100
    if big_bin_mask.any():
        max_bin_error = float(np.nanmax(bin_errors[big_bin_mask]))
    else:
        max_bin_error = float("nan")
    logger.info(
        "Calibration: max_bin_error (bins ≥100) = %.4f", max_bin_error,
    )
    for i in range(len(mids)):
        n = int(counts[i])
        flag = "" if n >= 100 else "  (sparse)"
        logger.info(
            "  bin %d  mid=%.2f  predicted=%s  observed=%s  n=%6d%s",
            i, mids[i],
            "nan" if np.isnan(pred_means[i]) else f"{pred_means[i]:.3f}",
            "nan" if np.isnan(obs_means[i]) else f"{obs_means[i]:.3f}",
            n, flag,
        )

    # ── Bar 3: greedy-threshold P&L sanity check ──
    naked_loss, force_close_loss = estimate_loss_priors(splits.train)
    logger.info(
        "P&L priors (estimated from training-set matured spreads): "
        "naked_loss=%.4f, force_close_loss=%.4f",
        naked_loss, force_close_loss,
    )

    best_thr, best_val_pnl = tune_threshold_on_val(
        splits.val, cal_val, naked_loss, force_close_loss,
    )
    logger.info(
        "Threshold tuned on val: thr=%.3f  val_pnl_estimate=£%.2f",
        best_thr, best_val_pnl,
    )

    test_pnl_total, test_pnl_per_day, opened_mask = greedy_threshold_pnl(
        splits.test, cal_test, best_thr, naked_loss, force_close_loss,
    )
    logger.info("Test greedy P&L estimate: total=£%.2f", test_pnl_total)
    for d, p in sorted(test_pnl_per_day.items()):
        logger.info("  %s  £%+.2f", d, p)

    # ── Bar gate evaluation (informational) ──
    bar1_pass = test_auc_cal >= 0.70
    bar2_pass = (not np.isnan(max_bin_error)) and max_bin_error <= 0.10
    pnl_min = min(test_pnl_per_day.values()) if test_pnl_per_day else 0.0
    pnl_max = max(test_pnl_per_day.values()) if test_pnl_per_day else 0.0
    # Bar 3's spirit per prompt: "non-catastrophic — not -£1000s per day".
    # We check the lower bound only; the upper bound is uncapped because
    # the simplified deterministic P&L estimate is known to over-state
    # winners (no execution slippage, no settlement variance modelled).
    # See findings.md "Bar 3 simulation simplifications".
    bar3_pass = pnl_min >= -1000.0
    bar3_strict_pass = (-100.0 <= pnl_min) and (pnl_max <= 100.0)

    logger.info(
        "Bars (informational): "
        "Bar1=%s (auc=%.3f), Bar2=%s (max_bin_err=%.3f), "
        "Bar3=%s (per-day pnl ∈ [%+.2f, %+.2f])",
        "PASS" if bar1_pass else "FAIL", test_auc_cal,
        "PASS" if bar2_pass else "FAIL", max_bin_error,
        "PASS" if bar3_pass else "FAIL", pnl_min, pnl_max,
    )

    # ── Plots ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_calibration_curve(
        raw_test, cal_test, splits.test["label"].to_numpy(),
        OUTPUT_DIR / "calibration_curve.png",
    )
    plot_feature_importance(booster, OUTPUT_DIR / "feature_importance.png")

    # ── eval_summary.json ──
    feature_imp = booster.feature_importance(importance_type="gain")
    feature_imp_pairs = sorted(
        zip(feature_names, feature_imp.tolist()),
        key=lambda kv: kv[1], reverse=True,
    )

    eval_summary: dict[str, Any] = {
        "test_auc": test_auc_cal,
        "test_auc_raw": test_auc_raw,
        "test_log_loss": test_logloss_cal,
        "train_auc": train_auc,
        "val_auc": val_auc,
        "test_auc_per_side": per_side_auc,
        "calibration": {
            "max_bin_error": max_bin_error,
            "bin_midpoints": [
                None if np.isnan(x) else float(x) for x in mids.tolist()
            ],
            "bin_predicted_means": [
                None if np.isnan(x) else float(x) for x in pred_means.tolist()
            ],
            "bin_observed_means": [
                None if np.isnan(x) else float(x) for x in obs_means.tolist()
            ],
            "bin_errors": [
                None if np.isnan(x) else float(x) for x in bin_errors.tolist()
            ],
            "bin_counts": [int(c) for c in counts.tolist()],
        },
        "greedy_pnl_test": {
            "per_day": {k: float(v) for k, v in test_pnl_per_day.items()},
            "total": float(test_pnl_total),
            "threshold_used": float(best_thr),
            "val_pnl_at_threshold": float(best_val_pnl),
            "naked_loss_estimate": float(naked_loss),
            "force_close_loss_estimate": float(force_close_loss),
            "opened_count": int(opened_mask.sum()),
            "opened_share": float(opened_mask.mean()) if len(opened_mask) else 0.0,
            "note": (
                "Per-row P&L estimated from outcome class with deterministic "
                "spread math; naked P&L is a heuristic 5× the median locked "
                "spread (race-outcome variance not simulated). See findings.md."
            ),
        },
        "bar_gate": {
            "bar1_test_auc_ge_0_70": bool(bar1_pass),
            "bar2_max_bin_error_le_0_10": bool(bar2_pass),
            "bar3_non_catastrophic_per_day_pnl_ge_-1000": bool(bar3_pass),
            "bar3_strict_per_day_pnl_within_100": bool(bar3_strict_pass),
        },
        "n_train_rows": int(len(splits.train)),
        "n_val_rows": int(len(splits.val)),
        "n_test_rows": int(len(splits.test)),
        "n_estimators_after_early_stop": int(booster.best_iteration),
        "feature_importance_gain": [
            {"feature": name, "gain": float(g)} for name, g in feature_imp_pairs
        ],
    }

    write_artefacts(booster, calibrator, spec, eval_summary, eval_history, wall)

    logger.info("Wrote artefacts to %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
