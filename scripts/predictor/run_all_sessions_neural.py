"""scripts/predictor/run_all_sessions_neural.py - neural-lineage orchestrator S04→S08.

This is the NEURAL lineage run, separate from the GBM lineage in run_all_sessions.py.

Key differences vs GBM lineage:
  - S04 base: conv1d/k3 + lstm/tw32 (S03 best-MAE neural cells, hardcoded)
  - Downselect criterion: median val MEAN_MAE (lower = better) — NOT dir_acc.
    GBM produced 0 dir_acc fires at k=5; neural models are expected to produce
    real directional signals. MAE is the fair primary criterion until we know.
  - Scoreboard: predictor_scoreboard_neural.csv (separate from GBM scoreboard)
  - Config dirs: configs/predictor/S04_neural/, S05_neural/, S06_neural/

Run:
    python scripts/predictor/run_all_sessions_neural.py
    python scripts/predictor/run_all_sessions_neural.py --start-from S05_neural
    python scripts/predictor/run_all_sessions_neural.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCOREBOARD_DEFAULT = REPO_ROOT / "registry" / "predictor_scoreboard_neural.csv"
CONFIG_ROOT = REPO_ROOT / "configs" / "predictor"
LOGS_DIR = REPO_ROOT / "logs"

SEEDS: tuple[int, ...] = (0, 1, 2)

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------- S03 neural leaders
# Source: S03 scoreboard, top neural cells by 7m MAE (session=smoke confirmed
# conv1d_k3 at 81% dir_acc; S03 sweep confirmed conv1d and lstm lead on MAE
# within neural families). Hardcoded so this orchestrator does not depend on
# the GBM scoreboard.
S04_NEURAL_CELLS = [
    {
        "family": "conv1d",
        "variant_label": "k3",
        "arch_kwargs": {"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1},
    },
    {
        "family": "lstm",
        "variant_label": "tw32",
        "arch_kwargs": {"time_window": 32, "hidden": 64, "layers": 2, "dropout": 0.1},
    },
]

S04_CORPORA: dict[str, list[str]] = {
    "V1": ["tvl_mask_29d"],
    "V2": ["tvl_mask_29d"],
    "V3": ["tvl_required_10d", "tvl_mask_29d"],
    "V4": ["tvl_required_10d", "tvl_mask_29d"],
    "V5": ["tvl_required_10d", "tvl_mask_29d"],
}

S06_HORIZON_SETS: dict[str, list[str]] = {
    "3m_7m_15m":  ["3m", "7m", "15m"],
    "1m_3m_7m":   ["1m", "3m", "7m"],
    "1m_7m_15m":  ["1m", "7m", "15m"],
    "7m_only":    ["7m"],
    "1m_only":    ["1m"],
    "3m_only":    ["3m"],
    "15m_only":   ["15m"],
}

SESSION_ORDER = ["S04_neural", "S05_neural", "S06_neural", "S07_neural", "S08_neural"]


# ----------------------------------------------------------------- shared

def base_training_kwargs(family: str) -> dict:
    if family == "transformer":
        return {
            "batch_size": 512,
            "learning_rate": 5e-4,
            "max_epochs": 20,
            "early_stopping_patience": 3,
        }
    return {
        "batch_size": 1024,
        "learning_rate": 1e-3,
        "max_epochs": 20,
        "early_stopping_patience": 3,
    }


def write_configs(cfgs: list[dict], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for cfg in cfgs:
        family = cfg["architecture"]["family"]
        variant = cfg["architecture"]["variant_label"]
        fv = cfg["dataset"]["feature_variant"]
        corpus = cfg["dataset"]["train_corpus"]
        horizons_str = "_".join(cfg["dataset"]["horizons"])
        formulation = cfg["output"]["formulation"]
        seed = cfg["seed"]
        fname = f"{family}_{variant}_{fv}_{corpus}_{horizons_str}_{formulation}_s{seed}.yaml"
        (out_dir / fname).write_text(
            yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        n += 1
    logger.info("wrote %d configs to %s", n, out_dir)
    return n


def run_matrix_session(internal_session: str, config_dir: Path, scoreboard: Path) -> int:
    """internal_session is passed to run_matrix --session (stored in scoreboard rows)."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "predictor" / "run_matrix.py"),
        "--session", internal_session,
        "--config-dir", str(config_dir),
        "--scoreboard", str(scoreboard),
    ]
    logger.info("run_matrix: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    return result.returncode


def git_commit(message: str) -> None:
    try:
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "add",
             "registry/", "configs/predictor/"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "commit", "-m", message],
            check=True,
        )
        logger.info("committed: %s", message)
    except subprocess.CalledProcessError as e:
        logger.warning("git commit failed (non-fatal): %s", e)


def read_scoreboard(scoreboard: Path) -> pd.DataFrame:
    if not scoreboard.exists():
        return pd.DataFrame()
    return pd.read_csv(scoreboard)


def _mean_mae(df: pd.DataFrame) -> pd.Series:
    mae_cols = [c for c in ["mae_3m", "mae_7m", "mae_15m"] if c in df.columns]
    if not mae_cols:
        raise RuntimeError("No mae_* columns in scoreboard — cannot downselect by MAE.")
    return df[mae_cols].mean(axis=1)


# ----------------------------------------------------------------- S04 neural

def generate_s04_configs(horizons: list[str] | None = None) -> list[dict]:
    if horizons is None:
        horizons = ["3m", "7m", "15m"]
    cfgs = []
    for cell in S04_NEURAL_CELLS:
        family = cell["family"]
        for fv, corpora in S04_CORPORA.items():
            for corpus in corpora:
                for seed in SEEDS:
                    cfgs.append({
                        "session": "S04_neural",
                        "seed": seed,
                        "dataset": {
                            "feature_variant": fv,
                            "train_corpus": corpus,
                            "horizons": horizons,
                        },
                        "architecture": {
                            "family": family,
                            "variant_label": cell["variant_label"],
                            "kwargs": cell["arch_kwargs"],
                        },
                        "output": {
                            "formulation": "pinball3",
                            "quantiles": [0.1, 0.5, 0.9],
                        },
                        "training": base_training_kwargs(family),
                        "device": "cuda",
                    })
    return cfgs


def run_s04(scoreboard: Path, dry_run: bool = False) -> None:
    cfgs = generate_s04_configs()
    out_dir = CONFIG_ROOT / "S04_neural"
    write_configs(cfgs, out_dir)
    logger.info(
        "S04_neural: %d configs (%d neural cells × 5 variants × 8 corpora avg × %d seeds). "
        "DOWNSELECT CRITERION: median mean_mae (lower = better).",
        len(cfgs), len(S04_NEURAL_CELLS), len(SEEDS),
    )
    if not dry_run:
        rc = run_matrix_session("S04_neural", out_dir, scoreboard)
        logger.info("S04_neural run_matrix rc=%d", rc)
        git_commit("feat(predictor): S04_neural feature-variant sweep (conv1d + lstm)")


# ----------------------------------------------------------------- S05 neural

def downselect_s04(df: pd.DataFrame) -> dict:
    """Best (arch, variant, fv, corpus) cell from S04_neural by median mean_mae."""
    s04 = df[df["session"] == "S04_neural"].copy()
    if s04.empty:
        raise RuntimeError("No S04_neural rows — run S04_neural first.")
    s04["mean_mae"] = _mean_mae(s04)
    grp = (
        s04.dropna(subset=["mean_mae"])
        .groupby(["architecture", "variant_label", "feature_variant", "train_corpus"])["mean_mae"]
        .median()
    )
    best_idx = grp.idxmin()
    arch, variant, fv, corpus = best_idx
    row = s04[
        (s04["architecture"] == arch)
        & (s04["variant_label"] == variant)
        & (s04["feature_variant"] == fv)
        & (s04["train_corpus"] == corpus)
    ].iloc[0]
    result = {
        "family": arch,
        "variant_label": variant,
        "feature_variant": fv,
        "train_corpus": corpus,
        "arch_kwargs": json.loads(row["arch_kwargs"]),
    }
    logger.info(
        "S04_neural best cell (by mean_mae): %s/%s %s %s (median mean_mae=%.4f)",
        arch, variant, fv, corpus, grp[best_idx],
    )
    return result


def generate_s05_configs(best_cell: dict, horizons: list[str] | None = None) -> list[dict]:
    """pinball5 on best S04_neural cell. gaussian/student_t/classification deferred."""
    if horizons is None:
        horizons = ["3m", "7m", "15m"]
    family = best_cell["family"]
    cfgs = []
    for seed in SEEDS:
        cfgs.append({
            "session": "S05_neural",
            "seed": seed,
            "dataset": {
                "feature_variant": best_cell["feature_variant"],
                "train_corpus": best_cell["train_corpus"],
                "horizons": horizons,
            },
            "architecture": {
                "family": family,
                "variant_label": best_cell["variant_label"],
                "kwargs": best_cell["arch_kwargs"],
            },
            "output": {
                "formulation": "pinball5",
                "quantiles": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
            "training": base_training_kwargs(family),
            "device": "cuda",
        })
    return cfgs


def run_s05(scoreboard: Path, dry_run: bool = False) -> None:
    df = read_scoreboard(scoreboard)
    best_cell = downselect_s04(df)
    cfgs = generate_s05_configs(best_cell)
    out_dir = CONFIG_ROOT / "S05_neural"
    write_configs(cfgs, out_dir)
    logger.info(
        "S05_neural: %d configs (pinball5 on %s/%s %s %s). "
        "NOTE: gaussian/student_t/classification deferred — require train_one.py changes.",
        len(cfgs),
        best_cell["family"], best_cell["variant_label"],
        best_cell["feature_variant"], best_cell["train_corpus"],
    )
    if not dry_run:
        rc = run_matrix_session("S05_neural", out_dir, scoreboard)
        logger.info("S05_neural run_matrix rc=%d", rc)
        git_commit("feat(predictor): S05_neural output-formulation sweep (pinball5)")


# ----------------------------------------------------------------- S06 neural

def downselect_s05(df: pd.DataFrame, s04_best: dict) -> dict:
    """Compare S04_neural (pinball3) vs S05_neural (pinball5) by median mean_mae."""
    arch = s04_best["family"]
    variant = s04_best["variant_label"]
    fv = s04_best["feature_variant"]
    corpus = s04_best["train_corpus"]

    candidates = df[
        (df["architecture"] == arch)
        & (df["variant_label"] == variant)
        & (df["feature_variant"] == fv)
        & (df["train_corpus"] == corpus)
        & (df["session"].isin(["S04_neural", "S05_neural"]))
    ].copy()

    default = {**s04_best, "formulation": "pinball3", "quantiles": [0.1, 0.5, 0.9]}

    if candidates.empty:
        logger.warning("S05_neural downselect: no candidates — defaulting to pinball3")
        return default

    candidates["mean_mae"] = _mean_mae(candidates)
    by_form = candidates.dropna(subset=["mean_mae"]).groupby("output_formulation")["mean_mae"].median()
    if by_form.empty:
        logger.warning("S05_neural downselect: mean_mae missing — defaulting to pinball3")
        return default

    logger.info("S04_neural / S05_neural formulation comparison (mean_mae, lower=better):\n%s",
                by_form.to_string())
    best_form = by_form.idxmin()
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9] if best_form == "pinball5" else [0.1, 0.5, 0.9]
    logger.info("S05_neural winner: %s (mean_mae=%.4f)", best_form, by_form[best_form])
    return {**s04_best, "formulation": best_form, "quantiles": quantiles}


def generate_s06_configs(best_cell: dict) -> list[dict]:
    family = best_cell["family"]
    cfgs = []
    for hset_name, horizons in S06_HORIZON_SETS.items():
        for seed in SEEDS:
            cfgs.append({
                "session": "S06_neural",
                "seed": seed,
                "dataset": {
                    "feature_variant": best_cell["feature_variant"],
                    "train_corpus": best_cell["train_corpus"],
                    "horizons": horizons,
                },
                "architecture": {
                    "family": family,
                    "variant_label": best_cell["variant_label"],
                    "kwargs": best_cell["arch_kwargs"],
                },
                "output": {
                    "formulation": best_cell["formulation"],
                    "quantiles": best_cell["quantiles"],
                },
                "training": base_training_kwargs(family),
                "device": "cuda",
            })
    return cfgs


def run_s06(scoreboard: Path, dry_run: bool = False) -> None:
    df = read_scoreboard(scoreboard)
    s04_best = downselect_s04(df)
    best_cell = downselect_s05(df, s04_best)
    cfgs = generate_s06_configs(best_cell)
    out_dir = CONFIG_ROOT / "S06_neural"
    write_configs(cfgs, out_dir)
    logger.info(
        "S06_neural: %d configs (%d horizon sets × %d seeds on %s/%s %s %s %s)",
        len(cfgs), len(S06_HORIZON_SETS), len(SEEDS),
        best_cell["family"], best_cell["variant_label"],
        best_cell["feature_variant"], best_cell["train_corpus"],
        best_cell["formulation"],
    )
    if not dry_run:
        rc = run_matrix_session("S06_neural", out_dir, scoreboard)
        logger.info("S06_neural run_matrix rc=%d", rc)
        git_commit("feat(predictor): S06_neural horizon-set sweep results")


# ----------------------------------------------------------------- S07 neural (skipped)

def run_s07_skipped() -> None:
    logger.info(
        "S07_neural SKIPPED: ema_post and temporal_loss smoothing require "
        "changes to train_one.py training loop — not yet implemented."
    )


# ----------------------------------------------------------------- S08 neural

def run_s08(scoreboard: Path) -> None:
    """Report top-N from neural scoreboard by backtest_pnl_k5_7m; write summary."""
    df = read_scoreboard(scoreboard)
    if df.empty or "backtest_pnl_k5_7m" not in df.columns:
        logger.warning("S08_neural: scoreboard empty or missing backtest columns")
        return

    prod = df[df["session"].isin(["S04_neural", "S05_neural", "S06_neural"])].copy()
    prod = prod.dropna(subset=["backtest_pnl_k5_7m"])
    if prod.empty:
        logger.warning("S08_neural: no rows with backtest_pnl_k5_7m")
        return

    # Also report by mean_mae for completeness alongside dir signal.
    prod["mean_mae"] = _mean_mae(prod)
    top3_backtest = prod.sort_values("backtest_pnl_k5_7m", ascending=False).head(3)
    top3_mae = prod.sort_values("mean_mae").head(3)

    summary_cols = [
        "experiment_id", "session", "architecture", "variant_label",
        "feature_variant", "train_corpus", "output_formulation", "horizons", "seed",
        "mean_mae", "mae_7m", "dir_acc_k5_7m", "dir_fires_k5_7m",
        "backtest_pnl_k5_7m", "backtest_winrate_k5_7m",
    ]
    summary_cols = [c for c in summary_cols if c in prod.columns]

    logger.info("S08_neural top-3 by backtest_pnl_k5_7m:\n%s",
                top3_backtest[summary_cols].to_string())
    logger.info("S08_neural top-3 by mean_mae:\n%s",
                top3_mae[summary_cols].to_string())

    any_positive = (prod["backtest_pnl_k5_7m"] > 0).any()
    logger.info(
        "S08_neural acceptance criterion (any positive backtest_pnl_k5_7m): %s",
        any_positive,
    )
    if not any_positive:
        logger.warning(
            "S08_neural WARNING: no candidate with positive k=5 backtest P&L. "
            "Review before proceeding to S09."
        )

    summary_path = REPO_ROOT / "registry" / "backtest_summary_neural.csv"
    top3_backtest[summary_cols].to_csv(summary_path, index=False)
    logger.info("backtest_summary_neural.csv written to %s", summary_path)
    logger.info(
        "S08_neural NOTE: k∈{3,10,20} sweeps deferred — "
        "require re-running inference with saved model weights."
    )
    git_commit("feat(predictor): S08_neural backtest summary")


# ----------------------------------------------------------------- main

def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.handlers.clear()
    root.addHandler(sh)
    root.addHandler(fh)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Neural-lineage predictor orchestrator S04_neural→S08_neural. "
                    "Downselects by mean MAE (not dir_acc). "
                    "Scoreboard: predictor_scoreboard_neural.csv.",
    )
    p.add_argument("--scoreboard", default=str(SCOREBOARD_DEFAULT))
    p.add_argument(
        "--start-from", default="S04_neural", choices=SESSION_ORDER,
        help="skip earlier sessions (assumes scoreboard already has their rows)",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="generate configs only; do not train or commit")
    args = p.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_path = LOGS_DIR / f"run_all_neural_{ts}.log"
    setup_logging(log_path)

    logger.info("=== run_all_sessions_neural.py start ===")
    logger.info("LINEAGE: neural (conv1d/k3 + lstm/tw32)")
    logger.info("DOWNSELECT CRITERION: mean_mae (lower = better) — NOT dir_acc")
    logger.info("log: %s", log_path)
    logger.info("scoreboard: %s", args.scoreboard)
    logger.info("start_from: %s  dry_run: %s", args.start_from, args.dry_run)

    scoreboard = Path(args.scoreboard)
    skip = SESSION_ORDER.index(args.start_from)

    t_start = time.time()
    try:
        if skip <= 0:
            logger.info("=== S04_neural: feature-variant sweep ===")
            run_s04(scoreboard, dry_run=args.dry_run)

        if skip <= 1:
            logger.info("=== S05_neural: output-formulation sweep ===")
            run_s05(scoreboard, dry_run=args.dry_run)

        if skip <= 2:
            logger.info("=== S06_neural: horizon-set sweep ===")
            run_s06(scoreboard, dry_run=args.dry_run)

        if skip <= 3:
            logger.info("=== S07_neural: smoothing sweep ===")
            run_s07_skipped()

        if skip <= 4:
            logger.info("=== S08_neural: backtest ===")
            if not args.dry_run:
                run_s08(scoreboard)
            else:
                logger.info("S08_neural skipped in dry-run mode.")

    except Exception:
        logger.exception("Neural orchestrator fatal error")
        return 1

    elapsed = time.time() - t_start
    logger.info("=== S04_neural–S08_neural complete in %.0fs ===", elapsed)
    logger.info(
        "STOP — S09 requires operator sign-off before running. "
        "Review registry/backtest_summary_neural.csv, pick top-3 candidates, "
        "then run S09 manually."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
