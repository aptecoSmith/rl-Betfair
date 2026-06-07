"""scripts/predictor/run_all_sessions.py - autonomous orchestrator S04→S08.

Runs sessions sequentially; each session generates configs, calls run_matrix,
commits results to git, then stops before S09 (operator sign-off required).

Run:
    python scripts/predictor/run_all_sessions.py
    python scripts/predictor/run_all_sessions.py --start-from S05   # resume
    python scripts/predictor/run_all_sessions.py --dry-run           # gen configs only
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

SCOREBOARD_DEFAULT = REPO_ROOT / "registry" / "predictor_scoreboard.csv"
CONFIG_ROOT = REPO_ROOT / "configs" / "predictor"
LOGS_DIR = REPO_ROOT / "logs"

SEEDS: tuple[int, ...] = (0, 1, 2)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- shared

def base_training_kwargs(family: str) -> dict:
    """Mirror generate_s03_configs.py — keep in sync."""
    if family == "gbm":
        return {
            "batch_size": 1024,
            "learning_rate": 0.05,
            "max_epochs": 1,
            "early_stopping_patience": 1,
        }
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


def run_matrix_session(session: str, config_dir: Path, scoreboard: Path) -> int:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "predictor" / "run_matrix.py"),
        "--session", session,
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


# ----------------------------------------------------------------- S04

# Per master_todo.md S04: V1/V2 train only on the large corpus (mask_29d);
# V3-V5 also run on tvl_required_10d so the TVL-gated features have a fair shot.
S04_CORPORA: dict[str, list[str]] = {
    "V1": ["tvl_mask_29d"],
    "V2": ["tvl_mask_29d"],
    "V3": ["tvl_required_10d", "tvl_mask_29d"],
    "V4": ["tvl_required_10d", "tvl_mask_29d"],
    "V5": ["tvl_required_10d", "tvl_mask_29d"],
}


def downselect_s03(df: pd.DataFrame) -> list[dict]:
    """Top-2 (architecture, variant_label) cells by median mean_mae across S03."""
    s03 = df[df["session"] == "S03"].copy()
    if s03.empty:
        raise RuntimeError("No S03 rows in scoreboard — S03 must be complete before S04.")
    s03["mean_mae"] = s03[["mae_3m", "mae_7m", "mae_15m"]].mean(axis=1)
    top2 = (
        s03.groupby(["architecture", "variant_label"])["mean_mae"]
        .median()
        .sort_values()
        .head(2)
    )
    logger.info("S03 top-2 cells (median mean_mae):\n%s", top2.to_string())
    cells = []
    for arch, variant in top2.index:
        row = s03[
            (s03["architecture"] == arch) & (s03["variant_label"] == variant)
        ].iloc[0]
        cells.append({
            "family": arch,
            "variant_label": variant,
            "arch_kwargs": json.loads(row["arch_kwargs"]),
        })
    return cells


def generate_s04_configs(
    cells: list[dict],
    horizons: list[str] | None = None,
) -> list[dict]:
    if horizons is None:
        horizons = ["3m", "7m", "15m"]
    cfgs = []
    for cell in cells:
        family = cell["family"]
        for fv, corpora in S04_CORPORA.items():
            for corpus in corpora:
                for seed in SEEDS:
                    cfgs.append({
                        "session": "S04",
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
                        "device": "cpu" if family == "gbm" else "cuda",
                    })
    return cfgs


def run_s04(scoreboard: Path, dry_run: bool = False) -> None:
    df = read_scoreboard(scoreboard)
    cells = downselect_s03(df)
    cfgs = generate_s04_configs(cells)
    out_dir = CONFIG_ROOT / "S04"
    write_configs(cfgs, out_dir)
    logger.info(
        "S04: %d configs (%d cells × 5 variants × %d corpora avg × %d seeds)",
        len(cfgs), len(cells), 8 // len(cells), len(SEEDS),
    )
    if not dry_run:
        rc = run_matrix_session("S04", out_dir, scoreboard)
        logger.info("S04 run_matrix rc=%d", rc)
        git_commit("feat(predictor): S04 feature-variant sweep results")


# ----------------------------------------------------------------- S05

def downselect_s04(df: pd.DataFrame) -> dict:
    """Best (arch, variant, fv, corpus) cell from S04 by median dir_acc_k5_7m."""
    s04 = df[df["session"] == "S04"].copy()
    if s04.empty:
        raise RuntimeError("No S04 rows — run S04 first.")

    if "dir_acc_k5_7m" in s04.columns and s04["dir_acc_k5_7m"].notna().any():
        metric, ascending = "dir_acc_k5_7m", False
    else:
        logger.warning("dir_acc_k5_7m missing — falling back to mae_7m for S04 downselect")
        metric, ascending = "mae_7m", True

    grp = (
        s04.dropna(subset=[metric])
        .groupby(["architecture", "variant_label", "feature_variant", "train_corpus"])[metric]
        .median()
    )
    best_idx = grp.idxmin() if ascending else grp.idxmax()
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
    logger.info("S04 best cell: %s / %s / %s (by %s=%.4f)",
                arch, variant, fv, metric, grp[best_idx])
    return result


def generate_s05_configs(
    best_cell: dict,
    horizons: list[str] | None = None,
) -> list[dict]:
    """pinball5 on best S04 cell. gaussian/student_t/classification deferred
    — they require output-head changes in models.py + train_one.py."""
    if horizons is None:
        horizons = ["3m", "7m", "15m"]
    family = best_cell["family"]
    cfgs = []
    for seed in SEEDS:
        cfgs.append({
            "session": "S05",
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
            "device": "cpu" if family == "gbm" else "cuda",
        })
    return cfgs


def run_s05(scoreboard: Path, dry_run: bool = False) -> None:
    df = read_scoreboard(scoreboard)
    best_cell = downselect_s04(df)
    cfgs = generate_s05_configs(best_cell)
    out_dir = CONFIG_ROOT / "S05"
    write_configs(cfgs, out_dir)
    logger.info(
        "S05: %d configs (pinball5 on %s/%s %s %s). "
        "NOTE: gaussian/student_t/classification deferred — require train_one.py changes.",
        len(cfgs),
        best_cell["family"], best_cell["variant_label"],
        best_cell["feature_variant"], best_cell["train_corpus"],
    )
    if not dry_run:
        rc = run_matrix_session("S05", out_dir, scoreboard)
        logger.info("S05 run_matrix rc=%d", rc)
        git_commit("feat(predictor): S05 output-formulation sweep (pinball5)")


# ----------------------------------------------------------------- S06

# S06 horizon-set sweep. The baseline (3m_7m_15m) is re-run within S06 so
# all rows are session=S06 and directly comparable.
S06_HORIZON_SETS: dict[str, list[str]] = {
    "3m_7m_15m":  ["3m", "7m", "15m"],
    "1m_3m_7m":   ["1m", "3m", "7m"],
    "1m_7m_15m":  ["1m", "7m", "15m"],
    "7m_only":    ["7m"],
    "1m_only":    ["1m"],
    "3m_only":    ["3m"],
    "15m_only":   ["15m"],
}


def downselect_s05(df: pd.DataFrame, s04_best: dict) -> dict:
    """Compare S04 (pinball3) vs S05 (pinball5) on same cell; pick by dir_acc_k5_7m.
    Returns extended cell dict with 'formulation' and 'quantiles' added."""
    arch = s04_best["family"]
    variant = s04_best["variant_label"]
    fv = s04_best["feature_variant"]
    corpus = s04_best["train_corpus"]

    candidates = df[
        (df["architecture"] == arch)
        & (df["variant_label"] == variant)
        & (df["feature_variant"] == fv)
        & (df["train_corpus"] == corpus)
        & (df["session"].isin(["S04", "S05"]))
    ].copy()

    default = {**s04_best, "formulation": "pinball3", "quantiles": [0.1, 0.5, 0.9]}

    if candidates.empty or "dir_acc_k5_7m" not in candidates.columns:
        logger.warning("S05 downselect: no candidates — defaulting to pinball3")
        return default

    by_form = (
        candidates.dropna(subset=["dir_acc_k5_7m"])
        .groupby("output_formulation")["dir_acc_k5_7m"]
        .median()
    )
    if by_form.empty:
        logger.warning("S05 downselect: dir_acc missing — defaulting to pinball3")
        return default

    logger.info("S04/S05 formulation comparison:\n%s", by_form.to_string())
    best_form = by_form.idxmax()
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9] if best_form == "pinball5" else [0.1, 0.5, 0.9]
    logger.info("S05 winner: %s (dir_acc_k5_7m=%.4f)", best_form, by_form[best_form])
    return {**s04_best, "formulation": best_form, "quantiles": quantiles}


def generate_s06_configs(best_cell: dict) -> list[dict]:
    family = best_cell["family"]
    cfgs = []
    for hset_name, horizons in S06_HORIZON_SETS.items():
        for seed in SEEDS:
            cfgs.append({
                "session": "S06",
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
                "device": "cpu" if family == "gbm" else "cuda",
            })
    return cfgs


def run_s06(scoreboard: Path, dry_run: bool = False) -> None:
    df = read_scoreboard(scoreboard)
    s04_best = downselect_s04(df)
    best_cell = downselect_s05(df, s04_best)
    cfgs = generate_s06_configs(best_cell)
    out_dir = CONFIG_ROOT / "S06"
    write_configs(cfgs, out_dir)
    logger.info(
        "S06: %d configs (%d horizon sets × %d seeds on %s/%s %s %s %s)",
        len(cfgs), len(S06_HORIZON_SETS), len(SEEDS),
        best_cell["family"], best_cell["variant_label"],
        best_cell["feature_variant"], best_cell["train_corpus"],
        best_cell["formulation"],
    )
    if not dry_run:
        rc = run_matrix_session("S06", out_dir, scoreboard)
        logger.info("S06 run_matrix rc=%d", rc)
        git_commit("feat(predictor): S06 horizon-set sweep results")


# ----------------------------------------------------------------- S07 (skipped)

def run_s07_skipped() -> None:
    logger.info(
        "S07 SKIPPED: ema_post and temporal_loss smoothing require changes to "
        "train_one.py training loop — not yet implemented. "
        "Implement in a follow-on session, then rerun with --start-from S07."
    )


# ----------------------------------------------------------------- S08

def run_s08(scoreboard: Path) -> None:
    """Report top-N by backtest_pnl_k5_7m; write backtest_summary.csv."""
    df = read_scoreboard(scoreboard)
    if df.empty or "backtest_pnl_k5_7m" not in df.columns:
        logger.warning("S08: scoreboard empty or missing backtest columns — nothing to report")
        return

    prod = df[df["session"].isin(["S03", "S04", "S05", "S06"])].copy()
    prod = prod.dropna(subset=["backtest_pnl_k5_7m"])
    if prod.empty:
        logger.warning("S08: no rows with backtest_pnl_k5_7m in S03-S06")
        return

    top3 = prod.sort_values("backtest_pnl_k5_7m", ascending=False).head(3)
    summary_cols = [
        "experiment_id", "session", "architecture", "variant_label",
        "feature_variant", "train_corpus", "output_formulation", "horizons", "seed",
        "dir_acc_k5_7m", "dir_fires_k5_7m",
        "backtest_pnl_k5_7m", "backtest_winrate_k5_7m", "mae_7m",
    ]
    summary_cols = [c for c in summary_cols if c in top3.columns]
    summary = top3[summary_cols].reset_index(drop=True)

    logger.info("S08 top-3 candidates (backtest_pnl_k5_7m):\n%s", summary.to_string())

    any_positive = (prod["backtest_pnl_k5_7m"] > 0).any()
    logger.info(
        "S08 acceptance criterion (any positive backtest_pnl_k5_7m): %s", any_positive,
    )
    if not any_positive:
        logger.warning(
            "S08 WARNING: no candidate has positive k=5 backtest P&L. "
            "Review S03-S06 rows before proceeding to S09."
        )

    summary_path = REPO_ROOT / "registry" / "backtest_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("backtest_summary.csv written to %s", summary_path)
    logger.info(
        "S08 NOTE: k∈{3,10,20} sweeps deferred — require re-running inference "
        "with different k values against saved model weights in registry/predictor/."
    )
    git_commit("feat(predictor): S08 backtest summary")


# ----------------------------------------------------------------- main

SESSION_ORDER = ["S04", "S05", "S06", "S07", "S08"]


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # stderr handler.
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.handlers.clear()
    root.addHandler(sh)
    root.addHandler(fh)


def main() -> int:
    p = argparse.ArgumentParser(description="Autonomous predictor orchestrator S04-S08.")
    p.add_argument("--scoreboard", default=str(SCOREBOARD_DEFAULT))
    p.add_argument(
        "--start-from", default="S04", choices=SESSION_ORDER,
        help="skip earlier sessions (assumes scoreboard already has their rows)",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="generate configs only; do not train or commit")
    args = p.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_path = LOGS_DIR / f"run_all_{ts}.log"
    setup_logging(log_path)

    logger.info("=== run_all_sessions.py start ===")
    logger.info("log: %s", log_path)
    logger.info("scoreboard: %s", args.scoreboard)
    logger.info("start_from: %s  dry_run: %s", args.start_from, args.dry_run)

    scoreboard = Path(args.scoreboard)
    skip = SESSION_ORDER.index(args.start_from)

    t_start = time.time()
    try:
        if skip <= 0:
            logger.info("=== S04: feature-variant sweep ===")
            run_s04(scoreboard, dry_run=args.dry_run)

        if skip <= 1:
            logger.info("=== S05: output-formulation sweep ===")
            run_s05(scoreboard, dry_run=args.dry_run)

        if skip <= 2:
            logger.info("=== S06: horizon-set sweep ===")
            run_s06(scoreboard, dry_run=args.dry_run)

        if skip <= 3:
            logger.info("=== S07: smoothing sweep ===")
            run_s07_skipped()

        if skip <= 4:
            logger.info("=== S08: backtest ===")
            if not args.dry_run:
                run_s08(scoreboard)
            else:
                logger.info("S08 skipped in dry-run mode.")

    except Exception:
        logger.exception("Orchestrator fatal error")
        return 1

    elapsed = time.time() - t_start
    logger.info("=== S04-S08 complete in %.0fs ===", elapsed)
    logger.info(
        "STOP — S09 requires operator sign-off before running. "
        "Review registry/backtest_summary.csv, pick top-3 candidates, "
        "then run S09 manually."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
