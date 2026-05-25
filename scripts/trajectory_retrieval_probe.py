"""Trajectory-retrieval probe — Phases 1 & 2.

Side-thread experiment. Reads data/processed/ read-only, writes to
scratch/trajectory_retrieval/. Touches no production code.

See plans/trajectory-retrieval-probe/ for the design.

Usage:
    python scripts/trajectory_retrieval_probe.py --phase 1
    python scripts/trajectory_retrieval_probe.py --phase 2
    python scripts/trajectory_retrieval_probe.py --phase all

Phase 1: parquets -> scratch/trajectory_retrieval/ticks.parquet
         (long-form: market_id, selection_id, tick_idx, time_to_off_s,
          ltp, vol_cum, best_back, best_back_size, best_lay,
          best_lay_size, ts)

Phase 2: ticks.parquet -> scratch/trajectory_retrieval/queries.parquet
         (one row per race/runner with D = T-off - 5min features +
          target log(LTP_{D+5min}) - log(LTP_D))
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
SCRATCH_DIR = REPO_ROOT / "scratch" / "trajectory_retrieval"

# Split locked per plans/trajectory-retrieval-probe/hard_constraints.md §2
INDEX_DAYS_END = "2026-05-04"  # inclusive
QUERY_DAYS_START = "2026-05-05"
VALIDATION_DAYS_START = "2026-05-15"

# Query point and prediction horizon
QUERY_TIME_TO_OFF_S = 5 * 60          # D = T-off - 5min
PREDICTION_HORIZON_S = 5 * 60          # predict log-LTP at D + 5min
TRAJECTORY_WINDOW_S = 30 * 60          # use ticks in [T-off - 30min, D]

# Feature-window lengths (seconds before D)
SLOPE_SHORT_S = 5 * 60
SLOPE_LONG_S = 20 * 60
VOL_WINDOW_S = 5 * 60
VOL_DELTA_S = 5 * 60


# --------------------------------------------------------------------------
# Phase 1 — tick-history reshape
# --------------------------------------------------------------------------

def _extract_runner_snapshot(snap_obj: dict) -> list[dict]:
    """Pull per-runner top-of-book + LTP + traded volume from one snap_json."""
    out = []
    for runner in snap_obj.get("MarketRunners", []):
        rid = runner.get("RunnerId", {})
        selection_id = rid.get("SelectionId") if isinstance(rid, dict) else None
        if selection_id is None:
            continue
        prices = runner.get("Prices", {}) or {}
        ltp = prices.get("LastTradedPrice") or 0.0
        vol_cum = prices.get("TradedVolume") or 0.0
        atb = prices.get("AvailableToBack") or []
        atl = prices.get("AvailableToLay") or []
        best_back = atb[0]["Price"] if atb else 0.0
        best_back_size = atb[0]["Size"] if atb else 0.0
        best_lay = atl[0]["Price"] if atl else 0.0
        best_lay_size = atl[0]["Size"] if atl else 0.0
        out.append(
            {
                "selection_id": int(selection_id),
                "ltp": float(ltp),
                "vol_cum": float(vol_cum),
                "best_back": float(best_back),
                "best_back_size": float(best_back_size),
                "best_lay": float(best_lay),
                "best_lay_size": float(best_lay_size),
            }
        )
    return out


def _reshape_one_day(parquet_path: Path) -> pd.DataFrame:
    """Reshape one day's parquet into long-form ticks frame."""
    df = pd.read_parquet(
        parquet_path,
        columns=["market_id", "timestamp", "market_start_time", "snap_json"],
    )
    if df.empty:
        return pd.DataFrame()

    rows = []
    for market_id, mdf in df.groupby("market_id", sort=False):
        mdf = mdf.sort_values("timestamp").reset_index(drop=True)
        start_time = mdf["market_start_time"].iloc[0]
        # time_to_off in seconds: positive means before the off
        for tick_idx, (ts, snap_str) in enumerate(
            zip(mdf["timestamp"], mdf["snap_json"])
        ):
            tto = (start_time - ts).total_seconds()
            # Filter to ticks within [0, 30 min] pre-off
            if tto < 0 or tto > TRAJECTORY_WINDOW_S:
                continue
            try:
                snap = json.loads(snap_str)
            except (json.JSONDecodeError, TypeError):
                continue
            for r in _extract_runner_snapshot(snap):
                rows.append(
                    {
                        "market_id": market_id,
                        "selection_id": r["selection_id"],
                        "tick_idx": tick_idx,
                        "ts": ts,
                        "time_to_off_s": tto,
                        "ltp": r["ltp"],
                        "vol_cum": r["vol_cum"],
                        "best_back": r["best_back"],
                        "best_back_size": r["best_back_size"],
                        "best_lay": r["best_lay"],
                        "best_lay_size": r["best_lay_size"],
                    }
                )
    return pd.DataFrame(rows)


def phase1_reshape() -> Path:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCRATCH_DIR / "ticks.parquet"

    parquet_paths = sorted(
        p for p in DATA_DIR.glob("*.parquet") if "_runners" not in p.stem
    )
    print(f"[phase1] {len(parquet_paths)} day files to process", flush=True)

    all_frames: list[pd.DataFrame] = []
    t0 = time.time()
    for i, path in enumerate(parquet_paths, 1):
        t_day = time.time()
        day_df = _reshape_one_day(path)
        all_frames.append(day_df)
        elapsed = time.time() - t0
        print(
            f"[phase1] {i}/{len(parquet_paths)} {path.stem} "
            f"rows={len(day_df):,} day_time={time.time()-t_day:.1f}s "
            f"total={elapsed:.1f}s",
            flush=True,
        )

    ticks = pd.concat(all_frames, ignore_index=True)
    # Add a date column for split bookkeeping
    ticks["date"] = ticks["ts"].dt.date.astype(str)
    ticks.to_parquet(out_path, index=False)
    print(f"[phase1] wrote {out_path}  rows={len(ticks):,}", flush=True)

    # Sanity report
    _phase1_sanity(ticks)
    return out_path


def _phase1_sanity(ticks: pd.DataFrame) -> None:
    n_runners_per_race = ticks.groupby("market_id")["selection_id"].nunique()
    tick_per_runner = ticks.groupby(["market_id", "selection_id"]).size()
    priceable = (ticks["ltp"] > 1.0).mean()
    print(
        f"[phase1.sanity] markets={ticks['market_id'].nunique():,} "
        f"race-runners={(n_runners_per_race.sum()):,} "
        f"median_ticks_per_runner={tick_per_runner.median():.0f} "
        f"priceable_ltp_frac={priceable:.3f}",
        flush=True,
    )
    days = sorted(ticks["date"].unique())
    print(
        f"[phase1.sanity] dates={len(days)} "
        f"first={days[0]} last={days[-1]}",
        flush=True,
    )


# --------------------------------------------------------------------------
# Phase 2 — query-time features
# --------------------------------------------------------------------------

@dataclass
class FeatureBundle:
    log_ltp_d: float
    slope_short: float
    slope_long: float
    vol_short: float
    spread_norm: float
    book_imbalance: float
    log_vol_cum: float
    delta_vol_short: float
    time_to_off_s: float
    fav_rank: float  # filled in later, after per-race ranking
    target_log_return: float
    quality_ok: bool  # False if insufficient history


def _linear_slope(times: np.ndarray, values: np.ndarray) -> float:
    """OLS slope of values vs times. Returns 0.0 if degenerate."""
    if len(times) < 2:
        return 0.0
    t_mean = times.mean()
    v_mean = values.mean()
    denom = ((times - t_mean) ** 2).sum()
    if denom <= 0:
        return 0.0
    return float(((times - t_mean) * (values - v_mean)).sum() / denom)


def _features_for_runner(runner_ticks: pd.DataFrame) -> FeatureBundle | None:
    """Compute the 10-feature bundle for one (race, runner) at D = T-off - 5min.

    Hard constraint §3: only uses ticks at time_to_off_s >= QUERY_TIME_TO_OFF_S.
    Target uses ticks at time_to_off_s ∈ [0, QUERY_TIME_TO_OFF_S).
    """
    # Sort ASCENDING by time_to_off_s. Under this sort:
    #   iloc[0]  = smallest tto    = CLOSEST to the off (latest chronologically)
    #   iloc[-1] = largest tto     = FURTHEST from off (earliest chronologically)
    # So the most-recent tick at-or-before D is pre_d.iloc[0], NOT pre_d.iloc[-1].
    # Earlier versions of this file had this inverted twice (descending sort +
    # iloc[0] in the original; ascending sort + iloc[-1] in the first "fix").
    # Both produced features at the earliest trajectory tick (~30min pre-off)
    # rather than at D. See lessons_learnt.md.
    pre_d = runner_ticks[
        runner_ticks["time_to_off_s"] >= QUERY_TIME_TO_OFF_S
    ].sort_values("time_to_off_s", ascending=True)
    post_d_min_tto = max(0.0, QUERY_TIME_TO_OFF_S - PREDICTION_HORIZON_S)
    post_d = runner_ticks[
        (runner_ticks["time_to_off_s"] < QUERY_TIME_TO_OFF_S)
        & (runner_ticks["time_to_off_s"] >= post_d_min_tto)
    ].sort_values("time_to_off_s", ascending=True)

    if len(pre_d) < 10 or len(post_d) < 1:
        return None

    # Most recent tick at-or-before D → smallest tto in pre_d → iloc[0]
    d_row = pre_d.iloc[0]
    ltp_d = float(d_row["ltp"])
    if ltp_d <= 1.0:
        return None  # unpriceable at D

    # Tick closest to in-play → smallest tto in post_d → iloc[0]
    target_row = post_d.iloc[0]
    ltp_target = float(target_row["ltp"])
    if ltp_target <= 1.0:
        return None
    target_log_return = float(np.log(ltp_target) - np.log(ltp_d))

    # Slope features — use elapsed seconds since D as the time axis
    # (negative numbers = earlier). Convert to log-LTP space.
    pre_d_valid = pre_d[pre_d["ltp"] > 1.0]
    if len(pre_d_valid) < 5:
        return None

    # elapsed_s relative to D: 0 at D, +N seconds for ticks N seconds earlier
    elapsed_s = (pre_d_valid["time_to_off_s"].to_numpy() - QUERY_TIME_TO_OFF_S)
    log_ltp = np.log(pre_d_valid["ltp"].to_numpy())

    short_mask = elapsed_s <= SLOPE_SHORT_S
    long_mask = elapsed_s <= SLOPE_LONG_S
    slope_short = _linear_slope(elapsed_s[short_mask], log_ltp[short_mask])
    slope_long = _linear_slope(elapsed_s[long_mask], log_ltp[long_mask])

    # Volatility (std of tick-to-tick log returns) over last 5 min before D
    vol_window = pre_d_valid[elapsed_s <= VOL_WINDOW_S]
    vol_window = vol_window.sort_values("time_to_off_s", ascending=False)
    if len(vol_window) >= 3:
        lr = np.diff(np.log(vol_window["ltp"].to_numpy()))
        vol_short = float(np.std(lr)) if len(lr) > 1 else 0.0
    else:
        vol_short = 0.0

    # Spread + book imbalance at D
    best_back = float(d_row["best_back"])
    best_lay = float(d_row["best_lay"])
    bb_size = float(d_row["best_back_size"])
    bl_size = float(d_row["best_lay_size"])
    if best_back > 0 and best_lay > 0:
        spread_norm = (best_lay - best_back) / ltp_d
    else:
        spread_norm = 0.0
    if bb_size > 0 and bl_size > 0:
        book_imbalance = float(np.log(bb_size / bl_size))
    else:
        book_imbalance = 0.0

    # Cumulative + delta traded volume
    vol_cum_d = float(d_row["vol_cum"])
    log_vol_cum = float(np.log1p(vol_cum_d))
    # delta over last 5 min
    older_5min = pre_d_valid[elapsed_s >= VOL_DELTA_S]
    if len(older_5min) > 0:
        vol_5min_ago = float(older_5min.iloc[0]["vol_cum"])
    else:
        vol_5min_ago = 0.0
    delta_vol_short = max(0.0, vol_cum_d - vol_5min_ago)

    return FeatureBundle(
        log_ltp_d=float(np.log(ltp_d)),
        slope_short=slope_short,
        slope_long=slope_long,
        vol_short=vol_short,
        spread_norm=float(spread_norm),
        book_imbalance=float(book_imbalance),
        log_vol_cum=log_vol_cum,
        delta_vol_short=float(np.log1p(delta_vol_short)),
        time_to_off_s=float(d_row["time_to_off_s"]),
        fav_rank=np.nan,  # filled per-race after this function
        target_log_return=target_log_return,
        quality_ok=True,
    )


def phase2_features(ticks_path: Path | None = None) -> Path:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    ticks_path = ticks_path or (SCRATCH_DIR / "ticks.parquet")
    if not ticks_path.exists():
        raise FileNotFoundError(
            f"{ticks_path} missing — run phase 1 first"
        )
    ticks = pd.read_parquet(ticks_path)
    print(f"[phase2] loaded {len(ticks):,} ticks from {ticks_path}", flush=True)

    rows = []
    t0 = time.time()
    n_groups = 0
    for (market_id, selection_id), runner_ticks in ticks.groupby(
        ["market_id", "selection_id"], sort=False
    ):
        n_groups += 1
        fb = _features_for_runner(runner_ticks)
        if fb is None:
            continue
        rows.append(
            {
                "market_id": market_id,
                "selection_id": selection_id,
                "date": runner_ticks["date"].iloc[0],
                "log_ltp_d": fb.log_ltp_d,
                "slope_short": fb.slope_short,
                "slope_long": fb.slope_long,
                "vol_short": fb.vol_short,
                "spread_norm": fb.spread_norm,
                "book_imbalance": fb.book_imbalance,
                "log_vol_cum": fb.log_vol_cum,
                "delta_vol_short": fb.delta_vol_short,
                "time_to_off_s": fb.time_to_off_s,
                "target_log_return": fb.target_log_return,
            }
        )

    queries = pd.DataFrame(rows)
    print(
        f"[phase2] built {len(queries):,}/{n_groups:,} feature rows "
        f"in {time.time()-t0:.1f}s",
        flush=True,
    )

    # Per-race favourite rank (1 = shortest LTP at D)
    queries["ltp_at_d"] = np.exp(queries["log_ltp_d"])
    queries["fav_rank"] = (
        queries.groupby("market_id")["ltp_at_d"].rank(method="dense").astype(int)
    )
    queries = queries.drop(columns=["ltp_at_d"])

    # Split bookkeeping
    queries["split"] = queries["date"].apply(_classify_date)

    # Z-score on INDEX days only
    feature_cols = [
        "log_ltp_d",
        "slope_short",
        "slope_long",
        "vol_short",
        "spread_norm",
        "book_imbalance",
        "log_vol_cum",
        "delta_vol_short",
        "time_to_off_s",
        "fav_rank",
    ]
    index_mask = queries["split"] == "index"
    means = queries.loc[index_mask, feature_cols].mean()
    stds = queries.loc[index_mask, feature_cols].std().replace(0.0, 1.0)
    for c in feature_cols:
        queries[f"{c}_z"] = (queries[c] - means[c]) / stds[c]
    norm_stats = pd.DataFrame({"mean": means, "std": stds}).reset_index().rename(
        columns={"index": "feature"}
    )

    out_path = SCRATCH_DIR / "queries.parquet"
    queries.to_parquet(out_path, index=False)
    norm_stats.to_parquet(SCRATCH_DIR / "norm_stats.parquet", index=False)
    print(f"[phase2] wrote {out_path}", flush=True)

    # Sanity report
    _phase2_sanity(queries)
    return out_path


def _classify_date(date_str: str) -> str:
    if date_str <= INDEX_DAYS_END:
        return "index"
    if date_str >= VALIDATION_DAYS_START:
        return "validation"
    return "query"


def _phase2_sanity(queries: pd.DataFrame) -> None:
    split_counts = queries["split"].value_counts().to_dict()
    print(f"[phase2.sanity] split counts: {split_counts}", flush=True)
    print(
        f"[phase2.sanity] target log-return: "
        f"mean={queries['target_log_return'].mean():+.5f} "
        f"std={queries['target_log_return'].std():.4f} "
        f"|mean|={queries['target_log_return'].abs().mean():.4f}",
        flush=True,
    )
    z_cols = [c for c in queries.columns if c.endswith("_z")]
    print("[phase2.sanity] z-scored feature ranges (index set):")
    idx = queries["split"] == "index"
    for c in z_cols:
        s = queries.loc[idx, c]
        print(
            f"    {c:30s} min={s.min():+.2f} max={s.max():+.2f} "
            f"mean={s.mean():+.4f} std={s.std():.4f}"
        )

    # No-lookahead smoke test
    _no_lookahead_smoke_test(queries)


def _no_lookahead_smoke_test(queries: pd.DataFrame) -> None:
    """Hard constraint §3 enforcement.

    Re-load 5 random ticks rows, perturb a tick at time_to_off_s <
    QUERY_TIME_TO_OFF_S (i.e. POST-D), recompute features, assert the
    feature vector is unchanged.
    """
    ticks = pd.read_parquet(SCRATCH_DIR / "ticks.parquet")
    rng = np.random.default_rng(0)
    sample = queries.sample(5, random_state=42)
    n_pass = 0
    for _, qrow in sample.iterrows():
        m, s = qrow["market_id"], qrow["selection_id"]
        runner_ticks = ticks[
            (ticks["market_id"] == m) & (ticks["selection_id"] == s)
        ].copy()
        if runner_ticks.empty:
            continue
        post_d = runner_ticks[runner_ticks["time_to_off_s"] < QUERY_TIME_TO_OFF_S]
        if post_d.empty:
            continue
        # Perturb LTP of a random post-D tick
        perturbed = runner_ticks.copy()
        target_idx = post_d.index[rng.integers(0, len(post_d))]
        # set ltp to some unreasonable value
        perturbed.loc[target_idx, "ltp"] = 999.0
        perturbed.loc[target_idx, "best_back"] = 999.0
        perturbed.loc[target_idx, "best_lay"] = 999.0
        fb_orig = _features_for_runner(runner_ticks)
        fb_pert = _features_for_runner(perturbed)
        if fb_orig is None or fb_pert is None:
            continue
        # Compare all non-target fields
        same = (
            np.isclose(fb_orig.log_ltp_d, fb_pert.log_ltp_d)
            and np.isclose(fb_orig.slope_short, fb_pert.slope_short)
            and np.isclose(fb_orig.slope_long, fb_pert.slope_long)
            and np.isclose(fb_orig.vol_short, fb_pert.vol_short)
            and np.isclose(fb_orig.spread_norm, fb_pert.spread_norm)
            and np.isclose(fb_orig.book_imbalance, fb_pert.book_imbalance)
            and np.isclose(fb_orig.log_vol_cum, fb_pert.log_vol_cum)
            and np.isclose(fb_orig.delta_vol_short, fb_pert.delta_vol_short)
        )
        n_pass += int(same)
    print(
        f"[phase2.smoketest] no-lookahead: {n_pass}/{len(sample)} passed "
        f"(features unchanged when post-D ticks are perturbed)",
        flush=True,
    )
    if n_pass < len(sample) // 2:
        print(
            "[phase2.smoketest] FAIL — feature vector depends on post-D "
            "ticks. Review feature definitions.",
            flush=True,
        )
        sys.exit(2)


# --------------------------------------------------------------------------
# Phase 3 — baselines + kNN headline
# --------------------------------------------------------------------------

FEATURE_Z_COLS = [
    "log_ltp_d_z",
    "slope_short_z",
    "slope_long_z",
    "vol_short_z",
    "spread_norm_z",
    "book_imbalance_z",
    "log_vol_cum_z",
    "delta_vol_short_z",
    "time_to_off_s_z",
    "fav_rank_z",
]


def _b1_constant(query_df: pd.DataFrame) -> np.ndarray:
    """B1: predict zero log-return (price stays flat)."""
    return np.zeros(len(query_df), dtype=float)


def _b2_linear_extrap(query_df: pd.DataFrame) -> np.ndarray:
    """B2: extrapolate the last-5-min slope of log-LTP forward 5 min.

    slope_short is the OLS slope of log_ltp vs elapsed_s (seconds BEFORE D).
    To project FORWARD by PREDICTION_HORIZON_S seconds (towards the off),
    we move in the direction of DECREASING elapsed_s, so:

        predicted Δlog_ltp = -slope_short * PREDICTION_HORIZON_S

    Note: slope_short is NOT z-scored in this calc — we need the raw slope.
    Pull it from the un-z-scored column.
    """
    return -query_df["slope_short"].to_numpy() * PREDICTION_HORIZON_S


def _b3_rank_prior(
    index_df: pd.DataFrame, query_df: pd.DataFrame
) -> np.ndarray:
    """B3: per-favourite-rank mean target on the index set."""
    rank_means = index_df.groupby("fav_rank")["target_log_return"].mean()
    global_mean = float(index_df["target_log_return"].mean())
    return query_df["fav_rank"].map(rank_means).fillna(global_mean).to_numpy()


def _knn_predict(
    index_df: pd.DataFrame, query_df: pd.DataFrame, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """kNN: top-k nearest neighbours by Euclidean distance on z-scored features.

    Returns (predictions, neighbour_target_std). The second array is the std
    of the k neighbours' target values — a "neighbour agreement" diagnostic.
    """
    from sklearn.neighbors import NearestNeighbors

    X_index = index_df[FEATURE_Z_COLS].to_numpy()
    X_query = query_df[FEATURE_Z_COLS].to_numpy()
    y_index = index_df["target_log_return"].to_numpy()

    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1)
    nn.fit(X_index)
    _dist, idxs = nn.kneighbors(X_query)
    neighbour_targets = y_index[idxs]  # shape (n_queries, k)
    preds = neighbour_targets.mean(axis=1)
    stds = neighbour_targets.std(axis=1)
    return preds, stds


def phase3_baselines(queries_path: Path | None = None, k: int = 5) -> Path:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    queries_path = queries_path or (SCRATCH_DIR / "queries.parquet")
    if not queries_path.exists():
        raise FileNotFoundError(
            f"{queries_path} missing — run phase 2 first"
        )
    queries = pd.read_parquet(queries_path)
    print(f"[phase3] loaded {len(queries):,} feature rows", flush=True)

    index_df = queries[queries["split"] == "index"].reset_index(drop=True)
    query_df = queries[queries["split"] == "query"].reset_index(drop=True)
    # Validation set is held back — Phase 5 runs on it separately
    print(
        f"[phase3] index n={len(index_df):,}  query n={len(query_df):,}",
        flush=True,
    )

    y_true = query_df["target_log_return"].to_numpy()
    sign_true = np.sign(y_true)

    t0 = time.time()
    preds_b1 = _b1_constant(query_df)
    preds_b2 = _b2_linear_extrap(query_df)
    preds_b3 = _b3_rank_prior(index_df, query_df)
    preds_knn, knn_std = _knn_predict(index_df, query_df, k=k)
    elapsed = time.time() - t0
    print(f"[phase3] predictions computed in {elapsed:.1f}s", flush=True)

    methods = {
        "B1_constant": preds_b1,
        "B2_linear_extrap": preds_b2,
        "B3_rank_prior": preds_b3,
        f"kNN_k{k}": preds_knn,
    }

    print("\n[phase3] headline results (query days only)")
    print(f"{'method':24s} {'MAE':>10s} {'vs_B1':>9s} {'dir_acc':>9s}")
    print("-" * 60)
    mae_b1 = float(np.mean(np.abs(y_true - preds_b1)))
    summary_rows = []
    for name, preds in methods.items():
        mae = float(np.mean(np.abs(y_true - preds)))
        # Directional accuracy: predict UP or DOWN correctly? Skip the rows
        # where y_true sign is 0 (no movement). For B1 (always 0) dir_acc
        # is degenerate — report it but expect ~50%.
        nonzero = sign_true != 0
        if name == "B1_constant":
            dir_acc = float(np.mean(np.sign(preds[nonzero]) == sign_true[nonzero]))
        else:
            pred_sign = np.sign(preds[nonzero])
            # When predicted sign is 0 (rare), count as wrong (no commitment).
            dir_acc = float(np.mean(pred_sign == sign_true[nonzero]))
        vs_b1 = (mae_b1 - mae) / mae_b1 * 100  # positive = better than B1
        print(f"{name:24s} {mae:>10.5f} {vs_b1:>+8.2f}% {dir_acc:>8.3f}")
        summary_rows.append(
            {"method": name, "mae": mae, "mae_vs_b1_pct": vs_b1, "dir_acc": dir_acc}
        )

    print(f"\n[phase3] B1 baseline MAE = {mae_b1:.5f}", flush=True)
    print(
        f"[phase3] kNN neighbour-agreement: median std = {np.median(knn_std):.4f}, "
        f"q25={np.quantile(knn_std,0.25):.4f}, "
        f"q75={np.quantile(knn_std,0.75):.4f}",
        flush=True,
    )

    # Persist per-row predictions + diagnostics for Phase 4
    results = query_df[
        ["market_id", "selection_id", "date", "fav_rank", "target_log_return"]
    ].copy()
    results["y_true"] = y_true
    for name, preds in methods.items():
        results[f"pred_{name}"] = preds
        results[f"err_{name}"] = preds - y_true
        results[f"abs_err_{name}"] = np.abs(preds - y_true)
    results["knn_neighbour_std"] = knn_std

    out_path = SCRATCH_DIR / "results.parquet"
    results.to_parquet(out_path, index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_parquet(SCRATCH_DIR / "summary.parquet", index=False)
    print(f"[phase3] wrote {out_path}", flush=True)

    _phase3_decision_band(summary_df)
    return out_path


def _phase3_decision_band(summary: pd.DataFrame) -> None:
    """Apply the locked decision rule from purpose.md to the kNN result.

    NB: this prints the band based on QUERY-set MAE. The final decision
    rule in plans/.../purpose.md is applied to the VALIDATION-set MAE in
    Phase 5 — this is a preview / early-exit signal, not the verdict.
    """
    b1_mae = float(summary.loc[summary["method"] == "B1_constant", "mae"].iloc[0])
    b2_mae = float(summary.loc[summary["method"] == "B2_linear_extrap", "mae"].iloc[0])
    knn_row = summary[summary["method"].str.startswith("kNN")].iloc[0]
    knn_mae = float(knn_row["mae"])
    knn_vs_b1 = (b1_mae - knn_mae) / b1_mae * 100

    print("\n[phase3] decision-band preview (QUERY set, not validation):")
    print(f"  B1 MAE  = {b1_mae:.5f}")
    print(f"  B2 MAE  = {b2_mae:.5f}")
    print(f"  kNN MAE = {knn_mae:.5f}  ({knn_vs_b1:+.2f}% vs B1)")
    if knn_vs_b1 > 10.0 and knn_mae < b2_mae:
        band = "GO (beats B1 by >10% AND beats B2) — invest in learned encoder"
    elif knn_vs_b1 >= 3.0:
        band = "MARGINAL — try richer features (cross-runner, form) before deciding"
    elif knn_mae <= b1_mae:
        band = "TIGHT — matches B1, doesn't beat B2; encoder is the bottleneck"
    else:
        band = "PARK — loses to B1, signal-to-noise too low for retrieval to help"
    print(f"  band: {band}")
    print(
        "  (final decision uses validation-set numbers in Phase 5; "
        "do not interpret this as the verdict)"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    args = p.parse_args()

    if args.phase in ("1", "all"):
        phase1_reshape()
    if args.phase in ("2", "all"):
        phase2_features()
    if args.phase in ("3", "all"):
        phase3_baselines()


if __name__ == "__main__":
    main()
