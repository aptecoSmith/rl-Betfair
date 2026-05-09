"""scripts/predictor/datasets.py - dataset loading + torch Dataset classes.

One source of truth for which feature columns each variant V1..V5
consumes, and how the per-tick parquet rows assemble into either
flat tabular tensors (MLP, GBM) or sequence windows (LSTM,
Transformer, Conv1D).

Hard-constraints sec 9 zero-handling: NaN feature values are
replaced with 0.0 at load time so downstream models never see NaN.
TVL columns are zero-filled when `tvl_available_flag == 0`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------- features

# V1: ladder + LTP + time-to-off + market-level basics.
V1_COLS: tuple[str, ...] = (
    "ltp",
    "back_p1", "back_p2", "back_p3",
    "back_s1", "back_s2", "back_s3",
    "lay_p1", "lay_p2", "lay_p3",
    "lay_s1", "lay_s2", "lay_s3",
    "traded_volume_runner",
    "num_active_runners",
    "time_to_off_sec",
)

# V2: V1 + per-runner trailing window stats.
V2_EXTRA: tuple[str, ...] = (
    "ltp_lag_1", "ltp_lag_5", "ltp_lag_10", "ltp_lag_30",
    "ltp_w32_mean", "ltp_w32_std", "ltp_w32_min", "ltp_w32_max",
    "ltp_w32_first", "ltp_w32_n",
)

# V3: V2 + TradedVolumeLadder.
V3_EXTRA: tuple[str, ...] = (
    "tvl_total", "tvl_at_ltp",
    "tvl_below_5t", "tvl_below_10t",
    "tvl_above_5t", "tvl_above_10t",
    "tvl_n_levels", "tvl_available_flag",
)

# V4: V3 + cross-runner.
V4_EXTRA: tuple[str, ...] = (
    "rank_in_market",
    "ltp_share",
    "ltp_zscore_in_market",
    "volume_share_in_market",
    "volume_zscore_in_market",
)

# V5: V4 + market-state.
V5_EXTRA: tuple[str, ...] = (
    "mkt_total_traded_volume",
    "mkt_avg_spread_ticks",
    "mkt_depth_total",
    "mkt_n_active",
)

VARIANT_COLS: dict[str, tuple[str, ...]] = {
    "V1": V1_COLS,
    "V2": V1_COLS + V2_EXTRA,
    "V3": V1_COLS + V2_EXTRA + V3_EXTRA,
    "V4": V1_COLS + V2_EXTRA + V3_EXTRA + V4_EXTRA,
    "V5": V1_COLS + V2_EXTRA + V3_EXTRA + V4_EXTRA + V5_EXTRA,
}


def feature_columns(variant: str) -> list[str]:
    if variant not in VARIANT_COLS:
        raise ValueError(f"unknown feature variant {variant!r}")
    return list(VARIANT_COLS[variant])


# ----------------------------------------------------------------- loading

DEFAULT_DATASET_DIR = Path(
    r"C:\Users\jsmit\source\repos\rl-betfair\data\predictor_dataset"
)


def load_split(
    split_name: str,
    feature_variant: str,
    horizons: list[str],
    train_corpus: str = "tvl_required_10d",
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> pd.DataFrame:
    """Load all parquets for a split, concat, return as one DataFrame.

    train_corpus:
      - tvl_required_10d: only dates with TVL available (>= 2026-04-26)
      - tvl_mask_29d:     all dates; TVL columns zero-filled where missing
    """
    from scripts.predictor.splits import (
        TRAIN_DATES, VAL_DATES, TEST_DATES,
        TVL_AVAILABLE_FROM,
    )
    if split_name == "train":
        date_set = TRAIN_DATES
    elif split_name == "val":
        date_set = VAL_DATES
    elif split_name == "test":
        date_set = TEST_DATES
    else:
        raise ValueError(f"unknown split {split_name!r}")

    if train_corpus == "tvl_required_10d":
        date_set = [d for d in date_set if d >= TVL_AVAILABLE_FROM]
    elif train_corpus == "tvl_mask_29d":
        pass
    else:
        raise ValueError(f"unknown train_corpus {train_corpus!r}")

    feat_cols = feature_columns(feature_variant)
    label_cols: list[str] = []
    for h in horizons:
        label_cols.extend([f"delta_ticks_{h}", f"label_exists_{h}"])
    keep_cols = (
        ["market_id", "selection_id", "timestamp", "tick_idx"]
        + feat_cols
        + label_cols
    )

    frames: list[pd.DataFrame] = []
    for d in date_set:
        path = dataset_dir / f"{d.isoformat()}.parquet"
        if not path.exists():
            logger.warning("missing dataset shard %s", path)
            continue
        # Load only needed columns.
        df = pd.read_parquet(path, columns=keep_cols)
        frames.append(df)

    if not frames:
        raise RuntimeError(f"no dataset shards found for {split_name}")

    out = pd.concat(frames, ignore_index=True)
    return out


def to_feature_tensor(
    df: pd.DataFrame,
    feature_variant: str,
) -> tuple[np.ndarray, list[str]]:
    """Extract feature columns -> dense float32 array, NaN -> 0.0 (sec 9)."""
    cols = feature_columns(feature_variant)
    arr = df[cols].to_numpy(dtype=np.float32, copy=True)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, cols


def to_label_tensors(
    df: pd.DataFrame,
    horizons: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack delta_ticks per horizon (n, n_horizons) plus an
    `exists` mask (n, n_horizons) of bool. NaN labels are kept as
    NaN; the mask determines what contributes to the loss.
    """
    n = len(df)
    y = np.full((n, len(horizons)), np.nan, dtype=np.float32)
    mask = np.zeros((n, len(horizons)), dtype=bool)
    for j, h in enumerate(horizons):
        y[:, j] = df[f"delta_ticks_{h}"].to_numpy(dtype=np.float32)
        mask[:, j] = df[f"label_exists_{h}"].to_numpy(dtype=bool)
    return y, mask


# ----------------------------------------------------------------- tabular Dataset


@dataclass
class TabularExamples:
    X: np.ndarray             # (n, F) float32
    y: np.ndarray             # (n, H) float32 (NaN where missing)
    mask: np.ndarray          # (n, H) bool
    feature_names: list[str]
    horizons: list[str]


class TabularDataset(Dataset):
    """Each (market, runner, tick) row is one independent example."""
    def __init__(self, ex: TabularExamples) -> None:
        self.ex = ex

    def __len__(self) -> int:
        return len(self.ex.X)

    def __getitem__(self, i: int):
        return (
            torch.from_numpy(self.ex.X[i]),
            torch.from_numpy(self.ex.y[i]),
            torch.from_numpy(self.ex.mask[i]),
        )


# ----------------------------------------------------------------- sequence Dataset


@dataclass
class SequenceExamples:
    """Pre-built sequence windows.

    For each (market, runner) trajectory we extract examples
    where the input is the last `window` ticks (left-padded with
    zeros if fewer ticks are available so the example index lines
    up with TabularExamples row indices) and the target is the
    label at the rightmost tick.
    """
    X: np.ndarray             # (n, window, F) float32
    y: np.ndarray             # (n, H) float32
    mask: np.ndarray          # (n, H) bool
    feature_names: list[str]
    horizons: list[str]
    window: int


def build_sequence_examples(
    df: pd.DataFrame,
    feature_variant: str,
    horizons: list[str],
    window: int,
) -> SequenceExamples:
    """Build per-(market, runner) windowed sequences. Left-pad with zeros."""
    cols = feature_columns(feature_variant)
    n_feat = len(cols)
    # Sort once globally so per-group is contiguous in idx order.
    df = df.sort_values(["market_id", "selection_id", "tick_idx"]).reset_index(drop=True)

    # Pre-build flat feature array + label array.
    flat_feat = df[cols].to_numpy(dtype=np.float32, copy=True)
    flat_feat = np.nan_to_num(flat_feat, nan=0.0, posinf=0.0, neginf=0.0)
    flat_y = np.stack(
        [df[f"delta_ticks_{h}"].to_numpy(dtype=np.float32) for h in horizons],
        axis=1,
    )
    flat_mask = np.stack(
        [df[f"label_exists_{h}"].to_numpy(dtype=bool) for h in horizons],
        axis=1,
    )

    # Group boundaries via groupby().indices is dict-of-arrays; faster:
    # detect changes in the (market_id, selection_id) tuple.
    keys = df[["market_id", "selection_id"]].agg(tuple, axis=1).to_numpy()
    # Use np.unique with return_inverse to get group ids; then groupby on those.
    # This avoids tuple comparisons per row in Python.
    _, group_ids = np.unique(keys, return_inverse=True)

    n_total = len(df)
    X_out = np.zeros((n_total, window, n_feat), dtype=np.float32)
    # Per-group, walk forward, fill window with last `window` rows of features.
    # Most efficient: for each group, slice the flat array.
    sort_idx = np.argsort(group_ids, kind="stable")
    sorted_groups = group_ids[sort_idx]
    # group_ids is already in monotonic order due to the dataframe sort, so
    # sort_idx == arange(n_total). We can iterate group boundaries by diff.
    boundaries = np.where(np.diff(sorted_groups) != 0)[0] + 1
    boundaries = np.concatenate([[0], boundaries, [n_total]])

    for b in range(len(boundaries) - 1):
        a, c = boundaries[b], boundaries[b + 1]
        idxs = sort_idx[a:c]  # original df indices of this group, in order
        L = c - a
        group_feat = flat_feat[idxs]  # (L, F)
        for k in range(L):
            lo = max(0, k + 1 - window)
            seg = group_feat[lo:k + 1]
            # Pad on the LEFT with zeros so newest tick is at index -1.
            if seg.shape[0] < window:
                pad = np.zeros((window - seg.shape[0], n_feat), dtype=np.float32)
                seg = np.concatenate([pad, seg], axis=0)
            X_out[idxs[k]] = seg

    return SequenceExamples(
        X=X_out,
        y=flat_y,
        mask=flat_mask,
        feature_names=list(cols),
        horizons=list(horizons),
        window=window,
    )


class SequenceDataset(Dataset):
    def __init__(self, ex: SequenceExamples) -> None:
        self.ex = ex

    def __len__(self) -> int:
        return len(self.ex.X)

    def __getitem__(self, i: int):
        return (
            torch.from_numpy(self.ex.X[i]),
            torch.from_numpy(self.ex.y[i]),
            torch.from_numpy(self.ex.mask[i]),
        )
