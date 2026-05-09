"""scripts/predictor/build_dataset.py - production labelling pipeline.

Session 01 of plans/price-direction-predictor. Reads raw daily
parquets and emits one persisted labelled-example parquet per date
under data/predictor_dataset/{date}.parquet.

Each output row = one (market_id, selection_id, tick) example with:
  - all feature columns (V1..V5 unioned -- variant selection happens
    at training-read time)
  - multi-horizon labels (delta_ticks, future_ltp, label_exists per
    horizon in HORIZONS_SEC)

Pre-off only (hard_constraints sec 1). Self-supervised (sec 2):
labels read straight from future LastTradedPrice in the same
parquet. No simulator. No oracle.

Idempotent: re-running on a date that already has a shard skips it
unless --rebuild is set.

Run:
    python scripts/predictor/build_dataset.py [--dates 2026-04-06,...]
                                              [--all]
                                              [--rebuild]
                                              [--out-dir data/predictor_dataset]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import deque
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.tick_ladder import ticks_between, snap_to_tick  # noqa: E402
from scripts.predictor.splits import (  # noqa: E402
    TRAIN_DATES,
    VAL_DATES,
    TEST_DATES,
    TVL_AVAILABLE_FROM,
    split_for_date,
    tvl_available_on,
)

logger = logging.getLogger(__name__)

# Horizons in seconds (union of all sweeps in master_todo).
# 30m dropped: polled feed starts at exactly 1800s pre-off, so 30m
# labels would have ~0% coverage on every day -- the target_ts
# always lands at or after market_start_time. Verified 2026-05-09
# across 10 sample days. If a future data extraction extends the
# pre-off window past 30 min this can be re-added without code
# changes other than this tuple.
HORIZONS_SEC: tuple[int, ...] = (60, 180, 420, 900)
HORIZON_NAMES: tuple[str, ...] = ("1m", "3m", "7m", "15m")

# Trailing-window length for V2 lag features.
WINDOW_LAG_TICKS: tuple[int, ...] = (1, 5, 10, 30)
WINDOW_AGG_TICKS = 32

DEFAULT_DATA_DIR = Path(
    r"C:\Users\jsmit\source\repos\rl-betfair\data\processed"
)
DEFAULT_OUT_DIR = Path(
    r"C:\Users\jsmit\source\repos\rl-betfair\data\predictor_dataset"
)


def signed_ticks(p_now: float, p_future: float) -> float:
    """Signed Betfair-tick distance. NaN if either price unpriceable.

    Sign convention: positive = price drifted UP (longer odds).
    Operator framing "horse comes in 10 ticks" = price falls = negative.
    """
    if not (np.isfinite(p_now) and np.isfinite(p_future)):
        return float("nan")
    if p_now <= 1.0 or p_future <= 1.0:
        return float("nan")
    if p_future == p_now:
        return 0.0
    n = ticks_between(p_now, p_future)
    return float(n) if p_future >= p_now else float(-n)


def _safe(x):
    """NaN-coerce None / null / weird values to NaN."""
    if x is None:
        return float("nan")
    try:
        f = float(x)
        return f if math.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _ladder_take(levels: list[dict] | None, k: int = 3) -> tuple[list[float], list[float]]:
    """Return (prices, sizes) padded with NaN to length k."""
    prices = [float("nan")] * k
    sizes = [float("nan")] * k
    if not levels:
        return prices, sizes
    for i, lvl in enumerate(levels[:k]):
        prices[i] = _safe(lvl.get("Price"))
        sizes[i] = _safe(lvl.get("Size"))
    return prices, sizes


def _tvl_features(tvl: list[dict] | None, ltp: float) -> dict[str, float]:
    """Compute V3 TVL features from TradedVolumeLadder.

    All features zero-fill when tvl is empty/None (hard_constraints sec 9).
    `tvl_available_flag` flags whether the source data was present at all.
    """
    out = {
        "tvl_total": 0.0,
        "tvl_at_ltp": 0.0,
        "tvl_below_5t": 0.0,
        "tvl_below_10t": 0.0,
        "tvl_above_5t": 0.0,
        "tvl_above_10t": 0.0,
        "tvl_n_levels": 0.0,
        "tvl_available_flag": 0.0,
    }
    if not tvl:
        return out
    out["tvl_available_flag"] = 1.0
    if ltp is None or not np.isfinite(ltp) or ltp <= 1.0:
        # Can't bucket without an LTP; total still computable.
        out["tvl_total"] = float(sum(_safe(lvl.get("Size")) for lvl in tvl))
        out["tvl_n_levels"] = float(len(tvl))
        return out
    ltp_snapped = snap_to_tick(float(ltp))
    total = 0.0
    n = 0
    for lvl in tvl:
        price = _safe(lvl.get("Price"))
        size = _safe(lvl.get("Size"))
        if not (np.isfinite(price) and np.isfinite(size)):
            continue
        n += 1
        total += size
        if abs(price - ltp_snapped) < 1e-9:
            out["tvl_at_ltp"] += size
            continue
        # Tick distance is computed on the ladder, not raw cents.
        td = ticks_between(ltp_snapped, price)
        if price < ltp_snapped:
            if td <= 5:
                out["tvl_below_5t"] += size
            if td <= 10:
                out["tvl_below_10t"] += size
        else:
            if td <= 5:
                out["tvl_above_5t"] += size
            if td <= 10:
                out["tvl_above_10t"] += size
    out["tvl_total"] = total
    out["tvl_n_levels"] = float(n)
    return out


def _market_state_features(
    runners_now: list[dict],
) -> dict[str, float]:
    """V5 market-state aggregates over the whole market this tick."""
    if not runners_now:
        return {
            "mkt_total_traded_volume": 0.0,
            "mkt_avg_spread_ticks": float("nan"),
            "mkt_depth_total": 0.0,
            "mkt_n_active": 0.0,
        }
    total_vol = 0.0
    spreads: list[float] = []
    depth_total = 0.0
    for r in runners_now:
        prices = r.get("Prices", {})
        ltp = _safe(prices.get("LastTradedPrice"))
        total_vol += _safe(prices.get("TradedVolume")) if np.isfinite(_safe(prices.get("TradedVolume"))) else 0.0
        atb = prices.get("AvailableToBack") or []
        atl = prices.get("AvailableToLay") or []
        if atb and atl:
            best_back = _safe(atb[0].get("Price"))
            best_lay = _safe(atl[0].get("Price"))
            if np.isfinite(best_back) and np.isfinite(best_lay) and best_lay > best_back:
                spreads.append(ticks_between(best_back, best_lay))
        for lvl in atb:
            depth_total += _safe(lvl.get("Size")) if np.isfinite(_safe(lvl.get("Size"))) else 0.0
        for lvl in atl:
            depth_total += _safe(lvl.get("Size")) if np.isfinite(_safe(lvl.get("Size"))) else 0.0
    return {
        "mkt_total_traded_volume": total_vol,
        "mkt_avg_spread_ticks": float(np.mean(spreads)) if spreads else float("nan"),
        "mkt_depth_total": depth_total,
        "mkt_n_active": float(len(runners_now)),
    }


def _cross_runner_features(
    runners_now: list[dict],
    sid: int,
) -> dict[str, float]:
    """V4 cross-runner features for the target runner within this market tick."""
    ltps: list[float] = []
    vols: list[float] = []
    target_ltp: float | None = None
    target_vol: float | None = None
    for r in runners_now:
        prices = r.get("Prices", {})
        rid = r.get("RunnerId", {}).get("SelectionId")
        ltp = _safe(prices.get("LastTradedPrice"))
        vol = _safe(prices.get("TradedVolume"))
        if not np.isfinite(ltp) or ltp <= 1.0:
            continue
        ltps.append(ltp)
        vols.append(vol if np.isfinite(vol) else 0.0)
        if rid == sid:
            target_ltp = ltp
            target_vol = vol if np.isfinite(vol) else 0.0
    if target_ltp is None or not ltps:
        return {
            "rank_in_market": float("nan"),
            "ltp_share": float("nan"),
            "ltp_zscore_in_market": float("nan"),
            "volume_share_in_market": float("nan"),
            "volume_zscore_in_market": float("nan"),
        }
    ltps_arr = np.asarray(ltps)
    vols_arr = np.asarray(vols)
    rank = int(np.searchsorted(np.sort(ltps_arr), target_ltp, side="left")) + 1
    inv = 1.0 / ltps_arr
    inv_target = 1.0 / target_ltp
    inv_sum = float(inv.sum())
    ltp_share = inv_target / inv_sum if inv_sum > 0 else float("nan")
    ltp_zscore = (
        (target_ltp - float(ltps_arr.mean())) / float(ltps_arr.std() + 1e-9)
        if len(ltps_arr) > 1
        else 0.0
    )
    vol_sum = float(vols_arr.sum())
    vol_share = (target_vol / vol_sum) if vol_sum > 0 else 0.0
    vol_zscore = (
        (target_vol - float(vols_arr.mean())) / float(vols_arr.std() + 1e-9)
        if len(vols_arr) > 1
        else 0.0
    )
    return {
        "rank_in_market": float(rank),
        "ltp_share": float(ltp_share),
        "ltp_zscore_in_market": float(ltp_zscore),
        "volume_share_in_market": float(vol_share),
        "volume_zscore_in_market": float(vol_zscore),
    }


def extract_one_day(parquet_path: Path) -> pd.DataFrame:
    """Read one day's parquet, walk per-market timelines, return labelled rows.

    See module docstring for output schema.
    """
    raw = pd.read_parquet(parquet_path)
    raw = raw.sort_values(["market_id", "timestamp"]).reset_index(drop=True)

    rows: list[dict] = []

    for mkt_id, mkt_df in raw.groupby("market_id"):
        mkt_df = mkt_df.sort_values("timestamp").reset_index(drop=True)
        market_start = mkt_df["market_start_time"].iloc[0]
        venue = mkt_df["venue"].iloc[0]
        n_active = mkt_df["number_of_active_runners"].iloc[0]

        # Per-market: per-runner timelines for both feature lookups (V2 lags
        # via deque) AND post-pass label assignment.
        # We build features in-tick (V1, V2, V3, V4, V5 all need only current
        # state + bounded history) and labels in a second pass once we have
        # the full per-runner LTP timeline.
        per_runner_recent: dict[int, deque] = {}
        per_runner_timeline_ts: dict[int, list[pd.Timestamp]] = {}
        per_runner_timeline_ltp: dict[int, list[float]] = {}
        per_runner_tick_idx: dict[int, int] = {}

        # First (and only) pass: walk ticks in order.
        for _i, row in mkt_df.iterrows():
            ts = row["timestamp"]
            in_play = bool(row["in_play"])

            # Pre-off filter (sec 1).
            if in_play:
                continue
            if ts >= market_start:
                continue

            try:
                snap = json.loads(row["snap_json"])
            except (TypeError, ValueError):
                continue
            runners_now = snap.get("MarketRunners", [])
            if not runners_now:
                continue

            mkt_state = _market_state_features(runners_now)

            for r in runners_now:
                sid = r.get("RunnerId", {}).get("SelectionId")
                if sid is None:
                    continue
                prices = r.get("Prices", {})
                ltp = _safe(prices.get("LastTradedPrice"))
                if not np.isfinite(ltp) or ltp <= 1.0:
                    # Unpriceable runner this tick; skip but DO NOT record
                    # a NaN row -- those are useless training examples.
                    continue

                back = prices.get("AvailableToBack") or []
                lay = prices.get("AvailableToLay") or []
                back_p, back_s = _ladder_take(back, k=3)
                lay_p, lay_s = _ladder_take(lay, k=3)

                # V1
                feat = {
                    "ltp": float(ltp),
                    "back_p1": back_p[0],
                    "back_p2": back_p[1],
                    "back_p3": back_p[2],
                    "back_s1": back_s[0],
                    "back_s2": back_s[1],
                    "back_s3": back_s[2],
                    "lay_p1": lay_p[0],
                    "lay_p2": lay_p[1],
                    "lay_p3": lay_p[2],
                    "lay_s1": lay_s[0],
                    "lay_s2": lay_s[1],
                    "lay_s3": lay_s[2],
                    "traded_volume_runner": _safe(prices.get("TradedVolume")),
                    "num_active_runners": float(n_active) if n_active is not None else float("nan"),
                    "time_to_off_sec": float((market_start - ts).total_seconds()),
                }

                # V2 lags + window aggregates over previous ticks.
                hist = per_runner_recent.setdefault(sid, deque(maxlen=WINDOW_AGG_TICKS))
                # Lags: index from the end of the deque (most recent prior).
                hist_list = list(hist)
                for lag in WINDOW_LAG_TICKS:
                    if lag <= len(hist_list):
                        feat[f"ltp_lag_{lag}"] = float(hist_list[-lag])
                    else:
                        feat[f"ltp_lag_{lag}"] = float("nan")
                if hist_list:
                    arr = np.asarray(hist_list, dtype=float)
                    feat["ltp_w32_mean"] = float(arr.mean())
                    feat["ltp_w32_std"] = float(arr.std(ddof=0))
                    feat["ltp_w32_min"] = float(arr.min())
                    feat["ltp_w32_max"] = float(arr.max())
                    feat["ltp_w32_first"] = float(arr[0])
                    feat["ltp_w32_n"] = float(len(arr))
                else:
                    feat["ltp_w32_mean"] = float("nan")
                    feat["ltp_w32_std"] = float("nan")
                    feat["ltp_w32_min"] = float("nan")
                    feat["ltp_w32_max"] = float("nan")
                    feat["ltp_w32_first"] = float("nan")
                    feat["ltp_w32_n"] = 0.0

                # V3 TVL.
                feat.update(_tvl_features(prices.get("TradedVolumeLadder"), ltp))

                # V4 cross-runner.
                feat.update(_cross_runner_features(runners_now, sid))

                # V5 market state.
                feat.update(mkt_state)

                # Output row.
                tick_idx = per_runner_tick_idx.get(sid, 0)
                row_out = {
                    "market_id": mkt_id,
                    "selection_id": int(sid),
                    "timestamp": ts,
                    "tick_idx": tick_idx,
                    "venue": venue,
                    "market_start_time": market_start,
                    **feat,
                }
                # Label placeholders -- filled in pass 2.
                for hn in HORIZON_NAMES:
                    row_out[f"future_ltp_{hn}"] = float("nan")
                    row_out[f"delta_ticks_{hn}"] = float("nan")
                    row_out[f"label_exists_{hn}"] = False
                rows.append(row_out)

                # Update per-runner state.
                hist.append(ltp)
                per_runner_timeline_ts.setdefault(sid, []).append(ts)
                per_runner_timeline_ltp.setdefault(sid, []).append(ltp)
                per_runner_tick_idx[sid] = tick_idx + 1

        # Pass 2: label assignment per (market, runner). Use forward-fill:
        # take the LTP of the FIRST tick at or after target_ts. If no such
        # tick exists (truncated by race off), label is NaN/False.
        # We need stable indexing: for each (market, sid), what is the
        # row index in `rows` for tick_idx=k? Build a lookup.
        # Since rows are appended in market-then-tick order, a simple
        # post-hoc index suffices for this market only.
        # NB: we re-walk just-appended rows of THIS market.
        # Build per-(sid) row index list.
        # rows for this market are the last N rows where market_id == mkt_id.
        # Find the start index of this market's rows in `rows`.
        # Easier: iterate per_runner_timeline_* in the order rows were emitted.
        # Build a (sid, tick_idx) -> rows-index map via reverse scan.
        if not per_runner_timeline_ts:
            continue

        # Find first index of this market's rows.
        # rows length minus all of this market's tick count gives start.
        n_market_rows = sum(len(v) for v in per_runner_timeline_ts.values())
        market_start_idx = len(rows) - n_market_rows

        # Walk this market's rows, look up label per (sid, ts).
        per_sid_ts_arr = {
            sid: np.asarray(tl) for sid, tl in per_runner_timeline_ts.items()
        }
        per_sid_ltp_arr = {
            sid: np.asarray(tl) for sid, tl in per_runner_timeline_ltp.items()
        }
        for j in range(market_start_idx, len(rows)):
            r_row = rows[j]
            sid = r_row["selection_id"]
            ts = r_row["timestamp"]
            ltp_now = r_row["ltp"]
            tl_ts = per_sid_ts_arr[sid]
            tl_ltp = per_sid_ltp_arr[sid]
            for hsec, hn in zip(HORIZONS_SEC, HORIZON_NAMES):
                target = ts + pd.Timedelta(seconds=hsec)
                if target >= market_start:
                    continue
                idx = int(np.searchsorted(tl_ts, target, side="left"))
                if idx >= len(tl_ts):
                    continue
                ltp_future = float(tl_ltp[idx])
                d = signed_ticks(ltp_now, ltp_future)
                if not np.isfinite(d):
                    continue
                r_row[f"future_ltp_{hn}"] = ltp_future
                r_row[f"delta_ticks_{hn}"] = d
                r_row[f"label_exists_{hn}"] = True

    return pd.DataFrame(rows)


def parse_dates_arg(s: str) -> list[date]:
    out: list[date] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(datetime.strptime(tok, "%Y-%m-%d").date())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dates", default=None, help="comma-sep YYYY-MM-DD")
    p.add_argument("--all", action="store_true", help="all 29 days")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument(
        "--no-test", action="store_true",
        help="hard_constraints sec 5: exclude test dates by default",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.all:
        dates = TRAIN_DATES + VAL_DATES + TEST_DATES
    elif args.dates:
        dates = parse_dates_arg(args.dates)
    else:
        raise SystemExit("--dates or --all required")

    if args.no_test:
        before = len(dates)
        dates = [d for d in dates if split_for_date(d) != "test"]
        skipped = before - len(dates)
        if skipped:
            logger.info("--no-test: skipping %d test date(s)", skipped)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    summary_rows: list[dict] = []
    for d in dates:
        date_str = d.isoformat()
        out_path = out_dir / f"{date_str}.parquet"
        if out_path.exists() and not args.rebuild:
            logger.info("skip %s (already exists)", date_str)
            df_existing = pd.read_parquet(out_path)
            summary_rows.append({
                "date": date_str,
                "split": split_for_date(d),
                "rows": len(df_existing),
                "tvl_available": tvl_available_on(d),
                "skipped": True,
            })
            continue

        in_path = data_dir / f"{date_str}.parquet"
        if not in_path.exists():
            logger.warning("missing input parquet for %s, skipping", date_str)
            continue

        t0 = time.time()
        logger.info("extracting %s ...", date_str)
        df = extract_one_day(in_path)
        elapsed = time.time() - t0

        # Hard checks (sec 1, sec 9). Pre-off only: every timestamp
        # MUST be strictly before its market_start_time.
        if not df.empty:
            offside = (df["timestamp"] >= df["market_start_time"]).any()
            if offside:
                raise RuntimeError(
                    f"{date_str}: post-off rows leaked into output (sec 1)"
                )
            # tvl_available_flag must be 0 or 1, never NaN.
            if df["tvl_available_flag"].isna().any():
                raise RuntimeError(
                    f"{date_str}: NaN tvl_available_flag (sec 9)"
                )

        df.to_parquet(out_path, index=False)
        logger.info(
            "  wrote %s rows in %.1fs to %s",
            f"{len(df):,}", elapsed, out_path,
        )
        summary_rows.append({
            "date": date_str,
            "split": split_for_date(d),
            "rows": len(df),
            "tvl_available": tvl_available_on(d),
            "skipped": False,
            "extract_seconds": round(elapsed, 1),
        })

    sdf = pd.DataFrame(summary_rows)
    if not sdf.empty:
        print()
        print(sdf.to_string(index=False))
        print()
        for split, gdf in sdf.groupby("split"):
            print(
                f"{split}: {len(gdf)} dates, "
                f"{gdf['rows'].sum():,} total rows"
            )


if __name__ == "__main__":
    main()
