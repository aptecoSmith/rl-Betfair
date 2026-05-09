"""Prototype: extract per-(market, runner, tick) delta-price-tick labels.

Session 01 of plans/price-direction-predictor. Reads one or more days
of parquet, walks each (market, runner) trajectory, and computes
signed delta-price-in-Betfair-ticks at horizons 1 / 3 / 7 minutes ahead.

Pre-off only (in_play=False AND timestamp < market_start_time --
hard_constraints sec 1). Self-supervised: labels come from future LTP
in the same parquet, never from a simulator (sec 2).

Run:
    python scripts/predictor/extract_labels_prototype.py \
        --date 2026-05-06 \
        [--data-dir C:/Users/jsmit/source/repos/rl-betfair/data/processed]

Prints summary stats; writes nothing to disk in this prototype. The
production pipeline (Session 02) will persist examples per-date.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running both as script and via -m from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.tick_ladder import ticks_between  # noqa: E402

# Horizons in seconds.
HORIZONS_SEC = (60, 180, 420)
HORIZON_NAMES = ("1m", "3m", "7m")


def signed_ticks_between(p_now: float, p_future: float) -> int | None:
    """Signed Betfair-tick distance: positive = price drifted (longer odds)."""
    if not (np.isfinite(p_now) and np.isfinite(p_future)):
        return None
    if p_now <= 1.0 or p_future <= 1.0:
        return None
    n = ticks_between(p_now, p_future)
    if p_future == p_now:
        return 0
    # Sign convention: p_future > p_now => price drifted UP => +ticks.
    # The operator framing was "horse comes in 10 ticks" = price falls
    # = negative Δticks. We keep the raw signed value here; downstream
    # decision rules can flip the sign per side.
    return n if p_future >= p_now else -n


def extract_one_day(parquet_path: Path) -> pd.DataFrame:
    """Walk one day's parquet, return one row per (market, runner, tick)."""
    raw = pd.read_parquet(parquet_path)
    raw = raw.sort_values(["market_id", "timestamp"]).reset_index(drop=True)

    rows: list[dict] = []
    skipped_post_off = 0
    skipped_in_play = 0
    skipped_no_ltp = 0

    for mkt_id, mkt_df in raw.groupby("market_id"):
        mkt_df = mkt_df.sort_values("timestamp").reset_index(drop=True)
        market_start = mkt_df["market_start_time"].iloc[0]

        # Per-runner timeline of (timestamp, LTP). Built once per market.
        # Each entry: list[(ts, ltp)] sorted by ts.
        per_runner_ts_ltp: dict[int, list[tuple[pd.Timestamp, float]]] = (
            defaultdict(list)
        )
        per_runner_tick_records: dict[int, list[dict]] = defaultdict(list)

        for _i, row in mkt_df.iterrows():
            ts = row["timestamp"]
            in_play = bool(row["in_play"])

            # Pre-off filter (§1).
            if in_play:
                skipped_in_play += 1
                continue
            if ts >= market_start:
                skipped_post_off += 1
                continue

            try:
                snap = json.loads(row["snap_json"])
            except (TypeError, ValueError):
                continue

            for r in snap.get("MarketRunners", []):
                sid = r.get("RunnerId", {}).get("SelectionId")
                if sid is None:
                    continue
                prices = r.get("Prices", {})
                ltp = prices.get("LastTradedPrice")
                if ltp is None or ltp <= 1.0:
                    skipped_no_ltp += 1
                    continue

                per_runner_ts_ltp[sid].append((ts, float(ltp)))
                per_runner_tick_records[sid].append(
                    {
                        "market_id": mkt_id,
                        "selection_id": sid,
                        "timestamp": ts,
                        "ltp": float(ltp),
                        "time_to_off_sec": (
                            (market_start - ts).total_seconds()
                        ),
                    }
                )

        # Now compute future-LTP labels per runner per tick.
        for sid, recs in per_runner_tick_records.items():
            timeline = per_runner_ts_ltp[sid]
            tl_ts = np.array([t for t, _ in timeline])
            tl_ltp = np.array([p for _, p in timeline])

            for rec in recs:
                t = rec["timestamp"]
                ltp_now = rec["ltp"]
                row_out = dict(rec)

                for h_sec, h_name in zip(HORIZONS_SEC, HORIZON_NAMES):
                    target_ts = t + pd.Timedelta(seconds=h_sec)

                    # Constraint: future LTP must still be pre-off.
                    if target_ts >= market_start:
                        row_out[f"future_ltp_{h_name}"] = np.nan
                        row_out[f"delta_ticks_{h_name}"] = np.nan
                        row_out[f"label_exists_{h_name}"] = False
                        continue

                    # Forward-fill: take the most recent LTP at or after
                    # target_ts. If no such tick exists (very near off),
                    # label is missing.
                    idx = np.searchsorted(tl_ts, target_ts, side="left")
                    if idx >= len(tl_ts):
                        row_out[f"future_ltp_{h_name}"] = np.nan
                        row_out[f"delta_ticks_{h_name}"] = np.nan
                        row_out[f"label_exists_{h_name}"] = False
                        continue

                    ltp_future = tl_ltp[idx]
                    delta = signed_ticks_between(ltp_now, ltp_future)
                    row_out[f"future_ltp_{h_name}"] = ltp_future
                    row_out[f"delta_ticks_{h_name}"] = delta
                    row_out[f"label_exists_{h_name}"] = delta is not None

                rows.append(row_out)

    df = pd.DataFrame(rows)
    df.attrs["skipped_post_off"] = skipped_post_off
    df.attrs["skipped_in_play"] = skipped_in_play
    df.attrs["skipped_no_ltp"] = skipped_no_ltp
    return df


def summarise(df: pd.DataFrame) -> None:
    print(f"Total examples (rows): {len(df):,}")
    print(f"Unique markets: {df['market_id'].nunique()}")
    print(f"Unique runners: {df['selection_id'].nunique()}")
    print()
    print(
        "Skipped during extraction: "
        f"in_play={df.attrs.get('skipped_in_play', 0)}, "
        f"post_off={df.attrs.get('skipped_post_off', 0)}, "
        f"no_ltp={df.attrs.get('skipped_no_ltp', 0)}"
    )
    print()

    for h_name in HORIZON_NAMES:
        col = f"delta_ticks_{h_name}"
        exists_col = f"label_exists_{h_name}"
        n_total = len(df)
        n_exists = df[exists_col].sum()
        coverage = n_exists / n_total if n_total else 0
        deltas = df.loc[df[exists_col], col].astype(float)
        zero_frac = (deltas == 0).mean() if len(deltas) else float("nan")
        abs_d = deltas.abs()

        print(f"--- horizon {h_name} ---")
        print(
            f"  coverage: {n_exists:,} / {n_total:,} = {coverage:.1%}"
            "  (rest truncated by race off)"
        )
        if len(deltas) == 0:
            print("  no examples")
            continue
        print(
            "  delta_ticks: "
            f"mean={deltas.mean():+.3f}  std={deltas.std():.3f}  "
            f"min={int(deltas.min())}  max={int(deltas.max())}  "
            f"zero_frac={zero_frac:.1%}"
        )
        # Magnitude distribution.
        for k in (1, 3, 5, 10, 20):
            frac = (abs_d >= k).mean()
            print(f"  P(|d| >= {k:>2}) = {frac:.1%}")
        # Direction.
        n_pos = (deltas > 0).sum()
        n_neg = (deltas < 0).sum()
        n_zero = (deltas == 0).sum()
        print(
            f"  signs: drift+ = {n_pos:,}  shorten- = {n_neg:,}  "
            f"flat = {n_zero:,}"
        )
        print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument(
        "--data-dir",
        default=r"C:\Users\jsmit\source\repos\rl-betfair\data\processed",
    )
    args = p.parse_args()

    pq = Path(args.data_dir) / f"{args.date}.parquet"
    if not pq.exists():
        raise SystemExit(f"missing parquet: {pq}")

    print(f"reading {pq}")
    df = extract_one_day(pq)
    print()
    summarise(df)


if __name__ == "__main__":
    main()
