"""CLI for the v2 offline direction-label scan — phase-13 S02.

::

    python -m training_v2.direction_label_cli scan \\
        --date 2026-05-03 \\
        --horizon-ticks 60 \\
        --threshold-ticks 5 \\
        --force-close-before-off-seconds 60

Per-day stdout::

    {date}: pre_race_ticks=T labels_total=N
            positive_back={X:.4f} ({k_back}/{N})
            positive_lay={Y:.4f} ({k_lay}/{N})
            both_positive={Z:.4f} ({k_both}/{N})
            wall={W:.1f}s horizon={H} thresh={T} fc={F}

Exit 1 if any day's wall exceeds 600s. Cache lands in
``data/direction_labels/{date}/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from training_v2.direction_label_scan import (
    _load_config,
    count_pre_race_ticks,
    save_labels,
    scan_day,
)


def _scan_one(
    date: str,
    data_dir: Path,
    config: dict,
    *,
    horizon_ticks: int | None,
    horizon_seconds: float | None,
    threshold_ticks: int,
    force_close: float,
) -> float:
    n_ticks = count_pre_race_ticks(date, data_dir)
    t0 = time.monotonic()
    labels = scan_day(
        date,
        data_dir,
        config,
        direction_horizon_ticks=horizon_ticks,
        direction_horizon_seconds=horizon_seconds,
        direction_threshold_ticks=threshold_ticks,
        force_close_before_off_seconds=force_close,
    )
    wall = time.monotonic() - t0

    save_labels(
        labels,
        date,
        data_dir,
        config,
        direction_horizon_ticks=horizon_ticks,
        direction_horizon_seconds=horizon_seconds,
        direction_threshold_ticks=threshold_ticks,
        force_close_before_off_seconds=force_close,
        total_pre_race_ticks=n_ticks,
    )

    n = len(labels)
    n_back = sum(1 for r in labels if r.label_back > 0.5)
    n_lay = sum(1 for r in labels if r.label_lay > 0.5)
    n_both = sum(
        1 for r in labels if r.label_back > 0.5 and r.label_lay > 0.5
    )
    pb = (n_back / n) if n > 0 else 0.0
    pl = (n_lay / n) if n > 0 else 0.0
    pboth = (n_both / n) if n > 0 else 0.0
    horizon_label = (
        f"horizon_seconds={horizon_seconds}" if horizon_seconds is not None
        else f"horizon_ticks={horizon_ticks}"
    )
    print(
        f"{date}: pre_race_ticks={n_ticks} labels_total={n}\n"
        f"        positive_back={pb:.4f} ({n_back}/{n})\n"
        f"        positive_lay={pl:.4f} ({n_lay}/{n})\n"
        f"        both_positive={pboth:.4f} ({n_both}/{n})\n"
        f"        wall={wall:.1f}s {horizon_label} "
        f"thresh={threshold_ticks} fc={force_close}",
        flush=True,
    )
    return wall


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Offline direction-label scan — phase-13 S02. Produces "
            "per-day .npz caches + header.json with per-side per-runner "
            "binary direction labels (label_back, label_lay)."
        ),
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    scan_p = sub.add_parser("scan", help="Scan one or more dates.")
    group = scan_p.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Single date YYYY-MM-DD.")
    group.add_argument("--dates", help="Comma-separated dates.")
    scan_p.add_argument(
        "--data-dir",
        default="data/processed",
        help="Processed data directory (default: data/processed).",
    )
    horizon_group = scan_p.add_mutually_exclusive_group(required=True)
    horizon_group.add_argument(
        "--horizon-ticks",
        type=int,
        default=None,
        help=(
            "Tick-count horizon for the favourable-move scan "
            "(v1_threshold_crossing mode, original 2026-05-06)."
        ),
    )
    horizon_group.add_argument(
        "--horizon-seconds",
        type=float,
        default=None,
        help=(
            "Clock-time horizon in seconds (v2_time_endpoint_signed_tick "
            "mode, 2026-05-24). Endpoint semantics: label = sign of "
            "(LTP at T+horizon) − (LTP at T) in ticks. Default = 420 "
            "(7 minutes) matches the betfair-predictors direction "
            "model's primary fire horizon."
        ),
    )
    scan_p.add_argument(
        "--threshold-ticks",
        type=int,
        required=True,
        help="Tick-count threshold for what counts as a favourable move.",
    )
    scan_p.add_argument(
        "--force-close-before-off-seconds",
        type=float,
        required=True,
        help=(
            "Force-close cutoff in wall seconds (matches "
            "config.constraints.force_close_before_off_seconds)."
        ),
    )
    args = ap.parse_args()

    dates: list[str] = (
        [d.strip() for d in args.dates.split(",")]
        if args.dates
        else [args.date]
    )

    config = _load_config()
    data_dir = Path(args.data_dir)

    over_budget = False
    for d in dates:
        wall = _scan_one(
            d,
            data_dir,
            config,
            horizon_ticks=args.horizon_ticks,
            horizon_seconds=args.horizon_seconds,
            threshold_ticks=args.threshold_ticks,
            force_close=args.force_close_before_off_seconds,
        )
        if wall > 600.0:
            over_budget = True
            print(
                f"  *** WALL EXCEEDED 600s on {d}: {wall:.1f}s ***",
                file=sys.stderr,
            )

    sys.exit(1 if over_budget else 0)


if __name__ == "__main__":
    main()
