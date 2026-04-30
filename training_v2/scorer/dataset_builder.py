"""Generate the Phase 0 supervised scorer dataset.

Walks every (date, market_id, runner_idx, tick_idx, side) opportunity
in ``data/processed/<date>.parquet`` and emits one row per opportunity
into ``data/scorer_v1/dataset/<date>.parquet`` with the locked feature
set + label + key columns for chronological splitting.

Per-day parquet shards keep memory bounded; Session 02 reads them all
via ``pyarrow.dataset`` or ``pd.concat`` on a directory glob.

CLI::

    python -m training_v2.scorer.dataset_builder \\
        --dates 2026-04-06 2026-04-07 \\
        --tick-stride 5 \\
        --out data/scorer_v1/dataset

The ``feature_spec.json`` is written next to the dataset directory and
captures the feature ordering + the function names that compute each.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from data.episode_builder import Day, load_day
from training_v2.scorer.feature_extractor import (
    FEATURE_NAMES,
    FeatureExtractor,
)
from training_v2.scorer.label_generator import (
    LabelGenerator,
    LabelOutcome,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _DayStats:
    date: str
    n_rows: int
    n_matured: int
    n_force_closed: int
    n_naked: int
    n_nan_label: int
    elapsed_sec: float
    outcome_counts: dict[str, int]


def build_day_dataset(
    day: Day,
    *,
    extractor: FeatureExtractor,
    labeler: LabelGenerator,
    tick_stride: int = 5,
    side_choices: tuple[str, ...] = ("back", "lay"),
) -> tuple[pd.DataFrame, _DayStats]:
    """Iterate every opportunity in ``day``; return one DataFrame + stats.

    The extractor's history state is built up tick-by-tick within each
    market and reset (``forget_market``) after the market is processed,
    so memory grows linearly with concurrent opportunity emission, not
    with day length.
    """
    t0 = time.perf_counter()
    rows: list[dict] = []
    outcome_counts: Counter[str] = Counter()

    for race in day.races:
        n_ticks = len(race.ticks)
        if n_ticks < 2:
            continue

        for j, tick in enumerate(race.ticks):
            # Maintain the rolling-window history for every tick (so the
            # 30s/60s windows are populated when we land on a stride-emit
            # tick).
            extractor.update_history(race, tick)

            if j % tick_stride != 0:
                continue
            if tick.in_play:
                # Skip in-play ticks — placement is pre-race only. The
                # label simulator returns INFEASIBLE_IN_PLAY too; we
                # short-circuit here to avoid the full simulation.
                continue

            for runner_idx, runner_snap in enumerate(tick.runners):
                if runner_snap.status != "ACTIVE":
                    continue
                for side in side_choices:
                    label_result = labeler.generate(race, j, runner_idx, side)
                    outcome_counts[label_result.outcome.value] += 1
                    feats = extractor.extract(race, j, runner_idx, side)
                    row: dict = {
                        "date": day.date,
                        "market_id": race.market_id,
                        "selection_id": runner_snap.selection_id,
                        "runner_idx": runner_idx,
                        "tick_idx": j,
                        "side": side,
                        "label": (
                            np.float32(label_result.label)
                            if label_result.label is not None
                            else np.float32(np.nan)
                        ),
                        "outcome": label_result.outcome.value,
                    }
                    # Feature names in declaration order — preserved
                    # because Python dicts are insertion-ordered (3.7+).
                    for name in FEATURE_NAMES:
                        row[name] = np.float32(feats[name])
                    rows.append(row)

        extractor.forget_market(race.market_id)

    df = pd.DataFrame(rows)
    elapsed = time.perf_counter() - t0
    stats = _DayStats(
        date=day.date,
        n_rows=len(df),
        n_matured=int((df.get("outcome", pd.Series(dtype=str)) == "matured").sum()) if not df.empty else 0,
        n_force_closed=int((df.get("outcome", pd.Series(dtype=str)) == "force_closed").sum()) if not df.empty else 0,
        n_naked=int((df.get("outcome", pd.Series(dtype=str)) == "naked").sum()) if not df.empty else 0,
        n_nan_label=int(df["label"].isna().sum()) if not df.empty else 0,
        elapsed_sec=elapsed,
        outcome_counts=dict(outcome_counts),
    )
    return df, stats


def write_feature_spec(out_path: Path) -> None:
    """Persist the feature contract — names + dtype + computing-function.

    Session 02 (and Phase 1) READ this to validate that the dataset
    they're consuming uses the same feature ordering they're indexing
    into. NEVER reorder; only append.
    """
    spec = {
        "version": "scorer_v1",
        "feature_count": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "dtype": "float32",
        "extractor_module": "training_v2.scorer.feature_extractor",
        "extractor_class": "FeatureExtractor",
        "extractor_method": "extract",
        "label_generator_module": "training_v2.scorer.label_generator",
        "label_generator_class": "LabelGenerator",
        "label_generator_method": "generate",
        "outcome_classes": [oc.value for oc in LabelOutcome],
        "key_columns": [
            "date", "market_id", "selection_id",
            "runner_idx", "tick_idx", "side",
        ],
        "label_column": "label",
        "label_dtype": "float32",
        "outcome_column": "outcome",
    }
    out_path.write_text(json.dumps(spec, indent=2))


def build_dataset(
    dates: Sequence[str],
    *,
    out_dir: Path | str,
    data_dir: Path | str = "data/processed",
    tick_stride: int = 5,
    side_choices: tuple[str, ...] = ("back", "lay"),
    starting_budget: float = 1000.0,
    arb_ticks: int | None | Literal["default"] = "default",
) -> list[_DayStats]:
    """Build per-day shards into ``out_dir``.

    Returns one ``_DayStats`` entry per successfully processed date for
    the operator's findings writeup.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    shard_dir = out_path / "dataset"
    shard_dir.mkdir(parents=True, exist_ok=True)

    write_feature_spec(out_path / "feature_spec.json")

    all_stats: list[_DayStats] = []
    for ds in dates:
        try:
            day = load_day(ds, data_dir=data_dir)
        except FileNotFoundError:
            logger.warning("Skipping %s — file not found", ds)
            continue
        labeler_kwargs: dict = {
            "starting_budget": starting_budget,
            "fill_mode": day.fill_mode,
        }
        if arb_ticks != "default":
            labeler_kwargs["arb_ticks"] = arb_ticks
        labeler = LabelGenerator(**labeler_kwargs)
        extractor = FeatureExtractor()
        df, stats = build_day_dataset(
            day,
            extractor=extractor,
            labeler=labeler,
            tick_stride=tick_stride,
            side_choices=side_choices,
        )
        if df.empty:
            logger.warning("No rows emitted for %s", ds)
        else:
            shard_path = shard_dir / f"{ds}.parquet"
            df.to_parquet(shard_path, index=False)
            logger.info(
                "%s: %d rows in %.1fs (matured=%d, force_closed=%d, "
                "naked=%d, nan=%d) → %s",
                ds, stats.n_rows, stats.elapsed_sec,
                stats.n_matured, stats.n_force_closed,
                stats.n_naked, stats.n_nan_label, shard_path,
            )
        all_stats.append(stats)
    return all_stats


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the Phase 0 supervised scorer dataset.",
    )
    p.add_argument(
        "--dates", nargs="+", required=True,
        help="ISO dates (YYYY-MM-DD) to process.",
    )
    p.add_argument(
        "--data-dir", default="data/processed",
        help="Directory containing the input parquet files.",
    )
    p.add_argument(
        "--out", default="data/scorer_v1",
        help="Output directory; per-day shards land in <out>/dataset/.",
    )
    p.add_argument(
        "--tick-stride", type=int, default=5,
        help="Emit every Nth tick. 1 = no sub-sampling.",
    )
    p.add_argument(
        "--starting-budget", type=float, default=1000.0,
        help="Per-opportunity simulator budget (must be enough to cover "
             "the equal-profit passive's liability at extreme prices).",
    )
    p.add_argument(
        "--arb-ticks", default="default",
        help="Fixed tick offset between aggressive and passive legs. "
             "Pass an integer (e.g. 20), 'none' to enable the dynamic "
             "min_arb_ticks_for_profit lookup (revert path once book "
             "depth widens — see label_generator._DEFAULT_ARB_TICKS), "
             "or 'default' to use the module's hardcoded default.",
    )
    return p.parse_args(argv)


def _main(argv: Sequence[str]) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    if args.arb_ticks == "default":
        arb_ticks: int | None | Literal["default"] = "default"
    elif args.arb_ticks.lower() == "none":
        arb_ticks = None
    else:
        arb_ticks = int(args.arb_ticks)
    stats = build_dataset(
        dates=args.dates,
        out_dir=args.out,
        data_dir=args.data_dir,
        tick_stride=args.tick_stride,
        starting_budget=args.starting_budget,
        arb_ticks=arb_ticks,
    )
    total_rows = sum(s.n_rows for s in stats)
    total_matured = sum(s.n_matured for s in stats)
    total_force = sum(s.n_force_closed for s in stats)
    total_naked = sum(s.n_naked for s in stats)
    total_nan = sum(s.n_nan_label for s in stats)
    feasible = total_rows - total_nan
    logger.info(
        "Done: %d total rows across %d days. matured=%d (%.2f%% of "
        "feasible), force_closed=%d, naked=%d, nan=%d",
        total_rows, len(stats), total_matured,
        100.0 * total_matured / max(feasible, 1),
        total_force, total_naked, total_nan,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
