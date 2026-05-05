"""CLI for the v2 offline arb oracle scan.

::

    python -m training_v2.oracle_cli scan --date 2026-04-29
    python -m training_v2.oracle_cli scan --dates 2026-04-29,2026-04-30

Per-line output for each scanned date::

    {date}: samples=N ticks=T density=N/T unique_arb_ticks=U \
unique_arb_density=U/T

A trailing ``*** LOW DENSITY ***`` warning fires when ``density <
0.001`` so a misconfigured scorer (e.g. wrong feature spec) is visible
without trawling logs. Cache lands in ``data/oracle_cache_v2/{date}/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agents_v2.env_shim import DEFAULT_SCORER_DIR, DiscreteActionShim
from data.episode_builder import load_day
from env.betfair_env import BetfairEnv
from training_v2.arb_oracle import (
    _load_config,
    count_pre_race_ticks,
    save_samples,
    scan_day,
)


def _scan_one(
    date: str,
    data_dir: Path,
    config: dict,
    scorer_dir: Path,
) -> None:
    n_ticks = count_pre_race_ticks(date, data_dir)
    samples = scan_day(date, data_dir, config, scorer_dir=scorer_dir)

    if samples:
        obs_dim = samples[0].obs.shape[0]
    else:
        # Empty-day path: derive obs_dim from a fresh shim so the cache
        # header still records the canonical width — keeps the v1 cache
        # contract that an empty .npz is shaped, not a placeholder.
        try:
            day = load_day(date, data_dir)
            env = BetfairEnv(day, config, scalping_mode=True)
            shim = DiscreteActionShim(env, scorer_dir=scorer_dir)
            obs_dim = int(shim.obs_dim)
        except Exception:
            obs_dim = 0

    save_samples(samples, date, data_dir, config, n_ticks, obs_dim)
    density = len(samples) / max(n_ticks, 1)
    unique_arb = len({s.tick_index for s in samples})
    unique_density = unique_arb / max(n_ticks, 1)
    warn = "  *** LOW DENSITY ***" if density < 0.001 else ""
    print(
        f"{date}: samples={len(samples)} ticks={n_ticks} "
        f"density={density:.4f} "
        f"unique_arb_ticks={unique_arb} unique_arb_density={unique_density:.4f}"
        f"{warn}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Offline arb oracle scan — v2. Produces per-day .npz caches "
            "with shim-extended observations (env obs + Phase 0 scorer "
            "features)."
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
    scan_p.add_argument(
        "--scorer-dir",
        default=str(DEFAULT_SCORER_DIR),
        help="Phase 0 scorer artefact directory (default: models/scorer_v1).",
    )
    args = ap.parse_args()

    dates: list[str] = (
        [d.strip() for d in args.dates.split(",")]
        if args.dates
        else [args.date]
    )

    config = _load_config()
    data_dir = Path(args.data_dir)
    scorer_dir = Path(args.scorer_dir)

    for d in dates:
        _scan_one(d, data_dir, config, scorer_dir)


if __name__ == "__main__":
    main()
