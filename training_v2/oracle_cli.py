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
    predictor_lean_obs: bool = False,
    predictor_bundle: object | None = None,
    use_race_outcome_predictor: bool = False,
    use_direction_predictor: bool = False,
) -> None:
    n_ticks = count_pre_race_ticks(date, data_dir)
    samples = scan_day(
        date, data_dir, config, scorer_dir=scorer_dir,
        predictor_lean_obs=predictor_lean_obs,
        predictor_bundle=predictor_bundle,
        use_race_outcome_predictor=use_race_outcome_predictor,
        use_direction_predictor=use_direction_predictor,
    )

    if samples:
        obs_dim = samples[0].obs.shape[0]
    else:
        # Empty-day path: derive obs_dim from a fresh shim so the cache
        # header still records the canonical width — keeps the v1 cache
        # contract that an empty .npz is shaped, not a placeholder.
        try:
            day = load_day(date, data_dir)
            env_kwargs = dict(
                scalping_mode=True,
                predictor_lean_obs=predictor_lean_obs,
            )
            if predictor_bundle is not None:
                env_kwargs["predictor_bundle"] = predictor_bundle
            if use_race_outcome_predictor:
                env_kwargs["use_race_outcome_predictor"] = True
            if use_direction_predictor:
                env_kwargs["use_direction_predictor"] = True
            env = BetfairEnv(day, config, **env_kwargs)
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
    scan_p.add_argument(
        "--predictor-lean-obs", action="store_true",
        help=(
            "Build the env with predictor_lean_obs=True (23 keys per "
            "runner instead of 143). Required when the downstream cohort "
            "uses --predictor-lean-obs — otherwise the cache's obs_dim "
            "won't match the trainer's shim and BC pretrain will refuse "
            "to load (see BetfairEnv LEAN_RUNNER_KEYS)."
        ),
    )
    scan_p.add_argument(
        "--predictor-bundle-manifests", nargs=3, default=None,
        metavar=("CHAMPION", "RANKER", "DIRECTION"),
        help=(
            "Three manifest.json paths (race-outcome champion, "
            "race-outcome ranker, direction predictor). When supplied "
            "together with --use-race-outcome-predictor and/or "
            "--use-direction-predictor, the env populates the "
            "predictor obs columns at scan time. Without this, those "
            "obs columns are zero-filled and diagnostic scripts that "
            "read them (e.g. tools/direction_signal_probe.py) see "
            "only zeros. 2026-05-24."
        ),
    )
    scan_p.add_argument(
        "--use-race-outcome-predictor", action="store_true",
        help="Inject race-outcome predictor outputs into obs at scan.",
    )
    scan_p.add_argument(
        "--use-direction-predictor", action="store_true",
        help="Inject direction predictor outputs into obs at scan.",
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

    predictor_bundle = None
    if args.predictor_bundle_manifests:
        from predictors import PredictorBundle
        champ, rank, dirm = args.predictor_bundle_manifests
        predictor_bundle = PredictorBundle.from_manifests(
            champion_manifest=Path(champ),
            ranker_manifest=Path(rank),
            direction_manifest=Path(dirm),
        )

    for d in dates:
        _scan_one(
            d, data_dir, config, scorer_dir,
            predictor_lean_obs=bool(args.predictor_lean_obs),
            predictor_bundle=predictor_bundle,
            use_race_outcome_predictor=bool(
                args.use_race_outcome_predictor,
            ),
            use_direction_predictor=bool(args.use_direction_predictor),
        )


if __name__ == "__main__":
    main()
