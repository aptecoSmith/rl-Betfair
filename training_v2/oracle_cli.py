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
    save_close_hold_samples,
    save_negative_samples,
    save_samples,
    scan_day,
    scan_day_close_hold,
    scan_day_negative,
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
    include_negative_samples: bool = False,
    negative_ratio: float = 2.0,
    include_close_hold_samples: bool = False,
    close_hold_force_close_seconds: float = 120.0,
    close_hold_max_per_pair: int = 5,
    maturation_conditioned: bool = False,
    maturation_fc_seconds: float = 120.0,
) -> None:
    n_ticks = count_pre_race_ticks(date, data_dir)
    samples = scan_day(
        date, data_dir, config, scorer_dir=scorer_dir,
        predictor_lean_obs=predictor_lean_obs,
        predictor_bundle=predictor_bundle,
        use_race_outcome_predictor=use_race_outcome_predictor,
        use_direction_predictor=use_direction_predictor,
        maturation_conditioned=maturation_conditioned,
        force_close_before_off_seconds=maturation_fc_seconds,
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

    neg_str = ""
    if include_negative_samples:
        # Use a deterministic seed derived from the date so two scans
        # of the same day produce byte-identical negative caches.
        neg_seed = abs(hash(("oracle_negative", date))) & 0x7FFFFFFF
        neg_samples = scan_day_negative(
            date, data_dir, config,
            positive_samples=samples,
            scorer_dir=scorer_dir,
            predictor_lean_obs=predictor_lean_obs,
            predictor_bundle=predictor_bundle,
            use_race_outcome_predictor=use_race_outcome_predictor,
            use_direction_predictor=use_direction_predictor,
            negative_ratio=float(negative_ratio),
            seed=neg_seed,
        )
        save_negative_samples(
            neg_samples, date, data_dir, config, n_ticks, obs_dim,
        )
        neg_str = (
            f" negative_samples={len(neg_samples)} "
            f"(ratio={negative_ratio:.2f})"
        )

    ch_str = ""
    if include_close_hold_samples:
        ch_seed = abs(hash(("oracle_close_hold", date))) & 0x7FFFFFFF
        ch_samples = scan_day_close_hold(
            date, data_dir, config,
            scorer_dir=scorer_dir,
            predictor_lean_obs=predictor_lean_obs,
            predictor_bundle=predictor_bundle,
            use_race_outcome_predictor=use_race_outcome_predictor,
            use_direction_predictor=use_direction_predictor,
            force_close_before_off_seconds=float(
                close_hold_force_close_seconds,
            ),
            max_samples_per_pair=int(close_hold_max_per_pair),
            seed=ch_seed,
        )
        save_close_hold_samples(
            ch_samples, date, data_dir, config, n_ticks, obs_dim,
        )
        n_close = sum(1 for s in ch_samples if s.target_action_class == 0)
        n_hold = sum(1 for s in ch_samples if s.target_action_class == 1)
        ch_str = (
            f" close_hold_samples={len(ch_samples)} "
            f"(close={n_close} hold={n_hold})"
        )

    print(
        f"{date}: samples={len(samples)} ticks={n_ticks} "
        f"density={density:.4f} "
        f"unique_arb_ticks={unique_arb} unique_arb_density={unique_density:.4f}"
        f"{neg_str}{ch_str}{warn}"
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
    scan_p.add_argument(
        "--include-negative-samples", action="store_true",
        help=(
            "BC label augmentation Phase A "
            "(``plans/bc-label-augmentation/``). Also walk the day and "
            "emit (tick, runner) negative-open samples — i.e. samples "
            "that are NOT in the positive arb set — and save them to "
            "``oracle_samples_negative.npz`` next to the positive cache. "
            "Subsampled at ``--negative-ratio`` × positives (default "
            "2x). Downstream BC pretrain targets ActionType.NOOP on "
            "these samples to give NOOP positive gradient and avoid "
            "softmax-decaying every other action class to zero."
        ),
    )
    scan_p.add_argument(
        "--negative-ratio", type=float, default=2.0, metavar="FLOAT",
        help=(
            "Subsample ratio for negative-open samples: emit roughly "
            "``ratio × len(positive_samples)`` negatives per day. "
            "Only consulted when --include-negative-samples is set. "
            "Default 2.0."
        ),
    )
    scan_p.add_argument(
        "--include-close-hold-samples", action="store_true",
        help=(
            "BC label augmentation Phase B "
            "(``plans/bc-label-augmentation/``). For each oracle-"
            "positive open, forward-walk the runner's ATB ladder to "
            "decide whether the hypothetical passive lay would fill "
            "naturally (HOLD/NOOP target) or be force-closed at T-N "
            "(CLOSE target). Synthesises a tiny pool of close-or-"
            "hold decision samples per pair with non-zero position "
            "dims on the opened runner. Cache lands at "
            "``oracle_samples_close_hold.npz`` next to the positive "
            "+ negative caches."
        ),
    )
    scan_p.add_argument(
        "--close-hold-force-close-seconds", type=float,
        default=120.0, metavar="SECONDS",
        help=(
            "Force-close threshold for the close/hold forward walk: "
            "any open pair whose passive hasn't filled by T-SECONDS "
            "is treated as force-close (CLOSE target). Matches the "
            "deployment ``constraints.force_close_before_off_"
            "seconds`` knob. Only consulted when "
            "--include-close-hold-samples is set. Default 120."
        ),
    )
    scan_p.add_argument(
        "--close-hold-max-per-pair", type=int, default=5,
        metavar="N",
        help=(
            "Cap on lifecycle samples emitted per oracle-positive "
            "open. Default 5. Only consulted when "
            "--include-close-hold-samples is set."
        ),
    )
    scan_p.add_argument(
        "--maturation-conditioned", action="store_true",
        help=(
            "Step 0.5 (plans/imitation-first). Emit an OPEN label ONLY "
            "if the passive lay would actually fill before T-fc on the "
            "real future price path (forward-walk), instead of every "
            "placeable spread. Step 0 found the un-gated 'spread "
            "placeable' oracle force-closes ~89%% of its labels on "
            "holdout. Default off = byte-identical 'spread placeable' "
            "oracle (the change stays behind this flag)."
        ),
    )
    scan_p.add_argument(
        "--maturation-fc-seconds", type=float, default=120.0,
        metavar="SECONDS",
        help=(
            "Force-close threshold for the --maturation-conditioned "
            "forward walk. Matches the deployment "
            "constraints.force_close_before_off_seconds. Default 120."
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
            include_negative_samples=bool(args.include_negative_samples),
            negative_ratio=float(args.negative_ratio),
            include_close_hold_samples=bool(
                args.include_close_hold_samples,
            ),
            close_hold_force_close_seconds=float(
                args.close_hold_force_close_seconds,
            ),
            close_hold_max_per_pair=int(args.close_hold_max_per_pair),
            maturation_conditioned=bool(args.maturation_conditioned),
            maturation_fc_seconds=float(args.maturation_fc_seconds),
        )


if __name__ == "__main__":
    main()
