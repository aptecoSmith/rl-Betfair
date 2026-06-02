"""Re-evaluate trained agents from a completed cohort against extra eval days.

Loads each agent's saved weights and runs eval rollouts (no PPO updates)
against a list of held-out days, then writes a new
``reeval_scoreboard.jsonl`` alongside the existing one in the cohort
directory. Per-day naked-pnl variance averages toward zero across
multiple eval days, giving a much cleaner picture of agent skill than
a single eval day's cash result.

Usage:
    python -m tools.reevaluate_cohort \\
        --cohort-dir registry/_phase7_s06_24agent_overnight_1777941123 \\
        --eval-days 2026-05-02 2026-05-03 2026-05-04 \\
        --device cuda \\
        --output reeval_scoreboard_3day.jsonl

To re-evaluate only the top-N agents by composite_score (faster):

    --top-n 24

Wall: ~45s per agent per eval day on cuda. 144 agents x 3 days ~ 5h.
24-top x 3 days ~ 50 min.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("reevaluate_cohort")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort-dir", required=True, type=Path)
    p.add_argument(
        "--eval-days", required=True, nargs="+", metavar="YYYY-MM-DD",
        help="One or more eval-day dates to evaluate against.",
    )
    p.add_argument(
        "--data-dir", default="data/processed", type=Path,
        help="Directory containing YYYY-MM-DD.parquet day files.",
    )
    p.add_argument(
        "--device", default="cuda",
        help="Torch device (cpu, cuda, cuda:N). Default cuda.",
    )
    p.add_argument(
        "--output", default=None,
        help=(
            "Output JSONL filename inside the cohort dir. "
            "Default: reeval_scoreboard_<n>day.jsonl"
        ),
    )
    p.add_argument(
        "--top-n", type=int, default=None,
        help=(
            "Re-evaluate only the top-N agents by existing composite_score. "
            "Default: all agents in the scoreboard."
        ),
    )
    p.add_argument(
        "--filter-agent-ids", nargs="*", default=None,
        help=(
            "Re-evaluate only the listed agent_ids (prefix match supported). "
            "Mutually exclusive with --top-n."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Per-agent eval seed (rolled deterministically per agent).",
    )
    p.add_argument(
        "--argmax-eval", action="store_true",
        help=(
            "Use deterministic (argmax) action selection for eval "
            "rollouts instead of stochastic sampling. Removes £100–£300 "
            "PnL swings caused by action-sampling RNG on identical "
            "weights + identical day. Output rows gain "
            "reeval_mode='argmax' when active."
        ),
    )
    # Predictor-integration data-bridging: if the cohort was trained
    # with predictors on, the cohort row's hyperparameters JSON
    # carries 3 `predictor_*_experiment_id` strings. Re-eval against
    # a different bundle is forbidden by hard_constraints.md §7.
    # When `--predictor-bundle-manifests` is supplied, the tool loads
    # the bundle and calls `bundle.validate_compatibility(hp)` per
    # agent row; mismatch raises `PredictorLoaderError`.
    p.add_argument(
        "--predictor-bundle-manifests", nargs=3, default=None,
        metavar=("CHAMPION", "RANKER", "DIRECTION"),
        help=(
            "Optional. Three paths to the manifest.json files for "
            "champion / ranker / direction predictors. When supplied, "
            "re-eval refuses if the cohort row's recorded "
            "predictor_*_experiment_id doesn't match the live bundle."
        ),
    )
    p.add_argument(
        "--direction-head-manifest", default=None,
        help=(
            "Path to a frozen direction-head dir (e.g. "
            "models/direction_head/sweep_c11). REQUIRED to reeval "
            "cohorts trained with --direction-head-manifest — without "
            "it the policy builds a default direction_prob_head whose "
            "shape mismatches the checkpoint and load_state_dict fails "
            "(0 rows written). 2026-05-28."
        ),
    )
    p.add_argument(
        "--use-race-outcome-predictor", action="store_true",
        help="Match training-time flag — env populates champion+ranker obs.",
    )
    p.add_argument(
        "--use-direction-predictor", action="store_true",
        help="Match training-time flag — env populates direction obs.",
    )
    p.add_argument(
        "--predictor-lean-obs", action="store_true",
        help=(
            "Match training-time flag — env uses the 23-col lean obs "
            "instead of the 143-col firehose. MUST be set to load "
            "checkpoints trained with this flag (otherwise input_proj "
            "shape mismatch on weight load)."
        ),
    )
    p.add_argument(
        "--strategy-mode", default=None,
        choices=["arb", "value_win", "value_each_way"],
        help=(
            "Match training-time strategy_mode. Value modes have a "
            "different obs/action shape (scalping_mode=False) so the "
            "obs_dim differs from arb mode. Defaults to None = "
            "config default (arb)."
        ),
    )
    p.add_argument(
        "--predictor-p-win-back-threshold", type=float, default=0.0,
        help="Match training-time p_win-back action-mask gate.",
    )
    p.add_argument(
        "--predictor-p-win-lay-threshold", type=float, default=1.0,
        help="Match training-time p_win-lay action-mask gate.",
    )
    p.add_argument(
        "--direction-gate-enabled", action="store_true",
        help="Match training-time direction-gate flag.",
    )
    p.add_argument(
        "--race-confidence-threshold", type=float, default=0.0,
        help="Match training-time race-confidence threshold.",
    )
    p.add_argument(
        "--lay-price-max", type=float, default=0.0,
        help=(
            "Match training-time lay-price cap. Refuses OPEN_LAY on "
            "runners whose LTP exceeds the cap. Default 0.0 = "
            "disabled."
        ),
    )
    p.add_argument(
        "--reward-overrides", action="append", default=[],
        metavar="KEY=VALUE",
        help=(
            "Apply a cohort-wide reward_overrides entry. Repeatable. "
            "Useful for retrospective probes — e.g. "
            "'force_close_before_off_seconds=60' to retroactively "
            "activate force-close-at-T-N on already-trained agents."
        ),
    )
    p.add_argument(
        "--market-type-filter", type=str, default=None,
        choices=["WIN", "EACH_WAY", "BOTH", "FREE_CHOICE"],
        help=(
            "Override the env's market_type_filter for reeval. "
            "WIN excludes Each Way markets; EACH_WAY keeps only EW; "
            "BOTH / FREE_CHOICE include everything (default). "
            "Probes whether removing a market class improves results "
            "for agents trained with market_type_filter=BOTH."
        ),
    )
    p.add_argument(
        "--starting-budget", type=float, default=None,
        metavar="GBP",
        help=(
            "Override the env's per-race starting_budget (default reads "
            "from config.yaml: 100.0). Lets you reeval trained agents at "
            "different capital allocations to test scale invariance / "
            "minimum viable deployment budget. The agent observes "
            "budget as a FRACTION of starting_budget so behaviour is "
            "approximately scale-invariant above the MIN_BET_STAKE=£2 "
            "floor (~2 %% of trained £100 budget; binds harder at lower "
            "budgets). See plans/EXPLORATIONS.md for the budget-sweep "
            "design rationale."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
    )
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    # Heavy imports deferred so --help is fast.
    import numpy as np
    import torch

    from agents_v2.discrete_policy import DiscreteLSTMPolicy
    from agents_v2.env_shim import DEFAULT_SCORER_DIR
    from training_v2.cohort.genes import CohortGenes
    from training_v2.cohort.worker import (
        EvalSummary,
        _build_env_for_day,
        _build_eval_bet_records,
        _build_per_agent_reward_overrides,
        _build_per_agent_scalping_overrides,
        _eval_rollout_stats,
        aggregate_eval_summaries,
        scalping_train_config,
    )
    from training_v2.discrete_ppo.rollout import RolloutCollector
    from registry.model_store import ModelStore

    cohort_dir: Path = args.cohort_dir
    if not cohort_dir.exists():
        raise SystemExit(f"--cohort-dir {cohort_dir} not found")
    scoreboard = cohort_dir / "scoreboard.jsonl"
    if not scoreboard.exists():
        raise SystemExit(f"{scoreboard} not found")
    weights_dir = cohort_dir / "weights"
    if not weights_dir.exists():
        raise SystemExit(f"{weights_dir} not found")

    # 2026-05-20: ALWAYS capture per-bet records to parquet during
    # reeval. This is deployment-critical behavioural data: what did
    # the agent back/lay, when, for how much, and how did it resolve.
    # Files land at cohort_dir/bet_logs/reeval_<output_stem>/<agent_id>/<date>.parquet
    # via the model_store helper.
    bet_log_store = ModelStore(
        db_path=cohort_dir / "models.db",
        weights_dir=weights_dir,
        bet_logs_dir=cohort_dir / "bet_logs",
    )

    eval_days: list[str] = list(args.eval_days)
    for ed in eval_days:
        ep = args.data_dir / f"{ed}.parquet"
        if not ep.exists():
            raise SystemExit(f"missing parquet: {ep}")

    output_name = args.output or (
        f"reeval_scoreboard_{len(eval_days)}day.jsonl"
    )
    output_path = cohort_dir / output_name

    # Predictor-bundle for compatibility validation (hard_constraints §7).
    predictor_bundle = None
    if args.predictor_bundle_manifests is not None:
        from predictors import PredictorBundle
        champion_m, ranker_m, direction_m = args.predictor_bundle_manifests
        predictor_bundle = PredictorBundle.from_manifests(
            champion_manifest=champion_m,
            ranker_manifest=ranker_m,
            direction_manifest=direction_m,
        )
        logger.info(
            "loaded predictor bundle: champion=%s ranker=%s direction=%s",
            predictor_bundle.champion_experiment_id,
            predictor_bundle.ranker_experiment_id,
            predictor_bundle.direction_experiment_id,
        )

    rows = []
    for line in scoreboard.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    logger.info(
        "Cohort scoreboard: %d agents", len(rows),
    )

    # Filter the agent list per CLI choice.
    if args.top_n is not None and args.filter_agent_ids:
        raise SystemExit("--top-n and --filter-agent-ids are mutually exclusive")
    if args.filter_agent_ids:
        prefixes = [pfx.lower() for pfx in args.filter_agent_ids]
        rows = [
            r for r in rows
            if any(r["agent_id"].lower().startswith(pfx) for pfx in prefixes)
        ]
        logger.info("Filtered to %d agents by --filter-agent-ids", len(rows))
    if args.top_n is not None:
        rows.sort(
            key=lambda r: -float(r.get("composite_score", r.get("eval_total_reward", 0))),
        )
        rows = rows[: int(args.top_n)]
        logger.info("Re-evaluating top %d by composite_score", len(rows))

    if not rows:
        raise SystemExit("No agents to re-evaluate after filtering")

    cfg = scalping_train_config()
    starting_budget = float(cfg["training"]["starting_budget"])
    # Budget-sweep support (2026-05-20): operator can override the
    # env's starting_budget per reeval to test scale invariance and
    # minimum viable deployment budget. Default None reads from
    # config.yaml (typically £100). Logged loudly so the JSONL
    # caller knows which budget produced its row.
    if args.starting_budget is not None:
        override_budget = float(args.starting_budget)
        logger.info(
            "starting_budget overridden: config=£%.2f -> CLI=£%.2f "
            "(budget-sweep mode; MIN_BET_STAKE=£2 becomes %.1f%% of "
            "budget — invariance breaks if this exceeds ~5%%)",
            starting_budget, override_budget,
            (2.0 / override_budget) * 100.0,
        )
        starting_budget = override_budget
    cfg["training"]["starting_budget"] = starting_budget
    max_runners = int(cfg["training"]["max_runners"])
    if args.strategy_mode is not None:
        cfg["training"]["strategy_mode"] = args.strategy_mode
        logger.info("strategy_mode overridden to %r", args.strategy_mode)

    t0 = time.perf_counter()
    n_done = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for row_idx, row in enumerate(rows):
            agent_id = row["agent_id"]
            model_id = row["model_id"]
            hp = row.get("hyperparameters", {})

            # Predictor-integration: refuse re-eval against a divergent
            # bundle (hard_constraints §7). Cohort rows that pre-date
            # the contract or were trained flag-off carry empty/missing
            # ids and pass through; live-bundle / cohort-row mismatch
            # raises `PredictorLoaderError`.
            if predictor_bundle is not None:
                predictor_bundle.validate_compatibility(hp)

            # Reconstruct the gene dataclass from the stored hp dict.
            try:
                genes = CohortGenes(**{
                    k: v for k, v in hp.items()
                    if k in CohortGenes.__dataclass_fields__
                })
            except Exception as e:
                logger.warning(
                    "[%d/%d] %s: skipping — could not rebuild genes (%s)",
                    row_idx + 1, len(rows), agent_id[:12], e,
                )
                continue

            weights_path = weights_dir / f"{model_id}.pt"
            if not weights_path.exists():
                logger.warning(
                    "[%d/%d] %s: no weights file at %s",
                    row_idx + 1, len(rows), agent_id[:12], weights_path,
                )
                continue

            agent_seed = (int(args.seed) + row_idx * 1_000_003) & 0x7FFFFFFF
            torch.manual_seed(agent_seed)
            np.random.seed(agent_seed)
            if str(args.device).startswith("cuda"):
                torch.cuda.manual_seed_all(agent_seed)

            # Per-agent reward + scalping overrides. The cohort recorded
            # which genes were enabled implicitly via non-default values
            # in hp; for reeval we treat ALL gene values as-is (every
            # field is enabled because the saved row carries actual
            # per-agent values, not cohort-default placeholders).
            enabled_set = frozenset(CohortGenes.__dataclass_fields__)
            # Parse --reward-overrides KEY=VALUE pairs into a dict to
            # layer on top of the per-agent gene overrides. Lets the
            # operator retroactively activate env mechanisms that
            # weren't active during the cohort's training (e.g.
            # force_close_before_off_seconds=60 to probe the T-N
            # close-sweep on already-trained agents).
            cli_reward_overrides: dict = {}
            for item in (args.reward_overrides or []):
                if "=" not in item:
                    raise SystemExit(
                        f"--reward-overrides expects key=value, got {item!r}",
                    )
                k, _, raw = item.partition("=")
                k = k.strip()
                raw = raw.strip()
                lo = raw.lower()
                if lo in ("true", "1"):
                    cli_reward_overrides[k] = True
                elif lo in ("false", "0"):
                    cli_reward_overrides[k] = False
                else:
                    try:
                        cli_reward_overrides[k] = float(raw)
                    except ValueError:
                        cli_reward_overrides[k] = raw
            per_agent_reward_overrides = _build_per_agent_reward_overrides(
                cohort_overrides=cli_reward_overrides or None,
                genes=genes,
                enabled_set=enabled_set,
            )
            per_agent_scalping_overrides = _build_per_agent_scalping_overrides(
                genes=genes,
                enabled_set=enabled_set,
            )

            # Build a sample env to size the policy. Use the FIRST eval
            # day for the sizing — env.obs_dim and shim.action_space are
            # day-agnostic. Forward the predictor flags so the env's
            # obs_dim matches the cohort's training-time setup;
            # otherwise the policy's input_proj layer won't match the
            # saved weights.
            try:
                env, shim = _build_env_for_day(
                    day_str=eval_days[0], data_dir=args.data_dir, cfg=cfg,
                    scorer_dir=DEFAULT_SCORER_DIR,
                    reward_overrides=per_agent_reward_overrides,
                    scalping_overrides=per_agent_scalping_overrides,
                    predictor_bundle=predictor_bundle,
                    use_race_outcome_predictor=bool(args.use_race_outcome_predictor),
                    use_direction_predictor=bool(args.use_direction_predictor),
                    predictor_lean_obs=bool(args.predictor_lean_obs),
                    predictor_p_win_back_threshold=float(args.predictor_p_win_back_threshold),
                    predictor_p_win_lay_threshold=float(args.predictor_p_win_lay_threshold),
                    direction_gate_enabled=bool(args.direction_gate_enabled),
                    race_confidence_threshold=float(args.race_confidence_threshold),
                    lay_price_max=float(args.lay_price_max),
                    market_type_filter=args.market_type_filter,
                )
            except Exception as e:
                logger.warning(
                    "[%d/%d] %s: failed to build env (%s)",
                    row_idx + 1, len(rows), agent_id[:12], e,
                )
                continue

            policy = DiscreteLSTMPolicy(
                obs_dim=shim.obs_dim,
                action_space=shim.action_space,
                hidden_size=int(genes.hidden_size),
                frozen_direction_head_path=(
                    args.direction_head_manifest
                    if args.direction_head_manifest else None
                ),
            )
            try:
                state = torch.load(
                    weights_path, weights_only=True, map_location="cpu",
                )
                if isinstance(state, dict) and "weights" in state:
                    state = state["weights"]
                policy.load_state_dict(state, strict=True)
            except Exception as e:
                logger.warning(
                    "[%d/%d] %s: failed to load weights (%s)",
                    row_idx + 1, len(rows), agent_id[:12], e,
                )
                continue
            policy.to(args.device)
            policy.eval()

            # Eval each day
            per_day_summaries: list[EvalSummary] = []
            per_day_peaks: list[dict] = []
            for ed in eval_days:
                eval_t0 = time.perf_counter()
                try:
                    _, eval_shim = _build_env_for_day(
                        day_str=ed, data_dir=args.data_dir, cfg=cfg,
                        scorer_dir=DEFAULT_SCORER_DIR,
                        reward_overrides=per_agent_reward_overrides,
                        scalping_overrides=per_agent_scalping_overrides,
                        predictor_bundle=predictor_bundle,
                        use_race_outcome_predictor=bool(args.use_race_outcome_predictor),
                        use_direction_predictor=bool(args.use_direction_predictor),
                        predictor_lean_obs=bool(args.predictor_lean_obs),
                        predictor_p_win_back_threshold=float(args.predictor_p_win_back_threshold),
                        predictor_p_win_lay_threshold=float(args.predictor_p_win_lay_threshold),
                        direction_gate_enabled=bool(args.direction_gate_enabled),
                        race_confidence_threshold=float(args.race_confidence_threshold),
                        lay_price_max=float(args.lay_price_max),
                        market_type_filter=args.market_type_filter,
                    )
                except Exception as e:
                    logger.warning(
                        "%s: failed to build env for %s (%s)",
                        agent_id[:12], ed, e,
                    )
                    continue
                eval_collector = RolloutCollector(
                    shim=eval_shim, policy=policy, device=str(args.device),
                )
                eval_batch = eval_collector.collect_episode(
                    deterministic=args.argmax_eval,
                )
                partial = _eval_rollout_stats(
                    batch=eval_batch,
                    last_info=eval_collector.last_info,
                    action_space=eval_shim.action_space,
                )

                # 2026-05-20: capture per-bet records to parquet.
                # Run-id namespaces this reeval's logs from any
                # in-sample bet_logs that may exist alongside.
                run_id = f"reeval_{output_path.stem}_{agent_id}"
                try:
                    bet_records = _build_eval_bet_records(
                        env=eval_shim.env,
                        day=eval_shim.env.day,
                        starting_budget=float(
                            eval_shim.env.starting_budget
                        ),
                    )
                    for r in bet_records:
                        r.run_id = run_id
                    if bet_records:
                        bet_log_store.write_bet_logs_parquet(
                            run_id=run_id, date=ed, records=bet_records,
                        )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "%s [%s]: bet-log capture failed (%s)",
                        agent_id[:12], ed, e,
                    )

                # 2026-05-20: read peak deployment-economics metrics
                # from the env's per-step trackers (set during
                # rollout, surfaced on info dict).
                last_info = eval_collector.last_info or {}
                peak_liability_day = float(
                    last_info.get("peak_open_liability", 0.0)
                )
                peak_drawdown_day = float(
                    last_info.get("peak_drawdown_from_high", 0.0)
                )
                per_day_summaries.append(EvalSummary(
                    eval_day=ed,
                    total_reward=partial.total_reward,
                    day_pnl=partial.day_pnl,
                    n_steps=partial.n_steps,
                    bet_count=partial.bet_count,
                    winning_bets=partial.winning_bets,
                    bet_precision=partial.bet_precision,
                    pnl_per_bet=partial.pnl_per_bet,
                    early_picks=partial.early_picks,
                    profitable=partial.profitable,
                    action_histogram=partial.action_histogram,
                    arbs_completed=partial.arbs_completed,
                    arbs_naked=partial.arbs_naked,
                    arbs_closed=partial.arbs_closed,
                    arbs_force_closed=partial.arbs_force_closed,
                    arbs_stop_closed=partial.arbs_stop_closed,
                    arbs_target_pnl_refused=partial.arbs_target_pnl_refused,
                    pairs_opened=partial.pairs_opened,
                    locked_pnl=partial.locked_pnl,
                    naked_pnl=partial.naked_pnl,
                    closed_pnl=partial.closed_pnl,
                    force_closed_pnl=partial.force_closed_pnl,
                    stop_closed_pnl=partial.stop_closed_pnl,
                    wall_time_sec=time.perf_counter() - eval_t0,
                ))
                # Parallel tracking of deployment-economics peaks
                # per day (kept outside EvalSummary to avoid touching
                # the dataclass schema).
                per_day_peaks.append({
                    "eval_day": ed,
                    "peak_open_liability": peak_liability_day,
                    "peak_drawdown_from_high": peak_drawdown_day,
                })
                logger.info(
                    "[%d/%d] %s [%s] reward=%+.2f pnl=%+.2f bets=%d "
                    "arbs=%d/%d locked=%+.2f naked=%+.2f wall=%.1fs",
                    row_idx + 1, len(rows), agent_id[:12], ed,
                    partial.total_reward, partial.day_pnl, partial.bet_count,
                    partial.arbs_completed, partial.arbs_naked,
                    partial.locked_pnl, partial.naked_pnl,
                    per_day_summaries[-1].wall_time_sec,
                )

            if not per_day_summaries:
                continue
            agg = aggregate_eval_summaries(per_day_summaries)
            n_completed = agg.arbs_completed + agg.arbs_closed
            mat_rate = (
                n_completed / agg.pairs_opened if agg.pairs_opened > 0 else 0.0
            )

            new_row = {
                "schema": "v2_cohort_reeval",
                "agent_id": agent_id,
                "model_id": model_id,
                "architecture_name": row.get("architecture_name"),
                "generation": row.get("generation"),
                "agent_idx": row.get("agent_idx"),
                "hyperparameters": hp,
                "training_days": row.get("training_days"),
                "original_eval_days": row.get("eval_days") or [row.get("eval_day")],
                "reeval_days": list(eval_days),
                # Aggregated reeval metrics (mean across reeval_days)
                "reeval_total_reward": agg.total_reward,
                "reeval_day_pnl": agg.day_pnl,
                "reeval_bet_count": agg.bet_count,
                "reeval_bet_precision": agg.bet_precision,
                "reeval_arbs_completed": agg.arbs_completed,
                "reeval_arbs_closed": agg.arbs_closed,
                "reeval_arbs_naked": agg.arbs_naked,
                "reeval_arbs_force_closed": agg.arbs_force_closed,
                "reeval_arbs_stop_closed": agg.arbs_stop_closed,
                "reeval_pairs_opened": agg.pairs_opened,
                "reeval_locked_pnl": agg.locked_pnl,
                "reeval_naked_pnl": agg.naked_pnl,
                "reeval_closed_pnl": agg.closed_pnl,
                "reeval_force_closed_pnl": agg.force_closed_pnl,
                "reeval_stop_closed_pnl": agg.stop_closed_pnl,
                "reeval_maturation_rate": mat_rate,
                "reeval_mode": "argmax" if args.argmax_eval else "stochastic",
                # 2026-05-20 deployment-economics telemetry.
                # peak_open_liability: max reserved capital observed
                # at any tick during the day (= what bank must
                # cover for ONE concurrent race; multiply by typical
                # race-clock concurrency for total bank estimate).
                # peak_drawdown_from_high: worst trough from running
                # high-water mark of day_pnl (bank also covers this).
                # Aggregates: max across the eval window.
                "reeval_peak_open_liability_max": max(
                    (p["peak_open_liability"] for p in per_day_peaks),
                    default=0.0,
                ),
                "reeval_peak_drawdown_from_high_max": max(
                    (p["peak_drawdown_from_high"] for p in per_day_peaks),
                    default=0.0,
                ),
                "reeval_peak_open_liability_mean": (
                    sum(p["peak_open_liability"] for p in per_day_peaks)
                    / len(per_day_peaks)
                ) if per_day_peaks else 0.0,
                "reeval_peak_drawdown_from_high_mean": (
                    sum(p["peak_drawdown_from_high"] for p in per_day_peaks)
                    / len(per_day_peaks)
                ) if per_day_peaks else 0.0,
                "reeval_per_day_peaks": per_day_peaks,
                # Per-day breakdown for variance inspection
                "reeval_per_day": [
                    {
                        "eval_day": s.eval_day,
                        "total_reward": s.total_reward,
                        "day_pnl": s.day_pnl,
                        "bet_count": s.bet_count,
                        "arbs_completed": s.arbs_completed,
                        "arbs_naked": s.arbs_naked,
                        "arbs_closed": s.arbs_closed,
                        "arbs_force_closed": s.arbs_force_closed,
                        "arbs_stop_closed": s.arbs_stop_closed,
                        "pairs_opened": s.pairs_opened,
                        "locked_pnl": s.locked_pnl,
                        "naked_pnl": s.naked_pnl,
                        "closed_pnl": s.closed_pnl,
                        "force_closed_pnl": s.force_closed_pnl,
                        "stop_closed_pnl": s.stop_closed_pnl,
                    }
                    for s in agg.per_day
                ],
            }
            out_f.write(json.dumps(new_row) + "\n")
            out_f.flush()

            n_done += 1
            wall_so_far = time.perf_counter() - t0
            avg_per_agent = wall_so_far / n_done
            remaining = (len(rows) - n_done) * avg_per_agent
            logger.info(
                "[%d/%d] %s AGG: pnl=%+.2f mr=%.3f locked=%+.2f naked=%+.2f "
                "(elapsed %.1fmin, ETA %.1fmin)",
                row_idx + 1, len(rows), agent_id[:12],
                agg.day_pnl, mat_rate, agg.locked_pnl, agg.naked_pnl,
                wall_so_far / 60.0, remaining / 60.0,
            )

    logger.info(
        "Re-eval complete in %.1fs. Wrote %d rows to %s",
        time.perf_counter() - t0, n_done, output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
