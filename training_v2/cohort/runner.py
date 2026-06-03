"""GA cohort orchestrator — Phase 3, Session 03 deliverable.

Drives ``n_agents`` workers across ``n_generations`` of training,
evaluating each agent on a held-out day, breeding the next generation
from the elite half of the previous, and writing a registry-shaped
scoreboard the v1 UI consumes unchanged (during the comparison window).

Concurrency: **sequential** for Session 03 (session prompt §3
"Concurrency for Session 03 = sequential"). v1's
``ThreadPoolExecutor`` worker pool is fragile at high N (OOM on a
shared GPU) and is the wrong hill for the first-run scaffolding. A
follow-on plan adds the worker pool if Session 04's wall time
demands it.

CLI:

    python -m training_v2.cohort.runner \\
        --n-agents 4 --generations 2 --days 7 \\
        --device cuda --seed 42 \\
        --output-dir registry/v2_dryrun_$(date +%s)

Output layout (mirrors v1):

    registry/{output-dir}/models.db
    registry/{output-dir}/weights/{model_id}.pt
    registry/{output-dir}/scoreboard.jsonl       (one row per agent)

The JSONL scoreboard duplicates the SQLite-stored data in a flat
shape the UI / findings.md aggregator can read without an SQL query;
v1 emits the same.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from registry.model_store import ModelStore
from training_v2.cohort.events import (
    WebSocketBroadcastServer,
    cohort_complete_event,
    cohort_started_event,
)
from training_v2.cohort.genes import (
    PHASE5_GENE_NAMES,
    CohortGenes,
    assert_in_range,
    crossover,
    mutate,
    sample_genes,
)
from training_v2.cohort.pbt import (
    PbtConfig,
    breed_pbt,
    init_pbt_population,
    make_rotations,
)
from training_v2.cohort.batched_worker import train_cluster_batched
from training_v2.cohort.multiproc_worker import (
    train_cluster_multiproc,
    prebuild_feature_cache,
    save_shared_cache_per_day,
    prebuild_static_obs_cache,
    model_store_paths,
    make_pool,
)
from training_v2.cohort.worker import (
    AgentResult,
    _build_env_for_day,
    _eval_rollout_stats,
    scalping_train_config,
    train_one_agent,
)
from training_v2.discrete_ppo.batched_rollout import cluster_agents_by_arch
from training_v2.discrete_ppo.rollout import RolloutCollector
from training_v2.discrete_ppo.train import select_days
from agents_v2.discrete_policy import DiscreteLSTMPolicy
from agents_v2.env_shim import DEFAULT_SCORER_DIR

# Pre-flight cache schema check (2026-05-24). The direction-label
# cache stores ``obs_schema_version`` from this module-level constant;
# ``direction_label_scan`` imports it from ``env.betfair_env``, so we
# read from there to keep the source of truth single-rooted.
from training_v2.direction_label_scan import (
    OBS_SCHEMA_VERSION as _DIRECTION_OBS_SCHEMA_VERSION,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed"


logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────


COMPOSITE_SCORE_MODE_TOTAL_REWARD = "total_reward"
COMPOSITE_SCORE_MODE_LOCKED_WEIGHTED = "locked_weighted"
# scalping-tight-naked-variance Phase 2A (2026-05-15). Variance-aware
# selection: surfaces low-σ_naked agents at the GA-selection step.
# Formula: ``mean_locked - 0.5 × σ_naked_daily + 0.25 × mean_naked``.
# Requires multi-day eval (``n_eval_days >= 2``); falls back to
# ``locked_weighted`` otherwise per hard_constraints §15.
COMPOSITE_SCORE_MODE_TIGHT_VARIANCE = "tight_variance"
# scalping-tight-naked-variance Phase 3 corrective (2026-05-16). Variance-
# aware Sharpe-style selection that NEVER reads naked-sign (mean_naked).
# Formula: ``mean_locked / (1 + naked_std_daily)``. Picked after the
# Phase 3b post-hoc analysis showed:
#   - score_a (pure_locked) picks the best held-out cohort because
#     locked is structurally signed (matched-arb count × spread cost).
#   - tight_variance's ``+ 0.25 × mean_naked`` term made the GA chase
#     in-sample naked luck — agents with in-sample +£342 reverted to
#     -£170 on 3-day held-out (cohort tnv_raceconf 1778852093,
#     agent ad42a47b).
# This mode keeps the variance-penalty discipline (Sharpe-like
# denominator) without reading naked-sign. Falls back to
# locked_weighted when ``len(per_day) < 2`` (σ undefined).
COMPOSITE_SCORE_MODE_LOCKED_PER_STD = "locked_per_std"
# scalping-tight-naked-variance tnv3 corrective (2026-05-17). The tnv2
# cohort revealed that ``locked_per_std`` selects for high-volume-open
# agents that pay £-for-£ in force_close cost. Locked goes up, naked_std
# stays tight, but ``day_pnl`` (the actual cash) goes negative because
# fc cost cancels the locked floor.
#
# Formula: ``day_pnl / (1 + naked_std_daily)``. ``day_pnl`` naturally
# includes force_close cost in its sum (= locked + naked + closed + fc)
# so the GA selects against agents that lose money to fc. Trade-off:
# re-introduces naked-sign reading at selection (which bit tnv1 at
# 3 in-sample-eval days), but at 10+ eval days the noise drops
# √(10/3) ≈ 1.8× — should be enough.
COMPOSITE_SCORE_MODE_DAY_PNL_PER_STD = "day_pnl_per_std"
# robust-phenotype R1 (2026-05-19). Sortino-shaped selector. The E3 full
# cohort surfaced agents with similar mean pnl but wildly different
# per-day spans — 571f6eda mean +£41 (worst day -£105, best +£313) vs
# 850522b9 mean +£65 (worst -£20, best +£160). The day_pnl_per_std
# selector treats positive and negative variance the same in the
# denominator, so it can't distinguish the left-tail-truncated 850522b9
# shape from the symmetric-high-variance 571f6eda shape.
#
# Formula (additive form per hard_constraints §4): score =
# mean(day_pnl) - λ × downside_deviation. downside_deviation =
# sqrt(mean(min(0, day_pnl)²)) — penalises ONLY sub-zero days.
# Default λ = 1.0; lives as a CLI flag. Mean of positive days is
# untouched; agents free to chase upside without selection pressure
# capping it.
#
# See plans/robust-phenotype/{purpose.md, hard_constraints.md §4}.
COMPOSITE_SCORE_MODE_SORTINO = "sortino"
COMPOSITE_SCORE_MODES = (
    COMPOSITE_SCORE_MODE_TOTAL_REWARD,
    COMPOSITE_SCORE_MODE_LOCKED_WEIGHTED,
    COMPOSITE_SCORE_MODE_TIGHT_VARIANCE,
    COMPOSITE_SCORE_MODE_LOCKED_PER_STD,
    COMPOSITE_SCORE_MODE_DAY_PNL_PER_STD,
    COMPOSITE_SCORE_MODE_SORTINO,
)
LOCKED_WEIGHTED_NAKED_COEFFICIENT = 0.25
# robust-phenotype R1 — default penalty weight on downside deviation.
# 1.0 = a £1 in worst-day downside cancels £1 of mean pnl. Tuneable
# via --sortino-lambda CLI flag.
SORTINO_DEFAULT_LAMBDA = 1.0
# Module-level mutable so the CLI flag can override without threading
# through every _composite_score call site. Set by the CLI handler
# (search for ``_SORTINO_LAMBDA = ``) before run_cohort() is invoked.
_SORTINO_LAMBDA: float = SORTINO_DEFAULT_LAMBDA
# scalping-tight-naked-variance hard_constraints.md §5 — module-level
# constants for grep-ability. Mirror values from the report tool
# (``tools/build_naked_variance_report.py``).
TIGHT_VARIANCE_VOL_COEF = 0.5
TIGHT_VARIANCE_NAKED_COEF = 0.25


def _composite_score(
    eval_stats, maturation_bonus_weight: float,
    composite_score_mode: str = COMPOSITE_SCORE_MODE_TOTAL_REWARD,
    sortino_lambda: float | None = None,
) -> float:
    """GA selection score. See module-level constants for modes.

    ``sortino_lambda``: only consumed when
    ``composite_score_mode == 'sortino'``. ``None`` (the default)
    reads the module-level ``_SORTINO_LAMBDA`` (settable via the
    ``--sortino-lambda`` CLI flag); passing an explicit value
    overrides for direct callers (tests).
    """
    if sortino_lambda is None:
        sortino_lambda = _SORTINO_LAMBDA
    """GA selection score.

    Two modes:

    - ``total_reward`` (default, byte-identical to pre-plan):
      ``total_reward + w × (matured + closed)`` where ``w`` is the
      ``maturation_bonus_weight`` knob. ``w = 0.0`` → score equals
      ``total_reward``. ``w > 0`` rewards both rate AND volume of
      matured / agent-closed pairs at a £-scale interpretable on the
      same scale as ``total_reward`` (~£100s per eval). Force-closed
      pairs are excluded — env bail-outs, not skill.

    - ``locked_weighted`` (scalping-locked-fitness-and-age-obs plan):
      ``locked_pnl + 0.25 × naked_pnl``. The 0.25 weight is locked
      (hard_constraints §9) — calibrated against the cross-agent
      naked/locked variance ratio observed in the predecessor
      lay-quality-gate cohort (σ_naked / σ_locked ≈ 4-5x). Surfaces
      structural locked-floor agents at the GA selection step
      instead of letting per-eval naked-pnl noise dominate. Ignores
      ``maturation_bonus_weight`` — this mode is a single-formula
      replacement, not an additive modification.

    The scoreboard row's ``composite_score`` field carries this exact
    value so downstream tooling sees what the GA actually selected on.
    """
    if composite_score_mode == COMPOSITE_SCORE_MODE_LOCKED_WEIGHTED:
        return (
            float(eval_stats.locked_pnl)
            + LOCKED_WEIGHTED_NAKED_COEFFICIENT * float(eval_stats.naked_pnl)
        )
    if composite_score_mode == COMPOSITE_SCORE_MODE_TIGHT_VARIANCE:
        # scalping-tight-naked-variance Phase 2A. Surface low-σ_naked
        # agents at the GA-selection step. ``eval_stats.per_day`` is
        # populated when the cohort runner uses ``--n-eval-days N>1``.
        # When ``len(per_day) < 2`` σ is undefined → fall back to
        # ``locked_weighted`` per hard_constraints §15.
        per_day = list(getattr(eval_stats, "per_day", []) or [])
        if len(per_day) < 2:
            return (
                float(eval_stats.locked_pnl)
                + LOCKED_WEIGHTED_NAKED_COEFFICIENT
                * float(eval_stats.naked_pnl)
            )
        naked_daily = [float(d.naked_pnl) for d in per_day]
        n = len(naked_daily)
        mean = sum(naked_daily) / n
        # Sample stddev (ddof=1) — matches the report tool.
        variance = sum((x - mean) ** 2 for x in naked_daily) / (n - 1)
        naked_std = variance ** 0.5
        return (
            float(eval_stats.locked_pnl)
            - TIGHT_VARIANCE_VOL_COEF * naked_std
            + TIGHT_VARIANCE_NAKED_COEF * float(eval_stats.naked_pnl)
        )
    if composite_score_mode == COMPOSITE_SCORE_MODE_LOCKED_PER_STD:
        # scalping-tight-naked-variance Phase 3 corrective. Sharpe-like
        # selection on locked floor only — never reads naked-sign.
        # The +1 in the denominator stabilises the score when σ is
        # near zero (degenerate cohort) and matches the report tool's
        # score_b/score_c convention.
        per_day = list(getattr(eval_stats, "per_day", []) or [])
        if len(per_day) < 2:
            return (
                float(eval_stats.locked_pnl)
                + LOCKED_WEIGHTED_NAKED_COEFFICIENT
                * float(eval_stats.naked_pnl)
            )
        naked_daily = [float(d.naked_pnl) for d in per_day]
        n = len(naked_daily)
        mean = sum(naked_daily) / n
        variance = sum((x - mean) ** 2 for x in naked_daily) / (n - 1)
        naked_std = variance ** 0.5
        return float(eval_stats.locked_pnl) / (1.0 + naked_std)
    if composite_score_mode == COMPOSITE_SCORE_MODE_DAY_PNL_PER_STD:
        # scalping-tight-naked-variance tnv3 corrective. Like
        # locked_per_std but uses ``day_pnl`` as the numerator so the
        # GA selects against agents that pay force_close cost (a hidden
        # drag in tnv2's locked-only metric). Fallback to
        # locked_weighted when n_eval_days < 2 per hard_constraints §15.
        per_day = list(getattr(eval_stats, "per_day", []) or [])
        if len(per_day) < 2:
            return (
                float(eval_stats.locked_pnl)
                + LOCKED_WEIGHTED_NAKED_COEFFICIENT
                * float(eval_stats.naked_pnl)
            )
        naked_daily = [float(d.naked_pnl) for d in per_day]
        n = len(naked_daily)
        mean = sum(naked_daily) / n
        variance = sum((x - mean) ** 2 for x in naked_daily) / (n - 1)
        naked_std = variance ** 0.5
        return float(eval_stats.day_pnl) / (1.0 + naked_std)
    if composite_score_mode == COMPOSITE_SCORE_MODE_SORTINO:
        # robust-phenotype R1 (2026-05-19). Sortino-shaped score:
        # ``score = mean(day_pnl) - λ × downside_deviation`` where
        # ``downside_deviation = sqrt(mean(min(0, day_pnl)²))``.
        # Penalises ONLY sub-zero days; positive-day variance is free.
        #
        # Fallback to locked_weighted when n_eval_days < 2 — Sortino
        # needs at least 2 days to estimate downside deviation.
        # Matches the other Sharpe-shaped selectors' fallback contract.
        per_day = list(getattr(eval_stats, "per_day", []) or [])
        if len(per_day) < 2:
            return (
                float(eval_stats.locked_pnl)
                + LOCKED_WEIGHTED_NAKED_COEFFICIENT
                * float(eval_stats.naked_pnl)
            )
        day_pnls = [float(d.day_pnl) for d in per_day]
        n = len(day_pnls)
        mean = sum(day_pnls) / n
        # Downside deviation: RMS of negative-only day_pnls. min(0, x)
        # zeroes positive days so they contribute nothing.
        downside_sq = sum(min(0.0, x) ** 2 for x in day_pnls) / n
        downside_dev = downside_sq ** 0.5
        return mean - float(sortino_lambda) * downside_dev
    n_completed = (
        int(eval_stats.arbs_completed) + int(eval_stats.arbs_closed)
    )
    return float(eval_stats.total_reward) + float(
        maturation_bonus_weight,
    ) * n_completed


# ── Early-stop helpers (scalping-tight-naked-variance tnv3, 2026-05-17) ──
# Thresholds for "improvement" on each of the three signals. Tuned so the
# default ``early_stop_patience=3`` only fires when the cohort has clearly
# exhausted exploration on all three axes simultaneously.
_EARLY_STOP_STD_IMPROVEMENT_THRESHOLD = 5.0   # £/day
_EARLY_STOP_COMPOSITE_REL_THRESHOLD = 0.01    # 1 %
_EARLY_STOP_BETA_REL_THRESHOLD = 0.10         # 10 %


def _gen_early_stop_stats(
    results: list,
    composite_score_mode: str,
    maturation_bonus_weight: float,
) -> dict:
    """Compute the three early-stop signals for a generation's results.

    Returns ``{median_std, median_composite, beta_med}``.
    """
    import statistics as _stats
    stds: list[float] = []
    composites: list[float] = []
    betas: list[float] = []
    for r in results:
        per_day = list(getattr(r.eval, "per_day", []) or [])
        if len(per_day) >= 2:
            nakeds = [float(d.naked_pnl) for d in per_day]
            n = len(nakeds)
            mean = sum(nakeds) / n
            var = sum((x - mean) ** 2 for x in nakeds) / (n - 1)
            stds.append(var ** 0.5)
        composites.append(_composite_score(
            r.eval, maturation_bonus_weight, composite_score_mode,
        ))
        try:
            betas.append(float(r.genes.naked_variance_penalty_beta))
        except AttributeError:
            betas.append(0.0)
    return {
        "median_std": _stats.median(stds) if stds else float("nan"),
        "median_composite": _stats.median(composites) if composites else float("nan"),
        "beta_med": _stats.median(betas) if betas else 0.0,
    }


def _early_stop_improved(history: list[dict]) -> list[str]:
    """Did the most-recent gen improve on the best of all prior gens
    on ANY of (median_std, median_composite, beta_med)?

    Returns a list naming the axes that improved (empty list = no
    improvement = stall counter advances).

    ``median_std`` improves when it DECREASES (tighter variance).
    ``median_composite`` improves when it INCREASES (better fitness).
    ``beta_med`` improves when it CHANGES by ≥ rel threshold (either
    direction — the GA might be exploring up or down).
    """
    if len(history) < 2:
        return ["initial"]  # not a stall on the first gen with data
    current = history[-1]
    prior = history[:-1]
    improved: list[str] = []
    # median_std: best so far is the MIN
    best_std = min(h["median_std"] for h in prior)
    if current["median_std"] + _EARLY_STOP_STD_IMPROVEMENT_THRESHOLD <= best_std:
        improved.append("median_std")
    # median_composite: best so far is the MAX
    best_comp = max(h["median_composite"] for h in prior)
    if (
        abs(best_comp) > 1e-9
        and (current["median_composite"] - best_comp) / abs(best_comp) >= _EARLY_STOP_COMPOSITE_REL_THRESHOLD
    ):
        improved.append("median_composite")
    elif current["median_composite"] > best_comp + 0.01:  # absolute fallback for near-zero
        improved.append("median_composite")
    # beta_med: improves on EITHER direction by ≥ rel threshold vs ANY
    # prior gen's value (the GA's exploration may revisit lower β too).
    prior_best_beta = prior[-1]["beta_med"]
    if prior_best_beta > 1e-9:
        rel_change = abs(current["beta_med"] - prior_best_beta) / prior_best_beta
        if rel_change >= _EARLY_STOP_BETA_REL_THRESHOLD:
            improved.append("beta_med")
    return improved


def _evaluate_agents_on_monitor_days(
    *,
    top_results: list[AgentResult],
    monitor_days: list[str],
    data_dir: Path,
    cfg: dict,
    device: str,
    reward_overrides: dict | None,
    predictor_bundle: object | None,
    use_race_outcome_predictor: bool,
    use_direction_predictor: bool,
    predictor_lean_obs: bool,
    predictor_p_win_back_threshold: float,
    predictor_p_win_lay_threshold: float,
    direction_gate_enabled: bool,
    race_confidence_threshold: float,
    lay_price_max: float,
    feature_cache: dict[str, list] | None = None,
) -> dict:
    """In-training overfit monitor (2026-05-22 anti-overfit, follow-on).

    Re-evaluate the top-K trained agents from this generation on a
    sealed ``monitor_days`` set. The result is NEVER used for selection
    or breeding — it is a tripwire metric that surfaces gen-on-gen when
    the in-training composite_score is rising on the rotating eval but
    monitor performance flattens or regresses (the classic overfit
    fingerprint).

    Each agent's weights are loaded from disk via ``r.weights_path``
    (set inside ``train_one_agent`` after model_store.save_weights).
    Architecture / hidden_size come from ``r.genes``.

    Returns a dict of the form::

        {
            "per_agent": [
                {"agent_id": str, "monitor_pnl_total": float,
                 "monitor_pnl_mean": float, "monitor_per_day": [{...}]},
                ...
            ],
            "cohort_monitor_pnl_mean": float,  # mean across top-K agents' totals
        }

    Cost: ``top_k × len(monitor_days)`` extra rollouts per generation.
    For top_k=3, monitor=14 days, gens=5: ~210 rollouts (~1-3h GPU).
    """
    import torch
    per_agent: list[dict] = []
    for r in top_results:
        if not r.weights_path:
            logger.warning(
                "monitor-eval: agent %s has empty weights_path; skipping",
                r.agent_id[:12],
            )
            continue
        # Build fresh env+shim from the FIRST monitor day just to size the
        # policy's obs space; build policy; load weights; then loop days.
        try:
            env0, shim0 = _build_env_for_day(
                day_str=monitor_days[0], data_dir=data_dir, cfg=cfg,
                scorer_dir=DEFAULT_SCORER_DIR,
                reward_overrides=reward_overrides,
                predictor_bundle=predictor_bundle,
                use_race_outcome_predictor=use_race_outcome_predictor,
                use_direction_predictor=use_direction_predictor,
                predictor_lean_obs=predictor_lean_obs,
                predictor_p_win_back_threshold=predictor_p_win_back_threshold,
        predictor_p_win_back_max_threshold=predictor_p_win_back_max_threshold,
                predictor_p_win_lay_threshold=predictor_p_win_lay_threshold,
                direction_gate_enabled=direction_gate_enabled,
                race_confidence_threshold=race_confidence_threshold,
                lay_price_max=lay_price_max,
                feature_cache=feature_cache,
            )
        except Exception as e:
            logger.warning(
                "monitor-eval: env build failed for %s on %s (%s); skipping agent",
                r.agent_id[:12], monitor_days[0], e,
            )
            continue
        policy = DiscreteLSTMPolicy(
            obs_dim=shim0.obs_dim,
            action_space=shim0.action_space,
            hidden_size=int(r.genes.hidden_size),
        )
        try:
            state = torch.load(
                r.weights_path, weights_only=True, map_location="cpu",
            )
            if isinstance(state, dict) and "weights" in state:
                state = state["weights"]
            policy.load_state_dict(state, strict=True)
        except Exception as e:
            logger.warning(
                "monitor-eval: load_state_dict failed for %s (%s); skipping",
                r.agent_id[:12], e,
            )
            continue
        policy.to(device)
        policy.eval()

        per_day: list[dict] = []
        total_pnl = 0.0
        for ed in monitor_days:
            try:
                _, shim = _build_env_for_day(
                    day_str=ed, data_dir=data_dir, cfg=cfg,
                    scorer_dir=DEFAULT_SCORER_DIR,
                    reward_overrides=reward_overrides,
                    predictor_bundle=predictor_bundle,
                    use_race_outcome_predictor=use_race_outcome_predictor,
                    use_direction_predictor=use_direction_predictor,
                    predictor_lean_obs=predictor_lean_obs,
                    predictor_p_win_back_threshold=predictor_p_win_back_threshold,
        predictor_p_win_back_max_threshold=predictor_p_win_back_max_threshold,
                    predictor_p_win_lay_threshold=predictor_p_win_lay_threshold,
                    direction_gate_enabled=direction_gate_enabled,
                    race_confidence_threshold=race_confidence_threshold,
                    lay_price_max=lay_price_max,
                    feature_cache=feature_cache,
                )
            except Exception as e:
                logger.warning(
                    "monitor-eval: env build failed for %s on %s (%s)",
                    r.agent_id[:12], ed, e,
                )
                continue
            collector = RolloutCollector(
                shim=shim, policy=policy, device=str(device),
            )
            batch = collector.collect_episode(deterministic=False)
            stats = _eval_rollout_stats(
                batch=batch,
                last_info=collector.last_info,
                action_space=shim.action_space,
            )
            per_day.append({"day": ed, "day_pnl": float(stats.day_pnl)})
            total_pnl += float(stats.day_pnl)

        n_days_done = len(per_day) or 1
        per_agent.append({
            "agent_id": r.agent_id,
            "monitor_pnl_total": total_pnl,
            "monitor_pnl_mean": total_pnl / n_days_done,
            "monitor_per_day": per_day,
        })

    cohort_mean = (
        sum(a["monitor_pnl_mean"] for a in per_agent) / len(per_agent)
        if per_agent else 0.0
    )
    return {
        "per_agent": per_agent,
        "cohort_monitor_pnl_mean": cohort_mean,
    }


# ── Pre-flight cache schema check (2026-05-24) ────────────────────────────


def _preflight_cache_schema_check(
    *,
    training_days: list[str],
    data_dir: Path,
    needs_oracle: bool,
    expected_oracle_obs_dim: int | None,
    needs_direction: bool,
    direction_horizon_ticks: int,
    direction_threshold_ticks: int,
    direction_force_close_seconds: float,
) -> None:
    """Fail-fast validation that all required caches match the env schema.

    Walks every training date and verifies header metadata for the
    caches that this run will actually consume. Raises a single
    :class:`ValueError` listing every stale or missing cache, grouped
    by cache type, with the exact re-scan command lines.

    The check is a no-op when neither flag is set (``needs_oracle`` and
    ``needs_direction`` both False) — covers the "byte-identical
    pre-patch run" contract.

    Read-only: never mutates any cache. Expected runtime <2s for a
    16-day list (each header.json is a few hundred bytes).

    Parameters
    ----------
    training_days:
        Dates to validate, in ``YYYY-MM-DD`` form. Each date is
        checked independently; failures accumulate.
    data_dir:
        Project ``data/processed`` (or test equivalent). The cache
        directories sit alongside, at ``data_dir.parent /
        oracle_cache_v2`` and ``data_dir.parent / direction_labels``
        (same convention as ``arb_oracle.load_samples`` and
        ``direction_label_scan.load_labels``).
    needs_oracle:
        True when any agent will run BC pretrain (operator override
        or per-agent ``bc_pretrain_steps > 0``).
    expected_oracle_obs_dim:
        The ``shim.obs_dim`` the worker will pass as
        ``expected_obs_dim`` to ``load_oracle_samples_for_dates``.
        Required when ``needs_oracle`` is True; ignored otherwise.
    needs_direction:
        True when any direction-cache consumer is active
        (``direction_prob_loss_weight > 0`` or
        ``bc_direction_target_weight > 0``).
    direction_horizon_ticks / direction_threshold_ticks /
    direction_force_close_seconds:
        The cache-naming triple. Defaults are 60 / 5 / 60.0; CLI /
        reward_overrides can change them.
    """
    if not needs_oracle and not needs_direction:
        return  # no-op fast path; byte-identical to pre-patch

    if needs_oracle and expected_oracle_obs_dim is None:
        raise ValueError(
            "_preflight_cache_schema_check: needs_oracle=True requires "
            "expected_oracle_obs_dim to be set.",
        )

    data_dir = Path(data_dir)
    oracle_failures: list[str] = []   # human-readable bullet lines
    oracle_stale_dates: list[str] = []  # for the re-scan command list
    direction_failures: list[str] = []
    direction_stale_dates: list[str] = []
    direction_stem = ""  # bound when needs_direction is True

    # ── Oracle cache check ─────────────────────────────────────────
    if needs_oracle:
        oracle_root = data_dir.parent / "oracle_cache_v2"
        for date in training_days:
            header_path = oracle_root / str(date) / "header.json"
            if not header_path.exists():
                oracle_failures.append(
                    f"{date}: header.json missing "
                    f"(expected at {header_path})"
                )
                oracle_stale_dates.append(date)
                continue
            try:
                header = json.loads(header_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                oracle_failures.append(
                    f"{date}: header.json unreadable ({exc})"
                )
                oracle_stale_dates.append(date)
                continue
            saved_dim = header.get("obs_dim")
            if saved_dim is None:
                oracle_failures.append(
                    f"{date}: header.json has no 'obs_dim' field "
                    "(pre-schema-bump cache)"
                )
                oracle_stale_dates.append(date)
                continue
            if int(saved_dim) != int(expected_oracle_obs_dim):
                oracle_failures.append(
                    f"{date}: obs_dim={int(saved_dim)} but env "
                    f"expects {int(expected_oracle_obs_dim)}"
                )
                oracle_stale_dates.append(date)

    # ── Direction-label cache check ────────────────────────────────
    if needs_direction:
        direction_root = data_dir.parent / "direction_labels"
        fc = float(direction_force_close_seconds)
        fc_token = f"{fc:g}".replace(".", "_")
        direction_stem = (
            f"horizon{int(direction_horizon_ticks)}"
            f"_thresh{int(direction_threshold_ticks)}"
            f"_fc{fc_token}"
        )
        for date in training_days:
            header_path = (
                direction_root / str(date)
                / f"{direction_stem}_header.json"
            )
            if not header_path.exists():
                direction_failures.append(
                    f"{date}: {direction_stem}_header.json missing "
                    f"(expected at {header_path})"
                )
                direction_stale_dates.append(date)
                continue
            try:
                header = json.loads(header_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                direction_failures.append(
                    f"{date}: {direction_stem}_header.json unreadable "
                    f"({exc})"
                )
                direction_stale_dates.append(date)
                continue
            saved_v = header.get("obs_schema_version")
            if saved_v is None:
                direction_failures.append(
                    f"{date}: header has no 'obs_schema_version' field "
                    "(pre-schema-bump cache)"
                )
                direction_stale_dates.append(date)
                continue
            if int(saved_v) != int(_DIRECTION_OBS_SCHEMA_VERSION):
                direction_failures.append(
                    f"{date}: obs_schema_version={int(saved_v)} but env "
                    f"expects {int(_DIRECTION_OBS_SCHEMA_VERSION)}"
                )
                direction_stale_dates.append(date)

    # ── Aggregate + raise ──────────────────────────────────────────
    if not oracle_failures and not direction_failures:
        return  # all caches good

    lines: list[str] = ["Pre-flight cache schema check FAILED."]
    if oracle_failures:
        lines.append("")
        lines.append(
            f"Oracle cache (data/oracle_cache_v2/<date>/header.json) "
            f"— {len(oracle_failures)} stale/missing:"
        )
        for f in oracle_failures:
            lines.append(f"  - {f}")
    if direction_failures:
        lines.append("")
        lines.append(
            f"Direction-label cache "
            f"(data/direction_labels/<date>/{direction_stem}_header.json) "
            f"— {len(direction_failures)} stale/missing:"
        )
        for f in direction_failures:
            lines.append(f"  - {f}")

    lines.append("")
    lines.append("Re-scan commands (one-shot fixes the whole training set):")
    if oracle_stale_dates:
        # De-dupe + sort for a deterministic, copy-pasteable command.
        unique_dates = sorted(set(oracle_stale_dates))
        lines.append(
            "  python -m training_v2.oracle_cli scan "
            f"--dates {','.join(unique_dates)} "
            "--predictor-lean-obs"
        )
    if direction_stale_dates:
        unique_dates = sorted(set(direction_stale_dates))
        lines.append(
            "  python -m training_v2.direction_label_cli scan "
            f"--dates {','.join(unique_dates)} "
            f"--horizon-ticks {int(direction_horizon_ticks)} "
            f"--threshold-ticks {int(direction_threshold_ticks)} "
            "--force-close-before-off-seconds "
            f"{int(direction_force_close_seconds)}"
        )
    lines.append("")
    lines.append(
        "Adjust the flags above to match the args used at launch "
        "(e.g. --max-runners) before re-launching the cohort."
    )

    raise ValueError("\n".join(lines))


def run_cohort(
    *,
    n_agents: int,
    n_generations: int,
    days: int,
    data_dir: Path,
    device: str,
    seed: int,
    output_dir: Path,
    mutation_rate: float = 0.1,
    breeding: str = "ga",
    pbt_config: "PbtConfig | None" = None,
    train_one_agent_fn: Callable[..., AgentResult] = train_one_agent,
    event_emitter: Callable[[dict], None] | None = None,
    reward_overrides: dict | None = None,
    enabled_set: frozenset[str] = frozenset(),
    batched: bool = False,
    parallel_agents: int = 0,
    maturation_bonus_weight: float = 0.0,
    n_eval_days: int | None = None,
    argmax_eval: bool = False,
    per_transition_credit: bool = False,
    bc_pretrain_steps_override: int | None = None,
    bc_learning_rate_override: float | None = None,
    bc_target_entropy_warmup_eps_override: int | None = None,
    bc_include_negative_samples: bool = False,
    bc_positive_weight: float = 1.0,
    bc_include_close_hold_samples: bool = False,
    arb_spread_target_lock_pct_override: float | None = None,
    predictor_bundle: object | None = None,
    predictor_manifests: "tuple | list | None" = None,
    strategy_mode: str | None = None,
    use_race_outcome_predictor: bool = False,
    predictor_lean_obs: bool = False,
    use_direction_predictor: bool = False,
    predictor_p_win_back_threshold: float = 0.0,
    predictor_p_win_back_max_threshold: float = 1.0,
    predictor_p_win_lay_threshold: float = 1.0,
    direction_gate_enabled: bool = False,
    mature_prob_open_threshold: float = 0.0,
    race_confidence_threshold: float = 0.0,
    lay_price_max: float = 0.0,
    exclude_days: list[str] | None = None,
    composite_score_mode: str = COMPOSITE_SCORE_MODE_TOTAL_REWARD,
    early_stop_patience: int = 0,
    early_stop_min_gens: int = 4,
    cohort_eval_days: list[str] | None = None,
    training_days_explicit: list[str] | None = None,
    monitor_days: list[str] | None = None,
    rotating_eval_sample: int = 0,
    monitor_eval_top_k: int = 0,
    monitor_early_stop_patience: int = 0,
    frozen_direction_head_path: "Path | None" = None,
    resume_from: "Path | None" = None,
) -> list[AgentResult]:
    """Run the cohort end-to-end. Returns one :class:`AgentResult` per agent.

    The list contains the LAST generation's agents in eval-reward
    descending order. Earlier-generation agents are persisted to the
    registry (one model row, one scoreboard row each) but not returned
    in the list — they're recoverable by querying the registry.

    Parameters
    ----------
    n_agents:
        Cohort size per generation. Phase 3 dry-run uses 4; Session 04
        scales to 12.
    n_generations:
        Number of generations to train. ``1`` = no breeding (initial
        cohort only). ``2`` = one breeding pass between generations.
    days:
        Number of recent training days to use. Last one is held out
        as the eval day; the remaining ``days-1`` are training days.
    train_one_agent_fn:
        Injection point for the lightweight integration test —
        defaults to the real :func:`train_one_agent`.
    event_emitter:
        Optional ``Callable[[dict], None]``. When supplied, called with
        v1-shape websocket events at run-start, per-agent points (via
        :func:`train_one_agent`), and run-complete (Session 04
        deliverable). Pass ``None`` for silent runs (unit tests,
        scripted use). The CLI's ``--emit-websocket`` flag wires
        :class:`training_v2.cohort.events.WebSocketBroadcastServer`
        in here.
    """
    if composite_score_mode not in COMPOSITE_SCORE_MODES:
        raise ValueError(
            f"composite_score_mode={composite_score_mode!r} must be one "
            f"of {COMPOSITE_SCORE_MODES}",
        )
    if n_agents < 2:
        raise ValueError(f"n_agents must be >= 2 for breeding, got {n_agents}")
    if n_generations < 1:
        raise ValueError(f"n_generations must be >= 1, got {n_generations}")
    # When using the explicit-lists path (2026-05-22 overfit fix),
    # days is ignored — the sizes come from cohort_eval_days +
    # training_days_explicit. Skip the >=2 guard in that case.
    if days < 2 and (cohort_eval_days is None and training_days_explicit is None):
        raise ValueError(
            f"days must be >= 2 (training + eval), got {days}",
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    bet_logs_dir = output_dir / "bet_logs"
    db_path = output_dir / "models.db"
    scoreboard_path = output_dir / "scoreboard.jsonl"

    model_store = ModelStore(
        db_path=db_path,
        weights_dir=weights_dir,
        bet_logs_dir=bet_logs_dir,
    )

    # ── Day selection ────────────────────────────────────────────────
    training_days, eval_pool = select_days(
        data_dir=data_dir, n_days=int(days), day_shuffle_seed=int(seed),
        n_eval_days=n_eval_days,
        exclude_days=list(exclude_days) if exclude_days else None,
        cohort_eval_days=cohort_eval_days,
        training_days_explicit=training_days_explicit,
        monitor_days=monitor_days,
    )
    # ``eval_pool`` is the full set the GA can sample from.
    # ``eval_days`` (used per-generation) is either the full pool or a
    # rotated subsample, set inside the gen loop below.
    monitor_pool = list(monitor_days) if monitor_days else []

    pool_summary = (
        eval_pool[0] if len(eval_pool) == 1
        else f"{eval_pool[0]}…{eval_pool[-1]} ({len(eval_pool)} days)"
    )
    rotation_note = (
        f"; rotating {rotating_eval_sample}/{len(eval_pool)} per gen"
        if rotating_eval_sample > 0 else ""
    )
    monitor_note = (
        f"; monitor pool {len(monitor_pool)} days"
        if monitor_pool else ""
    )
    logger.info(
        "Cohort: %d agents × %d generations on %d training days "
        "(eval pool=%s%s%s); device=%s output_dir=%s",
        n_agents, n_generations, len(training_days), pool_summary,
        rotation_note, monitor_note, device, output_dir,
    )
    if (
        rotating_eval_sample > 0
        and rotating_eval_sample > len(eval_pool)
    ):
        raise ValueError(
            f"--rotating-eval-sample {rotating_eval_sample} > "
            f"eval pool size {len(eval_pool)}",
        )
    if rotating_eval_sample > 0:
        logger.info(
            "Rotating eval is ON: each generation will sample %d days "
            "from the %d-day eval pool deterministically by "
            "(seed, generation_idx). Selection metrics are NOT "
            "comparable gen-over-gen in absolute terms (different days "
            "each gen); rely on the monitor metric for cross-gen "
            "trend if a monitor pool is set.",
            rotating_eval_sample, len(eval_pool),
        )

    # ── PBT promotion ladder setup (pbt-breeding Steps 2-3) ───────────
    # Opt-in via ``breeding="pbt"``. The GA path (default) is untouched
    # below (HC#1). PBT carries learned IDENTITY across generations via
    # warm-start + a day-rotation gauntlet; it has its OWN per-tier eval
    # and runs a fixed generation budget, so it is deliberately
    # incompatible with the GA-only optional machinery (reject, don't
    # silently ignore). ``rotations`` / ``pbt_specs`` stay unused on the
    # GA path.
    breeding = str(breeding).lower()
    if breeding not in ("ga", "pbt"):
        raise ValueError(f"breeding must be 'ga' or 'pbt', got {breeding!r}")
    pbt_specs: "list | None" = None
    pbt_rotations: list = []
    pbt_hall_of_fame: list = []   # [(spec, result)] R3 champions, frozen
    pbt_gen_metrics: list = []    # per-gen heritability/diversity rows
    if breeding == "pbt":
        _incompat = []
        if resume_from is not None:
            _incompat.append("--resume-from")
        if batched:
            _incompat.append("--batched")
        if monitor_days:
            _incompat.append("--monitor-days")
        if early_stop_patience > 0:
            _incompat.append("--early-stop-patience")
        if rotating_eval_sample > 0:
            _incompat.append("--rotating-eval-sample")
        if _incompat:
            raise ValueError(
                f"--breeding pbt is incompatible with "
                f"{', '.join(_incompat)} (PBT uses its own per-rotation "
                f"eval + a fixed generation budget for a paired A/B).",
            )
        if not (parallel_agents and int(parallel_agents) > 0):
            logger.warning(
                "--breeding pbt without --parallel-agents runs the SLOW "
                "sequential path. The multiprocess path (--parallel-agents "
                "N) is strongly recommended for real runs; sequential is "
                "fine for tests / tiny smokes.",
            )
        if pbt_config is None:
            pbt_config = PbtConfig(n_agents=int(n_agents))
        elif int(pbt_config.n_agents) != int(n_agents):
            raise ValueError(
                f"pbt_config.n_agents={pbt_config.n_agents} must equal "
                f"--n-agents={n_agents}.",
            )
        pbt_config.validate()
        # Rotations come from the FULL non-sealed pool the runner selected
        # (training_days ∪ eval_pool). The sealed final-test days are
        # excluded by the operator's day selection — never in
        # training_days/eval_pool — exactly as for the GA arm.
        nonsealed_pool = sorted(set(training_days) | set(eval_pool))
        pbt_rotations = make_rotations(
            nonsealed_pool, cohort_seed=int(seed),
            n_rotations=pbt_config.n_rotations,
            train_per_rotation=pbt_config.train_per_rotation,
            eval_per_rotation=pbt_config.eval_per_rotation,
        )
        logger.info(
            "── PBT ladder: %d agents, %d rotations of %d/%d (train/eval) "
            "from %d non-sealed days. R2=%d (%d elite), R3=%d (%d elite), "
            "freeze top-%d of R3, offspring perturb ±%.0f%%. ──",
            n_agents, pbt_config.n_rotations,
            pbt_config.train_per_rotation, pbt_config.eval_per_rotation,
            len(nonsealed_pool), pbt_config.r2_size,
            pbt_config.promote_from_r1, pbt_config.r3_size,
            pbt_config.promote_from_r2, pbt_config.freeze_top_r3,
            pbt_config.perturb_frac * 100.0,
        )
        for rot in pbt_rotations:
            logger.info(
                "   rotation %d: train=%s eval=%s",
                rot.index, list(rot.train_days), list(rot.eval_days),
            )

    # ── Initial population (gen 0) OR resume (ga-recipe-search §C) ────
    rng = random.Random(int(seed))
    start_generation = 0
    cohort: list[CohortGenes]
    parent_ids: list[tuple[str | None, str | None]]
    _resume = _load_resume_state(Path(resume_from)) if resume_from else None
    if _resume is not None:
        start_generation = int(_resume["generation"])
        cohort = _resume["cohort"]
        parent_ids = _resume["parent_ids"]
        rng.setstate(_resume["rng_state"])
        if len(cohort) != n_agents:
            raise ValueError(
                f"--resume-from cohort has {len(cohort)} agents but "
                f"--n-agents={n_agents}; they must match. Re-run with the "
                f"same --n-agents the checkpoint was created with."
            )
        logger.info(
            "RESUME: continuing at generation %d/%d with %d-agent cohort "
            "(checkpoint run_id=%s).",
            start_generation + 1, n_generations, len(cohort),
            _resume.get("run_id"),
        )
    elif breeding == "pbt":
        # Generation 0: n_agents fresh-blood lineages, all in tier 1 (the
        # rookie division). The pipeline fills on later gens as winners
        # promote to unseen rotations.
        pbt_specs = init_pbt_population(rng, pbt_config, enabled_set=enabled_set)
        cohort = [s.genes for s in pbt_specs]
        parent_ids = [(s.parent_model_id, None) for s in pbt_specs]
    else:
        cohort = [
            sample_genes(rng, enabled_set=enabled_set) for _ in range(n_agents)
        ]
        parent_ids = [(None, None) for _ in range(n_agents)]

    # ── Pre-flight cache schema check (2026-05-24) ────────────────────
    # Walk every training date and verify the caches this run will
    # consume actually match the env's current schema. Crash fast at
    # launch (with copy-pasteable re-scan commands) rather than 30s
    # into agent 1's BC step. See ``_preflight_cache_schema_check``
    # docstring + tests/test_v2_cohort_runner.py::TestPreflightCacheSchemaCheck.
    _preflight_ro = dict(reward_overrides or {})
    _needs_oracle = (
        (bc_pretrain_steps_override is not None
         and int(bc_pretrain_steps_override) > 0)
        or any(
            int(getattr(g, "bc_pretrain_steps", 0)) > 0 for g in cohort
        )
    )
    _needs_direction = (
        float(_preflight_ro.get("direction_prob_loss_weight", 0.0) or 0.0) > 0.0
        or float(
            _preflight_ro.get("bc_direction_target_weight", 0.0) or 0.0
        ) > 0.0
    )
    _oracle_expected_dim: int | None = None
    if _needs_oracle:
        # Build one env on the first training day to recover the
        # ``shim.obs_dim`` the worker will pass into BC. The worker
        # builds the same env per agent per day; an extra single
        # build at launch is cheap relative to the 28h cohort cost
        # of crashing 30s into agent 1.
        _cfg_for_preflight = scalping_train_config()
        try:
            # Phase-15 fix (2026-05-24): thread the predictor flags
            # so the pre-flight env matches the worker's env shape.
            # Without these, --predictor-lean-obs cohorts saw the
            # pre-flight derive obs_dim=2254 (full obs) and falsely
            # reject 574-dim caches. The worker rebuilds the env per
            # agent per day with these exact same flags downstream.
            _env_pf, _shim_pf = _build_env_for_day(
                day_str=training_days[0],
                data_dir=Path(data_dir),
                cfg=_cfg_for_preflight,
                scorer_dir=DEFAULT_SCORER_DIR,
                reward_overrides=reward_overrides,
                predictor_bundle=predictor_bundle,
                use_race_outcome_predictor=use_race_outcome_predictor,
                use_direction_predictor=use_direction_predictor,
                predictor_lean_obs=predictor_lean_obs,
                predictor_p_win_back_threshold=(
                    predictor_p_win_back_threshold
                ),
                predictor_p_win_back_max_threshold=(
                    predictor_p_win_back_max_threshold
                ),
                predictor_p_win_lay_threshold=(
                    predictor_p_win_lay_threshold
                ),
                direction_gate_enabled=direction_gate_enabled,
                race_confidence_threshold=race_confidence_threshold,
                lay_price_max=lay_price_max,
            )
            _oracle_expected_dim = int(_shim_pf.obs_dim)
        except Exception as _exc:
            # Defensive: if we can't build an env to derive obs_dim,
            # surface the failure rather than silently skipping the
            # check.
            raise ValueError(
                "Pre-flight cache schema check could not derive "
                "expected obs_dim (failed to build env for "
                f"{training_days[0]}): {_exc}"
            ) from _exc
    _preflight_cache_schema_check(
        training_days=list(training_days),
        data_dir=Path(data_dir),
        needs_oracle=_needs_oracle,
        expected_oracle_obs_dim=_oracle_expected_dim,
        needs_direction=_needs_direction,
        direction_horizon_ticks=int(
            _preflight_ro.get("direction_horizon_ticks", 60) or 60,
        ),
        direction_threshold_ticks=int(
            _preflight_ro.get("direction_threshold_ticks", 5) or 5,
        ),
        direction_force_close_seconds=float(
            _preflight_ro.get("direction_force_close_seconds", 60.0)
            or 60.0,
        ),
    )

    last_results: list[AgentResult] = []
    cohort_t0 = time.perf_counter()
    run_id = str(uuid.uuid4())
    total_agents_trained = 0

    # ── Early-stop state (scalping-tight-naked-variance tnv3) ───────────
    # ``_early_stop_history`` accumulates per-gen stats dicts.
    # ``_early_stop_stall`` counts consecutive non-improving gens.
    _early_stop_history: list[dict] = []
    _early_stop_stall = 0

    # ── Monitor-eval state (2026-05-22 in-training overfit tripwire) ──
    _monitor_history: list[dict] = []
    _monitor_stall = 0
    monitor_metrics_path = output_dir / "monitor_metrics.jsonl"
    # On resume, preserve completed gens' monitor metrics (the overfit
    # tripwire trend must survive a restart); only wipe on a fresh run.
    if _resume is None and monitor_metrics_path.exists():
        monitor_metrics_path.unlink()
    # Build a default training cfg here so the monitor eval can share it
    # with the per-agent training calls (cfg is otherwise constructed
    # per-agent inside ``train_one_agent``).
    cfg = scalping_train_config()

    # phase-3 Option F.1 — cohort-scoped feature cache. ``engineer_day``
    # output is a pure function of (date, env feature knobs); the
    # cohort runs N agents × G generations through the same date set,
    # so caching its output recovers ~10s/env-build × (N-1) agents per
    # gen × #days. Measured: cold env build 15.8s → warm 5.9s (63%).
    # Cohort-scoped (not gen-scoped) is safe because all agents share
    # the same feature knobs — only reward shaping / gene values
    # differ. Memory: ~40 MB/day × #unique days; well under 1 GB for
    # a typical 23-day cohort.
    feature_cache: dict[str, list] = {}

    # R5 multiprocess path: master engineered-feature cache, persisted across
    # generations so each unique day is engineered ONCE in the parent, then
    # written to per-day cache files the workers load. ``mp_pool`` is the
    # WARM persistent ProcessPoolExecutor — created lazily on the first
    # parallel generation and reused for the rest of the run, so workers stay
    # alive across generations (torch imported once, per-worker day cache
    # survives → gen 2+ skips both startup costs). Shut down in the finally
    # below. All empty / unused unless ``parallel_agents > 0``.
    mp_feature_cache: dict[str, list] = {}
    mp_pool = None

    # Initial eval_days = full pool (used for the cohort_started event;
    # per-gen rotation re-samples inside the loop).
    eval_days: list[str] = list(eval_pool)

    # ── Run-start event ──────────────────────────────────────────────
    if event_emitter is not None:
        try:
            event_emitter(cohort_started_event(
                run_id=run_id,
                n_generations=int(n_generations),
                n_agents=int(n_agents),
                train_days=list(training_days),
                eval_day=str(eval_days[0]),
                seed=int(seed),
            ))
        except Exception:
            logger.exception("event_emitter raised on cohort_started; continuing")

    # On resume, drop any stale rows the interrupted generation wrote and
    # APPEND (don't truncate completed generations); fresh runs truncate.
    _sb_mode = "w"
    if _resume is not None:
        _kept = _truncate_scoreboard_at_generation(
            scoreboard_path, start_generation,
        )
        _sb_mode = "a"
        logger.info(
            "RESUME: kept %d scoreboard rows from generations < %d.",
            _kept, start_generation,
        )
    with scoreboard_path.open(_sb_mode, encoding="utf-8") as sf:
        for generation in range(start_generation, n_generations):
            gen_t0 = time.perf_counter()
            # Checkpoint BEFORE training this generation so --resume-from
            # can re-run it cleanly after a crash (ga-recipe-search §C).
            _write_resume_state(
                output_dir, generation=generation, cohort=cohort,
                parent_ids=parent_ids, rng=rng, run_id=run_id,
                n_agents=n_agents, n_generations=n_generations,
            )
            # Per-generation eval rotation (2026-05-22 overfitting fix).
            # When rotating is ON, sample N days from the pool using a
            # deterministic RNG seeded by (cohort_seed, generation_idx)
            # so the same gen always picks the same days for any given
            # cohort_seed (debuggable / reproducible).
            if rotating_eval_sample > 0:
                rng_eval = random.Random((int(seed) << 16) ^ (generation + 1))
                eval_days = sorted(rng_eval.sample(eval_pool, rotating_eval_sample))
                logger.info(
                    "── Generation %d/%d ── rotated eval days: %s",
                    generation + 1, n_generations, eval_days,
                )
            else:
                eval_days = list(eval_pool)
                logger.info(
                    "── Generation %d/%d ──", generation + 1, n_generations,
                )
            agent_ids_gen = [str(uuid.uuid4()) for _ in cohort]
            per_agent_seeds = [
                (int(seed) * 1_000_003 + generation * 10_000 + i) & 0x7FFFFFFF
                for i in range(len(cohort))
            ]
            # Per-agent train/eval days + warm-start pointers. GA: every
            # agent trains on the gen's global days, cold-start. PBT: each
            # agent trains on ITS TIER's rotation, warm-starting from its
            # parent's on-disk weights (Step 1 ⊕ Steps 2-3).
            if breeding == "pbt":
                _agent_train_days = [
                    list(pbt_rotations[s.tier - 1].train_days)
                    for s in pbt_specs
                ]
                _agent_eval_days = [
                    list(pbt_rotations[s.tier - 1].eval_days)
                    for s in pbt_specs
                ]
                _agent_init_weights = [s.init_weights_path for s in pbt_specs]
            else:
                _agent_train_days = [list(training_days) for _ in cohort]
                _agent_eval_days = [list(eval_days) for _ in cohort]
                _agent_init_weights = [None for _ in cohort]
            for idx, genes in enumerate(cohort):
                assert_in_range(genes)
                logger.info(
                    "Generation %d agent %d/%d (id=%s) genes=%s",
                    generation + 1, idx + 1, n_agents,
                    agent_ids_gen[idx][:12], genes.to_dict(),
                )

            results: list[AgentResult] = [None] * len(cohort)  # type: ignore[list-item]

            if parallel_agents and int(parallel_agents) > 0 and not batched:
                # ── R5: parallel solo-agent processes (fast CPU path) ──
                # Train the whole cohort as N parallel worker PROCESSES, each
                # a single solo ``train_one_agent`` (the golden path) at its
                # own seed, single-threaded. Bit-identical to the sequential
                # ``else`` branch — proven by tests/test_v2_multiproc_cluster
                # + the R5 probes (shared-cache parallel == sequential). The
                # GPU-batched path only parallelised the forward; this
                # parallelises the WHOLE per-agent rollout across cores
                # (~7-9x cluster-day on a many-core box).
                #
                # NOTE: the spec kwargs below MUST stay in sync with the
                # sequential ``else`` branch's ``train_one_agent_fn(...)``
                # call — they are the same call, dispatched to a pool. A
                # typo'd / removed key surfaces immediately as a worker
                # TypeError (caught by the parallel-vs-sequential integration
                # smoke); ``tests/test_v2_multiproc_cluster.py`` guards the
                # worker plumbing (cache + store injection, key popping).
                # Behaviour-knob drift is a review item when either path
                # gains a kwarg.
                # Predictor support: the worker rebuilds the bundle from its
                # MANIFEST PATHS (bit-identical, no spawn-pickle of the model
                # object). We therefore need the manifests, not just the
                # loaded bundle object — the parent has the loaded bundle for
                # solo/batched, but multiprocess workers reload from paths.
                if predictor_bundle is not None and not predictor_manifests:
                    raise ValueError(
                        "parallel_agents (multiprocess) with a predictor "
                        "bundle requires predictor_manifests (the worker "
                        "rebuilds the bundle from manifest paths). The CLI "
                        "threads --predictor-bundle-manifests through; a "
                        "programmatic caller must pass predictor_manifests."
                    )
                mp_predictor_manifests = (
                    tuple(predictor_manifests)
                    if predictor_bundle is not None else None
                )
                # Warm persistent pool: create once, reuse across generations
                # so workers stay alive (torch imported once, per-worker day
                # cache survives → gen 2+ skips both startup costs).
                if mp_pool is None:
                    mp_pool = make_pool(int(parallel_agents))
                # Engineer each unique day ONCE (master cache persisted across
                # generations); write per-DAY cache files so a warm worker
                # deserialises each day at most once over the whole run.
                # engineer_day is a pure fn of day + cohort-fixed params, so
                # the shared cache is bit-identical to per-worker engineering.
                # Union over EVERY agent's train+eval days — for GA this is
                # just unique(training_days + eval_days); for PBT it spans
                # all in-use rotations so the cache covers every tier.
                gen_days = list(dict.fromkeys(
                    [d for tr in _agent_train_days for d in tr]
                    + [d for ev in _agent_eval_days for d in ev]))
                # shared-memory-day-cache (2026-06-02): on predictors-ON runs
                # — the OOM case, where each full-obs day's engineer_day DICTS
                # are ~1 GB duplicated master + N workers — bake the downstream
                # static_obs arrays ONCE (predictors baked in) and share them
                # read-only via memmap, so the OS page cache holds a single
                # physical copy across all processes. ~10-20x smaller per day
                # AND shared. Predictor-OFF runs keep the legacy per-day dict
                # cache (lower priority; not the OOM trigger). See
                # plans/shared-memory-day-cache/.
                use_static_obs_cache = (
                    predictor_bundle is not None
                    and bool(use_race_outcome_predictor)
                )
                if use_static_obs_cache:
                    static_obs_day_paths = prebuild_static_obs_cache(
                        gen_days, data_dir=data_dir,
                        cache_dir=output_dir / "mp_static_obs_cache",
                        predictor_bundle=predictor_bundle,
                        use_race_outcome_predictor=bool(
                            use_race_outcome_predictor),
                        use_direction_predictor=bool(use_direction_predictor),
                        predictor_lean_obs=bool(predictor_lean_obs),
                    )
                    day_cache_paths = None
                else:
                    prebuild_feature_cache(
                        gen_days, data_dir=data_dir, into=mp_feature_cache)
                    day_cache_paths = save_shared_cache_per_day(
                        mp_feature_cache, output_dir / "mp_cache", gen_days)
                    static_obs_day_paths = None
                store_paths = model_store_paths(model_store)
                specs: list[dict] = []
                for idx, genes in enumerate(cohort):
                    pa_id, pb_id = parent_ids[idx]
                    specs.append(dict(
                        agent_id=agent_ids_gen[idx],
                        genes=genes,
                        days_to_train=list(_agent_train_days[idx]),
                        eval_days=list(_agent_eval_days[idx]),
                        init_weights_path=_agent_init_weights[idx],
                        data_dir=data_dir,
                        device="cpu",       # multiprocess is CPU-parallel
                        seed=per_agent_seeds[idx],
                        model_store=None,   # worker rebuilds from paths
                        generation=generation,
                        parent_a_id=pa_id,
                        parent_b_id=pb_id,
                        event_emitter=None,  # callables can't cross spawn
                        agent_idx=int(idx),
                        n_agents=int(n_agents),
                        reward_overrides=reward_overrides,
                        enabled_set=enabled_set,
                        argmax_eval=argmax_eval,
                        per_transition_credit=per_transition_credit,
                        bc_pretrain_steps_override=bc_pretrain_steps_override,
                        bc_learning_rate_override=bc_learning_rate_override,
                        bc_target_entropy_warmup_eps_override=(
                            bc_target_entropy_warmup_eps_override),
                        bc_include_negative_samples=bc_include_negative_samples,
                        bc_positive_weight=bc_positive_weight,
                        bc_include_close_hold_samples=(
                            bc_include_close_hold_samples),
                        arb_spread_target_lock_pct_override=(
                            arb_spread_target_lock_pct_override),
                        predictor_bundle=None,   # worker reloads from manifests
                        strategy_mode=strategy_mode,
                        use_race_outcome_predictor=use_race_outcome_predictor,
                        predictor_lean_obs=predictor_lean_obs,
                        use_direction_predictor=use_direction_predictor,
                        predictor_p_win_back_threshold=(
                            predictor_p_win_back_threshold),
                        predictor_p_win_back_max_threshold=(
                            predictor_p_win_back_max_threshold),
                        predictor_p_win_lay_threshold=(
                            predictor_p_win_lay_threshold),
                        direction_gate_enabled=direction_gate_enabled,
                        mature_prob_open_threshold=mature_prob_open_threshold,
                        race_confidence_threshold=race_confidence_threshold,
                        lay_price_max=lay_price_max,
                        composite_score_mode=composite_score_mode,
                        feature_cache=None,
                        frozen_direction_head_path=frozen_direction_head_path,
                        _feature_cache_day_paths=day_cache_paths,
                        _static_obs_day_paths=static_obs_day_paths,
                        _model_store_paths=store_paths,
                        _predictor_manifests=mp_predictor_manifests,
                    ))
                logger.info(
                    "── Multiprocess: %d agents on warm pool (%d workers) ──",
                    len(specs), int(parallel_agents),
                )
                cluster_results = train_cluster_multiproc(
                    specs, n_workers=int(parallel_agents), executor=mp_pool)
                for idx, result in enumerate(cluster_results):
                    results[idx] = result
                    total_agents_trained += 1
            elif batched:
                # Cluster by architecture, run each cluster batched.
                # Cross-cluster scheduling is sequential (one cluster
                # consumes the GPU at a time — Session 02 prompt §2
                # "Cross-cluster scheduling. Sequential.").
                #
                # We dry-instantiate policies temporarily just to get
                # the cluster key from each agent's hidden_size; full
                # policy construction (under per-agent seed) happens
                # inside ``train_cluster_batched``.
                cluster_to_indices: dict[tuple, list[int]] = {}
                for i, g in enumerate(cohort):
                    key = (
                        "DiscreteLSTMPolicy",
                        int(g.hidden_size),
                    )
                    cluster_to_indices.setdefault(key, []).append(i)
                for cluster_key, idxs in cluster_to_indices.items():
                    logger.info(
                        "── Cluster %s: %d agents (batched) ──",
                        cluster_key, len(idxs),
                    )
                    cluster_results = train_cluster_batched(
                        agent_ids=[agent_ids_gen[i] for i in idxs],
                        genes_list=[cohort[i] for i in idxs],
                        days_to_train=list(training_days),
                        eval_days=list(eval_days),
                        data_dir=data_dir,
                        device=device,
                        seeds=[per_agent_seeds[i] for i in idxs],
                        model_store=model_store,
                        generation=generation,
                        parent_ids=[parent_ids[i] for i in idxs],
                        event_emitter=event_emitter,
                        agent_indices_in_cohort=[int(i) for i in idxs],
                        n_agents_in_cohort=int(n_agents),
                        reward_overrides=reward_overrides,
                        enabled_set=enabled_set,
                        argmax_eval=argmax_eval,
                    )
                    if per_transition_credit:
                        # Per-transition credit lives in the sequential
                        # trainer path; the batched cohort runner has
                        # not been wired through. Surface a clear
                        # warning rather than silently failing the gate.
                        logger.warning(
                            "per_transition_credit=True ignored under "
                            "--batched; flag has no effect on this run.",
                        )
                    if (
                        bc_pretrain_steps_override is not None
                        and int(bc_pretrain_steps_override) > 0
                    ):
                        # BC pretrain lives in the sequential per-agent
                        # path (worker.py); the batched cluster runner
                        # has not been wired through. Same surface as
                        # per_transition_credit above — warn so the
                        # operator knows the flag was a no-op.
                        logger.warning(
                            "--bc-pretrain-steps=%d ignored under "
                            "--batched; flag has no effect on this run.",
                            int(bc_pretrain_steps_override),
                        )
                    for k, i in enumerate(idxs):
                        results[i] = cluster_results[k]
                        total_agents_trained += 1
            else:
                for idx, genes in enumerate(cohort):
                    pa_id, pb_id = parent_ids[idx]
                    result = train_one_agent_fn(
                        agent_id=agent_ids_gen[idx],
                        genes=genes,
                        days_to_train=list(_agent_train_days[idx]),
                        eval_days=list(_agent_eval_days[idx]),
                        init_weights_path=_agent_init_weights[idx],
                        data_dir=data_dir,
                        device=device,
                        seed=per_agent_seeds[idx],
                        model_store=model_store,
                        generation=generation,
                        parent_a_id=pa_id,
                        parent_b_id=pb_id,
                        event_emitter=event_emitter,
                        agent_idx=int(idx),
                        n_agents=int(n_agents),
                        reward_overrides=reward_overrides,
                        enabled_set=enabled_set,
                        argmax_eval=argmax_eval,
                        per_transition_credit=per_transition_credit,
                        bc_pretrain_steps_override=bc_pretrain_steps_override,
                        bc_learning_rate_override=bc_learning_rate_override,
                        bc_target_entropy_warmup_eps_override=(
                            bc_target_entropy_warmup_eps_override
                        ),
                        bc_include_negative_samples=(
                            bc_include_negative_samples
                        ),
                        bc_positive_weight=bc_positive_weight,
                        bc_include_close_hold_samples=(
                            bc_include_close_hold_samples
                        ),
                        arb_spread_target_lock_pct_override=(
                            arb_spread_target_lock_pct_override
                        ),
                        predictor_bundle=predictor_bundle,
                        strategy_mode=strategy_mode,
                        use_race_outcome_predictor=use_race_outcome_predictor,
                        predictor_lean_obs=predictor_lean_obs,
                        use_direction_predictor=use_direction_predictor,
                        predictor_p_win_back_threshold=predictor_p_win_back_threshold,
        predictor_p_win_back_max_threshold=predictor_p_win_back_max_threshold,
                        predictor_p_win_lay_threshold=predictor_p_win_lay_threshold,
                        direction_gate_enabled=direction_gate_enabled,
                        mature_prob_open_threshold=mature_prob_open_threshold,
                        race_confidence_threshold=race_confidence_threshold,
                        lay_price_max=lay_price_max,
                        composite_score_mode=composite_score_mode,
                        feature_cache=feature_cache,
                        frozen_direction_head_path=(
                            frozen_direction_head_path
                        ),
                    )
                    results[idx] = result
                    total_agents_trained += 1
                    # Cohort-visibility S01a (2026-05-02): write the
                    # scoreboard row IMMEDIATELY so an operator (or
                    # tooling) reading scoreboard.jsonl mid-cohort sees
                    # per-agent results land at agent-completion cadence
                    # (~18 min on AMBER v2 wall) rather than at end-of-
                    # generation. The batched branch keeps its post-
                    # cluster write (agents in a batched cluster don't
                    # finish independently). See plans/rewrite/phase-3-
                    # followups/cohort-visibility/.
                    row = _agent_result_to_scoreboard_row(
                        result=result,
                        generation=generation,
                        agent_idx=idx,
                        eval_days=list(eval_days),
                        training_days=list(training_days),
                        maturation_bonus_weight=maturation_bonus_weight,
                        argmax_eval=argmax_eval,
                        composite_score_mode=composite_score_mode,
                    )
                    sf.write(json.dumps(row) + "\n")
                    sf.flush()

            if batched or (parallel_agents and int(parallel_agents) > 0):
                # Batched / multiprocess branches: write scoreboard rows after
                # the cluster has populated ``results`` (the agents don't
                # finish independently / live worker events don't cross the
                # spawn boundary — the solo ``else`` branch writes its rows
                # inline instead). Per-agent live visibility within a cluster
                # is documented in cohort-visibility/purpose.md.
                for idx, result in enumerate(results):
                    row = _agent_result_to_scoreboard_row(
                        result=result,
                        generation=generation,
                        agent_idx=idx,
                        eval_days=list(eval_days),
                        training_days=list(training_days),
                        maturation_bonus_weight=maturation_bonus_weight,
                        argmax_eval=argmax_eval,
                        composite_score_mode=composite_score_mode,
                    )
                    sf.write(json.dumps(row) + "\n")
                    sf.flush()

            # Sort by composite_score (descending) — equals total_reward
            # when ``maturation_bonus_weight = 0.0`` (byte-identical to
            # PBT: capture the spec↔result pairing in agent-INDEX order
            # BEFORE the in-place sort below scrambles it — breed_pbt ranks
            # within each tier itself.
            _pbt_pairs_this_gen = (
                list(zip(pbt_specs, results)) if breeding == "pbt" else None
            )
            if _pbt_pairs_this_gen is not None:
                _write_pbt_lineage(
                    output_dir / "pbt_lineage.jsonl",
                    generation=generation,
                    pairs=_pbt_pairs_this_gen,
                    score_fn=lambda res: _composite_score(
                        res.eval, maturation_bonus_weight,
                        composite_score_mode,
                    ),
                )
            # pre-2026-05-04). When > 0 the GA also rewards matured + agent-
            # closed pairs at the operator-specified £-scale. Ties are
            # broken by day_pnl (descending) so a higher cash-P&L agent
            # ranks above a higher-shaped-reward agent at the same total
            # — useful when eval rollouts produce similar shaped
            # contributions.
            results.sort(
                key=lambda r: (
                    -_composite_score(
                        r.eval, maturation_bonus_weight,
                        composite_score_mode,
                    ),
                    -float(r.eval.day_pnl),
                ),
            )
            gen_wall = time.perf_counter() - gen_t0
            logger.info(
                "Generation %d complete in %.1fs. Top-3 by composite_score "
                "(mode=%s, maturation_bonus_weight=%.3f):",
                generation + 1, gen_wall, composite_score_mode,
                maturation_bonus_weight,
            )
            for rank, r in enumerate(results[:3]):
                logger.info(
                    "  #%d agent=%s composite=%+.3f reward=%+.3f "
                    "pnl=%+.2f bets=%d matured=%d closed=%d genes=%s",
                    rank + 1, r.agent_id[:12],
                    _composite_score(
                        r.eval, maturation_bonus_weight,
                        composite_score_mode,
                    ),
                    r.eval.total_reward,
                    r.eval.day_pnl, r.eval.bet_count,
                    r.eval.arbs_completed, r.eval.arbs_closed,
                    r.genes.to_dict(),
                )

            last_results = results

            # ── Monitor eval (2026-05-22, in-training overfit tripwire) ──
            # Evaluate top-K agents on the sealed monitor_days. NOT used
            # for selection (already chose this gen's parents above), but
            # logged so the operator can watch cohort_monitor_pnl_mean
            # diverge from the in-training composite_score as a real-time
            # overfitting signal.
            if (
                monitor_eval_top_k > 0
                and monitor_days
            ):
                top_for_monitor = results[:int(monitor_eval_top_k)]
                logger.info(
                    "Monitor eval: top-%d agents x %d monitor days "
                    "(starting at gen %d)",
                    len(top_for_monitor), len(monitor_pool),
                    generation + 1,
                )
                monitor_t0 = time.perf_counter()
                m_eval = _evaluate_agents_on_monitor_days(
                    top_results=top_for_monitor,
                    monitor_days=list(monitor_pool),
                    data_dir=data_dir,
                    cfg=cfg,
                    device=str(device),
                    reward_overrides=reward_overrides,
                    predictor_bundle=predictor_bundle,
                    use_race_outcome_predictor=bool(use_race_outcome_predictor),
                    use_direction_predictor=bool(use_direction_predictor),
                    predictor_lean_obs=bool(predictor_lean_obs),
                    predictor_p_win_back_threshold=float(predictor_p_win_back_threshold),
                    predictor_p_win_back_max_threshold=float(predictor_p_win_back_max_threshold),
                    predictor_p_win_lay_threshold=float(predictor_p_win_lay_threshold),
                    direction_gate_enabled=bool(direction_gate_enabled),
                    race_confidence_threshold=float(race_confidence_threshold),
                    lay_price_max=float(lay_price_max),
                    feature_cache=feature_cache,
                )
                # Take this gen's GA selection metric (mean composite_score
                # across cohort) for comparison
                gen_composite_mean = float(
                    sum(
                        _composite_score(
                            r.eval, maturation_bonus_weight,
                            composite_score_mode,
                        )
                        for r in results
                    ) / len(results)
                )
                monitor_row = {
                    "generation": generation + 1,
                    "monitor_eval_top_k": int(monitor_eval_top_k),
                    "monitor_days": list(monitor_pool),
                    "wall_seconds": time.perf_counter() - monitor_t0,
                    "gen_composite_mean": gen_composite_mean,
                    **m_eval,
                }
                _monitor_history.append(monitor_row)
                # Write to a separate monitor_metrics.jsonl so post-hoc
                # tooling can plot eval vs monitor across gens.
                with monitor_metrics_path.open("a", encoding="utf-8") as mf:
                    mf.write(json.dumps(monitor_row) + "\n")
                logger.info(
                    "Monitor result gen %d: cohort_monitor_pnl_mean=%+.2f, "
                    "gen_composite_mean=%+.4f, wall=%.1fs",
                    generation + 1,
                    m_eval["cohort_monitor_pnl_mean"],
                    gen_composite_mean,
                    monitor_row["wall_seconds"],
                )

                # ── Monitor-driven early stop ──────────────────────────
                # Stop when monitor regresses for N consecutive gens past
                # min_gens. This is the OVERFIT-specific early stop —
                # distinct from the existing _early_stop_stall on
                # composite. Catches the case where composite keeps
                # rising on the rotating eval but monitor flattens or
                # falls (the classic overfit signal).
                if (
                    monitor_early_stop_patience > 0
                    and len(_monitor_history) >= early_stop_min_gens
                ):
                    best_so_far = max(
                        h["cohort_monitor_pnl_mean"] for h in _monitor_history[:-1]
                    ) if len(_monitor_history) > 1 else float("-inf")
                    current_m = m_eval["cohort_monitor_pnl_mean"]
                    if current_m + 0.5 < best_so_far:
                        # Treat as a regression (tolerance £0.50/d)
                        _monitor_stall += 1
                        logger.info(
                            "Monitor early-stop: gen %d regressed "
                            "(%.2f vs best %.2f). Stall %d/%d.",
                            generation + 1, current_m, best_so_far,
                            _monitor_stall, monitor_early_stop_patience,
                        )
                        if _monitor_stall >= monitor_early_stop_patience:
                            logger.info(
                                "MONITOR EARLY STOP at gen %d/%d "
                                "(monitor regression patience=%d). "
                                "Saved ~%d remaining gens.",
                                generation + 1, n_generations,
                                monitor_early_stop_patience,
                                n_generations - generation - 1,
                            )
                            break
                    else:
                        if _monitor_stall > 0:
                            logger.info(
                                "Monitor recovered (gen %d %.2f vs best %.2f). "
                                "Stall reset.",
                                generation + 1, current_m, best_so_far,
                            )
                        _monitor_stall = 0

            # ── Early-stop check (scalping-tight-naked-variance tnv3) ─────
            # Compare this gen's median_naked_std, median_composite_score,
            # and beta_med against the running best. Increment a stall
            # counter when NONE of the three improved by their threshold.
            # Break when the stall counter exceeds patience.
            #
            # Default ``early_stop_patience=0`` disables the check entirely
            # (byte-identical to pre-tnv3 runs). Recommended live values:
            # patience=3, min_gens=4.
            if early_stop_patience > 0:
                gen_stats = _gen_early_stop_stats(results, composite_score_mode, maturation_bonus_weight)
                _early_stop_history.append(gen_stats)
                if generation + 1 >= early_stop_min_gens:
                    improved = _early_stop_improved(_early_stop_history)
                    if improved:
                        _early_stop_stall = 0
                        logger.info(
                            "Early-stop check (gen %d): IMPROVED on %s. Stall reset to 0/%d.",
                            generation + 1, ", ".join(improved), early_stop_patience,
                        )
                    else:
                        _early_stop_stall += 1
                        logger.info(
                            "Early-stop check (gen %d): no improvement on any axis "
                            "(median_std=%.2f, median_composite=%.4f, beta_med=%.5f). "
                            "Stall %d/%d.",
                            generation + 1,
                            gen_stats["median_std"], gen_stats["median_composite"],
                            gen_stats["beta_med"],
                            _early_stop_stall, early_stop_patience,
                        )
                        if _early_stop_stall >= early_stop_patience:
                            logger.info(
                                "EARLY STOP at gen %d/%d "
                                "(patience=%d, min_gens=%d). "
                                "Saved ~%d remaining gens of compute.",
                                generation + 1, n_generations,
                                early_stop_patience, early_stop_min_gens,
                                n_generations - generation - 1,
                            )
                            break

            # ── Breed next generation if any left ─────────────────
            if generation < n_generations - 1:
                if breeding == "pbt":
                    def _pbt_score(res):
                        return _composite_score(
                            res.eval, maturation_bonus_weight,
                            composite_score_mode,
                        )
                    pbt_specs, _frozen = breed_pbt(
                        _pbt_pairs_this_gen, rng, pbt_config,
                        score_fn=_pbt_score, enabled_set=enabled_set,
                    )
                    pbt_hall_of_fame.extend(_frozen)
                    cohort = [s.genes for s in pbt_specs]
                    parent_ids = [
                        (s.parent_model_id, None) for s in pbt_specs
                    ]
                    if _frozen:
                        from datetime import datetime, timezone
                        _frozen_at = datetime.now(timezone.utc).isoformat(
                            timespec="seconds")
                        _write_pbt_hall_of_fame(
                            output_dir / "pbt_hall_of_fame.jsonl",
                            generation=generation, frozen=_frozen,
                            score_fn=_pbt_score, frozen_at=_frozen_at,
                        )
                        logger.info(
                            "PBT: %d R3 champion(s) FROZEN to the "
                            "hall-of-fame at %s (total %d). Winning "
                            "architectures: %s",
                            len(_frozen), _frozen_at, len(pbt_hall_of_fame),
                            ", ".join(sorted({
                                sp.genes.architecture for sp, _ in _frozen
                            })),
                        )
                    # Regenerate the live leaderboard.txt + model_register.csv
                    # each gen (cheap JSONL read + 2 writes) so the operator
                    # has fresh viewable files throughout an 18-20h run.
                    try:
                        from tools.pbt_leaderboard import regenerate as _regen
                        from datetime import datetime as _dt
                        from datetime import timezone as _tz
                        _nc, _nm = _regen(
                            output_dir,
                            now_iso=_dt.now(_tz.utc).isoformat(
                                timespec="seconds"),
                        )
                        logger.info(
                            "PBT leaderboard: %d R3 champions, %d model-rows "
                            "-> %s", _nc, _nm,
                            output_dir / "leaderboard.txt",
                        )
                    except Exception:
                        logger.exception(
                            "PBT leaderboard regenerate failed; continuing")
                else:
                    cohort, parent_ids = _breed_next_generation(
                        parents_ranked=results,
                        rng=rng,
                        n_agents=n_agents,
                        mutation_rate=mutation_rate,
                        model_store=model_store,
                        next_generation=generation + 1,
                        enabled_set=enabled_set,
                    )

    cohort_wall = time.perf_counter() - cohort_t0
    logger.info(
        "Cohort complete in %.1fs. Wrote %s + %s",
        cohort_wall, db_path, scoreboard_path,
    )

    # ── Run-complete event ──────────────────────────────────────────
    if event_emitter is not None:
        try:
            top_5 = [
                {
                    "model_id": r.model_id,
                    "composite_score": _composite_score(
                        r.eval, maturation_bonus_weight,
                        composite_score_mode,
                    ),
                    "pnl": float(r.eval.day_pnl),
                    "win_rate": float(r.eval.bet_precision),
                    "architecture": r.architecture_name,
                }
                for r in last_results[:5]
            ]
            best_model = None
            if last_results:
                br = last_results[0]
                best_model = {
                    "model_id": br.model_id,
                    "composite_score": _composite_score(
                        br.eval, maturation_bonus_weight,
                        composite_score_mode,
                    ),
                    "total_pnl": float(br.eval.day_pnl),
                    "win_rate": float(br.eval.bet_precision),
                    "architecture": br.architecture_name,
                }
            event_emitter(cohort_complete_event(
                run_id=run_id,
                status="completed",
                n_generations=int(n_generations),
                total_agents_trained=int(total_agents_trained),
                total_agents_evaluated=int(total_agents_trained),
                wall_time_seconds=float(cohort_wall),
                best_model=best_model,
                top_5=top_5,
            ))
        except Exception:
            logger.exception("event_emitter raised on cohort_complete; continuing")

    # Tear down the warm multiprocess pool (R5). On an exception escaping the
    # generation loop this line is skipped, but ProcessPoolExecutor registers
    # an atexit handler that terminates the workers at interpreter exit, so
    # they never leak past the process.
    if mp_pool is not None:
        mp_pool.shutdown(wait=True)

    return last_results


# ── Breeding ─────────────────────────────────────────────────────────────


def _breed_next_generation(
    *,
    parents_ranked: list[AgentResult],
    rng: random.Random,
    n_agents: int,
    mutation_rate: float,
    model_store: ModelStore | None,
    next_generation: int,
    enabled_set: frozenset[str] = frozenset(),
) -> tuple[list[CohortGenes], list[tuple[str | None, str | None]]]:
    """Top-half elites carry over verbatim; bottom-half bred + mutated.

    Returns ``(next_cohort_genes, parent_ids)`` where ``parent_ids[i]``
    is ``(parent_a_id, parent_b_id)`` for child ``i``. Elites have
    ``(None, None)`` because the registry already has their parent
    chain on the previous generation's row.
    """
    n_elites = max(1, n_agents // 2)
    elites = parents_ranked[:n_elites]
    elite_genes = [e.genes for e in elites]

    next_cohort: list[CohortGenes] = list(elite_genes)
    next_parent_ids: list[tuple[str | None, str | None]] = [
        (None, None) for _ in elite_genes
    ]

    n_children = n_agents - len(next_cohort)
    for _ in range(n_children):
        if len(elites) >= 2:
            a, b = rng.sample(elites, 2)
        else:
            a = b = elites[0]
        child = crossover(a.genes, b.genes, rng, enabled_set=enabled_set)
        child = mutate(
            child, rng, mutation_rate=mutation_rate,
            enabled_set=enabled_set,
        )
        assert_in_range(child)
        next_cohort.append(child)
        next_parent_ids.append((a.model_id, b.model_id))

        if model_store is not None:
            model_store.record_genetic_event(_make_genetic_event(
                generation=next_generation,
                child_id=None,  # child model_id created in worker; tied via parents
                parent_a_id=a.model_id,
                parent_b_id=b.model_id,
                child_genes=child,
            ))

    return next_cohort, next_parent_ids


def _pbt_naked_std(res) -> float:
    """Std of per-eval-day naked_pnl — a per-model naked-variance proxy
    (the deployment-critical noise channel; see the canonical metric panel).
    Needs >= 2 eval days; returns 0.0 otherwise."""
    try:
        import statistics
        per = list(getattr(res.eval, "per_day", []) or [])
        nks = [float(getattr(s, "naked_pnl", 0.0)) for s in per]
        if len(nks) >= 2:
            return float(statistics.pstdev(nks))
    except Exception:
        pass
    return 0.0


def _pbt_model_row(spec, res, *, generation: int, score: float) -> dict:
    """Full per-model record: lineage + architecture + FULL genes + every
    eval metric we rank on. The shared schema for the per-gen register
    (``pbt_lineage.jsonl``) and the R3 hall-of-fame
    (``pbt_hall_of_fame.jsonl``) — one source of truth so the leaderboard +
    register never drift."""
    ev = res.eval
    g = spec.genes
    return {
        "generation": int(generation),
        "agent_id": getattr(res, "agent_id", None),
        "model_id": getattr(res, "model_id", None),
        "lineage_id": spec.lineage_id,
        "tier": int(spec.tier),
        "role": spec.role,
        "rotations_seen": sorted(int(r) for r in spec.rotations_seen),
        "arch_name": getattr(res, "architecture_name", ""),
        "architecture": str(g.architecture),
        "hidden_size": int(g.hidden_size),
        "transformer_depth": int(g.transformer_depth),
        "transformer_heads": int(g.transformer_heads),
        "transformer_ctx_ticks": int(g.transformer_ctx_ticks),
        "init_weights_path": spec.init_weights_path,
        "parent_model_id": spec.parent_model_id,
        "score": float(score),
        "composite_score": float(score),
        "locked_pnl": float(getattr(ev, "locked_pnl", 0.0)),
        "naked_pnl": float(getattr(ev, "naked_pnl", 0.0)),
        "naked_std": _pbt_naked_std(res),
        "closed_pnl": float(getattr(ev, "closed_pnl", 0.0)),
        "force_closed_pnl": float(getattr(ev, "force_closed_pnl", 0.0)),
        "stop_closed_pnl": float(getattr(ev, "stop_closed_pnl", 0.0)),
        "day_pnl": float(getattr(ev, "day_pnl", 0.0)),
        "total_reward": float(getattr(ev, "total_reward", 0.0)),
        "bet_count": int(getattr(ev, "bet_count", 0)),
        "winning_bets": int(getattr(ev, "winning_bets", 0)),
        "bet_precision": float(getattr(ev, "bet_precision", 0.0)),
        "arbs_completed": int(getattr(ev, "arbs_completed", 0)),
        "arbs_naked": int(getattr(ev, "arbs_naked", 0)),
        "arbs_closed": int(getattr(ev, "arbs_closed", 0)),
        "arbs_force_closed": int(getattr(ev, "arbs_force_closed", 0)),
        "arbs_stop_closed": int(getattr(ev, "arbs_stop_closed", 0)),
        "pairs_opened": int(getattr(ev, "pairs_opened", 0)),
        "genes": g.to_dict(),
    }


def _write_pbt_lineage(
    path: "Path",
    *,
    generation: int,
    pairs: list,
    score_fn,
) -> None:
    """Append one rich JSONL row per agent this generation — the per-model
    parameter register + Step 4's heritability/diversity source. Carries the
    lineage/tier/role the scoreboard lacks PLUS the full genes + metrics."""
    with open(path, "a", encoding="utf-8") as f:
        for spec, res in pairs:
            row = _pbt_model_row(
                spec, res, generation=generation, score=float(score_fn(res)))
            f.write(json.dumps(row) + "\n")


def _write_pbt_hall_of_fame(
    path: "Path",
    *,
    generation: int,
    frozen: list,
    score_fn,
    frozen_at: str,
) -> None:
    """Append the R3 champions that froze this generation, stamped with the
    ``frozen_at`` datetime they SCORED in R3 — the leaderboard.txt source."""
    with open(path, "a", encoding="utf-8") as f:
        for spec, res in frozen:
            row = _pbt_model_row(
                spec, res, generation=generation, score=float(score_fn(res)))
            row["frozen_at"] = frozen_at
            f.write(json.dumps(row) + "\n")


def _make_genetic_event(
    *,
    generation: int,
    child_id: str | None,
    parent_a_id: str,
    parent_b_id: str,
    child_genes: CohortGenes,
):
    """Build a v1-shape ``GeneticEventRecord`` for a crossover+mutate event."""
    from registry.model_store import GeneticEventRecord
    return GeneticEventRecord(
        event_id=str(uuid.uuid4()),
        generation=int(generation),
        event_type="crossover",
        child_model_id=child_id,
        parent_a_id=parent_a_id,
        parent_b_id=parent_b_id,
        hyperparameter=None,
        parent_a_value=None,
        parent_b_value=None,
        inherited_from=None,
        mutation_delta=None,
        final_value=json.dumps(child_genes.to_dict()),
        selection_reason=None,
        human_summary=(
            f"Bred from parents {parent_a_id[:12]} × {parent_b_id[:12]}; "
            f"child genes={child_genes.to_dict()}"
        ),
    )


# ── Resume / checkpoint (ga-recipe-search §C, 2026-05-30) ──────────────────
#
# Multi-day GA runs need to survive a crash/reboot without restarting from
# generation 0. Breeding is PURELY gene-based (``_breed_next_generation``
# carries only ``e.genes``; each generation trains fresh from genes, no
# weight inheritance), so a checkpoint only needs the cohort genes +
# parent_ids + the breeding RNG state + the generation index — NO weights.
# We write one ``_resume_state.json`` at the START of each generation
# (overwritten each gen; only the latest is needed). ``--resume-from``
# loads it, drops any stale scoreboard rows from the interrupted gen, and
# re-runs from there. Idempotent: completed generations are skipped.

RESUME_STATE_FILENAME = "_resume_state.json"


def _write_resume_state(
    output_dir: Path,
    *,
    generation: int,
    cohort: "list[CohortGenes]",
    parent_ids: list[tuple[str | None, str | None]],
    rng: random.Random,
    run_id: str,
    n_agents: int,
    n_generations: int,
) -> None:
    """Persist the state needed to resume at ``generation``.

    ``rng.getstate()`` returns ``(version, internalstate_tuple, gauss)``;
    the middle tuple is JSON-encoded as a list and restored as a tuple by
    :func:`_load_resume_state`.
    """
    version, internalstate, gauss = rng.getstate()
    state = {
        "schema": "ga_resume_v1",
        "generation": int(generation),
        "run_id": str(run_id),
        "n_agents": int(n_agents),
        "n_generations": int(n_generations),
        "cohort": [g.to_dict() for g in cohort],
        "parent_ids": [list(p) for p in parent_ids],
        "rng_state": [version, list(internalstate), gauss],
    }
    tmp = output_dir / (RESUME_STATE_FILENAME + ".tmp")
    final = output_dir / RESUME_STATE_FILENAME
    # Atomic-ish: write to tmp then replace, so a crash mid-write can't
    # corrupt the checkpoint we'd resume from.
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f)
    tmp.replace(final)


def _load_resume_state(resume_dir: Path) -> dict | None:
    """Load ``_resume_state.json`` from *resume_dir*, or ``None`` if absent.

    Returns a dict with ``generation`` (int), ``cohort``
    (``list[CohortGenes]``), ``parent_ids`` (list of 2-tuples), and
    ``rng_state`` (a tuple ready for ``random.Random.setstate``).
    """
    path = Path(resume_dir) / RESUME_STATE_FILENAME
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    version, internalstate, gauss = raw["rng_state"]
    return {
        "generation": int(raw["generation"]),
        "run_id": raw.get("run_id"),
        "cohort": [CohortGenes(**g) for g in raw["cohort"]],
        "parent_ids": [tuple(p) for p in raw["parent_ids"]],
        "rng_state": (version, tuple(internalstate), gauss),
        "n_agents": int(raw.get("n_agents", len(raw["cohort"]))),
        "n_generations": int(raw.get("n_generations", 0)),
    }


def _truncate_scoreboard_at_generation(path: Path, min_gen: int) -> int:
    """Drop scoreboard rows with ``generation >= min_gen`` (in place).

    On resume we re-run the interrupted generation from its checkpointed
    cohort, so any partial rows it already wrote are stale and must be
    removed before we append fresh ones. Rows for completed generations
    (``generation < min_gen``) are preserved. Returns the number of rows
    kept. A missing file is a no-op (returns 0).
    """
    path = Path(path)
    if not path.exists():
        return 0
    kept_lines: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(row.get("generation", 0)) < min_gen:
                kept_lines.append(line if line.endswith("\n") else line + "\n")
    with path.open("w", encoding="utf-8") as f:
        f.writelines(kept_lines)
    return len(kept_lines)


# ── Scoreboard row builder ────────────────────────────────────────────────


def _agent_result_to_scoreboard_row(
    *,
    result: AgentResult,
    generation: int,
    agent_idx: int,
    eval_days: list[str],
    training_days: list[str],
    maturation_bonus_weight: float = 0.0,
    argmax_eval: bool = False,
    composite_score_mode: str = COMPOSITE_SCORE_MODE_TOTAL_REWARD,
) -> dict:
    """Flatten an :class:`AgentResult` into a v1-shape scoreboard row.

    The shape mirrors v1's ``scoreboard.jsonl`` rows: flat primitives,
    one row per agent, gene dict embedded under ``hyperparameters``.
    The UI reads both v1 and v2 rows during the comparison window.
    """
    return {
        "schema": "v2_cohort_scoreboard",
        "model_id": result.model_id,
        "agent_id": result.agent_id,
        "architecture_name": result.architecture_name,
        "generation": int(generation),
        "agent_idx": int(agent_idx),
        "hyperparameters": result.genes.to_dict(),
        "weights_path": result.weights_path,
        "run_id": result.run_id,
        "training_days": list(training_days),
        # ``eval_day`` retained as the FIRST eval day for peek_cohort
        # backward compat (peek displays a single string). ``eval_days``
        # is the full list — readers that want all N held-out days
        # should use that field.
        "eval_day": eval_days[0],
        "eval_days": list(eval_days),
        # Train aggregates
        "train_n_days": result.train.n_days,
        "train_total_steps": result.train.total_steps,
        "train_total_reward": result.train.total_reward,
        "train_mean_reward": result.train.mean_reward,
        "train_mean_pnl": result.train.mean_pnl,
        "train_mean_value_loss": result.train.mean_value_loss,
        "train_mean_policy_loss": result.train.mean_policy_loss,
        "train_mean_approx_kl": result.train.mean_approx_kl,
        "train_wall_time_sec": result.train.wall_time_sec,
        "train_per_day": result.train.per_day_rows,
        # Phase-13 S06 follow-up (2026-05-07). Aux-head loss
        # diagnostics on the cohort scoreboard. Default 0.0 means the
        # corresponding lever was disabled (or the cache was missing
        # and the head was un-supervised). Non-zero means the trainer
        # computed real BCE / NLL gradient. The plumbing closes the
        # gap S06 surfaced — pre-fix scoreboard rows could not
        # distinguish "head trained but didn't move policy" from
        # "head silently inert".
        "train_mean_fill_prob_bce": result.train.mean_fill_prob_bce,
        "train_mean_mature_prob_bce": result.train.mean_mature_prob_bce,
        "train_mean_risk_nll": result.train.mean_risk_nll,
        "train_mean_direction_back_bce": (
            result.train.mean_direction_back_bce
        ),
        "train_mean_direction_lay_bce": (
            result.train.mean_direction_lay_bce
        ),
        "train_total_direction_targets": (
            result.train.total_direction_targets
        ),
        "train_direction_prob_loss_weight_active": (
            result.train.direction_prob_loss_weight_active
        ),
        # Eval (held-out day)
        "eval_total_reward": result.eval.total_reward,
        "eval_day_pnl": result.eval.day_pnl,
        "eval_n_steps": result.eval.n_steps,
        "eval_bet_count": result.eval.bet_count,
        "eval_winning_bets": result.eval.winning_bets,
        "eval_bet_precision": result.eval.bet_precision,
        "eval_pnl_per_bet": result.eval.pnl_per_bet,
        "eval_early_picks": result.eval.early_picks,
        "eval_profitable": result.eval.profitable,
        "eval_action_histogram": result.eval.action_histogram,
        "eval_arbs_completed": result.eval.arbs_completed,
        "eval_arbs_naked": result.eval.arbs_naked,
        "eval_arbs_closed": result.eval.arbs_closed,
        "eval_arbs_force_closed": result.eval.arbs_force_closed,
        "eval_arbs_stop_closed": result.eval.arbs_stop_closed,
        "eval_arbs_target_pnl_refused": result.eval.arbs_target_pnl_refused,
        "eval_pairs_opened": result.eval.pairs_opened,
        "eval_locked_pnl": result.eval.locked_pnl,
        "eval_naked_pnl": result.eval.naked_pnl,
        "eval_closed_pnl": result.eval.closed_pnl,
        "eval_force_closed_pnl": result.eval.force_closed_pnl,
        "eval_stop_closed_pnl": result.eval.stop_closed_pnl,
        # Attribution counters (2026-05-24). Default to 0 / NaN so
        # pre-patch scoreboard.jsonl rows still parse with downstream
        # readers that use ``.get(name, default)``. Same additive
        # contract as ``train_mean_fill_prob_bce`` in Phase-13 S06.
        "eval_direction_gate_refusals": (
            result.eval.direction_gate_refusals
        ),
        "eval_pwin_back_gate_refusals": (
            result.eval.pwin_back_gate_refusals
        ),
        "eval_pwin_lay_gate_refusals": (
            result.eval.pwin_lay_gate_refusals
        ),
        "eval_arb_realised_lock_pct": (
            result.eval.arb_realised_lock_pct
        ),
        "eval_wall_time_sec": result.eval.wall_time_sec,
        # Composite — single scalar the UI sorts by AND the scalar GA
        # selection actually used. Equals ``total_reward`` when
        # ``maturation_bonus_weight = 0.0`` (byte-identical to pre-2026-05-04
        # rows). When > 0 includes ``w × (matured + closed)`` per pair.
        "composite_score": _composite_score(
            result.eval, maturation_bonus_weight, composite_score_mode,
        ),
        "composite_score_mode": str(composite_score_mode),
        "maturation_bonus_weight": float(maturation_bonus_weight),
        "eval_mode": "argmax" if argmax_eval else "stochastic",
    }


# ── CLI ──────────────────────────────────────────────────────────────────


def _parse_enabled_genes(items: list[str]) -> frozenset[str]:
    """Validate and dedupe ``--enable-gene`` values.

    Each value must name a Phase 5 gene (see
    :data:`training_v2.cohort.genes.PHASE5_GENE_NAMES`). Unknown
    names raise so a typo doesn't silently revert an agent to the
    cohort-wide default for the gene the operator thought they
    enabled.
    """
    enabled: set[str] = set()
    for name in items or []:
        name = name.strip()
        if name not in PHASE5_GENE_NAMES:
            raise ValueError(
                f"--enable-gene: unknown gene name {name!r}. "
                f"Valid: {sorted(PHASE5_GENE_NAMES)}"
            )
        enabled.add(name)
    return frozenset(enabled)


def _parse_reward_overrides(items: list[str]) -> dict:
    """Parse a list of ``key=value`` strings into a dict.

    Values are parsed as bool (``true``/``false``/``1``/``0``), then
    float, then fall back to string. The env's
    ``_REWARD_OVERRIDE_KEYS`` whitelist is the authoritative typecheck
    — passing an unknown key produces a one-time debug log inside
    ``BetfairEnv.__init__`` and is otherwise ignored.
    """
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(
                f"--reward-overrides expects key=value, got {item!r}"
            )
        key, _, raw = item.partition("=")
        key = key.strip()
        raw = raw.strip()
        lo = raw.lower()
        if lo in ("true", "1"):
            out[key] = True
        elif lo in ("false", "0"):
            out[key] = False
        else:
            try:
                out[key] = float(raw)
            except ValueError:
                out[key] = raw
    return out


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "v2 GA cohort runner (Phase 3 Session 03). Trains N agents "
            "across the locked Phase 3 gene schema, breeds elites into "
            "the next generation, and writes a v1-shape scoreboard."
        ),
    )
    p.add_argument(
        "--n-agents", type=int, default=4,
        help="Cohort size per generation. Default 4 (Phase 3 dry-run).",
    )
    p.add_argument(
        "--generations", type=int, default=2,
        help="Number of generations to train. Default 2.",
    )
    p.add_argument(
        "--days", type=int, default=7,
        help=(
            "Number of recent days to use. Last is held out as eval. "
            "Default 7."
        ),
    )
    p.add_argument(
        "--data-dir", default=str(DEFAULT_DATA_DIR),
        help="Directory containing YYYY-MM-DD.parquet day files.",
    )
    p.add_argument(
        "--exclude-days", nargs="+", default=[], metavar="YYYY-MM-DD",
        help=(
            "Day(s) to drop from the candidate pool BEFORE the "
            "most-recent-N selection. Use to keep held-out evaluation "
            "dates out of training even when --days is wider than the "
            "leak boundary. Default: empty (byte-identical to pre-flag "
            "behaviour). Example: --exclude-days 2026-04-28 2026-04-29 "
            "2026-04-30 lets you safely raise --days arbitrarily high."
        ),
    )
    p.add_argument(
        "--device", default="cpu",
        help="Torch device (cpu, cuda, cuda:N). Default cpu.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Cohort seed (drives gene sampling + day shuffle).",
    )
    p.add_argument(
        "--output-dir", required=True,
        help=(
            "Directory for the cohort's models.db, weights/, "
            "bet_logs/, and scoreboard.jsonl outputs."
        ),
    )
    p.add_argument(
        "--mutation-rate", type=float, default=0.1,
        help="Per-gene mutation probability for breeding. Default 0.1.",
    )
    # ── PBT promotion ladder (pbt-breeding) ──────────────────────────
    p.add_argument(
        "--breeding", choices=["ga", "pbt"], default="ga",
        help=(
            "Breeding mechanism. 'ga' (default) = the gene-only GA "
            "(re-trains from scratch each gen; byte-identical to before). "
            "'pbt' = Population-Based-Training promotion ladder: warm-start "
            "weight inheritance + a day-rotation gauntlet (fresh blood -> "
            "rotation 1; winning earns the next unseen rotation; R3 winners "
            "freeze to a hall-of-fame). Requires --parallel-agents > 0 and "
            "is incompatible with --batched / --resume-from / --monitor-days "
            "/ --early-stop-patience / --rotating-eval-sample."
        ),
    )
    p.add_argument("--pbt-rotations", type=int, default=3,
                   help="PBT: number of day-fold rotations. Default 3.")
    p.add_argument("--pbt-train-per-rotation", type=int, default=6,
                   help="PBT: train days per rotation. Default 6.")
    p.add_argument("--pbt-eval-per-rotation", type=int, default=4,
                   help="PBT: held-out eval days per rotation. Default 4.")
    p.add_argument("--pbt-r2-size", type=int, default=10,
                   help="PBT: steady-state tier-2 size. Default 10.")
    p.add_argument("--pbt-r3-size", type=int, default=6,
                   help="PBT: steady-state tier-3 size. Default 6.")
    p.add_argument("--pbt-promote-from-r1", type=int, default=5,
                   help="PBT: top-K of R1 promoted to R2 elites. Default 5.")
    p.add_argument("--pbt-promote-from-r2", type=int, default=3,
                   help="PBT: top-K of R2 promoted to R3 elites. Default 3.")
    p.add_argument("--pbt-freeze-top-r3", type=int, default=3,
                   help="PBT: top-K of R3 frozen to hall-of-fame/gen. Default 3.")
    p.add_argument("--pbt-perturb-frac", type=float, default=0.20,
                   help="PBT: offspring recipe perturbation ±frac. Default 0.20.")
    p.add_argument(
        "--emit-websocket", action="store_true",
        help=(
            "Start a websocket broadcast server on localhost:8002 and "
            "emit v1-shape cohort events to all connected clients. "
            "Mutually exclusive with a running v1 ``training.worker`` "
            "(port collision). The api / frontend connection chain "
            "works unchanged because the api connects as a CLIENT to "
            "ws://localhost:8002 (api/main.py::_worker_connection)."
        ),
    )
    p.add_argument(
        "--ws-host", default="localhost",
        help="Bind host for --emit-websocket. Default localhost.",
    )
    p.add_argument(
        "--reward-overrides", action="append", default=[],
        metavar="KEY=VALUE",
        help=(
            "Plan-level reward override (key=value). Repeatable. "
            "Values parse as bool ('true'/'false'/'1'/'0'), float, "
            "or string in that order. Whitelisted keys live in "
            "BetfairEnv._REWARD_OVERRIDE_KEYS. Example: "
            "--reward-overrides target_pnl_pair_sizing_enabled=true"
        ),
    )
    p.add_argument(
        "--enable-gene", action="append", default=[], metavar="NAME",
        help=(
            "Enable a Phase 5 gene to evolve per-agent (repeatable). "
            "Disabled genes use cohort-wide defaults so a launch "
            "without any --enable-gene flags is byte-identical to a "
            "pre-Phase-5 run. Cannot be combined with "
            "--reward-overrides for the same gene name. Valid names: "
            "open_cost, matured_arb_bonus_weight, "
            "mark_to_market_weight, naked_loss_scale, "
            "stop_loss_pnl_threshold, arb_spread_target_lock_pct, "
            "fill_prob_loss_weight, mature_prob_loss_weight, "
            "risk_loss_weight, alpha_lr, reward_clip, "
            "direction_gate_threshold, direction_prob_loss_weight."
        ),
    )
    p.add_argument(
        "--ws-port", type=int, default=8002,
        help=(
            "Bind port for --emit-websocket. Default 8002 (matches the v1 "
            "training_worker default in config.yaml so no api change is "
            "needed)."
        ),
    )
    p.add_argument(
        "--n-eval-days", type=int, default=None, metavar="N",
        help=(
            "Number of held-out eval days at the END of the day window "
            "(the rest become training days). Default: ``--days // 2`` so "
            "a 7-day window splits as 4 train + 3 eval (training gets the "
            "bigger half on odd day counts). Restores v1's 50/50 split. "
            "Pass ``1`` for the pre-2026-05-05 single-eval-day behaviour. "
            "Eval metrics on the scoreboard are MEANS across the N eval "
            "days, so per-agent naked-pnl variance averages down ~√N."
        ),
    )
    p.add_argument(
        "--cohort-eval-days", nargs="+", default=None, metavar="YYYY-MM-DD",
        help=(
            "EXPLICIT eval pool — overrides the chronological 'last "
            "n_eval_days from --days' auto-selection. Use to break the "
            "single-contiguous-week bias by selecting eval days that "
            "span multiple weeks (e.g. one from each of 4 different "
            "weeks). Combined with --rotating-eval-sample, lets the GA "
            "sample a different subset of these days each generation, "
            "forcing the policy to generalise across regimes rather "
            "than memorise one week's quirks. See "
            "plans/EXPERIMENTS.md 2026-05-22 overfitting-prevention "
            "entry for the motivating regression (+£72/d → −£200/d "
            "swing on a different eval week)."
        ),
    )
    p.add_argument(
        "--training-days-explicit", nargs="+", default=None,
        metavar="YYYY-MM-DD",
        help=(
            "EXPLICIT training days — overrides 'last n_days minus "
            "eval' auto-selection. Optional; defaults to "
            "(all_available - excludes - eval - monitor)."
        ),
    )
    p.add_argument(
        "--monitor-days", nargs="+", default=None, metavar="YYYY-MM-DD",
        help=(
            "OBSERVE-ONLY day set. Each generation, after the GA "
            "selection step, the top-K agents are also evaluated on "
            "these days and the per-day metrics are logged to the "
            "scoreboard / monitor_metrics.jsonl. NEVER used for "
            "selection — they are the overfitting tripwire. Default: "
            "none. Cost: M monitor days × top-K agents × rollouts "
            "per generation in extra compute."
        ),
    )
    p.add_argument(
        "--monitor-eval-top-k", type=int, default=0, metavar="K",
        help=(
            "If > 0, after each generation evaluate the top-K agents "
            "(by composite_score) on the --monitor-days set. The "
            "result is logged but NEVER used for selection / breeding. "
            "Serves as an overfitting tripwire: cohort_composite "
            "should track cohort_monitor_pnl; if composite rises and "
            "monitor flattens/falls, the GA is overfitting to the "
            "eval pool. Cost: K × len(monitor_days) extra rollouts "
            "per generation."
        ),
    )
    p.add_argument(
        "--monitor-early-stop-patience", type=int, default=0, metavar="N",
        help=(
            "Stop training when cohort_monitor_pnl_mean regresses for "
            "N consecutive generations past --early-stop-min-gens. "
            "Distinct from the composite-based early-stop: this fires "
            "when the OUT-of-eval-pool metric flattens/falls even if "
            "in-training composite keeps rising. Default 0 = disabled."
        ),
    )
    p.add_argument(
        "--rotating-eval-sample", type=int, default=0, metavar="N",
        help=(
            "Per-generation random sample size from --cohort-eval-days. "
            "0 (default) = use the full eval pool every generation "
            "(byte-identical pre-flag behaviour when --cohort-eval-days "
            "is fixed). N > 0 = each generation samples N days from "
            "the pool deterministically by (seed × generation). Forces "
            "agents to perform well across many different day-slices, "
            "not memorise one fixed eval set. Recommended: pool of 12-"
            "20 days, sample 8 per generation."
        ),
    )
    p.add_argument(
        "--early-stop-patience", type=int, default=0,
        help=(
            "GA early-stop: stop after this many consecutive non-improving "
            "generations. 0 (default) disables = byte-identical pre-tnv3. "
            "Recommended live value: 3. Improvement is judged on three "
            "signals (median naked_std, median composite_score, beta_med); "
            "ANY one of them improving by its threshold resets the counter."
        ),
    )
    p.add_argument(
        "--early-stop-min-gens", type=int, default=4,
        help=(
            "Generations to run unconditionally before the early-stop "
            "check is allowed to fire. Default 4 lets the GA establish "
            "a baseline (gen 0 = random, gens 1-3 = first breeding rounds)."
        ),
    )
    p.add_argument(
        "--composite-score-mode",
        default=COMPOSITE_SCORE_MODE_TOTAL_REWARD,
        choices=list(COMPOSITE_SCORE_MODES),
        help=(
            "GA selection scalar formula. "
            "`total_reward` (default, byte-identical to pre-plan): "
            "score = total_reward + maturation_bonus_weight x "
            "(arbs_completed + arbs_closed). "
            "`locked_weighted` (scalping-locked-fitness-and-age-obs "
            "plan): score = locked_pnl + 0.25 x naked_pnl. The 0.25 "
            "weight is locked (hard_constraints #9) — calibrated "
            "against the predecessor cohort's naked/locked variance "
            "ratio. Surfaces structural locked-floor agents at the "
            "GA selection step instead of letting per-eval naked-pnl "
            "noise dominate. Mutually exclusive with maturation_bonus_"
            "weight (locked_weighted ignores it; the registry column "
            "still records the active mode + score)."
        ),
    )
    p.add_argument(
        "--sortino-lambda", type=float, default=SORTINO_DEFAULT_LAMBDA,
        metavar="FLOAT",
        help=(
            "Penalty weight on the downside-deviation term in the "
            "sortino composite_score_mode. Default %(default)s = a "
            "GBP1 increase in downside_dev cancels GBP1 of mean pnl. "
            "Only consumed when --composite-score-mode=sortino. "
            "See plans/robust-phenotype/ for the formula."
        ),
    )
    p.add_argument(
        "--maturation-bonus-weight", type=float, default=0.0,
        metavar="FLOAT",
        help=(
            "GA selection-score bonus per matured-or-agent-closed pair "
            "(£-scale). Default 0.0 = byte-identical to pre-2026-05-04 "
            "selection (sort by total_reward only). When > 0 the GA "
            "sorts by ``total_reward + w × (arbs_completed + arbs_closed)`` "
            "so high-maturation lineages survive selection even when "
            "their total_reward is dragged down by the open_cost penalty. "
            "Force-closed pairs are excluded — they're env bail-outs, "
            "not skill, matching the strict mature_prob label semantics. "
            "Knob lives at runner level, not gene level: every agent in "
            "the cohort uses the same weight in the selection signal "
            "(the GA itself can still evolve mature_prob_loss_weight as "
            "a per-agent gene). The composite_score is persisted to "
            "scoreboard.jsonl so downstream tooling sees the actual "
            "selection scalar."
        ),
    )
    p.add_argument(
        "--batched", action="store_true",
        help=(
            "Use the batched cohort path (throughput-fix Session 02). "
            "Clusters agents by architecture (hidden_size) and shares "
            "one BatchedRolloutCollector per cluster per training day. "
            "Default OFF; the sequential per-agent path stays the "
            "default until at least one cohort run validates the "
            "batched path."
        ),
    )
    p.add_argument(
        "--parallel-agents", type=int, default=None, metavar="N",
        help=(
            "training-speedup-v2 R5: train the cohort as N parallel solo-"
            "agent PROCESSES (CPU, 1 thread each) — the fastest path on a "
            "many-core box (~9x cluster-day, bit-identical to the sequential "
            "path; each worker is the golden solo train_one_agent at its own "
            "seed). Warm persistent pool across generations + per-day shared "
            "feature cache. DEFAULT 16 (the measured throughput peak — see "
            "tools/measure_optimal_n.py; re-calibrate per machine). 0 = OFF "
            "(sequential). N is the concurrency CAP, not the cohort size: a "
            "cohort of M agents runs ceil(M/N) waves. Predictor runs ARE "
            "supported (workers rebuild the bundle from manifests). Yields to "
            "--batched (error only if both set explicitly)."
        ),
    )
    p.add_argument(
        "--argmax-eval", action="store_true",
        help=(
            "Use deterministic (argmax) action selection for eval "
            "rollouts instead of stochastic sampling. Training rollouts "
            "are always stochastic regardless of this flag. Removes "
            "£100–£300 PnL swings caused by action-sampling RNG on "
            "identical weights + identical day. Scoreboard rows gain "
            "eval_mode='argmax' when active."
        ),
    )
    p.add_argument(
        "--bc-pretrain-steps", type=int, default=None, metavar="N",
        help=(
            "Phase 8 S02. Cohort-wide pin for ``bc_pretrain_steps``. "
            "Each agent runs N supervised BC steps on the v2 oracle "
            "cache against its own training days BEFORE the PPO loop "
            "starts. Trains only ``actor_head``; the PPO optimiser is "
            "untouched (separate Adam). Default ``None`` = use each "
            "agent's per-gene ``bc_pretrain_steps`` value (default 0 "
            "= no BC = byte-identical to pre-S02). When set, the "
            "operator's pin overrides any per-agent gene draw "
            "cohort-wide. Requires populated v2 oracle caches "
            "(``python -m training_v2.oracle_cli scan --dates ...``); "
            "missing caches per training day emit a warning and BC "
            "trains on whatever pool the present caches produce."
        ),
    )
    p.add_argument(
        "--bc-learning-rate", type=float, default=None, metavar="LR",
        help=(
            "Phase 8 S02. Cohort-wide pin for ``bc_learning_rate``. "
            "Override the gene default (3e-4) for the BC pretrainer's "
            "Adam optimiser. Independent of the PPO learning rate. "
            "Operator escape hatch for tuning the BC loss surface "
            "without code changes — Phase 11 ("
            "plans/rewrite/phase-11-bc-gene-exploration/) will use "
            "this flag to sweep values once Phase 8 S03 confirms BC "
            "helps. Default ``None`` = leave each agent's gene at "
            "default."
        ),
    )
    p.add_argument(
        "--bc-target-entropy-warmup-eps", type=int, default=None,
        metavar="N",
        help=(
            "Phase 8 S02. Cohort-wide pin for "
            "``bc_target_entropy_warmup_eps``. Override the gene "
            "default (5) for the entropy-controller warmup window. "
            "Phase 11 sweep target. Default ``None`` = leave each "
            "agent's gene at default."
        ),
    )
    p.add_argument(
        "--bc-include-negative-samples", action="store_true",
        help=(
            "BC label augmentation Phase A "
            "(``plans/bc-label-augmentation/``). Also load the "
            "negative-sample cache (``oracle_samples_negative.npz`` "
            "per training day) and use it during BC pretrain to target "
            "the NOOP action class on (tick, runner) pairs that are "
            "NOT in the oracle's positive arb set. Adds positive "
            "gradient on NOOP so it doesn't softmax-decay across BC "
            "steps. Default OFF = byte-identical to pre-plan. Requires "
            "caches scanned with ``--include-negative-samples``."
        ),
    )
    p.add_argument(
        "--bc-positive-weight", type=float, default=1.0, metavar="FLOAT",
        help=(
            "BC label augmentation Phase A. Multiplicative weight on "
            "the positive (oracle + direction + dir-BCE) loss term "
            "when negative samples are active. Default 1.0 keeps the "
            "positive loss at unit scale; the negative-NOOP CE term "
            "is added with weight 1.0. Set < 1.0 to soften positive "
            "pressure relative to NOOP (more NOOP authority); > 1.0 "
            "to prioritise positives. Ignored when "
            "``--bc-include-negative-samples`` is off."
        ),
    )
    p.add_argument(
        "--bc-include-close-hold-samples", action="store_true",
        help=(
            "BC label augmentation Phase B "
            "(``plans/bc-label-augmentation/``). Also load the "
            "close/hold sample cache (``oracle_samples_close_hold.npz`` "
            "per training day) and use it during BC pretrain. Each "
            "sample carries a target action class — CLOSE on the "
            "open pair's runner_idx when the env would have force-"
            "closed the pair, NOOP when it would have matured "
            "naturally. Adds positive gradient on both CLOSE and "
            "NOOP for obs vectors that include an open-pair "
            "position signature. Default OFF = byte-identical to "
            "pre-plan. Requires caches scanned with ``--include-"
            "close-hold-samples``."
        ),
    )
    p.add_argument(
        "--arb-spread-target-lock-pct", type=float, default=None,
        metavar="PCT",
        help=(
            "Cohort-wide pin for ``arb_spread_target_lock_pct`` "
            "(Phase 5 gene, redesigned 2026-05-23). Fraction of "
            "aggressive stake the agent wants locked per scalped pair. "
            "The env passes this to ``min_arb_ticks_for_profit`` as the "
            "profit_floor and uses the returned tick count for the "
            "passive's offset. Higher values = wider passive = lower "
            "fill rate but more locked profit per fill. Range "
            "[0.005, 0.05]. Default ``None`` = leave each agent at its "
            "gene draw / default (0.02 = 2%% lock per pair). Mutually "
            "exclusive with --enable-gene arb_spread_target_lock_pct. "
            "See plans/force_close_and_arb_spread/findings.md."
        ),
    )
    p.add_argument(
        "--per-transition-credit", action="store_true",
        help=(
            "Phase 9 S02. Replace the per-slot mature_prob BCE label "
            "broadcast with per-transition credit assignment: each "
            "pair's strict-mature label lands on the SINGLE step "
            "where the pair was opened, not on every transition. "
            "Default OFF (byte-identical to Phase 7). When ON, the "
            "per-update log reports n_mature_targets and per-episode "
            "stats carry per_transition_credit_active=True. fill_prob "
            "and risk_nll stay on the per-slot path."
        ),
    )
    p.add_argument(
        "--resume-from", default=None, metavar="DIR",
        help=(
            "Resume an interrupted GA run from its output dir "
            "(ga-recipe-search §C). Loads <DIR>/_resume_state.json "
            "(cohort genes + parent_ids + RNG state + generation index), "
            "drops stale scoreboard rows from the interrupted generation, "
            "and continues from there. Completed generations are skipped. "
            "Use the SAME --n-agents and gene/reward flags as the original "
            "run. Default None = fresh run from generation 0."
        ),
    )
    # Predictor-integration (plans/predictor-integration/).
    p.add_argument(
        "--strategy-mode", default=None,
        choices=["arb", "value_win", "value_each_way"],
        help=(
            "Strategy mode (predictor-integration Session 03). "
            "`arb` = pair-trade scalping (default + byte-identical "
            "to pre-plan). `value_win` = single-shot back/lay "
            "informed by champion's calibrated p_win. "
            "`value_each_way` = single-shot EW (Session 06; needs "
            "Session 04 part 3+ env shim translation)."
        ),
    )
    p.add_argument(
        "--predictor-bundle-manifests", nargs=3, default=None,
        metavar=("CHAMPION", "RANKER", "DIRECTION"),
        help=(
            "Three paths to the betfair-predictors manifest.json "
            "files. When supplied, the runner constructs a "
            "PredictorBundle, threads it through to every agent, "
            "and tags the cohort row with the 3 experiment_ids "
            "(hard_constraints §7). Only useful when "
            "--use-race-outcome-predictor is also true."
        ),
    )
    p.add_argument(
        "--use-race-outcome-predictor", action="store_true",
        help=(
            "Inject champion+ranker outputs into the runner obs "
            "slice (predictor-integration data-bridging). Requires "
            "--predictor-bundle-manifests to be supplied. Default "
            "False; flag-off behaviour is byte-identical to pre-plan."
        ),
    )
    p.add_argument(
        "--predictor-lean-obs", action="store_true",
        help=(
            "AGENT-AS-HUMAN + 2 ADVISORS mode. Strips the env's "
            "RUNNER_KEYS down to 23: just the predictors' outputs + "
            "minimum market state (back/lay price, spread, velocity). "
            "The agent gets the advisors' opinions, NOT the firehose. "
            "Use with --use-race-outcome-predictor for the real "
            "experiment. Default False = full 143-col obs."
        ),
    )
    p.add_argument(
        "--use-direction-predictor", action="store_true",
        help=(
            "Inject per-tick direction predictor outputs (price-mover "
            "advisor) into the obs slice. Adds ~6s per env "
            "construction (V2 ladder window construction + batched "
            "Conv1D forward) — minimal overhead. Requires "
            "--predictor-bundle-manifests. Most useful for scalping/"
            "arb mode (per the predictor's `signal_description`)."
        ),
    )
    p.add_argument(
        "--predictor-p-win-back-threshold", type=float, default=0.0,
        help=(
            "Action-mask gate: refuse OPEN_BACK on runners whose "
            "champion p_win is below this. Default 0.0 = gate "
            "disabled. Only meaningful when "
            "--use-race-outcome-predictor is set. See "
            "plans/scalping-pwin-gate/."
        ),
    )
    p.add_argument(
        "--predictor-p-win-back-max-threshold", type=float, default=1.0,
        help=(
            "Action-mask gate (added 2026-05-27): refuse OPEN_BACK "
            "on runners whose champion p_win is ABOVE this. Combined "
            "with --predictor-p-win-back-threshold, defines a pwin "
            "BAND for back-leg selection. Default 1.0 = no upper "
            "bound. Motivated by Round 9 EV-by-pwin analysis showing "
            "p_win 0.40-0.50 has negative naked EV (-£0.19/pair) "
            "while p_win 0.30-0.35 peaks at +£9.49/pair."
        ),
    )
    p.add_argument(
        "--predictor-p-win-lay-threshold", type=float, default=1.0,
        help=(
            "Action-mask gate: refuse OPEN_LAY on runners whose "
            "champion p_win is above this. Default 1.0 = gate "
            "disabled."
        ),
    )
    p.add_argument(
        "--direction-gate-enabled", action="store_true",
        help=(
            "Action-mask gate: also refuse OPEN_LAY on runners "
            "where dir_fire_drift did NOT fire at the current tick. "
            "Requires --use-direction-predictor. Composes with "
            "--predictor-p-win-back-threshold / "
            "--predictor-p-win-lay-threshold (champion gate). See "
            "plans/scalping-direction-gate/."
        ),
    )
    p.add_argument(
        "--mature-prob-open-threshold", type=float, default=0.0,
        help=(
            "Policy-side action-mask gate: refuse OPEN_BACK/OPEN_LAY "
            "on runners whose own mature_prob_head sigmoid output is "
            "below this threshold. Default 0.0 = gate disabled. "
            "Refusals surface in the direction_gate_refusals counter. "
            "Effective threshold anneals from 0.0 up to this value "
            "over mature_gate_warmup_eps episodes to avoid cold-start "
            "collapse. See plans/recipe-expansion-and-robustness/ "
            "(Path C)."
        ),
    )
    p.add_argument(
        "--race-confidence-threshold", type=float, default=0.0,
        help=(
            "Per-race action-mask gate: refuse all opens/closes in "
            "races where max(champion p_win) across runners is below "
            "this. Default 0.0 = disabled. Requires "
            "--use-race-outcome-predictor. See "
            "plans/scalping-race-confidence-gate/."
        ),
    )
    p.add_argument(
        "--lay-price-max", type=float, default=0.0,
        help=(
            "Per-(tick, runner) lay-price cap: refuse OPEN_LAY on "
            "runners whose current LTP exceeds this. Default 0.0 = "
            "disabled. Allowed range [0, 1000]. Requires "
            "--use-race-outcome-predictor (the cap composes with "
            "the predictor-driven lay gate). See "
            "plans/scalping-lay-quality-gate/."
        ),
    )
    p.add_argument(
        "--direction-head-manifest", default=None, metavar="PATH",
        help=(
            "Path to a directory containing weights.pt + "
            "manifest.json for a pre-trained shared direction head. "
            "When supplied, every agent's direction_prob_head is "
            "loaded from this path and frozen (requires_grad=False). "
            "Forces direction_prob_loss_weight + "
            "bc_direction_target_weight to 0 (frozen weights aren't "
            "trainable). Mutually exclusive with `--enable-gene "
            "direction_prob_loss_weight` and `--enable-gene "
            "bc_direction_target_weight`. See "
            "plans/shared-direction-head/."
        ),
    )
    return p.parse_args(argv)


def _resolve_parallel_agents(
    parallel_agents_arg: "int | None", *, batched: bool,
) -> int:
    """Resolve the ``--parallel-agents`` worker count (training-speedup-v2 R5).

    Default (arg is ``None``) is **16** — the measured throughput peak
    (``tools/measure_optimal_n.py``) — so the fast multiprocess path is ON by
    default. The only conflict is ``--batched``: it takes precedence, raising
    ONLY if BOTH were set explicitly, otherwise the default 16 silently yields
    to batched (returns 0). Predictor runs ARE supported (workers rebuild the
    bundle from manifests — see the multiprocess branch in ``run_cohort``), so
    they no longer disable multiprocess.

    Returns the resolved worker count (0 = sequential / off). This mirrors the
    `_resolve_<knob>` pattern from the Path-A precedence foot-gun lesson
    (CLAUDE.md) so the CLI default and the explicit value are both tested.
    """
    explicit = parallel_agents_arg is not None
    pa = int(parallel_agents_arg) if explicit else 16
    if pa <= 0:
        return 0
    if batched:
        if explicit:
            raise SystemExit(
                "--parallel-agents and --batched are mutually exclusive: "
                "--parallel-agents runs N solo agents as parallel CPU "
                "processes; --batched runs one GPU-batched cluster. Pick one."
            )
        return 0
    return pa


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    # robust-phenotype R1. Set the module-level sortino_lambda from
    # the CLI BEFORE any _composite_score calls happen (the function
    # reads _SORTINO_LAMBDA when its ``sortino_lambda`` param is
    # left None). Default value preserves byte-identity when the
    # flag is unset.
    global _SORTINO_LAMBDA
    _SORTINO_LAMBDA = float(args.sortino_lambda)

    server: WebSocketBroadcastServer | None = None
    emitter: Callable[[dict], None] | None = None
    if args.emit_websocket:
        server = WebSocketBroadcastServer(host=args.ws_host, port=args.ws_port)
        server.start()
        emitter = server

    reward_overrides = _parse_reward_overrides(args.reward_overrides)
    enabled_set = _parse_enabled_genes(args.enable_gene)
    # Mutual-exclusion guard. Operator must pick one source of truth
    # per knob per run: either evolve the gene per-agent
    # (``--enable-gene``) or fix it cohort-wide
    # (``--reward-overrides``), not both. See
    # ``plans/rewrite/phase-5-restore-genes/purpose.md``
    # §"Hard constraints" item 5.
    collision = enabled_set & set(reward_overrides)
    if collision:
        raise ValueError(
            "Cannot combine --enable-gene with --reward-overrides for "
            f"the same gene name(s): {sorted(collision)}. Operator "
            "must pick one source of truth per knob per run. Either "
            "evolve the gene per-agent (--enable-gene) or fix it "
            "cohort-wide (--reward-overrides), not both."
        )
    if reward_overrides:
        logger.info("reward_overrides: %s", reward_overrides)
    if enabled_set:
        logger.info("Phase 5 enabled genes: %s", sorted(enabled_set))

    # plans/shared-direction-head/hard_constraints.md §4: a frozen
    # direction head has requires_grad=False on its parameters, so
    # any supervised gradient ON the head is silently ignored. To
    # avoid the misleading "BCE is being computed but the head
    # isn't changing" footgun, refuse the combination at launch.
    if args.direction_head_manifest:
        _bad_genes = {
            "direction_prob_loss_weight",
            "bc_direction_target_weight",
        } & enabled_set
        if _bad_genes:
            raise ValueError(
                "--direction-head-manifest loads a FROZEN shared "
                "direction head. Cannot combine with "
                f"--enable-gene for: {sorted(_bad_genes)}. The "
                "head's weights are frozen, so a per-agent loss "
                "weight is meaningless. Either drop the "
                "--enable-gene flag(s) (the head trains itself "
                "off your training data) or drop "
                "--direction-head-manifest (and let each agent "
                "train its own head)."
            )
        _bad_overrides = (
            {"direction_prob_loss_weight",
             "bc_direction_target_weight"}
            & set(reward_overrides)
        )
        if _bad_overrides and any(
            float(reward_overrides[k]) > 0
            for k in _bad_overrides
        ):
            logger.warning(
                "--direction-head-manifest is set and the operator "
                "passed --reward-overrides %s — those weights will "
                "be forced to 0 inside the worker (head is frozen).",
                sorted(_bad_overrides),
            )
        logger.info(
            "Loading FROZEN direction head from %s — "
            "direction_prob_loss_weight + bc_direction_target_weight "
            "will be forced to 0 in trainer_hp.",
            args.direction_head_manifest,
        )
    if float(args.maturation_bonus_weight) != 0.0:
        logger.info(
            "GA selection composite_score = total_reward + %.3f × "
            "(arbs_completed + arbs_closed)",
            float(args.maturation_bonus_weight),
        )
    if args.argmax_eval:
        logger.info("Eval mode: argmax (deterministic action + Beta.mean stake)")
    if args.per_transition_credit:
        logger.info("Per-transition mature_prob credit: ENABLED")
    if args.bc_pretrain_steps is not None:
        logger.info(
            "BC pretrain: cohort-wide pin %d steps (overrides per-agent gene)",
            int(args.bc_pretrain_steps),
        )
    if args.bc_learning_rate is not None:
        logger.info(
            "BC pretrain: cohort-wide pin learning_rate=%g",
            float(args.bc_learning_rate),
        )
    if args.bc_target_entropy_warmup_eps is not None:
        logger.info(
            "BC pretrain: cohort-wide pin target_entropy_warmup_eps=%d",
            int(args.bc_target_entropy_warmup_eps),
        )
    if args.arb_spread_target_lock_pct is not None:
        logger.info(
            "Scalping: cohort-wide pin arb_spread_target_lock_pct=%g",
            float(args.arb_spread_target_lock_pct),
        )
        if "arb_spread_target_lock_pct" in enabled_set:
            raise ValueError(
                "Cannot combine --arb-spread-target-lock-pct with "
                "--enable-gene arb_spread_target_lock_pct (one source "
                "of truth per knob per run).",
            )

    # Predictor bundle (predictor-integration data-bridging).
    predictor_bundle = None
    if args.predictor_bundle_manifests is not None:
        if not args.use_race_outcome_predictor:
            logger.warning(
                "--predictor-bundle-manifests supplied but "
                "--use-race-outcome-predictor not set; bundle will be "
                "loaded for registry tagging but env will not consume "
                "predictor obs.",
            )
        from predictors import PredictorBundle
        champ, rank, dirm = args.predictor_bundle_manifests
        predictor_bundle = PredictorBundle.from_manifests(
            champion_manifest=champ,
            ranker_manifest=rank,
            direction_manifest=dirm,
        )
        logger.info(
            "predictor bundle loaded: champion=%s ranker=%s direction=%s",
            predictor_bundle.champion_experiment_id,
            predictor_bundle.ranker_experiment_id,
            predictor_bundle.direction_experiment_id,
        )
    elif args.use_race_outcome_predictor:
        raise SystemExit(
            "--use-race-outcome-predictor requires "
            "--predictor-bundle-manifests CHAMPION RANKER DIRECTION",
        )

    parallel_agents = _resolve_parallel_agents(
        args.parallel_agents, batched=bool(args.batched),
    )

    try:
        run_cohort(
            n_agents=args.n_agents,
            n_generations=args.generations,
            days=args.days,
            data_dir=Path(args.data_dir),
            device=args.device,
            seed=args.seed,
            output_dir=Path(args.output_dir),
            mutation_rate=args.mutation_rate,
            breeding=args.breeding,
            pbt_config=(
                PbtConfig(
                    n_agents=int(args.n_agents),
                    n_rotations=int(args.pbt_rotations),
                    train_per_rotation=int(args.pbt_train_per_rotation),
                    eval_per_rotation=int(args.pbt_eval_per_rotation),
                    r2_size=int(args.pbt_r2_size),
                    r3_size=int(args.pbt_r3_size),
                    promote_from_r1=int(args.pbt_promote_from_r1),
                    promote_from_r2=int(args.pbt_promote_from_r2),
                    freeze_top_r3=int(args.pbt_freeze_top_r3),
                    perturb_frac=float(args.pbt_perturb_frac),
                ) if args.breeding == "pbt" else None
            ),
            event_emitter=emitter,
            reward_overrides=reward_overrides or None,
            enabled_set=enabled_set,
            batched=bool(args.batched),
            parallel_agents=parallel_agents,
            maturation_bonus_weight=float(args.maturation_bonus_weight),
            n_eval_days=(
                int(args.n_eval_days) if args.n_eval_days is not None else None
            ),
            argmax_eval=bool(args.argmax_eval),
            per_transition_credit=bool(args.per_transition_credit),
            bc_pretrain_steps_override=(
                int(args.bc_pretrain_steps)
                if args.bc_pretrain_steps is not None else None
            ),
            bc_learning_rate_override=(
                float(args.bc_learning_rate)
                if args.bc_learning_rate is not None else None
            ),
            bc_target_entropy_warmup_eps_override=(
                int(args.bc_target_entropy_warmup_eps)
                if args.bc_target_entropy_warmup_eps is not None else None
            ),
            bc_include_negative_samples=bool(args.bc_include_negative_samples),
            bc_positive_weight=float(args.bc_positive_weight),
            bc_include_close_hold_samples=bool(
                args.bc_include_close_hold_samples,
            ),
            arb_spread_target_lock_pct_override=(
                float(args.arb_spread_target_lock_pct)
                if args.arb_spread_target_lock_pct is not None else None
            ),
            predictor_bundle=predictor_bundle,
            predictor_manifests=args.predictor_bundle_manifests,
            strategy_mode=args.strategy_mode,
            use_race_outcome_predictor=bool(args.use_race_outcome_predictor),
            predictor_lean_obs=bool(args.predictor_lean_obs),
            use_direction_predictor=bool(args.use_direction_predictor),
            predictor_p_win_back_threshold=float(args.predictor_p_win_back_threshold),
            predictor_p_win_back_max_threshold=float(args.predictor_p_win_back_max_threshold),
            predictor_p_win_lay_threshold=float(args.predictor_p_win_lay_threshold),
            direction_gate_enabled=bool(args.direction_gate_enabled),
            mature_prob_open_threshold=float(args.mature_prob_open_threshold),
            race_confidence_threshold=float(args.race_confidence_threshold),
            lay_price_max=float(args.lay_price_max),
            exclude_days=list(args.exclude_days) if args.exclude_days else None,
            composite_score_mode=str(args.composite_score_mode),
            early_stop_patience=int(args.early_stop_patience),
            early_stop_min_gens=int(args.early_stop_min_gens),
            cohort_eval_days=(
                list(args.cohort_eval_days) if args.cohort_eval_days else None
            ),
            training_days_explicit=(
                list(args.training_days_explicit)
                if args.training_days_explicit else None
            ),
            monitor_days=(
                list(args.monitor_days) if args.monitor_days else None
            ),
            rotating_eval_sample=int(args.rotating_eval_sample),
            monitor_eval_top_k=int(args.monitor_eval_top_k),
            monitor_early_stop_patience=int(args.monitor_early_stop_patience),
            frozen_direction_head_path=(
                Path(args.direction_head_manifest)
                if args.direction_head_manifest else None
            ),
            resume_from=(
                Path(args.resume_from) if args.resume_from else None
            ),
        )
    finally:
        if server is not None:
            # Give clients a beat to receive the final cohort_complete
            # event before closing the listen socket.
            time.sleep(0.5)
            server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
