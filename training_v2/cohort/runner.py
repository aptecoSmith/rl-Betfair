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
from training_v2.cohort.batched_worker import train_cluster_batched
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
    train_one_agent_fn: Callable[..., AgentResult] = train_one_agent,
    event_emitter: Callable[[dict], None] | None = None,
    reward_overrides: dict | None = None,
    enabled_set: frozenset[str] = frozenset(),
    batched: bool = False,
    maturation_bonus_weight: float = 0.0,
    n_eval_days: int | None = None,
    argmax_eval: bool = False,
    per_transition_credit: bool = False,
    bc_pretrain_steps_override: int | None = None,
    bc_learning_rate_override: float | None = None,
    bc_target_entropy_warmup_eps_override: int | None = None,
    arb_spread_target_lock_pct_override: float | None = None,
    predictor_bundle: object | None = None,
    strategy_mode: str | None = None,
    use_race_outcome_predictor: bool = False,
    predictor_lean_obs: bool = False,
    use_direction_predictor: bool = False,
    predictor_p_win_back_threshold: float = 0.0,
    predictor_p_win_lay_threshold: float = 1.0,
    direction_gate_enabled: bool = False,
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

    # ── Initial population (gen 0) ───────────────────────────────────
    rng = random.Random(int(seed))
    cohort: list[CohortGenes] = [
        sample_genes(rng, enabled_set=enabled_set) for _ in range(n_agents)
    ]
    parent_ids: list[tuple[str | None, str | None]] = [
        (None, None) for _ in range(n_agents)
    ]

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
    if monitor_metrics_path.exists():
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

    with scoreboard_path.open("w", encoding="utf-8") as sf:
        for generation in range(n_generations):
            gen_t0 = time.perf_counter()
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
            for idx, genes in enumerate(cohort):
                assert_in_range(genes)
                logger.info(
                    "Generation %d agent %d/%d (id=%s) genes=%s",
                    generation + 1, idx + 1, n_agents,
                    agent_ids_gen[idx][:12], genes.to_dict(),
                )

            results: list[AgentResult] = [None] * len(cohort)  # type: ignore[list-item]

            if batched:
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
                        days_to_train=list(training_days),
                        eval_days=list(eval_days),
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
                        arb_spread_target_lock_pct_override=(
                            arb_spread_target_lock_pct_override
                        ),
                        predictor_bundle=predictor_bundle,
                        strategy_mode=strategy_mode,
                        use_race_outcome_predictor=use_race_outcome_predictor,
                        predictor_lean_obs=predictor_lean_obs,
                        use_direction_predictor=use_direction_predictor,
                        predictor_p_win_back_threshold=predictor_p_win_back_threshold,
                        predictor_p_win_lay_threshold=predictor_p_win_lay_threshold,
                        direction_gate_enabled=direction_gate_enabled,
                        race_confidence_threshold=race_confidence_threshold,
                        lay_price_max=lay_price_max,
                        composite_score_mode=composite_score_mode,
                        feature_cache=feature_cache,
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

            if batched:
                # Batched branch: write scoreboard rows after the cluster
                # loop above has populated ``results``. Per-agent live
                # visibility within a batched cluster needs the batched-
                # rollout collector to emit per-agent sub-events — out
                # of scope for this plan; documented as a known
                # limitation in cohort-visibility/purpose.md.
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
            "risk_loss_weight, alpha_lr, reward_clip."
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
    return p.parse_args(argv)


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
            event_emitter=emitter,
            reward_overrides=reward_overrides or None,
            enabled_set=enabled_set,
            batched=bool(args.batched),
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
            arb_spread_target_lock_pct_override=(
                float(args.arb_spread_target_lock_pct)
                if args.arb_spread_target_lock_pct is not None else None
            ),
            predictor_bundle=predictor_bundle,
            strategy_mode=args.strategy_mode,
            use_race_outcome_predictor=bool(args.use_race_outcome_predictor),
            predictor_lean_obs=bool(args.predictor_lean_obs),
            use_direction_predictor=bool(args.use_direction_predictor),
            predictor_p_win_back_threshold=float(args.predictor_p_win_back_threshold),
            predictor_p_win_lay_threshold=float(args.predictor_p_win_lay_threshold),
            direction_gate_enabled=bool(args.direction_gate_enabled),
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
