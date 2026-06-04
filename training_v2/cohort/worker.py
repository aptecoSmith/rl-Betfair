"""Single-agent training driver — Phase 3, Session 03 deliverable.

Trains one agent across N days (multi-day loop from Session 02), runs
one rollout-only eval pass on the held-out day, writes weights +
scoreboard row to the registry, and returns the :class:`AgentResult`.

The worker re-uses Session 02's per-day re-bind pattern:
- One :class:`DiscretePPOTrainer` is built once.
- For each day, we build a fresh ``BetfairEnv`` + ``DiscreteActionShim``
  and re-bind the trainer's ``shim`` / ``_collector`` (Phase 1's hidden
  state is reset to zero each day per ``RolloutCollector.collect_episode``).
- After all training days, we run one final rollout on ``eval_day`` with
  ``policy.eval()`` and NO PPO update — same collector, no gradient.
- Eval-day metrics + train summary populate one v1-shape registry row.

Hard constraints (session 03 prompt §"Hard constraints"):

- No env edits.
- No re-import of v1 trainer / worker / runner.
- Locked Phase 3 gene schema only.
- Registry shape matches v1 exactly (``EvaluationDayRecord`` fields
  filled where v2 has data; left at default for fields v2 doesn't
  populate yet — Session 04 may extend).
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from agents_v2.action_space import ActionType
from agents_v2.policy_factory import build_policy, policy_arch_name
from agents_v2.env_shim import DEFAULT_SCORER_DIR, DiscreteActionShim
from data.episode_builder import load_day
from env.betfair_env import (
    ACTION_SCHEMA_VERSION,
    OBS_SCHEMA_VERSION,
    BetfairEnv,
)
from registry.model_store import (
    EvaluationBetRecord,
    EvaluationDayRecord,
    ModelStore,
)
from training_v2.cohort.events import (
    agent_training_complete_event,
    agent_training_started_event,
    episode_complete_event,
)
from training_v2.cohort.genes import (
    PHASE5_GENE_NAMES,
    CohortGenes,
    assert_in_range,
    is_gpu_lane_eligible,
)
from training_v2.discrete_ppo.bc_pretrain import (
    DiscreteBCPretrainer,
    build_direction_bce_label_map,
    build_direction_target_map,
    load_direction_labels_for_dates,
    load_close_hold_samples_for_dates,
    load_negative_samples_for_dates,
    load_oracle_samples_for_dates,
    measure_post_bc_entropy,
)
from training_v2.discrete_ppo.rollout import RolloutCollector
from training_v2.discrete_ppo.trainer import DiscretePPOTrainer, EpisodeStats


logger = logging.getLogger(__name__)


__all__ = [
    "AgentResult",
    "TrainSummary",
    "EvalSummary",
    "arch_name_for_genes",
    "scalping_train_config",
    "load_warm_start_weights",
    "train_one_agent",
    "_build_per_agent_reward_overrides",
    "_build_per_agent_scalping_overrides",
    "_build_trainer_hp",
]


# Phase 5 (2026-05-03): genes that flow through the env's
# ``reward_overrides`` passthrough. ``alpha_lr`` is trainer-only and
# not currently consumed by either path (the v2 trainer doesn't yet
# accept an alpha_lr hyperparameter — see
# plans/rewrite/phase-5-restore-genes/session_prompts/02_*.md
# §"Stop conditions"; the gene is still recorded on the CohortGenes
# row so future trainer support can read it without a schema bump).
_PHASE5_GENES_VIA_REWARD_OVERRIDES: frozenset[str] = frozenset({
    "open_cost",
    "matured_arb_bonus_weight",
    "mark_to_market_weight",
    "naked_loss_scale",
    "stop_loss_pnl_threshold",
    "fill_prob_loss_weight",
    "mature_prob_loss_weight",
    "risk_loss_weight",
    "reward_clip",
    # scalping-tight-naked-variance Phase 2A (2026-05-15). Flows
    # through reward_overrides to env's _compute_scalping_reward_terms.
    "naked_variance_penalty_beta",
})

_PHASE5_GENES_VIA_SCALPING_OVERRIDES: frozenset[str] = frozenset({
    # Price-adaptive arb_spread, redesigned 2026-05-23
    # (plans/force_close_and_arb_spread/). Always active in the env
    # when scalping_mode=True; gene-evolved when opted in via
    # --enable-gene arb_spread_target_lock_pct, otherwise pinned to
    # the cohort-wide default 0.02 (2% lock per pair).
    "arb_spread_target_lock_pct",
})

# Phase 7 Session 02 (2026-05-04). The three auxiliary-head loss
# weights are read by the trainer directly from a per-agent ``hp``
# dict (NOT the env's ``reward_overrides`` passthrough). The worker
# pre-merges any cohort-level ``--reward-overrides <key>=<value>`` into
# the hp dict before constructing the trainer (Path A in
# ``plans/rewrite/phase-7-port-aux-heads/session_prompts/
# 02_wire_bce_loss_in_trainer.md``). This is the load-bearing fix for
# the v1↔v2 hp-dict-precedence trap: v2's ``CohortGenes.to_dict``
# always populates these keys with their default 0.0, so a v1-style
# ``hp.get(name, config_fallback)`` would return 0.0 and silently
# swallow the override. See ``lessons_learnt.md``.
_PHASE7_TRAINER_HP_KEYS: frozenset[str] = frozenset({
    "fill_prob_loss_weight",
    "mature_prob_loss_weight",
    "risk_loss_weight",
    # fc-cost-probe D (2026-05-17): same Path-A precedence as the
    # other aux-head weights. Default 0.0 in genes; cohort-wide pin
    # via ``--reward-overrides fc_prob_loss_weight=3.0``.
    "fc_prob_loss_weight",
})

# Phase-13 (2026-05-06). Direction-prob aux head — phase-13 S03.
# Same Path-A precedence as the Phase 7 keys: read by the trainer
# directly from the per-agent ``hp`` dict, NOT via env-side
# ``reward_overrides`` passthrough. The four label-defining knobs
# resolve the offline cache stem at trainer init.
_PHASE13_TRAINER_HP_KEYS: frozenset[str] = frozenset({
    "direction_prob_loss_weight",
    "direction_horizon_ticks",
    "direction_threshold_ticks",
    "direction_force_close_seconds",
    "bc_direction_target_weight",
    # Phase-15 S02: BC direction-BCE pos_weight on/off knob.
    # Default True; can be toggled via reward-overrides.
    "direction_bce_use_pos_weight",
})

# Phase-14 S03 (2026-05-07). Direction-gate keys consumed by the
# policy at construction time (NOT the trainer). They flow through
# the same Path-A precedence as Phase 7 / 13 keys: the worker reads
# them from the per-agent ``hp`` dict, which it pre-merges from
# cohort-level ``--reward-overrides``. ``direction_gate_enabled`` is
# a cohort-wide bool; ``direction_gate_threshold`` is a Phase 5
# gene that evolves per-agent when the operator opts in via
# ``--enable-gene direction_gate_threshold``.
_PHASE14_TRAINER_HP_KEYS: frozenset[str] = frozenset({
    "direction_gate_enabled",
    "direction_gate_threshold",
    "direction_gate_warmup_eps",
})


def _resolve_direction_gate_enabled(
    *, cli_flag: bool, trainer_hp: dict,
) -> bool:
    """Return whether the policy-side direction gate should be active.

    Two sources can enable the gate:

    - ``cli_flag``: ``True`` when the operator passed
      ``--direction-gate-enabled`` to the cohort runner.
    - ``trainer_hp["direction_gate_enabled"]``: ``True`` when the
      operator passed ``--reward-overrides direction_gate_enabled=true``
      OR (in principle) when a per-agent gene draws ``True``. The
      gene's dataclass default is ``False``, so absent an explicit
      override this contributes ``False``.

    **OR semantics**: enable if EITHER source says enable. This was
    a 2026-05-24 fix — the prior implementation used
    ``trainer_hp.get("direction_gate_enabled", cli_flag)``, which
    failed because ``to_dict()`` always populates the key with the
    gene default ``False``, so the CLI flag was silently overridden.
    Result: ``--direction-gate-enabled`` only flipped the ENV-side
    gate (built via ``_build_env_for_day``); the POLICY-side gate
    (action-mask + ``gate_refusals`` counter) stayed OFF unless the
    operator also passed ``--reward-overrides direction_gate_enabled=true``.

    Regression-guarded in
    ``tests/test_v2_direction_gate.py::TestResolvePolicyGateEnabled``.
    """
    return bool(cli_flag) or bool(
        trainer_hp.get("direction_gate_enabled", False)
    )


# ── Public dataclasses ────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainSummary:
    """Aggregate metrics over the multi-day training pass."""

    n_days: int
    total_steps: int
    total_reward: float
    mean_reward: float
    mean_pnl: float
    mean_value_loss: float
    mean_policy_loss: float
    mean_approx_kl: float
    wall_time_sec: float
    per_day_rows: list[dict] = field(default_factory=list)
    # Phase-13 S06 follow-up (2026-05-07). Per-update aux-head loss
    # diagnostics aggregated across training days. Zero when the
    # corresponding weight is 0.0 OR the trainer's per-update path
    # didn't compute the term (e.g. cache missing, head un-supervised).
    # The plumbing through to the scoreboard is what S06's NULL-result
    # caveat #1 specifically asked for: without these on disk we can
    # not verify from a cohort artefact alone whether the head
    # actually trained.
    mean_fill_prob_bce: float = 0.0
    mean_mature_prob_bce: float = 0.0
    mean_risk_nll: float = 0.0
    mean_direction_back_bce: float = 0.0
    mean_direction_lay_bce: float = 0.0
    total_direction_targets: int = 0
    direction_prob_loss_weight_active: float = 0.0


@dataclass(frozen=True)
class EvalSummary:
    """Eval-day metrics from the rollout-only pass.

    When the cohort runs ``--n-eval-days N > 1``, the worker evaluates
    the trained policy against each held-out day independently and
    builds a single aggregate ``EvalSummary`` via :func:`aggregate_eval_summaries`
    — every numeric field becomes the MEAN across the N per-day
    summaries. ``eval_day`` carries the FIRST eval-day date string
    (kept as a string for backward-compat with the single-eval-day
    callers / scoreboard schema); ``per_day`` carries the unaggregated
    list when N > 1.

    The mean choice keeps fields' interpretation stable across runs
    with different ``n_eval_days`` settings — a reported ``day_pnl=-£50``
    means "expected to lose £50 per held-out day" regardless of how many
    eval days were evaluated. ``composite_score = total_reward + w *
    (arbs_completed + arbs_closed)`` then represents per-day skill, not
    a sum that scales with eval-budget choice.
    """

    eval_day: str
    total_reward: float
    day_pnl: float
    n_steps: int
    bet_count: int
    winning_bets: int
    bet_precision: float
    pnl_per_bet: float
    early_picks: int
    profitable: bool
    action_histogram: dict[str, int]
    arbs_completed: int = 0
    arbs_naked: int = 0
    arbs_closed: int = 0
    arbs_force_closed: int = 0
    arbs_stop_closed: int = 0
    arbs_target_pnl_refused: int = 0
    pairs_opened: int = 0
    locked_pnl: float = 0.0
    naked_pnl: float = 0.0
    closed_pnl: float = 0.0
    force_closed_pnl: float = 0.0
    stop_closed_pnl: float = 0.0
    # Attribution counters (2026-05-24). Pure additive fields with
    # falsy defaults so pre-2026-05-24 rows round-trip unchanged.
    # See ``env/betfair_env.py`` __init__ for field semantics.
    direction_gate_refusals: int = 0
    pwin_back_gate_refusals: int = 0
    pwin_lay_gate_refusals: int = 0
    arb_realised_lock_pct: float = float("nan")
    wall_time_sec: float = 0.0
    per_day: list["EvalSummary"] = field(default_factory=list)


def _nanmean_attr(summaries: list["EvalSummary"], attr: str) -> float:
    """Mean over a list, skipping NaNs. Returns NaN if every value is NaN.

    Used for ``arb_realised_lock_pct`` aggregation across per-day
    summaries: a per-day NaN means "no filled pairs this day" rather
    than "lock-pct was zero", so a plain mean would conflate the two
    and a zero-default would be misleading.
    """
    vals: list[float] = []
    for s in summaries:
        v = float(getattr(s, attr, float("nan")))
        if v == v:  # NaN ≠ NaN
            vals.append(v)
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def aggregate_eval_summaries(
    summaries: list[EvalSummary],
) -> EvalSummary:
    """Combine N per-day eval summaries into a single mean-aggregate.

    All numeric fields are MEANED across the input summaries (counts
    too — ``bet_count=421`` on a 3-day aggregate means "421 bets per
    day on average"). ``action_histogram`` keys are union-merged with
    summed counts then divided by N. ``profitable`` is True iff the
    MEAN ``day_pnl`` is positive. ``eval_day`` is set to the FIRST
    summary's eval_day for backward-compat with the single-string field;
    callers needing the full date list should read ``per_day``.

    A single-element input is returned with ``per_day=[input]`` (so the
    aggregate carries the same data and the caller can still introspect
    the per-day list uniformly).

    Implementation notes:
    - ``bet_precision`` is recomputed from MEAN(winning_bets) /
      MEAN(bet_count) so it's a valid ratio rather than a mean of
      ratios (which would be biased by per-day denominator differences).
    - ``pnl_per_bet`` similarly recomputed from MEAN(day_pnl) /
      MEAN(bet_count).
    - ``wall_time_sec`` is SUMMED (operator wants total compute spent,
      not "per-day wall").
    """
    if not summaries:
        raise ValueError("aggregate_eval_summaries: empty input")
    if len(summaries) == 1:
        s = summaries[0]
        # Wrap in per_day list so downstream code can iterate uniformly.
        return EvalSummary(
            eval_day=s.eval_day, total_reward=s.total_reward,
            day_pnl=s.day_pnl, n_steps=s.n_steps,
            bet_count=s.bet_count, winning_bets=s.winning_bets,
            bet_precision=s.bet_precision, pnl_per_bet=s.pnl_per_bet,
            early_picks=s.early_picks, profitable=s.profitable,
            action_histogram=dict(s.action_histogram),
            arbs_completed=s.arbs_completed, arbs_naked=s.arbs_naked,
            arbs_closed=s.arbs_closed,
            arbs_force_closed=s.arbs_force_closed,
            arbs_stop_closed=s.arbs_stop_closed,
            arbs_target_pnl_refused=s.arbs_target_pnl_refused,
            pairs_opened=s.pairs_opened,
            locked_pnl=s.locked_pnl, naked_pnl=s.naked_pnl,
            closed_pnl=s.closed_pnl,
            force_closed_pnl=s.force_closed_pnl,
            stop_closed_pnl=s.stop_closed_pnl,
            direction_gate_refusals=s.direction_gate_refusals,
            pwin_back_gate_refusals=s.pwin_back_gate_refusals,
            pwin_lay_gate_refusals=s.pwin_lay_gate_refusals,
            arb_realised_lock_pct=s.arb_realised_lock_pct,
            wall_time_sec=s.wall_time_sec,
            per_day=[s],
        )

    n = len(summaries)
    mean = lambda attr: sum(getattr(s, attr) for s in summaries) / n
    total = lambda attr: sum(getattr(s, attr) for s in summaries)

    merged_hist: dict[str, float] = {}
    for s in summaries:
        for k, v in s.action_histogram.items():
            merged_hist[k] = merged_hist.get(k, 0.0) + v
    mean_hist = {k: int(round(v / n)) for k, v in merged_hist.items()}

    mean_winning = mean("winning_bets")
    mean_bet_count = mean("bet_count")
    mean_day_pnl = mean("day_pnl")
    bet_precision = (
        mean_winning / mean_bet_count if mean_bet_count > 0 else 0.0
    )
    pnl_per_bet = (
        mean_day_pnl / mean_bet_count if mean_bet_count > 0 else 0.0
    )

    return EvalSummary(
        eval_day=summaries[0].eval_day,
        total_reward=mean("total_reward"),
        day_pnl=mean_day_pnl,
        n_steps=int(round(mean("n_steps"))),
        bet_count=int(round(mean_bet_count)),
        winning_bets=int(round(mean_winning)),
        bet_precision=bet_precision,
        pnl_per_bet=pnl_per_bet,
        early_picks=int(round(mean("early_picks"))),
        profitable=mean_day_pnl > 0.0,
        action_histogram=mean_hist,
        arbs_completed=int(round(mean("arbs_completed"))),
        arbs_naked=int(round(mean("arbs_naked"))),
        arbs_closed=int(round(mean("arbs_closed"))),
        arbs_force_closed=int(round(mean("arbs_force_closed"))),
        arbs_stop_closed=int(round(mean("arbs_stop_closed"))),
        arbs_target_pnl_refused=int(round(mean("arbs_target_pnl_refused"))),
        pairs_opened=int(round(mean("pairs_opened"))),
        locked_pnl=mean("locked_pnl"),
        naked_pnl=mean("naked_pnl"),
        closed_pnl=mean("closed_pnl"),
        force_closed_pnl=mean("force_closed_pnl"),
        stop_closed_pnl=mean("stop_closed_pnl"),
        # Attribution counters (2026-05-24). Means match the rest
        # of the per-day rollup. ``arb_realised_lock_pct`` mean
        # filters out per-day NaNs (no filled pairs ⇒ undefined,
        # not zero); if every day was NaN the aggregate stays NaN.
        direction_gate_refusals=int(round(mean("direction_gate_refusals"))),
        pwin_back_gate_refusals=int(round(mean("pwin_back_gate_refusals"))),
        pwin_lay_gate_refusals=int(round(mean("pwin_lay_gate_refusals"))),
        arb_realised_lock_pct=_nanmean_attr(
            summaries, "arb_realised_lock_pct",
        ),
        wall_time_sec=total("wall_time_sec"),
        per_day=list(summaries),
    )


@dataclass(frozen=True)
class AgentResult:
    """Outcome of one agent's full train + eval cycle."""

    agent_id: str
    model_id: str
    architecture_name: str
    genes: CohortGenes
    train: TrainSummary
    eval: EvalSummary
    weights_path: str
    run_id: str


# ── Architecture-name discriminator (session 03 prompt §4) ────────────────


def arch_name_for_genes(genes: CohortGenes) -> str:
    """Registry ``architecture_name`` discriminator — delegates to the
    single source of truth in :func:`agents_v2.policy_factory.policy_arch_name`.

    LSTM agents keep the prior ``v2_discrete_ppo_lstm_h{hidden_size}``
    name (byte-identical discriminator for existing cohorts); transformer
    agents (pbt-breeding Step 1b) carry their depth/heads/ctx in the name
    so the registry's weight-shape hash + UI never confuse the two and
    weights never cross-load. ``v1`` weights use a different prefix
    entirely, so v2 cohorts never collide with v1 cohorts.
    """
    return policy_arch_name(genes)


_ARCH_DESCRIPTION = (
    "v2 discrete-action PPO + LSTM (Phase 3 cohort). "
    "Masked categorical over {NOOP, OPEN_BACK_i, OPEN_LAY_i, CLOSE_i}, "
    "Beta stake head, per-runner value head. Hidden size encoded in "
    "architecture_name suffix."
)


# ── Default config (matches train.py's _scalping_train_config) ────────────


def scalping_train_config(max_runners: int = 14) -> dict:
    """Phase 2 / 3 scalping baseline — same shape as ``train.py``.

    Returned fresh per call so the worker can mutate it locally without
    leaking changes between agents.
    """
    return {
        "training": {
            "max_runners": max_runners,
            "starting_budget": 100.0,
            "max_bets_per_race": 20,
            "scalping_mode": True,
            "betting_constraints": {
                "max_back_price": 50.0,
                "max_lay_price": None,
                "min_seconds_before_off": 0,
                "force_close_before_off_seconds": 0,
            },
        },
        "actions": {"force_aggressive": True},
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "efficiency_penalty": 0.01,
            "commission": 0.05,
            "mark_to_market_weight": 0.0,
        },
    }


# ── Helpers ──────────────────────────────────────────────────────────────


def _build_env_for_day(
    *,
    day_str: str,
    data_dir: Path,
    cfg: dict,
    scorer_dir: Path,
    reward_overrides: dict | None = None,
    scalping_overrides: dict | None = None,
    predictor_bundle: object | None = None,
    use_race_outcome_predictor: bool | None = None,
    use_direction_predictor: bool | None = None,
    predictor_lean_obs: bool = False,
    predictor_p_win_back_threshold: float = 0.0,
    predictor_p_win_back_max_threshold: float = 1.0,
    predictor_p_win_lay_threshold: float = 1.0,
    direction_gate_enabled: bool = False,
    race_confidence_threshold: float = 0.0,
    lay_price_max: float = 0.0,
    market_type_filter: str | None = None,
    emit_debug_features: bool = False,
    feature_cache: dict[str, list] | None = None,
    static_obs_cache: dict | None = None,
) -> tuple[BetfairEnv, DiscreteActionShim]:
    """Build a BetfairEnv + DiscreteActionShim for a single day.

    2026-05-22 perf phase 2: ``emit_debug_features`` defaults to
    False here. The env's default is True, but per-tick per-runner
    OBI / microprice / traded_delta / mid_drift / book_churn
    computations in `_get_info` consume ~25-40% of step time and are
    NEVER read by the training rollout collector (only by tests +
    replay UI). Setting False here speeds up cohort training by
    8-12 min/agent with zero impact on the GA signal. See
    plans/cohort_training_speedup/phase_2.md.

    Pass True explicitly when the caller actually needs the debug
    features (e.g. ad-hoc evaluation scripts that read
    ``info["debug_features"]``).
    """
    day = load_day(day_str, data_dir=data_dir)
    env_kwargs = dict(
        reward_overrides=reward_overrides,
        scalping_overrides=scalping_overrides,
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
        emit_debug_features=emit_debug_features,
    )
    if market_type_filter is not None:
        env_kwargs["market_type_filter"] = market_type_filter
    # phase-3 Option F.1: the env constructor accepts a
    # feature_cache dict keyed by ``day.date``. When the cohort
    # runner threads its per-run cache through, ``engineer_day``
    # output is reused across agents in the same cohort, recovering
    # ~10s of the 16s env-build wall (63% reduction, measured).
    if feature_cache is not None:
        env_kwargs["feature_cache"] = feature_cache
    # shared-memory-day-cache (2026-06-02): the cross-process memmap path.
    # When present for this day the env skips engineer_day + predictor
    # inference and reads the pre-baked shared arrays + gate caches.
    if static_obs_cache is not None:
        env_kwargs["static_obs_cache"] = static_obs_cache
    env = BetfairEnv(day, cfg, **env_kwargs)
    shim = DiscreteActionShim(env, scorer_dir=scorer_dir)
    return env, shim


def _build_per_agent_reward_overrides(
    *,
    cohort_overrides: dict | None,
    genes: CohortGenes,
    enabled_set: frozenset[str],
) -> dict | None:
    """Combine cohort-level reward_overrides with this agent's
    enabled-gene values.

    Cohort-level overrides apply to ALL agents (e.g.
    ``force_close_before_off_seconds=60``). Enabled-gene values are
    per-agent (e.g. ``open_cost`` = this agent's gene draw).

    Phase 5 invariant: enabled-gene names cannot collide with
    cohort-level override keys (CLI guard at runner enforces).

    Returns ``None`` when nothing would be passed — keeps the
    pre-Phase-5 ``reward_overrides=None`` byte-identity for legacy
    launches (no cohort overrides, no enabled genes).
    """
    out: dict = dict(cohort_overrides or {})
    for name in PHASE5_GENE_NAMES:
        if name in enabled_set and name in _PHASE5_GENES_VIA_REWARD_OVERRIDES:
            out[name] = float(getattr(genes, name))
    return out or None


def _build_trainer_hp(
    *,
    cohort_overrides: dict | None,
    genes: CohortGenes,
    enabled_set: frozenset[str],
) -> dict:
    """Pre-merge cohort-level reward_overrides into the per-agent hp dict.

    Phase 7 Session 02 — Path A. The trainer reads
    ``fill_prob_loss_weight`` / ``mature_prob_loss_weight`` /
    ``risk_loss_weight`` from this dict ONLY (no config fallback). The
    precedence cascade for each trainer-side key:

    1. ``cohort_overrides[key]`` — operator's cohort-wide pin via
       ``--reward-overrides``. Wins iff present.
    2. ``genes.<key>`` — per-agent gene draw. Wins iff the gene is in
       ``enabled_set`` (operator opted in via ``--enable-gene``).
    3. The gene's default (always present in ``genes.to_dict()``,
       0.0 for the three S02 keys).

    Other ``CohortGenes`` fields are passed through verbatim so the
    trainer can read them too if it wants. Worker sole responsibility:
    one source of truth before the trainer is constructed.
    """
    hp: dict = dict(genes.to_dict())
    overrides = dict(cohort_overrides or {})
    for name in _PHASE7_TRAINER_HP_KEYS:
        if name in overrides:
            # Cohort-wide pin wins over the gene default. (If the
            # operator also enabled the gene the runner's CLI guard
            # rejected the launch — see ``training_v2/cohort/runner.py``
            # mutual-exclusion check.)
            hp[name] = float(overrides[name])
        elif name in enabled_set:
            hp[name] = float(getattr(genes, name))
        # else: gene default already in ``hp`` from genes.to_dict().

    # Phase-13 (2026-05-06). Direction-prob aux head: same Path-A
    # precedence — cohort-level overrides win over the gene default.
    # ``direction_prob_loss_weight`` is the on/off knob; the three
    # cache-resolution keys are passed through verbatim so the trainer
    # can build the cache-stem regardless of override order.
    for name in _PHASE13_TRAINER_HP_KEYS:
        if name in overrides:
            value = overrides[name]
            if name in (
                "direction_horizon_ticks", "direction_threshold_ticks",
            ):
                hp[name] = int(value)
            elif name == "direction_bce_use_pos_weight":
                # Phase-15 S02: bool knob.
                if isinstance(value, str):
                    hp[name] = value.lower() in ("1", "true", "yes")
                else:
                    hp[name] = bool(value)
            else:
                hp[name] = float(value)
        # Phase-13 S05 — also pass the gene's own value through if set
        # (default 0.0). The BC pretrain reads ``hp["bc_direction_
        # target_weight"]`` to decide whether to load direction labels.

    # Phase-14 S03 (2026-05-07). Direction-gate keys.
    # ``direction_gate_enabled`` is a cohort-wide bool — operator
    # opts-in via ``--reward-overrides direction_gate_enabled=true``.
    # The threshold is a Phase 5 gene; if the operator enabled it
    # via ``--enable-gene direction_gate_threshold`` the per-agent
    # gene draw lands in ``hp`` already (genes.to_dict path); the
    # cohort-wide override (if any) takes precedence.
    if "direction_gate_enabled" in overrides:
        v = overrides["direction_gate_enabled"]
        if isinstance(v, str):
            v = v.lower() in ("1", "true", "yes")
        hp["direction_gate_enabled"] = bool(v)
    if "direction_gate_threshold" in overrides:
        hp["direction_gate_threshold"] = float(
            overrides["direction_gate_threshold"],
        )
    elif "direction_gate_threshold" in enabled_set:
        # GA-evolved gene; pull the agent's draw.
        hp["direction_gate_threshold"] = float(
            getattr(genes, "direction_gate_threshold"),
        )
    # Phase-14 S06: warmup window (operator-controlled).
    if "direction_gate_warmup_eps" in overrides:
        hp["direction_gate_warmup_eps"] = int(
            overrides["direction_gate_warmup_eps"],
        )
    return hp


def _build_per_agent_scalping_overrides(
    *,
    genes: CohortGenes,
    enabled_set: frozenset[str],
) -> dict | None:
    """Build the per-agent ``scalping_overrides`` dict from gene values.

    Currently only ``arb_spread_target_lock_pct`` lives in
    scalping_overrides; other Phase 5 genes flow via ``reward_overrides``.
    """
    out: dict = {}
    for name in _PHASE5_GENES_VIA_SCALPING_OVERRIDES:
        if name in enabled_set:
            out[name] = float(getattr(genes, name))
    return out or None


def _rebind_trainer(
    trainer: DiscretePPOTrainer,
    shim: DiscreteActionShim,
) -> None:
    """Mirror of ``train.py::_rebind_trainer_for_day``.

    Same policy, same optimiser — only the env-bound rollout state
    changes per day. Phase 1 resets the LSTM hidden state to zero at
    every ``collect_episode`` call, so day boundaries are clean
    episode boundaries (multi-day prompt §2).
    """
    trainer.shim = shim
    trainer.action_space = shim.action_space
    trainer.max_runners = shim.max_runners
    # phase-3 Option A — rebind the collector to the trainer's rollout
    # device, not its update device. Single-device runs see them equal
    # so the rebind is byte-identical to pre-Option-A.
    trainer._collector = RolloutCollector(
        shim=shim, policy=trainer.policy,
        device=str(trainer.rollout_device),
    )
    # Phase-13 S03 — direction-prob caches per-day labels keyed by
    # date+config; clear on rebind so the new day's labels load
    # cleanly. The cache is small (one numpy grid per (date, knob)
    # tuple) so this only affects training-time label-load latency.
    if hasattr(trainer, "_direction_label_cache"):
        trainer._direction_label_cache.clear()


def _eval_rollout_stats(
    *,
    batch,
    last_info: dict,
    action_space,
) -> EvalSummary:
    """Build :class:`EvalSummary` from a rollout-only pass on the eval day.

    Phase 4 Session 06 (2026-05-02): consumes a
    :class:`training_v2.discrete_ppo.transition.RolloutBatch` instead
    of ``list[Transition]``. The action histogram and total-reward
    sum read from the batch's pre-stacked arrays — same numbers as
    pre-Session-06.
    """
    n_steps = int(batch.n_steps)
    total_reward = float(batch.per_runner_reward.sum())
    hist: dict[str, int] = {}
    action_idx_arr = batch.action_idx
    for i in range(n_steps):
        kind, _runner = action_space.decode(int(action_idx_arr[i]))
        hist[kind.name] = hist.get(kind.name, 0) + 1

    day_pnl = float(last_info.get("day_pnl", 0.0))
    bet_count = int(last_info.get("bet_count", 0))
    winning_bets = int(last_info.get("winning_bets", 0))
    # ``early_picks`` isn't an env-level rollup; sum it from per-race
    # records when present, else default 0.
    race_records = last_info.get("race_records", []) or []
    early_picks = int(sum(
        int(getattr(r, "early_picks", 0)) for r in race_records
    ))
    arbs_completed = int(last_info.get("arbs_completed", 0))
    arbs_naked = int(last_info.get("arbs_naked", 0))
    arbs_closed = int(last_info.get("arbs_closed", 0))
    arbs_force_closed = int(last_info.get("arbs_force_closed", 0))
    arbs_stop_closed = int(last_info.get("arbs_stop_closed", 0))
    arbs_target_pnl_refused = int(
        last_info.get("arbs_target_pnl_refused", 0)
    )
    pairs_opened = int(last_info.get("pairs_opened", 0))
    locked_pnl = float(last_info.get("locked_pnl", 0.0))
    naked_pnl = float(last_info.get("naked_pnl", 0.0))
    closed_pnl = float(last_info.get("scalping_closed_pnl", 0.0))
    force_closed_pnl = float(
        last_info.get("scalping_force_closed_pnl", 0.0)
    )
    stop_closed_pnl = float(
        last_info.get("scalping_stop_closed_pnl", 0.0)
    )
    # Attribution counters (2026-05-24). Pre-change rollouts emit no
    # such keys; readers must default-tolerate. Defaults: integer
    # counters → 0; arb_realised_lock_pct → NaN (no filled pairs).
    direction_gate_refusals = int(
        last_info.get("direction_gate_refusals", 0)
    )
    pwin_back_gate_refusals = int(
        last_info.get("pwin_back_gate_refusals", 0)
    )
    pwin_lay_gate_refusals = int(
        last_info.get("pwin_lay_gate_refusals", 0)
    )
    arb_realised_lock_pct = float(
        last_info.get("arb_realised_lock_pct", float("nan"))
    )

    bet_precision = (
        float(winning_bets) / float(bet_count) if bet_count > 0 else 0.0
    )
    pnl_per_bet = day_pnl / bet_count if bet_count > 0 else 0.0

    return EvalSummary(
        eval_day="",  # filled by caller
        total_reward=total_reward,
        day_pnl=day_pnl,
        n_steps=n_steps,
        bet_count=bet_count,
        winning_bets=winning_bets,
        bet_precision=bet_precision,
        pnl_per_bet=pnl_per_bet,
        early_picks=early_picks,
        profitable=day_pnl > 0.0,
        action_histogram=hist,
        arbs_completed=arbs_completed,
        arbs_naked=arbs_naked,
        arbs_closed=arbs_closed,
        arbs_force_closed=arbs_force_closed,
        arbs_stop_closed=arbs_stop_closed,
        arbs_target_pnl_refused=arbs_target_pnl_refused,
        pairs_opened=pairs_opened,
        locked_pnl=locked_pnl,
        naked_pnl=naked_pnl,
        closed_pnl=closed_pnl,
        force_closed_pnl=force_closed_pnl,
        stop_closed_pnl=stop_closed_pnl,
        direction_gate_refusals=direction_gate_refusals,
        pwin_back_gate_refusals=pwin_back_gate_refusals,
        pwin_lay_gate_refusals=pwin_lay_gate_refusals,
        arb_realised_lock_pct=arb_realised_lock_pct,
    )


# ── Per-bet log capture (Phase 2a) ──────────────────────────────────────


def _build_eval_bet_records(
    *,
    env: BetfairEnv,
    day,
    starting_budget: float,
) -> list[EvaluationBetRecord]:
    """Build :class:`EvaluationBetRecord` list from the day's settled bets.

    Captures predictor context (per-runner pwin + race max pwin) when
    the predictor is active, and derives a per-pair lifecycle
    classification (``matured`` / ``agent_closed`` / ``force_closed`` /
    ``stop_closed`` / ``naked`` / ``directional``) for each bet.

    ``run_id`` is left as an empty string and must be patched by the
    caller before writing to the registry — the run_id is only known
    after ``create_evaluation_run`` returns. Mirrors the v1 pattern in
    ``training/evaluator.py``.

    Returns an empty list when ``env.all_settled_bets`` is empty so
    the caller can short-circuit the write.
    """
    bets = list(env.all_settled_bets)
    if not bets:
        return []

    # Lookup tables built once per day.
    race_by_market = {r.market_id: r for r in day.races}
    market_to_race_idx = {r.market_id: i for i, r in enumerate(day.races)}
    pwin_by_race: list[dict[int, float]] = getattr(
        env, "_race_p_win_by_race", [],
    )

    # Pre-compute per-pair lifecycle classification. We touch every
    # bet in two passes: first pass groups by pair_id and decides
    # the category; second pass stamps it on each leg's record.
    pair_legs: dict[str, list] = {}
    for b in bets:
        pid = getattr(b, "pair_id", None)
        if pid is not None:
            pair_legs.setdefault(pid, []).append(b)

    pair_outcome: dict[str, str] = {}
    for pid, legs in pair_legs.items():
        any_stop = any(getattr(b, "stop_close", False) for b in legs)
        any_force = any(getattr(b, "force_close", False) for b in legs)
        any_close = any(getattr(b, "close_leg", False) for b in legs)
        if any_stop:
            pair_outcome[pid] = "stop_closed"
        elif any_force:
            pair_outcome[pid] = "force_closed"
        elif any_close:
            pair_outcome[pid] = "agent_closed"
        elif len(legs) >= 2:
            pair_outcome[pid] = "matured"
        else:
            pair_outcome[pid] = "naked"

    records: list[EvaluationBetRecord] = []
    for bet in bets:
        race = race_by_market.get(bet.market_id)
        race_idx = market_to_race_idx.get(bet.market_id, -1)

        runner_name = ""
        tick_timestamp = ""
        seconds_to_off = 0.0
        runner_pwin: float | None = None
        race_max_pwin: float | None = None

        if race is not None:
            meta = race.runner_metadata.get(bet.selection_id)
            if meta is not None:
                runner_name = meta.runner_name

            if 0 <= bet.tick_index < len(race.ticks):
                tick = race.ticks[bet.tick_index]
                tick_timestamp = tick.timestamp.isoformat()
                seconds_to_off = (
                    race.market_start_time - tick.timestamp
                ).total_seconds()

        if 0 <= race_idx < len(pwin_by_race):
            race_pwins = pwin_by_race[race_idx]
            if race_pwins:
                runner_pwin = float(race_pwins.get(bet.selection_id, 0.0))
                race_max_pwin = float(max(race_pwins.values()))

        pid = getattr(bet, "pair_id", None)
        if pid is None:
            final_outcome = "directional"
        else:
            final_outcome = pair_outcome.get(pid, "naked")

        records.append(EvaluationBetRecord(
            run_id="",  # patched by caller after create_evaluation_run
            date=day.date,
            market_id=bet.market_id,
            tick_timestamp=tick_timestamp,
            seconds_to_off=seconds_to_off,
            runner_id=bet.selection_id,
            runner_name=runner_name,
            action=bet.side.value,
            price=bet.average_price,
            stake=bet.matched_stake,
            matched_size=bet.matched_stake,
            outcome=bet.outcome.value,
            pnl=bet.pnl,
            opportunity_window_s=0.0,
            is_each_way=bet.is_each_way,
            each_way_divisor=bet.each_way_divisor,
            number_of_places=bet.number_of_places,
            settlement_type=bet.settlement_type,
            effective_place_odds=bet.effective_place_odds,
            starting_budget=starting_budget,
            pair_id=pid,
            fill_prob_at_placement=getattr(bet, "fill_prob_at_placement", None),
            predicted_locked_pnl_at_placement=getattr(
                bet, "predicted_locked_pnl_at_placement", None,
            ),
            predicted_locked_stddev_at_placement=getattr(
                bet, "predicted_locked_stddev_at_placement", None,
            ),
            # v2-aux-head-bet-plumbing (2026-05-24) — additional per-runner
            # aux-head snapshots the v2 rollout collector stamps at
            # placement time. ``getattr`` with default-None keeps the
            # call site backward-compatible with v1 bets and any legacy
            # ``Bet`` objects without these slots.
            mature_prob_at_placement=getattr(
                bet, "mature_prob_at_placement", None,
            ),
            direction_back_prob_at_placement=getattr(
                bet, "direction_back_prob_at_placement", None,
            ),
            direction_lay_prob_at_placement=getattr(
                bet, "direction_lay_prob_at_placement", None,
            ),
            close_leg=bool(getattr(bet, "close_leg", False)),
            force_close=bool(getattr(bet, "force_close", False)),
            stop_close=bool(getattr(bet, "stop_close", False)),
            runner_champion_p_win=runner_pwin,
            race_max_pwin=race_max_pwin,
            final_outcome=final_outcome,
        ))

    return records


# ── PBT warm-start (plans/pbt-breeding Step 1) ──────────────────────────


def load_warm_start_weights(
    policy: "torch.nn.Module",
    init_weights_path: "str | Path",
) -> None:
    """Load an inherited ``state_dict`` into ``policy`` in place (strict).

    PBT warm-start (``plans/pbt-breeding`` Step 1, HC#5/#10). The whole
    PBT mechanism rests on weight inheritance being REAL: a warm-started
    child must load the parent's ACTUAL trained weights so its gen-0
    forward reproduces the parent's final forward before any new gradient
    step. (The gene-only GA threw the weights away and re-trained from
    scratch every generation, so a champion's identity never reproduced.)

    ``init_weights_path`` points at a registry weights file
    (``registry/weights/<model_id>.pt``) saved by
    :meth:`ModelStore.save_weights`, which wraps the raw ``state_dict`` as
    ``{"weights": ..., "obs_schema_version": N, "action_schema_version":
    M}``. We unwrap that envelope (mirroring :meth:`ModelStore.load_weights`)
    and ``load_state_dict(..., strict=True)``.

    **Strict is load-bearing (HC#10).** The warm-start contract requires
    the child's weight shapes to match the parent's exactly. A structural-
    gene mismatch (different ``architecture`` / ``hidden_size`` / aux-head
    layout) therefore raises a ``RuntimeError`` HERE — loud — rather than
    silently truncating the inherited tensor into a garbled brain. The
    breed step (Step 2) keeps structural genes frozen within a lineage so
    this never fires in normal operation; if it does, the lineage
    bookkeeping has a bug.

    This is THE single warm-start load path — both the worker
    (``train_one_agent``) and any reeval/factory caller load through this
    function so the inherited brain is reconstructed identically (HC#11).
    The policy is mutated in place; nothing is returned. Forward-match
    gate: ``tests/test_v2_pbt_warm_start.py``.
    """
    path = Path(init_weights_path)
    if not path.exists():
        raise FileNotFoundError(
            f"init_weights_path does not exist: {path}",
        )
    # map_location='cpu': the policy is still on CPU at warm-start time
    # (the trainer moves it to ``device`` afterwards), and we don't want
    # to allocate GPU memory for the checkpoint. ``load_state_dict`` copies
    # values into the existing params regardless of the loaded tensors'
    # device, so this is correct for CUDA runs too.
    raw = torch.load(str(path), weights_only=True, map_location="cpu")
    if isinstance(raw, dict) and "weights" in raw:
        state_dict = raw["weights"]
    else:
        # Tolerate a bare state_dict (forward-compat with any caller that
        # saved one without the ModelStore version envelope).
        state_dict = raw
    policy.load_state_dict(state_dict, strict=True)


def _load_bc_oracle_or_skip(
    *,
    dates: list[str],
    data_dir: Path,
    obs_dim: int,
    agent_id: str,
) -> list | None:
    """Load BC oracle samples, or return ``None`` (skip BC) when the oracle's
    obs config doesn't match this agent's ``obs_dim``.

    The oracle cache is scanned at ONE obs config (one ``obs_dim``).
    ``load_oracle_samples_for_dates`` does a STRICT obs_dim check and RAISES
    ``ValueError`` on a mismatch. An agent whose obs config differs — e.g. a
    lean-obs fresh-blood lineage (``obs_dim`` 574) when only a full-obs oracle
    (2254) exists — must therefore SKIP BC gracefully, NOT crash the worker.
    Returns ``None`` on mismatch so the caller falls through to its
    "no samples -> skip BC" path; the agent still trains via PPO (+ warm-start
    for non-fresh lineages). pbt-breeding 2026-06-04: a lean agent crashed the
    multiprocess pool here ("Cache obs_dim=2254 but caller expects 574").
    Guarded by ``tests/test_v2_bc_obs_dim_mismatch.py``.
    """
    try:
        return load_oracle_samples_for_dates(
            dates=list(dates),
            data_dir=data_dir,
            expected_obs_dim=int(obs_dim),
        )
    except ValueError as err:
        logger.warning(
            "Agent %s: BC oracle obs_dim mismatch (%s) — this agent's obs "
            "config (obs_dim=%d) has no matching oracle cache; SKIPPING BC "
            "(it still trains via PPO + warm-start). Scan a matching oracle "
            "to enable BC for this obs config.",
            agent_id, err, int(obs_dim),
        )
        return None


# ── Main entry point ────────────────────────────────────────────────────


def train_one_agent(
    *,
    agent_id: str,
    genes: CohortGenes,
    days_to_train: list[str],
    eval_days: list[str] | None = None,
    eval_day: str | None = None,  # legacy single-day kwarg
    data_dir: Path,
    device: str,
    seed: int,
    model_store: ModelStore | None = None,
    scorer_dir: Path = DEFAULT_SCORER_DIR,
    generation: int = 0,
    parent_a_id: str | None = None,
    parent_b_id: str | None = None,
    starting_budget: float = 100.0,
    event_emitter: Callable[[dict], None] | None = None,
    agent_idx: int = 0,
    n_agents: int = 1,
    reward_overrides: dict | None = None,
    enabled_set: frozenset[str] = frozenset(),
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
    composite_score_mode: str = "total_reward",
    feature_cache: dict[str, list] | None = None,
    static_obs_cache: dict | None = None,
    frozen_direction_head_path: "Path | None" = None,
    init_weights_path: "str | Path | None" = None,
    gpu_policy_lane: bool = False,
) -> AgentResult:
    """Train one agent through ``days_to_train`` and eval on ``eval_days``.

    Parameters
    ----------
    agent_id:
        Stable string identifier — used as ``model_id`` if
        ``model_store`` is provided. UUIDs preferred so the registry
        primary key doesn't collide.
    genes:
        Locked Phase 3 schema (:class:`CohortGenes`).
    days_to_train, eval_day:
        Already-resolved date strings. The runner does the day
        selection / shuffle; the worker takes the result.
    model_store:
        Optional :class:`ModelStore`. When supplied the worker creates
        a model row, saves weights, creates an evaluation_run, and
        records one ``evaluation_days`` row for ``eval_day``. When
        ``None`` (e.g. unit-test runs), the worker still computes the
        full :class:`AgentResult` but doesn't write to disk — used by
        Session 03's lightweight integration test.
    event_emitter:
        Optional ``Callable[[dict], None]``. When supplied, called with
        v1-shape websocket events at the agent-start, per-episode, and
        agent-complete points (Session 04 deliverable). Pass ``None``
        for silent runs (unit tests, scripted use). Per-episode events
        carry detail strings the frontend's ``extractChartData`` regex
        parses for live reward / loss charts.
    agent_idx, n_agents:
        Position within the current generation's cohort. Used only for
        the ``progress`` snapshot's ``completed`` / ``total`` fields on
        emitted events. Defaults of ``0`` / ``1`` produce a sensible
        single-agent snapshot when the worker is invoked outside a
        cohort context.
    init_weights_path:
        PBT warm-start (``plans/pbt-breeding`` Step 1). When set, the
        agent loads an inherited ``state_dict`` (a parent's or its own
        prior-generation weights, saved at
        ``registry/weights/<model_id>.pt``) into the freshly-built
        policy BEFORE any BC / PPO, then continues with a PPO fine-tune
        — this is what makes a champion's learned identity heritable
        across generations. ``None`` (the default) = cold-start =
        byte-identical to the pre-pbt gene-only path (HC#1). A
        warm-started agent SKIPS BC pretrain (it already has a trained
        actor_head; re-running BC would overwrite it). The load is
        strict, so a structural-gene mismatch raises (HC#10).

    Returns
    -------
    :class:`AgentResult` carrying the genes, train summary, and eval
    summary. The runner sorts agents by ``result.eval.total_reward``
    for breeding selection.
    """
    assert_in_range(genes)
    if not days_to_train:
        raise ValueError("days_to_train must contain at least one date.")
    # Accept legacy ``eval_day=...`` kwarg (single string) for backward
    # compat with existing callers and the integration-test stub.
    if eval_days is None:
        if eval_day is None:
            raise ValueError(
                "Either eval_days (list) or eval_day (single string) "
                "must be provided.",
            )
        eval_days = [str(eval_day)]
    if not eval_days:
        raise ValueError("eval_days must contain at least one date.")

    # GPU policy lane (plans/pbt-gpu-forward, 2026-06-04): a big-ctx transformer
    # routes its WHOLE policy — the batch=1 attention forward AND the batched
    # PPO update — to CUDA; the env stays on CPU. The O(ctx^2) forward wins on
    # GPU (measured 6.3x at ctx256) where an LSTM's batch=1 GEMV does not, so
    # only is_gpu_lane_eligible() agents flip. Resolved ONCE here so the seed,
    # policy build, BC, and trainer all see the right device. Default off →
    # device unchanged → byte-identical to the pure-CPU R5 path.
    _gpu_lane = (
        bool(gpu_policy_lane)
        and is_gpu_lane_eligible(genes)
        and torch.cuda.is_available()
    )
    if _gpu_lane:
        device = "cuda"

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    cfg = scalping_train_config()
    cfg["training"]["starting_budget"] = float(starting_budget)
    max_runners = int(cfg["training"]["max_runners"])

    # Predictor-integration: inject strategy_mode + observations flag
    # into cfg so the env's resolution path picks them up. Both are
    # optional kwargs at the agent level — the runner threads them
    # through from CLI (`--strategy-mode`, `--use-race-outcome-predictor`).
    if strategy_mode is not None:
        cfg["training"]["strategy_mode"] = strategy_mode
    if use_race_outcome_predictor:
        cfg.setdefault("observations", {})
        cfg["observations"]["use_race_outcome_predictor"] = True
    if use_direction_predictor:
        cfg.setdefault("observations", {})
        cfg["observations"]["use_direction_predictor"] = True

    # Phase 5 (2026-05-03): combine cohort-level overrides with this
    # agent's enabled-gene values. Disabled genes contribute nothing,
    # so a launch without ``--enable-gene`` flags reproduces the
    # pre-Phase-5 reward_overrides shape byte-identically.
    per_agent_reward_overrides = _build_per_agent_reward_overrides(
        cohort_overrides=reward_overrides,
        genes=genes,
        enabled_set=enabled_set,
    )
    per_agent_scalping_overrides = _build_per_agent_scalping_overrides(
        genes=genes,
        enabled_set=enabled_set,
    )
    # Cohort-wide pin for arb_spread_target_lock_pct wins over the
    # gene/enable path. The runner's CLI guard rejects
    # --arb-spread-target-lock-pct combined with --enable-gene
    # arb_spread_target_lock_pct (one source of truth per knob).
    if arb_spread_target_lock_pct_override is not None:
        per_agent_scalping_overrides = dict(per_agent_scalping_overrides or {})
        per_agent_scalping_overrides["arb_spread_target_lock_pct"] = float(
            arb_spread_target_lock_pct_override,
        )
    # Phase 7 Session 02 (Path A). Pre-merge any reward_overrides for
    # trainer-side keys into the per-agent hp dict so the trainer's
    # ``hp.get(name, 0.0)`` reads the override. NEVER add a config
    # fallback inside the trainer — that would re-introduce the v1
    # precedence trap (lessons_learnt.md).
    trainer_hp = _build_trainer_hp(
        cohort_overrides=reward_overrides,
        genes=genes,
        enabled_set=enabled_set,
    )
    # Phase 9 S02 — cohort-wide flag (NOT a gene). Threaded through hp
    # so the trainer's ``hp.get("per_transition_credit", False)`` read
    # picks it up. The default ``False`` keeps Phase 7 byte-identity
    # for runs that don't pass the flag (hard_constraints.md §1, §6).
    trainer_hp["per_transition_credit"] = bool(per_transition_credit)

    # ── Build first-day env + shim to size the policy ────────────────
    first_day = days_to_train[0]
    logger.info(
        "Agent %s: loading first day %s from %s",
        agent_id, first_day, data_dir,
    )
    env, shim = _build_env_for_day(
        day_str=first_day, data_dir=data_dir, cfg=cfg, scorer_dir=scorer_dir,
        reward_overrides=per_agent_reward_overrides,
        scalping_overrides=per_agent_scalping_overrides,
        predictor_bundle=predictor_bundle,
        predictor_lean_obs=predictor_lean_obs,
        predictor_p_win_back_threshold=predictor_p_win_back_threshold,
        predictor_p_win_back_max_threshold=predictor_p_win_back_max_threshold,
        predictor_p_win_lay_threshold=predictor_p_win_lay_threshold,
        direction_gate_enabled=direction_gate_enabled,
        race_confidence_threshold=race_confidence_threshold,
        lay_price_max=lay_price_max,
        feature_cache=feature_cache,
        static_obs_cache=static_obs_cache,
    )

    # Phase-14 S03 — direction gate config flows through trainer_hp
    # (which the cohort runner constructs from CohortGenes +
    # --reward-overrides). When ``direction_gate_enabled`` isn't in
    # the override dict, fall back to the function-arg (the cohort
    # runner's ``--direction-gate-enabled`` CLI flag). The gene
    # default (False) only wins when neither the override nor the
    # CLI flag is set. The threshold gene evolves per-agent if the
    # operator opted in via ``--enable-gene direction_gate_threshold``.
    #
    # 2026-05-24 (commit `<this one>`) — ACTUAL CODE FIX. The earlier
    # comment claimed this had been fixed but the code below was
    # unchanged. The bug: ``trainer_hp`` is built from CohortGenes via
    # ``to_dict()`` which ALWAYS includes ``"direction_gate_enabled":
    # False`` (the dataclass default). So ``.get(key, fallback)``
    # never reached the fallback — the function-arg (the CLI flag's
    # value) was silently discarded and replaced with False from the
    # gene dict. Result: ``--direction-gate-enabled`` only enabled the
    # ENV-side gate (which uses dir_fire_drift from the upstream
    # Conv1D predictor) and the POLICY-side gate (which uses
    # max(direction_back_prob, direction_lay_prob) >= threshold from
    # the frozen C11 head) stayed OFF. Verified in cohort
    # ``_recipe_sensitivity_sweep_1779661887``: every agent showed
    # ``gate_refusals=0`` despite a non-noop threshold gene draw.
    #
    # Fix: OR semantics — enable if EITHER the CLI flag OR the
    # reward-override says enable. See _resolve_direction_gate_enabled
    # for the documented contract and the regression test in
    # tests/test_v2_direction_gate.py::TestResolvePolicyGateEnabled.
    direction_gate_enabled = _resolve_direction_gate_enabled(
        cli_flag=direction_gate_enabled,
        trainer_hp=trainer_hp,
    )
    direction_gate_threshold = float(
        trainer_hp.get(
            "direction_gate_threshold",
            getattr(genes, "direction_gate_threshold", 0.5),
        ),
    )

    # fc-cost-probe D (2026-05-17): policy-arch flag for the
    # strict-fc aux head. Read from cohort-level reward_overrides
    # (passthrough whitelist). Default False = byte-identical to
    # pre-probe-D arch.
    enable_fc_prob_head = bool(
        (per_agent_reward_overrides or {}).get(
            "enable_fc_prob_head", False,
        ),
    )
    # 2026-05-24 fix: env.active_runner_dim is REQUIRED — pre-fix
    # the cohort silently defaulted to 143 (full obs) when the env
    # was actually emitting 23-dim per-runner lean obs, producing
    # structurally garbage head input. If a future env refactor
    # drops the attribute, raise immediately rather than re-introduce
    # the silent fallback. See plans/direction-predictor-label-
    # alignment/.
    if not hasattr(shim.env, "active_runner_dim"):
        raise RuntimeError(
            "shim.env is missing `active_runner_dim` — the env "
            "must expose its per-runner block dim (23 under lean "
            "obs, 143 under full obs) so the policy can size "
            "direction_prob_head correctly. The cohort runner "
            "REFUSES to fall back to a silent default of 143 "
            "after the 2026-05-24 incident."
        )
    # Path C (2026-05-30): mature_prob open-gate. Plumbed as a single
    # float (gate enabled iff > 0.0) straight to the policy ctor — NOT
    # via trainer_hp.get(), which would hit the Path-A precedence foot
    # gun (CohortGenes.to_dict() always populates the key, swallowing
    # the CLI value). Log when active so the operator can confirm
    # wiring on the first agent (the foot-gun detection rule).
    if float(mature_prob_open_threshold) > 0.0:
        logger.info(
            "Agent %s: mature_prob open-gate ACTIVE — threshold=%.4f "
            "(opens masked where mature_prob < threshold; refusals "
            "surface in direction_gate_refusals)",
            agent_id, float(mature_prob_open_threshold),
        )
    # ── Build the policy via the SINGLE factory (pbt-breeding HC#11) ──
    # ``build_policy`` reads the structural genes (architecture + sizing)
    # off ``genes`` and dispatches to DiscreteLSTMPolicy /
    # DiscreteTransformerPolicy. With ``genes.architecture == "lstm"``
    # (every existing cohort + the gene-only GA) this is byte-identical to
    # the prior inline ``DiscreteLSTMPolicy(...)`` construction.
    # tools/reevaluate_cohort.py builds through the SAME factory, so the
    # trained policy and the held-out re-eval policy are the same module —
    # the input_norm divergence that bit us is structurally impossible now.
    #
    # input_norm (full obs): the 2254-d predictor-injected obs has raw
    # dims up to ~190k that dominate the input_proj Linear and drown the
    # well-scaled features (imitation-first Step 1b / memory
    # feedback_full_obs_needs_input_norm). Register per-dim (mean, std)
    # buffers; stats are set from the BC oracle obs just below (before
    # BC/PPO) for fresh blood, or INHERITED via warm-start (Step 1).
    # Default-unset buffers are (0, 1) → no-op, so this is safe even when
    # BC is off.
    policy = build_policy(
        genes,
        obs_dim=shim.obs_dim,
        action_space=shim.action_space,
        runner_dim=int(shim.env.active_runner_dim),
        input_norm=True,
        direction_gate_enabled=direction_gate_enabled,
        direction_gate_threshold=direction_gate_threshold,
        mature_prob_open_threshold=float(mature_prob_open_threshold),
        enable_fc_prob_head=enable_fc_prob_head,
        frozen_direction_head_path=frozen_direction_head_path,
    )

    # ── PBT warm-start (plans/pbt-breeding Step 1) ───────────────────
    # Load an inherited brain (a parent's or this lineage's own
    # prior-generation weights) into the freshly-built policy BEFORE any
    # BC / PPO. This is what makes a champion's learned IDENTITY (its
    # weights), not just its recipe, heritable across generations — the
    # gene-only GA re-trained every agent from scratch each gen, so
    # champions never reproduced. ``init_weights_path=None`` (the
    # default) is cold-start = byte-identical to the pre-pbt path (HC#1).
    # The load is strict: a structural-gene mismatch (architecture /
    # hidden_size / aux-head layout) raises here rather than silently
    # truncating (HC#10). The forward-match gate
    # (tests/test_v2_pbt_warm_start.py) proves the inherited brain
    # reproduces the parent's forward on a fixed obs before any new
    # gradient step (HC#5).
    warm_started = False
    if init_weights_path is not None:
        load_warm_start_weights(policy, init_weights_path)
        warm_started = True
        logger.info(
            "Agent %s: WARM-START — loaded inherited weights from %s "
            "(strict); will skip BC and continue with a PPO fine-tune.",
            agent_id, init_weights_path,
        )

    # 2026-05-24: when a shared frozen direction head is loaded, the
    # supervised loss weights MUST be zero (hard_constraints §4 of
    # plans/shared-direction-head/). Force them here even if the
    # operator's gene overrides set them — the head's parameters are
    # frozen so any non-zero supervised gradient would attempt to
    # update parameters whose `requires_grad=False`, which torch
    # would silently ignore but logging the BCE metric anyway gives
    # a misleading impression of "head is being trained."
    if frozen_direction_head_path is not None:
        if trainer_hp.get("direction_prob_loss_weight", 0.0) > 0.0:
            logger.info(
                "Agent %s: FROZEN direction head loaded — forcing "
                "direction_prob_loss_weight 0.0 (was %.4f)",
                agent_id,
                float(trainer_hp.get("direction_prob_loss_weight", 0.0)),
            )
            trainer_hp["direction_prob_loss_weight"] = 0.0
        if trainer_hp.get("bc_direction_target_weight", 0.0) > 0.0:
            logger.info(
                "Agent %s: FROZEN direction head loaded — forcing "
                "bc_direction_target_weight 0.0 (was %.4f)",
                agent_id,
                float(trainer_hp.get("bc_direction_target_weight", 0.0)),
            )
            trainer_hp["bc_direction_target_weight"] = 0.0

    # phase-3 Option A: when training on CUDA, the rollout's policy
    # forward at batch=1 is dominated by per-op kernel-launch overhead;
    # running it on CPU is ~33% faster end-to-end while the PPO update
    # (mini-batch=64+) still benefits from CUDA. Single-device runs
    # ("--device cpu") keep the rollout on CPU and so retain byte-
    # identical behaviour. The trainer handles the policy + optimiser
    # round-trip per episode.
    # rollout_device: the GPU lane keeps the batch=1 forward ON cuda — a big-ctx
    # transformer's attention forward wins there (the whole point of the lane).
    # The default split-device path (an incidental --device cuda on an LSTM)
    # keeps the launch-bound batch=1 forward on CPU and only the update on GPU.
    if _gpu_lane:
        rollout_device = "cuda"
    else:
        rollout_device = "cpu" if str(device) == "cuda" else device
    trainer = DiscretePPOTrainer(
        policy=policy,
        shim=shim,
        learning_rate=float(genes.learning_rate),
        gamma=0.99,
        gae_lambda=float(genes.gae_lambda),
        clip_range=float(genes.clip_range),
        entropy_coeff=float(genes.entropy_coeff),
        value_coeff=float(genes.value_coeff),
        ppo_epochs=4,
        mini_batch_size=int(genes.mini_batch_size),
        max_grad_norm=0.5,
        device=device,
        rollout_device=rollout_device,
        hp=trainer_hp,
    )

    # ── Phase 8 S02: BC pretrain (optional) ──────────────────────────
    # Runs BEFORE the day loop, in-place on ``policy``. The
    # ``bc_pretrain_steps_override`` from the cohort runner pins the
    # per-agent gene cohort-wide; absent the override each agent uses
    # its own ``genes.bc_pretrain_steps`` (default 0 = no-op,
    # byte-identical to pre-S02). The PPO optimiser already exists
    # but its state is untouched by BC — the BC pretrainer holds its
    # own Adam over actor_head parameters only and the per-param
    # freeze restores ``requires_grad=True`` on exit.
    bc_steps = (
        int(bc_pretrain_steps_override)
        if bc_pretrain_steps_override is not None
        else int(getattr(genes, "bc_pretrain_steps", 0))
    )
    # PBT warm-start (Step 1): a warm-started agent inherits a trained
    # actor_head. Re-running BC would overwrite it from this agent's
    # oracle obs (and re-set the input_norm buffers away from the
    # parent's inherited stats), partially undoing the inheritance.
    # Warm-start is a pure PPO fine-tune — skip BC. Fresh blood / cold
    # agents are unaffected (init_weights_path is None for them, so
    # warm_started is False and BC runs exactly as before — HC#1).
    if warm_started and bc_steps > 0:
        logger.info(
            "Agent %s: warm-start active — skipping BC pretrain "
            "(%d steps) to preserve the inherited weights.",
            agent_id, bc_steps,
        )
        bc_steps = 0
    # Pre-existing scoping fix (caught by predictor-integration smoke
    # 2026-05-10): `direction_bce_label_map` is only assigned inside
    # the `if bc_steps > 0:` branch but referenced unconditionally at
    # line 1308 in the post-PPO direction-BCE diagnostic. Default to
    # None up-front so the diagnostic short-circuits cleanly when BC
    # is off.
    direction_bce_label_map = None
    bc_samples = None
    if bc_steps > 0:
        bc_lr = (
            float(bc_learning_rate_override)
            if bc_learning_rate_override is not None
            else float(getattr(genes, "bc_learning_rate", 3e-4))
        )
        bc_warmup_eps = (
            int(bc_target_entropy_warmup_eps_override)
            if bc_target_entropy_warmup_eps_override is not None
            else int(getattr(genes, "bc_target_entropy_warmup_eps", 5))
        )
        # Push the operator's override into the trainer's warmup
        # state — it was constructed reading the gene default. Skip
        # when no override (gene default already in place).
        if bc_target_entropy_warmup_eps_override is not None:
            trainer._bc_warmup_eps = bc_warmup_eps
        bc_samples = _load_bc_oracle_or_skip(
            dates=list(days_to_train),
            data_dir=data_dir,
            obs_dim=int(shim.obs_dim),
            agent_id=str(agent_id),
        )
        # input_norm: set per-dim standardization stats from the BC oracle
        # obs BEFORE BC + PPO (so the LSTM trains on normalized obs). Only
        # when the policy was built with input_norm=True and BC samples
        # exist. See the policy-construction comment above.
        if bc_samples and getattr(policy, "_input_norm_enabled", False):
            _norm_src = np.stack(
                [s.obs for s in bc_samples[:100000]], axis=0,
            ).astype(np.float64)
            policy.set_input_norm_stats(
                _norm_src.mean(axis=0), _norm_src.std(axis=0),
            )
            logger.info(
                "Agent %s: input_norm stats set from %d BC oracle obs "
                "(std range [%.3g, %.3g])",
                agent_id, len(_norm_src),
                float(_norm_src.std(axis=0).min()),
                float(_norm_src.std(axis=0).max()),
            )
            del _norm_src
        if not bc_samples:
            logger.warning(
                "Agent %s: bc_pretrain_steps=%d but no oracle samples "
                "loaded across %d training day(s); skipping BC. Run "
                "`python -m training_v2.oracle_cli scan --dates %s` "
                "to populate caches.",
                agent_id, bc_steps, len(days_to_train),
                ",".join(days_to_train),
            )
        else:
            # Phase-13 S05 — direction-targeted BC layered with the
            # oracle target. Operator-controlled via
            # ``--reward-overrides bc_direction_target_weight=X``;
            # defaults to 0.0 (oracle-only, byte-identical to phase-8).
            bc_dir_w = float(
                trainer_hp.get("bc_direction_target_weight", 0.0) or 0.0
            )
            direction_target_map: dict[tuple[int, int], int] | None = None
            if bc_dir_w > 0.0:
                # The direction labels are keyed by tick_index aligned
                # with the SAME pre-race tick numbering that
                # ``arb_oracle.scan_day`` uses (verified by both modules
                # consuming the env's deterministic tick walk). We
                # collapse the per-day label lists into a single global
                # ``(tick_index, runner_idx) → action`` map by stitching
                # them per-day under their offsets — the oracle samples
                # carry day-local tick indices, NOT global-across-days,
                # so the map's keys are also day-local. This works
                # because BC is keyed on a per-day shuffled mini-batch
                # and each oracle sample comes from one day's cache;
                # cross-day key collision is benign (same tick index in
                # different days favouring the same direction is just
                # the same target action).
                day_to_labels = load_direction_labels_for_dates(
                    dates=list(days_to_train),
                    data_dir=data_dir,
                    direction_horizon_ticks=int(
                        trainer_hp.get("direction_horizon_ticks", 60),
                    ),
                    direction_threshold_ticks=int(
                        trainer_hp.get(
                            "direction_threshold_ticks", 5,
                        ),
                    ),
                    force_close_before_off_seconds=float(
                        trainer_hp.get(
                            "direction_force_close_seconds", 60.0,
                        ),
                    ),
                )
                merged: list = []
                for _d, labs in day_to_labels.items():
                    merged.extend(labs)
                direction_target_map = build_direction_target_map(
                    merged, shim.action_space,
                )
                # Phase-15 S02 amendment: also build the raw binary
                # label map so BC can train direction_prob_head
                # directly via BCE-with-logits.
                direction_bce_label_map = build_direction_bce_label_map(
                    merged, shim.action_space,
                )
                logger.info(
                    "Agent %s: direction BC enabled (weight=%.3f, "
                    "%d unambiguous targets across %d day(s); "
                    "direction-head BCE pool: %d entries)",
                    agent_id, bc_dir_w,
                    len(direction_target_map), len(days_to_train),
                    len(direction_bce_label_map),
                )
            else:
                direction_bce_label_map = None
            # BC label augmentation Phase A
            # (``plans/bc-label-augmentation/``). Load the negative-
            # open cache when the CLI flag is set; otherwise pass
            # ``None`` so the pretrainer's negative-sample code path
            # stays gated off (byte-identical to pre-plan).
            bc_negative_samples = None
            if bc_include_negative_samples:
                bc_negative_samples = load_negative_samples_for_dates(
                    dates=list(days_to_train),
                    data_dir=data_dir,
                    expected_obs_dim=int(shim.obs_dim),
                )
                if not bc_negative_samples:
                    logger.warning(
                        "Agent %s: --bc-include-negative-samples set "
                        "but no negative cache loaded across %d "
                        "training day(s); BC will run positive-only. "
                        "Run `python -m training_v2.oracle_cli scan "
                        "--include-negative-samples --dates %s` to "
                        "populate.",
                        agent_id, len(days_to_train),
                        ",".join(days_to_train),
                    )
                else:
                    logger.info(
                        "Agent %s: BC negative-sample augmentation "
                        "active (%d negatives across %d day(s), "
                        "positive_weight=%.3f)",
                        agent_id, len(bc_negative_samples),
                        len(days_to_train), float(bc_positive_weight),
                    )
            # BC label augmentation Phase B
            # (``plans/bc-label-augmentation/``). Load the close/hold
            # cache when the CLI flag is set; otherwise pass ``None``
            # so the pretrainer's close/hold code path stays gated off
            # (byte-identical to Phase A).
            bc_close_hold_samples = None
            if bc_include_close_hold_samples:
                bc_close_hold_samples = (
                    load_close_hold_samples_for_dates(
                        dates=list(days_to_train),
                        data_dir=data_dir,
                        expected_obs_dim=int(shim.obs_dim),
                    )
                )
                if not bc_close_hold_samples:
                    logger.warning(
                        "Agent %s: --bc-include-close-hold-samples "
                        "set but no close/hold cache loaded across "
                        "%d training day(s); BC will run without "
                        "close/hold augmentation. Run `python -m "
                        "training_v2.oracle_cli scan --include-"
                        "close-hold-samples --dates %s` to populate.",
                        agent_id, len(days_to_train),
                        ",".join(days_to_train),
                    )
                else:
                    n_close = sum(
                        1 for s in bc_close_hold_samples
                        if s.target_action_class == 0
                    )
                    n_hold = sum(
                        1 for s in bc_close_hold_samples
                        if s.target_action_class == 1
                    )
                    logger.info(
                        "Agent %s: BC close/hold augmentation active "
                        "(%d samples across %d day(s): close=%d "
                        "hold=%d)",
                        agent_id, len(bc_close_hold_samples),
                        len(days_to_train), n_close, n_hold,
                    )
            bc_history = DiscreteBCPretrainer(
                lr=bc_lr, batch_size=64, seed=int(seed),
            ).pretrain(
                policy=policy,
                samples=bc_samples,
                n_steps=bc_steps,
                direction_target_map=direction_target_map,
                direction_target_weight=bc_dir_w,
                # Phase-15 S02: re-use bc_dir_w as the direction-
                # head BCE weight too. Until/unless we split it into
                # a separate gene, the same knob controls both the
                # actor-CE pull (toward the direction-derived action)
                # and the predictor-BCE pull (toward calibrated
                # binary outputs). Both are useful supervision; the
                # second is the load-bearing one for the gate.
                direction_bce_label_map=direction_bce_label_map,
                direction_bce_weight=bc_dir_w,
                # Phase-15 S02 amendment: pos_weight defaults TRUE
                # but can be turned off via reward_overrides
                # ``direction_bce_use_pos_weight=false``. v8 smoke
                # showed pos_weight may bias the predictor away
                # from true calibration (the loss optimum shifts
                # under pos_weight); v9 tests vanilla BCE.
                direction_bce_use_pos_weight=bool(
                    trainer_hp.get(
                        "direction_bce_use_pos_weight", True,
                    )
                ),
                negative_samples=bc_negative_samples,
                positive_weight=float(bc_positive_weight),
                close_hold_samples=bc_close_hold_samples,
            )
            post_bc_entropy = measure_post_bc_entropy(policy, bc_samples)
            trainer.set_post_bc_entropy(post_bc_entropy)
            # Phase-15 S02 diagnostic: measure post-BC direction
            # BCE so we can see whether BC actually calibrated
            # direction_prob_head. Without this measurement we
            # only see end-of-day BCE (post-PPO), which can't
            # distinguish "BC didn't move the head" from "BC
            # moved it then PPO un-calibrated it".
            post_bc_dir_bce_str = ""
            if direction_bce_label_map and bc_dir_w > 0.0:
                import torch as _torch
                _device = next(policy.parameters()).device
                _back_total = 0.0
                _lay_total = 0.0
                _n_total = 0
                # Eval in batches of 256 for memory.
                _n_eval = min(2048, len(bc_samples))
                _eval_idx = list(range(_n_eval))
                with _torch.no_grad():
                    for _start in range(0, _n_eval, 256):
                        _chunk = bc_samples[_start:_start + 256]
                        _obs = _torch.tensor(
                            np.stack([s.obs for s in _chunk], axis=0),
                            dtype=_torch.float32,
                            device=_device,
                        )
                        _out = policy(_obs)
                        _bb = _out.direction_back_logits_per_runner
                        _bl = _out.direction_lay_logits_per_runner
                        for _i, _s in enumerate(_chunk):
                            _key = (
                                int(_s.tick_index),
                                int(_s.runner_idx),
                            )
                            _labels = direction_bce_label_map.get(_key)
                            if _labels is None:
                                continue
                            _runner = int(_s.runner_idx)
                            _lb = float(_labels[0])
                            _ll = float(_labels[1])
                            _bb_logit = float(_bb[_i, _runner].item())
                            _bl_logit = float(_bl[_i, _runner].item())
                            # BCE-with-logits: log(1+exp(-x)) if y=1
                            # else log(1+exp(x)). Stable form:
                            #   max(x, 0) - x*y + log(1+exp(-|x|))
                            import math as _math
                            def _bce(x: float, y: float) -> float:
                                return (
                                    max(x, 0.0) - x * y
                                    + _math.log1p(_math.exp(-abs(x)))
                                )
                            _back_total += _bce(_bb_logit, _lb)
                            _lay_total += _bce(_bl_logit, _ll)
                            _n_total += 1
                if _n_total > 0:
                    _bb_mean = _back_total / _n_total
                    _bl_mean = _lay_total / _n_total
                    post_bc_dir_bce_str = (
                        f" post_bc_dir_bce_back={_bb_mean:.4f} "
                        f"lay={_bl_mean:.4f} (n={_n_total})"
                    )
            logger.info(
                "Agent %s: BC pretrain done — steps=%d samples=%d "
                "final_ce=%.4f post_entropy=%.3f (warmup_eps=%d, "
                "bc_lr=%g)%s",
                agent_id, bc_steps, len(bc_samples),
                bc_history.final_ce_loss, post_bc_entropy,
                bc_warmup_eps, bc_lr,
                post_bc_dir_bce_str,
            )
            # Phase-15 S02 amendment 3: freeze direction_prob_head
            # AFTER BC so PPO cannot drift the calibrated state.
            # Smoke v7 (registry/_phase15_smoke_bcv2_*) showed BC
            # achieves direction BCE 0.26/0.35 (better than the
            # supervised probe's 0.4-0.6 target), but 364 PPO updates
            # with the 0.1-weighted auxiliary BCE drag the head back
            # to the no-skill baseline ~1.05. The detach prevented
            # actor-pathway gradient from touching the head; the
            # auxiliary BCE itself was found to drift it. Freezing
            # eliminates both as a corruption source — the head is
            # treated as a fixed predictor (the operator's literal
            # "make it a per-horse feature" framing). LayerNorm gamma
            # / beta are also frozen so the input normalisation stays
            # at its BC-fitted values.
            if direction_bce_label_map and bc_dir_w > 0.0:
                _frozen_count = 0
                for _name, _p in policy.named_parameters():
                    if "direction_prob_head" in _name:
                        _p.requires_grad_(False)
                        _frozen_count += 1
                logger.info(
                    "Agent %s: direction_prob_head frozen post-BC "
                    "(%d parameter tensors); PPO cannot drift the "
                    "calibrated predictor.",
                    agent_id, _frozen_count,
                )

    arch_name = arch_name_for_genes(genes)
    if event_emitter is not None:
        try:
            event_emitter(agent_training_started_event(
                agent_id=str(agent_id),
                architecture_name=arch_name,
                generation=int(generation),
                agent_idx=int(agent_idx),
                n_agents=int(n_agents),
                genes=genes.to_dict(),
            ))
        except Exception:
            logger.exception("event_emitter raised on agent_training_started; continuing")

    # ── Registry: create model row up-front so we have a model_id ────
    # Predictor-integration Session 03 / data-bridging follow-on
    # (hard_constraints.md §7): the cohort row captures the
    # `strategy_mode` + the 3 predictor `experiment_id`s when a
    # bundle is in play, so re-eval tooling can refuse on mismatch.
    # Stuffed into the existing `hyperparameters` JSON column rather
    # than added as new SQL columns (avoids schema migration; see
    # `incoming/predictor-integration-data-bridging.md` recommendation).
    hp_for_registry = dict(genes.to_dict())
    hp_for_registry["strategy_mode"] = str(
        cfg.get("training", {}).get("strategy_mode", "arb")
    )
    if predictor_bundle is not None:
        hp_for_registry["predictor_champion_experiment_id"] = str(
            getattr(predictor_bundle, "champion_experiment_id", "")
        )
        hp_for_registry["predictor_ranker_experiment_id"] = str(
            getattr(predictor_bundle, "ranker_experiment_id", "")
        )
        hp_for_registry["predictor_direction_experiment_id"] = str(
            getattr(predictor_bundle, "direction_experiment_id", "")
        )
    if model_store is not None:
        model_id = model_store.create_model(
            generation=int(generation),
            architecture_name=arch_name,
            architecture_description=_ARCH_DESCRIPTION,
            hyperparameters=hp_for_registry,
            parent_a_id=parent_a_id,
            parent_b_id=parent_b_id,
            model_id=str(agent_id),
        )
    else:
        model_id = str(agent_id)

    # ── Multi-day training loop (Session 02) ─────────────────────────
    train_t0 = time.perf_counter()
    per_day_rows: list[dict] = []
    aggregate_hist: Counter[str] = Counter()
    total_steps = 0
    total_reward = 0.0

    for day_idx, day_str in enumerate(days_to_train):
        if day_idx > 0:
            logger.info(
                "Agent %s: day %d/%d loading %s",
                agent_id, day_idx + 1, len(days_to_train), day_str,
            )
            _, new_shim = _build_env_for_day(
                day_str=day_str, data_dir=data_dir, cfg=cfg,
                scorer_dir=scorer_dir,
                reward_overrides=per_agent_reward_overrides,
                scalping_overrides=per_agent_scalping_overrides,
                predictor_bundle=predictor_bundle,
                predictor_lean_obs=predictor_lean_obs,
                predictor_p_win_back_threshold=predictor_p_win_back_threshold,
        predictor_p_win_back_max_threshold=predictor_p_win_back_max_threshold,
                predictor_p_win_lay_threshold=predictor_p_win_lay_threshold,
                direction_gate_enabled=direction_gate_enabled,
                race_confidence_threshold=race_confidence_threshold,
                lay_price_max=lay_price_max,
                feature_cache=feature_cache,
                static_obs_cache=static_obs_cache,
            )
            _rebind_trainer(trainer, new_shim)

        stats: EpisodeStats = trainer.train_episode()
        total_steps += int(stats.n_steps)
        total_reward += float(stats.total_reward)
        for k, v in (stats.action_histogram or {}).items():
            aggregate_hist[k] += int(v)
        per_day_rows.append({
            "day_idx": day_idx,
            "day_str": day_str,
            "n_steps": int(stats.n_steps),
            "total_reward": float(stats.total_reward),
            "day_pnl": float(stats.day_pnl),
            "value_loss_mean": float(stats.value_loss_mean),
            "policy_loss_mean": float(stats.policy_loss_mean),
            "approx_kl_mean": float(stats.approx_kl_mean),
            "entropy_mean": float(stats.entropy_mean),
            "wall_time_sec": float(stats.wall_time_sec),
            # Phase-13 S06 follow-up (2026-05-07). Per-day aux-head BCE
            # / NLL means and direction-targets count. Surfaces a per-
            # day curve for the operator to confirm BCE trends down.
            "fill_prob_bce_mean": float(stats.fill_prob_bce_mean),
            "mature_prob_bce_mean": float(stats.mature_prob_bce_mean),
            "risk_nll_mean": float(stats.risk_nll_mean),
            "direction_back_bce_mean": float(
                stats.direction_back_bce_mean,
            ),
            "direction_lay_bce_mean": float(
                stats.direction_lay_bce_mean,
            ),
            "n_direction_targets": int(stats.n_direction_targets),
            "direction_prob_loss_weight_active": float(
                stats.direction_prob_loss_weight_active,
            ),
        })
        # Phase-15 (2026-05-24). Surface direction_gate_refusals +
        # arb_realised_lock_pct on the per-day log line so the
        # operator's in-flight monitor can detect (a) the gate
        # over-refusing and starving training, (b) the
        # arb_spread_target_lock_pct formula not delivering the
        # promised fraction. Both default to safe sentinels (0 / NaN)
        # on pre-Phase-15 runs.
        _arb_lock = stats.arb_realised_lock_pct
        _arb_lock_str = (
            f"{_arb_lock:+.3f}" if _arb_lock == _arb_lock else "nan"
        )
        logger.info(
            "Agent %s day %d/%d [%s] reward=%+.3f pnl=%+.2f "
            "value_loss=%.4f approx_kl=%.4f dir_bce_back=%.4f "
            "dir_bce_lay=%.4f n_dir_targets=%d "
            "gate_refusals=%d arb_realised_lock=%s wall=%.1fs",
            agent_id, day_idx + 1, len(days_to_train), day_str,
            stats.total_reward, stats.day_pnl,
            stats.value_loss_mean, stats.approx_kl_mean,
            stats.direction_back_bce_mean,
            stats.direction_lay_bce_mean,
            stats.n_direction_targets,
            stats.direction_gate_refusals,
            _arb_lock_str,
            stats.wall_time_sec,
        )
        if event_emitter is not None:
            try:
                event_emitter(episode_complete_event(
                    agent_id=str(agent_id),
                    architecture_name=arch_name,
                    generation=int(generation),
                    day_idx=int(day_idx),
                    n_days=len(days_to_train),
                    day_str=str(day_str),
                    episode_idx=int(day_idx),
                    total_reward=float(stats.total_reward),
                    day_pnl=float(stats.day_pnl),
                    value_loss_mean=float(stats.value_loss_mean),
                    policy_loss_mean=float(stats.policy_loss_mean),
                    approx_kl_mean=float(stats.approx_kl_mean),
                    n_steps=int(stats.n_steps),
                ))
            except Exception:
                logger.exception("event_emitter raised on episode_complete; continuing")

    train_wall = time.perf_counter() - train_t0
    n_d = max(len(days_to_train), 1)
    # Phase-13 S06 follow-up — direction-prob weight is constant
    # across an agent's training days (set once on trainer init), so
    # the "active" value on the summary is the value the trainer used.
    # ``trainer_hp`` was built before the loop; reading it here keeps
    # the summary self-contained even if a future plan starts varying
    # per-day genes.
    dir_weight_active = float(
        trainer_hp.get("direction_prob_loss_weight", 0.0) or 0.0
    )
    train_summary = TrainSummary(
        n_days=len(days_to_train),
        total_steps=total_steps,
        total_reward=total_reward,
        mean_reward=total_reward / n_d,
        mean_pnl=float(np.mean([r["day_pnl"] for r in per_day_rows])),
        mean_value_loss=float(
            np.mean([r["value_loss_mean"] for r in per_day_rows])
        ),
        mean_policy_loss=float(
            np.mean([r["policy_loss_mean"] for r in per_day_rows])
        ),
        mean_approx_kl=float(
            np.mean([r["approx_kl_mean"] for r in per_day_rows])
        ),
        wall_time_sec=train_wall,
        per_day_rows=per_day_rows,
        mean_fill_prob_bce=float(
            np.mean([r["fill_prob_bce_mean"] for r in per_day_rows])
        ),
        mean_mature_prob_bce=float(
            np.mean([r["mature_prob_bce_mean"] for r in per_day_rows])
        ),
        mean_risk_nll=float(
            np.mean([r["risk_nll_mean"] for r in per_day_rows])
        ),
        mean_direction_back_bce=float(
            np.mean(
                [r["direction_back_bce_mean"] for r in per_day_rows],
            )
        ),
        mean_direction_lay_bce=float(
            np.mean(
                [r["direction_lay_bce_mean"] for r in per_day_rows],
            )
        ),
        total_direction_targets=int(
            sum(r["n_direction_targets"] for r in per_day_rows)
        ),
        direction_prob_loss_weight_active=dir_weight_active,
    )

    # Phase-15 S02 diagnostic: measure direction BCE on the SAME
    # oracle pool used by the post-BC measurement, AFTER all PPO
    # updates. If the freeze worked, this should be ~identical to
    # the post-BC value. If it drifted upward, freeze failed (or
    # is bypassed by some Adam quirk).
    if (
        direction_bce_label_map
        and bc_dir_w > 0.0
        and bc_samples
    ):
        import torch as _torch
        _device2 = next(policy.parameters()).device
        _back_total2 = 0.0
        _lay_total2 = 0.0
        _n_total2 = 0
        _n_eval2 = min(2048, len(bc_samples))
        with _torch.no_grad():
            for _start2 in range(0, _n_eval2, 256):
                _chunk2 = bc_samples[_start2:_start2 + 256]
                _obs2 = _torch.tensor(
                    np.stack([s.obs for s in _chunk2], axis=0),
                    dtype=_torch.float32,
                    device=_device2,
                )
                _out2 = policy(_obs2)
                _bb2 = _out2.direction_back_logits_per_runner
                _bl2 = _out2.direction_lay_logits_per_runner
                for _i2, _s2 in enumerate(_chunk2):
                    _key2 = (
                        int(_s2.tick_index),
                        int(_s2.runner_idx),
                    )
                    _labels2 = direction_bce_label_map.get(_key2)
                    if _labels2 is None:
                        continue
                    _runner2 = int(_s2.runner_idx)
                    _lb2 = float(_labels2[0])
                    _ll2 = float(_labels2[1])
                    _bb_logit2 = float(_bb2[_i2, _runner2].item())
                    _bl_logit2 = float(_bl2[_i2, _runner2].item())
                    import math as _math2
                    def _bce2(x: float, y: float) -> float:
                        return (
                            max(x, 0.0) - x * y
                            + _math2.log1p(_math2.exp(-abs(x)))
                        )
                    _back_total2 += _bce2(_bb_logit2, _lb2)
                    _lay_total2 += _bce2(_bl_logit2, _ll2)
                    _n_total2 += 1
        if _n_total2 > 0:
            _bb_mean2 = _back_total2 / _n_total2
            _bl_mean2 = _lay_total2 / _n_total2
            logger.info(
                "Agent %s: POST-PPO direction BCE on BC oracle pool: "
                "back=%.4f lay=%.4f (n=%d)",
                agent_id, _bb_mean2, _bl_mean2, _n_total2,
            )

    # ── Eval rollouts (no PPO update) ───────────────────────────────
    # When ``eval_days`` has more than one date, we run an eval rollout
    # against each day independently and then aggregate the per-day
    # summaries into a single mean-aggregate. This averages out the
    # per-day naked-pnl variance that dominated the 2026-05-05
    # 24-agent cohort's signal (~£200 day_pnl spread on identical-gene
    # agents from naked-luck alone).
    per_day_summaries: list[EvalSummary] = []
    # Phase 2a (2026-05-13) — per-bet log capture. Records are built
    # inside the loop (the env is recreated per day and goes out of
    # scope when the next iteration starts) and the writer is called
    # after ``create_evaluation_run`` returns the real run_id.
    per_day_bet_records: list[tuple[str, list[EvaluationBetRecord]]] = []
    for ed in eval_days:
        eval_t0 = time.perf_counter()
        logger.info(
            "Agent %s: eval rollout on held-out day %s", agent_id, ed,
        )
        _, eval_shim = _build_env_for_day(
            day_str=ed, data_dir=data_dir, cfg=cfg,
            scorer_dir=scorer_dir,
            reward_overrides=per_agent_reward_overrides,
            scalping_overrides=per_agent_scalping_overrides,
            predictor_bundle=predictor_bundle,
            predictor_lean_obs=predictor_lean_obs,
            predictor_p_win_back_threshold=predictor_p_win_back_threshold,
        predictor_p_win_back_max_threshold=predictor_p_win_back_max_threshold,
            predictor_p_win_lay_threshold=predictor_p_win_lay_threshold,
            direction_gate_enabled=direction_gate_enabled,
            race_confidence_threshold=race_confidence_threshold,
            lay_price_max=lay_price_max,
            feature_cache=feature_cache,
            static_obs_cache=static_obs_cache,
        )
        # phase-3 Option A — eval has no PPO update so the rollout
        # always uses the trainer's rollout_device (CPU when on CUDA).
        # The policy was parked on rollout_device by the last
        # _update_from_batch ``finally`` block; eval just continues
        # from there.
        eval_collector = RolloutCollector(
            shim=eval_shim, policy=policy,
            device=str(trainer.rollout_device),
        )
        eval_batch = eval_collector.collect_episode(deterministic=argmax_eval)
        eval_summary_partial = _eval_rollout_stats(
            batch=eval_batch,
            last_info=eval_collector.last_info,
            action_space=eval_shim.action_space,
        )
        # Capture per-bet records BEFORE the env goes out of scope on
        # the next loop iteration. ``_build_eval_bet_records`` returns
        # an empty list when no bets were placed, which the writer
        # handles by short-circuiting.
        try:
            day_records = _build_eval_bet_records(
                env=eval_shim.env,
                day=eval_shim.env.day,
                starting_budget=float(eval_shim.env.starting_budget),
            )
        except Exception:
            logger.exception(
                "Agent %s: _build_eval_bet_records failed on %s; "
                "continuing without bet log",
                agent_id, ed,
            )
            day_records = []
        per_day_bet_records.append((ed, day_records))
        eval_wall = time.perf_counter() - eval_t0
        per_day_summaries.append(EvalSummary(
            eval_day=ed,
            total_reward=eval_summary_partial.total_reward,
            day_pnl=eval_summary_partial.day_pnl,
            n_steps=eval_summary_partial.n_steps,
            bet_count=eval_summary_partial.bet_count,
            winning_bets=eval_summary_partial.winning_bets,
            bet_precision=eval_summary_partial.bet_precision,
            pnl_per_bet=eval_summary_partial.pnl_per_bet,
            early_picks=eval_summary_partial.early_picks,
            profitable=eval_summary_partial.profitable,
            action_histogram=eval_summary_partial.action_histogram,
            arbs_completed=eval_summary_partial.arbs_completed,
            arbs_naked=eval_summary_partial.arbs_naked,
            arbs_closed=eval_summary_partial.arbs_closed,
            arbs_force_closed=eval_summary_partial.arbs_force_closed,
            arbs_stop_closed=eval_summary_partial.arbs_stop_closed,
            arbs_target_pnl_refused=eval_summary_partial.arbs_target_pnl_refused,
            pairs_opened=eval_summary_partial.pairs_opened,
            locked_pnl=eval_summary_partial.locked_pnl,
            naked_pnl=eval_summary_partial.naked_pnl,
            closed_pnl=eval_summary_partial.closed_pnl,
            force_closed_pnl=eval_summary_partial.force_closed_pnl,
            stop_closed_pnl=eval_summary_partial.stop_closed_pnl,
            direction_gate_refusals=(
                eval_summary_partial.direction_gate_refusals
            ),
            pwin_back_gate_refusals=(
                eval_summary_partial.pwin_back_gate_refusals
            ),
            pwin_lay_gate_refusals=(
                eval_summary_partial.pwin_lay_gate_refusals
            ),
            arb_realised_lock_pct=(
                eval_summary_partial.arb_realised_lock_pct
            ),
            wall_time_sec=eval_wall,
        ))
        logger.info(
            "Agent %s eval [%s] reward=%+.3f pnl=%+.2f bets=%d "
            "precision=%.3f arbs=%d/%d locked=%+.2f naked=%+.2f wall=%.1fs",
            agent_id, ed, per_day_summaries[-1].total_reward,
            per_day_summaries[-1].day_pnl, per_day_summaries[-1].bet_count,
            per_day_summaries[-1].bet_precision,
            per_day_summaries[-1].arbs_completed,
            per_day_summaries[-1].arbs_naked,
            per_day_summaries[-1].locked_pnl,
            per_day_summaries[-1].naked_pnl,
            per_day_summaries[-1].wall_time_sec,
        )

    eval_summary = aggregate_eval_summaries(per_day_summaries)
    if len(eval_days) > 1:
        logger.info(
            "Agent %s eval AGGREGATE across %d days: "
            "reward=%+.3f pnl=%+.2f bets=%d arbs=%d/%d locked=%+.2f "
            "naked=%+.2f (wall_sum=%.1fs)",
            agent_id, len(eval_days),
            eval_summary.total_reward, eval_summary.day_pnl,
            eval_summary.bet_count, eval_summary.arbs_completed,
            eval_summary.arbs_naked, eval_summary.locked_pnl,
            eval_summary.naked_pnl, eval_summary.wall_time_sec,
        )

    # ── Registry writes ─────────────────────────────────────────────
    weights_path = ""
    run_id = ""
    if model_store is not None:
        weights_path = model_store.save_weights(
            model_id=model_id,
            state_dict=policy.state_dict(),
            obs_schema_version=OBS_SCHEMA_VERSION,
            action_schema_version=ACTION_SCHEMA_VERSION,
        )
        run_id = model_store.create_evaluation_run(
            model_id=model_id,
            train_cutoff_date=days_to_train[-1],
            test_days=list(eval_days),
        )
        # Phase 2a (2026-05-13) — write per-bet parquet logs now that
        # ``run_id`` is known. Records were captured inside the eval-day
        # loop with ``run_id=""``; patch the real id before writing.
        for ed_, records in per_day_bet_records:
            if not records:
                continue
            for r in records:
                r.run_id = run_id
            try:
                model_store.write_bet_logs_parquet(
                    run_id=run_id, date=ed_, records=records,
                )
            except Exception:
                logger.exception(
                    "Agent %s: write_bet_logs_parquet failed for %s; "
                    "continuing — bet log will be missing for this day",
                    agent_id, ed_,
                )
        # Persist one EvaluationDayRecord per eval day. peek_cohort
        # already aggregates the per-day rows when summarising —
        # writing them per-day keeps the registry's lineage detail
        # intact rather than collapsing to a single row of means.
        for per_day in eval_summary.per_day or [eval_summary]:
            model_store.record_evaluation_day(EvaluationDayRecord(
                run_id=run_id,
                date=per_day.eval_day,
                day_pnl=per_day.day_pnl,
                bet_count=per_day.bet_count,
                winning_bets=per_day.winning_bets,
                bet_precision=per_day.bet_precision,
                pnl_per_bet=per_day.pnl_per_bet,
                early_picks=per_day.early_picks,
                profitable=per_day.profitable,
                starting_budget=float(starting_budget),
                arbs_completed=per_day.arbs_completed,
                arbs_naked=per_day.arbs_naked,
                locked_pnl=per_day.locked_pnl,
                naked_pnl=per_day.naked_pnl,
                arbs_closed=per_day.arbs_closed,
                arbs_force_closed=per_day.arbs_force_closed,
                arbs_stop_closed=per_day.arbs_stop_closed,
                arbs_target_pnl_refused=per_day.arbs_target_pnl_refused,
                pairs_opened=per_day.pairs_opened,
                closed_pnl=per_day.closed_pnl,
                force_closed_pnl=per_day.force_closed_pnl,
                stop_closed_pnl=per_day.stop_closed_pnl,
            ))
        # Composite score: by default = eval-day total_reward
        # (byte-identical to pre-plan). When
        # ``composite_score_mode == "locked_weighted"`` (scalping-
        # locked-fitness-and-age-obs plan) the registry's
        # composite_score column instead records
        # ``locked_pnl + 0.25 × naked_pnl`` so the model row matches
        # what the GA actually selected on at the runner level. The
        # 0.25 weight is locked in hard_constraints §9.
        if composite_score_mode == "locked_weighted":
            composite_score_value = (
                float(eval_summary.locked_pnl)
                + 0.25 * float(eval_summary.naked_pnl)
            )
        else:
            composite_score_value = float(eval_summary.total_reward)
        model_store.update_composite_score(
            model_id=model_id,
            score=composite_score_value,
        )

    # Drop trainer to release optimiser-state tensors before the next
    # agent constructs its own. Mirrors v1 ``training/run_training.py:799``.
    del trainer
    if device.startswith("cuda"):
        try:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    if event_emitter is not None:
        try:
            event_emitter(agent_training_complete_event(
                agent_id=str(agent_id),
                architecture_name=arch_name,
                generation=int(generation),
                agent_idx=int(agent_idx),
                n_agents=int(n_agents),
                eval_total_reward=float(eval_summary.total_reward),
                eval_day_pnl=float(eval_summary.day_pnl),
                eval_bet_count=int(eval_summary.bet_count),
                eval_bet_precision=float(eval_summary.bet_precision),
            ))
        except Exception:
            logger.exception("event_emitter raised on agent_training_complete; continuing")

    return AgentResult(
        agent_id=str(agent_id),
        model_id=str(model_id),
        architecture_name=arch_name,
        genes=genes,
        train=train_summary,
        eval=eval_summary,
        weights_path=str(weights_path),
        run_id=str(run_id),
    )
