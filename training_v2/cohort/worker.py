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
from agents_v2.discrete_policy import DiscreteLSTMPolicy
from agents_v2.env_shim import DEFAULT_SCORER_DIR, DiscreteActionShim
from data.episode_builder import load_day
from env.betfair_env import (
    ACTION_SCHEMA_VERSION,
    OBS_SCHEMA_VERSION,
    BetfairEnv,
)
from registry.model_store import EvaluationDayRecord, ModelStore
from training_v2.cohort.events import (
    agent_training_complete_event,
    agent_training_started_event,
    episode_complete_event,
)
from training_v2.cohort.genes import CohortGenes, assert_in_range
from training_v2.discrete_ppo.rollout import RolloutCollector
from training_v2.discrete_ppo.trainer import DiscretePPOTrainer, EpisodeStats


logger = logging.getLogger(__name__)


__all__ = [
    "AgentResult",
    "TrainSummary",
    "EvalSummary",
    "arch_name_for_genes",
    "scalping_train_config",
    "train_one_agent",
]


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


@dataclass(frozen=True)
class EvalSummary:
    """Eval-day metrics from the rollout-only pass."""

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
    locked_pnl: float = 0.0
    naked_pnl: float = 0.0
    wall_time_sec: float = 0.0


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
    """``v2_discrete_ppo_lstm_h{hidden_size}`` — registry discriminator.

    Different ``hidden_size`` → different ``arch_name``. The registry's
    weight-shape hash adds a second layer of protection (state dict
    shapes differ across hidden sizes), but the arch_name is the
    primary discriminator the UI / scoreboard reads. ``v1`` weights
    use a different prefix entirely, so v2 cohorts never collide with
    v1 cohorts in the same registry.
    """
    return f"v2_discrete_ppo_lstm_h{int(genes.hidden_size)}"


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
) -> tuple[BetfairEnv, DiscreteActionShim]:
    day = load_day(day_str, data_dir=data_dir)
    env = BetfairEnv(day, cfg)
    shim = DiscreteActionShim(env, scorer_dir=scorer_dir)
    return env, shim


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
    trainer._collector = RolloutCollector(
        shim=shim, policy=trainer.policy, device=str(trainer.device),
    )


def _eval_rollout_stats(
    *,
    transitions: list,
    last_info: dict,
    action_space,
) -> EvalSummary:
    """Build :class:`EvalSummary` from a rollout-only pass on the eval day."""
    n_steps = len(transitions)
    total_reward = float(
        sum(float(tr.per_runner_reward.sum()) for tr in transitions),
    )
    hist: dict[str, int] = {}
    for tr in transitions:
        kind, _runner = action_space.decode(int(tr.action_idx))
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
    locked_pnl = float(last_info.get("locked_pnl", 0.0))
    naked_pnl = float(last_info.get("naked_pnl", 0.0))

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
        locked_pnl=locked_pnl,
        naked_pnl=naked_pnl,
    )


# ── Main entry point ────────────────────────────────────────────────────


def train_one_agent(
    *,
    agent_id: str,
    genes: CohortGenes,
    days_to_train: list[str],
    eval_day: str,
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
) -> AgentResult:
    """Train one agent through ``days_to_train`` and eval on ``eval_day``.

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

    Returns
    -------
    :class:`AgentResult` carrying the genes, train summary, and eval
    summary. The runner sorts agents by ``result.eval.total_reward``
    for breeding selection.
    """
    assert_in_range(genes)
    if not days_to_train:
        raise ValueError("days_to_train must contain at least one date.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    cfg = scalping_train_config()
    cfg["training"]["starting_budget"] = float(starting_budget)
    max_runners = int(cfg["training"]["max_runners"])

    # ── Build first-day env + shim to size the policy ────────────────
    first_day = days_to_train[0]
    logger.info(
        "Agent %s: loading first day %s from %s",
        agent_id, first_day, data_dir,
    )
    env, shim = _build_env_for_day(
        day_str=first_day, data_dir=data_dir, cfg=cfg, scorer_dir=scorer_dir,
    )

    policy = DiscreteLSTMPolicy(
        obs_dim=shim.obs_dim,
        action_space=shim.action_space,
        hidden_size=int(genes.hidden_size),
    )

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
    if model_store is not None:
        model_id = model_store.create_model(
            generation=int(generation),
            architecture_name=arch_name,
            architecture_description=_ARCH_DESCRIPTION,
            hyperparameters=genes.to_dict(),
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
        })
        logger.info(
            "Agent %s day %d/%d [%s] reward=%+.3f pnl=%+.2f "
            "value_loss=%.4f approx_kl=%.4f wall=%.1fs",
            agent_id, day_idx + 1, len(days_to_train), day_str,
            stats.total_reward, stats.day_pnl,
            stats.value_loss_mean, stats.approx_kl_mean,
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
    )

    # ── Eval rollout (no PPO update) ────────────────────────────────
    eval_t0 = time.perf_counter()
    logger.info("Agent %s: eval rollout on held-out day %s", agent_id, eval_day)
    _, eval_shim = _build_env_for_day(
        day_str=eval_day, data_dir=data_dir, cfg=cfg,
        scorer_dir=scorer_dir,
    )
    eval_collector = RolloutCollector(
        shim=eval_shim, policy=policy, device=device,
    )
    transitions = eval_collector.collect_episode()
    eval_summary_partial = _eval_rollout_stats(
        transitions=transitions,
        last_info=eval_collector.last_info,
        action_space=eval_shim.action_space,
    )
    eval_wall = time.perf_counter() - eval_t0
    eval_summary = EvalSummary(
        eval_day=eval_day,
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
        locked_pnl=eval_summary_partial.locked_pnl,
        naked_pnl=eval_summary_partial.naked_pnl,
        wall_time_sec=eval_wall,
    )
    logger.info(
        "Agent %s eval [%s] reward=%+.3f pnl=%+.2f bets=%d "
        "precision=%.3f arbs=%d/%d locked=%+.2f naked=%+.2f wall=%.1fs",
        agent_id, eval_day, eval_summary.total_reward,
        eval_summary.day_pnl, eval_summary.bet_count,
        eval_summary.bet_precision, eval_summary.arbs_completed,
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
            test_days=[eval_day],
        )
        model_store.record_evaluation_day(EvaluationDayRecord(
            run_id=run_id,
            date=eval_day,
            day_pnl=eval_summary.day_pnl,
            bet_count=eval_summary.bet_count,
            winning_bets=eval_summary.winning_bets,
            bet_precision=eval_summary.bet_precision,
            pnl_per_bet=eval_summary.pnl_per_bet,
            early_picks=eval_summary.early_picks,
            profitable=eval_summary.profitable,
            starting_budget=float(starting_budget),
            arbs_completed=eval_summary.arbs_completed,
            arbs_naked=eval_summary.arbs_naked,
            locked_pnl=eval_summary.locked_pnl,
            naked_pnl=eval_summary.naked_pnl,
        ))
        # Composite score = eval-day total_reward. Phase 3 keeps the
        # ranking simple; Session 04 may add a multi-component score.
        model_store.update_composite_score(
            model_id=model_id,
            score=float(eval_summary.total_reward),
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
