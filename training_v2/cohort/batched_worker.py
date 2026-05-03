"""Batched per-cluster training driver — throughput-fix Session 02.

Sibling of :mod:`training_v2.cohort.worker` that runs a CLUSTER of
``N`` architecturally-identical agents through their training days
in lock-step, sharing a :class:`BatchedRolloutCollector` per day.
The PPO update path stays per-agent (each agent owns its own
optimiser, its own trajectory, its own KL early-stop budget); only
rollout collection is batched.

Entry point: :func:`train_cluster_batched`. The cohort runner's
``train_cohort_batched`` function calls this once per architecture
cluster (sequential across clusters — see Session 02 prompt §2
"Cross-cluster scheduling").

Hard constraints (Session 02 prompt §"Hard constraints"):

* No env edits.
* No re-import of v1 trainer / policy / rollout / worker pool.
* Hidden-state contract is unchanged per agent.
* Per-agent self-parity is the load-bearing correctness guard.
* Per-agent RNG independence via ``BatchedRolloutCollector``'s
  save/restore mechanism (see ``batched_rollout.py``).
* ``Transition`` shape unchanged.

The ``--batched`` cohort runner flag defaults OFF; the sequential
:func:`train_one_agent` path stays the default until at least one
cohort run validates the batched path. Deleting the sequential path
is a Session 03 question.
"""

from __future__ import annotations

import gc
import logging
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

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
from training_v2.cohort.worker import (
    _ARCH_DESCRIPTION,
    AgentResult,
    EvalSummary,
    TrainSummary,
    _build_env_for_day,
    _eval_rollout_stats,
    arch_name_for_genes,
    scalping_train_config,
)
from training_v2.discrete_ppo.batched_rollout import BatchedRolloutCollector
from training_v2.discrete_ppo.rollout import RolloutCollector
from training_v2.discrete_ppo.trainer import DiscretePPOTrainer, EpisodeStats


logger = logging.getLogger(__name__)


__all__ = ["train_cluster_batched"]


# ── Public API ────────────────────────────────────────────────────────────


def train_cluster_batched(
    *,
    agent_ids: list[str],
    genes_list: list[CohortGenes],
    days_to_train: list[str],
    eval_day: str,
    data_dir: Path,
    device: str,
    seeds: list[int],
    model_store: ModelStore | None = None,
    scorer_dir: Path = DEFAULT_SCORER_DIR,
    generation: int = 0,
    parent_ids: list[tuple[str | None, str | None]] | None = None,
    starting_budget: float = 100.0,
    event_emitter: Callable[[dict], None] | None = None,
    agent_indices_in_cohort: list[int] | None = None,
    n_agents_in_cohort: int = 1,
    reward_overrides: dict | None = None,
) -> list[AgentResult]:
    """Train ``N`` arch-compatible agents in lock-step with a batched rollout.

    All agents in the cluster MUST share architecture (same
    ``hidden_size``, same ``policy_class``). The runner clusters
    agents via :func:`training_v2.discrete_ppo.batched_rollout.
    cluster_agents_by_arch` and calls this function once per cluster.

    Per training day, this function:

    1. Builds N envs + shims for the day.
    2. Rebinds each per-agent trainer to its own (env, shim).
    3. Builds a :class:`BatchedRolloutCollector` over all N
       (shim, policy) pairs.
    4. Runs one :meth:`collect_episode_batch` call → ``N`` per-agent
       transition lists.
    5. Calls :meth:`DiscretePPOTrainer.update_from_rollout` per
       agent — GAE + PPO update.

    Eval rollout (held-out day) is sequential per agent for now —
    eval is rollout-only (no PPO update), so the batched path's
    only saving is reduced kernel-launch latency. Future work may
    batch eval too; the speed bar in Session 02 is the training
    cohort wall.

    Returns one :class:`AgentResult` per agent in input order.
    """
    n = len(agent_ids)
    if n == 0:
        raise ValueError("train_cluster_batched: empty cluster")
    if len(genes_list) != n:
        raise ValueError(
            f"genes_list length {len(genes_list)} != agent_ids {n}",
        )
    if len(seeds) != n:
        raise ValueError(f"seeds length {len(seeds)} != agent_ids {n}")
    if not days_to_train:
        raise ValueError("days_to_train must contain at least one date.")
    for g in genes_list:
        assert_in_range(g)

    if parent_ids is None:
        parent_ids = [(None, None)] * n
    if agent_indices_in_cohort is None:
        agent_indices_in_cohort = list(range(n))

    # Validate cluster-key compatibility up-front (cheap; saves a
    # late failure deep in BatchedRolloutCollector).
    first_hs = int(genes_list[0].hidden_size)
    for g in genes_list[1:]:
        if int(g.hidden_size) != first_hs:
            raise ValueError(
                "train_cluster_batched: all agents must share hidden_size; "
                f"got {sorted({int(g.hidden_size) for g in genes_list})!r}",
            )

    cfg = scalping_train_config()
    cfg["training"]["starting_budget"] = float(starting_budget)
    max_runners = int(cfg["training"]["max_runners"])

    # ── Per-agent RNG seeding (top-level torch.manual_seed) ──────────
    # Each agent's :class:`DiscreteLSTMPolicy` is constructed in turn
    # under its own ``torch.manual_seed(seed)`` so initial weights are
    # deterministic per-seed. The rollout-time RNG is then driven by
    # the per-agent state in :class:`BatchedRolloutCollector`.
    policies: list[DiscreteLSTMPolicy] = []
    trainers: list[DiscretePPOTrainer] = []
    arch_names: list[str] = []
    model_ids: list[str] = []

    # First-day env+shim is needed to size each policy. We build N
    # envs+shims here so each agent's policy is seeded under its own
    # generator. The collector below replaces these with fresh per-day
    # envs.
    first_day = days_to_train[0]
    envs = []
    shims = []
    for i in range(n):
        torch.manual_seed(int(seeds[i]) & 0x7FFFFFFF)
        np.random.seed(int(seeds[i]) & 0x7FFFFFFF)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(int(seeds[i]) & 0x7FFFFFFF)

        env_i, shim_i = _build_env_for_day(
            day_str=first_day, data_dir=data_dir, cfg=cfg,
            scorer_dir=scorer_dir,
            reward_overrides=reward_overrides,
        )
        envs.append(env_i)
        shims.append(shim_i)

        policy = DiscreteLSTMPolicy(
            obs_dim=shim_i.obs_dim,
            action_space=shim_i.action_space,
            hidden_size=int(genes_list[i].hidden_size),
        )
        policies.append(policy)

        trainer = DiscretePPOTrainer(
            policy=policy,
            shim=shim_i,
            learning_rate=float(genes_list[i].learning_rate),
            gamma=0.99,
            gae_lambda=float(genes_list[i].gae_lambda),
            clip_range=float(genes_list[i].clip_range),
            entropy_coeff=float(genes_list[i].entropy_coeff),
            value_coeff=float(genes_list[i].value_coeff),
            ppo_epochs=4,
            mini_batch_size=int(genes_list[i].mini_batch_size),
            max_grad_norm=0.5,
            device=device,
        )
        trainers.append(trainer)

        arch = arch_name_for_genes(genes_list[i])
        arch_names.append(arch)

        if model_store is not None:
            mid = model_store.create_model(
                generation=int(generation),
                architecture_name=arch,
                architecture_description=_ARCH_DESCRIPTION,
                hyperparameters=genes_list[i].to_dict(),
                parent_a_id=parent_ids[i][0],
                parent_b_id=parent_ids[i][1],
                model_id=str(agent_ids[i]),
            )
        else:
            mid = str(agent_ids[i])
        model_ids.append(mid)

        if event_emitter is not None:
            try:
                event_emitter(agent_training_started_event(
                    agent_id=str(agent_ids[i]),
                    architecture_name=arch,
                    generation=int(generation),
                    agent_idx=int(agent_indices_in_cohort[i]),
                    n_agents=int(n_agents_in_cohort),
                    genes=genes_list[i].to_dict(),
                ))
            except Exception:
                logger.exception("event_emitter raised on agent_training_started; continuing")

    # ── Multi-day training loop (batched) ────────────────────────────
    train_t0 = time.perf_counter()
    per_day_rows_per_agent: list[list[dict]] = [[] for _ in range(n)]
    aggregate_hist_per_agent: list[Counter[str]] = [Counter() for _ in range(n)]
    total_steps_per_agent = [0] * n
    total_reward_per_agent = [0.0] * n

    for day_idx, day_str in enumerate(days_to_train):
        if day_idx > 0:
            # Build fresh envs+shims for this day.
            envs = []
            shims = []
            for i in range(n):
                _, new_shim = _build_env_for_day(
                    day_str=day_str, data_dir=data_dir, cfg=cfg,
                    scorer_dir=scorer_dir,
                    reward_overrides=reward_overrides,
                )
                shims.append(new_shim)
                envs.append(new_shim.env)
                # Rebind trainer to the new shim (mirror of
                # ``worker._rebind_trainer`` — the trainer's collector
                # isn't used in the batched path but we keep the
                # invariants tidy in case the trainer is reused for
                # the eval rollout in solo mode below).
                trainers[i].shim = new_shim
                trainers[i].action_space = new_shim.action_space
                trainers[i].max_runners = new_shim.max_runners
                trainers[i]._collector = RolloutCollector(
                    shim=new_shim, policy=policies[i], device=device,
                )

        # ── Build batched collector and collect ────────────────────
        # Per-day per-agent seeds offset the agent's base seed by the
        # day index — same shape as solo mode where each day's
        # ``train_episode`` consumes RNG starting from whatever state
        # the previous day left global. Here we seed the per-agent
        # private RNG state from a stable derived seed so cross-run
        # reproducibility holds.
        day_seeds = [
            (int(seeds[i]) ^ (day_idx + 1) * 0x9E3779B9) & 0x7FFFFFFF
            for i in range(n)
        ]
        collector = BatchedRolloutCollector(
            shims=shims,
            policies=policies,
            device=device,
            seeds=day_seeds,
        )
        transitions_per_agent = collector.collect_episode_batch()

        # ── Per-agent PPO update + bookkeeping ──────────────────────
        for i in range(n):
            transitions = transitions_per_agent[i]
            last_info = collector.last_infos[i]
            stats: EpisodeStats = trainers[i].update_from_rollout(
                transitions=transitions,
                last_info=last_info,
            )
            total_steps_per_agent[i] += int(stats.n_steps)
            total_reward_per_agent[i] += float(stats.total_reward)
            for k, v in (stats.action_histogram or {}).items():
                aggregate_hist_per_agent[i][k] += int(v)
            per_day_rows_per_agent[i].append({
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
                "value_loss=%.4f approx_kl=%.4f wall=%.1fs (batched)",
                agent_ids[i], day_idx + 1, len(days_to_train), day_str,
                stats.total_reward, stats.day_pnl,
                stats.value_loss_mean, stats.approx_kl_mean,
                stats.wall_time_sec,
            )
            if event_emitter is not None:
                try:
                    event_emitter(episode_complete_event(
                        agent_id=str(agent_ids[i]),
                        architecture_name=arch_names[i],
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
    train_summaries: list[TrainSummary] = []
    for i in range(n):
        n_d = max(len(days_to_train), 1)
        train_summaries.append(TrainSummary(
            n_days=len(days_to_train),
            total_steps=total_steps_per_agent[i],
            total_reward=total_reward_per_agent[i],
            mean_reward=total_reward_per_agent[i] / n_d,
            mean_pnl=float(np.mean(
                [r["day_pnl"] for r in per_day_rows_per_agent[i]],
            )),
            mean_value_loss=float(np.mean(
                [r["value_loss_mean"] for r in per_day_rows_per_agent[i]],
            )),
            mean_policy_loss=float(np.mean(
                [r["policy_loss_mean"] for r in per_day_rows_per_agent[i]],
            )),
            mean_approx_kl=float(np.mean(
                [r["approx_kl_mean"] for r in per_day_rows_per_agent[i]],
            )),
            wall_time_sec=train_wall,  # cluster-wide wall; per-agent
                                       # share would need finer profiling
            per_day_rows=per_day_rows_per_agent[i],
        ))

    # ── Eval rollout (per-agent, sequential) ────────────────────────
    # Sequential because eval is rollout-only — the gain from
    # batching it is bounded by the active-set shrinkage benefit on
    # one episode. The training loop is the dominant cost in cohort
    # wall, not the eval pass.
    eval_summaries: list[EvalSummary] = []
    for i in range(n):
        eval_t0 = time.perf_counter()
        logger.info(
            "Agent %s: eval rollout on held-out day %s",
            agent_ids[i], eval_day,
        )
        _, eval_shim = _build_env_for_day(
            day_str=eval_day, data_dir=data_dir, cfg=cfg,
            scorer_dir=scorer_dir,
            reward_overrides=reward_overrides,
        )
        eval_collector = RolloutCollector(
            shim=eval_shim, policy=policies[i], device=device,
        )
        eval_batch = eval_collector.collect_episode()
        partial = _eval_rollout_stats(
            batch=eval_batch,
            last_info=eval_collector.last_info,
            action_space=eval_shim.action_space,
        )
        eval_wall = time.perf_counter() - eval_t0
        eval_summaries.append(EvalSummary(
            eval_day=eval_day,
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
            wall_time_sec=eval_wall,
        ))

    # ── Registry writes (per agent) ──────────────────────────────────
    weights_paths: list[str] = [""] * n
    run_ids: list[str] = [""] * n
    if model_store is not None:
        for i in range(n):
            wp = model_store.save_weights(
                model_id=model_ids[i],
                state_dict=policies[i].state_dict(),
                obs_schema_version=OBS_SCHEMA_VERSION,
                action_schema_version=ACTION_SCHEMA_VERSION,
            )
            weights_paths[i] = wp
            rid = model_store.create_evaluation_run(
                model_id=model_ids[i],
                train_cutoff_date=days_to_train[-1],
                test_days=[eval_day],
            )
            run_ids[i] = rid
            es = eval_summaries[i]
            model_store.record_evaluation_day(EvaluationDayRecord(
                run_id=rid,
                date=eval_day,
                day_pnl=es.day_pnl,
                bet_count=es.bet_count,
                winning_bets=es.winning_bets,
                bet_precision=es.bet_precision,
                pnl_per_bet=es.pnl_per_bet,
                early_picks=es.early_picks,
                profitable=es.profitable,
                starting_budget=float(starting_budget),
                arbs_completed=es.arbs_completed,
                arbs_naked=es.arbs_naked,
                locked_pnl=es.locked_pnl,
                naked_pnl=es.naked_pnl,
            ))
            model_store.update_composite_score(
                model_id=model_ids[i],
                score=float(es.total_reward),
            )

    # ── Cleanup ─────────────────────────────────────────────────────
    del trainers
    if device.startswith("cuda"):
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ── Emit per-agent training-complete events ─────────────────────
    if event_emitter is not None:
        for i in range(n):
            try:
                event_emitter(agent_training_complete_event(
                    agent_id=str(agent_ids[i]),
                    architecture_name=arch_names[i],
                    generation=int(generation),
                    agent_idx=int(agent_indices_in_cohort[i]),
                    n_agents=int(n_agents_in_cohort),
                    eval_total_reward=float(eval_summaries[i].total_reward),
                    eval_day_pnl=float(eval_summaries[i].day_pnl),
                    eval_bet_count=int(eval_summaries[i].bet_count),
                    eval_bet_precision=float(eval_summaries[i].bet_precision),
                ))
            except Exception:
                logger.exception("event_emitter raised on agent_training_complete; continuing")

    return [
        AgentResult(
            agent_id=str(agent_ids[i]),
            model_id=str(model_ids[i]),
            architecture_name=arch_names[i],
            genes=genes_list[i],
            train=train_summaries[i],
            eval=eval_summaries[i],
            weights_path=str(weights_paths[i]),
            run_id=str(run_ids[i]),
        )
        for i in range(n)
    ]
