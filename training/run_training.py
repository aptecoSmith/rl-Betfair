"""
training/run_training.py -- Full generational training orchestrator.

Runs the complete training pipeline:

1. Initialise population (generation 0) or load survivors from previous gen
2. For each generation:
   a. Train all agents on training days
   b. Evaluate all agents on test days
   c. Score and rank via scoreboard
   d. Apply discard policy
   e. Select survivors (tournament + elitism)
   f. Breed next generation (crossover + mutation)
   g. Log genetic events
3. Repeat for N generations

Two-level ProgressTracker at every stage:
- Outer tracker: agents/generation
- Inner tracker: episodes (training) or test days (evaluation)

All progress events flow to a shared asyncio.Queue for WebSocket.

Usage::

    orchestrator = TrainingOrchestrator(config, model_store)
    result = orchestrator.run(
        train_days=train_days,
        test_days=test_days,
        n_generations=5,
        n_epochs=3,
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import threading
import time
import traceback
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

import torch

from typing import TYPE_CHECKING

from agents.population_manager import (
    AgentRecord,
    BreedingRecord,
    PopulationManager,
    SelectionResult,
)
from agents.ppo_trainer import PPOTrainer, TrainingStats
from data.episode_builder import Day
from env.betfair_env import ACTION_SCHEMA_VERSION, OBS_SCHEMA_VERSION
from registry.model_store import ModelStore
from registry.scoreboard import ModelScore, Scoreboard
from agents.bc_pretrainer import BCPretrainer, measure_entropy
from training.arb_annealing import effective_naked_loss_scale
from training.arb_oracle import load_samples, order_days_by_density
from training.evaluator import Evaluator
from training.perf_log import gpu_memory_summary, perf_log
from training.progress_tracker import ProgressTracker, RunProgressTracker

# Historical timing file — read on wizard load, written at end of each run.
# Best-effort only: missing/corrupt → fall back to defaults.
HISTORICAL_TIMING_PATH = Path("logs/training/last_run_timing.json")
# Legacy default from Session 4.6 (Jul 2024). Used when no historical
# timing file exists yet.
DEFAULT_SECONDS_PER_AGENT_PER_DAY = 12.0

if TYPE_CHECKING:
    from training.training_plan import PlanRegistry, TrainingPlan

logger = logging.getLogger(__name__)


# -- Result dataclass ----------------------------------------------------------


@dataclass
class GenerationResult:
    """Results from one generation."""

    generation: int
    training_stats: dict[str, TrainingStats]  # model_id → stats
    scores: list[ModelScore]
    selection: SelectionResult | None
    discarded: list[str]
    children: list[AgentRecord]
    breeding_records: list[BreedingRecord]


@dataclass
class TrainingRunResult:
    """Results from a full multi-generation run."""

    run_id: str
    generations: list[GenerationResult] = field(default_factory=list)
    final_rankings: list[ModelScore] = field(default_factory=list)


# -- Orchestrator --------------------------------------------------------------


class TrainingOrchestrator:
    """Full generational training loop.

    Parameters
    ----------
    config : dict
        Project config (from config.yaml).
    model_store : ModelStore | None
        Registry for persistence.  Pass None for test-only runs.
    progress_queue : asyncio.Queue | None
        Shared queue for all progress/phase events (WebSocket consumption).
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        config: dict,
        model_store: ModelStore | None = None,
        progress_queue: asyncio.Queue | None = None,
        device: str | None = None,
        stop_event: threading.Event | None = None,
        finish_event: threading.Event | None = None,
        skip_training_event: threading.Event | None = None,
        stop_after_current_eval_event: threading.Event | None = None,
        training_plan: "TrainingPlan | None" = None,
        plan_registry: "PlanRegistry | None" = None,
        stud_model_ids: list[str] | None = None,
    ) -> None:
        self.config = config
        self.model_store = model_store
        self.progress_queue = progress_queue
        self.stop_event = stop_event
        self.finish_event = finish_event
        self.skip_training_event = skip_training_event
        self.stop_after_current_eval_event = stop_after_current_eval_event
        # Optional Session-4 planner integration.  Both default to None
        # so config.yaml-based launches keep working unchanged.
        self.training_plan = training_plan
        self.plan_registry = plan_registry

        # Per-plan starting_budget override: patch config so all
        # downstream consumers (env, evaluator, scoreboard) pick it up.
        if training_plan is not None and training_plan.starting_budget is not None:
            config.setdefault("training", {})["starting_budget"] = training_plan.starting_budget

        # Per-plan reward_overrides: merge into config["reward"] so every
        # env instance the orchestrator constructs picks them up. Unknown
        # keys are rejected at create time (API layer) rather than silently
        # swallowed here, so anything that reaches this point is safe to
        # apply directly.
        if training_plan is not None and training_plan.reward_overrides:
            reward_cfg = config.setdefault("reward", {})
            for key, value in training_plan.reward_overrides.items():
                reward_cfg[key] = value

        # Arb-signal-cleanup Session 03 (2026-04-21). Per-plan cohort label
        # recorded on every episodes.jsonl row so the validator can
        # attribute per-criterion pass/fail to the cohort that produced
        # the agents. "" / None → "ungrouped" at log time.
        if training_plan is not None:
            cohort = training_plan.plan_cohort or ""
            config.setdefault("training", {})["plan_cohort"] = cohort

        # GPU auto-detection (config.training.device overrides auto-detect)
        config_device = config.get("training", {}).get("device")
        if device is not None:
            self.device = device
        elif config_device is not None:
            self.device = config_device
        elif torch.cuda.is_available():
            self.device = "cuda"
            logger.info(
                "GPU detected: %s (VRAM: %.1f GB). Using CUDA.",
                torch.cuda.get_device_name(0),
                torch.cuda.get_device_properties(0).total_memory / 1e9,
            )
        else:
            self.device = "cpu"
            logger.warning(
                "WARNING: No CUDA GPU detected — falling back to CPU. "
                "Training will be significantly slower. "
                "Ensure CUDA-enabled PyTorch is installed: "
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126"
            )

        # Config-level GPU requirement check
        require_gpu = config.get("training", {}).get("require_gpu", False)
        if require_gpu and self.device == "cpu":
            raise RuntimeError(
                "config.yaml has training.require_gpu=true but no CUDA GPU "
                "was detected. Install CUDA-enabled PyTorch or set "
                "require_gpu=false."
            )

        # Shared feature cache: engineer_day() is expensive and deterministic
        # for a given Day — cache results so they're computed once per day,
        # not once per agent per day per phase.
        self.feature_cache: dict[str, list] = {}

        self._stopped = False
        # Eval timing: track how long each agent eval takes (for time estimates)
        self._eval_times: list[float] = []
        self._eval_total_agents: int = 0
        self._eval_completed_agents: int = 0
        # Overall-run tracker: carries ETA across phase boundaries.
        self._run_tracker: RunProgressTracker | None = None
        self._run_phase_label: str = ""
        # Historical-timing accumulators (written to last_run_timing.json).
        self._train_wall_s: float = 0.0
        self._train_agent_days: int = 0
        self._eval_wall_s: float = 0.0
        self._eval_agent_days: int = 0
        # Run-level counters for the end-of-run summary.
        self._run_start_time: float = 0.0
        self._total_agents_trained: int = 0
        self._total_agents_evaluated: int = 0
        # Stud models (Issue 13): hand-picked parents forced into every
        # generation's breeding regardless of selection.
        self.stud_model_ids: list[str] = list(stud_model_ids or [])
        # Adaptive breeding (Issue 09) — per-run state.
        self._consecutive_bad_gens: int = 0
        self._effective_mutation_rate: float | None = None

        self.pop_manager = PopulationManager(config, model_store)
        self.evaluator = Evaluator(
            config, model_store, progress_queue=progress_queue, device=self.device,
            feature_cache=self.feature_cache,
        )
        self.scoreboard = Scoreboard(model_store, config) if model_store else None

    def _check_stop(self) -> bool:
        """Return True if a stop has been requested."""
        if self.stop_event is not None and self.stop_event.is_set():
            if not self._stopped:
                self._stopped = True
                logger.info("Stop requested — halting training")
                self._emit_phase_complete("run_stopped", {})
            return True
        return False

    def _check_finish(self) -> bool:
        """Return True if an early finish has been requested."""
        return self.finish_event is not None and self.finish_event.is_set()

    def _check_skip_training(self) -> bool:
        """Return True if training should be skipped (eval_all granularity)."""
        return self.skip_training_event is not None and self.skip_training_event.is_set()

    def _check_stop_after_current_eval(self) -> bool:
        """Return True if we should stop after the current eval completes."""
        return self.stop_after_current_eval_event is not None and self.stop_after_current_eval_event.is_set()

    def run(
        self,
        train_days: list[Day],
        test_days: list[Day],
        n_generations: int = 1,
        n_epochs: int = 1,
        seed: int | None = None,
        reevaluate_garaged: bool = False,
        reevaluate_min_score: float | None = None,
        start_generation: int = 0,
    ) -> TrainingRunResult:
        """Execute the full generational training loop.

        Parameters
        ----------
        train_days :
            Days used for training (earliest chronologically).
        test_days :
            Days used for evaluation (latest chronologically).
            If empty, evaluation is skipped with a warning.
        n_generations :
            Number of generations to run (within this session).
        n_epochs :
            Number of training passes over the training days per agent.
        seed :
            Optional RNG seed for population initialisation.
        start_generation :
            Generation offset for session resumption.  When > 0 the
            orchestrator loads survivors from the model store instead
            of creating a fresh random population.

        Returns
        -------
        TrainingRunResult with full results from all generations.
        """
        run_id = str(uuid.uuid4())
        result = TrainingRunResult(run_id=run_id)
        self._run_start_time = time.time()

        if not train_days:
            logger.warning("No training days provided — aborting run.")
            return result

        if not test_days:
            logger.warning(
                "No test days available. Using training days for evaluation "
                "(results will be optimistic — do not trust for ranking)."
            )
            test_days = train_days

        train_cutoff = train_days[-1].date

        # Emit informational events so the UI activity log shows the
        # effective configuration for this run.
        self._emit_info(f"Run {run_id[:8]} starting")
        self._emit_info(
            f"Data: {len(train_days)} train days, {len(test_days)} test days"
        )
        pop_cfg = self.config.get("population", {})
        cap = pop_cfg.get("max_mutations_per_child")
        cap_str = f", cap {cap}/child" if cap is not None else ""
        pool_str = pop_cfg.get("breeding_pool", "run_only")
        self._emit_info(
            f"Population: {pop_cfg.get('size', '?')} agents, "
            f"{pop_cfg.get('n_elite', '?')} elite, "
            f"{int(pop_cfg.get('selection_top_pct', 0) * 100)}% survival, "
            f"{int(pop_cfg.get('mutation_rate', 0) * 100)}% mutation{cap_str} "
            f"(breeding_pool: {pool_str})"
        )
        self._emit_info(
            f"Training: {n_generations} generation(s), {n_epochs} epoch(s) per day"
        )
        arch_choices = (
            self.config.get("hyperparameters", {})
            .get("search_ranges", {})
            .get("architecture_name", {})
            .get("choices", [])
        )
        if arch_choices:
            self._emit_info(f"Architectures: {', '.join(arch_choices)}")
        # Betting constraints — the whole point of surfacing these
        constraints = self.config.get("training", {}).get("betting_constraints", {})
        max_back = constraints.get("max_back_price")
        max_lay = constraints.get("max_lay_price")
        min_secs = constraints.get("min_seconds_before_off", 0)
        self._emit_info(
            f"Constraints: max_back={max_back if max_back is not None else 'unlimited'}, "
            f"max_lay={max_lay if max_lay is not None else 'unlimited'}, "
            f"min_seconds_before_off={min_secs}"
        )
        self._emit_info(
            f"Budget: \u00a3{self.config.get('training', {}).get('starting_budget', 100.0)} per race, "
            f"max {self.config.get('training', {}).get('max_bets_per_race', 20)} bets/race"
        )

        # -- Initialise or resume population --
        end_generation = start_generation + n_generations  # exclusive upper bound
        # Overall run tracker: one tick per agent-phase (train OR eval).
        pop_size = int(self.config.get("population", {}).get("size", 0)) or 0
        run_total = max(1, n_generations * pop_size * 2)
        self._run_tracker = RunProgressTracker(
            total=run_total,
            label=f"Run {run_id[:8]} — starting",
        )
        self._run_tracker.reset_timer()
        self._emit_phase_start("training", {
            "run_id": run_id,
            "n_generations": n_generations,
            "start_generation": start_generation,
            "train_days": len(train_days),
            "test_days": len(test_days),
        })

        # Backfill any models missing newly-added HP keys so crossover
        # and mutation don't KeyError on stale records.
        patched = self.pop_manager.backfill_hyperparameters()
        if patched:
            logger.info("Backfilled hyperparameters on %d existing model(s)", patched)

        if start_generation > 0 and self.model_store is not None:
            # Resuming from a previous session — load survivors from the
            # model store rather than creating a fresh random population.
            logger.info("Resuming from generation %d — loading survivors from model store", start_generation)
            self._emit_info(f"Session resume: loading survivors from generation {start_generation - 1}")
            agents = self.pop_manager.load_active_agents()
            if not agents:
                logger.warning(
                    "No active agents found in model store for resume — "
                    "falling back to fresh population"
                )
                seed_point = self._resolve_seed_point(run_id)
                agents = self.pop_manager.initialise_population(
                    generation=start_generation, seed=seed, plan=self.training_plan,
                    seed_point=seed_point,
                )
        else:
            # Fresh start — initialise population from scratch.
            seed_point = self._resolve_seed_point(run_id)
            agents = self.pop_manager.initialise_population(
                generation=0, seed=seed, plan=self.training_plan,
                seed_point=seed_point,
            )

        # Track which architectures the plan ever saw, so we can flag
        # any that die out across generations in the outcome record.
        self._planner_known_architectures: set[str] = (
            {a.architecture_name for a in agents}
            if self.training_plan is not None
            else set()
        )

        for gen in range(start_generation, end_generation):
            if self._check_stop():
                break

            # Finish requested between generations: skip remaining, evaluate, exit
            finishing_between_gens = self._check_finish() and gen > start_generation
            if finishing_between_gens:
                logger.info("Finish requested — skipping to final evaluation (gen %d)", gen)
                self._emit_phase_start("finishing_early", {
                    "generation": gen,
                    "skipped_generations": end_generation - gen,
                })

            logger.info("=== Generation %d ===", gen)
            try:
                gen_result = self._run_generation(
                    generation=gen,
                    agents=agents,
                    train_days=train_days,
                    test_days=test_days,
                    train_cutoff=train_cutoff,
                    n_epochs=n_epochs,
                    is_last=(gen == end_generation - 1) or finishing_between_gens,
                    skip_training=finishing_between_gens,
                )
            except Exception:
                logger.exception("Generation %d failed", gen)
                self._emit_phase_complete("generation_error", {
                    "generation": gen,
                    "error": traceback.format_exc(),
                })
                break
            result.generations.append(gen_result)

            # If finish was requested (either between gens or mid-training),
            # evaluate what we have and exit
            if finishing_between_gens or self._check_finish():
                if self.scoreboard is not None:
                    result.final_rankings = self.scoreboard.update_scores()
                else:
                    result.final_rankings = gen_result.scores
                break

            # Prepare next generation's agents
            if gen < end_generation - 1:
                # Survivors carry over, children are new
                survivor_agents = []
                if gen_result.selection is not None:
                    for mid in gen_result.selection.survivors:
                        try:
                            agent = self.pop_manager.load_agent(mid)
                            survivor_agents.append(agent)
                        except Exception:
                            logger.warning(
                                "Could not load survivor %s, skipping",
                                mid,
                                exc_info=True,
                            )
                agents = survivor_agents + gen_result.children
            else:
                # Final generation — compute final rankings
                if self.scoreboard is not None:
                    result.final_rankings = self.scoreboard.update_scores()
                else:
                    result.final_rankings = gen_result.scores

        # Re-evaluate garaged models on the current test data
        if reevaluate_garaged and self.model_store is not None and not self._check_stop():
            self._reevaluate_garaged(test_days, train_cutoff, reevaluate_min_score)
            # Recompute rankings to include re-evaluated garaged models
            if self.scoreboard is not None:
                result.final_rankings = self.scoreboard.update_scores()

        # Persist historical timing (best-effort) so the next wizard run
        # has realistic per-agent-per-day rates instead of the 12s default.
        self._persist_historical_timing(run_id)

        # Status: stopped > completed. run_error is emitted separately from
        # the worker's except block; this event always fires (even on stop).
        if self._stopped:
            status = "stopped"
        else:
            status = "completed"

        summary = self._build_run_summary(
            run_id=run_id,
            result=result,
            status=status,
            n_generations_requested=n_generations,
        )
        self._emit_phase_complete("run_complete", summary)

        return result

    def _reevaluate_garaged(
        self,
        test_days: list[Day],
        train_cutoff: str,
        min_score: float | None = None,
    ) -> None:
        """Re-evaluate garaged models on the current test data."""
        garaged = self.model_store.list_garaged_models()
        if min_score is not None:
            garaged = [
                m for m in garaged
                if m.composite_score is not None and m.composite_score >= min_score
            ]
        if not garaged:
            logger.info("No garaged models to re-evaluate")
            return

        self._emit_phase_start("reevaluating_garaged", {
            "model_count": len(garaged),
            "test_days": len(test_days),
        })

        tracker = ProgressTracker(
            total=len(garaged),
            label=f"Re-evaluating {len(garaged)} garaged model(s)",
        )
        tracker.reset_timer()

        for model_rec in garaged:
            if self._check_stop():
                break
            self._publish_progress(
                "reevaluating_garaged", tracker,
                detail=f"Re-evaluating {model_rec.model_id[:12]}",
            )
            try:
                agent = self.pop_manager.load_agent(model_rec.model_id)
                mtf = (agent.hyperparameters or {}).get("market_type_filter", "BOTH")
                self.evaluator.evaluate(
                    model_id=model_rec.model_id,
                    policy=agent.policy,
                    test_days=test_days,
                    train_cutoff_date=train_cutoff,
                    market_type_filter=mtf,
                    hyperparameters=agent.hyperparameters or None,
                )
            except Exception:
                logger.exception(
                    "Failed to re-evaluate garaged model %s",
                    model_rec.model_id[:12],
                )
            tracker.tick()
            self._publish_progress("reevaluating_garaged", tracker)

        self._emit_phase_complete("reevaluating_garaged", {
            "models_reevaluated": tracker.completed,
        })

    def _run_generation(
        self,
        generation: int,
        agents: list[AgentRecord],
        train_days: list[Day],
        test_days: list[Day],
        train_cutoff: str,
        n_epochs: int,
        is_last: bool,
        skip_training: bool = False,
    ) -> GenerationResult:
        """Run one full generation: train → evaluate → score → select → breed.

        If *skip_training* is True the training phase is skipped entirely
        and only evaluation + scoring runs.  Used by the "finish up" flow.
        """

        training_stats: dict[str, TrainingStats] = {}

        if skip_training:
            # Jump straight to evaluation
            self._emit_phase_complete("training", {
                "generation": generation,
                "agents_trained": 0,
                "skipped": True,
            })
        else:
            # ── Phase: Training ──────────────────────────────────────────────
            self._emit_phase_start("training", {
                "generation": generation,
                "agent_count": len(agents),
            })

            outer_tracker = ProgressTracker(
                total=len(agents),
                label=f"Generation {generation} — training {len(agents)} agents",
            )
            outer_tracker.reset_timer()
            if self._run_tracker is not None:
                self._run_tracker.set_label(f"Generation {generation} — training")

            for agent in agents:
                if self._check_stop() or self._check_finish() or self._check_skip_training():
                    if self._check_skip_training() and not self._check_stop():
                        logger.info("Skip-training event set — jumping to evaluation")
                        self._emit_info("Skipping remaining training — evaluating existing models...")
                    break
                self._publish_progress(
                    "training", outer_tracker,
                    detail=f"Training agent {agent.model_id[:12]} ({agent.architecture_name})",
                )

                # Arb-curriculum Session 03: apply generation-level
                # naked_loss_scale annealing before the trainer sees the
                # HP dict so the env receives the interpolated effective
                # scale, not the raw gene value.
                hp = dict(agent.hyperparameters)
                anneal_schedule = (
                    self.training_plan.naked_loss_anneal
                    if self.training_plan is not None else None
                )
                if anneal_schedule is not None and "naked_loss_scale" in hp:
                    hp["naked_loss_scale"] = effective_naked_loss_scale(
                        float(hp["naked_loss_scale"]),
                        current_gen=generation,
                        schedule=anneal_schedule,
                    )

                trainer = PPOTrainer(
                    policy=agent.policy,
                    config=self.config,
                    hyperparams=hp,
                    progress_queue=self.progress_queue,
                    device=self.device,
                    feature_cache=self.feature_cache,
                    model_id=agent.model_id,
                    architecture_name=agent.architecture_name,
                )

                # Arb-curriculum Session 04: BC pretrain before first
                # PPO rollout. Only runs when scalping_mode is on and
                # bc_pretrain_steps > 0 in the agent's gene.
                _bc_steps = int(hp.get("bc_pretrain_steps", 0) or 0)
                _scalping_on = self.config.get(
                    "training", {}
                ).get("scalping_mode", False)
                if _scalping_on and _bc_steps > 0:
                    _all_samples: list = []
                    for _date in [d.date for d in train_days]:
                        try:
                            _all_samples.extend(
                                load_samples(
                                    _date,
                                    Path("data/processed"),
                                    strict=True,
                                )
                            )
                        except FileNotFoundError:
                            logger.warning(
                                "BC: oracle cache missing for %s; "
                                "skipping date",
                                _date,
                            )
                        except ValueError as _exc:
                            logger.warning(
                                "BC: oracle schema mismatch for %s "
                                "(%s); skipping date",
                                _date, _exc,
                            )
                    if not _all_samples:
                        logger.warning(
                            "BC requested (steps=%d) but no oracle "
                            "samples available; skipping BC for "
                            "agent %s",
                            _bc_steps, agent.model_id[:12],
                        )
                    else:
                        _bc_lr = float(
                            hp.get("bc_learning_rate", 3e-4) or 3e-4
                        )
                        _bc = BCPretrainer(lr=_bc_lr)
                        _bc_history = _bc.pretrain(
                            agent.policy, _all_samples,
                            n_steps=_bc_steps,
                        )
                        trainer._bc_loss_history = _bc_history
                        trainer._bc_pretrain_steps_done = len(
                            _bc_history.signal_losses
                        )
                        trainer._post_bc_entropy = measure_entropy(
                            agent.policy, _all_samples[:256],
                        )
                        trainer._bc_target_entropy_warmup_eps = int(
                            hp.get(
                                "bc_target_entropy_warmup_eps", 5
                            ) or 5
                        )
                        logger.info(
                            "BC pretrain: agent=%s steps=%d "
                            "signal_loss=%.4f arb_spread_loss=%.4f "
                            "post_bc_entropy=%.1f",
                            agent.model_id[:12], _bc_steps,
                            _bc_history.final_signal_loss,
                            _bc_history.final_arb_spread_loss,
                            trainer._post_bc_entropy,
                        )

                # Arb-curriculum Session 05: per-agent curriculum day ordering.
                # Reorder train_days per oracle density; membership preserved.
                _curriculum_mode = self.config.get("training", {}).get(
                    "curriculum_day_order", "random"
                )
                _rng = random.Random(hash(agent.model_id) & 0xFFFFFFFF)
                _date_to_day = {d.date: d for d in train_days}
                _dates_ordered = order_days_by_density(
                    [d.date for d in train_days],
                    _curriculum_mode,
                    Path("data/oracle_cache"),
                    _rng,
                )
                _ordered_train_days = [_date_to_day[dt] for dt in _dates_ordered]
                logger.info(
                    "Curriculum mode=%s agent=%s day order: %s",
                    _curriculum_mode, agent.model_id[:12],
                    [d[:10] for d in _dates_ordered[:5]],
                )

                train_start = time.time()
                with perf_log(
                    logger,
                    f"Train agent {agent.model_id[:12]}",
                    log_gpu=(self.device == "cuda"),
                ):
                    stats = trainer.train(_ordered_train_days, n_epochs=n_epochs)
                self._train_wall_s += time.time() - train_start
                self._train_agent_days += len(train_days) * n_epochs
                self._total_agents_trained += 1
                training_stats[agent.model_id] = stats

                # Save updated weights after training
                if self.model_store is not None:
                    self.model_store.save_weights(
                        agent.model_id, agent.policy.state_dict(),
                        obs_schema_version=OBS_SCHEMA_VERSION,
                        action_schema_version=ACTION_SCHEMA_VERSION,
                    )

                outer_tracker.tick()
                if self._run_tracker is not None:
                    self._run_tracker.tick()
                self._publish_progress(
                    "training", outer_tracker,
                    detail=(
                        f"Agent {agent.model_id[:12]} done | "
                        f"mean_reward={stats.mean_reward:+.3f} | "
                        f"mean_pnl={stats.mean_pnl:+.2f}"
                    ),
                )

                # Memory-leak fix (2026-04-24). Each trainer
                # allocates optimiser-state tensors (Adam moments,
                # 2× the policy weight count on GPU), rollout
                # tensors (obs / action / log_prob / hidden_state —
                # the last of which is ~80 MB per 5000-tick rollout
                # on ctx=256 transformer), and feature-encoder
                # intermediates. Dropping the local ``trainer``
                # reference lets Python release those blocks; the
                # ``empty_cache()`` then returns the reserved VRAM
                # to the PyTorch allocator pool so the NEXT agent's
                # trainer doesn't inherit the high-water mark. The
                # agent.policy itself stays alive via the ``agents``
                # list — that's intentional (evaluator needs it).
                del trainer
                if self.device == "cuda":
                    try:
                        import gc
                        import torch
                        gc.collect()
                        torch.cuda.empty_cache()
                    except Exception:
                        logger.warning(
                            "Between-agent GPU cleanup raised; "
                            "continuing", exc_info=True,
                        )

            self._emit_phase_complete("training", {
                "generation": generation,
                "agents_trained": len(agents),
            })

        if self._check_stop():
            return GenerationResult(
                generation=generation, training_stats=training_stats,
                scores=[], selection=None, discarded=[], children=[], breeding_records=[],
            )

        # ── Phase: Evaluation ────────────────────────────────────────────
        self._emit_phase_start("evaluating", {
            "generation": generation,
            "agent_count": len(agents),
            "test_days": len(test_days),
        })

        eval_tracker = ProgressTracker(
            total=len(agents),
            label=f"Generation {generation} — evaluating {len(agents)} agents",
        )
        eval_tracker.reset_timer()
        eval_phase_start = time.time()
        if self._run_tracker is not None:
            self._run_tracker.set_label(f"Generation {generation} — evaluating")

        # Reset eval timing for time estimates
        self._eval_total_agents = len(agents)
        self._eval_completed_agents = 0

        # Determine parallelism: CPU-bound rollout benefits from threads
        # (torch releases GIL during forward pass). Cap at available cores.
        max_eval_workers = min(
            len(agents),
            self.config.get("training", {}).get("eval_workers", 1),
            os.cpu_count() or 1,
        )

        def _eval_agent(agent: AgentRecord) -> None:
            """Evaluate a single agent (thread-safe: no shared mutable state)."""
            evaluator = Evaluator(
                config=self.config,
                model_store=self.model_store,
                progress_queue=self.progress_queue,
                device=self.device,
                feature_cache=self.feature_cache,
            )
            mtf = (agent.hyperparameters or {}).get("market_type_filter", "BOTH")
            evaluator.evaluate(
                model_id=agent.model_id,
                policy=agent.policy,
                test_days=test_days,
                train_cutoff_date=train_cutoff,
                market_type_filter=mtf,
                hyperparameters=agent.hyperparameters or None,
            )

        agents_evaluated = 0
        if max_eval_workers > 1:
            logger.info(
                "Parallel evaluation: %d agents across %d workers",
                len(agents), max_eval_workers,
            )
            with ThreadPoolExecutor(max_workers=max_eval_workers) as pool:
                futures = [pool.submit(_eval_agent, agent) for agent in agents]
                for future in futures:
                    future.result()  # raises if eval failed
                    eval_tracker.tick()
                    if self._run_tracker is not None:
                        self._run_tracker.tick()
                    self._total_agents_evaluated += 1
                    agents_evaluated += 1
        else:
            for agent in agents:
                # stop_event always overrides (escalation)
                if self._check_stop():
                    break
                self._publish_progress(
                    "evaluating", eval_tracker,
                    detail=f"Evaluating agent {agent.model_id[:12]}",
                )
                eval_start = time.time()
                with perf_log(
                    logger,
                    f"Eval agent {agent.model_id[:12]}",
                    log_gpu=(self.device == "cuda"),
                ):
                    _eval_agent(agent)
                eval_elapsed = time.time() - eval_start
                self._eval_times.append(eval_elapsed)
                self._eval_completed_agents += 1
                self._total_agents_evaluated += 1
                eval_tracker.tick()
                if self._run_tracker is not None:
                    self._run_tracker.tick()
                agents_evaluated += 1
                # After completing this agent's eval, check if we should stop
                if self._check_stop_after_current_eval():
                    logger.info("Stop-after-current-eval — stopping after agent %s", agent.model_id[:12])
                    self._emit_info("Stopping after current evaluation...")
                    break

        # Accumulate eval wall time for historical-timing persistence.
        self._eval_wall_s += time.time() - eval_phase_start
        self._eval_agent_days += agents_evaluated * len(test_days)

        self._emit_phase_complete("evaluating", {
            "generation": generation,
            "agents_evaluated": agents_evaluated,
        })

        # ── Phase: Scoring ───────────────────────────────────────────────
        self._emit_phase_start("scoring", {"generation": generation})

        scores: list[ModelScore] = []
        if self.scoreboard is not None:
            scores = self.scoreboard.update_scores()
        else:
            # Fallback for tests without a model store: build minimal scores
            for agent in agents:
                scores.append(ModelScore(
                    model_id=agent.model_id,
                    win_rate=0.0,
                    mean_daily_pnl=0.0,
                    sharpe=0.0,
                    bet_precision=0.0,
                    pnl_per_bet=0.0,
                    efficiency=0.0,
                    composite_score=0.0,
                    test_days=len(test_days),
                    profitable_days=0,
                ))

        self._emit_phase_complete("scoring", {
            "generation": generation,
            "models_scored": len(scores),
            "top_score": scores[0].composite_score if scores else 0.0,
        })

        # ── Phase: Selection & breeding ──────────────────────────────────
        selection: SelectionResult | None = None
        discarded: list[str] = []
        children: list[AgentRecord] = []
        breeding_records: list[BreedingRecord] = []

        if not is_last and scores:
            self._emit_phase_start("selecting", {"generation": generation})

            # Scope selection to this run's population — the scoreboard
            # persists scores for every active model, but only the agents
            # in *this* generation should participate in selection/breeding
            # (run_only, the default). When the operator opts into
            # include_garaged or full_registry the external models join
            # the *pool* but stay parent-only — they don't take survivor
            # slots in the next generation.
            run_ids = {a.model_id for a in agents}
            run_scores = [s for s in scores if s.model_id in run_ids]

            # Discard policy (scoped to this run's agents)
            discarded = self.pop_manager.apply_discard_policy(run_scores)

            # ── Adaptive breeding: detect bad generation ──
            pop_cfg_local = self.config.get("population", {})
            base_mutation_rate = pop_cfg_local.get("mutation_rate", 0.3)
            bad_gen_threshold = float(
                pop_cfg_local.get("bad_generation_threshold", 0.0) or 0.0
            )
            bad_gen_policy = pop_cfg_local.get("bad_generation_policy", "persist")
            adaptive = bool(pop_cfg_local.get("adaptive_mutation", False))
            increment = float(pop_cfg_local.get("adaptive_mutation_increment", 0.1))
            cap = float(pop_cfg_local.get("adaptive_mutation_cap", 0.8))

            best_score = (
                max((s.composite_score for s in run_scores), default=0.0)
                if run_scores else 0.0
            )
            is_bad = (
                bad_gen_threshold > 0.0 and best_score < bad_gen_threshold
            )
            policy_inject_top = False
            policy_boost = False

            if is_bad:
                self._consecutive_bad_gens += 1
                self._emit_info(
                    f"Generation {generation} underperformed "
                    f"(best={best_score:.4f}, threshold={bad_gen_threshold:.4f}) "
                    f"— policy={bad_gen_policy}"
                )
                if bad_gen_policy == "boost_mutation":
                    policy_boost = True
                elif bad_gen_policy == "inject_top":
                    policy_inject_top = True
            else:
                if self._consecutive_bad_gens > 0:
                    self._emit_info(
                        f"Generation {generation} recovered "
                        f"(best={best_score:.4f}) — resetting adaptive state"
                    )
                self._consecutive_bad_gens = 0

            # Effective mutation rate.
            if adaptive and self._consecutive_bad_gens > 0:
                effective_mutation = min(
                    cap,
                    base_mutation_rate + increment * self._consecutive_bad_gens,
                )
            elif policy_boost:
                effective_mutation = min(cap, base_mutation_rate + increment)
            else:
                effective_mutation = base_mutation_rate
            self._effective_mutation_rate = effective_mutation
            if effective_mutation != base_mutation_rate:
                self._emit_info(
                    f"Mutation rate: {base_mutation_rate:.2f} → "
                    f"{effective_mutation:.2f}"
                )

            # Build the breeding pool according to config.
            breeding_pool_mode = pop_cfg_local.get("breeding_pool", "run_only")
            external_ids: set[str] = set()
            if breeding_pool_mode == "include_garaged" and self.model_store is not None:
                try:
                    for g in self.model_store.list_garaged_models():
                        if g.model_id not in run_ids:
                            external_ids.add(g.model_id)
                except Exception:
                    logger.warning(
                        "Could not enumerate garaged models for breeding pool",
                        exc_info=True,
                    )
            elif breeding_pool_mode == "full_registry":
                # All scored models that aren't in this run already.
                for s in scores:
                    if s.model_id not in run_ids:
                        external_ids.add(s.model_id)
            elif breeding_pool_mode != "run_only":
                logger.warning(
                    "Unknown breeding_pool mode %r — falling back to run_only",
                    breeding_pool_mode,
                )


            # inject_top policy: pull top garaged models as parent-only.
            if policy_inject_top and self.model_store is not None:
                try:
                    garaged = self.model_store.list_garaged_models()
                    # Already sorted by composite_score desc.
                    top_n = 5
                    injected = 0
                    for g in garaged:
                        if injected >= top_n:
                            break
                        if g.model_id in run_ids or g.model_id in external_ids:
                            continue
                        external_ids.add(g.model_id)
                        injected += 1
                    if injected:
                        self._emit_info(
                            f"inject_top: added {injected} top garaged "
                            f"model(s) as parent-only"
                        )
                    else:
                        self._emit_info(
                            "inject_top: no eligible garaged models to inject"
                        )
                except Exception:
                    logger.warning(
                        "inject_top: could not enumerate garaged models",
                        exc_info=True,
                    )

            # Pool scores = run scores (post-discard) + external scores.
            active_scores = [s for s in run_scores if s.model_id not in discarded]
            external_scores = [s for s in scores if s.model_id in external_ids]
            pool_scores = active_scores + external_scores

            if external_ids:
                self._emit_info(
                    f"Breeding pool ({breeding_pool_mode}): "
                    f"{len(active_scores)} run + {len(external_scores)} external "
                    f"(parent-only)"
                )

            if pool_scores:
                pool_selection = self.pop_manager.select(pool_scores)

                # Split survivors into run-survivors (carry over) and
                # external (parent-only). External survivors do NOT take
                # slots in the next generation and are NOT re-trained;
                # they only contribute hyperparameters via crossover.
                run_survivor_ids = [
                    mid for mid in pool_selection.survivors
                    if mid not in external_ids
                ]
                external_parent_ids = [
                    mid for mid in pool_selection.survivors
                    if mid in external_ids
                ]
                selection = SelectionResult(
                    elites=[
                        mid for mid in pool_selection.elites
                        if mid not in external_ids
                    ],
                    survivors=run_survivor_ids,
                    eliminated=pool_selection.eliminated,
                    ranked_scores=pool_selection.ranked_scores,
                    external_parents=external_parent_ids,
                )

                self._emit_phase_complete("selecting", {
                    "generation": generation,
                    "survivors": len(selection.survivors),
                    "eliminated": len(selection.eliminated),
                    "discarded": len(discarded),
                    "external_parents": len(external_parent_ids),
                })

                # Breeding
                self._emit_phase_start("breeding", {
                    "generation": generation,
                    "survivors": len(selection.survivors),
                    "external_parents": len(external_parent_ids),
                })

                # Use adaptive/effective mutation rate when set, else the
                # base config value.
                mutation_rate = (
                    self._effective_mutation_rate
                    if self._effective_mutation_rate is not None
                    else self.config["population"].get("mutation_rate", 0.3)
                )
                max_mutations = self.config["population"].get(
                    "max_mutations_per_child"
                )
                # Resolve studs: only load HP for IDs present in registry
                # with both weights and hyperparameters. Validation already
                # happened at API start, but be defensive — log and skip
                # any stud that's been deleted or corrupted between launch
                # and this generation.
                stud_ids: list[str] = []
                if self.stud_model_ids and self.model_store is not None:
                    for sid in self.stud_model_ids:
                        rec = self.model_store.get_model(sid)
                        if rec is None:
                            logger.warning("Stud %s not found — skipping", sid[:12])
                            continue
                        if not rec.hyperparameters:
                            logger.warning("Stud %s has no HP — skipping", sid[:12])
                            continue
                        stud_ids.append(sid)
                    if stud_ids:
                        self._emit_info(
                            f"Studs: {len(stud_ids)} guaranteed parent(s) — "
                            + ", ".join(s[:12] for s in stud_ids)
                        )

                children, breeding_records = self.pop_manager.breed(
                    selection_result=selection,
                    generation=generation + 1,
                    mutation_rate=mutation_rate,
                    max_mutations=max_mutations,
                    external_parent_ids=external_parent_ids,
                    stud_parent_ids=stud_ids,
                )

                self._emit_phase_complete("breeding", {
                    "generation": generation,
                    "children_bred": len(children),
                })

                # Log genetic events
                self.pop_manager.log_generation(
                    generation=generation,
                    selection_result=selection,
                    breeding_records=breeding_records,
                    discarded=discarded,
                )

        # Purge discarded models (weights, eval data) to free space
        if discarded and self.model_store is not None:
            purged = self.model_store.purge_discarded()
            if purged:
                logger.info(
                    "Purged %d discarded model(s): %s",
                    len(purged),
                    ", ".join(mid[:12] for mid in purged),
                )

        # GPU memory summary at end of generation
        mem = gpu_memory_summary()
        if mem:
            logger.info("Generation %d complete | %s", generation, mem)
            torch.cuda.reset_peak_memory_stats()

        # Session-4 planner outcome callback.  Records best/mean fitness
        # and any architectures that have died out so the UI can show a
        # post-hoc summary alongside the original plan configuration.
        self._record_plan_outcome(generation, agents, scores)

        return GenerationResult(
            generation=generation,
            training_stats=training_stats,
            scores=scores,
            selection=selection,
            discarded=discarded,
            children=children,
            breeding_records=breeding_records,
        )

    # -- Session-4 planner outcome callback ------------------------------------

    def _record_plan_outcome(
        self,
        generation: int,
        agents: list[AgentRecord],
        scores: list[ModelScore],
    ) -> None:
        """Append a :class:`GenerationOutcome` to the plan, if one is set.

        Silent no-op when ``training_plan`` or ``plan_registry`` is None
        so the legacy ``config.yaml``-only launch path is unaffected.
        """
        if self.training_plan is None or self.plan_registry is None:
            return

        # Lazy import to avoid a hard dependency at module load time --
        # the planner module is optional from this orchestrator's POV.
        from training.training_plan import GenerationOutcome

        composite_scores = [
            s.composite_score for s in scores
            if s.composite_score is not None
        ]
        best = max(composite_scores) if composite_scores else 0.0
        mean = (
            sum(composite_scores) / len(composite_scores)
            if composite_scores else 0.0
        )

        alive = sorted({a.architecture_name for a in agents})
        if not self._planner_known_architectures:
            self._planner_known_architectures = set(alive)
        died = sorted(self._planner_known_architectures - set(alive))

        outcome = GenerationOutcome(
            generation=generation,
            recorded_at=datetime.now(timezone.utc).isoformat(),
            best_fitness=float(best),
            mean_fitness=float(mean),
            architectures_alive=alive,
            architectures_died=died,
            n_agents=len(agents),
        )
        try:
            self.plan_registry.record_outcome(self.training_plan.plan_id, outcome)
        except Exception:
            logger.exception(
                "Failed to record planner outcome for plan %s gen %d",
                self.training_plan.plan_id, generation,
            )

    @property
    def eval_rate_s(self) -> float | None:
        """Average seconds per model evaluation, or None if no data."""
        if not self._eval_times:
            return None
        return sum(self._eval_times) / len(self._eval_times)

    @property
    def unevaluated_count(self) -> int:
        """Number of models not yet evaluated in the current generation."""
        return max(0, self._eval_total_agents - self._eval_completed_agents)

    # -- Exploration strategy --------------------------------------------------

    def _resolve_seed_point(self, run_id: str) -> dict | None:
        """Resolve the plan's exploration strategy to a seed point (or None).

        Returns ``None`` for ``"random"`` (legacy behaviour).  For all
        other strategies, records the exploration run in the DB and returns
        the seed point dict to pass to ``initialise_population()``.
        """
        plan = self.training_plan
        if plan is None or plan.exploration_strategy == "random":
            return None

        from agents.population_manager import parse_search_ranges
        from training.training_plan import (
            generate_coverage_seed,
            generate_sobol_points,
            historical_agents_from_model_store,
        )

        hp_specs = parse_search_ranges(
            plan.hp_ranges or self.config["hyperparameters"]["search_ranges"]
        )

        strategy = plan.exploration_strategy
        seed_point: dict
        coverage_snapshot: dict | None = None

        if strategy == "sobol":
            skip = 0
            if self.model_store is not None:
                skip = self.model_store.get_exploration_run_count()
            points = generate_sobol_points(hp_specs, n_points=1, skip=skip)
            seed_point = points[0]
            self._emit_info(f"Sobol seed point #{skip + 1}: {seed_point}")

        elif strategy == "coverage":
            history = historical_agents_from_model_store(self.model_store)
            seed_point, report = generate_coverage_seed(hp_specs, history)
            coverage_snapshot = report.to_dict()
            gaps = len(report.poorly_covered_genes)
            self._emit_info(
                f"Coverage seed targeting {gaps} poorly-covered gene(s): {seed_point}"
            )

        elif strategy == "manual":
            if not plan.manual_seed_point:
                self._emit_info("Manual strategy with no seed point — falling back to random")
                return None
            seed_point = dict(plan.manual_seed_point)
            self._emit_info(f"Manual seed point: {seed_point}")

        else:
            logger.warning("Unknown exploration strategy %r — falling back to random", strategy)
            return None

        # Record in exploration log.
        if self.model_store is not None:
            self.model_store.record_exploration_run(
                run_id=run_id,
                seed_point=seed_point,
                strategy=strategy,
                coverage_before=coverage_snapshot,
            )

        return seed_point

    # -- Progress event helpers ------------------------------------------------

    def _emit_info(self, message: str) -> None:
        """Emit a lightweight progress event purely to log an info message
        in the frontend activity log and stdout. Used for run-start config
        summaries and other informational breadcrumbs."""
        event = {
            "event": "progress",
            "phase": "info",
            "detail": message,
            "timestamp": time.time(),
        }
        logger.info(message)
        self._put_event(event)

    def _emit_phase_start(self, phase: str, summary: dict) -> None:
        """Emit a phase_start event."""
        event = {
            "event": "phase_start",
            "phase": phase,
            "timestamp": time.time(),
            "summary": summary,
        }
        logger.info("Phase start: %s %s", phase, summary)
        self._put_event(event)

    def _emit_phase_complete(self, phase: str, summary: dict) -> None:
        """Emit a phase_complete event."""
        event = {
            "event": "phase_complete",
            "phase": phase,
            "timestamp": time.time(),
            "summary": summary,
        }
        logger.info("Phase complete: %s %s", phase, summary)
        self._put_event(event)

    def _publish_progress(
        self,
        phase: str,
        tracker: ProgressTracker,
        detail: str = "",
    ) -> None:
        """Publish a progress event with the outer tracker state."""
        event = {
            "event": "progress",
            "phase": phase,
            "process": tracker.to_dict(),
            "detail": detail,
            "timestamp": time.time(),
        }
        if self._run_tracker is not None:
            event["overall"] = self._run_tracker.to_dict()
        # Include eval stats so the frontend can compute time estimates
        if phase == "evaluating":
            event["unevaluated_count"] = self.unevaluated_count
            event["eval_rate_s"] = self.eval_rate_s
        self._put_event(event)

    def _persist_historical_timing(self, run_id: str) -> None:
        """Write per-agent-per-day timing to last_run_timing.json.

        Best-effort only — any IO error is logged and swallowed so timing
        persistence never breaks a run.
        """
        try:
            train_rate = None
            eval_rate = None
            # Train rate is seconds per (agent × day × epoch). Epochs are
            # already baked into self._train_agent_days.
            if self._train_agent_days > 0:
                train_rate = self._train_wall_s / self._train_agent_days
            if self._eval_agent_days > 0:
                eval_rate = self._eval_wall_s / self._eval_agent_days
            if train_rate is None and eval_rate is None:
                return  # nothing useful to save

            payload = {
                "run_id": run_id,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "train_seconds_per_agent_per_day": train_rate,
                "eval_seconds_per_agent_per_day": eval_rate,
                "total_agents_trained": self._total_agents_trained,
                "total_agents_evaluated": self._total_agents_evaluated,
            }
            HISTORICAL_TIMING_PATH.parent.mkdir(parents=True, exist_ok=True)
            HISTORICAL_TIMING_PATH.write_text(json.dumps(payload, indent=2))
            logger.info(
                "Saved historical timing: train=%.2fs/agent-day eval=%.2fs/agent-day",
                train_rate or 0.0, eval_rate or 0.0,
            )
        except Exception:
            logger.exception("Failed to persist historical timing (non-fatal)")

    def _build_run_summary(
        self,
        run_id: str,
        result: TrainingRunResult,
        status: str,
        n_generations_requested: int,
        error_message: str | None = None,
    ) -> dict:
        """Assemble the enriched run_complete summary dict.

        Used by the UI to render the end-of-run modal.
        """
        wall_time = max(0.0, time.time() - self._run_start_time)
        rankings = result.final_rankings or []

        best_model: dict | None = None
        top_5: list[dict] = []
        if rankings:
            # Guard against empty composite scores.
            sorted_rank = sorted(
                rankings,
                key=lambda s: (
                    s.composite_score if s.composite_score is not None else -1e9
                ),
                reverse=True,
            )
            top_n = sorted_rank[:5]
            for s in top_n:
                arch_name = "unknown"
                if self.model_store is not None:
                    rec = self.model_store.get_model(s.model_id)
                    if rec is not None:
                        arch_name = rec.architecture_name
                top_5.append({
                    "model_id": s.model_id,
                    "composite_score": s.composite_score,
                    "pnl": s.mean_daily_pnl,
                    "win_rate": s.win_rate,
                    "architecture": arch_name,
                })
            if top_5:
                best = top_5[0]
                best_model = {
                    "model_id": best["model_id"],
                    "composite_score": best["composite_score"],
                    "total_pnl": best["pnl"],
                    "win_rate": best["win_rate"],
                    "architecture": best["architecture"],
                }

        # Population summary from model_store (active / discarded / garaged).
        survived = 0
        discarded = 0
        garaged = 0
        if self.model_store is not None:
            try:
                for rec in self.model_store.list_models():
                    if rec.garaged:
                        garaged += 1
                    elif rec.status == "active":
                        survived += 1
                    elif rec.status == "discarded":
                        discarded += 1
            except Exception:
                logger.exception("Failed to build population summary (non-fatal)")

        return {
            "run_id": run_id,
            "status": status,
            "generations_completed": len(result.generations),
            "generations_requested": n_generations_requested,
            "total_agents_trained": self._total_agents_trained,
            "total_agents_evaluated": self._total_agents_evaluated,
            "wall_time_seconds": round(wall_time, 2),
            "best_model": best_model,
            "top_5": top_5,
            "population_summary": {
                "survived": survived,
                "discarded": discarded,
                "garaged": garaged,
            },
            "error_message": error_message,
            # Keep the old key so any existing consumers still see it.
            "final_rankings": len(rankings),
        }

    def _put_event(self, event: dict) -> None:
        """Put an event on the progress queue (non-blocking).

        Handles both ``asyncio.Queue`` and ``queue.Queue`` (thread-safe).
        """
        if self.progress_queue is not None:
            try:
                self.progress_queue.put_nowait(event)
            except Exception:
                pass  # drop if consumer is behind (Full for either queue type)
