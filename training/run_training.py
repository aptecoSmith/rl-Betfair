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
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import torch

from agents.population_manager import (
    AgentRecord,
    BreedingRecord,
    PopulationManager,
    SelectionResult,
)
from agents.ppo_trainer import PPOTrainer, TrainingStats
from data.episode_builder import Day
from registry.model_store import ModelStore
from registry.scoreboard import ModelScore, Scoreboard
from training.evaluator import Evaluator
from training.perf_log import gpu_memory_summary, perf_log
from training.progress_tracker import ProgressTracker

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
    ) -> None:
        self.config = config
        self.model_store = model_store
        self.progress_queue = progress_queue
        self.stop_event = stop_event
        self.finish_event = finish_event

        # GPU auto-detection
        if device is not None:
            self.device = device
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

    def run(
        self,
        train_days: list[Day],
        test_days: list[Day],
        n_generations: int = 1,
        n_epochs: int = 1,
        seed: int | None = None,
        reevaluate_garaged: bool = False,
        reevaluate_min_score: float | None = None,
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
            Number of generations to run.
        n_epochs :
            Number of training passes over the training days per agent.
        seed :
            Optional RNG seed for population initialisation.

        Returns
        -------
        TrainingRunResult with full results from all generations.
        """
        run_id = str(uuid.uuid4())
        result = TrainingRunResult(run_id=run_id)

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

        # -- Generation 0: initialise population --
        self._emit_phase_start("training", {
            "run_id": run_id,
            "n_generations": n_generations,
            "train_days": len(train_days),
            "test_days": len(test_days),
        })

        agents = self.pop_manager.initialise_population(generation=0, seed=seed)

        for gen in range(n_generations):
            if self._check_stop():
                break

            # Finish requested between generations: skip remaining, evaluate, exit
            finishing_between_gens = self._check_finish() and gen > 0
            if finishing_between_gens:
                logger.info("Finish requested — skipping to final evaluation (gen %d)", gen)
                self._emit_phase_start("finishing_early", {
                    "generation": gen,
                    "skipped_generations": n_generations - gen,
                })

            logger.info("=== Generation %d ===", gen)
            gen_result = self._run_generation(
                generation=gen,
                agents=agents,
                train_days=train_days,
                test_days=test_days,
                train_cutoff=train_cutoff,
                n_epochs=n_epochs,
                is_last=(gen == n_generations - 1) or finishing_between_gens,
                skip_training=finishing_between_gens,
            )
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
            if gen < n_generations - 1:
                # Survivors carry over, children are new
                survivor_agents = []
                if gen_result.selection is not None:
                    for mid in gen_result.selection.survivors:
                        try:
                            agent = self.pop_manager.load_agent(mid)
                            survivor_agents.append(agent)
                        except Exception:
                            logger.warning(
                                "Could not load survivor %s, skipping", mid,
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

        self._emit_phase_complete("run_complete", {
            "run_id": run_id,
            "generations_completed": n_generations,
            "final_rankings": len(result.final_rankings),
        })

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
                self.evaluator.evaluate(
                    model_id=model_rec.model_id,
                    policy=agent.policy,
                    test_days=test_days,
                    train_cutoff_date=train_cutoff,
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

            for agent in agents:
                if self._check_stop() or self._check_finish():
                    break
                self._publish_progress(
                    "training", outer_tracker,
                    detail=f"Training agent {agent.model_id[:12]} ({agent.architecture_name})",
                )

                trainer = PPOTrainer(
                    policy=agent.policy,
                    config=self.config,
                    hyperparams=agent.hyperparameters,
                    progress_queue=self.progress_queue,
                    device=self.device,
                    feature_cache=self.feature_cache,
                )
                with perf_log(
                    logger,
                    f"Train agent {agent.model_id[:12]}",
                    log_gpu=(self.device == "cuda"),
                ):
                    stats = trainer.train(train_days, n_epochs=n_epochs)
                training_stats[agent.model_id] = stats

                # Save updated weights after training
                if self.model_store is not None:
                    self.model_store.save_weights(
                        agent.model_id, agent.policy.state_dict(),
                    )

                outer_tracker.tick()
                self._publish_progress(
                    "training", outer_tracker,
                    detail=(
                        f"Agent {agent.model_id[:12]} done | "
                        f"mean_reward={stats.mean_reward:+.3f} | "
                        f"mean_pnl={stats.mean_pnl:+.2f}"
                    ),
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
                device=self.device,
                feature_cache=self.feature_cache,
            )
            evaluator.evaluate(
                model_id=agent.model_id,
                policy=agent.policy,
                test_days=test_days,
                train_cutoff_date=train_cutoff,
            )

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
        else:
            for agent in agents:
                self._publish_progress(
                    "evaluating", eval_tracker,
                    detail=f"Evaluating agent {agent.model_id[:12]}",
                )
                with perf_log(
                    logger,
                    f"Eval agent {agent.model_id[:12]}",
                    log_gpu=(self.device == "cuda"),
                ):
                    _eval_agent(agent)
                eval_tracker.tick()

        self._emit_phase_complete("evaluating", {
            "generation": generation,
            "agents_evaluated": len(agents),
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

            # Discard policy
            discarded = self.pop_manager.apply_discard_policy(scores)

            # Filter out discarded from selection
            active_scores = [s for s in scores if s.model_id not in discarded]
            if active_scores:
                selection = self.pop_manager.select(active_scores)

                self._emit_phase_complete("selecting", {
                    "generation": generation,
                    "survivors": len(selection.survivors),
                    "eliminated": len(selection.eliminated),
                    "discarded": len(discarded),
                })

                # Breeding
                self._emit_phase_start("breeding", {
                    "generation": generation,
                    "survivors": len(selection.survivors),
                })

                mutation_rate = self.config["population"].get("mutation_rate", 0.3)
                children, breeding_records = self.pop_manager.breed(
                    selection_result=selection,
                    generation=generation + 1,
                    mutation_rate=mutation_rate,
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

        return GenerationResult(
            generation=generation,
            training_stats=training_stats,
            scores=scores,
            selection=selection,
            discarded=discarded,
            children=children,
            breeding_records=breeding_records,
        )

    # -- Progress event helpers ------------------------------------------------

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
        self._put_event(event)

    def _put_event(self, event: dict) -> None:
        """Put an event on the progress queue (non-blocking).

        Handles both ``asyncio.Queue`` and ``queue.Queue`` (thread-safe).
        """
        if self.progress_queue is not None:
            try:
                self.progress_queue.put_nowait(event)
            except Exception:
                pass  # drop if consumer is behind (Full for either queue type)
