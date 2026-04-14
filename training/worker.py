"""Standalone training worker process.

Runs independently from the API.  Accepts commands (start/stop/status)
over a WebSocket server and streams progress events back to connected
clients (typically the API, which proxies them to the Angular frontend).

Rich terminal output shows live progress with progress bars and tables.

Usage::

    python -m training.worker          # default port 8002
    python -m training.worker --port 8003
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import queue as thread_queue
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import websockets
import websockets.server
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from registry.model_store import ModelStore
from registry.scoreboard import Scoreboard
from training.training_plan import PlanRegistry, TrainingPlan
from training.ipc import (
    DEFAULT_WORKER_HOST,
    DEFAULT_WORKER_PORT,
    CMD_EVALUATE,
    CMD_FINISH,
    CMD_START,
    CMD_STOP,
    CMD_STATUS,
    STOP_EVAL_ALL,
    STOP_EVAL_CURRENT,
    STOP_IMMEDIATE,
    make_error_msg,
    make_evaluate_started_msg,
    make_event_msg,
    make_started_msg,
    make_status_msg,
    parse_message,
)

logger = logging.getLogger(__name__)
console = Console()


# ── Async bridge queue ─────────────────────────────────────────────


class _AsyncBridgeQueue:
    """Drop-in ``queue.Queue`` substitute that pushes events directly into
    the asyncio event loop via ``loop.call_soon_threadsafe``.

    This eliminates GIL contention between the training thread and the
    asyncio event bridge.  The previous approach used
    ``asyncio.to_thread(queue.get, timeout=…)`` which dispatched to the
    default ``ThreadPoolExecutor``, starving the training thread on
    Windows / Python 3.14.

    Only ``put_nowait`` is meaningful — the handler runs in the event
    loop thread.  ``empty`` / ``get_nowait`` are stubs so callers that
    drain the queue (e.g. reset logic) are no-ops instead of errors.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        handler: object,
    ) -> None:
        self._loop = loop
        self._handler = handler

    def put_nowait(self, item: dict) -> None:  # noqa: N802
        self._loop.call_soon_threadsafe(self._handler, item)

    def empty(self) -> bool:
        return True

    def get_nowait(self) -> None:  # noqa: N802
        raise thread_queue.Empty


# ── Worker ──────────────────────────────────────────────────────────


class TrainingWorker:
    """WebSocket server that manages training runs."""

    def __init__(self, config: dict, host: str, port: int, config_path: str = "config.yaml") -> None:
        self.config = config
        self.host = host
        self.port = port
        self.config_path = config_path

        # ModelStore (shared DB via WAL)
        db_path = config["paths"]["registry_db"]
        weights_dir = config["paths"]["model_weights"]
        bet_logs_dir = config["paths"].get("bet_logs") or str(Path(db_path).parent / "bet_logs")
        self.store = ModelStore(
            db_path=db_path,
            weights_dir=weights_dir,
            bet_logs_dir=bet_logs_dir,
        )

        # Plan registry — same path convention as the API
        plan_root = (
            config.get("paths", {}).get("training_plans")
            or str(Path(db_path).parent / "training_plans")
        )
        self.plan_registry = PlanRegistry(plan_root)

        # Training state
        self.running = False
        self.stop_event = threading.Event()
        self.finish_event = threading.Event()
        self.skip_training_event = threading.Event()
        self.stop_after_current_eval_event = threading.Event()
        self.progress_queue: thread_queue.Queue | _AsyncBridgeQueue = thread_queue.Queue()
        self.training_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Active plan tracking (for status updates on terminal events)
        self._active_plan_id: str | None = None

        # Latest state for catch-up on connect / status queries
        self.latest_event: dict | None = None
        self.latest_process: dict | None = None
        self.latest_item: dict | None = None
        self.latest_overall: dict | None = None

        # Connected WebSocket clients
        self.clients: set[websockets.server.ServerConnection] = set()

        # Rich progress display
        self._progress: Progress | None = None
        self._process_task_id = None
        self._item_task_id = None
        self._live: Live | None = None

    # ── State snapshot ──────────────────────────────────────────────

    @staticmethod
    def _apply_run_overrides(base_config: dict, params: dict) -> dict:
        """Return a deep-copied config with per-run overrides applied.

        Handles:
        - population_size override (and auto-scaled n_elite)
        - architectures override (restricts the architecture_name choice set)
        - betting_constraints overrides (max_back_price, max_lay_price,
          min_seconds_before_off)

        Fields in *params* that are None or missing fall through to the
        base config values.
        """
        run_config = copy.deepcopy(base_config)

        # Population size
        population_size = params.get("population_size")
        if population_size is not None:
            run_config.setdefault("population", {})["size"] = population_size
            run_config["population"]["n_elite"] = max(1, population_size // 10)

        # Architecture selection
        architectures = params.get("architectures")
        if architectures is not None and len(architectures) > 0:
            run_config.setdefault("hyperparameters", {}).setdefault("search_ranges", {})
            arch_spec = run_config["hyperparameters"]["search_ranges"].get("architecture_name", {})
            arch_spec["type"] = "str_choice"
            arch_spec["choices"] = list(architectures)
            run_config["hyperparameters"]["search_ranges"]["architecture_name"] = arch_spec
            if len(architectures) == 1:
                run_config.setdefault("training", {})["architecture"] = architectures[0]

        # Market type filter restriction (same pattern as architecture selection)
        market_type_filters = params.get("market_type_filters")
        if market_type_filters is not None and len(market_type_filters) > 0:
            run_config.setdefault("hyperparameters", {}).setdefault("search_ranges", {})
            mtf_spec = run_config["hyperparameters"]["search_ranges"].get("market_type_filter", {})
            mtf_spec["type"] = "str_choice"
            mtf_spec["choices"] = list(market_type_filters)
            run_config["hyperparameters"]["search_ranges"]["market_type_filter"] = mtf_spec

        # Starting budget override
        if params.get("starting_budget") is not None:
            run_config.setdefault("training", {})["starting_budget"] = params["starting_budget"]

        # Betting constraints
        bc_cfg = run_config.setdefault("training", {}).setdefault("betting_constraints", {})
        if params.get("max_back_price") is not None:
            bc_cfg["max_back_price"] = params["max_back_price"]
        if params.get("max_lay_price") is not None:
            bc_cfg["max_lay_price"] = params["max_lay_price"]
        if params.get("min_seconds_before_off") is not None:
            bc_cfg["min_seconds_before_off"] = params["min_seconds_before_off"]

        # Mutation cap and breeding pool — per-run population overrides.
        if params.get("max_mutations_per_child") is not None:
            run_config.setdefault("population", {})["max_mutations_per_child"] = (
                params["max_mutations_per_child"]
            )
        if params.get("breeding_pool") is not None:
            run_config.setdefault("population", {})["breeding_pool"] = (
                params["breeding_pool"]
            )

        # Adaptive breeding overrides (Issue 09).
        if params.get("mutation_rate") is not None:
            run_config.setdefault("population", {})["mutation_rate"] = (
                params["mutation_rate"]
            )
        for key in (
            "bad_generation_threshold",
            "bad_generation_policy",
            "adaptive_mutation",
            "adaptive_mutation_increment",
            "adaptive_mutation_cap",
        ):
            if params.get(key) is not None:
                run_config.setdefault("population", {})[key] = params[key]

        return run_config

    def _reload_config_from_disk(self) -> bool:
        """Reload config.yaml from disk into self.config.

        Called at the start of each training run so changes made via the
        Admin UI (e.g. betting constraints) take effect without requiring
        a worker restart.

        Returns True on success, False on any error (in which case
        self.config is left unchanged).
        """
        try:
            with open(self.config_path) as f:
                fresh_config = yaml.safe_load(f)
            if fresh_config is None:
                console.print(f"[yellow]config.yaml is empty — keeping in-memory config[/yellow]")
                return False
            self.config = fresh_config
            console.print("[dim]Reloaded config.yaml from disk[/dim]")
            return True
        except Exception as e:
            console.print(f"[yellow]Could not reload config.yaml: {e} — using in-memory config[/yellow]")
            return False

    def _state_msg(self) -> str:
        return make_status_msg(
            running=self.running,
            latest_event=self.latest_event,
            latest_process=self.latest_process,
            latest_item=self.latest_item,
            latest_overall=self.latest_overall,
        )

    # ── WebSocket handler ───────────────────────────────────────────

    async def _handle_client(
        self, websocket: websockets.server.ServerConnection,
    ) -> None:
        self.clients.add(websocket)
        peer = websocket.remote_address
        console.print(f"[dim]Client connected: {peer}[/dim]")

        # Send current state so mid-run clients get caught up
        try:
            await websocket.send(self._state_msg())
        except Exception:
            self.clients.discard(websocket)
            return

        try:
            async for raw in websocket:
                msg = parse_message(raw)
                await self._dispatch(msg, websocket)
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            console.print(f"[dim]Client disconnected: {peer}[/dim]")

    async def _dispatch(
        self, msg: dict, ws: websockets.server.ServerConnection,
    ) -> None:
        msg_type = msg.get("type")

        if msg_type == CMD_STATUS:
            await ws.send(self._state_msg())

        elif msg_type == CMD_START:
            if self.running:
                await ws.send(make_error_msg("Training already in progress"))
                return
            await self._start_training(msg, ws)

        elif msg_type == CMD_EVALUATE:
            if self.running:
                await ws.send(make_error_msg(
                    "Worker is busy — cannot start evaluation while another job is running"
                ))
                return
            await self._start_evaluation(msg, ws)

        elif msg_type == CMD_STOP:
            if not self.running:
                await ws.send(make_error_msg("No training run in progress"))
                return
            granularity = msg.get("granularity", STOP_IMMEDIATE)
            if granularity == STOP_IMMEDIATE:
                self.stop_event.set()
                console.print("[yellow]Stop requested (immediate) — halting after current agent[/yellow]")
            elif granularity == STOP_EVAL_CURRENT:
                self.stop_after_current_eval_event.set()
                console.print("[yellow]Stop requested (eval_current) — finishing current evaluation then stopping[/yellow]")
            elif granularity == STOP_EVAL_ALL:
                self.finish_event.set()
                self.skip_training_event.set()
                console.print("[yellow]Stop requested (eval_all) — skipping training, evaluating all models[/yellow]")
            else:
                await ws.send(make_error_msg(f"Invalid stop granularity: {granularity}"))
                return
            # Send status back so the API's pending Future resolves
            await ws.send(self._state_msg())

        elif msg_type == CMD_FINISH:
            if not self.running:
                await ws.send(make_error_msg("No training run in progress"))
                return
            self.finish_event.set()
            console.print("[yellow]Finish requested — will evaluate current population then stop[/yellow]")
            await ws.send(self._state_msg())

        else:
            await ws.send(make_error_msg(f"Unknown command: {msg_type}"))

    # ── Training lifecycle ──────────────────────────────────────────

    async def _start_training(
        self, params: dict, ws: websockets.server.ServerConnection,
    ) -> None:
        from data.episode_builder import load_days

        # Resolve data
        data_dir = self.config["paths"]["processed_data"]
        processed = Path(data_dir)
        dates = sorted(
            f.stem
            for f in processed.glob("*.parquet")
            if not f.stem.endswith("_runners") and f.stem != ".gitkeep"
        )
        if not dates:
            await ws.send(make_error_msg("No extracted data available — import days first"))
            return

        # Chronological train/test split (~50/50), overridden by explicit dates
        if params.get("train_dates") is not None:
            train_dates = sorted(params["train_dates"])
        else:
            split = max(1, len(dates) // 2)
            train_dates = dates[:split]
        if params.get("test_dates") is not None:
            test_dates = sorted(params["test_dates"])
        else:
            split = max(1, len(dates) // 2)
            test_dates = dates[split:]

        run_id = str(uuid.uuid4())

        # Load training plan if plan_id provided
        plan_id = params.get("plan_id")
        training_plan: TrainingPlan | None = None
        if plan_id is not None:
            try:
                training_plan = self.plan_registry.load(plan_id)
                console.print(f"[dim]Loaded training plan: {training_plan.name} ({plan_id[:8]}…)[/dim]")
            except (KeyError, ValueError) as exc:
                await ws.send(make_error_msg(f"Failed to load plan {plan_id}: {exc}"))
                return

        n_generations = params.get("n_generations", 3)
        n_epochs = params.get("n_epochs", 3)
        population_size = params.get("population_size")
        seed = params.get("seed")
        reevaluate_garaged = params.get("reevaluate_garaged", False)
        reevaluate_min_score = params.get("reevaluate_min_score")
        start_generation = params.get("start_generation", 0)

        # Session splitting: on initial launch (start_generation == 0),
        # scope n_generations to the first session's range if the plan
        # defines generations_per_session.
        if (
            training_plan is not None
            and training_plan.generations_per_session is not None
            and start_generation == 0
        ):
            boundaries = training_plan.session_boundaries()
            if boundaries:
                s0_start, s0_end = boundaries[0]
                n_generations = s0_end - s0_start + 1
                console.print(
                    f"[dim]Session splitting: running session 1/{len(boundaries)} "
                    f"(gen {s0_start}-{s0_end})[/dim]"
                )

        # Reload config from disk so changes made via the Admin UI
        # (e.g. betting constraints, reevaluate_garaged_default) take
        # effect on the next run without needing to restart the worker.
        self._reload_config_from_disk()

        # Re-resolve data dir from (possibly) updated config
        data_dir = self.config["paths"]["processed_data"]

        # Build run config with all per-run overrides applied
        run_config = self._apply_run_overrides(self.config, params)

        # Mark plan as running
        if training_plan is not None:
            try:
                from datetime import datetime, timezone
                self.plan_registry.set_status(
                    training_plan.plan_id,
                    "running",
                    started_at=datetime.now(timezone.utc).isoformat(),
                    current_generation=start_generation,
                )
            except Exception:
                logger.exception("Failed to set plan status to running")

        # Track the active plan_id so the event handler can update status
        self._active_plan_id: str | None = plan_id
        # Stash params for auto-continue (next session re-uses same config)
        self._last_start_params: dict = dict(params)

        # Reset
        self.stop_event.clear()
        self.finish_event.clear()
        self.skip_training_event.clear()
        self.stop_after_current_eval_event.clear()
        self.running = True
        self.latest_event = None
        self.latest_process = None
        self.latest_item = None
        self.latest_overall = None

        # Clear any stale events from the queue
        while not self.progress_queue.empty():
            try:
                self.progress_queue.get_nowait()
            except thread_queue.Empty:
                break

        # Acknowledge — include plan_id so the frontend can cross-link
        await self._broadcast(make_started_msg(run_id, train_dates, test_dates, plan_id=plan_id))

        console.print()
        console.rule(f"[bold green]Training Run {run_id[:8]}[/bold green]")
        if training_plan:
            console.print(f"  Plan: {training_plan.name} ({training_plan.plan_id[:8]}…)")
        if start_generation > 0:
            console.print(f"  Generations: {n_generations} (starting at gen {start_generation})  |  Epochs: {n_epochs}")
        else:
            console.print(f"  Generations: {n_generations}  |  Epochs: {n_epochs}")
        console.print(f"  Population:  {run_config['population']['size']}")
        console.print(f"  Train days:  {len(train_dates)}  |  Test days: {len(test_dates)}")
        console.print()

        def _run_in_thread() -> None:
            from training.run_training import TrainingOrchestrator

            try:
                console.print("[dim]Loading training data...[/dim]")
                train_days = load_days(train_dates, data_dir=data_dir, progress_queue=self.progress_queue)
                test_days_loaded = load_days(test_dates, data_dir=data_dir, progress_queue=self.progress_queue)
                console.print(f"[dim]Loaded {len(train_days)} train days, {len(test_days_loaded)} test days[/dim]")

                orch = TrainingOrchestrator(
                    config=run_config,
                    model_store=self.store,
                    progress_queue=self.progress_queue,
                    stop_event=self.stop_event,
                    finish_event=self.finish_event,
                    skip_training_event=self.skip_training_event,
                    stop_after_current_eval_event=self.stop_after_current_eval_event,
                    training_plan=training_plan,
                    plan_registry=self.plan_registry if training_plan else None,
                    stud_model_ids=params.get("stud_model_ids") or None,
                )

                orch.run(
                    train_days=train_days,
                    test_days=test_days_loaded,
                    n_generations=n_generations,
                    n_epochs=n_epochs,
                    seed=seed,
                    reevaluate_garaged=reevaluate_garaged,
                    reevaluate_min_score=reevaluate_min_score,
                    start_generation=start_generation,
                )
            except Exception as exc:
                logger.exception("Training run failed")
                # Write crash file for post-mortem debugging
                crash_dir = Path("logs/crashes")
                crash_dir.mkdir(parents=True, exist_ok=True)
                crash_file = crash_dir / f"crash_{time.strftime('%Y%m%d_%H%M%S')}.log"
                crash_file.write_text(traceback.format_exc())
                logger.info("Crash details written to %s", crash_file)
                try:
                    self.progress_queue.put_nowait({
                        "event": "phase_complete",
                        "phase": "run_error",
                        "timestamp": time.time(),
                        "summary": {"error": str(exc)},
                    })
                except thread_queue.Full:
                    pass

        self.training_thread = threading.Thread(
            target=_run_in_thread, daemon=True, name="training-run",
        )
        self.training_thread.start()

    # ── Evaluation lifecycle ────────────────────────────────────────

    async def _start_evaluation(
        self, params: dict, ws: websockets.server.ServerConnection,
    ) -> None:
        """Handle CMD_EVALUATE — run the Evaluator on N models × M test days."""
        from data.episode_builder import load_days

        model_ids: list[str] = list(params.get("model_ids") or [])
        test_dates_raw: list[str] | None = params.get("test_dates")

        if not model_ids:
            await ws.send(make_error_msg("No model_ids provided"))
            return

        # Validate models exist + have weights
        missing: list[str] = []
        no_weights: list[str] = []
        for mid in model_ids:
            rec = self.store.get_model(mid)
            if rec is None:
                missing.append(mid)
            elif not rec.weights_path:
                no_weights.append(mid)
        if missing:
            await ws.send(make_error_msg(f"Model(s) not found: {missing}"))
            return
        if no_weights:
            await ws.send(make_error_msg(f"Model(s) have no saved weights: {no_weights}"))
            return

        # Reload config so admin-edited constraints take effect
        self._reload_config_from_disk()
        data_dir = self.config["paths"]["processed_data"]
        processed = Path(data_dir)
        all_dates = sorted(
            f.stem
            for f in processed.glob("*.parquet")
            if not f.stem.endswith("_runners") and f.stem != ".gitkeep"
        )
        if not all_dates:
            await ws.send(make_error_msg("No extracted data available — import days first"))
            return

        if test_dates_raw is None:
            test_dates = all_dates
        else:
            unknown = [d for d in test_dates_raw if d not in all_dates]
            if unknown:
                await ws.send(make_error_msg(f"Test dates not found in data: {unknown}"))
                return
            test_dates = sorted(test_dates_raw)

        job_id = str(uuid.uuid4())

        # Reset state
        self.stop_event.clear()
        self.finish_event.clear()
        self.skip_training_event.clear()
        self.stop_after_current_eval_event.clear()
        self.running = True
        self.latest_event = None
        self.latest_process = None
        self.latest_item = None
        self.latest_overall = None
        self._active_plan_id = None

        while not self.progress_queue.empty():
            try:
                self.progress_queue.get_nowait()
            except thread_queue.Empty:
                break

        await self._broadcast(make_evaluate_started_msg(job_id, model_ids, test_dates))

        console.print()
        console.rule(f"[bold cyan]Manual Evaluation {job_id[:8]}[/bold cyan]")
        console.print(f"  Models: {len(model_ids)}  |  Test days: {len(test_dates)}")
        console.print()

        config_for_run = copy.deepcopy(self.config)

        def _run_in_thread() -> None:
            from training.evaluator import Evaluator
            from agents.architecture_registry import create_policy
            from env.betfair_env import (
                ACTIONS_PER_RUNNER,
                AGENT_STATE_DIM,
                MARKET_DIM,
                OBS_SCHEMA_VERSION,
                POSITION_DIM,
                RUNNER_DIM,
                VELOCITY_DIM,
            )

            try:
                self.progress_queue.put_nowait({
                    "event": "phase_start",
                    "phase": "evaluating",
                    "timestamp": time.time(),
                    "summary": {
                        "model_count": len(model_ids),
                        "day_count": len(test_dates),
                        "manual_evaluation": True,
                    },
                })

                console.print("[dim]Loading test data...[/dim]")
                test_days = load_days(
                    test_dates, data_dir=data_dir, progress_queue=self.progress_queue,
                )
                console.print(f"[dim]Loaded {len(test_days)} test days[/dim]")

                max_runners = config_for_run["training"]["max_runners"]
                obs_dim = (
                    MARKET_DIM
                    + VELOCITY_DIM
                    + (RUNNER_DIM * max_runners)
                    + AGENT_STATE_DIM
                    + (POSITION_DIM * max_runners)
                )
                action_dim = max_runners * ACTIONS_PER_RUNNER

                evaluator = Evaluator(
                    config=config_for_run,
                    model_store=self.store,
                    progress_queue=self.progress_queue,
                )

                train_cutoff = test_dates[0] if test_dates else ""
                completed = 0
                failed: list[tuple[str, str]] = []

                for idx, mid in enumerate(model_ids):
                    if self.stop_event.is_set():
                        break

                    self.progress_queue.put_nowait({
                        "event": "progress",
                        "phase": "evaluating",
                        "process": {
                            "label": f"Evaluating {len(model_ids)} models",
                            "completed": idx,
                            "total": len(model_ids),
                            "pct": 100.0 * idx / max(len(model_ids), 1),
                            "item_eta_human": "",
                            "process_eta_human": "",
                        },
                        "detail": f"Evaluating model {mid[:12]} ({idx+1}/{len(model_ids)})",
                    })

                    try:
                        record = self.store.get_model(mid)
                        if record is None:
                            raise ValueError(f"Model {mid} disappeared from registry")
                        hp = record.hyperparameters or {}
                        arch_name = record.architecture_name
                        market_type_filter = hp.get("market_type_filter", "BOTH")

                        policy = create_policy(
                            name=arch_name,
                            obs_dim=obs_dim,
                            action_dim=action_dim,
                            max_runners=max_runners,
                            hyperparams=hp,
                        )
                        state_dict = self.store.load_weights(
                            mid,
                            expected_obs_schema_version=OBS_SCHEMA_VERSION,
                        )
                        policy.load_state_dict(state_dict)

                        evaluator.evaluate(
                            model_id=mid,
                            policy=policy,
                            test_days=test_days,
                            train_cutoff_date=train_cutoff,
                            market_type_filter=market_type_filter,
                        )
                        completed += 1
                    except Exception as exc:
                        logger.exception("Evaluation failed for model %s", mid)
                        failed.append((mid, str(exc)))
                        try:
                            self.progress_queue.put_nowait({
                                "event": "progress",
                                "phase": "evaluating",
                                "detail": f"Model {mid[:12]} failed: {exc}",
                            })
                        except thread_queue.Full:
                            pass

                self.progress_queue.put_nowait({
                    "event": "phase_complete",
                    "phase": "run_complete",
                    "timestamp": time.time(),
                    "summary": {
                        "manual_evaluation": True,
                        "models_evaluated": completed,
                        "models_failed": len(failed),
                        "failed_model_ids": [m for m, _ in failed],
                    },
                })
            except Exception as exc:
                logger.exception("Manual evaluation crashed")
                crash_dir = Path("logs/crashes")
                crash_dir.mkdir(parents=True, exist_ok=True)
                crash_file = crash_dir / f"crash_eval_{time.strftime('%Y%m%d_%H%M%S')}.log"
                crash_file.write_text(traceback.format_exc())
                logger.info("Crash details written to %s", crash_file)
                try:
                    self.progress_queue.put_nowait({
                        "event": "phase_complete",
                        "phase": "run_error",
                        "timestamp": time.time(),
                        "summary": {"error": str(exc), "manual_evaluation": True},
                    })
                except thread_queue.Full:
                    pass

        self.training_thread = threading.Thread(
            target=_run_in_thread, daemon=True, name="manual-evaluation",
        )
        self.training_thread.start()

    # ── Event bridge ────────────────────────────────────────────────

    def _handle_event(self, event: dict) -> None:
        """Process a progress event in the asyncio event loop thread.

        Called via ``loop.call_soon_threadsafe`` from the training thread
        (through ``_AsyncBridgeQueue.put_nowait``).  This replaces the old
        ``_bridge_events`` coroutine that polled a ``queue.Queue`` via
        ``asyncio.to_thread`` — which caused GIL contention that starved
        the training thread on Windows / Python 3.14.

        Must be synchronous (``call_soon_threadsafe`` requirement).
        Broadcasts to WebSocket clients via ``asyncio.create_task``.
        """
        # Update local state
        self.latest_event = event

        if event.get("process"):
            self.latest_process = event["process"]
        if event.get("item"):
            self.latest_item = event["item"]
        if event.get("overall"):
            self.latest_overall = event["overall"]

        is_terminal = False
        if event.get("event") == "phase_start":
            self.running = True
            self.latest_item = None
        elif (
            event.get("event") == "run_complete"
            or (
                event.get("event") == "phase_complete"
                and event.get("phase") in (
                    "run_complete", "run_stopped", "run_error",
                )
            )
        ):
            self.running = False
            self.latest_process = None
            self.latest_item = None
            self.latest_overall = None
            is_terminal = True

            # Update plan status on terminal events
            plan_id = getattr(self, "_active_plan_id", None)
            if plan_id is not None:
                phase = event.get("phase", "")
                if phase in ("run_error", "run_stopped"):
                    # Error or user-stop: mark failed, clear plan
                    try:
                        from datetime import datetime, timezone
                        self.plan_registry.set_status(
                            plan_id, "failed",
                            completed_at=datetime.now(timezone.utc).isoformat(),
                        )
                    except Exception:
                        logger.exception("Failed to set plan status to failed")
                    self._active_plan_id = None
                else:
                    # Successful run_complete — check for remaining sessions
                    self._handle_session_complete(plan_id)

        # Broadcast to WS clients (async — schedule as a task)
        msg = make_event_msg(event)
        asyncio.create_task(self._broadcast(msg))

        # Print to terminal
        self._print_event(event, is_terminal)

    def _handle_session_complete(self, plan_id: str) -> None:
        """Handle plan session completion — auto-continue or pause.

        Called from _handle_event when a plan-based run completes
        successfully.  Decides whether to launch the next session,
        pause, or mark the plan as fully completed.
        """
        from datetime import datetime, timezone

        try:
            plan = self.plan_registry.load(plan_id)
        except Exception:
            logger.exception("Failed to load plan %s for session check", plan_id)
            self._active_plan_id = None
            return

        if plan.has_remaining_sessions:
            # Advance to next session
            try:
                plan = self.plan_registry.advance_session(plan_id)
            except Exception:
                logger.exception("Failed to advance session for plan %s", plan_id)
                self._active_plan_id = None
                return

            if plan.auto_continue:
                # Auto-launch next session
                console.print(
                    f"[bold green]Auto-continuing to session {plan.current_session + 1}"
                    f"/{plan.total_sessions}[/bold green]"
                )
                self._launch_next_session(plan)
            else:
                # Pause and wait for manual continue
                try:
                    self.plan_registry.set_status(plan_id, "paused")
                except Exception:
                    logger.exception("Failed to set plan status to paused")
                console.print(
                    f"[yellow]Session {plan.current_session}/{plan.total_sessions} "
                    f"complete — paused, waiting for Continue[/yellow]"
                )
                self._active_plan_id = None
        else:
            # All sessions done
            try:
                self.plan_registry.set_status(
                    plan_id, "completed",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
            except Exception:
                logger.exception("Failed to set plan status to completed")
            self._active_plan_id = None

    def _launch_next_session(self, plan: TrainingPlan) -> None:
        """Launch the next session of a plan via auto-continue.

        Reuses the stashed start params but with updated generation range.
        """
        boundaries = plan.session_boundaries()
        session_idx = plan.current_session
        if session_idx >= len(boundaries):
            logger.warning("Session index %d out of bounds for plan %s", session_idx, plan.plan_id)
            self._active_plan_id = None
            return

        start_gen, end_gen = boundaries[session_idx]
        n_gens_this_session = end_gen - start_gen + 1

        # Build params from the stashed start params
        params = dict(getattr(self, "_last_start_params", {}))
        params["n_generations"] = n_gens_this_session
        params["start_generation"] = start_gen
        params["plan_id"] = plan.plan_id

        # Schedule the next session start in the event loop
        # (we're in a sync callback, so use create_task)
        async def _start():
            # Create a dummy websocket reference — we'll broadcast to all clients
            # Use the first connected client, or None
            ws = next(iter(self.clients), None)
            if ws is None:
                logger.warning("No WebSocket clients connected for auto-continue")
                return
            await self._start_training(params, ws)

        asyncio.create_task(_start())

    async def _check_dead_thread(self) -> None:
        """Periodically check if the training thread died without sending
        a terminal event."""
        while True:
            await asyncio.sleep(2.0)
            if self.running and self.training_thread and not self.training_thread.is_alive():
                self.running = False
                self.latest_process = None
                self.latest_item = None
                self.latest_overall = None
                console.print("[red]Training thread exited unexpectedly[/red]")
                # Update plan status if a plan-based run died
                plan_id = self._active_plan_id
                if plan_id is not None:
                    try:
                        from datetime import datetime, timezone
                        self.plan_registry.set_status(
                            plan_id,
                            "failed",
                            completed_at=datetime.now(timezone.utc).isoformat(),
                        )
                    except Exception:
                        logger.exception("Failed to set plan status to failed (dead thread)")
                    self._active_plan_id = None

    async def _broadcast(self, msg: str) -> None:
        dead: set = set()
        for client in self.clients:
            try:
                await client.send(msg)
            except Exception:
                dead.add(client)
        self.clients -= dead

    # Keepalive is handled by websockets library's built-in ping/pong
    # (ping_interval=30 on the server). No application-level pings needed.

    # ── Rich terminal output ────────────────────────────────────────

    def _print_event(self, event: dict, is_terminal: bool) -> None:
        evt_type = event.get("event", "")
        phase = event.get("phase", "")
        detail = event.get("detail", "")
        summary = event.get("summary", {})

        if evt_type == "phase_start":
            gen = event.get("generation") or summary.get("generation", "")
            pop = summary.get("population_size", "")
            label = phase.replace("_", " ").title()
            parts = [f"[bold cyan]{label}[/bold cyan]"]
            if gen != "":
                parts.append(f"Generation {gen}")
            if pop:
                parts.append(f"{pop} agents")
            console.print()
            console.rule(" | ".join(parts))

        elif evt_type == "progress":
            proc = event.get("process", {})
            completed = proc.get("completed", 0)
            total = proc.get("total", 0)
            pct = proc.get("pct", 0)
            process_eta = proc.get("process_eta_human", "")
            item = event.get("item", {})
            item_completed = item.get("completed", 0) if item else 0
            item_total = item.get("total", 0) if item else 0

            # Build a compact progress line
            parts = []
            if total:
                parts.append(f"[{completed}/{total}]")
            if detail:
                parts.append(detail)
            if item and item_total:
                parts.append(f"(step {item_completed}/{item_total})")
            if process_eta:
                parts.append(f"[dim]ETA {process_eta}[/dim]")

            console.print("  ".join(parts))

        elif evt_type == "phase_complete":
            if phase == "run_complete":
                console.print()
                console.rule("[bold green]Training Complete[/bold green]")
            elif phase == "run_stopped":
                console.print()
                console.rule("[bold yellow]Training Stopped[/bold yellow]")
            elif phase == "run_error":
                console.print()
                err = summary.get("error", "Unknown error")
                console.rule(f"[bold red]Training Error: {err}[/bold red]")
            else:
                # Phase-level completion (scoring, selecting, breeding etc.)
                label = phase.replace("_", " ").title()
                if summary:
                    info = "  ".join(f"{k}={v}" for k, v in summary.items())
                    console.print(f"  [green]{label} complete[/green]  {info}")
                else:
                    console.print(f"  [green]{label} complete[/green]")

        elif evt_type == "run_complete":
            console.print()
            console.rule("[bold green]Training Complete[/bold green]")

    # ── Main serve loop ─────────────────────────────────────────────

    async def serve(self) -> None:
        self._loop = asyncio.get_running_loop()

        # Replace the plain thread_queue.Queue with a bridge that pushes
        # events directly into the asyncio event loop.  All producers
        # (TrainingOrchestrator, Evaluator, load_days) call put_nowait()
        # unchanged — the _AsyncBridgeQueue dispatches via
        # loop.call_soon_threadsafe, eliminating the GIL contention that
        # the old asyncio.to_thread(queue.get) polling loop caused.
        self.progress_queue = _AsyncBridgeQueue(self._loop, self._handle_event)

        dead_thread_task = asyncio.create_task(self._check_dead_thread())

        async with websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10,
        ):
            console.rule("[bold green]Training Worker[/bold green]")
            console.print(f"  Listening on [bold]ws://{self.host}:{self.port}[/bold]")
            console.print(f"  Press Ctrl+C to stop")
            console.print()

            try:
                await asyncio.Future()  # run forever
            except asyncio.CancelledError:
                pass
            finally:
                dead_thread_task.cancel()


# ── Entry point ─────────────────────────────────────────────────────


_SAFE_PROCESS_PREFIXES = ("python", "node", "npm")


def _get_process_name(pid: int) -> str | None:
    """Return the image name (e.g. 'python.exe') for a PID, or None."""
    try:
        import subprocess as _sp
        result = _sp.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) >= 2:
                return parts[0].strip('"').lower()
    except Exception:
        pass
    return None


def _clear_port(port: int) -> None:
    """Kill any stale process listening on the given port.

    Safety: only kills python/node/npm processes. Never uses /T (tree kill)
    to avoid accidentally terminating system or GPU processes.
    """
    if sys.platform != "win32":
        return  # Unix handles this via SO_REUSEADDR or manual kill
    try:
        import subprocess as _sp
        result = _sp.run(["netstat", "-ano"], capture_output=True, text=True, timeout=10)
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = int(parts[-1])
                if pid > 0:
                    proc_name = _get_process_name(pid)
                    if proc_name and any(proc_name.startswith(p) for p in _SAFE_PROCESS_PREFIXES):
                        console.print(f"[yellow]Killing stale {proc_name} on port {port} (PID {pid})[/yellow]")
                        _sp.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=10)
                    else:
                        console.print(f"[yellow]Skipping unknown process on port {port} (PID {pid}, name={proc_name!r})[/yellow]")
        import time as _time
        _time.sleep(0.5)  # Brief pause to let the OS release the port
    except Exception as exc:
        console.print(f"[yellow]Warning: could not clear port {port}: {exc}[/yellow]")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="rl-betfair training worker")
    parser.add_argument("--port", type=int, default=None, help="WebSocket port")
    parser.add_argument("--host", type=str, default=None, help="Bind host")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    worker_cfg = config.get("training_worker", {})
    host = args.host or worker_cfg.get("host", DEFAULT_WORKER_HOST)
    port = args.port or worker_cfg.get("port", DEFAULT_WORKER_PORT)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s",
    )
    # Suppress noisy websockets library logging (logs every frame/connection)
    logging.getLogger("websockets").setLevel(logging.WARNING)

    # Kill any stale process holding our port before starting
    _clear_port(port)

    worker = TrainingWorker(config=config, host=host, port=port, config_path=args.config)

    try:
        asyncio.run(worker.serve())
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
    except OSError as exc:
        if "10048" in str(exc) or "address already in use" in str(exc).lower():
            console.print(f"[red bold]Port {port} is still in use after cleanup attempt.[/red bold]")
            console.print(f"[red]Run: stop-training.bat  or manually kill the process on port {port}[/red]")
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
