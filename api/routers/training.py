"""Training status, start/stop endpoints, and WebSocket for live progress.

Training runs in a separate worker process (training/worker.py).  This
router sends commands to the worker via WebSocket IPC and proxies progress
events back to the Angular frontend.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect

from api.schemas import (
    ArchitectureInfo,
    FinishTrainingResponse,
    GeneticsInfo,
    HyperparamSchemaEntry,
    ProgressSnapshot,
    ResumeTrainingRequest,
    ResumeTrainingResponse,
    StartTrainingRequest,
    StartTrainingResponse,
    StopTrainingResponse,
    TrainingStatus,
)
from training.ipc import (
    make_finish_cmd,
    make_start_cmd,
    make_stop_cmd,
    parse_message,
    EVT_ERROR,
    VALID_STOP_GRANULARITIES,
    STOP_EVAL_ALL,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["training"])


def _state(request: Request) -> dict:
    return request.app.state.training_state


def _worker_ws(request: Request):
    """Return the live worker WebSocket or None."""
    return getattr(request.app.state, "worker_ws", None)


async def _send_to_worker(request: Request, msg: str, timeout: float = 5.0) -> dict:
    """Send a command to the worker and wait for the response.

    The response is delivered by the background worker connection task
    in main.py via ``app.state.worker_pending_response``.

    Raises HTTPException(503) if the worker is unreachable.
    """
    ws = _worker_ws(request)
    if ws is None:
        raise HTTPException(
            503,
            "Training worker is not available. "
            "Start it with: python -m training.worker",
        )
    loop = asyncio.get_event_loop()
    fut: asyncio.Future = loop.create_future()
    request.app.state.worker_pending_response = fut
    try:
        await ws.send(msg)
        return await asyncio.wait_for(fut, timeout=timeout)
    except asyncio.TimeoutError:
        raise HTTPException(504, "Training worker did not respond in time")
    except ConnectionError as exc:
        raise HTTPException(503, f"Lost connection to training worker: {exc}")
    except Exception as exc:
        raise HTTPException(503, f"Training worker error: {exc}")
    finally:
        request.app.state.worker_pending_response = None


@router.get("/training/status")
def get_training_status(request: Request):
    """Current run snapshot: phase, process ETA, item ETA, generation."""
    state = _state(request)

    # If the worker connection is down and we think we're running,
    # report that the worker is disconnected.
    worker_connected = getattr(request.app.state, "worker_connected", False)
    if state["running"] and not worker_connected:
        return TrainingStatus(
            running=True,
            phase="worker_disconnected",
            detail="Training worker connection lost — training may still be running in the worker process",
            worker_connected=False,
            plan_id=state.get("plan_id"),
        )

    if not state["running"]:
        return TrainingStatus(running=False, worker_connected=worker_connected, plan_id=state.get("plan_id"))

    latest = state.get("latest_event")
    if not latest:
        return TrainingStatus(running=True, worker_connected=worker_connected, plan_id=state.get("plan_id"))

    def _snap(d: dict | None) -> ProgressSnapshot | None:
        if not d:
            return None
        return ProgressSnapshot(
            label=d.get("label", ""),
            completed=d.get("completed", 0),
            total=d.get("total", 0),
            pct=d.get("pct", 0.0),
            item_eta_human=d.get("item_eta_human", ""),
            process_eta_human=d.get("process_eta_human", ""),
        )

    # Use independently-tracked snapshots so the poll response
    # always contains overall/process/item, even when the latest
    # event only carried one of them.
    process = _snap(state.get("latest_process"))
    item = _snap(state.get("latest_item"))
    overall = _snap(state.get("latest_overall"))

    return TrainingStatus(
        running=True,
        phase=latest.get("phase"),
        generation=latest.get("generation"),
        process=process,
        item=item,
        overall=overall,
        detail=latest.get("detail"),
        last_agent_score=latest.get("last_agent_score"),
        worker_connected=worker_connected,
        unevaluated_count=latest.get("unevaluated_count"),
        eval_rate_s=latest.get("eval_rate_s"),
        plan_id=state.get("plan_id"),
    )


@router.get("/training/episodes")
def get_training_episodes(
    request: Request,
    since_ts: str | None = None,
    limit: int = 5000,
):
    """Return recent rows from ``logs/training/episodes.jsonl``.

    Powers the learning-curves UI. Reads the jsonl log directly rather
    than going through the WebSocket event stream — the stream already
    emits some of this data but as parsed detail strings, not the full
    per-episode record the UI needs.

    Query params:

    - ``since_ts``: ISO timestamp. Only return episodes whose ``timestamp``
      strictly exceeds this. Used by the frontend to poll for deltas.
    - ``limit``: cap the number of rows returned (default 5000). When
      ``since_ts`` is omitted we return the last ``limit`` rows of the
      file; when ``since_ts`` is supplied we return up to ``limit`` rows
      after that timestamp (but the delta is usually tiny so the cap is
      a safety net).

    Response shape::

        {
          "episodes": [ {...}, ... ],
          "latest_ts": "2026-04-17T13:48:03.142Z" | null,
          "truncated": true | false
        }

    ``latest_ts`` echoes the timestamp of the last returned row, so the
    client can pass it back as ``since_ts`` on the next poll.
    """
    config = request.app.state.config
    log_path = Path(config["paths"]["logs"]) / "training" / "episodes.jsonl"
    if not log_path.exists():
        return {"episodes": [], "latest_ts": None, "truncated": False}

    # Read backwards one line at a time when no since_ts, so we don't
    # parse the whole file on every request. With since_ts, do a forward
    # scan but short-circuit once we've crossed the cutoff.
    rows: list[dict] = []
    try:
        with log_path.open("r", encoding="utf-8") as f:
            all_lines = f.readlines()
    except OSError as exc:
        raise HTTPException(500, f"Failed to read episodes log: {exc}")

    # Timestamps in episodes.jsonl are Unix epoch floats. Accept either
    # a numeric string (preferred, what the frontend sends) or an ISO
    # timestamp (parsed on the spot). Invalid input is treated as "no
    # filter" rather than 400'ing so a stale cursor never hangs the UI.
    since_epoch: float | None = None
    if since_ts is not None:
        try:
            since_epoch = float(since_ts)
        except ValueError:
            try:
                from datetime import datetime
                iso = since_ts.replace("Z", "+00:00")
                since_epoch = datetime.fromisoformat(iso).timestamp()
            except (ValueError, TypeError):
                since_epoch = None

    if since_epoch is None:
        candidates = all_lines[-(limit * 2):] if limit > 0 else all_lines
    else:
        candidates = all_lines

    for line in candidates:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if since_epoch is not None:
            ts = row.get("timestamp")
            try:
                if ts is None or float(ts) <= since_epoch:
                    continue
            except (ValueError, TypeError):
                continue
        rows.append(row)

    truncated = len(rows) > limit
    if truncated:
        rows = rows[-limit:] if since_ts is None else rows[:limit]

    latest_ts = rows[-1].get("timestamp") if rows else None
    return {"episodes": rows, "latest_ts": latest_ts, "truncated": truncated}


@router.get("/training/info")
def get_training_info(request: Request):
    """Return data availability and estimated training duration info.

    Historical rates come from ``logs/training/last_run_timing.json``
    (written at the end of each completed run). Missing/corrupt file →
    fall back to the legacy 12s default so the wizard still works on
    a fresh install.
    """
    config = request.app.state.config
    data_dir = config["paths"]["processed_data"]
    processed = Path(data_dir)
    dates = sorted(
        f.stem
        for f in processed.glob("*.parquet")
        if not f.stem.endswith("_runners") and f.stem != ".gitkeep"
    )
    population_size = config.get("population", {}).get("size", 50)
    split = max(1, len(dates) // 2)
    store = request.app.state.store
    garage_count = len(store.list_garaged_models())

    # Historical rates from last completed run (if any).
    from training.run_training import (
        HISTORICAL_TIMING_PATH, DEFAULT_SECONDS_PER_AGENT_PER_DAY,
    )
    train_rate: float = DEFAULT_SECONDS_PER_AGENT_PER_DAY
    eval_rate: float = DEFAULT_SECONDS_PER_AGENT_PER_DAY
    based_on_last_run = False
    try:
        if HISTORICAL_TIMING_PATH.exists():
            raw = json.loads(HISTORICAL_TIMING_PATH.read_text())
            t = raw.get("train_seconds_per_agent_per_day")
            e = raw.get("eval_seconds_per_agent_per_day")
            if isinstance(t, (int, float)) and t > 0:
                train_rate = float(t)
            if isinstance(e, (int, float)) and e > 0:
                eval_rate = float(e)
            based_on_last_run = isinstance(t, (int, float)) or isinstance(e, (int, float))
    except Exception:
        # Corrupt file — fall back silently to defaults.
        based_on_last_run = False

    return {
        "available_days": len(dates),
        "train_days": split,
        "test_days": len(dates) - split,
        "population_size": population_size,
        "dates": dates,
        # Legacy field — kept for backward compatibility with older frontends.
        "seconds_per_agent_per_day": train_rate,
        "train_seconds_per_agent_per_day": train_rate,
        "eval_seconds_per_agent_per_day": eval_rate,
        "timing_based_on_last_run": based_on_last_run,
        "garage_count": garage_count,
        "reevaluate_garaged_default": config.get("training", {}).get(
            "reevaluate_garaged_default", True,
        ),
        "starting_budget_default": float(
            config.get("training", {}).get("starting_budget", 20.0)
        ),
    }


@router.get("/training/architectures", response_model=list[ArchitectureInfo])
def get_architectures(request: Request):
    """Return the list of available policy architectures with descriptions."""
    from agents.architecture_registry import REGISTRY

    # Which architectures are currently selected by default (from config hyperparameter choices)
    config = request.app.state.config
    default_choices = (
        config.get("hyperparameters", {})
        .get("search_ranges", {})
        .get("architecture_name", {})
        .get("choices", [])
    )
    result = []
    for name, cls in REGISTRY.items():
        result.append(ArchitectureInfo(
            name=name,
            description=cls.description,
        ))
    return result


@router.get("/training/genetics", response_model=GeneticsInfo)
def get_genetics(request: Request):
    """Return genetic algorithm configuration (read-only)."""
    config = request.app.state.config
    pop_cfg = config.get("population", {})
    return GeneticsInfo(
        population_size=pop_cfg.get("size", 50),
        n_elite=pop_cfg.get("n_elite", 5),
        selection_top_pct=pop_cfg.get("selection_top_pct", 0.5),
        mutation_rate=pop_cfg.get("mutation_rate", 0.3),
    )


@router.get(
    "/training/hyperparameter-schema",
    response_model=list[HyperparamSchemaEntry],
)
def get_hyperparameter_schema(request: Request):
    """Return the full hyperparameter search-range schema for the UI.

    The Session 8 UI renders one widget per entry in this list, dispatched
    by ``type``. Returning ``source_file`` lets the schema-inspector page
    point developers at the canonical definition without grepping.

    No hardcoded gene list — adding/removing a gene in ``config.yaml``
    automatically flows into the UI on the next reload.
    """
    config = request.app.state.config
    raw = (
        config.get("hyperparameters", {})
        .get("search_ranges", {})
    )
    entries: list[HyperparamSchemaEntry] = []
    for name, defn in raw.items():
        entries.append(
            HyperparamSchemaEntry(
                name=name,
                type=defn["type"],
                min=defn.get("min"),
                max=defn.get("max"),
                choices=defn.get("choices"),
                source_file=f"config.yaml#hyperparameters.search_ranges.{name}",
            )
        )
    return entries


@router.get("/training/architectures/defaults")
def get_architecture_defaults(request: Request):
    """Return the list of architectures currently used by default."""
    config = request.app.state.config
    choices = (
        config.get("hyperparameters", {})
        .get("search_ranges", {})
        .get("architecture_name", {})
        .get("choices", [])
    )
    return {"defaults": choices}


@router.post("/training/start", response_model=StartTrainingResponse)
async def start_training(request: Request, body: StartTrainingRequest):
    """Start a multi-generation training run via the training worker.

    Returns immediately with the run configuration.  Progress is streamed
    via the ``/ws/training`` WebSocket.
    """
    state = _state(request)
    if state["running"]:
        raise HTTPException(409, "A training run is already in progress")

    # Compute train/test split for the response (worker does same split)
    config = request.app.state.config
    data_dir = config["paths"]["processed_data"]
    processed = Path(data_dir)
    dates = sorted(
        f.stem
        for f in processed.glob("*.parquet")
        if not f.stem.endswith("_runners") and f.stem != ".gitkeep"
    )
    if not dates:
        raise HTTPException(400, "No extracted data available — import days first")

    # Default: chronological 50/50 split
    split = max(1, len(dates) // 2)
    train_dates = dates[:split]
    test_dates = dates[split:]

    # Override with user-supplied dates if provided
    if body.train_dates is not None:
        missing = [d for d in body.train_dates if d not in dates]
        if missing:
            raise HTTPException(400, f"Train dates not found in data: {missing}")
        train_dates = sorted(body.train_dates)
    if body.test_dates is not None:
        missing = [d for d in body.test_dates if d not in dates]
        if missing:
            raise HTTPException(400, f"Test dates not found in data: {missing}")
        test_dates = sorted(body.test_dates)

    # Validate architecture names if provided.  We validate against the
    # config choices (not the runtime REGISTRY) because architecture_registry
    # is only populated when agents.policy_network is imported, and that
    # pulls in torch — which we deliberately avoid in the API process.
    if body.architectures is not None:
        if not body.architectures:
            raise HTTPException(400, "At least one architecture must be selected")
        valid_choices = (
            config.get("hyperparameters", {})
            .get("search_ranges", {})
            .get("architecture_name", {})
            .get("choices", [])
        )
        unknown = [a for a in body.architectures if a not in valid_choices]
        if unknown:
            raise HTTPException(400, f"Unknown architectures: {unknown}")

    # Load training plan if plan_id provided
    plan_data = None
    if body.plan_id is not None:
        plan_registry = getattr(request.app.state, "plan_registry", None)
        if plan_registry is None:
            raise HTTPException(503, "Plan registry not configured")
        try:
            plan = plan_registry.load(body.plan_id)
        except KeyError:
            raise HTTPException(404, f"No such plan: {body.plan_id}")
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        plan_data = plan.to_dict()

    # Budget override validation
    if body.starting_budget is not None and body.starting_budget <= 0:
        raise HTTPException(400, "starting_budget must be positive")

    # Market type filter validation
    if body.market_type_filters is not None:
        if not body.market_type_filters:
            raise HTTPException(400, "At least one market type filter must be selected")
        valid_filters = {"WIN", "EACH_WAY", "BOTH", "FREE_CHOICE"}
        unknown = [f for f in body.market_type_filters if f not in valid_filters]
        if unknown:
            raise HTTPException(400, f"Unknown market type filters: {unknown}")

    # Mutation cap validation
    if body.max_mutations_per_child is not None and body.max_mutations_per_child < 1:
        raise HTTPException(400, "max_mutations_per_child must be >= 1 (or null)")

    # Breeding pool validation
    if body.breeding_pool is not None:
        valid_pools = {"run_only", "include_garaged", "full_registry"}
        if body.breeding_pool not in valid_pools:
            raise HTTPException(
                400,
                f"Unknown breeding_pool '{body.breeding_pool}'. "
                f"Must be one of: {sorted(valid_pools)}",
            )

    # Stud model validation (Issue 13)
    if body.stud_model_ids:
        if len(body.stud_model_ids) > 5:
            raise HTTPException(
                400, "stud_model_ids: at most 5 studs allowed",
            )
        store = getattr(request.app.state, "store", None)
        if store is None:
            raise HTTPException(503, "Model store not configured")
        missing: list[str] = []
        no_weights: list[str] = []
        no_hp: list[str] = []
        for sid in body.stud_model_ids:
            rec = store.get_model(sid)
            if rec is None:
                missing.append(sid)
                continue
            if not rec.weights_path:
                no_weights.append(sid)
            if not rec.hyperparameters:
                no_hp.append(sid)
        if missing:
            raise HTTPException(400, f"Stud model(s) not found: {missing}")
        if no_weights:
            raise HTTPException(
                400, f"Stud model(s) have no saved weights: {no_weights}",
            )
        if no_hp:
            raise HTTPException(
                400, f"Stud model(s) have no hyperparameters: {no_hp}",
            )

    # Adaptive breeding validation (Issue 09)
    if body.bad_generation_policy is not None:
        valid_policies = {"persist", "boost_mutation", "inject_top"}
        if body.bad_generation_policy not in valid_policies:
            raise HTTPException(
                400,
                f"Unknown bad_generation_policy '{body.bad_generation_policy}'. "
                f"Must be one of: {sorted(valid_policies)}",
            )
    if body.mutation_rate is not None and not (0.0 <= body.mutation_rate <= 1.0):
        raise HTTPException(400, "mutation_rate must be between 0.0 and 1.0")
    if body.adaptive_mutation_increment is not None and not (
        0.0 <= body.adaptive_mutation_increment <= 1.0
    ):
        raise HTTPException(
            400, "adaptive_mutation_increment must be between 0.0 and 1.0",
        )
    if body.adaptive_mutation_cap is not None and not (
        0.0 <= body.adaptive_mutation_cap <= 1.0
    ):
        raise HTTPException(
            400, "adaptive_mutation_cap must be between 0.0 and 1.0",
        )

    # Send start command to worker
    cmd = make_start_cmd(
        n_generations=body.n_generations,
        n_epochs=body.n_epochs,
        population_size=body.population_size,
        seed=body.seed,
        reevaluate_garaged=body.reevaluate_garaged,
        reevaluate_min_score=body.reevaluate_min_score,
        train_dates=train_dates if body.train_dates is not None else None,
        test_dates=test_dates if body.test_dates is not None else None,
        architectures=body.architectures,
        max_back_price=body.max_back_price,
        max_lay_price=body.max_lay_price,
        min_seconds_before_off=body.min_seconds_before_off,
        starting_budget=body.starting_budget,
        market_type_filters=body.market_type_filters,
        plan_id=body.plan_id,
        max_mutations_per_child=body.max_mutations_per_child,
        breeding_pool=body.breeding_pool,
        stud_model_ids=body.stud_model_ids,
        mutation_rate=body.mutation_rate,
        bad_generation_threshold=body.bad_generation_threshold,
        bad_generation_policy=body.bad_generation_policy,
        adaptive_mutation=body.adaptive_mutation,
        adaptive_mutation_increment=body.adaptive_mutation_increment,
        adaptive_mutation_cap=body.adaptive_mutation_cap,
        scalping_mode=body.scalping_mode,
        smoke_test_first=body.smoke_test_first,
    )
    # The probe runs inside the worker as a pre-flight to the full run;
    # the full population only starts if the probe passes. Give the
    # worker up to the probe-budget + some slack before the API gives
    # up on the synchronous acknowledgement.
    start_timeout = 1800.0 if body.smoke_test_first else 30.0
    resp = await _send_to_worker(request, cmd, timeout=start_timeout)

    if resp.get("type") == EVT_ERROR:
        # The worker reports a failed smoke test as an EVT_ERROR with a
        # structured ``smoke_test_result`` payload so the frontend can
        # open its failure modal without guessing. Legacy errors
        # (malformed config, worker busy) have no smoke_test_result and
        # surface as a plain string detail.
        smoke_result = resp.get("smoke_test_result")
        if smoke_result is not None:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": resp.get("message", "Smoke test failed"),
                    "smoke_test_result": smoke_result,
                },
            )
        raise HTTPException(409, resp.get("message", "Worker rejected start request"))

    run_id = resp.get("run_id", "unknown")

    return StartTrainingResponse(
        run_id=run_id,
        train_days=train_dates,
        test_days=test_dates,
        n_generations=body.n_generations,
        n_epochs=body.n_epochs,
        smoke_test_result=resp.get("smoke_test_result"),
    )


@router.post("/training/stop", response_model=StopTrainingResponse)
async def stop_training(request: Request, granularity: str = "immediate"):
    """Request a graceful stop of the current training run.

    The ``granularity`` query parameter controls how aggressively the run
    winds down:

    - ``immediate`` (default): halt after the current agent finishes.
    - ``eval_current``: finish the agent currently being evaluated, then stop.
    - ``eval_all``: skip remaining training, evaluate every model, then stop.
    """
    state = _state(request)
    if not state["running"]:
        raise HTTPException(409, "No training run is in progress")

    if granularity not in VALID_STOP_GRANULARITIES:
        raise HTTPException(
            422,
            f"Invalid granularity '{granularity}'. "
            f"Must be one of: {', '.join(sorted(VALID_STOP_GRANULARITIES))}",
        )

    # Send stop command to worker
    resp = await _send_to_worker(request, make_stop_cmd(granularity=granularity))

    if resp.get("type") == EVT_ERROR:
        raise HTTPException(409, resp.get("message", "Worker rejected stop request"))

    detail_map = {
        "immediate": "Stop requested — training will halt after the current agent completes",
        "eval_current": "Stop requested — finishing current evaluation then stopping",
        "eval_all": "Stop requested — skipping training, evaluating all models then stopping",
    }
    return StopTrainingResponse(detail=detail_map[granularity])


@router.post("/training/resume", response_model=ResumeTrainingResponse)
async def resume_training(request: Request, body: ResumeTrainingRequest):
    """Resume a paused or failed training plan from its next incomplete session.

    Loads the plan, determines which session to resume from, and sends a
    start command to the worker with the correct generation offset.
    """
    state = _state(request)
    if state["running"]:
        raise HTTPException(409, "A training run is already in progress")

    plan_registry = getattr(request.app.state, "plan_registry", None)
    if plan_registry is None:
        raise HTTPException(503, "Plan registry not configured")

    try:
        plan = plan_registry.load(body.plan_id)
    except KeyError:
        raise HTTPException(404, f"No such plan: {body.plan_id}")

    if plan.status not in ("paused", "failed"):
        raise HTTPException(
            400,
            f"Plan status is '{plan.status}' — only 'paused' or 'failed' plans can be resumed",
        )

    # Determine which session to resume from
    boundaries = plan.session_boundaries()
    session_idx = plan.current_session

    # If the plan crashed mid-session, replay that session from scratch
    if session_idx >= len(boundaries):
        raise HTTPException(400, "All sessions already completed")

    start_gen, end_gen = boundaries[session_idx]
    n_gens = end_gen - start_gen + 1

    # Compute train/test split (same logic as start_training)
    config = request.app.state.config
    data_dir = config["paths"]["processed_data"]
    processed = Path(data_dir)
    dates = sorted(
        f.stem
        for f in processed.glob("*.parquet")
        if not f.stem.endswith("_runners") and f.stem != ".gitkeep"
    )
    if not dates:
        raise HTTPException(400, "No extracted data available — import days first")

    split = max(1, len(dates) // 2)
    train_dates = dates[:split]
    test_dates = dates[split:]

    cmd = make_start_cmd(
        n_generations=n_gens,
        n_epochs=plan.n_epochs,
        population_size=plan.population_size,
        seed=plan.seed,
        architectures=plan.architectures,
        starting_budget=plan.starting_budget,
        plan_id=plan.plan_id,
        start_generation=start_gen,
        max_mutations_per_child=plan.max_mutations_per_child,
        breeding_pool=plan.breeding_pool,
        stud_model_ids=plan.stud_model_ids or None,
        mutation_rate=plan.mutation_rate,
        bad_generation_threshold=plan.bad_generation_threshold,
        bad_generation_policy=plan.bad_generation_policy,
        adaptive_mutation=plan.adaptive_mutation,
        adaptive_mutation_increment=plan.adaptive_mutation_increment,
        adaptive_mutation_cap=plan.adaptive_mutation_cap,
    )
    resp = await _send_to_worker(request, cmd, timeout=30.0)

    if resp.get("type") == EVT_ERROR:
        raise HTTPException(409, resp.get("message", "Worker rejected resume request"))

    return ResumeTrainingResponse(
        run_id=resp.get("run_id", "unknown"),
        session=session_idx + 1,
        start_generation=start_gen,
        n_generations=n_gens,
    )


@router.post("/training/finish", response_model=FinishTrainingResponse)
async def finish_training(request: Request):
    """Request early finish: evaluate current population, then complete normally.

    Unlike stop, this still runs evaluation and scoring on the current
    population so the run produces usable results.
    """
    state = _state(request)
    if not state["running"]:
        raise HTTPException(409, "No training run is in progress")

    resp = await _send_to_worker(request, make_finish_cmd())

    if resp.get("type") == EVT_ERROR:
        raise HTTPException(409, resp.get("message", "Worker rejected finish request"))

    return FinishTrainingResponse(
        detail="Finish requested — will evaluate current population then complete",
    )


@router.websocket("/ws/training")
async def ws_training(websocket: WebSocket):
    """Stream training progress events to a client.

    Events are consumed from the main progress queue by a background task
    in main.py and pushed into each client's per-connection mailbox.
    The handler drains the mailbox and forwards to the WebSocket, while
    also listening for client disconnect.

    Clients that connect mid-run receive the latest state immediately.
    """
    await websocket.accept()

    state = websocket.app.state.training_state

    # Send latest state on connect so mid-run clients get caught up
    if state.get("latest_event"):
        await websocket.send_text(json.dumps(state["latest_event"]))

    # Per-client mailbox — the queue consumer in main.py pushes here
    mailbox: asyncio.Queue[str] = asyncio.Queue()

    async def send_fn(msg: str) -> None:
        await mailbox.put(msg)

    websocket.app.state.ws_clients.add(send_fn)

    try:
        receive_task: asyncio.Task | None = None
        mailbox_task: asyncio.Task | None = None

        while True:
            # Start both tasks if not already running
            if receive_task is None:
                receive_task = asyncio.ensure_future(websocket.receive_text())
            if mailbox_task is None:
                mailbox_task = asyncio.ensure_future(mailbox.get())

            done, _ = await asyncio.wait(
                {receive_task, mailbox_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if mailbox_task in done:
                msg = mailbox_task.result()
                await websocket.send_text(msg)
                mailbox_task = None

            if receive_task in done:
                # Client sent something (or disconnected — triggers exception)
                receive_task.result()  # propagates WebSocketDisconnect
                receive_task = None
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        # Cancel any pending tasks
        for t in (receive_task, mailbox_task):
            if t and not t.done():
                t.cancel()
        websocket.app.state.ws_clients.discard(send_fn)
