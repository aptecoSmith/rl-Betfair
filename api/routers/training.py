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
        )

    if not state["running"]:
        return TrainingStatus(running=False, worker_connected=worker_connected)

    latest = state.get("latest_event")
    if not latest:
        return TrainingStatus(running=True, worker_connected=worker_connected)

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
    # always contains both process and item, even when the latest
    # event only carried one of them.
    process = _snap(state.get("latest_process"))
    item = _snap(state.get("latest_item"))

    return TrainingStatus(
        running=True,
        phase=latest.get("phase"),
        generation=latest.get("generation"),
        process=process,
        item=item,
        detail=latest.get("detail"),
        last_agent_score=latest.get("last_agent_score"),
        worker_connected=worker_connected,
    )


@router.get("/training/info")
def get_training_info(request: Request):
    """Return data availability and estimated training duration info."""
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
    return {
        "available_days": len(dates),
        "train_days": split,
        "test_days": len(dates) - split,
        "population_size": population_size,
        "dates": dates,
        # Benchmark: ~12s per agent per day (from Session 4.6)
        "seconds_per_agent_per_day": 12.0,
        "garage_count": garage_count,
        "reevaluate_garaged_default": config.get("training", {}).get(
            "reevaluate_garaged_default", True,
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
    )
    resp = await _send_to_worker(request, cmd, timeout=30.0)

    if resp.get("type") == EVT_ERROR:
        raise HTTPException(409, resp.get("message", "Worker rejected start request"))

    run_id = resp.get("run_id", "unknown")

    return StartTrainingResponse(
        run_id=run_id,
        train_days=train_dates,
        test_days=test_dates,
        n_generations=body.n_generations,
        n_epochs=body.n_epochs,
    )


@router.post("/training/stop", response_model=StopTrainingResponse)
async def stop_training(request: Request):
    """Request a graceful stop of the current training run.

    The run will halt after the current agent completes its training/evaluation.
    """
    state = _state(request)
    if not state["running"]:
        raise HTTPException(409, "No training run is in progress")

    # Send stop command to worker
    resp = await _send_to_worker(request, make_stop_cmd())

    if resp.get("type") == EVT_ERROR:
        raise HTTPException(409, resp.get("message", "Worker rejected stop request"))

    return StopTrainingResponse(
        detail="Stop requested — training will halt after the current agent completes",
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
