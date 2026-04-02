"""Training status, start/stop endpoints, and WebSocket for live progress."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect

from api.schemas import (
    ProgressSnapshot,
    StartTrainingRequest,
    StartTrainingResponse,
    StopTrainingResponse,
    TrainingStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["training"])


def _state(request: Request) -> dict:
    return request.app.state.training_state


@router.get("/training/status")
def get_training_status(request: Request):
    """Current run snapshot: phase, process ETA, item ETA, generation."""
    state = _state(request)
    if not state["running"]:
        return TrainingStatus(running=False)

    latest = state.get("latest_event")
    if not latest:
        return TrainingStatus(running=True)

    process = None
    if latest.get("process"):
        p = latest["process"]
        process = ProgressSnapshot(
            label=p.get("label", ""),
            completed=p.get("completed", 0),
            total=p.get("total", 0),
            pct=p.get("pct", 0.0),
            item_eta_human=p.get("item_eta_human", ""),
            process_eta_human=p.get("process_eta_human", ""),
        )

    item = None
    if latest.get("item"):
        i = latest["item"]
        item = ProgressSnapshot(
            label=i.get("label", ""),
            completed=i.get("completed", 0),
            total=i.get("total", 0),
            pct=i.get("pct", 0.0),
            item_eta_human=i.get("item_eta_human", ""),
            process_eta_human=i.get("process_eta_human", ""),
        )

    return TrainingStatus(
        running=True,
        phase=latest.get("phase"),
        generation=latest.get("generation"),
        process=process,
        item=item,
        detail=latest.get("detail"),
        last_agent_score=latest.get("last_agent_score"),
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
    return {
        "available_days": len(dates),
        "train_days": split,
        "test_days": len(dates) - split,
        "population_size": population_size,
        "dates": dates,
        # Benchmark: ~12s per agent per day (from Session 4.6)
        "seconds_per_agent_per_day": 12.0,
    }


@router.post("/training/start", response_model=StartTrainingResponse)
async def start_training(request: Request, body: StartTrainingRequest):
    """Start a multi-generation training run in the background.

    Returns immediately with the run configuration. Progress is streamed
    via the ``/ws/training`` WebSocket.
    """
    state = _state(request)
    if state["running"]:
        raise HTTPException(409, "A training run is already in progress")

    config = request.app.state.config
    data_dir = config["paths"]["processed_data"]

    # Find available dates from Parquet files
    processed = Path(data_dir)
    dates = sorted(
        f.stem
        for f in processed.glob("*.parquet")
        if not f.stem.endswith("_runners") and f.stem != ".gitkeep"
    )
    if not dates:
        raise HTTPException(400, "No extracted data available — import days first")

    # Chronological train/test split (~50/50)
    split = max(1, len(dates) // 2)
    train_dates = dates[:split]
    test_dates = dates[split:]

    run_id = str(uuid.uuid4())

    # Reset stop event for this run
    request.app.state.stop_event.clear()

    # Apply population size override from request
    import copy
    run_config = copy.deepcopy(config)
    if body.population_size is not None:
        run_config["population"]["size"] = body.population_size
        # Scale n_elite proportionally (at least 1)
        run_config["population"]["n_elite"] = max(1, body.population_size // 10)

    async def _run_training():
        """Background coroutine that runs the orchestrator in a thread."""
        from data.episode_builder import load_days
        from training.run_training import TrainingOrchestrator

        state["running"] = True
        state["latest_event"] = None

        # The orchestrator runs in a thread but asyncio.Queue is NOT
        # thread-safe. Use a thread-safe queue and a bridge task.
        import queue as thread_queue
        thread_q: thread_queue.Queue = thread_queue.Queue()
        async_q: asyncio.Queue = request.app.state.progress_queue
        loop = asyncio.get_event_loop()

        # Bridge: move events from thread-safe queue to asyncio queue
        bridge_running = True

        async def _bridge():
            while bridge_running or not thread_q.empty():
                try:
                    event = await asyncio.to_thread(thread_q.get, timeout=0.5)
                    await async_q.put(event)
                except Exception:
                    await asyncio.sleep(0.2)

        bridge_task = asyncio.create_task(_bridge())

        try:
            train_days = load_days(train_dates, data_dir=data_dir)
            test_days = load_days(test_dates, data_dir=data_dir)

            orch = TrainingOrchestrator(
                config=run_config,
                model_store=request.app.state.store,
                progress_queue=thread_q,
                stop_event=request.app.state.stop_event,
            )

            # Run in thread — the orchestrator is blocking/CPU-bound
            await asyncio.to_thread(
                orch.run,
                train_days=train_days,
                test_days=test_days,
                n_generations=body.n_generations,
                n_epochs=body.n_epochs,
                seed=body.seed,
            )
        except Exception:
            logger.exception("Training run failed")
            try:
                request.app.state.progress_queue.put_nowait({
                    "event": "phase_complete",
                    "phase": "run_error",
                    "summary": {"error": "Training run failed"},
                })
            except asyncio.QueueFull:
                pass
        finally:
            bridge_running = False
            await bridge_task
            state["running"] = False
            request.app.state.training_task = None

    task = asyncio.create_task(_run_training())
    request.app.state.training_task = task

    return StartTrainingResponse(
        run_id=run_id,
        train_days=train_dates,
        test_days=test_dates,
        n_generations=body.n_generations,
        n_epochs=body.n_epochs,
    )


@router.post("/training/stop", response_model=StopTrainingResponse)
def stop_training(request: Request):
    """Request a graceful stop of the current training run.

    The run will halt after the current agent completes its training/evaluation.
    """
    state = _state(request)
    if not state["running"]:
        raise HTTPException(409, "No training run is in progress")

    request.app.state.stop_event.set()
    return StopTrainingResponse(
        detail="Stop requested — training will halt after the current agent completes",
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
