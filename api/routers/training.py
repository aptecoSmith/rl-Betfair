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


@router.get("/training/status", response_model=TrainingStatus)
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

    async def _run_training():
        """Background coroutine that runs the orchestrator in a thread."""
        from data.episode_builder import load_days
        from training.run_training import TrainingOrchestrator

        state["running"] = True
        state["latest_event"] = None
        try:
            train_days = load_days(train_dates, data_dir=data_dir)
            test_days = load_days(test_dates, data_dir=data_dir)

            orch = TrainingOrchestrator(
                config=config,
                model_store=request.app.state.store,
                progress_queue=request.app.state.progress_queue,
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
    """Broadcast progress events from the orchestrator's asyncio.Queue.

    Clients that connect mid-run receive the latest state immediately.
    """
    await websocket.accept()

    state = websocket.app.state.training_state
    queue: asyncio.Queue = websocket.app.state.progress_queue

    # Send latest state on connect so mid-run clients get caught up
    if state.get("latest_event"):
        await websocket.send_text(json.dumps(state["latest_event"]))

    try:
        while True:
            # Wait for next event from the orchestrator queue
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_text(json.dumps({"event": "ping"}))
                continue

            # Update latest state for future mid-run connections
            state["latest_event"] = event
            if event.get("event") == "phase_start":
                state["running"] = True
            elif (
                event.get("event") == "run_complete"
                or (
                    event.get("event") == "phase_complete"
                    and event.get("phase") in ("run_complete", "run_stopped", "run_error")
                )
            ):
                state["running"] = False

            await websocket.send_text(json.dumps(event))

    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
