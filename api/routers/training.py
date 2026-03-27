"""Training status endpoint and WebSocket for live progress events."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from api.schemas import TrainingStatus, ProgressSnapshot

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
            elif event.get("event") == "run_complete":
                state["running"] = False

            await websocket.send_text(json.dumps(event))

    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
