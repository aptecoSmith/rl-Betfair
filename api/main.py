"""FastAPI application with CORS and lifespan that opens the registry DB."""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from registry.model_store import ModelStore
from registry.scoreboard import Scoreboard

from api.routers import models, training, replay, system, admin


def _load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open registry DB on startup, close on shutdown."""
    config = _load_config()
    db_path = config["paths"]["registry_db"]
    weights_dir = config["paths"]["model_weights"]
    bet_logs_dir = str(Path(db_path).parent / "bet_logs")

    store = ModelStore(
        db_path=db_path,
        weights_dir=weights_dir,
        bet_logs_dir=bet_logs_dir,
    )
    scoreboard = Scoreboard(store=store, config=config)
    progress_queue: asyncio.Queue = asyncio.Queue()

    app.state.config = config
    app.state.store = store
    app.state.scoreboard = scoreboard
    app.state.progress_queue = progress_queue
    app.state.training_state = {
        "running": False,
        "latest_event": None,
        "latest_process": None,
        "latest_item": None,
    }
    app.state.stop_event = threading.Event()
    app.state.training_task = None
    # Set of connected WebSocket send callbacks for broadcasting
    app.state.ws_clients: set = set()

    # Background task: drain progress_queue into training_state and broadcast
    async def _queue_consumer():
        state = app.state.training_state
        queue = app.state.progress_queue
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send keepalive to all WebSocket clients
                import json
                ping = json.dumps({"event": "ping"})
                dead = set()
                for send_fn in app.state.ws_clients:
                    try:
                        await send_fn(ping)
                    except Exception:
                        dead.add(send_fn)
                app.state.ws_clients -= dead
                continue
            except asyncio.CancelledError:
                break

            # Update training_state for the status endpoint
            state["latest_event"] = event

            # Track process and item snapshots independently so the
            # status endpoint can return both even when only one is
            # present in the current event.
            if event.get("process"):
                state["latest_process"] = event["process"]
            if event.get("item"):
                state["latest_item"] = event["item"]

            if event.get("event") == "phase_start":
                state["running"] = True
                # New phase — clear stale item progress
                state["latest_item"] = None
            elif (
                event.get("event") == "run_complete"
                or (
                    event.get("event") == "phase_complete"
                    and event.get("phase") in (
                        "run_complete", "run_stopped", "run_error",
                        "extracting",
                    )
                )
            ):
                state["running"] = False
                state["latest_process"] = None
                state["latest_item"] = None

            # Broadcast to all connected WebSocket clients
            import json
            msg = json.dumps(event)
            dead = set()
            for send_fn in app.state.ws_clients:
                try:
                    await send_fn(msg)
                except Exception:
                    dead.add(send_fn)
            app.state.ws_clients -= dead

    consumer_task = asyncio.create_task(_queue_consumer())

    yield

    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass


def create_app() -> FastAPI:
    app = FastAPI(title="rl-betfair API", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:4200"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(models.router)
    app.include_router(training.router)
    app.include_router(replay.router)
    app.include_router(system.router)
    app.include_router(admin.router)

    return app


app = create_app()
