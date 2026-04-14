"""FastAPI application with CORS and lifespan that opens the registry DB."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure app-level logging so our logger.info/error calls reach uvicorn's output
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logging.getLogger("websockets").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

from registry.model_store import ModelStore
from registry.scoreboard import Scoreboard

from api.routers import models, training, replay, system, admin, training_plans, exploration
from training.training_plan import PlanRegistry
from training.ipc import (
    DEFAULT_WORKER_HOST,
    DEFAULT_WORKER_PORT,
    EVT_EVENT,
    EVT_STATUS,
    EVT_STARTED,
    EVT_ERROR,
    make_status_cmd,
    parse_message,
)


load_dotenv()


def _load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open registry DB on startup, close on shutdown."""
    config = _load_config()
    app.state.config_path = str(Path("config.yaml").resolve())
    db_path = config["paths"]["registry_db"]
    weights_dir = config["paths"]["model_weights"]
    bet_logs_dir = config["paths"].get("bet_logs") or str(Path(db_path).parent / "bet_logs")

    store = ModelStore(
        db_path=db_path,
        weights_dir=weights_dir,
        bet_logs_dir=bet_logs_dir,
    )
    scoreboard = Scoreboard(store=store, config=config)
    # Session-4 plan registry: same registry/ root as model_store, new
    # subfolder.  Path is overrideable via config so deployments can
    # relocate it without code changes.
    plan_root = (
        config.get("paths", {}).get("training_plans")
        or str(Path(db_path).parent / "training_plans")
    )
    plan_registry = PlanRegistry(plan_root)
    progress_queue: asyncio.Queue = asyncio.Queue()

    app.state.config = config
    app.state.store = store
    app.state.scoreboard = scoreboard
    app.state.plan_registry = plan_registry
    app.state.progress_queue = progress_queue
    app.state.training_state = {
        "running": False,
        "latest_event": None,
        "latest_process": None,
        "latest_item": None,
        "latest_overall": None,
        "plan_id": None,
    }
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
            if event.get("overall"):
                state["latest_overall"] = event["overall"]

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
                state["latest_overall"] = None
                state["plan_id"] = None

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

    # ── Training worker connection ──────────────────────────────────
    # Maintain a persistent WebSocket connection to the training worker.
    # Events received from the worker are pushed into progress_queue so
    # the existing _queue_consumer + frontend WS broadcast work unchanged.

    worker_cfg = config.get("training_worker", {})
    worker_host = worker_cfg.get("host", DEFAULT_WORKER_HOST)
    worker_port = worker_cfg.get("port", DEFAULT_WORKER_PORT)

    # Shared mutable reference so the training router can send commands
    app.state.worker_ws = None
    app.state.worker_connected = False

    # When the training router sends a command (start/stop), it sets a
    # Future here.  The connection loop resolves it with the next
    # non-event message (status/started/error) from the worker.
    app.state.worker_pending_response: asyncio.Future | None = None

    async def _worker_connection():
        import websockets

        url = f"ws://{worker_host}:{worker_port}"
        while True:
            try:
                async with websockets.connect(url) as ws:
                    app.state.worker_ws = ws
                    app.state.worker_connected = True
                    logger.info("Connected to training worker at %s", url)

                    # Request current state to sync up
                    await ws.send(make_status_cmd())

                    async for raw in ws:
                        msg = parse_message(raw)
                        msg_type = msg.get("type")

                        if msg_type == EVT_EVENT:
                            # Push the training event into our progress_queue
                            await progress_queue.put(msg["payload"])

                        elif msg_type == EVT_STATUS:
                            # Sync local training_state from worker
                            state = app.state.training_state
                            state["running"] = msg.get("running", False)
                            if msg.get("latest_event"):
                                state["latest_event"] = msg["latest_event"]
                            if msg.get("latest_process"):
                                state["latest_process"] = msg["latest_process"]
                            if msg.get("latest_item"):
                                state["latest_item"] = msg["latest_item"]
                            if msg.get("latest_overall"):
                                state["latest_overall"] = msg["latest_overall"]
                            # Resolve pending response if any
                            pending = app.state.worker_pending_response
                            if pending and not pending.done():
                                pending.set_result(msg)

                        elif msg_type == EVT_STARTED:
                            # Track active plan_id for the status endpoint
                            state = app.state.training_state
                            state["plan_id"] = msg.get("plan_id")
                            # Resolve pending response
                            pending = app.state.worker_pending_response
                            if pending and not pending.done():
                                pending.set_result(msg)

                        elif msg_type == EVT_ERROR:
                            logger.warning(
                                "Training worker error: %s", msg.get("message")
                            )
                            # Resolve pending response with the error
                            pending = app.state.worker_pending_response
                            if pending and not pending.done():
                                pending.set_result(msg)

                        # Ignore pings and unknown types

            except asyncio.CancelledError:
                break
            except Exception:
                app.state.worker_ws = None
                app.state.worker_connected = False
                # Cancel any pending response
                pending = app.state.worker_pending_response
                if pending and not pending.done():
                    pending.set_exception(
                        ConnectionError("Lost connection to training worker")
                    )
                # Retry after backoff
                await asyncio.sleep(3)

    worker_task = asyncio.create_task(_worker_connection())

    yield

    worker_task.cancel()
    consumer_task.cancel()
    for t in (worker_task, consumer_task):
        try:
            await t
        except asyncio.CancelledError:
            pass


def create_app() -> FastAPI:
    app = FastAPI(title="rl-betfair API", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:4202"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(models.router)
    app.include_router(training.router)
    app.include_router(replay.router)
    app.include_router(system.router)
    app.include_router(admin.router)
    app.include_router(training_plans.router)
    app.include_router(exploration.router)

    return app


app = create_app()
