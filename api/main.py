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
    }
    app.state.stop_event = threading.Event()
    app.state.training_task = None

    yield


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
