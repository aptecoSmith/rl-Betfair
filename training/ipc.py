"""Shared IPC protocol constants and message helpers.

Used by both the training worker (server) and the API (client).
"""

from __future__ import annotations

import json

# ── Defaults ────────────────────────────────────────────────────────

DEFAULT_WORKER_HOST = "localhost"
DEFAULT_WORKER_PORT = 8002

# ── Message types ───────────────────────────────────────────────────

# Commands (API → Worker)
CMD_START = "start"
CMD_STOP = "stop"
CMD_STATUS = "status"

# Events (Worker → API)
EVT_EVENT = "event"        # wraps a training progress event
EVT_STATUS = "status"      # full state snapshot
EVT_STARTED = "started"    # training accepted
EVT_ERROR = "error"        # error message
EVT_PING = "ping"          # keepalive


# ── Message builders ────────────────────────────────────────────────

def make_start_cmd(
    n_generations: int = 3,
    n_epochs: int = 3,
    population_size: int | None = None,
    seed: int | None = None,
    reevaluate_garaged: bool = False,
    reevaluate_min_score: float | None = None,
) -> str:
    return json.dumps({
        "type": CMD_START,
        "n_generations": n_generations,
        "n_epochs": n_epochs,
        "population_size": population_size,
        "seed": seed,
        "reevaluate_garaged": reevaluate_garaged,
        "reevaluate_min_score": reevaluate_min_score,
    })


def make_stop_cmd() -> str:
    return json.dumps({"type": CMD_STOP})


def make_status_cmd() -> str:
    return json.dumps({"type": CMD_STATUS})


def make_event_msg(payload: dict) -> str:
    return json.dumps({"type": EVT_EVENT, "payload": payload})


def make_status_msg(
    running: bool,
    latest_event: dict | None,
    latest_process: dict | None,
    latest_item: dict | None,
) -> str:
    return json.dumps({
        "type": EVT_STATUS,
        "running": running,
        "latest_event": latest_event,
        "latest_process": latest_process,
        "latest_item": latest_item,
    })


def make_started_msg(run_id: str, train_days: list[str], test_days: list[str]) -> str:
    return json.dumps({
        "type": EVT_STARTED,
        "run_id": run_id,
        "train_days": train_days,
        "test_days": test_days,
    })


def make_error_msg(message: str) -> str:
    return json.dumps({"type": EVT_ERROR, "message": message})


def parse_message(raw: str) -> dict:
    return json.loads(raw)
