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
CMD_FINISH = "finish"
CMD_STATUS = "status"
CMD_EVALUATE = "evaluate"

# Stop granularity levels (most → least aggressive)
STOP_IMMEDIATE = "immediate"
STOP_EVAL_CURRENT = "eval_current"
STOP_EVAL_ALL = "eval_all"

VALID_STOP_GRANULARITIES = {STOP_IMMEDIATE, STOP_EVAL_CURRENT, STOP_EVAL_ALL}

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
    train_dates: list[str] | None = None,
    test_dates: list[str] | None = None,
    architectures: list[str] | None = None,
    max_back_price: float | None = None,
    max_lay_price: float | None = None,
    min_seconds_before_off: int | None = None,
    starting_budget: float | None = None,
    market_type_filters: list[str] | None = None,
    plan_id: str | None = None,
    start_generation: int = 0,
    max_mutations_per_child: int | None = None,
    breeding_pool: str | None = None,
    stud_model_ids: list[str] | None = None,
    mutation_rate: float | None = None,
    bad_generation_threshold: float | None = None,
    bad_generation_policy: str | None = None,
    adaptive_mutation: bool | None = None,
    adaptive_mutation_increment: float | None = None,
    adaptive_mutation_cap: float | None = None,
    scalping_mode: bool | None = None,
    smoke_test_first: bool = False,
) -> str:
    return json.dumps({
        "type": CMD_START,
        "n_generations": n_generations,
        "n_epochs": n_epochs,
        "population_size": population_size,
        "seed": seed,
        "reevaluate_garaged": reevaluate_garaged,
        "reevaluate_min_score": reevaluate_min_score,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "architectures": architectures,
        "max_back_price": max_back_price,
        "max_lay_price": max_lay_price,
        "min_seconds_before_off": min_seconds_before_off,
        "starting_budget": starting_budget,
        "market_type_filters": market_type_filters,
        "plan_id": plan_id,
        "start_generation": start_generation,
        "max_mutations_per_child": max_mutations_per_child,
        "breeding_pool": breeding_pool,
        "stud_model_ids": stud_model_ids,
        "mutation_rate": mutation_rate,
        "bad_generation_threshold": bad_generation_threshold,
        "bad_generation_policy": bad_generation_policy,
        "adaptive_mutation": adaptive_mutation,
        "adaptive_mutation_increment": adaptive_mutation_increment,
        "adaptive_mutation_cap": adaptive_mutation_cap,
        "scalping_mode": scalping_mode,
        "smoke_test_first": bool(smoke_test_first),
    })


def make_stop_cmd(granularity: str = STOP_IMMEDIATE) -> str:
    return json.dumps({"type": CMD_STOP, "granularity": granularity})


def make_finish_cmd() -> str:
    return json.dumps({"type": CMD_FINISH})


def make_status_cmd() -> str:
    return json.dumps({"type": CMD_STATUS})


def make_event_msg(payload: dict) -> str:
    return json.dumps({"type": EVT_EVENT, "payload": payload})


def make_status_msg(
    running: bool,
    latest_event: dict | None,
    latest_process: dict | None,
    latest_item: dict | None,
    latest_overall: dict | None = None,
) -> str:
    return json.dumps({
        "type": EVT_STATUS,
        "running": running,
        "latest_event": latest_event,
        "latest_process": latest_process,
        "latest_item": latest_item,
        "latest_overall": latest_overall,
    })


def make_started_msg(
    run_id: str,
    train_days: list[str],
    test_days: list[str],
    plan_id: str | None = None,
    smoke_test_result: dict | None = None,
) -> str:
    payload = {
        "type": EVT_STARTED,
        "run_id": run_id,
        "train_days": train_days,
        "test_days": test_days,
        "plan_id": plan_id,
    }
    if smoke_test_result is not None:
        # Session 04 (naked-clip-and-stability). Surfaces the probe
        # outcome alongside the launch ack so the synchronous HTTP
        # reply can carry pass-case diagnostics. None when the probe
        # was skipped (checkbox OFF or legacy client).
        payload["smoke_test_result"] = smoke_test_result
    return json.dumps(payload)


def make_evaluate_cmd(
    model_ids: list[str],
    test_dates: list[str] | None = None,
) -> str:
    return json.dumps({
        "type": CMD_EVALUATE,
        "model_ids": model_ids,
        "test_dates": test_dates,
    })


def make_evaluate_started_msg(
    job_id: str,
    model_ids: list[str],
    test_dates: list[str],
) -> str:
    return json.dumps({
        "type": EVT_STARTED,
        "kind": "evaluate",
        "run_id": job_id,
        "model_ids": model_ids,
        "test_dates": test_dates,
        "train_days": [],
        "test_days": test_dates,
        "plan_id": None,
    })


def make_error_msg(message: str, smoke_test_result: dict | None = None) -> str:
    payload: dict = {"type": EVT_ERROR, "message": message}
    if smoke_test_result is not None:
        # Session 04 (naked-clip-and-stability). A failing smoke-test
        # probe is reported as an error so the launch's pending future
        # resolves on the API side, but the structured probe outcome
        # travels with it for the UI failure modal.
        payload["smoke_test_result"] = smoke_test_result
    return json.dumps(payload)


def parse_message(raw: str) -> dict:
    return json.loads(raw)
