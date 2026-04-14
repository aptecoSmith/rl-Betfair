"""Manual standalone evaluation endpoints.

Sends CMD_EVALUATE to the training worker.  Progress is streamed via the
existing ``/ws/training`` WebSocket — frontend reuses the training-monitor
progress components.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from api.routers.training import _send_to_worker
from api.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    EvaluateStatus,
    ProgressSnapshot,
)
from training.ipc import EVT_ERROR, make_evaluate_cmd

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluate", tags=["evaluation"])


def _state(request: Request) -> dict:
    return request.app.state.training_state


@router.post("", response_model=EvaluateResponse)
async def start_evaluation(request: Request, body: EvaluateRequest):
    """Kick off a standalone evaluation batch.

    Validates inputs upfront — missing models or unknown dates fail the
    request before the worker is contacted.  Returns 409 if the worker is
    already busy with a training run or a previous evaluation.
    """
    state = _state(request)
    if state["running"]:
        raise HTTPException(409, "Worker is busy — another job is already running")

    if not body.model_ids:
        raise HTTPException(400, "model_ids must contain at least one model")

    store = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(503, "Model store not configured")

    missing: list[str] = []
    no_weights: list[str] = []
    for mid in body.model_ids:
        rec = store.get_model(mid)
        if rec is None:
            missing.append(mid)
        elif not rec.weights_path:
            no_weights.append(mid)
    if missing:
        raise HTTPException(400, f"Model(s) not found: {missing}")
    if no_weights:
        raise HTTPException(400, f"Model(s) have no saved weights: {no_weights}")

    config = request.app.state.config
    data_dir = config["paths"]["processed_data"]
    processed = Path(data_dir)
    available_dates = sorted(
        f.stem
        for f in processed.glob("*.parquet")
        if not f.stem.endswith("_runners") and f.stem != ".gitkeep"
    )
    if not available_dates:
        raise HTTPException(400, "No extracted data available — import days first")

    if body.test_dates is None:
        test_dates = available_dates
    else:
        if not body.test_dates:
            raise HTTPException(400, "test_dates must contain at least one date (or be null for all)")
        unknown = [d for d in body.test_dates if d not in available_dates]
        if unknown:
            raise HTTPException(400, f"Test dates not found in data: {unknown}")
        test_dates = sorted(body.test_dates)

    cmd = make_evaluate_cmd(model_ids=list(body.model_ids), test_dates=test_dates)
    resp = await _send_to_worker(request, cmd, timeout=30.0)

    if resp.get("type") == EVT_ERROR:
        raise HTTPException(409, resp.get("message", "Worker rejected evaluate request"))

    return EvaluateResponse(
        accepted=True,
        job_id=resp.get("run_id", "unknown"),
        model_count=len(body.model_ids),
        day_count=len(test_dates),
    )


@router.get("/status", response_model=EvaluateStatus)
def evaluation_status(request: Request):
    """Return the current evaluation progress (or idle)."""
    state = _state(request)
    latest = state.get("latest_event") or {}

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

    summary = latest.get("summary") or {}
    is_manual = bool(summary.get("manual_evaluation"))

    return EvaluateStatus(
        running=bool(state.get("running")),
        phase=latest.get("phase"),
        detail=latest.get("detail"),
        process=_snap(state.get("latest_process")),
        item=_snap(state.get("latest_item")),
        manual_evaluation=is_manual,
    )
