"""Admin endpoints for data management, agent deletion, import, and reset."""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from api.schemas import (
    AdminAgentEntry,
    AdminAgentsResponse,
    AdminDeleteResponse,
    BackupDay,
    BackupDaysResponse,
    ExtractedDay,
    ExtractedDaysResponse,
    ImportDayRequest,
    ImportDayResponse,
    ImportRangeRequest,
    ImportRangeResponse,
    ResetRequest,
    ResetResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# ── Helpers ──────────────────────────────────────────────────────────


def _processed_dir(request: Request) -> Path:
    return Path(request.app.state.config["paths"]["processed_data"])


def _backup_dir(request: Request) -> Path:
    return Path(request.app.state.config["paths"]["backup_data"])


def _get_extracted_dates(processed: Path) -> list[str]:
    """Return sorted list of YYYY-MM-DD dates that have a ticks Parquet file."""
    if not processed.exists():
        return []
    dates = []
    for f in sorted(processed.glob("*.parquet")):
        if f.stem.endswith("_runners"):
            continue
        dates.append(f.stem)
    return dates


def _day_metadata(processed: Path, date_str: str) -> ExtractedDay:
    """Build metadata for one extracted day."""
    ticks_path = processed / f"{date_str}.parquet"
    file_size = ticks_path.stat().st_size if ticks_path.exists() else 0

    tick_count = 0
    race_count = 0
    if ticks_path.exists():
        try:
            df = pd.read_parquet(ticks_path, columns=["market_id"])
            tick_count = len(df)
            race_count = df["market_id"].nunique()
        except Exception:
            pass

    return ExtractedDay(
        date=date_str,
        tick_count=tick_count,
        race_count=race_count,
        file_size_bytes=file_size,
    )


# ── GET endpoints ────────────────────────────────────────────────────


@router.get("/days", response_model=ExtractedDaysResponse)
async def list_extracted_days(request: Request):
    """List all extracted days with metadata."""
    processed = _processed_dir(request)
    dates = _get_extracted_dates(processed)
    days = [_day_metadata(processed, d) for d in dates]
    return ExtractedDaysResponse(days=days)


@router.get("/backup-days", response_model=BackupDaysResponse)
async def list_backup_days(request: Request):
    """List dates available in backup folder not yet in data/processed/."""
    processed = _processed_dir(request)
    backup = _backup_dir(request)

    existing = set(_get_extracted_dates(processed))

    backup_dates: list[BackupDay] = []
    if backup.exists():
        for f in sorted(backup.glob("*.parquet")):
            if f.stem.endswith("_runners"):
                continue
            if f.stem not in existing:
                backup_dates.append(BackupDay(date=f.stem))

    return BackupDaysResponse(days=backup_dates)


@router.get("/agents", response_model=AdminAgentsResponse)
async def list_agents(request: Request):
    """List all models (active + discarded) for admin management."""
    store = request.app.state.store
    models = store.list_models()
    agents = [
        AdminAgentEntry(
            model_id=m.model_id,
            generation=m.generation,
            architecture_name=m.architecture_name,
            status=m.status,
            composite_score=m.composite_score,
            created_at=m.created_at,
        )
        for m in models
    ]
    return AdminAgentsResponse(agents=agents)


# ── DELETE endpoints ─────────────────────────────────────────────────


@router.delete("/days/{date_str}", response_model=AdminDeleteResponse)
async def delete_day(date_str: str, request: Request, confirm: bool = False):
    """Delete a day's extracted Parquet files and evaluation records referencing that date.

    Requires ``confirm=true`` query parameter to prevent accidental deletion.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required: pass ?confirm=true",
        )
    # Validate date format
    try:
        date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {date_str}")

    processed = _processed_dir(request)
    ticks_path = processed / f"{date_str}.parquet"
    runners_path = processed / f"{date_str}_runners.parquet"

    if not ticks_path.exists() and not runners_path.exists():
        raise HTTPException(status_code=404, detail=f"No extracted data for {date_str}")

    # Delete Parquet files
    deleted_files = []
    for p in [ticks_path, runners_path]:
        if p.exists():
            p.unlink()
            deleted_files.append(p.name)

    # Delete evaluation records referencing this date
    store = request.app.state.store
    deleted_eval_days = _delete_evaluation_days_for_date(store, date_str)

    detail = (
        f"Deleted {', '.join(deleted_files)}"
        f" and {deleted_eval_days} evaluation day record(s) for {date_str}"
    )
    logger.info(detail)
    return AdminDeleteResponse(deleted=True, detail=detail)


def _delete_evaluation_days_for_date(store, date_str: str) -> int:
    """Delete all evaluation_days rows for a given date. Returns count deleted."""
    conn = store._get_conn()
    try:
        cursor = conn.execute(
            "DELETE FROM evaluation_days WHERE date = ?", (date_str,)
        )
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


@router.delete("/agents/{model_id}", response_model=AdminDeleteResponse)
async def delete_agent(model_id: str, request: Request, confirm: bool = False):
    """Delete a model: weights, registry record, evaluation data, genetic events.

    Requires ``confirm=true`` query parameter to prevent accidental deletion.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required: pass ?confirm=true",
        )
    store = request.app.state.store

    model = store.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    # 1. Delete weights file
    if model.weights_path:
        wp = Path(model.weights_path)
        if wp.exists():
            wp.unlink()

    # 2. Delete bet log Parquets for all evaluation runs of this model
    eval_runs = _get_all_evaluation_runs(store, model_id)
    for run in eval_runs:
        run_dir = store.bet_logs_dir / run["run_id"]
        if run_dir.exists():
            shutil.rmtree(run_dir)

    # 3. Delete evaluation_days, evaluation_runs from DB
    conn = store._get_conn()
    try:
        for run in eval_runs:
            conn.execute("DELETE FROM evaluation_days WHERE run_id = ?", (run["run_id"],))
        conn.execute("DELETE FROM evaluation_runs WHERE model_id = ?", (model_id,))

        # 4. Delete genetic events where this model is the child
        conn.execute("DELETE FROM genetic_events WHERE child_model_id = ?", (model_id,))

        # 5. Delete the model record itself
        conn.execute("DELETE FROM models WHERE model_id = ?", (model_id,))

        conn.commit()
    finally:
        conn.close()

    detail = (
        f"Deleted model {model_id}: weights, {len(eval_runs)} evaluation run(s), "
        f"genetic events, registry record"
    )
    logger.info(detail)
    return AdminDeleteResponse(deleted=True, detail=detail)


def _get_all_evaluation_runs(store, model_id: str) -> list[dict]:
    """Get all evaluation runs for a model (not just latest)."""
    conn = store._get_conn()
    try:
        rows = conn.execute(
            "SELECT run_id, model_id FROM evaluation_runs WHERE model_id = ?",
            (model_id,),
        ).fetchall()
        return [{"run_id": r["run_id"], "model_id": r["model_id"]} for r in rows]
    finally:
        conn.close()


# ── POST endpoints ───────────────────────────────────────────────────


@router.post("/import-day", response_model=ImportDayResponse)
async def import_day(body: ImportDayRequest, request: Request):
    """Import a single day from MySQL. Runs extractor for that date."""
    try:
        target = date.fromisoformat(body.date)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {body.date}")

    config = request.app.state.config

    try:
        from data.extractor import DataExtractor

        extractor = DataExtractor(config)
        ok = extractor.extract_date(target)
    except Exception as e:
        logger.exception("Import failed for %s", body.date)
        return ImportDayResponse(
            success=False, date=body.date, detail=f"Extraction failed: {e}"
        )

    if ok:
        return ImportDayResponse(
            success=True, date=body.date, detail=f"Successfully extracted {body.date}"
        )
    return ImportDayResponse(
        success=False, date=body.date, detail=f"No tick data found for {body.date}"
    )


@router.post("/import-range", response_model=ImportRangeResponse)
async def import_range(body: ImportRangeRequest, request: Request):
    """Import a date range. Returns immediately with job ID; progress via WebSocket."""
    try:
        start = date.fromisoformat(body.start_date)
        end = date.fromisoformat(body.end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")

    if start > end:
        raise HTTPException(
            status_code=400, detail="start_date must be <= end_date"
        )

    config = request.app.state.config
    processed = _processed_dir(request)
    progress_queue = request.app.state.progress_queue
    training_state = request.app.state.training_state

    # Build list of dates to import
    all_dates = []
    d = start
    from datetime import timedelta

    while d <= end:
        date_str = d.isoformat()
        if body.force or not (processed / f"{date_str}.parquet").exists():
            all_dates.append(d)
        d += timedelta(days=1)

    if not all_dates:
        return ImportRangeResponse(
            job_id="", dates_queued=0, detail="All dates already extracted"
        )

    job_id = str(uuid.uuid4())

    async def _run_import():
        import time

        training_state["running"] = True
        try:
            from data.extractor import DataExtractor

            extractor = DataExtractor(config)
            total = len(all_dates)

            await progress_queue.put({
                "event": "phase_start",
                "timestamp": time.time(),
                "phase": "extracting",
                "summary": {"job_id": job_id, "total_dates": total},
            })

            for i, target_date in enumerate(all_dates):
                try:
                    extractor.extract_date(target_date)
                except Exception:
                    logger.exception("Failed to extract %s", target_date)

                await progress_queue.put({
                    "event": "progress",
                    "timestamp": time.time(),
                    "phase": "extracting",
                    "process": {
                        "label": "Importing days",
                        "completed": i + 1,
                        "total": total,
                        "pct": round((i + 1) / total * 100, 1),
                        "item_eta_human": "",
                        "process_eta_human": "",
                    },
                    "detail": f"Extracted {target_date.isoformat()}",
                })

            await progress_queue.put({
                "event": "phase_complete",
                "timestamp": time.time(),
                "phase": "extracting",
                "summary": {"job_id": job_id, "dates_imported": total},
            })
        finally:
            training_state["running"] = False

    asyncio.create_task(_run_import())

    return ImportRangeResponse(
        job_id=job_id,
        dates_queued=len(all_dates),
        detail=f"Queued {len(all_dates)} date(s) for import",
    )


@router.post("/reset", response_model=ResetResponse)
async def reset_system(body: ResetRequest, request: Request):
    """Delete all models, scoreboard, genetic events, evaluation data. Preserves Parquet."""
    if body.confirm != "DELETE_EVERYTHING":
        raise HTTPException(
            status_code=400,
            detail="Confirmation required: body must contain {\"confirm\": \"DELETE_EVERYTHING\"}",
        )

    store = request.app.state.store

    # Delete all weight files
    weights_deleted = 0
    if store.weights_dir.exists():
        for f in store.weights_dir.glob("*.pt"):
            f.unlink()
            weights_deleted += 1

    # Delete all bet log Parquets
    bet_dirs_deleted = 0
    if store.bet_logs_dir.exists():
        for d in store.bet_logs_dir.iterdir():
            if d.is_dir():
                shutil.rmtree(d)
                bet_dirs_deleted += 1

    # Clear all DB tables (order matters for foreign keys)
    conn = store._get_conn()
    try:
        conn.execute("DELETE FROM evaluation_days")
        conn.execute("DELETE FROM evaluation_runs")
        conn.execute("DELETE FROM genetic_events")
        conn.execute("DELETE FROM models")
        conn.commit()
    finally:
        conn.close()

    detail = (
        f"Reset complete: deleted {weights_deleted} weight file(s), "
        f"{bet_dirs_deleted} bet log dir(s), cleared all DB tables"
    )
    logger.info(detail)
    return ResetResponse(reset=True, detail=detail)
