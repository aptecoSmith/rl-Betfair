"""Admin endpoints for data management, agent deletion, import, and reset."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
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
    BettingConstraints,
    ExtractedDay,
    ExtractedDaysResponse,
    ImportDayRequest,
    ImportDayResponse,
    ImportRangeRequest,
    ImportRangeResponse,
    LogPathsResponse,
    LogSubdir,
    MysqlDatesResponse,
    ResetRequest,
    ResetResponse,
    RestoreRequest,
    RestoreResponse,
    StreamrecorderBackup,
    StreamrecorderBackupsResponse,
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


@router.get("/mysql-dates", response_model=MysqlDatesResponse)
async def list_mysql_dates(request: Request):
    """List dates available in MySQL databases (cold + hot).

    Returns available=false gracefully if MySQL is unreachable.
    """
    try:
        from data.extractor import DataExtractor

        extractor = DataExtractor(request.app.state.config)
        dates = extractor.get_available_dates()
        return MysqlDatesResponse(
            dates=[d.isoformat() for d in sorted(dates)],
            available=True,
        )
    except Exception as exc:
        logger.warning("Could not query MySQL dates: %s", exc)
        return MysqlDatesResponse(dates=[], available=False)


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
            garaged=m.garaged,
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

    if not store.get_model(model_id):
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    store.delete_model(model_id)

    detail = f"Deleted model {model_id}: weights, evaluation data, genetic events, registry record"
    logger.info(detail)
    return AdminDeleteResponse(deleted=True, detail=detail)


@router.post("/purge-discarded", response_model=AdminDeleteResponse)
async def purge_discarded(request: Request):
    """Delete all discarded, non-garaged models and their artefacts."""
    store = request.app.state.store
    purged = store.purge_discarded()
    detail = f"Purged {len(purged)} discarded model(s)" if purged else "No discarded models to purge"
    logger.info(detail)
    return AdminDeleteResponse(deleted=bool(purged), detail=detail)


@router.post("/purge-incompatible", response_model=AdminDeleteResponse)
async def purge_incompatible(request: Request, dry_run: bool = False):
    """Delete all active models whose obs_schema_version != current.

    Pass ``?dry_run=true`` to see which models *would* be deleted without
    actually removing them.
    """
    from env.betfair_env import OBS_SCHEMA_VERSION

    store = request.app.state.store
    ids = store.purge_incompatible(OBS_SCHEMA_VERSION, dry_run=dry_run)
    action = "Would purge" if dry_run else "Purged"
    detail = (
        f"{action} {len(ids)} incompatible model(s) (obs_schema != {OBS_SCHEMA_VERSION})"
        if ids
        else f"No incompatible models found (all match obs_schema {OBS_SCHEMA_VERSION})"
    )
    logger.info(detail)
    return AdminDeleteResponse(deleted=bool(ids), detail=detail)


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
        logger.info("[import] Job %s started: %d date(s)", job_id, len(all_dates))
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
                    ok = extractor.extract_date(target_date)
                    if ok:
                        logger.info("[import] Extracted %s", target_date)
                    else:
                        logger.warning("[import] No tick data for %s", target_date)
                except Exception:
                    logger.exception("[import] Failed to extract %s", target_date)

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
    skip_garaged = not body.clear_garage

    # Build set of garaged model IDs and their evaluation run IDs to preserve
    garaged_ids: set[str] = set()
    garaged_run_ids: set[str] = set()
    if skip_garaged:
        for m in store.list_garaged_models():
            garaged_ids.add(m.model_id)
        if garaged_ids:
            conn_tmp = store._get_conn()
            try:
                placeholders = ",".join("?" for _ in garaged_ids)
                rows = conn_tmp.execute(
                    f"SELECT run_id FROM evaluation_runs WHERE model_id IN ({placeholders})",
                    list(garaged_ids),
                ).fetchall()
                garaged_run_ids = {r["run_id"] for r in rows}
            finally:
                conn_tmp.close()

    # Delete weight files (skip garaged)
    weights_deleted = 0
    if store.weights_dir.exists():
        for f in store.weights_dir.glob("*.pt"):
            if skip_garaged and f.stem in garaged_ids:
                continue
            f.unlink()
            weights_deleted += 1

    # Delete bet log Parquets (skip garaged run dirs)
    bet_dirs_deleted = 0
    if store.bet_logs_dir.exists():
        for d in store.bet_logs_dir.iterdir():
            if d.is_dir():
                if skip_garaged and d.name in garaged_run_ids:
                    continue
                shutil.rmtree(d)
                bet_dirs_deleted += 1

    # Clear DB tables (order matters for foreign keys; skip garaged rows)
    conn = store._get_conn()
    try:
        if garaged_ids and skip_garaged:
            placeholders = ",".join("?" for _ in garaged_ids)
            ids = list(garaged_ids)
            conn.execute(
                f"DELETE FROM evaluation_days WHERE run_id IN "
                f"(SELECT run_id FROM evaluation_runs WHERE model_id NOT IN ({placeholders}))",
                ids,
            )
            conn.execute(
                f"DELETE FROM evaluation_runs WHERE model_id NOT IN ({placeholders})",
                ids,
            )
            conn.execute(
                f"DELETE FROM genetic_events WHERE child_model_id IS NULL "
                f"OR child_model_id NOT IN ({placeholders})",
                ids,
            )
            conn.execute(
                f"DELETE FROM models WHERE model_id NOT IN ({placeholders})",
                ids,
            )
        else:
            conn.execute("DELETE FROM evaluation_days")
            conn.execute("DELETE FROM evaluation_runs")
            conn.execute("DELETE FROM genetic_events")
            conn.execute("DELETE FROM models")
        conn.commit()
    finally:
        conn.close()

    garage_note = f" (preserved {len(garaged_ids)} garaged model(s))" if garaged_ids and skip_garaged else ""
    detail = (
        f"Reset complete: deleted {weights_deleted} weight file(s), "
        f"{bet_dirs_deleted} bet log dir(s), cleared DB tables{garage_note}"
    )
    logger.info(detail)
    return ResetResponse(reset=True, detail=detail)


# ── Streamrecorder restore ──────────────────────────────────────────


def _streamrecorder_dir(request: Request) -> Path:
    return Path(request.app.state.config["paths"]["streamrecorder_backups"])


def _mysql_bin(request: Request) -> str:
    raw = request.app.state.config["paths"]["mysql_bin"]
    # Convert MSYS/Git-Bash paths (/c/...) to Windows paths (C:\...)
    if re.match(r"^/[a-zA-Z]/", raw):
        raw = raw[1].upper() + ":" + raw[2:]
    resolved = str(Path(raw))
    logger.debug("[config] mysql_bin resolved: %s → %s", request.app.state.config["paths"]["mysql_bin"], resolved)
    return resolved


# Pattern: coldData-2026-04-02_223000.sql.gz  or  hotData-2026-04-02_223000.sql.gz
_BACKUP_RE = re.compile(
    r"^(coldData|hotData)-(\d{4}-\d{2}-\d{2})(?:_(\d{6}))?\.sql\.gz$"
)


def _scan_backups(backup_dir: Path, extracted: set[str]) -> list[StreamrecorderBackup]:
    """Scan StreamRecorder backup folder and return latest cold+hot pair per date."""
    if not backup_dir.exists():
        return []

    # Collect all .sql.gz files grouped by (kind, date) → list of (timestamp, filename)
    by_date: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for f in backup_dir.iterdir():
        m = _BACKUP_RE.match(f.name)
        if not m:
            continue
        kind, date_str, ts = m.group(1), m.group(2), m.group(3) or "000000"
        by_date.setdefault(date_str, {}).setdefault(kind, []).append((ts, f.name))

    results: list[StreamrecorderBackup] = []
    for date_str in sorted(by_date.keys()):
        groups = by_date[date_str]
        cold_list = groups.get("coldData", [])
        hot_list = groups.get("hotData", [])
        if not cold_list or not hot_list:
            continue  # need both cold + hot

        # Pick latest timestamp for each
        cold_list.sort(key=lambda x: x[0], reverse=True)
        hot_list.sort(key=lambda x: x[0], reverse=True)

        cold_ts, cold_file = cold_list[0]
        hot_ts, hot_file = hot_list[0]

        cold_path = backup_dir / cold_file
        hot_path = backup_dir / hot_file

        results.append(StreamrecorderBackup(
            date=date_str,
            timestamp=max(cold_ts, hot_ts),
            cold_file=cold_file,
            hot_file=hot_file,
            cold_size_bytes=cold_path.stat().st_size,
            hot_size_bytes=hot_path.stat().st_size,
            already_extracted=date_str in extracted,
        ))

    return results


@router.get("/streamrecorder-backups", response_model=StreamrecorderBackupsResponse)
async def list_streamrecorder_backups(request: Request):
    """Scan StreamRecorder backup folder for available cold+hot pairs."""
    backup_dir = _streamrecorder_dir(request)
    processed = _processed_dir(request)
    extracted = set(_get_extracted_dates(processed))
    backups = _scan_backups(backup_dir, extracted)
    return StreamrecorderBackupsResponse(
        backups=backups,
        backup_dir=str(backup_dir.resolve()),
    )


def _run_restore(mysql_bin: str, backup_dir: Path, db_name: str, gz_file: str,
                 user: str, password: str) -> str:
    """Decompress a .sql.gz via Python gzip and pipe into MySQL. Returns error string or empty."""
    import gzip as gzip_mod

    gz_path = backup_dir / gz_file
    logger.info("[restore] Piping %s into %s (mysql_bin=%s)", gz_file, db_name, mysql_bin)

    mysql_cmd = [mysql_bin, "-u", user, f"-p{password}", db_name]

    try:
        with gzip_mod.open(gz_path, "rb") as f:
            sql_data = f.read()
    except Exception as e:
        logger.error("[restore] gzip decompression failed for %s: %s", gz_file, e)
        return f"gzip error: {e}"

    try:
        mysql_proc = subprocess.Popen(
            mysql_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _, mysql_err = mysql_proc.communicate(input=sql_data, timeout=600)

        if mysql_proc.returncode != 0:
            mysql_msg = mysql_err.decode(errors="replace").strip()
            logger.error("[restore] mysql import failed for %s: %s", gz_file, mysql_msg)
            return mysql_msg
        logger.info("[restore] Successfully restored %s into %s", gz_file, db_name)
        return ""
    except subprocess.TimeoutExpired:
        mysql_proc.kill()
        logger.error("[restore] Restore timed out for %s (10 min limit)", gz_file)
        return "Restore timed out (10 min limit)"
    except FileNotFoundError as e:
        logger.error("[restore] Command not found: %s", e)
        return f"Command not found: {e}"


def _drop_and_create_db(mysql_bin: str, db_name: str, user: str, password: str) -> str:
    """Drop and recreate a MySQL database. Returns error string or empty on success."""
    sql = f"DROP DATABASE IF EXISTS `{db_name}`; CREATE DATABASE `{db_name}`;"
    cmd = [mysql_bin, "-u", user, f"-p{password}", "-e", sql]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            msg = result.stderr.decode(errors="replace").strip()
            logger.error("[restore] Failed to recreate DB %s: %s", db_name, msg)
            return msg
        logger.info("[restore] Recreated database %s", db_name)
        return ""
    except FileNotFoundError as e:
        logger.error("[restore] Command not found: %s", e)
        return f"Command not found: {e}"


@router.post("/restore-backups", response_model=RestoreResponse)
async def restore_backups(body: RestoreRequest, request: Request):
    """Restore selected dates from StreamRecorder backups into MySQL, then extract Parquet."""
    if not body.dates:
        logger.warning("[restore] Rejected: no dates selected")
        raise HTTPException(status_code=400, detail="No dates selected")

    logger.info("[restore] Request received for dates: %s", body.dates)
    backup_dir = _streamrecorder_dir(request)
    processed = _processed_dir(request)
    extracted = set(_get_extracted_dates(processed))
    all_backups = _scan_backups(backup_dir, extracted)
    backup_by_date = {b.date: b for b in all_backups}

    # Validate all requested dates exist
    missing = [d for d in body.dates if d not in backup_by_date]
    if missing:
        logger.warning("[restore] Rejected: missing backups for %s", missing)
        raise HTTPException(
            status_code=400,
            detail=f"No backups found for: {', '.join(missing)}",
        )

    config = request.app.state.config
    mysql_bin = _mysql_bin(request)
    progress_queue = request.app.state.progress_queue
    training_state = request.app.state.training_state

    # Read credentials from env
    user = os.environ.get("DB_USER", "root")
    password = os.environ.get("DB_PASSWORD", "")
    cold_db = config["database"]["cold_data_db"]
    hot_db = config["database"]["hot_data_db"]

    job_id = str(uuid.uuid4())
    dates_to_restore = sorted(body.dates)

    async def _run_restore_job():
        import time

        async def _with_heartbeat(blocking_fn, detail: str, process_snapshot: dict):
            """Run *blocking_fn* in a thread while sending heartbeat events every 5 s."""
            logger.info("[restore] %s", detail)
            task = asyncio.get_event_loop().run_in_executor(None, blocking_fn)
            elapsed = 0
            while True:
                try:
                    result = await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
                    logger.info("[restore] Done: %s (%.0fs)", detail, elapsed)
                    return result
                except asyncio.TimeoutError:
                    elapsed += 5
                    logger.info("[restore] Still working: %s (%ds elapsed)", detail, elapsed)
                    await progress_queue.put({
                        "event": "progress",
                        "timestamp": time.time(),
                        "phase": "restoring",
                        "process": process_snapshot,
                        "detail": detail,
                    })

        training_state["running"] = True
        total = len(dates_to_restore)
        logger.info("[restore] Job %s started: %d date(s) — %s", job_id, total, dates_to_restore)

        try:
            await progress_queue.put({
                "event": "phase_start",
                "timestamp": time.time(),
                "phase": "restoring",
                "summary": {"job_id": job_id, "total_dates": total},
            })

            succeeded = []
            failed = []
            for i, date_str in enumerate(dates_to_restore):
                backup = backup_by_date[date_str]
                steps_per_date = 3  # drop+create+restore cold, hot, extract
                base_progress = i * steps_per_date

                def _make_process(step_offset: int) -> dict:
                    completed = base_progress + step_offset
                    return {
                        "label": "Restoring backups",
                        "completed": completed,
                        "total": total * steps_per_date,
                        "pct": round(completed / (total * steps_per_date) * 100, 1),
                        "item_eta_human": "",
                        "process_eta_human": "",
                    }

                # Step 1: Restore coldData
                step1_detail = f"Restoring coldData for {date_str}..."
                await progress_queue.put({
                    "event": "progress",
                    "timestamp": time.time(),
                    "phase": "restoring",
                    "process": _make_process(0),
                    "detail": step1_detail,
                })

                err = await _with_heartbeat(
                    lambda: _drop_and_create_db(mysql_bin, cold_db, user, password),
                    step1_detail, _make_process(0),
                )
                if err:
                    logger.error("Failed to recreate %s: %s", cold_db, err)
                    failed.append(date_str)
                    await progress_queue.put({
                        "event": "progress",
                        "timestamp": time.time(),
                        "phase": "restoring",
                        "detail": f"ERROR recreating {cold_db}: {err}",
                    })
                    continue

                err = await _with_heartbeat(
                    lambda: _run_restore(mysql_bin, backup_dir, cold_db, backup.cold_file, user, password),
                    step1_detail, _make_process(0),
                )
                if err:
                    logger.error("Failed to restore cold for %s: %s", date_str, err)
                    failed.append(date_str)
                    await progress_queue.put({
                        "event": "progress",
                        "timestamp": time.time(),
                        "phase": "restoring",
                        "detail": f"ERROR restoring coldData for {date_str}: {err}",
                    })
                    continue

                # Step 2: Restore hotData
                step2_detail = f"Restoring hotData for {date_str}..."
                await progress_queue.put({
                    "event": "progress",
                    "timestamp": time.time(),
                    "phase": "restoring",
                    "process": _make_process(1),
                    "detail": step2_detail,
                })

                err = await _with_heartbeat(
                    lambda: _drop_and_create_db(mysql_bin, hot_db, user, password),
                    step2_detail, _make_process(1),
                )
                if err:
                    logger.error("Failed to recreate %s: %s", hot_db, err)
                    failed.append(date_str)
                    await progress_queue.put({
                        "event": "progress",
                        "timestamp": time.time(),
                        "phase": "restoring",
                        "detail": f"ERROR recreating {hot_db}: {err}",
                    })
                    continue

                err = await _with_heartbeat(
                    lambda: _run_restore(mysql_bin, backup_dir, hot_db, backup.hot_file, user, password),
                    step2_detail, _make_process(1),
                )
                if err:
                    logger.error("Failed to restore hot for %s: %s", date_str, err)
                    failed.append(date_str)
                    await progress_queue.put({
                        "event": "progress",
                        "timestamp": time.time(),
                        "phase": "restoring",
                        "detail": f"ERROR restoring hotData for {date_str}: {err}",
                    })
                    continue

                # Step 3: Extract Parquet
                step3_detail = f"Extracting Parquet for {date_str}..."
                await progress_queue.put({
                    "event": "progress",
                    "timestamp": time.time(),
                    "phase": "restoring",
                    "process": _make_process(2),
                    "detail": step3_detail,
                    "sub_process": {
                        "label": f"Extracting Parquet for {date_str}",
                        "completed": 0,
                        "total": 6,
                    },
                })

                try:
                    from data.extractor import DataExtractor

                    extractor = DataExtractor(config)
                    target = date.fromisoformat(date_str)

                    # Thread-safe progress callback — DataExtractor runs in an
                    # executor thread, so we hop back onto the event loop to
                    # push onto the asyncio queue.
                    loop = asyncio.get_running_loop()
                    captured_date = date_str

                    def _on_extract_progress(
                        step: int, total: int, message: str,
                    ) -> None:
                        event = {
                            "event": "progress",
                            "timestamp": time.time(),
                            "phase": "restoring",
                            "process": _make_process(2),
                            "detail": f"[{captured_date}] {message}",
                            "sub_process": {
                                "label": f"Extracting Parquet for {captured_date}",
                                "completed": step,
                                "total": total,
                            },
                        }
                        loop.call_soon_threadsafe(progress_queue.put_nowait, event)

                    await _with_heartbeat(
                        lambda: extractor.extract_date(
                            target, on_progress=_on_extract_progress,
                        ),
                        step3_detail, _make_process(2),
                    )
                except Exception:
                    logger.exception("Extraction failed for %s", date_str)
                    failed.append(date_str)
                    await progress_queue.put({
                        "event": "progress",
                        "timestamp": time.time(),
                        "phase": "restoring",
                        "detail": f"ERROR extracting {date_str}",
                    })
                    continue

                # Verify parquet file was actually created
                processed = Path(config["paths"]["processed_data"])
                parquet_file = processed / f"{date_str}.parquet"
                if not parquet_file.exists():
                    logger.error("Parquet file not created for %s", date_str)
                    failed.append(date_str)
                    await progress_queue.put({
                        "event": "progress",
                        "timestamp": time.time(),
                        "phase": "restoring",
                        "detail": f"ERROR: no parquet file produced for {date_str}",
                    })
                    continue

                succeeded.append(date_str)
                logger.info("[restore] %s completed successfully", date_str)
                await progress_queue.put({
                    "event": "progress",
                    "timestamp": time.time(),
                    "phase": "restoring",
                    "process": {
                        "label": "Restoring backups",
                        "completed": base_progress + 3,
                        "total": total * steps_per_date,
                        "pct": round((base_progress + 3) / (total * steps_per_date) * 100, 1),
                        "item_eta_human": "",
                        "process_eta_human": "",
                    },
                    "detail": f"Completed {date_str}",
                })

            logger.info(
                "[restore] Job %s finished: %d succeeded, %d failed%s",
                job_id, len(succeeded), len(failed),
                f" (failed: {failed})" if failed else "",
            )
            await progress_queue.put({
                "event": "phase_complete",
                "timestamp": time.time(),
                "phase": "restoring",
                "summary": {
                    "job_id": job_id,
                    "dates_restored": len(succeeded),
                    "dates_failed": len(failed),
                    "failed_dates": failed,
                },
            })
        finally:
            training_state["running"] = False

    asyncio.create_task(_run_restore_job())

    return RestoreResponse(
        job_id=job_id,
        dates_queued=len(dates_to_restore),
        detail=f"Queued {len(dates_to_restore)} date(s) for restore + extraction",
    )


# ── Betting Constraints ─────────────────────────────────────────────


@router.get("/config/constraints", response_model=BettingConstraints)
async def get_betting_constraints(request: Request):
    """Return the current betting constraint settings from config."""
    constraints = request.app.state.config.get("training", {}).get(
        "betting_constraints", {}
    )
    training = request.app.state.config.get("training", {})
    return BettingConstraints(
        max_back_price=constraints.get("max_back_price"),
        max_lay_price=constraints.get("max_lay_price"),
        min_seconds_before_off=constraints.get("min_seconds_before_off", 0),
        reevaluate_garaged_default=training.get("reevaluate_garaged_default", True),
    )


@router.post("/config/constraints", response_model=BettingConstraints)
async def update_betting_constraints(body: BettingConstraints, request: Request):
    """Update betting constraints and persist to config.yaml."""
    import yaml

    # Update in-memory config
    config = request.app.state.config
    config.setdefault("training", {}).setdefault("betting_constraints", {})
    config["training"]["betting_constraints"]["max_back_price"] = body.max_back_price
    config["training"]["betting_constraints"]["max_lay_price"] = body.max_lay_price
    config["training"]["betting_constraints"]["min_seconds_before_off"] = body.min_seconds_before_off
    config["training"]["reevaluate_garaged_default"] = body.reevaluate_garaged_default

    # Read-modify-write the on-disk config to preserve comments and other keys
    config_path = Path(getattr(request.app.state, "config_path", "config.yaml"))
    if config_path.exists():
        with open(config_path) as f:
            on_disk = yaml.safe_load(f) or {}
    else:
        on_disk = {}
    on_disk.setdefault("training", {})["betting_constraints"] = {
        "max_back_price": body.max_back_price,
        "max_lay_price": body.max_lay_price,
        "min_seconds_before_off": body.min_seconds_before_off,
    }
    on_disk.setdefault("training", {})["reevaluate_garaged_default"] = body.reevaluate_garaged_default
    with open(config_path, "w") as f:
        yaml.dump(on_disk, f, default_flow_style=False, sort_keys=False)

    return body


# ── Log Paths ──────────────────────────────────────────────────────────


@router.get("/log-paths", response_model=LogPathsResponse)
async def get_log_paths(request: Request):
    """Return the logs root and stats for each subdirectory."""
    logs_root = Path(request.app.state.config.get("paths", {}).get("logs", "logs"))
    subdirs: list[LogSubdir] = []
    if logs_root.exists():
        for entry in sorted(logs_root.iterdir()):
            if entry.is_dir():
                files = list(entry.rglob("*"))
                files = [f for f in files if f.is_file()]
                total_size = sum(f.stat().st_size for f in files)
                subdirs.append(LogSubdir(
                    name=entry.name,
                    file_count=len(files),
                    total_size_bytes=total_size,
                ))
    return LogPathsResponse(logs_root=str(logs_root), subdirs=subdirs)
