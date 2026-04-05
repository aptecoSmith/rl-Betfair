"""System metrics and service control endpoints."""

from __future__ import annotations

import logging
import os
import re
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from api.schemas import (
    GpuMetrics,
    ServiceActionResponse,
    ServiceStatus,
    ServicesResponse,
    SystemMetrics,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["system"])

VALID_SERVICES = {"worker", "api", "frontend"}
VALID_ACTIONS = {"start", "stop", "restart"}

SERVICE_PORTS = {
    "worker": 8002,
    "api": 8001,
    "frontend": 4202,
}


# ── GPU metrics ─────────────────────────────────────────────────────


def _get_gpu_metrics() -> GpuMetrics | None:
    """Read GPU utilisation and VRAM via pynvml.  Returns None if unavailable."""
    try:
        import pynvml  # noqa: F811 — wraps nvidia-ml-py

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = None
        try:
            temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            pass

        pynvml.nvmlShutdown()

        return GpuMetrics(
            name=name,
            utilisation_pct=util.gpu,
            memory_used_mb=round(mem.used / 1024 / 1024),
            memory_total_mb=round(mem.total / 1024 / 1024),
            temperature_c=temp,
        )
    except Exception as exc:
        logger.debug("GPU metrics unavailable: %s", exc)
        return None


@router.get("/system/metrics", response_model=SystemMetrics)
def get_system_metrics():
    """Return current CPU, RAM, GPU, and disk stats."""
    import psutil

    vm = psutil.virtual_memory()
    disk = psutil.disk_io_counters()
    disk_usage = psutil.disk_usage("/")

    return SystemMetrics(
        cpu_pct=psutil.cpu_percent(interval=0.1),
        ram_used_mb=round(vm.used / 1024 / 1024),
        ram_total_mb=round(vm.total / 1024 / 1024),
        ram_pct=vm.percent,
        disk_read_mb_s=round(disk.read_bytes / 1024 / 1024, 1) if disk else 0,
        disk_write_mb_s=round(disk.write_bytes / 1024 / 1024, 1) if disk else 0,
        disk_used_gb=round(disk_usage.used / 1024 / 1024 / 1024, 1),
        disk_total_gb=round(disk_usage.total / 1024 / 1024 / 1024, 1),
        gpu=_get_gpu_metrics(),
    )


# ── Service control ─────────────────────────────────────────────────


def _is_port_listening(port: int) -> bool:
    """Check if a TCP port has a listener."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except (ConnectionRefusedError, OSError):
        return False


def _kill_by_port(port: int) -> list[int]:
    """Kill processes listening on a port. Returns list of killed PIDs."""
    killed = []
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                if parts:
                    pid = int(parts[-1])
                    if pid > 0:
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(pid)],
                            capture_output=True, timeout=10,
                        )
                        killed.append(pid)
    except Exception as exc:
        logger.warning("Failed to kill by port %d: %s", port, exc)
    return killed


def _kill_by_cmdline(pattern: str) -> list[int]:
    """Kill processes matching a command line pattern via wmic. Returns PIDs."""
    killed = []
    try:
        result = subprocess.run(
            ["wmic", "process", "where",
             f"CommandLine like '%{pattern}%'",
             "get", "ProcessId", "/format:csv"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    pid = int(parts[-1])
                    if pid > 0 and pid != os.getpid():
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(pid)],
                            capture_output=True, timeout=10,
                        )
                        killed.append(pid)
                except ValueError:
                    continue
    except Exception as exc:
        logger.warning("Failed to kill by pattern '%s': %s", pattern, exc)
    return killed


def _creation_flags() -> int:
    """Return subprocess creation flags for detached processes on Windows."""
    if sys.platform == "win32":
        return subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    return 0


# ── Service-specific start/stop ─────────────────────────────────────


def _stop_worker() -> str:
    killed = _kill_by_port(SERVICE_PORTS["worker"])
    killed += _kill_by_cmdline("training.worker")
    if killed:
        return f"Worker stopped (PIDs: {killed})"
    return "Worker was not running"


def _start_worker() -> str:
    if _is_port_listening(SERVICE_PORTS["worker"]):
        raise HTTPException(409, "Worker is already running on port 8002")
    subprocess.Popen(
        [sys.executable, "-m", "training.worker"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=_creation_flags(),
    )
    return "Worker starting on port 8002"


def _stop_frontend() -> str:
    killed = _kill_by_port(SERVICE_PORTS["frontend"])
    killed += _kill_by_cmdline("ng serve")
    if killed:
        return f"Frontend stopped (PIDs: {killed})"
    return "Frontend was not running"


def _start_frontend() -> str:
    if _is_port_listening(SERVICE_PORTS["frontend"]):
        raise HTTPException(409, "Frontend is already running on port 4202")
    frontend_dir = str(Path.cwd() / "frontend")
    subprocess.Popen(
        ["npm", "start"],
        cwd=frontend_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=_creation_flags(),
        shell=True,
    )
    return "Frontend starting on port 4202"


def _stop_api() -> str:
    # Schedule self-termination after a brief delay so the HTTP response can be sent
    def _delayed_kill():
        import time
        time.sleep(1.5)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_delayed_kill, daemon=True).start()
    return "API shutting down in 1.5 seconds"


def _restart_api() -> str:
    # Spawn a replacement API process, then schedule self-kill
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--reload", "--reload-exclude", ".claude",
         "--reload-exclude", "*.log", "--port", "8001"],
        creationflags=_creation_flags(),
    )

    def _delayed_kill():
        import time
        time.sleep(1.5)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_delayed_kill, daemon=True).start()
    return "API restarting — new instance spawning on port 8001"


# ── Endpoints ───────────────────────────────────────────────────────


@router.get("/system/health")
def health_check():
    """Simple health check for reconnection polling."""
    return {"status": "ok"}


@router.get("/system/services", response_model=ServicesResponse)
def get_services(request: Request):
    """Return running status of all 3 services."""
    worker_connected = getattr(request.app.state, "worker_connected", False)
    return ServicesResponse(services=[
        ServiceStatus(
            name="worker",
            port=SERVICE_PORTS["worker"],
            running=worker_connected or _is_port_listening(SERVICE_PORTS["worker"]),
        ),
        ServiceStatus(
            name="api",
            port=SERVICE_PORTS["api"],
            running=True,
        ),
        ServiceStatus(
            name="frontend",
            port=SERVICE_PORTS["frontend"],
            running=_is_port_listening(SERVICE_PORTS["frontend"]),
        ),
    ])


@router.post(
    "/system/services/{service}/{action}",
    response_model=ServiceActionResponse,
)
def control_service(service: str, action: str):
    """Start, stop, or restart a service."""
    if service not in VALID_SERVICES:
        raise HTTPException(400, f"Unknown service: {service}")
    if action not in VALID_ACTIONS:
        raise HTTPException(400, f"Unknown action: {action}")

    try:
        if action == "stop":
            if service == "worker":
                detail = _stop_worker()
            elif service == "api":
                detail = _stop_api()
            elif service == "frontend":
                detail = _stop_frontend()

        elif action == "start":
            if service == "worker":
                detail = _start_worker()
            elif service == "api":
                # Can't start the API from itself — it's already running
                raise HTTPException(409, "API is already running (you're talking to it)")
            elif service == "frontend":
                detail = _start_frontend()

        elif action == "restart":
            if service == "worker":
                _stop_worker()
                import time
                time.sleep(1)
                detail = _start_worker()
            elif service == "api":
                detail = _restart_api()
            elif service == "frontend":
                _stop_frontend()
                import time
                time.sleep(1)
                detail = _start_frontend()

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Service control failed: {exc}")

    return ServiceActionResponse(
        service=service,
        action=action,
        success=True,
        detail=detail,
    )
