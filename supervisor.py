"""
supervisor.py — Lightweight process manager for rl-betfair.

Manages the training worker, API backend, and Angular frontend as
subprocesses.  Exposes a REST API on port 9000 for the Admin page
to query status, start/stop/restart processes, and view logs.

Usage:
    python supervisor.py          # start all services
    python supervisor.py --idle   # start supervisor only, no auto-start
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).parent.resolve()
_log_dir = ROOT / "logs"
_log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_dir / "supervisor.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("supervisor")

# ── Process definitions ──────────────────────────────────────────────────

PROCESS_DEFS = {
    "worker": {
        "cmd": [sys.executable, "-m", "training.worker"],
        "cwd": str(ROOT),
        "port": 8002,
        "label": "Training Worker",
    },
    "api": {
        "cmd": [
            sys.executable, "-m", "uvicorn", "api.main:app",
            "--reload",
            "--reload-exclude", ".claude",
            "--reload-exclude", "tests",
            "--reload-exclude", "frontend",
            "--reload-exclude", "registry",
            "--reload-exclude", "logs",
            "--port", "8001",
        ],
        "cwd": str(ROOT),
        "port": 8001,
        "label": "API Backend",
    },
    "frontend": {
        "cmd": ["npm.cmd" if sys.platform == "win32" else "npm", "start"],
        "cwd": str(ROOT / "frontend"),
        "port": 4202,
        "label": "Angular Frontend",
    },
}

SUPERVISOR_PORT = 9000
MAX_LOG_LINES = 200


# ── Managed process ─────────────────────────────────────────────────────

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


class ManagedProcess:
    """Wraps a subprocess with log capture and lifecycle management."""

    def __init__(self, name: str, cmd: list[str], cwd: str, port: int, label: str) -> None:
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.port = port
        self.label = label
        self._proc: subprocess.Popen | None = None
        self._started_at: float | None = None
        self._log_buffer: collections.deque[str] = collections.deque(maxlen=MAX_LOG_LINES)
        self._reader_thread: threading.Thread | None = None
        self._log_file: open | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc and self._proc.poll() is None else None

    @property
    def uptime_seconds(self) -> float | None:
        if self._started_at and self.running:
            return time.time() - self._started_at
        return None

    def _clear_port(self) -> None:
        """Kill any stale process holding this service's port."""
        if sys.platform != "win32":
            return
        try:
            result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, timeout=10)
            for line in result.stdout.splitlines():
                if f":{self.port}" in line and "LISTENING" in line:
                    parts = line.split()
                    pid = int(parts[-1])
                    if pid > 0:
                        logger.info("Killing stale process on port %d (PID %d)", self.port, pid)
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=10)
            time.sleep(0.5)
        except Exception as exc:
            logger.warning("Could not clear port %d: %s", self.port, exc)

    def start(self) -> bool:
        if self.running:
            return False

        self._clear_port()

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        self._proc = subprocess.Popen(
            self.cmd,
            cwd=self.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        self._started_at = time.time()
        self._log_buffer.clear()

        # Open persistent log file (append mode, rotates by name)
        log_path = LOG_DIR / f"{self.name}.log"
        self._log_file = open(log_path, "a", encoding="utf-8")
        self._log_file.write(f"\n{'='*60}\n")
        self._log_file.write(f"  {self.label} started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._log_file.write(f"  PID {self._proc.pid} | Port {self.port}\n")
        self._log_file.write(f"{'='*60}\n")
        self._log_file.flush()

        self._reader_thread = threading.Thread(
            target=self._read_output, daemon=True, name=f"log-{self.name}",
        )
        self._reader_thread.start()

        logger.info("Started %s (PID %d)", self.label, self._proc.pid)
        return True

    def stop(self) -> bool:
        if not self.running:
            return False

        pid = self._proc.pid
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True, timeout=10,
                )
            else:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                self._proc.wait(timeout=5)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

        self._proc = None
        self._started_at = None
        logger.info("Stopped %s (was PID %d)", self.label, pid)
        return True

    def restart(self) -> None:
        self.stop()
        time.sleep(1)
        self.start()

    def get_logs(self, lines: int = 50) -> list[str]:
        buf = list(self._log_buffer)
        return buf[-lines:]

    def status_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "status": "running" if self.running else "stopped",
            "pid": self.pid,
            "port": self.port,
            "uptime_seconds": round(self.uptime_seconds, 1) if self.uptime_seconds else None,
        }

    def _read_output(self) -> None:
        try:
            for line in iter(self._proc.stdout.readline, b""):
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    self._log_buffer.append(text)
                    if self._log_file:
                        try:
                            self._log_file.write(text + "\n")
                            self._log_file.flush()
                        except Exception:
                            pass
        except Exception:
            pass
        finally:
            # Process exited — log the exit code
            exit_code = None
            if self._proc is not None:
                try:
                    exit_code = self._proc.wait(timeout=5)
                except Exception:
                    pass
            if exit_code is not None and exit_code != 0:
                msg = f"CRASHED: {self.label} exited with code {exit_code} after {self._uptime_str()}"
                logger.error(msg)
                self._log_buffer.append(msg)
                if self._log_file:
                    try:
                        self._log_file.write(f"\n*** {msg} ***\n")
                    except Exception:
                        pass
            elif exit_code == 0:
                msg = f"{self.label} exited cleanly (code 0) after {self._uptime_str()}"
                logger.info(msg)
            if self._log_file:
                try:
                    self._log_file.close()
                except Exception:
                    pass
                self._log_file = None

    def _uptime_str(self) -> str:
        if self._started_at is None:
            return "unknown duration"
        secs = time.time() - self._started_at
        if secs < 60:
            return f"{secs:.0f}s"
        mins = secs / 60
        if mins < 60:
            return f"{mins:.0f}m"
        return f"{mins / 60:.1f}h"


# ── FastAPI app ──────────────────────────────────────────────────────────

app = FastAPI(title="rl-betfair supervisor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4202"],
    allow_methods=["*"],
    allow_headers=["*"],
)

processes: dict[str, ManagedProcess] = {}


@app.get("/api/processes")
async def get_processes():
    return {name: p.status_dict() for name, p in processes.items()}


@app.post("/api/processes/{name}/start")
async def start_process(name: str):
    p = processes.get(name)
    if p is None:
        return {"error": f"Unknown process: {name}"}
    if p.running:
        return {"status": "already_running", **p.status_dict()}
    p.start()
    time.sleep(0.5)
    return {"status": "started", **p.status_dict()}


@app.post("/api/processes/{name}/stop")
async def stop_process(name: str):
    p = processes.get(name)
    if p is None:
        return {"error": f"Unknown process: {name}"}
    if not p.running:
        return {"status": "already_stopped", **p.status_dict()}
    p.stop()
    return {"status": "stopped", **p.status_dict()}


@app.post("/api/processes/{name}/restart")
async def restart_process(name: str):
    p = processes.get(name)
    if p is None:
        return {"error": f"Unknown process: {name}"}
    p.restart()
    time.sleep(0.5)
    return {"status": "restarted", **p.status_dict()}


@app.get("/api/processes/{name}/logs")
async def get_logs(name: str, lines: int = 50):
    p = processes.get(name)
    if p is None:
        return {"error": f"Unknown process: {name}"}
    return {"name": name, "logs": p.get_logs(lines)}


# ── Entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="rl-betfair supervisor")
    parser.add_argument("--idle", action="store_true", help="Start without auto-launching services")
    parser.add_argument("--port", type=int, default=SUPERVISOR_PORT, help="Supervisor API port")
    args = parser.parse_args()

    for name, defn in PROCESS_DEFS.items():
        processes[name] = ManagedProcess(
            name=name,
            cmd=defn["cmd"],
            cwd=defn["cwd"],
            port=defn["port"],
            label=defn["label"],
        )

    if not args.idle:
        logger.info("Starting all processes...")
        for name in ["worker", "api", "frontend"]:
            processes[name].start()
            if name in ("worker", "api"):
                time.sleep(2)
        logger.info("All processes started.")

    logger.info("Supervisor API on http://localhost:%d", args.port)

    config = uvicorn.Config(app, host="0.0.0.0", port=args.port, log_level="warning")
    server = uvicorn.Server(config)

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down all processes...")
        for p in processes.values():
            p.stop()
        logger.info("Supervisor stopped.")


if __name__ == "__main__":
    main()
