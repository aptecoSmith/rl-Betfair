"""Helpers for test fixtures that need to manage TCP ports.

Used by ``test_training_worker`` and ``test_e2e_training`` to recover
from a stale worker process orphaned by a previous failed/interrupted
test run instead of refusing to start.

Scope is intentionally narrow:
- Only kills processes whose listening socket is bound to ``port`` on
  ``127.0.0.1`` / ``0.0.0.0``.
- Only used for the test worker port (currently 18002), never for
  any production / dev port.  Caller must be explicit about which
  port to free.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time

_SAFE_PROCESS_PREFIXES = ("python", "node", "npm")


def _get_process_name_win(pid: int) -> str | None:
    """Return the image name (e.g. 'python.exe') for a PID, or None."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) >= 2:
                return parts[0].strip('"').lower()
    except Exception:
        pass
    return None


def port_free(port: int) -> bool:
    """Return True if a TCP connect to ``127.0.0.1:port`` is refused."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return False
    except (ConnectionRefusedError, OSError):
        return True


def _pids_on_port_windows(port: int) -> list[int]:
    """Return PIDs holding ``port`` in LISTENING state on Windows."""
    try:
        out = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    pids: set[int] = set()
    needle = f":{port} "
    for line in out.splitlines():
        if needle not in line or "LISTENING" not in line:
            continue
        parts = line.split()
        if len(parts) >= 5 and parts[-1].isdigit():
            pids.add(int(parts[-1]))
    return sorted(pids)


def _pids_on_port_unix(port: int) -> list[int]:
    """Return PIDs holding ``port`` on macOS / Linux via ``lsof``."""
    try:
        out = subprocess.run(
            ["lsof", "-tiTCP:" + str(port), "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    return sorted({int(p) for p in out.split() if p.strip().isdigit()})


def kill_stale_on_port(port: int, *, wait_seconds: float = 5.0) -> list[int]:
    """If anything is listening on ``port``, kill it. Return killed PIDs.

    Intended to recover from an orphaned test worker. Caller must
    pass a *test-only* port — passing a production port is a bug.
    Polls for up to ``wait_seconds`` after the kill so the next port
    bind doesn't race the OS releasing the socket.
    """
    if port_free(port):
        return []

    if sys.platform == "win32":
        pids = _pids_on_port_windows(port)
    else:
        pids = _pids_on_port_unix(port)

    killed: list[int] = []
    for pid in pids:
        try:
            if sys.platform == "win32":
                # Verify process is python/node before killing — never use /T
                proc_name = _get_process_name_win(pid)
                if not (proc_name and any(proc_name.startswith(p) for p in _SAFE_PROCESS_PREFIXES)):
                    continue  # skip unknown/system processes
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True, timeout=5,
                )
            else:
                subprocess.run(
                    ["kill", "-9", str(pid)],
                    capture_output=True, timeout=5,
                )
            killed.append(pid)
        except (FileNotFoundError, subprocess.SubprocessError):
            pass

    # Wait for the OS to release the socket
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if port_free(port):
            break
        time.sleep(0.2)

    return killed
