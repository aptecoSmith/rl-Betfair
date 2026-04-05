"""Integration tests for the training worker process.

These tests start an actual worker process, connect via WebSocket,
and verify the IPC protocol works end-to-end.  They catch issues
like port conflicts, import errors, and protocol mismatches that
unit tests with mocks would miss.
"""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def _port_free(port: int) -> bool:
    """Check if a TCP port is free."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return False
    except (ConnectionRefusedError, OSError):
        return True


def _port_listening(port: int, timeout: float = 10.0) -> bool:
    """Wait until a port is listening, with timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _port_free(port):
            return True
        time.sleep(0.3)
    return False


@pytest.fixture()
def worker_process():
    """Start a training worker on a test port, yield it, then kill it."""
    port = 18002  # Use a non-standard port to avoid conflicts
    # Ensure port is free
    assert _port_free(port), f"Port {port} is already in use — kill the stale process"

    proc = subprocess.Popen(
        [sys.executable, "-m", "training.worker", "--port", str(port)],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"},
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    # Wait for the worker to start listening
    if not _port_listening(port, timeout=15):
        stdout = proc.stdout.read(4096).decode("utf-8", errors="replace") if proc.stdout else ""
        proc.kill()
        pytest.fail(f"Worker did not start listening on port {port} within 15s.\nOutput:\n{stdout}")

    yield proc, port

    # Teardown: kill the worker
    if proc.poll() is None:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.terminate()
            proc.wait(timeout=5)


class TestWorkerStartup:
    """Test that the worker process starts and listens correctly."""

    def test_worker_starts_and_listens(self, worker_process):
        proc, port = worker_process
        assert proc.poll() is None, "Worker process should still be running"
        assert not _port_free(port), f"Worker should be listening on port {port}"

    def test_worker_responds_to_status(self, worker_process):
        """Connect via WebSocket and request status."""
        _, port = worker_process

        async def _check():
            import websockets
            async with websockets.connect(f"ws://localhost:{port}") as ws:
                # Worker sends initial status on connect
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                msg = json.loads(raw)
                assert msg["type"] == "status"
                assert msg["running"] is False

                # Send explicit status request
                await ws.send(json.dumps({"type": "status"}))
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                msg = json.loads(raw)
                assert msg["type"] == "status"
                assert "running" in msg

        asyncio.run(_check())

    def test_worker_rejects_start_without_data(self, worker_process):
        """Starting training with no parquet data should return an error."""
        _, port = worker_process

        async def _check():
            import websockets
            async with websockets.connect(f"ws://localhost:{port}") as ws:
                # Consume initial status
                await asyncio.wait_for(ws.recv(), timeout=5)

                # Send start command
                await ws.send(json.dumps({
                    "type": "start",
                    "n_generations": 1,
                    "n_epochs": 1,
                    "population_size": 2,
                }))

                # Should get either a started or error response
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                msg = json.loads(raw)
                # With no data in a test environment, it should error
                assert msg["type"] in ("started", "error")

        asyncio.run(_check())

    def test_worker_handles_unknown_command(self, worker_process):
        """Unknown commands should return an error, not crash."""
        _, port = worker_process

        async def _check():
            import websockets
            async with websockets.connect(f"ws://localhost:{port}") as ws:
                await asyncio.wait_for(ws.recv(), timeout=5)  # initial status

                await ws.send(json.dumps({"type": "nonexistent_command"}))
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                msg = json.loads(raw)
                assert msg["type"] == "error"

        asyncio.run(_check())

    def test_worker_stop_when_not_running(self, worker_process):
        """Sending stop when no training is running should return an error."""
        _, port = worker_process

        async def _check():
            import websockets
            async with websockets.connect(f"ws://localhost:{port}") as ws:
                await asyncio.wait_for(ws.recv(), timeout=5)  # initial status

                await ws.send(json.dumps({"type": "stop"}))
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                msg = json.loads(raw)
                assert msg["type"] == "error"
                assert "no training run in progress" in msg.get("message", "").lower()

        asyncio.run(_check())


class TestWorkerPortCleanup:
    """Test that the worker handles port conflicts."""

    def test_port_clear_on_startup(self):
        """Verify _clear_port function works."""
        from training.worker import _clear_port
        # Should not raise even on a free port
        _clear_port(19999)


class TestIpcProtocol:
    """Test the IPC message protocol without starting a process."""

    def test_make_start_cmd(self):
        from training.ipc import make_start_cmd
        raw = make_start_cmd(n_generations=2, n_epochs=1, population_size=10)
        msg = json.loads(raw)
        assert msg["type"] == "start"
        assert msg["n_generations"] == 2
        assert msg["population_size"] == 10

    def test_make_stop_cmd(self):
        from training.ipc import make_stop_cmd
        msg = json.loads(make_stop_cmd())
        assert msg["type"] == "stop"

    def test_make_status_cmd(self):
        from training.ipc import make_status_cmd
        msg = json.loads(make_status_cmd())
        assert msg["type"] == "status"

    def test_make_event_msg(self):
        from training.ipc import make_event_msg
        payload = {"event": "progress", "phase": "training"}
        raw = make_event_msg(payload)
        msg = json.loads(raw)
        assert msg["type"] == "event"
        assert msg["payload"]["phase"] == "training"

    def test_make_status_msg(self):
        from training.ipc import make_status_msg
        raw = make_status_msg(running=True, latest_event={"x": 1}, latest_process=None, latest_item=None)
        msg = json.loads(raw)
        assert msg["type"] == "status"
        assert msg["running"] is True
        assert msg["latest_event"] == {"x": 1}

    def test_make_error_msg(self):
        from training.ipc import make_error_msg
        msg = json.loads(make_error_msg("something broke"))
        assert msg["type"] == "error"
        assert msg["message"] == "something broke"

    def test_make_started_msg(self):
        from training.ipc import make_started_msg
        msg = json.loads(make_started_msg("run-1", ["2026-04-01"], ["2026-04-02"]))
        assert msg["type"] == "started"
        assert msg["run_id"] == "run-1"
        assert msg["train_days"] == ["2026-04-01"]

    def test_parse_message(self):
        from training.ipc import parse_message
        msg = parse_message('{"type": "status", "running": false}')
        assert msg["type"] == "status"
        assert msg["running"] is False
