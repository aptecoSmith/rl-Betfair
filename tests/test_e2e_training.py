"""End-to-end test: start worker, run a minimal training, verify results.

This test exercises the full training pipeline through the actual WebSocket
interface — the same path a user takes from the UI.  It catches integration
issues (port conflicts, broken IPC, missing data) that unit tests miss.

Uses real parquet data from data/processed/ if available (at least 2 days),
otherwise writes synthetic data to a temp directory.

Starts a real worker process on a non-standard port to avoid conflicts.

Running this test
-----------------
Run with ``-s`` to see live progress (pytest captures stdout by default)::

    python -m pytest tests/test_e2e_training.py -xvs

Typical timeline with real data (39 races, ~9 000 ticks per day):

    0-5s     Worker startup + WebSocket handshake
    5-30s    Training phase (GPU/CPU spike visible)
    30-90s   Evaluation phase — usually 15-20s of actual compute
             (feature eng ~4s + step loop ~12s at ~800 steps/s).
             CPU/GPU will appear idle because individual forward
             passes are tiny.

The evaluation step loop is **model-dependent**: an aggressive model
that places many passive orders causes ``PassiveOrderBook.on_tick()``
to slow down (O(orders * ticks) per race).  A conservative model
finishes in ~15s; an aggressive one may take 5+ minutes on the same
data.  If CPU/GPU spike and then go idle but the test keeps waiting,
this is what's happening — the step loop is still running, just slowly.

The heartbeat line ``... waiting for events (phase: evaluating)``
confirms the WebSocket is alive.  The evaluator prints a timing
breakdown when it finishes (``env_init=Xs step_loop=Ys ...``) so you
can see exactly where the time went.

All ``print()`` output goes through ``_print()`` which forces
``flush=True`` — without this, Python buffers stdout and you see
nothing until the test finishes or times out.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import yaml

from tests._port_utils import kill_stale_on_port, port_free as _port_free

ROOT = Path(__file__).parent.parent
WORKER_PORT = 18002


def _print(*args: object, **kwargs: object) -> None:
    """Print with immediate flush so progress is visible in real time."""
    print(*args, **kwargs, flush=True)  # type: ignore[arg-type]


# ── Helpers ──────────────────────────────────────────────────────────


def _wait_for_ws(port: int, timeout: float = 30.0) -> bool:
    """Wait until a WebSocket connection can be established."""
    async def _try():
        import websockets
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                async with websockets.connect(
                    f"ws://127.0.0.1:{port}", open_timeout=3,
                ) as ws:
                    await asyncio.wait_for(ws.recv(), timeout=3)
                    return True
            except Exception:
                await asyncio.sleep(1)
        return False
    return asyncio.run(_try())


def _has_real_data() -> bool:
    processed = ROOT / "data" / "processed"
    if not processed.exists():
        return False
    parquets = [f for f in processed.glob("*.parquet") if not f.stem.endswith("_runners")]
    return len(parquets) >= 2


def _write_synthetic_parquets(data_dir: Path) -> list[str]:
    from tests.test_episode_builder import _write_parquet_pair
    dates = ["2026-01-01", "2026-01-02"]
    for d in dates:
        _write_parquet_pair(data_dir, date_str=d, n_ticks=10, n_runners=3, n_markets=2)
    return dates


def _make_test_config(tmp_dir: Path, data_dir: str) -> Path:
    db_path = str(tmp_dir / "test_models.db")
    weights_dir = str(tmp_dir / "weights")
    os.makedirs(weights_dir, exist_ok=True)

    config = {
        "database": {"host": "localhost", "port": 3306, "cold_data_db": "coldData", "hot_data_db": "hotDataRefactored"},
        "population": {"size": 1, "n_elite": 1, "selection_top_pct": 0.5, "mutation_rate": 0.3},
        "reward": {
            "early_pick_bonus_min": 1.2, "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300, "commission": 0.05,
            "efficiency_penalty": 0.01, "precision_bonus": 1.0,
            "coefficients": {"win_rate": 0.35, "sharpe": 0.30, "mean_daily_pnl": 0.15, "efficiency": 0.20},
        },
        "paths": {
            "processed_data": data_dir,
            "backup_data": str(tmp_dir / "backup"),
            "model_weights": weights_dir,
            "logs": str(tmp_dir / "logs"),
            "registry_db": db_path,
            "streamrecorder_backups": str(tmp_dir / "backups"),
            "mysql_bin": "mysql",
        },
        "training": {
            "architecture": "ppo_lstm_v1",
            "starting_budget": 100.0,
            "max_runners": 14,
            "max_bets_per_race": 20,
            "require_gpu": False,
            # Force CPU: single-step LSTM inference is faster on CPU than
            # GPU due to per-step CPU↔GPU transfer overhead (no batching).
            "device": "cpu",
            "retraining": {"min_days": 30},
        },
        "discard_policy": {"min_win_rate": 0.0, "min_mean_pnl": -9999, "min_sharpe": -9999},
        "hyperparameters": {
            "search_ranges": {
                "learning_rate": {"type": "float_log", "min": 1e-4, "max": 5e-4},
                "ppo_clip_epsilon": {"type": "float", "min": 0.2, "max": 0.2},
                "entropy_coefficient": {"type": "float", "min": 0.01, "max": 0.01},
                "lstm_hidden_size": {"type": "int_choice", "choices": [64]},
                "mlp_hidden_size": {"type": "int_choice", "choices": [64]},
                "mlp_layers": {"type": "int", "min": 1, "max": 1},
                "early_pick_bonus_min": {"type": "float", "min": 1.1, "max": 1.1},
                "early_pick_bonus_max": {"type": "float", "min": 1.3, "max": 1.3},
                "reward_efficiency_penalty": {"type": "float", "min": 0.01, "max": 0.01},
                "reward_precision_bonus": {"type": "float", "min": 1.0, "max": 1.0},
                "architecture_name": {"type": "str_choice", "choices": ["ppo_lstm_v1"]},
            },
        },
        "training_worker": {"host": "localhost", "port": WORKER_PORT},
    }

    config_path = tmp_dir / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def e2e_env():
    tmp_dir = Path(tempfile.mkdtemp(prefix="rl_e2e_"))

    data_dir = str(tmp_dir / "processed")
    os.makedirs(data_dir, exist_ok=True)

    # Always use synthetic data: the e2e test verifies the pipeline works
    # end-to-end, not real-data performance.  Real data (9 000+ ticks) causes
    # the worker subprocess step loop to hang due to GIL contention between
    # the training thread and the asyncio event bridge.  Synthetic data
    # (10 ticks × 3 runners × 2 markets) completes in seconds regardless.
    _write_synthetic_parquets(Path(data_dir))

    config_path = _make_test_config(tmp_dir, data_dir)
    parquet_count = len([f for f in Path(data_dir).glob("*.parquet") if not f.stem.endswith("_runners")])
    _print(f"\n  E2E setup: config={config_path}")
    _print(f"  Data dir: {data_dir} ({parquet_count} days)")
    _print(f"  Using real data: {_has_real_data()}")
    _print(f"  DB: {tmp_dir / 'test_models.db'}")
    yield {"tmp_dir": tmp_dir, "config_path": config_path, "data_dir": data_dir}
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def worker_proc(e2e_env):
    # Recover from a stale worker orphaned by a previous failed run.
    killed = kill_stale_on_port(WORKER_PORT)
    if killed:
        _print(f"  [test_e2e_training] killed stale PIDs on port {WORKER_PORT}: {killed}")
    assert _port_free(WORKER_PORT), f"Port {WORKER_PORT} still in use after kill"

    proc = subprocess.Popen(
        [sys.executable, "-m", "training.worker",
         "--port", str(WORKER_PORT), "--host", "0.0.0.0",
         "--config", str(e2e_env["config_path"])],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    _print(f"  Waiting for worker WebSocket on port {WORKER_PORT}...")
    if not _wait_for_ws(WORKER_PORT, timeout=30):
        out = proc.stdout.read(8192).decode("utf-8", errors="replace") if proc.stdout else ""
        proc.kill()
        pytest.fail(f"Worker WebSocket not ready on port {WORKER_PORT} within 30s.\nOutput:\n{out}")

    _print(f"  Worker ready (PID {proc.pid})")
    yield proc

    if proc.poll() is None:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.terminate()
            proc.wait(timeout=5)


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.timeout(600)
def test_full_training_flow(worker_proc, e2e_env):
    """Single end-to-end test: connect, verify idle, start training,
    wait for completion, verify models in DB, verify idle again.

    Uses a single asyncio.run() and a single WebSocket connection to
    avoid connection issues between test methods.
    """
    assert worker_proc.poll() is None, "Worker process is not running"
    t0 = time.monotonic()

    def _elapsed() -> str:
        return f"[{time.monotonic() - t0:.1f}s]"

    async def _run():
        import websockets

        _print(f"\n  {_elapsed()} Connecting to worker on ws://127.0.0.1:{WORKER_PORT}")
        # Generous ping timeout: the worker's asyncio event loop can stall
        # during CPU-bound phases (model saving, scoring) that hold the GIL,
        # preventing it from responding to WebSocket pings promptly.
        async with websockets.connect(
            f"ws://127.0.0.1:{WORKER_PORT}",
            open_timeout=10,
            ping_interval=30,
            ping_timeout=600,
        ) as ws:
            # ── Step 1: Verify idle ───────────────────────────────
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            msg = json.loads(raw)
            _print(f"  {_elapsed()} Initial status: running={msg.get('running')}")
            assert msg["type"] == "status", f"Expected status, got {msg['type']}"
            assert msg["running"] is False, "Worker should be idle before training"

            # ── Step 2: Start training ────────────────────────────
            _print(f"  {_elapsed()} Sending start command (1 agent, 1 gen, 1 epoch)")
            await ws.send(json.dumps({
                "type": "start",
                "n_generations": 1,
                "n_epochs": 1,
                "population_size": 1,
                "seed": 42,
            }))

            # Wait for started response — may take time if data loading is slow
            _print(f"  {_elapsed()} Waiting for 'started' (data loading can take up to 60s)...")
            for attempt in range(6):
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                    msg = json.loads(raw)
                    msg_type = msg.get("type")
                    _print(f"  {_elapsed()} Response: type={msg_type}")
                    if msg_type == "error":
                        pytest.skip(f"Worker could not start: {msg.get('message')}")
                    if msg_type == "started":
                        break
                except asyncio.TimeoutError:
                    _print(f"  {_elapsed()} ... still loading data ({(attempt + 1) * 10}s / 60s)")
                    continue
            else:
                pytest.fail("Did not receive 'started' response within 60s")

            # ── Step 3: Wait for completion (10 min timeout) ──────
            _print(f"  {_elapsed()} Training started, waiting for completion...")
            _print(f"  {'-' * 70}")
            deadline = time.monotonic() + 600
            completed = False
            event_count = 0
            phase_timers: dict[str, float] = {}   # phase -> start time
            phase_durations: list[tuple[str, float]] = []  # (phase, seconds)
            current_phase: str | None = None
            last_heartbeat = time.monotonic()

            def _progress_bar(pct: float, width: int = 30) -> str:
                filled = int(width * pct / 100)
                return f"[{'#' * filled}{'.' * (width - filled)}] {pct:5.1f}%"

            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                except asyncio.TimeoutError:
                    # Heartbeat so watchers know we're still alive
                    elapsed_since_hb = time.monotonic() - last_heartbeat
                    if elapsed_since_hb >= 15:
                        phase_label = f" (phase: {current_phase})" if current_phase else ""
                        _print(f"  {_elapsed()} ... waiting for events{phase_label}")
                        last_heartbeat = time.monotonic()
                    continue
                msg = json.loads(raw)
                event_count += 1

                if msg.get("type") == "event":
                    payload = msg["payload"]
                else:
                    payload = msg

                event_type = payload.get("event")
                phase = payload.get("phase")
                detail = str(payload.get("detail", ""))
                summary = payload.get("summary", {})

                # ── Phase transitions: always print prominently ──
                if event_type == "phase_start":
                    current_phase = phase
                    phase_timers[phase] = time.monotonic()
                    summary_str = " | ".join(f"{k}={v}" for k, v in summary.items()) if summary else ""
                    _print(f"  {_elapsed()} >> PHASE: {phase} {summary_str}")

                elif event_type == "phase_complete":
                    duration = time.monotonic() - phase_timers.get(phase, time.monotonic())
                    phase_durations.append((phase, duration))
                    summary_str = " | ".join(f"{k}={v}" for k, v in summary.items()) if summary else ""
                    _print(f"  {_elapsed()} OK DONE:  {phase} ({duration:.1f}s) {summary_str}")

                # ── Progress events: compact bar + ETA ──
                elif event_type == "progress":
                    proc = payload.get("process")
                    item = payload.get("item")
                    if proc and proc.get("total"):
                        bar = _progress_bar(proc.get("pct", 0))
                        eta = proc.get("process_eta_human", "")
                        label = proc.get("label", phase or "")
                        eta_str = f" ETA {eta}" if eta else ""
                        parts = [f"  {_elapsed()} {bar}{eta_str}  {label}"]
                        # Show nested item progress (e.g. evaluation steps) inline
                        if item and item.get("total"):
                            parts.append(f"  step {item['completed']}/{item['total']}")
                        _print("".join(parts))
                    elif detail:
                        _print(f"  {_elapsed()} i: {detail[:100]}")

                # ── Other events (info, errors): always print ──
                else:
                    _print(f"  {_elapsed()} #{event_count} event={event_type} phase={phase} {detail[:80]}")

                last_heartbeat = time.monotonic()

                if event_type in ("run_complete",) or (
                    event_type == "phase_complete"
                    and phase in ("run_complete", "run_stopped")
                ):
                    total_elapsed = time.monotonic() - t0
                    _print(f"  {'-' * 70}")
                    _print(f"  {_elapsed()} Training completed! ({event_count} events, {total_elapsed:.0f}s total)")
                    if phase_durations:
                        _print(f"  Phase breakdown:")
                        for p_name, p_dur in phase_durations:
                            _print(f"    {p_name:.<30s} {p_dur:6.1f}s")
                    completed = True
                    break
                if event_type == "phase_complete" and phase == "run_error":
                    pytest.fail(f"Training run failed: {payload}")

            assert completed, (
                f"Training did not complete within 10 minutes. "
                f"Received {event_count} events. Worker alive: {worker_proc.poll() is None}"
            )

            # ── Step 4: Verify idle again ─────────────────────────
            _print(f"  {_elapsed()} Verifying worker is idle...")
            await ws.send(json.dumps({"type": "status"}))
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            msg = json.loads(raw)
            _print(f"  {_elapsed()} Post-training status: running={msg.get('running')}")
            assert msg["type"] == "status"
            assert msg["running"] is False, "Worker should be idle after training"

    asyncio.run(_run())

    # ── Step 5: Verify models in DB ───────────────────────────
    db_path = e2e_env["tmp_dir"] / "test_models.db"
    _print(f"  {_elapsed()} Checking DB: {db_path}")
    assert db_path.exists(), "Registry DB was not created"

    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    conn.close()

    _print(f"  {_elapsed()} Models in DB: {count}")
    assert count >= 1, f"Expected at least 1 model in DB, got {count}"
    _print(f"  {_elapsed()} E2E test PASSED")
