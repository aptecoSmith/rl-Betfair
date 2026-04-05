"""Standalone training worker process.

Runs independently from the API.  Accepts commands (start/stop/status)
over a WebSocket server and streams progress events back to connected
clients (typically the API, which proxies them to the Angular frontend).

Rich terminal output shows live progress with progress bars and tables.

Usage::

    python -m training.worker          # default port 8002
    python -m training.worker --port 8003
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import queue as thread_queue
import sys
import threading
import time
import uuid
from pathlib import Path

import websockets
import websockets.server
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from registry.model_store import ModelStore
from registry.scoreboard import Scoreboard
from training.ipc import (
    DEFAULT_WORKER_HOST,
    DEFAULT_WORKER_PORT,
    CMD_START,
    CMD_STOP,
    CMD_STATUS,
    make_error_msg,
    make_event_msg,
    make_started_msg,
    make_status_msg,
    parse_message,
)

logger = logging.getLogger(__name__)
console = Console()


# ── Worker ──────────────────────────────────────────────────────────


class TrainingWorker:
    """WebSocket server that manages training runs."""

    def __init__(self, config: dict, host: str, port: int) -> None:
        self.config = config
        self.host = host
        self.port = port

        # ModelStore (shared DB via WAL)
        db_path = config["paths"]["registry_db"]
        weights_dir = config["paths"]["model_weights"]
        bet_logs_dir = str(Path(db_path).parent / "bet_logs")
        self.store = ModelStore(
            db_path=db_path,
            weights_dir=weights_dir,
            bet_logs_dir=bet_logs_dir,
        )

        # Training state
        self.running = False
        self.stop_event = threading.Event()
        self.progress_queue: thread_queue.Queue = thread_queue.Queue()
        self.training_thread: threading.Thread | None = None

        # Latest state for catch-up on connect / status queries
        self.latest_event: dict | None = None
        self.latest_process: dict | None = None
        self.latest_item: dict | None = None

        # Connected WebSocket clients
        self.clients: set[websockets.server.ServerConnection] = set()

        # Rich progress display
        self._progress: Progress | None = None
        self._process_task_id = None
        self._item_task_id = None
        self._live: Live | None = None

    # ── State snapshot ──────────────────────────────────────────────

    def _state_msg(self) -> str:
        return make_status_msg(
            running=self.running,
            latest_event=self.latest_event,
            latest_process=self.latest_process,
            latest_item=self.latest_item,
        )

    # ── WebSocket handler ───────────────────────────────────────────

    async def _handle_client(
        self, websocket: websockets.server.ServerConnection,
    ) -> None:
        self.clients.add(websocket)
        peer = websocket.remote_address
        console.print(f"[dim]Client connected: {peer}[/dim]")

        # Send current state so mid-run clients get caught up
        try:
            await websocket.send(self._state_msg())
        except Exception:
            self.clients.discard(websocket)
            return

        try:
            async for raw in websocket:
                msg = parse_message(raw)
                await self._dispatch(msg, websocket)
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            console.print(f"[dim]Client disconnected: {peer}[/dim]")

    async def _dispatch(
        self, msg: dict, ws: websockets.server.ServerConnection,
    ) -> None:
        msg_type = msg.get("type")

        if msg_type == CMD_STATUS:
            await ws.send(self._state_msg())

        elif msg_type == CMD_START:
            if self.running:
                await ws.send(make_error_msg("Training already in progress"))
                return
            await self._start_training(msg, ws)

        elif msg_type == CMD_STOP:
            if not self.running:
                await ws.send(make_error_msg("No training run in progress"))
                return
            self.stop_event.set()
            console.print("[yellow]Stop requested — halting after current agent[/yellow]")
            # Send status back so the API's pending Future resolves
            await ws.send(self._state_msg())

        else:
            await ws.send(make_error_msg(f"Unknown command: {msg_type}"))

    # ── Training lifecycle ──────────────────────────────────────────

    async def _start_training(
        self, params: dict, ws: websockets.server.ServerConnection,
    ) -> None:
        from data.episode_builder import load_days

        # Resolve data
        data_dir = self.config["paths"]["processed_data"]
        processed = Path(data_dir)
        dates = sorted(
            f.stem
            for f in processed.glob("*.parquet")
            if not f.stem.endswith("_runners") and f.stem != ".gitkeep"
        )
        if not dates:
            await ws.send(make_error_msg("No extracted data available — import days first"))
            return

        # Chronological train/test split (~50/50)
        split = max(1, len(dates) // 2)
        train_dates = dates[:split]
        test_dates = dates[split:]

        run_id = str(uuid.uuid4())
        n_generations = params.get("n_generations", 3)
        n_epochs = params.get("n_epochs", 3)
        population_size = params.get("population_size")
        seed = params.get("seed")
        reevaluate_garaged = params.get("reevaluate_garaged", False)
        reevaluate_min_score = params.get("reevaluate_min_score")

        # Apply population size override
        run_config = copy.deepcopy(self.config)
        if population_size is not None:
            run_config["population"]["size"] = population_size
            run_config["population"]["n_elite"] = max(1, population_size // 10)

        # Reset
        self.stop_event.clear()
        self.running = True
        self.latest_event = None
        self.latest_process = None
        self.latest_item = None

        # Clear any stale events from the queue
        while not self.progress_queue.empty():
            try:
                self.progress_queue.get_nowait()
            except thread_queue.Empty:
                break

        # Acknowledge
        await self._broadcast(make_started_msg(run_id, train_dates, test_dates))

        console.print()
        console.rule(f"[bold green]Training Run {run_id[:8]}[/bold green]")
        console.print(f"  Generations: {n_generations}  |  Epochs: {n_epochs}")
        console.print(f"  Population:  {run_config['population']['size']}")
        console.print(f"  Train days:  {len(train_dates)}  |  Test days: {len(test_dates)}")
        console.print()

        def _run_in_thread() -> None:
            from training.run_training import TrainingOrchestrator

            try:
                console.print("[dim]Loading training data...[/dim]")
                train_days = load_days(train_dates, data_dir=data_dir)
                test_days_loaded = load_days(test_dates, data_dir=data_dir)
                console.print(f"[dim]Loaded {len(train_days)} train days, {len(test_days_loaded)} test days[/dim]")

                orch = TrainingOrchestrator(
                    config=run_config,
                    model_store=self.store,
                    progress_queue=self.progress_queue,
                    stop_event=self.stop_event,
                )

                orch.run(
                    train_days=train_days,
                    test_days=test_days_loaded,
                    n_generations=n_generations,
                    n_epochs=n_epochs,
                    seed=seed,
                    reevaluate_garaged=reevaluate_garaged,
                    reevaluate_min_score=reevaluate_min_score,
                )
            except Exception:
                logger.exception("Training run failed")
                try:
                    self.progress_queue.put_nowait({
                        "event": "phase_complete",
                        "phase": "run_error",
                        "timestamp": time.time(),
                        "summary": {"error": "Training run failed"},
                    })
                except thread_queue.Full:
                    pass

        self.training_thread = threading.Thread(
            target=_run_in_thread, daemon=True, name="training-run",
        )
        self.training_thread.start()

    # ── Event bridge ────────────────────────────────────────────────

    async def _bridge_events(self) -> None:
        """Drain the thread-safe queue → broadcast to WS clients + terminal."""
        while True:
            try:
                event = await asyncio.to_thread(
                    self.progress_queue.get, timeout=1.0,
                )
            except Exception:
                # Timeout — send keepalive if running
                if self.running and self.training_thread and not self.training_thread.is_alive():
                    # Thread died without a terminal event
                    self.running = False
                    self.latest_process = None
                    self.latest_item = None
                    console.print("[red]Training thread exited unexpectedly[/red]")
                continue

            # Update local state
            self.latest_event = event

            if event.get("process"):
                self.latest_process = event["process"]
            if event.get("item"):
                self.latest_item = event["item"]

            is_terminal = False
            if event.get("event") == "phase_start":
                self.running = True
                self.latest_item = None
            elif (
                event.get("event") == "run_complete"
                or (
                    event.get("event") == "phase_complete"
                    and event.get("phase") in (
                        "run_complete", "run_stopped", "run_error",
                    )
                )
            ):
                self.running = False
                self.latest_process = None
                self.latest_item = None
                is_terminal = True

            # Broadcast to WS clients
            msg = make_event_msg(event)
            await self._broadcast(msg)

            # Print to terminal
            self._print_event(event, is_terminal)

    async def _broadcast(self, msg: str) -> None:
        dead: set = set()
        for client in self.clients:
            try:
                await client.send(msg)
            except Exception:
                dead.add(client)
        self.clients -= dead

    # Keepalive is handled by websockets library's built-in ping/pong
    # (ping_interval=30 on the server). No application-level pings needed.

    # ── Rich terminal output ────────────────────────────────────────

    def _print_event(self, event: dict, is_terminal: bool) -> None:
        evt_type = event.get("event", "")
        phase = event.get("phase", "")
        detail = event.get("detail", "")
        summary = event.get("summary", {})

        if evt_type == "phase_start":
            gen = event.get("generation") or summary.get("generation", "")
            pop = summary.get("population_size", "")
            label = phase.replace("_", " ").title()
            parts = [f"[bold cyan]{label}[/bold cyan]"]
            if gen != "":
                parts.append(f"Generation {gen}")
            if pop:
                parts.append(f"{pop} agents")
            console.print()
            console.rule(" | ".join(parts))

        elif evt_type == "progress":
            proc = event.get("process", {})
            completed = proc.get("completed", 0)
            total = proc.get("total", 0)
            pct = proc.get("pct", 0)
            process_eta = proc.get("process_eta_human", "")
            item = event.get("item", {})
            item_completed = item.get("completed", 0) if item else 0
            item_total = item.get("total", 0) if item else 0

            # Build a compact progress line
            parts = []
            if total:
                parts.append(f"[{completed}/{total}]")
            if detail:
                parts.append(detail)
            if item and item_total:
                parts.append(f"(step {item_completed}/{item_total})")
            if process_eta:
                parts.append(f"[dim]ETA {process_eta}[/dim]")

            console.print("  ".join(parts))

        elif evt_type == "phase_complete":
            if phase == "run_complete":
                console.print()
                console.rule("[bold green]Training Complete[/bold green]")
            elif phase == "run_stopped":
                console.print()
                console.rule("[bold yellow]Training Stopped[/bold yellow]")
            elif phase == "run_error":
                console.print()
                err = summary.get("error", "Unknown error")
                console.rule(f"[bold red]Training Error: {err}[/bold red]")
            else:
                # Phase-level completion (scoring, selecting, breeding etc.)
                label = phase.replace("_", " ").title()
                if summary:
                    info = "  ".join(f"{k}={v}" for k, v in summary.items())
                    console.print(f"  [green]{label} complete[/green]  {info}")
                else:
                    console.print(f"  [green]{label} complete[/green]")

        elif evt_type == "run_complete":
            console.print()
            console.rule("[bold green]Training Complete[/bold green]")

    # ── Main serve loop ─────────────────────────────────────────────

    async def serve(self) -> None:
        bridge_task = asyncio.create_task(self._bridge_events())

        async with websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10,
        ):
            console.rule("[bold green]Training Worker[/bold green]")
            console.print(f"  Listening on [bold]ws://{self.host}:{self.port}[/bold]")
            console.print(f"  Press Ctrl+C to stop")
            console.print()

            try:
                await asyncio.Future()  # run forever
            except asyncio.CancelledError:
                pass
            finally:
                bridge_task.cancel()


# ── Entry point ─────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="rl-betfair training worker")
    parser.add_argument("--port", type=int, default=None, help="WebSocket port")
    parser.add_argument("--host", type=str, default=None, help="Bind host")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    worker_cfg = config.get("training_worker", {})
    host = args.host or worker_cfg.get("host", DEFAULT_WORKER_HOST)
    port = args.port or worker_cfg.get("port", DEFAULT_WORKER_PORT)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s",
    )
    # Suppress noisy websockets library logging (logs every frame/connection)
    logging.getLogger("websockets").setLevel(logging.WARNING)

    worker = TrainingWorker(config=config, host=host, port=port)

    try:
        asyncio.run(worker.serve())
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")


if __name__ == "__main__":
    main()
