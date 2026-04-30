"""Websocket-event adapter — Phase 3, Session 04 deliverable.

Translates v2 trainer / worker / runner events into the **exact event
shape** that v1's frontend already consumes via
``frontend/src/app/services/training.service.ts`` +
``frontend/src/app/models/training.model.ts``::WSEvent.

Phase 3 hard constraint §3 (rewrite README §"Out of scope"; phase-3
purpose §"Hard constraints" §3): **no frontend code changes.** All
schema work happens here. Fields v1's UI doesn't render are dropped
(see "drop the extras" note below); fields v2 doesn't yet have are
left absent (the frontend already tolerates absence per its
``Partial<...>`` typing).

The adapter is pure event-construction. The runner/worker call the
factories and route the results to a caller-supplied
``Callable[[dict], None]`` emitter. Two emitter implementations live in
this file:

- ``QueueEventEmitter``: wraps a ``queue.Queue`` (or anything with a
  ``put_nowait`` method — including v1's ``_AsyncBridgeQueue``). This
  is the integration point if the v2 cohort runs **inside** v1's
  training-worker process (``python -m training.worker`` proxies the
  queue onto its websocket broadcast pipeline).
- ``WebSocketBroadcastServer``: a self-contained mini-broadcaster that
  binds the same ``localhost:8002`` endpoint the v1 worker uses, so the
  api / frontend connection chain works unchanged. Used by the
  ``--emit-websocket`` CLI flag on ``training_v2.cohort.runner`` when
  the operator launches the cohort directly (without the v1 worker).
  Mutually exclusive with a running v1 worker (port collision).

Schema reference (read-not-imported per Phase 3 §3):

- ``training/run_training.py::TrainingOrchestrator._emit_phase_start /
  _emit_phase_complete / _emit_info / _publish_progress`` — call sites
  + payload shapes.
- ``training/progress_tracker.py::ProgressTracker.to_dict`` — the
  ``process``/``item``/``overall`` snapshot shape.
- ``frontend/src/app/models/training.model.ts::WSEvent`` — the
  authoritative shape the UI consumes.

The frontend's ``training.service.ts::extractChartData`` parses the
``detail`` string with the regexes ``reward=([+-]?[\\d.]+)`` and
``loss=([\\d.]+)``. ``episode_complete_event`` formats its detail
string to match these so the per-episode reward / loss charts populate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol


logger = logging.getLogger(__name__)


__all__ = [
    "CohortEvent",
    "EventEmitter",
    "QueueEventEmitter",
    "WebSocketBroadcastServer",
    "make_progress_snapshot",
    "cohort_started_event",
    "agent_training_started_event",
    "episode_complete_event",
    "agent_training_complete_event",
    "cohort_complete_event",
    "info_event",
    "phase_start_evaluating_event",
    "phase_complete_evaluating_event",
]


# Event names — names are v1's, NOT the abstract names in the session
# prompt. The frontend's WSEvent.event field is checked against these
# string literals (training.service.ts::updateStatusFromEvent).
EVENT_PHASE_START = "phase_start"
EVENT_PHASE_COMPLETE = "phase_complete"
EVENT_PROGRESS = "progress"

# Phase names — string literals the UI conditions on
# (training.service.ts::updateStatusFromEvent::isRunEnd).
PHASE_TRAINING = "training"
PHASE_EVALUATING = "evaluating"
PHASE_RUN_COMPLETE = "run_complete"
PHASE_INFO = "info"


# ── Event-dict typing ─────────────────────────────────────────────────────


CohortEvent = dict[str, Any]
"""A v1-shape websocket event dict.

Required keys: ``event``, ``timestamp``. Optional: ``phase``,
``summary``, ``process``, ``item``, ``overall``, ``detail``,
``generation``, ``last_agent_score``. The shape mirrors
``frontend/src/app/models/training.model.ts::WSEvent``.
"""


# ── Emitter abstractions ─────────────────────────────────────────────────


class EventEmitter(Protocol):
    """Anything callable as ``emit(event_dict)``.

    The runner / worker take ``Callable[[dict], None] | None``; this
    Protocol is documentation only — concrete callers use a plain
    function or one of the helper classes below.
    """

    def __call__(self, event: CohortEvent) -> None: ...


class QueueEventEmitter:
    """Push events onto a queue-like object.

    Compatible with both ``queue.Queue`` (thread-safe) and v1's
    ``training/worker.py::_AsyncBridgeQueue`` (asyncio-bridged). The
    only contract is a non-blocking ``put_nowait(event)`` method that
    accepts a plain dict.

    Drops the event if the queue's ``put_nowait`` raises (matches v1's
    ``_put_event`` swallow-on-Full pattern — back-pressure on a UI
    queue should never crash training).
    """

    def __init__(self, queue: Any) -> None:
        if not hasattr(queue, "put_nowait"):
            raise TypeError(
                f"QueueEventEmitter requires a put_nowait()-capable object, "
                f"got {type(queue).__name__}"
            )
        self._queue = queue

    def __call__(self, event: CohortEvent) -> None:
        try:
            self._queue.put_nowait(event)
        except Exception:
            # Drop on backpressure, same as v1's _put_event. Training
            # rolls on; a slow consumer just sees a gap in the UI.
            pass


# ── ProgressSnapshot helper ──────────────────────────────────────────────


def _fmt_eta(seconds: float | None) -> str:
    """Match the human-readable ETA format from
    ``training/progress_tracker.py::_fmt`` so the UI columns line up.
    """
    if seconds is None or seconds < 0 or math.isnan(seconds):
        return "—"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    hours = seconds // 3600
    remainder = seconds - hours * 3600
    minutes = remainder // 60
    return f"{hours}h {minutes}m" if minutes else f"{hours}h"


def make_progress_snapshot(
    *,
    label: str,
    completed: int,
    total: int,
    item_eta_s: float | None = None,
    process_eta_s: float | None = None,
) -> dict[str, Any]:
    """Build a v1-shape ProgressTracker snapshot dict.

    Mirrors ``training/progress_tracker.py::ProgressTracker.to_dict``.
    The frontend reads ``label``, ``completed``, ``total``, ``pct``,
    ``item_eta_human``, ``process_eta_human`` (model: ProgressSnapshot
    in training.model.ts).
    """
    if total <= 0:
        pct = 0.0
    else:
        pct = round(min(completed, total) / total * 100, 1)
    return {
        "label": str(label),
        "completed": int(completed),
        "total": int(total),
        "pct": float(pct),
        "item_eta_s": item_eta_s,
        "process_eta_s": process_eta_s,
        "item_eta_human": _fmt_eta(item_eta_s),
        "process_eta_human": _fmt_eta(process_eta_s),
    }


# ── Event factories ──────────────────────────────────────────────────────


def _now() -> float:
    """Unix-seconds timestamp; matches v1's ``time.time()`` usage."""
    return time.time()


def cohort_started_event(
    *,
    run_id: str,
    n_generations: int,
    n_agents: int,
    train_days: list[str],
    eval_day: str,
    seed: int,
) -> CohortEvent:
    """Run-start event. v1 emits this as ``phase_start "training"``
    with a ``summary`` carrying run-level config. The frontend treats
    any non-end ``phase_start`` as "the run is live" via
    ``setRunning(true)`` (training.service.ts::updateStatusFromEvent).
    """
    return {
        "event": EVENT_PHASE_START,
        "phase": PHASE_TRAINING,
        "timestamp": _now(),
        "summary": {
            "run_id": str(run_id),
            "n_generations": int(n_generations),
            "n_agents": int(n_agents),
            "train_days": list(train_days),
            "eval_day": str(eval_day),
            # v1 emits ``test_days`` as a count (line 376 of
            # run_training.py); we follow the same shape so the UI's
            # "X test days" label string formats unchanged.
            "test_days": 1,
            "seed": int(seed),
        },
    }


def agent_training_started_event(
    *,
    agent_id: str,
    architecture_name: str,
    generation: int,
    agent_idx: int,
    n_agents: int,
    genes: Mapping[str, Any] | None = None,
    item_eta_s: float | None = None,
    process_eta_s: float | None = None,
) -> CohortEvent:
    """Per-agent kick-off. Matches v1's
    ``run_training.py::_publish_progress("training", outer_tracker,
    detail=f"Training agent ...")`` shape.

    The ``process`` snapshot is the per-generation outer tracker
    (one tick per agent completed within this generation).
    """
    detail_genes = ""
    if genes:
        # Format a one-line gene summary for the activity log. Keep
        # ASCII / no Unicode so the frontend ActivityLog renders cleanly
        # in any terminal.
        detail_genes = " | " + ", ".join(
            f"{k}={_fmt_gene_val(v)}" for k, v in sorted(genes.items())
        )
    detail = (
        f"Training agent {agent_id[:12]} ({architecture_name})"
        f"{detail_genes}"
    )
    return {
        "event": EVENT_PROGRESS,
        "phase": PHASE_TRAINING,
        "timestamp": _now(),
        "generation": int(generation),
        "process": make_progress_snapshot(
            label=f"Generation {generation} — training {n_agents} agents",
            completed=int(agent_idx),
            total=int(n_agents),
            item_eta_s=item_eta_s,
            process_eta_s=process_eta_s,
        ),
        "detail": detail,
    }


def _fmt_gene_val(val: Any) -> str:
    """Compact gene value for the activity-log detail string."""
    if isinstance(val, float):
        if abs(val) < 1e-2 or abs(val) >= 1e3:
            return f"{val:.2e}"
        return f"{val:.3f}"
    return str(val)


def episode_complete_event(
    *,
    agent_id: str,
    architecture_name: str,
    generation: int,
    day_idx: int,
    n_days: int,
    day_str: str,
    episode_idx: int,
    total_reward: float,
    day_pnl: float,
    value_loss_mean: float,
    policy_loss_mean: float,
    approx_kl_mean: float,
    n_steps: int,
    bet_count: int | None = None,
    force_close_count: int | None = None,
) -> CohortEvent:
    """Per-(agent × day) episode completion.

    Detail string is formatted to match the regex the frontend uses to
    extract chart data (``training.service.ts::extractChartData``)::

        reward=([+-]?[\\d.]+)   →  rewardHistory chart
        loss=([\\d.]+)          →  lossHistory chart

    so the existing per-episode chart streams reward + value_loss
    without any frontend change. Anything beyond the regex hooks is
    informational only.
    """
    bits = [
        f"Episode {episode_idx} [{day_str}]",
        f"Day {day_idx + 1}/{n_days}",
        f"reward={total_reward:+.3f}",
        # The "loss" the regex picks up is the value loss — the most
        # informative single scalar of training health on PPO. Policy
        # loss is also surfaced for the activity log but won't hit the
        # chart (the regex matches the FIRST loss=X).
        f"loss={value_loss_mean:.4f}",
        f"P&L=£{day_pnl:+.2f}",
        f"steps={n_steps}",
        f"kl={approx_kl_mean:.4f}",
        f"policy_loss={policy_loss_mean:.4f}",
    ]
    if bet_count is not None:
        bits.append(f"bets={bet_count}")
    if force_close_count is not None:
        bits.append(f"fc={force_close_count}")
    bits.append(f"agent={agent_id[:12]} {architecture_name}")
    detail = " | ".join(bits)
    return {
        "event": EVENT_PROGRESS,
        "phase": PHASE_TRAINING,
        "timestamp": _now(),
        "generation": int(generation),
        "detail": detail,
        # ``last_agent_score`` is what v1 puts on the orchestrator-side
        # ``progress`` events to feed the right-side score widget. We
        # use total_reward as the per-episode signal — a Phase 3 agent's
        # eval-day score lands later via agent_training_complete.
        "last_agent_score": float(total_reward),
    }


def agent_training_complete_event(
    *,
    agent_id: str,
    architecture_name: str,
    generation: int,
    agent_idx: int,
    n_agents: int,
    eval_total_reward: float,
    eval_day_pnl: float,
    eval_bet_count: int,
    eval_bet_precision: float,
    item_eta_s: float | None = None,
    process_eta_s: float | None = None,
) -> CohortEvent:
    """Per-agent completion: train + eval done, weights/scoreboard
    written, advance the outer tracker by one tick.

    Uses the same ``progress`` shape as the kick-off event so the UI's
    progress bar advances naturally; the score widget refreshes via
    ``last_agent_score``.
    """
    detail = (
        f"Agent {agent_id[:12]} ({architecture_name}) eval reward="
        f"{eval_total_reward:+.3f} | P&L=£{eval_day_pnl:+.2f} | "
        f"bets={eval_bet_count} precision={eval_bet_precision:.3f}"
    )
    return {
        "event": EVENT_PROGRESS,
        "phase": PHASE_TRAINING,
        "timestamp": _now(),
        "generation": int(generation),
        "process": make_progress_snapshot(
            label=f"Generation {generation} — training {n_agents} agents",
            completed=int(agent_idx) + 1,
            total=int(n_agents),
            item_eta_s=item_eta_s,
            process_eta_s=process_eta_s,
        ),
        "detail": detail,
        "last_agent_score": float(eval_total_reward),
    }


def phase_start_evaluating_event(
    *,
    generation: int,
    n_agents: int,
) -> CohortEvent:
    """Mirror v1's ``_emit_phase_start("evaluating", ...)`` for the
    eval-day rollout pass. v2 doesn't run a separate evaluating phase
    (eval is folded into the worker) — but the UI's phase header
    benefits from seeing the explicit transition. Optional; the
    runner only emits when the eval rollout is meaningful work.
    """
    return {
        "event": EVENT_PHASE_START,
        "phase": PHASE_EVALUATING,
        "timestamp": _now(),
        "generation": int(generation),
        "summary": {
            "generation": int(generation),
            "agent_count": int(n_agents),
            "test_days": 1,
        },
    }


def phase_complete_evaluating_event(
    *,
    generation: int,
    agents_evaluated: int,
) -> CohortEvent:
    """Counterpart to :func:`phase_start_evaluating_event`."""
    return {
        "event": EVENT_PHASE_COMPLETE,
        "phase": PHASE_EVALUATING,
        "timestamp": _now(),
        "generation": int(generation),
        "summary": {
            "generation": int(generation),
            "agents_evaluated": int(agents_evaluated),
        },
    }


def cohort_complete_event(
    *,
    run_id: str,
    status: str,
    n_generations: int,
    total_agents_trained: int,
    total_agents_evaluated: int,
    wall_time_seconds: float,
    best_model: dict[str, Any] | None,
    top_5: list[dict[str, Any]],
    error_message: str | None = None,
) -> CohortEvent:
    """Run-end event. The frontend treats this as the canonical
    "training finished" signal (training.service.ts::isRunEnd checks
    for ``phase_complete`` with phase ∈ {run_complete, run_stopped,
    run_error, extracting}). The summary shape matches
    ``RunCompleteSummary`` in training.model.ts so the run-summary
    panel renders without any frontend change.
    """
    if status not in {"completed", "stopped", "error"}:
        raise ValueError(
            f"cohort_complete status must be one of "
            f"completed/stopped/error, got {status!r}"
        )
    summary: dict[str, Any] = {
        "run_id": str(run_id),
        "status": str(status),
        "generations_completed": int(n_generations),
        "generations_requested": int(n_generations),
        "total_agents_trained": int(total_agents_trained),
        "total_agents_evaluated": int(total_agents_evaluated),
        "wall_time_seconds": float(wall_time_seconds),
        "best_model": best_model,
        "top_5": list(top_5),
        "population_summary": {
            # Phase 3 first cohort doesn't track survival/discard
            # — every agent gets evaluated, no garaging. Zeroes are
            # the honest answer; the UI tolerates them.
            "survived": 0,
            "discarded": 0,
            "garaged": 0,
        },
        "error_message": error_message,
    }
    return {
        "event": EVENT_PHASE_COMPLETE,
        "phase": PHASE_RUN_COMPLETE,
        "timestamp": _now(),
        "summary": summary,
    }


def info_event(message: str) -> CohortEvent:
    """Lightweight info-log event. Matches v1's
    ``run_training.py::_emit_info`` shape; populates the activity log."""
    return {
        "event": EVENT_PROGRESS,
        "phase": PHASE_INFO,
        "timestamp": _now(),
        "detail": str(message),
    }


# ── Optional self-hosted broadcaster ──────────────────────────────────────


class WebSocketBroadcastServer:
    """Minimal websocket server that broadcasts cohort events to all
    connected clients.

    Wraps the existing v1 ``ipc.make_event_msg`` envelope shape::

        {"type": "event", "payload": <event_dict>}

    so the API's ``_worker_connection`` task in ``api/main.py`` parses
    the messages without any change. The api connects as a CLIENT to
    ``ws://{host}:{port}`` (the v1 ``training_worker`` host/port from
    config.yaml — default ``localhost:8002``).

    **Operator constraint.** Mutually exclusive with a running v1
    ``python -m training.worker`` process — both bind the same port.
    The 12-agent cohort run takes the v1 worker offline for the
    duration; relaunch v1 worker after the cohort completes.

    Threading model: the server runs its own asyncio event loop in a
    background thread. Cohort events arrive from the runner's main
    thread via ``emit(event)``; the server schedules a broadcast on
    its loop using ``call_soon_threadsafe``.
    """

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 8002,
        ping_interval: float = 20.0,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.ping_interval = float(ping_interval)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server: Any = None  # websockets.Server
        self._clients: set[Any] = set()
        self._started = threading.Event()
        self._latest_event: CohortEvent | None = None

    # -- Lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Start the server thread; blocks until the listen socket is open."""
        if self._thread is not None:
            raise RuntimeError("WebSocketBroadcastServer already started")
        self._thread = threading.Thread(
            target=self._run_loop,
            name="v2-cohort-ws",
            daemon=True,
        )
        self._thread.start()
        if not self._started.wait(timeout=10.0):
            raise RuntimeError(
                f"WebSocketBroadcastServer failed to bind {self.host}:{self.port} "
                f"within 10s — is a v1 training worker still running?"
            )
        logger.info(
            "WebSocketBroadcastServer listening on ws://%s:%d",
            self.host, self.port,
        )

    def stop(self, *, timeout: float = 5.0) -> None:
        """Shut down the server thread cleanly."""
        if self._thread is None:
            return
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._shutdown_request)
        self._thread.join(timeout=timeout)
        self._thread = None
        self._loop = None

    @contextmanager
    def running(self):
        """Context-manager helper for ``with server.running(): ...``."""
        self.start()
        try:
            yield self
        finally:
            self.stop()

    # -- Public emit ------------------------------------------------------

    def __call__(self, event: CohortEvent) -> None:
        """Emitter callable: schedule a broadcast on the server loop."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        self._latest_event = event
        loop.call_soon_threadsafe(self._broadcast, event)

    @property
    def latest_event(self) -> CohortEvent | None:
        return self._latest_event

    # -- Internals (run on the asyncio thread) ---------------------------

    def _run_loop(self) -> None:
        try:
            import websockets  # local import — websockets is heavy
        except ImportError as exc:
            logger.error(
                "WebSocketBroadcastServer needs the `websockets` package: %s", exc
            )
            self._started.set()  # unblock start(); caller will raise
            return

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._serve(websockets))
        except Exception:
            logger.exception("WebSocketBroadcastServer event loop crashed")
        finally:
            try:
                loop.close()
            except Exception:
                pass

    async def _serve(self, websockets_mod: Any) -> None:
        async def _handler(ws):
            self._clients.add(ws)
            try:
                # Send a synthetic "running=true" status on connect so
                # mid-run clients catch up immediately. Mirrors
                # ``training/worker.py::_handle_connection``'s behaviour.
                if self._latest_event is not None:
                    await ws.send(_envelope(self._latest_event))
                async for _msg in ws:
                    # The v2 broadcaster doesn't accept commands —
                    # commands belong to v1's worker. Drop client
                    # messages quietly so the connection stays alive.
                    pass
            except Exception:
                pass
            finally:
                self._clients.discard(ws)

        self._server = await websockets_mod.serve(
            _handler, self.host, self.port,
            ping_interval=self.ping_interval,
        )
        self._started.set()
        # Wait for shutdown_request; closing self._server returns control
        try:
            await self._server.wait_closed()
        except Exception:
            pass

    def _broadcast(self, event: CohortEvent) -> None:
        if not self._clients:
            return
        envelope = _envelope(event)
        for client in list(self._clients):
            try:
                # Schedule each client send. Errors close the client.
                asyncio.ensure_future(_safe_send(client, envelope))
            except Exception:
                self._clients.discard(client)

    def _shutdown_request(self) -> None:
        server = self._server
        if server is not None:
            try:
                server.close()
            except Exception:
                pass


async def _safe_send(client: Any, envelope: str) -> None:
    try:
        await client.send(envelope)
    except Exception:
        try:
            await client.close()
        except Exception:
            pass


def _envelope(event: CohortEvent) -> str:
    """Wrap an event dict in v1's ipc envelope so the api parses it
    via ``parse_message`` → ``msg_type == EVT_EVENT``.

    Mirrors ``training/ipc.py::make_event_msg`` exactly. We don't import
    ipc to honour the rewrite hard-constraint §3 (no v1 imports from
    v2). The string literal ``"event"`` is the constant ``EVT_EVENT``.
    """
    return json.dumps({"type": "event", "payload": event})


# ── Smoke-test of the schema match (run as script for sanity) ────────────


if __name__ == "__main__":  # pragma: no cover
    # Print one of each event so the operator can eyeball the shape
    # against the WSEvent interface in training.model.ts.
    import pprint
    pp = pprint.PrettyPrinter(indent=2, width=100)
    pp.pprint(cohort_started_event(
        run_id="r-xyz", n_generations=4, n_agents=12,
        train_days=["2026-04-01", "2026-04-02"],
        eval_day="2026-04-03", seed=42,
    ))
    pp.pprint(agent_training_started_event(
        agent_id="a" * 32, architecture_name="v2_discrete_ppo_lstm_h128",
        generation=0, agent_idx=2, n_agents=12,
        genes={"learning_rate": 1.2e-4, "hidden_size": 128},
    ))
    pp.pprint(episode_complete_event(
        agent_id="a" * 32,
        architecture_name="v2_discrete_ppo_lstm_h128",
        generation=0, day_idx=2, n_days=7, day_str="2026-04-02",
        episode_idx=2, total_reward=1.234, day_pnl=3.40,
        value_loss_mean=0.0042, policy_loss_mean=-0.012,
        approx_kl_mean=0.018, n_steps=11872, bet_count=42,
    ))
    pp.pprint(cohort_complete_event(
        run_id="r-xyz", status="completed", n_generations=4,
        total_agents_trained=48, total_agents_evaluated=48,
        wall_time_seconds=5040.0,
        best_model={
            "model_id": "best", "composite_score": 12.3,
            "total_pnl": 4.5, "win_rate": 0.55,
            "architecture": "v2_discrete_ppo_lstm_h128",
        },
        top_5=[],
    ))
