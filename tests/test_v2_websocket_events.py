"""Schema regression for the Phase 3 Session 04 cohort event adapter.

The frontend's authoritative shape is in
``frontend/src/app/models/training.model.ts::WSEvent``. The adapter in
``training_v2/cohort/events.py`` builds dicts that round-trip through
JSON and match that shape field-for-field. These tests pin the field
names + types so a refactor that drops or renames a key is caught
before it breaks the live UI.

The frontend's chart-population logic
(``frontend/src/app/services/training.service.ts::extractChartData``)
parses the ``detail`` string with two regexes. The ``episode_complete``
test asserts the regexes still match — that's the load-bearing
guarantee for the per-episode reward / loss live charts.

Tests do NOT spin up a websocket server (the broadcaster is integration-
tested by the live run; unit tests stay fast). The schema match is the
primary deliverable per the session prompt §"Tests".
"""

from __future__ import annotations

import json
import queue
import re

import pytest

from training_v2.cohort.events import (
    EVENT_PHASE_COMPLETE,
    EVENT_PHASE_START,
    EVENT_PROGRESS,
    PHASE_EVALUATING,
    PHASE_INFO,
    PHASE_RUN_COMPLETE,
    PHASE_TRAINING,
    QueueEventEmitter,
    agent_training_complete_event,
    agent_training_started_event,
    cohort_complete_event,
    cohort_started_event,
    episode_complete_event,
    info_event,
    make_progress_snapshot,
    phase_complete_evaluating_event,
    phase_start_evaluating_event,
)


# ── ProgressSnapshot ─────────────────────────────────────────────────


def test_progress_snapshot_matches_v1_shape():
    """Mirrors ``training/progress_tracker.py::ProgressTracker.to_dict``.

    Frontend reads label/completed/total/pct/item_eta_human/process_eta_human
    (training.model.ts::ProgressSnapshot interface). The other two _s
    fields are optional but harmless.
    """
    snap = make_progress_snapshot(
        label="Generation 0 — training 12 agents",
        completed=3,
        total=12,
        item_eta_s=42.0,
        process_eta_s=378.0,
    )
    expected_keys = {
        "label", "completed", "total", "pct",
        "item_eta_s", "process_eta_s",
        "item_eta_human", "process_eta_human",
    }
    assert set(snap.keys()) == expected_keys
    assert snap["pct"] == pytest.approx(25.0)
    assert snap["item_eta_human"] == "42s"
    assert snap["process_eta_human"] == "6m 18s"


def test_progress_snapshot_zero_total_does_not_div_zero():
    snap = make_progress_snapshot(
        label="empty", completed=0, total=0,
    )
    assert snap["pct"] == 0.0
    assert snap["item_eta_human"] == "—"


# ── cohort_started ───────────────────────────────────────────────────


def test_cohort_started_field_names_and_types():
    ev = cohort_started_event(
        run_id="r-xyz",
        n_generations=4,
        n_agents=12,
        train_days=["2026-04-01", "2026-04-02"],
        eval_day="2026-04-03",
        seed=42,
    )
    # WSEvent surface
    assert ev["event"] == EVENT_PHASE_START
    assert ev["phase"] == PHASE_TRAINING
    assert isinstance(ev["timestamp"], float)
    assert ev["timestamp"] > 0
    # summary content
    s = ev["summary"]
    assert s["run_id"] == "r-xyz"
    assert s["n_generations"] == 4
    assert s["n_agents"] == 12
    assert s["train_days"] == ["2026-04-01", "2026-04-02"]
    assert s["eval_day"] == "2026-04-03"
    assert s["seed"] == 42
    # ``test_days`` is a count (matches v1's run_training.py:376)
    assert isinstance(s["test_days"], int)


def test_cohort_started_round_trips_through_json():
    ev = cohort_started_event(
        run_id="r-xyz", n_generations=2, n_agents=4,
        train_days=["2026-04-01"], eval_day="2026-04-02", seed=7,
    )
    re_parsed = json.loads(json.dumps(ev))
    assert re_parsed == ev


# ── agent_training_started ───────────────────────────────────────────


def test_agent_training_started_progress_shape():
    """Matches v1's ``_publish_progress("training", ...)`` shape:
    ``event=progress, phase=training, process=<snapshot>, detail=<str>``."""
    ev = agent_training_started_event(
        agent_id="a" * 32,
        architecture_name="v2_discrete_ppo_lstm_h128",
        generation=0,
        agent_idx=2,
        n_agents=12,
        genes={"learning_rate": 1.2e-4, "hidden_size": 128},
    )
    assert ev["event"] == EVENT_PROGRESS
    assert ev["phase"] == PHASE_TRAINING
    assert ev["generation"] == 0
    assert "Training agent" in ev["detail"]
    assert "v2_discrete_ppo_lstm_h128" in ev["detail"]
    # Process snapshot has the v1 ProgressTracker.to_dict keys
    assert ev["process"]["completed"] == 2
    assert ev["process"]["total"] == 12


# ── episode_complete (load-bearing for the live chart) ───────────────


REWARD_RE = re.compile(r"reward=([+-]?[\d.]+)")
LOSS_RE = re.compile(r"loss=([\d.]+)")


def test_episode_complete_detail_matches_frontend_chart_regex():
    """The frontend's ``training.service.ts::extractChartData`` parses
    ``detail`` with these two regexes. If the format string ever drifts,
    the per-episode chart silently goes blank — this test pins it.

    See ``training.service.ts`` lines ~227 (``rewardMatch``) and
    ~228 (``lossMatch``).
    """
    ev = episode_complete_event(
        agent_id="a" * 32,
        architecture_name="v2_discrete_ppo_lstm_h128",
        generation=1,
        day_idx=2,
        n_days=7,
        day_str="2026-04-02",
        episode_idx=2,
        total_reward=1.234,
        day_pnl=3.40,
        value_loss_mean=0.0042,
        policy_loss_mean=-0.012,
        approx_kl_mean=0.018,
        n_steps=11872,
        bet_count=42,
    )
    detail = ev["detail"]
    rm = REWARD_RE.search(detail)
    lm = LOSS_RE.search(detail)
    assert rm is not None, f"reward regex didn't match {detail!r}"
    assert lm is not None, f"loss regex didn't match {detail!r}"
    assert float(rm.group(1)) == pytest.approx(1.234)
    # The first ``loss=`` in the string is the value loss.
    assert float(lm.group(1)) == pytest.approx(0.0042)


def test_episode_complete_round_trips_through_json():
    ev = episode_complete_event(
        agent_id="ag1", architecture_name="v2_discrete_ppo_lstm_h64",
        generation=0, day_idx=0, n_days=1, day_str="2026-04-25",
        episode_idx=0, total_reward=-0.5, day_pnl=-1.2,
        value_loss_mean=0.1, policy_loss_mean=0.0,
        approx_kl_mean=0.02, n_steps=100,
    )
    re_parsed = json.loads(json.dumps(ev))
    assert re_parsed == ev
    assert re_parsed["last_agent_score"] == pytest.approx(-0.5)


# ── agent_training_complete ──────────────────────────────────────────


def test_agent_training_complete_advances_progress_tracker():
    """The completed agent counts toward the outer tracker's
    ``completed`` field (one tick after the kick-off snapshot)."""
    ev = agent_training_complete_event(
        agent_id="ag1",
        architecture_name="v2_discrete_ppo_lstm_h64",
        generation=0,
        agent_idx=2,
        n_agents=12,
        eval_total_reward=2.5,
        eval_day_pnl=4.0,
        eval_bet_count=20,
        eval_bet_precision=0.6,
    )
    assert ev["event"] == EVENT_PROGRESS
    assert ev["phase"] == PHASE_TRAINING
    # agent_idx=2 → completed = idx + 1 = 3 (the agent JUST finished)
    assert ev["process"]["completed"] == 3
    assert ev["process"]["total"] == 12
    assert ev["last_agent_score"] == pytest.approx(2.5)


# ── phase_start/complete evaluating (optional event pair) ────────────


def test_phase_start_evaluating_event():
    ev = phase_start_evaluating_event(generation=1, n_agents=12)
    assert ev["event"] == EVENT_PHASE_START
    assert ev["phase"] == PHASE_EVALUATING
    assert ev["summary"]["agent_count"] == 12


def test_phase_complete_evaluating_event():
    ev = phase_complete_evaluating_event(generation=1, agents_evaluated=12)
    assert ev["event"] == EVENT_PHASE_COMPLETE
    assert ev["phase"] == PHASE_EVALUATING


# ── cohort_complete (terminal event) ─────────────────────────────────


def test_cohort_complete_terminal_phase_runs_run_complete():
    """The frontend ``isRunEnd`` check accepts ``phase_complete`` with
    phase ∈ {run_complete, run_stopped, run_error, extracting}. We use
    ``run_complete`` for normal termination so the UI flips
    ``running=false`` (training.service.ts line ~182)."""
    ev = cohort_complete_event(
        run_id="r-xyz",
        status="completed",
        n_generations=4,
        total_agents_trained=48,
        total_agents_evaluated=48,
        wall_time_seconds=5040.0,
        best_model={
            "model_id": "best",
            "composite_score": 12.3,
            "total_pnl": 4.5,
            "win_rate": 0.55,
            "architecture": "v2_discrete_ppo_lstm_h128",
        },
        top_5=[],
    )
    assert ev["event"] == EVENT_PHASE_COMPLETE
    assert ev["phase"] == PHASE_RUN_COMPLETE
    s = ev["summary"]
    # RunCompleteSummary in training.model.ts
    for k in (
        "run_id", "status", "generations_completed",
        "generations_requested", "total_agents_trained",
        "total_agents_evaluated", "wall_time_seconds",
        "best_model", "top_5", "population_summary",
    ):
        assert k in s, f"missing key {k!r} in summary"
    assert s["status"] == "completed"
    # population_summary always carries survived/discarded/garaged ints
    ps = s["population_summary"]
    for k in ("survived", "discarded", "garaged"):
        assert isinstance(ps[k], int)


def test_cohort_complete_round_trips_through_json():
    ev = cohort_complete_event(
        run_id="r-xyz", status="stopped", n_generations=2,
        total_agents_trained=8, total_agents_evaluated=8,
        wall_time_seconds=1234.5,
        best_model=None, top_5=[],
        error_message=None,
    )
    re_parsed = json.loads(json.dumps(ev))
    assert re_parsed == ev


def test_cohort_complete_rejects_invalid_status():
    with pytest.raises(ValueError):
        cohort_complete_event(
            run_id="r", status="frobbed",
            n_generations=1,
            total_agents_trained=0, total_agents_evaluated=0,
            wall_time_seconds=0.0,
            best_model=None, top_5=[],
        )


# ── info ─────────────────────────────────────────────────────────────


def test_info_event_shape():
    ev = info_event("Booting up the cohort.")
    assert ev["event"] == EVENT_PROGRESS
    assert ev["phase"] == PHASE_INFO
    assert ev["detail"] == "Booting up the cohort."


# ── QueueEventEmitter integration ────────────────────────────────────


def test_queue_event_emitter_pushes_dicts_onto_queue():
    q: queue.Queue = queue.Queue()
    emitter = QueueEventEmitter(q)
    ev = info_event("hello")
    emitter(ev)
    received = q.get_nowait()
    assert received is ev
    assert q.empty()


def test_queue_event_emitter_drops_on_backpressure():
    """v1's ``_put_event`` swallows Full silently (training rolls on,
    UI just sees a gap). The emitter does the same so a slow / full
    consumer can't crash the trainer."""
    q: queue.Queue = queue.Queue(maxsize=1)
    emitter = QueueEventEmitter(q)
    emitter(info_event("first"))
    # Queue is now full; the second call must NOT raise.
    emitter(info_event("second — dropped"))
    # First event still there; second silently dropped.
    assert q.qsize() == 1


def test_queue_event_emitter_rejects_non_queue_at_construction():
    """Misuse should fail loudly at construction, not silently at
    emit-time."""
    with pytest.raises(TypeError):
        QueueEventEmitter("not a queue")


# ── End-to-end: full event sequence through a queue ──────────────────


def test_full_cohort_lifecycle_through_queue_emitter():
    """Walk the full v1 lifecycle of events the runner would emit, push
    them through the queue emitter, and verify each round-trips through
    JSON without loss. This is the integration shape the api consumes
    via ``api/main.py::_worker_connection``."""
    q: queue.Queue = queue.Queue()
    emit = QueueEventEmitter(q)

    emit(cohort_started_event(
        run_id="r1", n_generations=2, n_agents=4,
        train_days=["2026-04-01"], eval_day="2026-04-02", seed=1,
    ))
    emit(agent_training_started_event(
        agent_id="ag-1", architecture_name="v2_discrete_ppo_lstm_h64",
        generation=0, agent_idx=0, n_agents=4,
        genes={"learning_rate": 1e-4},
    ))
    emit(episode_complete_event(
        agent_id="ag-1", architecture_name="v2_discrete_ppo_lstm_h64",
        generation=0, day_idx=0, n_days=1, day_str="2026-04-01",
        episode_idx=0, total_reward=0.1, day_pnl=0.5,
        value_loss_mean=0.01, policy_loss_mean=-0.001,
        approx_kl_mean=0.02, n_steps=500,
    ))
    emit(agent_training_complete_event(
        agent_id="ag-1", architecture_name="v2_discrete_ppo_lstm_h64",
        generation=0, agent_idx=0, n_agents=4,
        eval_total_reward=0.2, eval_day_pnl=0.6,
        eval_bet_count=10, eval_bet_precision=0.5,
    ))
    emit(cohort_complete_event(
        run_id="r1", status="completed", n_generations=2,
        total_agents_trained=8, total_agents_evaluated=8,
        wall_time_seconds=120.0,
        best_model=None, top_5=[],
    ))

    received = []
    while not q.empty():
        ev = q.get_nowait()
        # Each must serialise as JSON without raising.
        json.dumps(ev)
        received.append(ev)
    assert len(received) == 5
    # Order preserved.
    assert received[0]["event"] == EVENT_PHASE_START
    assert received[-1]["event"] == EVENT_PHASE_COMPLETE
    assert received[-1]["phase"] == PHASE_RUN_COMPLETE
