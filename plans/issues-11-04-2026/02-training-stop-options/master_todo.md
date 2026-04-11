# Master TODO — Training Stop Options

Ordered session list. Tick boxes as sessions land.

When a session completes:
1. Tick its box here.
2. Add an entry to `progress.md`.
3. Append any learnings to `lessons_learnt.md`.

---

## Phase 1 — Backend granularity

- [ ] **Session 01 — IPC + worker: stop granularity parameter**

  Extend the stop command to carry a `granularity` field.

  - `training/ipc.py` — add `STOP_EVAL_ALL`, `STOP_EVAL_CURRENT`,
    `STOP_IMMEDIATE` constants.  Update `make_stop_cmd()` to accept
    a `granularity` parameter.
  - `training/worker.py` — dispatch on granularity:
    - `immediate` → sets `stop_event` (current behaviour).
    - `eval_current` → sets new `stop_after_current_eval_event`.
    - `eval_all` → sets `finish_event` + `skip_training_event`.
  - Add `skip_training_event` and `stop_after_current_eval_event`
    threading.Events to the worker, passed to the orchestrator.

  **Tests:**
  - Worker receives stop with granularity=immediate → stop_event set.
  - Worker receives stop with granularity=eval_all → finish_event
    + skip_training_event set.
  - Worker receives stop with granularity=eval_current →
    stop_after_current_eval_event set.

- [ ] **Session 02 — Orchestrator: handle new events**

  `training/run_training.py` — add checks for the new events:

  - `skip_training_event`: if set during training phase, skip
    remaining agent training and jump to evaluation.  If already
    in eval phase, no effect.
  - `stop_after_current_eval_event`: if set during eval phase,
    complete the current agent's evaluation, then stop.  Don't
    evaluate remaining agents.  If set during training, skip
    training and don't evaluate.
  - Emit appropriate phase events so the frontend knows what's
    happening (e.g. "Skipping to evaluation...",
    "Finishing current evaluation...").

  **Tests:**
  - Orchestrator in training phase + skip_training → jumps to eval.
  - Orchestrator in eval phase + stop_after_current_eval → finishes
    current agent then stops.
  - Orchestrator in eval phase + skip_training (already past
    training) → no effect, eval continues normally.

## Phase 2 — API + frontend

- [ ] **Session 03 — API endpoint: granularity parameter**

  - `api/routers/training.py` — update `POST /training/stop` to
    accept `granularity` query param (default: `"immediate"` for
    backward compat).
  - Remove or keep separate `POST /training/finish` — this becomes
    equivalent to `stop?granularity=eval_all`.  Recommend keeping
    it as an alias for backward compat.
  - Add `unevaluated_count` and `eval_rate_s_per_model` to the
    status endpoint response so the frontend can show time estimates.

  **Tests:**
  - API stop with granularity=eval_all → correct IPC command sent.
  - API stop without granularity → defaults to immediate.
  - Status endpoint returns unevaluated_count.

- [ ] **Session 04 — Frontend: stop dialog**

  Replace the current "Stop Training" button with a button that
  opens a dialog offering three choices:

  - "Evaluate all generated models" (eval_all)
  - "Finish current evaluation only" (eval_current)
  - "Stop immediately" (immediate)

  Each option shows an estimated time:
  - eval_all: `unevaluated_count × avg_eval_time`
  - eval_current: remaining % of current eval × avg_eval_time
  - immediate: "< 10 seconds"

  "Finish Up" button remains as-is (it maps to eval_all, the
  current finish behaviour).

  - `training-monitor.html` — dialog template with radio options.
  - `training-monitor.ts` — dialog state, time estimate computation,
    API call with selected granularity.
  - `training-monitor.scss` — dialog styles.

  **Tests:**
  - Dialog opens on Stop Training click.
  - Each option calls API with correct granularity.
  - Time estimates update from WebSocket status.
  - Escalation: after choosing eval_all, user can re-open dialog
    and escalate to immediate.

---

## Summary

| Session | What | Phase |
|---------|------|-------|
| 01 | IPC + worker granularity | 1 |
| 02 | Orchestrator new event handling | 1 |
| 03 | API endpoint + status additions | 2 |
| 04 | Frontend stop dialog | 2 |

Total: 4 sessions. Sessions 01-02 are backend-only. Sessions 03-04
deliver the user-facing feature.
