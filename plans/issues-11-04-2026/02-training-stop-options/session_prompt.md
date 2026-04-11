# Training Stop Options — All Sessions (01–04)

Work through sessions sequentially. Complete each session fully
(code + tests + progress.md entry) before moving to the next.
Commit after each session.

## Before you start — read these

- `plans/issues-11-04-2026/02-training-stop-options/purpose.md` —
  why this work exists and the proposed dialog UX.
- `plans/issues-11-04-2026/02-training-stop-options/hard_constraints.md`
  — non-negotiables.
- `plans/issues-11-04-2026/02-training-stop-options/master_todo.md` —
  session breakdown with tests.

Also read the existing stop/finish infrastructure:
- `training/ipc.py` — current command types and message builders.
- `training/worker.py` — how `stop_event` and `finish_event` are
  set and passed to the orchestrator.
- `training/run_training.py` — `_check_stop()` and `_check_finish()`
  methods, and where they're called in the generation loop.
- `frontend/src/app/training-monitor/` — current button layout and
  WebSocket event handling.

---

## Session 01 — IPC + worker: stop granularity parameter

### Context

`training/ipc.py` defines `CMD_STOP = "stop"` and `CMD_FINISH = "finish"`.
`make_stop_cmd()` creates `{"type": "stop"}` with no parameters.
The worker sets `self.stop_event` on CMD_STOP and `self.finish_event`
on CMD_FINISH.

### What to do

1. In `training/ipc.py`:
   - Add constants: `STOP_EVAL_ALL = "eval_all"`,
     `STOP_EVAL_CURRENT = "eval_current"`,
     `STOP_IMMEDIATE = "immediate"`.
   - Update `make_stop_cmd(granularity: str = "immediate")` to
     include `{"type": "stop", "granularity": granularity}`.

2. In `training/worker.py`:
   - Add two new `threading.Event` objects:
     `self.skip_training_event = threading.Event()`
     `self.stop_after_current_eval_event = threading.Event()`
   - Dispatch on `CMD_STOP` granularity:
     - `immediate` → `self.stop_event.set()` (as now).
     - `eval_current` → `self.stop_after_current_eval_event.set()`.
     - `eval_all` → `self.finish_event.set()` +
       `self.skip_training_event.set()`.
   - Pass the new events to `TrainingOrchestrator.__init__()`.

3. Clear the new events alongside existing ones when starting a
   new training run.

### Tests

- Unit test: worker receives stop with each granularity → correct
  events are set.
- Unit test: stop without granularity defaults to immediate.

### Exit criteria

- All tests pass. `progress.md` updated. Commit.

---

## Session 02 — Orchestrator: handle new events

### Context

`training/run_training.py` checks `self._stop_event.is_set()` at
lines ~299 and ~514 (between generations and between agents).
`self._finish_event.is_set()` is checked at lines ~303-309 (between
generations — skips remaining gens, runs eval on current pop).

### What to do

1. Accept `skip_training_event` and `stop_after_current_eval_event`
   in `__init__()`.

2. **`skip_training_event` handling:**
   - In `_run_generation()`, before the agent training loop: if
     `skip_training_event.is_set()`, skip the training phase
     entirely and go straight to evaluation.
   - Emit a progress event: `"Skipping training — evaluating
     existing models..."`.

3. **`stop_after_current_eval_event` handling:**
   - In the evaluation loop (where agents are evaluated one by one):
     after completing each agent's evaluation, check
     `stop_after_current_eval_event.is_set()`.
   - If set, break out of the eval loop.  The current agent's
     results are already written.
   - If set during the training phase (before eval starts): treat
     as immediate stop (nothing to "finish current eval" on).
   - Emit a progress event: `"Stopping after current evaluation..."`.

4. **Escalation:** If `stop_event` is set at any point, it overrides
   everything (including mid-eval_all or mid-eval_current).

### Tests

- Orchestrator in training phase + skip_training_event → jumps to
  eval loop.
- Orchestrator in eval phase + stop_after_current_eval_event →
  completes current agent, stops.
- Orchestrator mid-eval_all + stop_event → immediate stop (override).
- Full run with no events → unchanged behaviour (regression).

### Exit criteria

- All tests pass. `progress.md` updated. Commit.

---

## Session 03 — API endpoint: granularity parameter

### Context

`api/routers/training.py` has `POST /training/stop` (line ~334) and
`POST /training/finish` (line ~355).  The status endpoint returns
phase, progress bars, and detail text.

### What to do

1. Update `POST /training/stop` to accept `granularity` query param:
   ```python
   @router.post("/stop")
   def stop_training(
       request: Request,
       granularity: str = "immediate",
   ):
   ```
   Validate granularity is one of: `immediate`, `eval_current`,
   `eval_all`.

2. Keep `POST /training/finish` as an alias for
   `stop?granularity=eval_all`.

3. Add to the status response:
   - `unevaluated_count: int` — models not yet evaluated in the
     current generation.
   - `eval_rate_s: float | None` — average seconds per model eval
     (computed from completed evals in this run).

   These may require the orchestrator to emit additional progress
   data, or the worker to track eval timing.

### Tests

- API stop with granularity=eval_all → correct IPC command sent.
- API stop without granularity → defaults to immediate.
- API stop with invalid granularity → 422 error.
- Status endpoint includes unevaluated_count.

### Exit criteria

- All tests pass. `progress.md` updated. Commit.

---

## Session 04 — Frontend: stop dialog

### Context

`training-monitor.html` has "Finish Up" (line ~79) and
"Stop Training" (line ~85) buttons. The component tracks
`isStopping` and `isFinishing` signals.

### What to do

1. Replace the "Stop Training" button with one that opens a dialog.
   Keep "Finish Up" as-is.

2. Dialog template (see `purpose.md` for the sketch):
   - Three radio options with descriptions and time estimates.
   - Cancel and Confirm buttons.
   - Time estimates computed from status endpoint data:
     - eval_all: `unevaluated_count × eval_rate_s`
     - eval_current: `(1 - current_progress) × eval_rate_s`
     - immediate: "< 10 seconds"

3. On Confirm, call `api.stopTraining(granularity)`.

4. After confirming, allow escalation: if the user clicks
   "Stop Training" again while a non-immediate stop is in progress,
   re-open the dialog with the option to escalate.

5. Style the dialog to match the existing dark theme.  Reference
   the stake breakdown dialog already in the codebase for patterns
   (`race-replay` or `recommendations` pages).

### Tests

- Dialog opens on Stop Training click.
- Cancel closes dialog without sending command.
- Each option sends correct granularity to API.
- Time estimates render (even if approximate).
- Escalation: after choosing eval_all, re-opening dialog shows
  eval_current and immediate only (eval_all already in progress).

### Exit criteria

- All tests pass. `progress.md` updated. Commit.

---

## Cross-session rules

- Run `pytest tests/ -x` after each session. All tests must pass.
- Do not touch training logic, scoring, or evaluation correctness.
- Do not "improve" unrelated code. Scope is tight.
- Commit after each session with a clear message referencing the
  session number.
