# Progress — Training Stop Options

One entry per completed session.

---

## Session 01 — IPC + worker: stop granularity parameter (2026-04-12)

- Added `STOP_IMMEDIATE`, `STOP_EVAL_CURRENT`, `STOP_EVAL_ALL` constants
  and `VALID_STOP_GRANULARITIES` set to `training/ipc.py`.
- Updated `make_stop_cmd()` to accept a `granularity` parameter
  (default: `"immediate"` for backward compat).
- Added `skip_training_event` and `stop_after_current_eval_event`
  threading.Events to `TrainingWorker`.
- Worker dispatches on granularity: `immediate` → stop_event,
  `eval_current` → stop_after_current_eval_event,
  `eval_all` → finish_event + skip_training_event.
- Events cleared on new run start.
- 6 new tests in `test_training_worker.py`.

## Session 02 — Orchestrator: handle new events (2026-04-12)

- `TrainingOrchestrator.__init__` accepts `skip_training_event` and
  `stop_after_current_eval_event`.
- Training loop checks `_check_skip_training()` — if set, breaks out
  of agent training and jumps to evaluation.
- Eval loop checks `_check_stop_after_current_eval()` after each
  agent completes — if set, breaks after the current eval.
- `stop_event` always overrides (escalation) during eval loop.
- Added `eval_rate_s` and `unevaluated_count` properties for time
  estimates. Eval timing tracked per agent.
- Progress events during evaluation include `unevaluated_count` and
  `eval_rate_s` for frontend consumption.
- 6 new tests in `test_orchestrator.py`.

## Session 03 — API endpoint: granularity parameter (2026-04-12)

- `POST /training/stop` now accepts `granularity` query param
  (default: `"immediate"`). Validates against `VALID_STOP_GRANULARITIES`.
- `POST /training/finish` unchanged (backward compat alias).
- `TrainingStatus` schema extended with `unevaluated_count: int | None`
  and `eval_rate_s: float | None`.
- Status endpoint reads these from the latest event.
- 5 new tests in `test_api_training.py`.

## Session 04 — Frontend: stop dialog (2026-04-12)

- "Stop Training" button now opens a dialog with three radio options:
  - Evaluate all generated models (eval_all)
  - Finish current evaluation only (eval_current)
  - Stop immediately (immediate)
- Each option shows a time estimate computed from `unevaluated_count`
  and `eval_rate_s` from the status endpoint / WebSocket events.
- Cancel closes dialog without sending any command.
- Escalation: after choosing eval_all, re-opening dialog shows only
  eval_current and immediate. After eval_current, only immediate.
- `TrainingStatus` and `WSEvent` interfaces extended with new fields.
- `TrainingService` passes `unevaluated_count` and `eval_rate_s`
  through WS events and poll status.
- `ApiService.stopTraining()` accepts granularity parameter.
- Dialog styled to match dark theme.
- 9 new tests in `training-monitor.spec.ts`.
