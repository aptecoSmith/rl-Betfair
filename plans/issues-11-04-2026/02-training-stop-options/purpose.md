# 02 — Training Stop Options (Finish vs Stop with Granularity)

## Problem

The training monitor has two buttons — "Finish Up" and "Stop Training" —
but the distinction isn't clear and neither offers enough control.  The
operator wants to choose *how much* evaluation happens before shutdown:

| Scenario | What the operator wants |
|---|---|
| "I've generated enough models, just evaluate them all" | Stop breeding, evaluate entire current population |
| "I need to leave soon, wrap up what you're doing" | Finish current agent's eval, then stop |
| "Kill it now" | Abort immediately (current behaviour of Stop) |

## Current behaviour

- **Stop** (`stop_event`): Halts after the current agent finishes
  its training or eval step.  No further evaluation runs.  Results
  for partially-evaluated populations are incomplete.
- **Finish** (`finish_event`): Completes the current generation
  (all agents train), then evaluates the full population, scores,
  and exits.  No next generation is bred.

The gap: there's no option to say "stop training new models but
evaluate everything we've already generated" without waiting for the
current generation's training to complete first.

## Proposed UX: dialog on Stop Training button

Replace the current two-button layout with:

- **"Finish Up"** — unchanged (complete current generation + eval)
- **"Stop Training"** — opens a dialog with three choices:

### Stop Training dialog

```
┌──────────────────────────────────────────────────────────┐
│  Stop Training                                      [✕]  │
│                                                          │
│  Choose how to wind down:                                │
│                                                          │
│  ○ Evaluate all generated models                         │
│    Stop breeding. Evaluate every model in the current    │
│    population that hasn't been evaluated yet.            │
│    Estimated time: ~12 min (8 models remaining)          │
│                                                          │
│  ○ Finish current evaluation only                        │
│    Complete the model currently being evaluated,          │
│    then stop. Other unevaluated models are skipped.      │
│    Estimated time: ~2 min                                │
│                                                          │
│  ○ Stop immediately                                      │
│    Abort now. Current evaluation results are discarded.  │
│    Estimated time: < 10 seconds                          │
│                                                          │
│              [Cancel]              [Confirm]              │
└──────────────────────────────────────────────────────────┘
```

The dialog shows estimated time where possible (computed from the
current eval rate × remaining models).

## Implementation

### 1. New stop granularity enum

In `training/ipc.py`, extend the finish/stop commands to carry a
`granularity` field:

```python
STOP_EVAL_ALL = "eval_all"         # Stop breeding, eval full population
STOP_EVAL_CURRENT = "eval_current" # Finish current eval, then stop
STOP_IMMEDIATE = "immediate"       # Abort now (current Stop behaviour)
```

### 2. Worker command handling (`training/worker.py`)

- `CMD_STOP` with `granularity: "immediate"` → sets `stop_event` (as now)
- `CMD_STOP` with `granularity: "eval_current"` → sets a new
  `stop_after_current_eval_event`
- `CMD_STOP` with `granularity: "eval_all"` → sets `finish_event` +
  a new `skip_training_event` (or reuse `finish_event` with a flag
  that means "don't wait for training to complete")

### 3. Orchestrator changes (`training/run_training.py`)

The orchestrator currently checks `stop_event` and `finish_event` at
generation boundaries and between agents.  Add checks for the new
granularities:

**`eval_all` ("evaluate all generated models"):**
- If in the training phase: skip remaining training immediately,
  jump to evaluation of whatever models exist.  This is subtly
  different from current Finish which waits for training to complete.
- If already in eval phase: continue as normal.

**`eval_current` ("finish current evaluation only"):**
- If in eval phase: let the current agent's eval finish, then stop.
  Don't evaluate remaining agents.
- If in training phase: skip remaining training, don't evaluate
  anything (there's nothing being evaluated yet).

**`immediate` ("stop now"):**
- Same as current `stop_event`.

### 4. API endpoint changes (`api/routers/training.py`)

Merge stop and finish into a single endpoint (or keep both for
backward compat) that accepts a `granularity` parameter:

```python
@router.post("/stop")
def stop_training(request: Request, granularity: str = "immediate"):
    ...
```

### 5. Frontend dialog (`training-monitor/`)

- New `StopDialog` component (or inline template in training-monitor)
- Three radio options as sketched above
- Time estimates computed from WebSocket status:
  - `eval_all`: `(unevaluated_count) × avg_eval_time`
  - `eval_current`: remaining time on current eval (from progress %)
  - `immediate`: "< 10 seconds"
- Confirm calls the API with selected granularity

### 6. WebSocket status additions

The frontend needs to know:
- How many models are waiting for evaluation (to show the estimate)
- Current eval progress rate (seconds per model)
- Which phase we're in (training vs eval) to disable irrelevant options

Some of this is already in the progress events; may need a small
addition to surface `unevaluated_count`.

## Files touched

| Layer | File | Change |
|---|---|---|
| IPC | `training/ipc.py` | Add granularity to stop command |
| Worker | `training/worker.py` | Handle new granularities |
| Orchestrator | `training/run_training.py` | New event checks, skip-to-eval logic |
| API | `api/routers/training.py` | Accept granularity param |
| Frontend | `training-monitor.html` | Stop dialog template |
| Frontend | `training-monitor.ts` | Dialog state, time estimates |
| Frontend | `training-monitor.scss` | Dialog styles |

## Edge cases

- **Stop during breeding (between generations):** `eval_all` should
  still evaluate the current population even though breeding hasn't
  started yet.
- **Stop when no models need eval:** `eval_all` becomes equivalent
  to `immediate` — dialog should grey out the option or show
  "0 models remaining".
- **Multiple stop requests:** Once a stop granularity is chosen,
  the dialog should show the committed choice and allow escalation
  (e.g. "eval_all" → user can escalate to "immediate" if they
  change their mind).
