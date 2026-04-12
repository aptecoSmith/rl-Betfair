# Session: ETA Overhaul

Read `CLAUDE.md` and `plans/issues-12-04-2026/02-eta-overhaul/` before
starting. Follow `master_todo.md` in order. Mark items done as you go and
update `progress.md` at the end.

## Context

The ETA system has three problems:

1. **Wizard estimate is fiction.** `api/routers/training.py:160` hardcodes
   `seconds_per_agent_per_day: 12.0`. The frontend (`training-monitor.ts:414-416`)
   splits this 60/40 between train/eval with no basis. Real rates vary 3-5x.

2. **No overall ETA.** `ProgressTracker` (`training/progress_tracker.py`)
   tracks one phase at a time. When training finishes and eval starts,
   the process tracker resets. There's no tracker spanning the full run.

3. **Labels are misleading.** "PROCESS" and "ITEM" in
   `training-monitor.html:15,31` don't convey what they actually track.
   The item bar's `process_eta_human` during eval means "remaining test
   days for this agent", not total remaining time.

## What to do

### 1. Historical timing

At the end of a completed run in `training/run_training.py`, compute:
- `train_seconds_per_agent_per_day` = total training wall time / (agents × train_days × epochs)
- `eval_seconds_per_agent_per_day` = total eval wall time / (agents × test_days)

Persist to `logs/training/last_run_timing.json`. In `GET /api/training/info`
(`api/routers/training.py:160`), read this file and return the historical
rates. Fall back to `12.0` if the file doesn't exist.

Update `training-monitor.ts::estimatedDuration()` to use separate train
and eval rates instead of the 60/40 hack. Show "(based on last run)" or
"(default estimate)" after the duration string.

### 2. Overall run tracker

Add a `RunProgressTracker` class (or extend `ProgressTracker`) that:
- Is initialised at run start with the total work shape
- Gets ticked as each agent-phase completes (not just each item)
- Carries its rolling window across phase transitions
- Publishes as `overall` in the WebSocket progress event

In `run_training.py`, the run loop structure is:
```
for generation in range(n_generations):
    train all agents        ← outer tracker: agents, inner: episodes
    evaluate all agents     ← outer tracker: agents, inner: test days
```

The overall tracker should tick once per agent completion (both train
and eval), giving it a consistent unit of work. Total = n_generations
× pop × 2 (train + eval per agent).

### 3. Frontend three-tier bars

In `training-monitor.html`, change the bar layout to three tiers:
- **OVERALL** (new, top) — reads `status().overall`, label like
  "Generation 1/3 — training", shows full-run ETA
- **PHASE** (was PROCESS) — reads `status().process`, label like
  "Training 5/50 agents", shows phase ETA
- **CURRENT** (was ITEM) — reads `status().item`, label like
  "Agent abc123 — episode 7/12", shows current-item ETA

Update `training.service.ts` to read `event.overall` from WebSocket
events and store it alongside process/item.

## Key files

| File | What to change |
|------|----------------|
| `training/progress_tracker.py` | Add `RunProgressTracker` or extend existing |
| `training/run_training.py` | Initialise + tick overall tracker, persist timing at end |
| `api/routers/training.py` | Read historical timing for `/api/training/info` |
| `frontend/src/app/models/training.model.ts` | Add overall snapshot type |
| `frontend/src/app/services/training.service.ts` | Read overall from WS |
| `frontend/src/app/training-monitor/training-monitor.ts` | New computed for overall bar, fix estimatedDuration |
| `frontend/src/app/training-monitor/training-monitor.html` | Three-tier bar layout, relabel |
| `frontend/src/app/training-monitor/training-monitor.scss` | Style overall bar |

## Constraints

- `ProgressTracker` is used by other code — don't break existing callers.
- The rolling window approach is sound — keep it. The issue is scope, not
  algorithm.
- Don't change the WebSocket event schema in a breaking way — add the
  `overall` field alongside existing `process` and `item`.
- `python -m pytest tests/ --timeout=120 -q` must pass.
- `cd frontend && ng build` must be clean.

## Commit

Single commit: `feat: ETA overhaul — historical timing, overall run tracker, three-tier bars`
Push: `git push all`
