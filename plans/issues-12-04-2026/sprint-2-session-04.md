# Sprint 2, Session 4: ETA Overhaul + Training End Summary (Issues 02 + 12)

Two monitoring improvements in one session. Read `CLAUDE.md` first,
then the issue folders listed below.

---

## Part 1: ETA Overhaul (Issue 02)

Read `plans/issues-12-04-2026/02-eta-overhaul/` for full context.

### Summary

1. Replace hardcoded 12s/agent/day wizard estimate with historical
   timing from last completed run. Persist to
   `logs/training/last_run_timing.json`. Separate train vs eval rates.
2. Add overall run tracker spanning all generations and phases.
   Publishes as `overall` field in WebSocket progress events.
3. Relabel bars: OVERALL (top) → PHASE (middle) → CURRENT (bottom).

### Key files

- `training/progress_tracker.py` — RunProgressTracker
- `training/run_training.py` — initialise + tick overall tracker, persist timing
- `api/routers/training.py:160` — read historical timing
- `frontend/src/app/training-monitor/` — three-tier bars, estimatedDuration fix

---

## Part 2: Training End Summary Modal (Issue 12)

Read `plans/issues-12-04-2026/12-training-end-summary/` for full context.

### Summary

1. Enrich `run_complete` event with formatted summary: best model,
   top 5, population stats, wall time, status.
2. Auto-opening modal on training completion with action buttons
   (View Scoreboard, View Best Model, Start New Run, Dismiss).
3. Replace raw JSON `<pre>` dump with compact idle-state card.

### Key files

- `training/run_training.py` — enrich run_complete event
- `frontend/src/app/training-monitor/` — modal + compact card

---

## Commits

Two separate commits:
1. `feat: ETA overhaul — historical timing, overall run tracker, three-tier bars`
2. `feat: training end summary modal with best model, top 5, and action buttons`

Push: `git push all`

## Verify

- `python -m pytest tests/ --timeout=120 -q` — all green
- `cd frontend && ng build` — clean
