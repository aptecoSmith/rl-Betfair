# Progress — Training End Summary Modal

## Session 1 — 2026-04-14 ✅

Shipped as two commits:
1. Backend enrichment landed with the ETA overhaul: run_complete event
   now carries status, wall_time_seconds, generations_completed /
   _requested, total_agents_trained / _evaluated, best_model, top_5,
   population_summary, error_message.
2. Frontend modal in
   `feat: training end summary modal with best model, top 5, and action buttons`.

Modal behaviour:
- Auto-opens on run_complete / phase_complete(run_stopped|run_error).
- Header colour/label varies by status (green / amber / red).
- Shows formatted duration, generation count (with "stopped early"
  tag if truncated), agents trained/evaluated, best-model card with
  score/P&L/win-rate/architecture, top-5 table, population chips, and
  error_message in a red box when present.
- Action buttons: Dismiss, Start New Run, View Best Model (→ model
  detail), View Scoreboard.
- Auto-dismisses when a new run starts.
- Idle state shows a compact "Last run {status} · Best: abc123…
  (score X.XX, P&L X.XX)" card with a "View summary" button that
  re-opens the modal. The raw JSON `<pre>` dump is gone.
- The summary is fully backward-compatible: if a legacy minimal
  run_complete event arrives, status is derived from the phase and all
  missing fields render as fallbacks.

Tests: `tests/test_run_summary_and_timing.py::TestRunCompleteSummary`
asserts every required key is present and that best_model lines up
with top_5[0].
