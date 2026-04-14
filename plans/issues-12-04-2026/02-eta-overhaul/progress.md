# Progress — ETA Overhaul

## Session 1 — 2026-04-14 ✅

Shipped in commit `feat: ETA overhaul — historical timing, overall run tracker, three-tier bars`.

- Added `RunProgressTracker` (subclass of `ProgressTracker` with a
  mutable label) — carries the rolling ETA window across phase
  transitions.
- Orchestrator initialises one at run start (total = n_gens × pop × 2),
  ticks once per agent train and per agent eval, and publishes it as
  `overall` in every progress event.
- Historical per-agent-per-day train + eval rates written to
  `logs/training/last_run_timing.json` at run end. Best-effort — any
  IO error logs and returns cleanly.
- `GET /api/training/info` reads the file and returns
  `train_seconds_per_agent_per_day`, `eval_seconds_per_agent_per_day`,
  and `timing_based_on_last_run` alongside the legacy field.
- Frontend: three-tier bars relabelled OVERALL / PHASE / CURRENT.
  Wizard's `estimatedDuration()` uses the separate train/eval rates
  and suffixes "(based on last run)" or "(default estimate)".
- IPC + worker state propagate `latest_overall` alongside the other
  snapshots so mid-run connecting clients get the overall bar too.
- Tests: `RunProgressTracker.set_label` + cross-phase ETA carry,
  progress-event `overall` field, timing-file persistence, run_complete
  summary shape.
