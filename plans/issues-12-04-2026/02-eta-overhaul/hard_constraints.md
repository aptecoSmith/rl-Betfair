# Hard Constraints

- `ProgressTracker` API must not break — existing callers (evaluator,
  run_training phase trackers) continue to work unchanged.
- WebSocket progress event schema is additive only — `overall` is a new
  field alongside `process` and `item`, not a replacement.
- Historical timing file is best-effort — if it's missing or corrupt,
  fall back to the 12s default. Never crash on a missing timing file.
- The overall ETA must not jump wildly at phase transitions. The rolling
  window should carry real timing data across phases.
- All tests pass: `python -m pytest tests/ --timeout=120 -q`.
- Frontend builds clean: `ng build`.
