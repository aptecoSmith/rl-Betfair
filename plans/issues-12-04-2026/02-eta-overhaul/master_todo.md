# Master TODO — ETA Overhaul

## Session 1: Historical timing + overall run tracker

### Backend — persist and serve historical timing

- [ ] At end of each completed run, compute actual seconds_per_agent_per_day
      from elapsed time and write it to a config-accessible location
      (e.g. `logs/training/last_run_timing.json` or an AppSettings-style
      key in the registry DB)
- [ ] `GET /api/training/info` — read historical rate and return it as
      `seconds_per_agent_per_day` instead of hardcoded 12.0; fall back
      to 12.0 if no history
- [ ] Store separate rates for training vs evaluation phases so the
      wizard estimate doesn't need the arbitrary 60/40 split

### Backend — overall run tracker

- [ ] Create a `RunProgressTracker` in `training/progress_tracker.py`
      that models the full run shape: generations × (train + eval) phases
- [ ] Initialise it in `run_training.py` at run start with:
      total_work_units = n_generations × (pop × train_days × epochs + pop × test_days)
- [ ] Tick the run tracker alongside existing phase/item trackers so it
      accumulates real elapsed time across phase boundaries
- [ ] Publish a third `overall` field in progress WebSocket events
- [ ] Ensure phase transitions don't cause the overall ETA to jump
      (the rolling window should carry across phases)

### Frontend — three-tier progress display

- [ ] Add `OverallProgressSnapshot` to training model (or reuse existing
      `ProgressSnapshot` type)
- [ ] Update `training.service.ts` to read `event.overall` from WebSocket
- [ ] Relabel bars: "OVERALL" (top), "PHASE" (middle), "CURRENT" (bottom)
- [ ] Overall bar always visible while running; phase and current bars
      show/hide as they do today
- [ ] Overall bar shows: "Generation 1/3 — training", ETA for full run
- [ ] Phase bar shows: "Training 5/50 agents", ETA for this phase
- [ ] Current bar shows: "Agent abc123 — episode 7/12", ETA for this item

### Frontend — wizard estimate improvement

- [ ] `estimatedDuration()` — use historical `train_rate_s` and
      `eval_rate_s` from `/api/training/info` instead of hardcoded
      12.0 with 60/40 split
- [ ] Show "(based on last run)" or "(default estimate)" so user knows
      confidence level

### Tests

- [ ] Test `RunProgressTracker` — tick across phase boundaries, ETA
      doesn't reset
- [ ] Test historical timing persistence — write after run, read on
      next `/api/training/info`
- [ ] Test WebSocket event includes `overall` field
- [ ] Existing progress tracker tests still pass

### Verify

- [ ] `python -m pytest tests/ --timeout=120 -q` — all green
- [ ] `cd frontend && ng build` — clean
- [ ] Manual check: start a short training run, verify all three bars
      display sensibly and overall ETA doesn't jump at phase transitions
