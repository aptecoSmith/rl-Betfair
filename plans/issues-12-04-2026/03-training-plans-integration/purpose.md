# 03 — Training Plans Integration

## What

1. Fix the "save does nothing" bug on the training plans page.
2. Wire saved plans into the training launch flow — add a "Start plan"
   button, pass plan_id through the API to the worker/orchestrator.
3. Show plan status and progress — which plan is running, current
   generation, link back to the training monitor. Mark completed.
4. Session splitting — let a plan define multiple training sessions
   (e.g. 3 gens each). Auto-continue to the next session when one
   finishes. Resume from the last completed session.

## Why

- Training plans are fully built (UI, persistence, validation, coverage
  analysis, exploration strategies) but completely disconnected from
  actual training. They're save-only artefacts.
- The orchestrator already accepts `training_plan=` and
  `plan_registry=` kwargs, and `PopulationManager.initialise_population`
  accepts a `plan=` kwarg. The backend plumbing exists — the last-mile
  wiring from API → worker → orchestrator is missing.
- Without session splitting, long plans (50 agents × 5 generations) are
  all-or-nothing. A crash at generation 4 means starting over. Breaking
  into sessions with auto-continue gives manageable checkpoints.
