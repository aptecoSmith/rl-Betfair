# Sprint 2, Session 1: Training Plans — Fix Save + Launch (Issue 03, Session 1)

Read `CLAUDE.md` and `plans/issues-12-04-2026/03-training-plans-integration/`
before starting. Follow session 1 of `master_todo.md`.

## Scope

The critical path — make training plans functional end-to-end:

1. Debug and fix the "save does nothing" bug
2. Add `plan_id` to `POST /api/training/start`
3. Worker loads plan, passes to orchestrator
4. "Start plan" button on plan detail view
5. Add `n_generations`, `n_epochs` to TrainingPlan model

See `plans/issues-12-04-2026/03-training-plans-integration/session_prompt.md`
for full details, key files, and constraints.

## Commit

`feat: wire training plans into launch flow + fix save bug`
Push: `git push all`
