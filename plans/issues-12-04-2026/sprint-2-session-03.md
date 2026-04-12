# Sprint 2, Session 3: Training Plans — Session Splitting (Issue 03, Session 3)

Read `CLAUDE.md` and `plans/issues-12-04-2026/03-training-plans-integration/`
before starting. Follow session 3 of `master_todo.md`.

## Scope

Break large plans into manageable training sessions with checkpoints:

1. `generations_per_session` concept on TrainingPlan
2. Session boundary computation
3. Auto-continue: when a session finishes and `auto_continue=true`,
   launch the next session automatically
4. Resume: `POST /api/training/resume` to pick up from last completed session
5. Frontend: session breakdown display, continue button, per-session timing

See session 3 of `master_todo.md` for the full task list.

## Commit

`feat: training plan session splitting + auto-continue + resume`
Push: `git push all`
