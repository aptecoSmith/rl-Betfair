# Sprint 2, Session 2: Training Plans — Status + Progress (Issue 03, Session 2)

Read `CLAUDE.md` and `plans/issues-12-04-2026/03-training-plans-integration/`
before starting. Follow session 2 of `master_todo.md`.

## Scope

Give visibility into which plan is running and how far through it is:

1. Add `status` field to TrainingPlan (draft/running/completed/failed)
2. Track `current_generation` and `started_at`
3. Update status on start, generation complete, run finish, crash
4. Status badges on plan cards in list view
5. Progress indicator in detail view
6. Cross-links: plan detail ↔ training monitor

See session 2 of `master_todo.md` for the full task list.

## Commit

`feat: training plan status tracking + progress display`
Push: `git push all`
