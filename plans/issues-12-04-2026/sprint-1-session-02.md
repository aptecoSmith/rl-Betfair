# Sprint 1, Session 2: Log Consolidation + Training Plans Help Text (Issues 01 + 06)

Two quick wins in one session. Read `CLAUDE.md` first, then the issue
folders listed below.

---

## Part 1: Log Consolidation (Issue 01)

Read `plans/issues-12-04-2026/01-log-consolidation/` for full context.

### Summary

Move `registry/bet_logs/` to `logs/bet_logs/` so all log output lives
under one tree. Add `paths.bet_logs` to config.yaml. Update the 3
production call sites that hardcode the path. Add a
`GET /api/admin/log-paths` endpoint and a "Log Files" card to the
admin page.

### Key changes

1. `config.yaml` — add `paths.bet_logs: logs/bet_logs`
2. `api/main.py:54` — read from config instead of `db_path.parent / "bet_logs"`
3. `training/worker.py:122` — same
4. `registry/model_store.py:173` — default fallback to `logs/bet_logs`
5. Move existing `registry/bet_logs/` → `logs/bet_logs/`
6. New endpoint: `GET /api/admin/log-paths`
7. Admin page: "Log Files" card with copy-path button

---

## Part 2: Training Plans Help Text (Issue 06)

Read `plans/issues-12-04-2026/06-training-plans-help-text/` for full context.

### Summary

Add explanatory help text to every field and section in the training
plans editor. Frontend-only. Follow the wizard page's existing
`.help-text` / `.field-help` patterns.

### Key changes

1. `training-plans.html` — add help text below every input and section
2. `training-plans.scss` — style consistently with wizard
3. Check `app-gene-editor` component — add gene description display
   if not already present

Draft copy for all fields is in `master_todo.md`.

---

## Commits

Two separate commits:
1. `feat: consolidate logs under logs/ + add log paths UI`
2. `feat: add explanatory help text to training plans editor`

Push: `git push all`

## Verify

- `python -m pytest tests/ --timeout=120 -q` — all green
- `cd frontend && ng build` — clean
