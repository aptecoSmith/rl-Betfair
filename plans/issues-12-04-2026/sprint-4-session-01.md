# Sprint 4, Session 1: Manual Evaluation — Backend (Issue 04, Session 1)

Read `CLAUDE.md` and `plans/issues-12-04-2026/04-manual-evaluation/`
before starting. Follow session 1 of `master_todo.md`.

## Scope

Build the standalone evaluation path — worker command + API:

1. Add `CMD_EVALUATE` to worker — accepts model_ids + test_dates
2. For each model: load weights, reconstruct policy, run evaluator
3. Progress via existing WebSocket
4. Add `POST /api/evaluate` endpoint with validation
5. Add `GET /api/evaluate/status` endpoint
6. Per-model error handling (skip + continue)

See `plans/issues-12-04-2026/04-manual-evaluation/session_prompt.md`
for full details.

## Commit

`feat: standalone evaluation — worker command + API endpoint`
Push: `git push all`
