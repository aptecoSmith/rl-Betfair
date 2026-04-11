# Sprint 3 — Training Stop Dialog (4 sessions)

One issue: granular stop options for the training monitor.

## Before you start

- Verify Sprints 1–2 landed: tests green, budget configurable, market
  type filter gene working.
- Read `plans/issues-11-04-2026/order.md` for context.

## Issue 02 — Training Stop Options (4 sessions)

Read the full plan folder: `plans/issues-11-04-2026/02-training-stop-options/`

Start with `purpose.md` (includes the dialog sketch), then
`hard_constraints.md`, then work through `session_prompt.md`
sessions 01–04 in order.

Summary of sessions:
1. IPC + worker: add `granularity` parameter to stop command
   (`immediate`, `eval_current`, `eval_all`)
2. Orchestrator: handle new threading events (`skip_training_event`,
   `stop_after_current_eval_event`)
3. API endpoint: accept granularity param, add `unevaluated_count`
   and `eval_rate_s` to status response
4. Frontend: stop dialog with three radio options and time estimates

The existing "Finish Up" button stays as-is. "Stop Training" gets a
dialog offering three choices with estimated time for each.

Key constraint: escalation is always allowed (eval_all → eval_current
→ immediate), de-escalation is not.

No ai-betfair knock-on.

**Exit per session:** All tests pass, `progress.md` updated, commit.

---

## Sprint complete

After all four sessions:
1. Full test suite green.
2. Push: `git push origin master`.
3. Start a training run, then test the stop dialog — verify each
   granularity option works and time estimates are reasonable.
