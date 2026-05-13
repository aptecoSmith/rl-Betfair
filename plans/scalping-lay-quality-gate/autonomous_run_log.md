# Autonomous run log — scalping-lay-quality-gate

Per-iteration log of the autonomous run. Each iteration writes
one entry using the template at the bottom of
`session_prompts/00_autonomous_full_run.md`.

## 2026-05-13 — Phase 0, iteration 1

**State entering iteration:** plan folder existed with only
`session_prompts/00_autonomous_full_run.md`; predecessor
`scalping-race-confidence-gate` complete (mean +£39.40/day,
3/5 profitable held-out).
**Work done:**
- Verified no active cohort process is running
  (`Get-Process python` showed only the predecessor watcher).
- Created `plans/scalping-lay-quality-gate/README.md`.
- Created `plans/scalping-lay-quality-gate/hard_constraints.md`.
- Created `plans/scalping-lay-quality-gate/master_todo.md`.
- Created `session_prompts/01_probe.md` through
  `06_compare_and_verdict.md`.
- Created this `autonomous_run_log.md`.
**Tests run:** none (scaffolding only).
**Decisions made:**
- Inherit `race_confidence_threshold = 0.50` (predecessor's
  smoke-PASS value).
- Phase 1 will set `predictor_p_win_lay_threshold` and
  `lay_price_max` empirically.
- Two reeval watchers (force_close=0 + 120) per
  `memory/project_force_close_train_vs_deploy.md`.
- Phase 2a and 2b will be committed separately so variables
  are separable for analysis.
**Outstanding for this phase:** commit scaffold with message
`plan(scalping-lay-quality-gate): scaffold next stack-on plan`.
**Next iteration's focus:** Phase 1 — re-run
`tools/probe_lay_outcome_distribution.py` on 2026-04-28/29/30
to set Phase 3 defaults.
