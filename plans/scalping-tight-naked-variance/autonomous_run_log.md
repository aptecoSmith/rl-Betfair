# Autonomous run log — scalping-tight-naked-variance

Per-iteration log of the autonomous run. Each iteration appends one
entry using the template at the bottom of
`session_prompts/00_autonomous_full_run.md`.

The plan exits at the end of Phase 1d (verdict commits). Phase 2 is
operator-gated per `hard_constraints.md §22`.

<!-- iterations append below this line -->

## 2026-05-14 20:48 — Phase 0+1, iteration 1 (BLOCKED on 1c)

**State entering iteration:** plan scaffolded, no work yet.
**Work done:**
- Phase 0 complete: `tools/build_naked_variance_report.py` +
  `tests/test_naked_variance_report.py`. Five tests pass. Committed
  as `83a21a6`.
- Phase 1a complete: verified `reevaluate_cohort.py` already accepts
  `--filter-agent-ids`. No code change.
- Phase 1b complete: ran the report on raceconf + layq. Wrote
  `naked_variance_report.csv` and `phase1_top5_union.txt` into each
  cohort dir. 14 unique agents per cohort.
- Phase 1c BLOCKED. All 14 raceconf agents failed weight load on
  the first eval day with `input_proj.0.weight` shape mismatch
  ([64, 504] checkpoint vs [64, 574] current). Verified directly:
  a saved checkpoint has `obs_schema_version=8`; current
  `env.betfair_env.OBS_SCHEMA_VERSION` is 9 (commit `045174d`,
  today at 14:54). Stop condition #3 fired.
- Phase 1d: wrote `phase1_verdict.md` documenting the block and
  presenting three remediation options for operator decision.

**Tests run:** `pytest tests/test_naked_variance_report.py -v` → 5/5
pass.
**Decisions made:** STOPPED at stop condition #3 per the autonomous
prompt. Did NOT auto-revert `045174d`; did NOT auto-create a
worktree (operator decision per `hard_constraints.md §22` spirit).
Did NOT launch Phase 2.
**Outstanding for this phase:** held-out reeval (8 jobs) once
operator picks a remediation path.
**Next iteration's focus:** wait for operator input.
