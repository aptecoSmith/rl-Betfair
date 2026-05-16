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

## 2026-05-15 12:13 — Phase 1c complete, Phase 1d done

**State entering iteration:** Phase 1c reeval chain (8 jobs) launched
from per-cohort eval worktrees at 22:02 BST 2026-05-14.
**Work done:**
- All 8 reevals completed cleanly between 22:02 and 11:33 BST,
  14 rows per JSONL. Total wall: 13h 31m.
- Wrote `C:/tmp/phase1_analyze.py` to roll the 8 JSONLs into the
  per-selector × per-cohort × per-window × per-fc verdict table.
- Generated `phase1_verdict_table.csv` (40 cells) and amended
  `phase1_verdict.md` with the headline numbers, full per-cell
  table, and band verdict.
**Tests run:** None this iteration; analysis only.
**Decisions made:** All 20 fc-paired cells land in REGRESSION
territory (fc=120 mean < 0). No band cleared. Phase 1 hypothesis
(variance-aware selection over existing populations surfaces
deployable agents) is REJECTED on held-out data. Phase 2 (retrain
with variance-aware reward + fc=120 in training) is the only
remaining path per the plan; recipes documented in verdict.md.
**Outstanding for this phase:** verdict + analysis CSV need
committing.
**Next iteration's focus:** Phase 1 EXITS here. Operator gate per
hard_constraints.md s22 — operator picks Phase 2A / Phase 2B /
BOTH / retire.

## 2026-05-16 07:48 — Phase 2A complete

**State entering iteration:** Cohort _predictor_SCALPING_tnv_raceconf_
1778852093 complete (96/96 agents at 02:34). Phase 3 reeval (4 jobs)
underway since 02:40.
**Work done:**
- All 4 Phase 3 reevals completed cleanly at 07:30. 10-agent union
  top-5 reevald on 2 windows x 2 fc settings = 4 JSONLs.
- Analysed results vs Phase 1 explicit null:
  - new fc=0 top-5 mean PnL: -£9.73 vs null -£40.50 = +£30.77 BETTER
  - new fc=120 top-5 mean PnL: -£20.09 vs null -£16.92 = tied
  - Best agent 32ed9e32 (gen 0, beta=0.00133): -£0.76/day fc=120 new,
    4/7 profitable days, naked_std £101 (Modest band's std ceiling).
- Wrote `findings.md` with full verdict, in-sample gen trend, and
  three follow-on recipes (default: re-run Phase 2A WITH fc=120 in
  training).
**Tests run:** None this iteration; analysis only.
**Decisions made:** Phase 2A complete. No band cleared but variance-
penalty mechanism CONFIRMED to produce selection pressure in-sample
(46% max-span reduction gen 0-3, beta_med 0.016 -> 0.030). The
train-vs-deploy fc asymmetry from the layq predecessor remains —
operator's decision to defer fc=120 in training was the limiting
factor.
**Outstanding for this phase:** verdict + log need committing.
**Next iteration's focus:** plan exits here; operator decides
follow-on (Option A: re-run with fc=120; Option B: narrow sweep;
Option C: deploy 32ed9e32 directly).
