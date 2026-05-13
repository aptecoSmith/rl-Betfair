# 00 — Autonomous full run

You are driving the **scalping-race-confidence-gate** plan to
completion. Each invocation is one iteration. `/loop` re-fires
this prompt. **Operator interaction is NOT available** — make
every decision yourself using the documents + defaults below.

Deliverable: a held-out reeval verdict (≥3/5 profitable on
2026-04-28/29/30 AND mean > pwin-gate's −£13) for a scalping
cohort that stacks a per-race confidence filter on top of the
pwin-gate's per-runner filter.

## Read FIRST every iteration

1. `plans/scalping-race-confidence-gate/README.md` — design +
   hypothesis + success bar
2. `plans/scalping-race-confidence-gate/hard_constraints.md` —
   12 inviolable constraints
3. `plans/scalping-race-confidence-gate/master_todo.md` — session
   table + after-session defaults
4. `plans/scalping-race-confidence-gate/autonomous_run_log.md` —
   your own progress log (create on iteration 1)
5. The relevant session prompt for the current phase.

## Each iteration — in order

1. **Read state** — `autonomous_run_log.md` tells you where you
   are.
2. **Check stop conditions** (below) — if any apply, write final
   entry, surface a paragraph, do NOT schedule next wakeup.
3. **Pick the next concrete sub-step** from the current
   session's prompt.
4. **Do the work** — edit code, run tests, launch processes.
5. **Log** with the template at the end of this file.
6. **Decide whether the session is done** — if yes, commit and
   move on.
7. **Schedule next iteration** via `ScheduleWakeup` with a
   pacing-appropriate delay.

## Stop conditions

Loop ends (no next wakeup) when ANY of:

1. **Session 04 findings.md is written and committed** — plan
   complete.
2. **Session 02 pre-flight smoke FAILS** any of three thresholds
   (hard_constraints §3). Stop, write diagnostic. Do NOT launch
   the 12h cohort.
3. **Cohort process crashed** with a Traceback in the log.
4. **A hard_constraint is about to be violated** to make
   progress.
5. **Three consecutive iterations on the same sub-step without
   progress.**

## Pacing

- 60–270s during active code/tests
- 900–1800s waiting on smoke/cohort partial
- 3600s max heartbeat during cohort mid-flight

Re-fire prompt verbatim:
`/loop @plans/scalping-race-confidence-gate/session_prompts/00_autonomous_full_run.md`

## Session-by-session execution

### Session 01: Implement gate
Follow `session_prompts/01_implement_gate.md`. Edits in env +
compute_mask + worker + runner + reeval + tests. ~5 iterations
to complete. Success: all tests pass.

### Session 02: Pre-flight smoke
Follow `session_prompts/02_smoke_test.md`. Write smoke tool, run
on 2026-05-04, evaluate against §3 thresholds. Binary
PASS/FAIL.

### Session 03: Launch cohort
Follow `session_prompts/03_launch_cohort.md`. Mirror the
pwin-gate launch + `--race-confidence-threshold 0.30`. Arm
watcher. Sleep into 1h heartbeat mode until 96 rows.

### Session 04: Compare + verdict
Follow `session_prompts/04_compare_and_verdict.md`. Read JSONL,
compute table, write findings.md, commit, STOP.

## Default decisions (no operator)

| Question | Default |
|---|---|
| `race_confidence_threshold` | 0.50 (revised 2026-05-13 after 0.30 smoke FAIL) |
| `predictor_p_win_back_threshold` | 0.20 (same as predecessor) |
| `predictor_p_win_lay_threshold` | 0.40 (same as predecessor) |
| Cohort size | 12 × 8 × 6 |
| Seed | 42 |
| Mutation rate | 0.2 |
| Enabled genes | same 6 safety genes |
| Eval window | 2026-04-28/29/30 (locked) |
| Smoke day | 2026-05-04 |
| If a test fails | fix in same iteration; one retry; stop on third |
| If a launch fails | read traceback; one fix retry; stop if still failing |

## What NOT to do

- Do NOT tweak `race_confidence_threshold` between sessions or
  mid-cohort. The default 0.50 is locked (revised 2026-05-13;
  see autonomous_run_log.md for the probe + decision).
- Do NOT use `p_placed`, `segment_strong_flag`, or per-runner
  averages instead of `max(p_win)`. See hard_constraints §9.
- Do NOT silence failing tests.
- Do NOT push to origin. Commit locally only.
- Do NOT skip the smoke if you're confident the gate works.
  The smoke is the load-bearing safety check (the
  direction-gate plan was saved by this discipline; same here).

## What you SHOULD do

- Commit at clean session boundaries.
- Log per iteration to `autonomous_run_log.md`.
- Use specific `git add <file>`, never `.`.
- Use `run_in_background=True` for the cohort + watcher +
  status updater (long processes).
- Read the cohort `status.txt` file rather than re-running
  ad-hoc python.

## Log entry template

```markdown
## YYYY-MM-DD HH:MM — Session NN, iteration M

**State entering iteration:** one sentence.
**Work done:** bullet list with file paths / test names.
**Tests run:** what was run, what passed/failed.
**Decisions made:** any defaults applied.
**Outstanding for this session:** what's left.
**Next iteration's focus:** specific concrete next step.
```

## After plan exit

When Session 04 commits `findings.md`:

1. Surface the headline result in one paragraph (success bar
   pass/fail, mean held-out vs pwin-gate's −£13).
2. Recommend next plan from README's "What success looks like"
   branches.
3. Stop scheduling.
