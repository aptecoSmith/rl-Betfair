# 00 — Autonomous full run

You are driving the **scalping-direction-gate** plan to completion.
Each invocation of this prompt is one iteration. The `/loop`
runtime re-fires it. **Operator interaction is NOT available** —
make every decision yourself by following the documents and the
defaults in this prompt.

The deliverable: a held-out reeval verdict (≥3/5 profitable on
2026-04-28/29/30) for a scalping cohort that stacks the direction
predictor's drift signal on top of the pwin-gate's champion
filter.

## Read FIRST every iteration

Short documents. Re-read every iteration so you stay aligned:

1. `plans/scalping-direction-gate/README.md` — plan summary,
   asymmetric gate design, hypothesis, success bar.
2. `plans/scalping-direction-gate/hard_constraints.md` — 10
   inviolable constraints.
3. `plans/scalping-direction-gate/master_todo.md` — session
   table + per-session deliverables + after-session defaults.
4. `plans/scalping-direction-gate/autonomous_run_log.md` — your
   own progress log (create on iteration 1 if absent).

The relevant session prompt for the current phase:

- `session_prompts/01_implement_gate.md` (Session 01)
- `session_prompts/02_smoke_test.md` (Session 02)
- `session_prompts/03_launch_cohort.md` (Session 03)
- `session_prompts/04_compare_and_verdict.md` (Session 04)

## Each iteration — steps in order

1. **Read state.** Read `autonomous_run_log.md`. Identify the
   current session and sub-step. If the log is empty, start
   Session 01.
2. **Check stop conditions** (see below). If any apply, write a
   final log entry, surface a one-paragraph summary, and DO NOT
   call ScheduleWakeup. Loop ends.
3. **Pick the next concrete deliverable.** One sub-step at a
   time. The session prompts list specific deliverables in
   order; pick the first one not yet done.
4. **Do the work.** Edit code, run tests, launch processes.
5. **Log progress.** Append a dated entry to
   `autonomous_run_log.md` using the template at the end of
   this prompt.
6. **Decide whether the session is done.** If the session's
   success bar is met, commit the session's work and move on.
7. **Schedule next iteration.** Call `ScheduleWakeup` with a
   delay appropriate to what's outstanding (see "Pacing"
   below).

## Stop conditions

The loop ends (do NOT schedule next wakeup) when ANY of:

1. **Session 04 findings.md is written and committed.** Plan
   complete.
2. **Session 02 pre-flight smoke FAILS** any of the three
   threshold checks (hard_constraints §3). Stop, write
   diagnostic, do NOT launch the 12h cohort.
3. **The cohort process crashed** with a Traceback or "Process
   exited" marker in the log. Stop and surface — this is a
   genuine operator-input situation.
4. **A hard_constraint is about to be violated** to make
   progress. Stop.
5. **Three consecutive iterations on the same sub-step without
   progress.** Stop and surface.

## Pacing

`ScheduleWakeup.delaySeconds`:

- **60–270s (cache-warm)** — actively iterating on code/tests.
- **900–1800s (idle window)** — waiting on smoke / cohort
  partial. Status file refreshes every 60s automatically; the
  loop just samples it.
- **3600s (max heartbeat)** — cohort mid-generation, no
  interesting log line for >30 min.

Pass `/loop @plans/scalping-direction-gate/session_prompts/
00_autonomous_full_run.md` verbatim back to ScheduleWakeup each
iteration so the loop re-fires.

## Session-by-session execution

### Session 01: Implement direction gate

Follow `session_prompts/01_implement_gate.md`. Roughly:

1. Edit `env/betfair_env.py` — add kwarg, cache, validation.
2. Edit `agents_v2/action_space.py::compute_mask` — apply the
   asymmetric drift filter on OPEN_LAY only.
3. Wire CLI flag through runner + worker + reeval.
4. Write tests in `tests/test_agents_v2_action_space.py::
   TestDirectionGate`.
5. Run tests; iterate until green.
6. Commit when full action-space suite passes.

Success bar: all new tests pass, all 32+ existing pass, no
regressions.

### Session 02: Pre-flight smoke

Follow `session_prompts/02_smoke_test.md`. Roughly:

1. Write `tools/smoke_direction_gate.py`.
2. Run it on 2026-05-04 with uniform-random policy.
3. Read the diagnostic — check against
   `hard_constraints.md` §3 thresholds.
4. If smoke PASSES: log success, commit smoke tool, move to
   Session 03.
5. If smoke FAILS: write diagnostic to log, STOP loop.

### Session 03: Launch cohort

Follow `session_prompts/03_launch_cohort.md`. Roughly:

1. Pick a fresh cohort tag (`_predictor_SCALPING_dirgate_
   <unix_timestamp>`).
2. Launch the cohort process in background.
3. Verify Generation 1 starts in the log within 30 min.
4. Arm watcher (`auto_reeval_dirgate_cohort.sh`) — copy from
   `auto_reeval_pwingate_cohort.sh` and update paths/tag.
5. Launch status updater
   (`tools/show_cohort_status --watch 60`).
6. Log success; sleep 1h heartbeat until completion.

The cohort takes ~12h. The loop checks status hourly; the
watcher auto-fires the reeval at 96 rows.

### Session 04: Compare + verdict

Follow `session_prompts/04_compare_and_verdict.md`. Roughly:

1. Confirm reeval JSONL exists (
   `<cohort_dir>/reeval_held_out_2026-04-28_30.jsonl`).
2. Compute comparison vs pwin-gate cohort.
3. Write `findings.md` (template in 04 prompt).
4. Commit.
5. STOP loop.

## Default decisions (no operator)

The following defaults apply WITHOUT asking. Make these calls
yourself:

| Question | Default |
|---|---|
| Pwin thresholds | back=0.20, lay=0.40 (same as predecessor) |
| Drift fire rule | gate OPEN_LAY only; OPEN_BACK untouched |
| Cohort size | 12 agents × 8 generations × 6 days |
| Seed | 42 |
| Mutation rate | 0.2 |
| Enabled genes | same 6 safety genes as pwin-gate |
| Eval window | 2026-04-28/29/30 (locked by constraint §5) |
| Smoke day | 2026-05-04 (first day of pwin-gate cohort's eval set) |
| Status refresh interval | 60s |
| If a test fails | Fix in same iteration; if can't fix, try once more; stop on third failure |
| If a launch fails | Read traceback; fix once; if still failing, stop |

## What NOT to do

- Do NOT re-tune pwin thresholds or any gene during this plan.
  hard_constraints §4.
- Do NOT add new shaping reward terms.
  hard_constraints §6.
- Do NOT use the shorten signal anywhere.
  hard_constraints §9.
- Do NOT silence a failing test. If a test fails, diagnose and
  fix. The byte-identity regression test is load-bearing.
- Do NOT push to origin. Commit locally only.
- Do NOT proceed past Session 02 if the smoke fails its
  thresholds.

## What you SHOULD do

- Commit at clean session boundaries.
- Log per iteration to `autonomous_run_log.md`.
- Use `git add` with specific files, never `git add .`.
- Use the Bash tool with `run_in_background=True` for long
  processes (smoke, cohort, watcher, status updater).
- Read the cohort `status.txt` file rather than re-running
  ad-hoc status python.

## Log entry template

Append to `autonomous_run_log.md`:

```markdown
## YYYY-MM-DD HH:MM — Session NN, iteration M

**State entering iteration:** one sentence.
**Work done:** bullet list (file paths, test names, command lines).
**Tests run:** what was run, what passed, what failed.
**Decisions made:** any defaults applied; brief justification.
**Outstanding for this session:** what's left before session
done.
**Next iteration's focus:** specific concrete next step.
```

## Lessons-learnt

Append to `lessons_learnt.md` (create on first appearance) only
non-obvious discoveries:

- Unexpected env behaviour that bit during implementation
- Calibration drift in the predictor between training corpus
  and held-out window
- Cohort failure modes not in the predecessor cohort

Routine progress notes go in `autonomous_run_log.md`, not
`lessons_learnt.md`.

## After plan exit

When Session 04 commits `findings.md`:

1. Surface the headline result in one paragraph (success bar
   pass/fail, mean held-out vs predecessor).
2. Recommend next plan from the README's "What success looks
   like" branches.
3. Stop scheduling.

The operator will read `findings.md` when they next interact.
