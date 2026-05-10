# 00 — Autonomous full run

You are driving the **predictor-integration** plan in
`C:\Users\jsmit\source\repos\rl-betfair\plans\predictor-integration\`
to completion. Each invocation of this prompt is one iteration —
read state, do focused work toward the next deliverable, log
progress, then stop. The `/loop` runtime re-fires this prompt;
each fire is one block of work.

The deliverable is a working integration of the
`betfair-predictors` production champions into v2's observation
space + a strategy-mode switch (`arb` / `value_win` /
`value_each_way`) + a three-way comparison verdict.

## Read FIRST every iteration

These are short. Read them every iteration so you stay aligned
even if context is fresh:

1. `plans/predictor-integration/README.md` — the plan summary,
   recommendation, and phasing table.
2. `plans/predictor-integration/hard_constraints.md` — the 13
   cross-session invariants. **Treat these as inviolable.**
3. `plans/predictor-integration/master_todo.md` — session
   table + per-session-boundary operator decisions.
4. `plans/predictor-integration/autonomous_run_log.md` — your
   own progress log (created on iteration 1 if absent). The
   "where am I" anchor. **Tail it before deciding what to do.**

These are referenced by the session you're working on:

- The relevant `session_prompts/0N_*.md` for the current
  session.
- `plans/predictor-integration/predictor_contracts.md` (for
  Sessions 01, 02, 04).
- `plans/predictor-integration/strategy_modes.md` (for Sessions
  03, 04, 05, 06).
- `plans/predictor-integration/integration_contract.md` (for
  Sessions 01, 02, 03, 04).

## Each iteration

Steps in order:

1. **Read state.** Read `autonomous_run_log.md`. Identify the
   current session (e.g. "Session 02 in progress, awaiting
   byte-identical regression test"). If the log is empty, you
   are on Session 01.
2. **Check stop conditions** (see "Stop conditions" below). If
   any apply, write a final entry to the log, summarise to the
   operator, and DO NOT call ScheduleWakeup. Loop ends.
3. **Pick the next concrete deliverable.** This is one
   sub-step of the current session — typically one file edit
   or one test that needs to pass. Sessions are bounded; pick
   whichever bounded chunk is next.
4. **Do the work.** Edit code; run tests; verify locally with
   `pytest <specific-tests>` or whatever the session prompt's
   success bar says.
5. **Log progress.** Append a dated entry to
   `autonomous_run_log.md` (template below). Include: what was
   done; what passed; what's left for the session; what to
   do next iteration.
6. **Decide whether the session is done.** If the session's
   success bar (per `master_todo.md` / the session prompt) is
   met AND the byte-identical regression test still passes,
   commit the session's work with a clear message and move to
   the next session.
7. **Schedule next iteration.** Call `ScheduleWakeup` with a
   delay appropriate to what's outstanding (see "Pacing"
   below).

## Log entry template

Append to `autonomous_run_log.md`:

```markdown
## YYYY-MM-DD HH:MM — Session NN, iteration M

**State entering iteration:** one sentence.
**Work done:** bullet list (file paths, test names).
**Tests run:** what was run, what passed, what failed.
**Outstanding for this session:** what's left before session
done.
**Next iteration's focus:** specific concrete next step.
**Operator decisions pending:** any (per master_todo.md
"After Session NN" sections); blank if none.
```

## Stop conditions

Loop ends (do NOT schedule next wake) when ANY of:

1. **Session 07 findings.md is written and committed.** Plan
   is complete.
2. **The byte-identical regression test fails** and the cause
   is non-obvious. Stop. Do NOT silence the test or weaken its
   assertion. Log the failure, summarise to the operator, and
   await guidance.
3. **A hard_constraint is about to be violated** to make
   progress (e.g. Session 05 smoke produces no gradient signal
   and the temptation is to add shaping per §3 — that's a stop
   condition, not a unilateral fix).
4. **A smoke cohort fails its success bar** in a way that
   indicates the integration is genuinely broken (e.g. zero
   bets across all 12 agents on Session 05's value-win smoke;
   `is_each_way == True` on < 50% of bets in Session 06 paired
   with zero P&L signal). The cohort completed, but the
   verdict is "the wiring is broken." Stop and diagnose; do
   NOT proceed to the next mode's cohort or to Session 07.
5. **Three iterations on the same sub-step without progress.**
   Stop and ask. Do not fight a problem indefinitely.

**NOTE — operator-decision points are NOT stop conditions.**
Per the operator's autonomous-run mandate, default to the
recommendation in `master_todo.md`'s "After Session NN"
sections, log the choice in `autonomous_run_log.md`, and
continue. The recommendations:

- After 01: sibling-import via `sys.path.insert` (don't
  invest in installable package yet).
- After 02: `use_direction_predictor` opt-in per cohort,
  default off until per-tick cost is profiled.
- After 03: new genes always present at default 0 in
  `CohortGenes.to_dict()` (Path A pattern).
- After 04: back-only EW for the smoke (lay-EW is a follow-on).
- After 05: no shaping added even if signal looks sparse —
  that's a hard_constraints §3 violation, escalate via stop
  condition §3 above.
- After 06: same — no place-specific reward.

If a decision point genuinely demands operator input (i.e. the
recommendation doesn't apply because reality diverged from the
plan's assumptions), THAT triggers stop condition §3 above.

## Pacing

`ScheduleWakeup` `delaySeconds`:

- **60–270s (cache-warm)** — when actively iterating on code
  and tests; the next iteration is doing the next file edit
  or running the next test.
- **1200–1800s (idle window)** — when waiting on a cohort run
  (Sessions 05, 06, 07) to make progress. The cohort emits log
  files; the next iteration tails the latest log, checks
  process state, and decides whether to continue waiting or
  the cohort has completed.
- **3600s (max)** — only when a cohort is mid-generation and
  the next interesting log line is known to be > 30 minutes
  away. Don't burn iterations on dead-air.

Pass the **same /loop input** verbatim back to ScheduleWakeup
each iteration:
`/loop @plans/predictor-integration/session_prompts/00_autonomous_full_run.md`.

## Cohort run discipline (Sessions 05, 06, 07)

The loop launches cohorts directly. **No operator sign-off
required** — the operator has authorised autonomous training
runs for this plan. Discipline:

1. **Launch via the existing CLI.** Look at
   `run_phase8_overnight.sh`, `training_v2/discrete_ppo/train.py`,
   and `training_v2/cohort/worker.py` for the canonical
   invocation. Use `--device cuda` (per the operator's
   memory: always GPU for cohorts). Tee the launch log into
   `registry/<cohort_tag>.log` so the next iteration can tail
   it.
2. **Run in background.** Launch via `run_in_background=True`
   on the Bash tool (or a Monitor task arm if it suits).
   Don't block on the cohort; schedule a 1200s wakeup.
3. **Monitor by tailing the log.** Each iteration:
   - Tail the last ~200 lines of the cohort log.
   - Check whether the latest generation marker has advanced
     since the previous iteration.
   - Check for crash markers (`Traceback`, `CUDA out of
     memory`, `Process exited with code`).
   - If still running and no crash: schedule next wakeup;
     log "gen N/6 in progress" and stop the iteration.
   - If completed cleanly: read the registry row + episode
     JSONL; evaluate against the smoke success bar from
     `session_prompts/0N_*.md`; commit findings; move on.
   - If crashed: stop (this is stop condition §4 — verify
     it's a real failure, log the trace, await operator).
4. **Wall-clock estimates.** Smokes are ~4 hours each on
   GPU. Real cohorts (Session 07's three concurrent runs)
   are ~6–12 hours total. The loop can absorb either at the
   1200–3600s wake cadence; don't try to compress.
5. **Concurrent cohorts (Session 07).** Launch all three
   modes' cohorts in parallel as separate background processes
   (the operator's GPU host has the headroom — see Phase 8's
   parallel-cohort precedent in `run_phase8_overnight.sh`).
   Each iteration tails all three logs.
6. **Crash recovery.** If one of three Session-07 cohorts
   crashes, stop ALL three (don't let the survivors finish
   and produce a misleading two-mode comparison). Diagnose;
   surface to operator.

## What to commit, when

Commit at clean session boundaries:

## What to commit, when

Commit at clean session boundaries:

- After Session 01: predictor loader + tests pass.
- After Session 02: obs wiring + byte-identical regression
  test passes.
- After Session 03: strategy-mode switch + three smoke tests
  pass (value_each_way smoke skipped if Session 04 hasn't
  landed).
- After Session 04: each-way action surface + EW settlement
  tests pass.
- After Session 05: value-win smoke cohort findings.
- After Session 06: value-each-way smoke cohort findings.
- After Session 07: three-way comparison findings.md.

Use the project's commit-message convention from `git log`:
prefix with `feat(predictor-integration)`,
`test(predictor-integration)`, `docs(predictor-integration)`,
or `fix(predictor-integration)` as appropriate. Reference the
session number. Co-Authored-By footer per repo convention.

## Hard constraints — no shortcuts

These are restated for emphasis. From
`hard_constraints.md`:

1. Flag-off byte-identical to pre-plan (regression test
   ALWAYS passes).
2. Don't touch env / matcher / bet_manager mechanics.
3. Don't re-derive EW settlement (`plans/ew-settlement/`
   complete; reuse the path verbatim).
4. No new shaped reward terms in value modes.
5. Predictor weights FROZEN.
6. Predictor `experiment_id` captured in every cohort row.
7. Three modes trained separately, evaluated jointly.
8. Predictor outputs are observations, not authority.
9. Loader robustness: no silent fallbacks.
10. Don't refactor `agents_v2/discrete_policy.py` (no policy
    shape change).
11. Don't retire the v2 internal scorer or aux heads in this
    plan.
12. Don't expand scope. Out-of-scope items go to `incoming/`
    of the appropriate target repo.

## Operator-override discipline

If you genuinely need to violate a hard_constraint to make
progress (e.g. the integration reveals an env bug that must be
fixed before Session 05 can run), DO NOT just fix it
inline. Instead:

1. Stop the loop (this iteration is the last one).
2. Write the diagnosis to `autonomous_run_log.md` under a
   "Operator override needed" heading.
3. Surface to the operator with a specific question:
   "Session 0N requires touching <module> to fix <issue>;
   `hard_constraints.md` §N forbids it. Fix it as a separate
   plan first, or temporarily lift the constraint here?"
4. Await guidance.

## Lessons learnt accumulation

Per the convention from
`plans/rewrite/phase-7-port-aux-heads/lessons_learnt.md`,
append a `lessons_learnt.md` entry whenever you discover a
non-obvious thing during execution — failure modes that
weren't anticipated in the plan, design choices that turned
out load-bearing, etc. Cross-cutting lessons get promoted to
CLAUDE.md when the plan exits.

## What you should NOT do

- Do NOT skip reading `hard_constraints.md` because it feels
  repetitive. The constraints exist because the codebase has
  been bitten by exactly the issues they describe.
- Do NOT batch sessions. Each session has its own commit and
  its own success bar.
- Do NOT silence a failing test. If a test fails, diagnose
  the cause; if it's a real failure, stop and surface to the
  operator. The byte-identical regression test is the
  load-bearing one.
- Do NOT push to `origin master` mid-loop. Commit locally;
  the operator pushes when they review the work.
- Do NOT modify `data/extractor.py` or `data/episode_builder.py`
  in this plan. EW data is already in the parquet pipeline.
- Do NOT add new shaping terms to value modes if signal looks
  sparse. That's a hard_constraints §3 violation; stop and
  escalate.
- Do NOT proceed to Session 06 if Session 05's smoke fails its
  success bar, or to Session 07 if Session 06's smoke fails.
  A failed smoke means the integration is broken; cascading
  to the next mode wastes GPU and produces misleading data.

## First iteration: bootstrap

If `autonomous_run_log.md` does not yet exist, create it with
this header and start Session 01:

```markdown
# Predictor-integration autonomous run log

This file tracks per-iteration progress through the
predictor-integration plan. Newest entries at the bottom.
The autonomous-run prompt
(`session_prompts/00_autonomous_full_run.md`) reads this file
to decide what to do next.

```

Then read `session_prompts/01_predictor_loader.md` and start
on its first deliverable: `predictors/__init__.py` +
`predictors/loader.py`.

## After plan exit

When Session 07 lands and findings.md is committed:

1. Update `plans/INDEX.md` row 21's "(latest)" tag if a new
   plan supersedes it.
2. Promote lessons_learnt to CLAUDE.md per the
   `plans/INDEX.md` convention.
3. Set `plan: predictor-integration` `status: closed` in
   `README.md` frontmatter.
4. Surface the plan-level verdict to the operator.

Do NOT schedule another wake. The loop is done.
