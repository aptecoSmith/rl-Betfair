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
2. **An operator-decision point is hit** (per
   `master_todo.md`'s "After Session NN" sections — e.g. after
   Session 01 the sibling-import vs install-package decision;
   after Session 05 the reward-shape escalation if smoke
   produces zero gradient signal). Surface the decision
   explicitly to the operator and stop.
3. **The byte-identical regression test fails** and the cause
   is non-obvious. Stop. Do NOT silence the test or weaken its
   assertion. Log the failure, summarise to the operator, and
   await guidance.
4. **A hard_constraint is about to be violated** to make
   progress (e.g. Session 05 smoke produces no gradient signal
   and the temptation is to add shaping per §3 — that's a stop
   condition, not a unilateral fix).
5. **The cohort runs in Sessions 05/06/07 require GPU time the
   operator hasn't committed.** Surface the wall-clock estimate
   and stop; the operator launches the cohort.
6. **Three iterations on the same sub-step without progress.**
   Stop and ask. Do not fight a problem indefinitely.

## Pacing

`ScheduleWakeup` `delaySeconds`:

- **60–270s (cache-warm)** — when actively iterating on code
  and tests; the next iteration is doing the next file edit
  or running the next test.
- **1200–1800s (idle window)** — when waiting on a real cohort
  run (Sessions 05, 06, 07) to finish on the GPU host. The
  cohort emits log files; the next iteration reads the latest
  log and decides whether to continue.
- **Operator-asleep windows** — never schedule across an
  expected operator-asleep window (typical UK overnight).
  Sessions 05/06/07 cohorts are operator-launched; they run
  while the operator is awake to monitor.

Pass the **same /loop input** verbatim back to ScheduleWakeup
each iteration:
`/loop @plans/predictor-integration/session_prompts/00_autonomous_full_run.md`.

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
- Do NOT run cohorts (Sessions 05/06/07) without operator
  go-ahead. Cohort runs cost GPU-hours.
- Do NOT push to `origin master` mid-loop. Commit locally;
  the operator pushes when they review the work.
- Do NOT modify `data/extractor.py` or `data/episode_builder.py`
  in this plan. EW data is already in the parquet pipeline.

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
