# Session 22 — P1d: P1 re-train and decision-gate comparison

## Before you start — read these

- `../purpose.md`
- `../hard_constraints.md` — constraint 9 (CPU-only during
  development) does NOT apply to this session: this is an
  integration/training session and may use the GPU if
  available. But constraint 11 (no full training runs in unit
  tests) still applies — nothing here goes into the fast
  feedback set.
- `../proposals.md` P1
- `../master_todo.md` Phase 1 decision gate
- `../open_questions.md` Q3 — **eval metric must be settled
  before this session starts.** If Q3 is still open, stop and
  get the operator to answer it.
- `../progress.md` — confirm sessions 19, 20, 21 have landed.
- `../integration_testing.md` — this session adds to it.
- `../lessons_learnt.md`

## Goal

Train one policy on the new P1 obs vector (OBI + microprice +
traded_delta + mid_drift) and compare it against the pre-P1
baseline on the held-out 9-day eval window. This is the
**decision gate at the end of Phase 1**: the result informs
whether Phase 1 continues to P2 / P5, or whether the gains are
already large enough to stop and skip Phase 2 entirely.

This is **not** a feature-adding session. No new feature goes in.
The only code changes permitted are (a) eval-script tweaks to
surface the metric from Q3, and (b) bug fixes for anything that
breaks during the run.

## Inputs — constraints to obey

1. **Eval metric is whatever Q3 resolved to.** Do not invent
   one. If Q3 says "raw P&L on eval window" then that's the
   number; if Q3 says "Sharpe-ish", use that. Record the metric
   and the exact value for both policies in `progress.md`.
2. **Same hyperparameters for both runs.** The only difference
   between baseline and P1 policies is the obs vector. Reuse
   the most recent production hyperparameters from session 11's
   real run (or the latest available).
3. **Fresh init for both.** Do not warm-start the P1 policy
   from the baseline weights — the obs dim changed, so the
   input layer is different, and warm-starting would require
   a layer-surgery step that is out of scope.
4. **Gradient check on the new columns.** Early in training
   (after N gradient steps, pick N), assert that the gradient
   norm on the *new* input-layer columns is non-zero. If it's
   zero, the features are being ignored and something is wired
   wrong.

## Steps

1. **Confirm Q3 resolution.** Read `open_questions.md`. If Q3
   is unresolved, stop. Otherwise write the chosen metric into
   the top of this session file under a `## Q3 resolution`
   heading.

2. **Baseline run.** Train one policy on the pre-P1 obs vector
   using the same hyperparameters you will use for the P1 run.
   If a baseline already exists from session 11 or later on
   the same git SHA, reuse it — but confirm it was trained with
   the same hyperparameter set.

3. **P1 run.** Train one policy on the new obs vector. Log the
   gradient-norm sanity check. If it fires (gradient norm on
   new columns is zero after N steps), stop and debug before
   continuing.

4. **Eval both policies on the 9-day eval window.** Use the
   existing evaluator. Record per-day numbers, not just the
   aggregate.

5. **Write up the comparison in `progress.md`.** Include:
    - The eval metric (from Q3).
    - Per-day numbers for both policies.
    - Aggregate number for both.
    - Whether the P1 policy is better, worse, or noise-level.
    - A one-line recommendation: *continue to P2*, *stop Phase
      1 here*, or *P1 made things worse, investigate before
      continuing*.

6. **Write up the surprises in `lessons_learnt.md`.** This
   session exists to produce data; the data almost always
   contains at least one thing nobody predicted. Write it
   down.

## Tests added

To `integration_testing.md`:
- **P1 training run on 1-day fixture.** Asserts gradient norm
  on new columns is non-zero after N steps.
- **Comparison run.** P1 vs baseline on the 9-day eval
  window; results recorded in `progress.md`. Not pass/fail —
  this is a data-collection test.

No new unit tests. P1 unit tests landed in sessions 19–21.

## Manual tests

- **Open the evaluator UI (if one exists) and spot-check three
  races.** Confirm the P1 policy is making plausibly-different
  decisions on races where OBI or traded_delta was strongly
  signed. If the policy is indistinguishable from the baseline
  on visibly-informative ticks, the features may be too noisy
  to use or the training run was too short. Note either way.

## Session exit criteria

- Both policies trained.
- Comparison numbers recorded in `progress.md`.
- Recommendation ticked in `master_todo.md` Phase 1 decision
  gate (one of: *continue*, *stop*, *regress-investigate*).
- `lessons_learnt.md` entry with at least one surprise. If
  literally nothing surprised you, the entry says so
  explicitly — that's itself a data point.
- Commit.

## Do not

- Do not add new features in this session. The comparison is
  only meaningful if the only variable is "obs dim".
- Do not tune hyperparameters per policy. Same HP for both.
- Do not skip the gradient-norm sanity check. A zero-gradient
  column is a silent bug that makes the whole comparison
  worthless.
- Do not short-cut by warm-starting P1 from the baseline.
- Do not decide the continue/stop question without writing
  the reasoning into `progress.md`. A bare tick box with no
  rationale is worse than a hesitant sentence.
- Do not begin session 23 (P2) before the recommendation is
  recorded. If you do, you are pre-empting the gate.
