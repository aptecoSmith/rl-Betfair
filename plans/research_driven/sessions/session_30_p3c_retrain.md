# Session 30 — P3c: Phase 2 re-train + diversity check + decision gate

## Before you start — read these

- `../purpose.md`
- `../hard_constraints.md` — constraint 11 (no full training
  runs in unit tests) still applies; this session is the
  training session itself.
- `../proposals.md` P3 and P4 acceptance criteria
- `../master_todo.md` Phase 2 decision gate
- `../open_questions.md` Q3 — the same eval metric used in
  session 22 applies here. If Q3 has been revised, the new
  metric takes precedence but the session 22 results must
  be re-evaluated under the new metric for a fair
  comparison.
- `../progress.md` — confirm sessions 25–29 have all landed
  and P4 + P3 are both fully complete.
- `../integration_testing.md`
- `../lessons_learnt.md`

## Goal

Train one policy with the full research-driven feature set
(P1 obs + P2 shaping + P3 action space + P4 queue model) and
compare it on the held-out 9-day eval window against the
Phase 1 policy from session 22.

This is the **Phase 2 decision gate**. Its outcome decides
whether the execution-aware simulator is worth shipping to
`ai-betfair`, or whether the Phase 1 selection-only policy is
already good enough and the extra code should stay in the
simulator but not in production.

## Inputs — constraints to obey

1. **Fresh init.** Action-space change from session 28
   invalidated all checkpoints. The P3+P4 policy trains from
   scratch. No warm-starting.
2. **Same hyperparameters as session 22** where possible.
   Some may need to be adjusted because the action space is
   larger and exploration is harder — if you change any,
   document which and why in `progress.md`.
3. **Diversity check is mandatory, not optional.** A P3
   policy that collapses to "always aggressive" or "always
   passive" has defeated the point of the work. The
   aggression histogram must be non-trivial in the final
   trained policy. Same for cancel rate — at least one race
   per eval day must have a non-zero cancel count.
4. **Baseline sanity check before comparison.** Before
   running the full training, confirm that with
   `actions.force_aggressive=true` the new codebase
   reproduces the Phase 1 policy's aggressive-only
   behaviour. If that regression check fails, something
   broke in sessions 25–29 that the unit tests missed; stop
   and debug before training.

## Steps

1. **Confirm decision-gate prerequisites.** Sessions 25–29
   all landed. All unit tests green. The Phase 1 baseline
   from session 22 is still loadable and its eval numbers
   are documented in `progress.md`. If any of this is
   missing, stop.

2. **Regression sanity check.** With
   `actions.force_aggressive=true`, run a 1-day fixture
   with the Phase 1 policy's weights (or the session 22
   baseline) and assert the day's bet-by-bet P&L matches
   the pre-P3 baseline exactly. This is the "we didn't
   break aggressive while adding passive" check.

3. **P3+P4 training run.** Train a fresh policy with the
   new action space. Log the aggression distribution and
   cancel rate throughout training — not just at eval time.

4. **Diversity assertions during training.** By step N
   (pick a meaningful mid-training checkpoint), the
   aggression histogram must show at least some fraction
   of passive actions (say ≥ 5 %, or whatever you decide
   is "not collapsed"). If the policy collapses, stop and
   investigate before burning the full training budget.

5. **Eval on the 9-day window.** Under the Q3 metric.
   Record per-day numbers for the P3+P4 policy, the
   Phase 1 P1+P2 policy, and the pre-everything baseline.

6. **Cancel-rate and passive-fill-rate sanity.** In eval,
   at least one race per eval day shows a non-zero cancel
   count, and at least one race per eval day shows a
   non-zero passive fill. If either is flat at zero across
   the whole eval window, the policy is using one regime
   only and the feature has not earned its cost.

7. **Decision-gate writeup.** In `progress.md`, record:
    - The Q3 metric.
    - Per-day eval numbers for all three policies.
    - Aggression histogram summary (mean, variance, mode).
    - Cancel rate summary.
    - Passive fill rate summary.
    - A recommendation, with reasoning, one of:
       - **Ship P3+P4 policy to `ai-betfair`** — the
         execution-aware policy meaningfully beat the
         selection-only policy.
       - **Keep P3+P4 code in the simulator, ship the
         Phase 1 policy** — the code paths are correct
         but the training gain does not justify the
         additional `ai-betfair` deployment cost (§3 of
         `downstream_knockon.md`).
       - **Regression — investigate before shipping
         anything** — the P3+P4 policy underperformed
         both baselines; something is wrong.

## Tests added

To `integration_testing.md`:
- **Phase 2 training run on 1-day fixture with diversity
  assertions.**
- **Regression run** (`force_aggressive=true` reproduces
  Phase 1 policy).
- **Comparison run** on the 9-day eval window.

No new unit tests — unit tests for P3/P4 landed in sessions
25–29.

## Manual tests

- **Watch one eval race with the P3+P4 policy** and confirm
  the operator can see all three regimes in use: an
  aggressive cross, a passive rest-then-fill, and a passive
  cancel. If the operator can't spot all three across 3
  eval races, the diversity assertions may have been too
  lenient; note in `lessons_learnt.md`.
- **Spot-check one race where P3+P4 outperformed Phase 1
  by the largest margin.** What did the P3+P4 policy do
  differently that Phase 1 couldn't? If you can't tell, the
  improvement is noise, not signal.
- **Spot-check one race where P3+P4 underperformed Phase 1
  by the largest margin.** Same question, opposite sign.

## Session exit criteria

- Regression sanity check passed.
- P3+P4 policy trained.
- Diversity assertions passed (aggression not collapsed,
  cancel rate non-trivial, passive fill rate non-trivial).
- Eval numbers recorded in `progress.md` for all three
  policies.
- Recommendation filed in `progress.md` with reasoning.
- `master_todo.md` Phase 2 decision gate ticked with the
  recommendation (one of: ship / keep-code-only / regress-
  investigate).
- `lessons_learnt.md` entry with at least one surprise.
- If the recommendation is *ship*, the
  `downstream_knockon.md` §3 items for `ai-betfair` are
  confirmed queued in the `ai-betfair` repo (cross-repo
  coordination) before deployment can proceed.
- Commit.

## Do not

- Do not ship a new policy to `ai-betfair` if the
  phantom-fill bug (R-1) is still open. Hard gate from
  `hard_constraints.md` #8. The recommendation can still be
  *ship*, but the actual deployment waits.
- Do not skip the regression sanity check to save time.
  That check is specifically there to catch the kind of
  silent break that ships a broken policy and looks fine
  on the eval metric because the bug is in a seldom-
  triggered code path.
- Do not revise the Q3 metric in this session. If Q3 needs
  revising, that's its own operator decision and it has to
  happen *before* this session starts, not during it. A
  session that invents its own success metric is a session
  that justifies its own outcome.
- Do not ship a collapsed policy as successful. An
  aggression histogram flattened to one mode, or a zero
  cancel rate, means the policy is not using the new
  action space — and therefore P3+P4 has *not* earned its
  cost regardless of the eval number.
- Do not begin follow-on research-driven work in this
  session. If the decision gate says *regress-
  investigate*, file the investigation as a new session
  prompt under `sessions/` and stop.
