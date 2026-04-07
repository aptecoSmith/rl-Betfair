# Session 13 — Promote `mini_batch_size` and `ppo_epochs` to genes

## Before you start — read these

- `../purpose.md`
- `../hard_constraints.md` — constraint 7 (sampled ≠ used) applies
- `../master_todo.md` (Session 13)
- `../progress.md` — confirm Session 11's results justify this work.
  If the Session 11 analysis did not flag PPO inner-loop variance as
  worth exploring, re-read `next_steps.md` entry #3 and reconsider
  whether this session is still warranted.
- `../lessons_learnt.md`
- `../../arch-exploration/session_2_ppo_schema.md` — the session
  that **explicitly excluded** these two genes, with reasoning.
  Read its "Out of scope" block carefully.
- `../ui_additions.md`
- `../initial_testing.md`

## Goal

Add `mini_batch_size` and `ppo_epochs` to the genetic search space
as discrete choice genes, with narrow safe ranges. Leave
`max_grad_norm` out unless Session 11 results say otherwise (the
backlog says it's the least interesting of the three).

## Why these two were originally excluded

Quoting Session 2 of arch-exploration: *"mini_batch_size interacts
with the rollout length, and clip_epsilon + learning_rate together
can blow up training if both are extreme."* Now that we have an
honest Session 11 baseline, we know what "stable" looks like. That
lets us pick narrow, safe ranges instead of avoiding the knobs
entirely.

## Scope

**In scope:**

1. **`mini_batch_size`** as an `int_choice` gene. Candidate values:
   `{32, 64, 128}`. Do **not** use `int` with a free range — the
   interaction with GPU memory and rollout length is non-linear
   and unsafe to mutate smoothly.

2. **`ppo_epochs`** as an `int` or `int_choice` gene. Candidate
   range: `{2, 3, 4, 5, 6}` or `int` in `[2, 6]`. Narrow. Higher
   values waste compute without obvious return; lower values lose
   the benefit of PPO's clipping.

3. **Rollout-length sanity.** Before adding either gene, confirm
   the current `rollout_length` × `mini_batch_size` math still
   works at the extremes. If any `(rollout_length, mini_batch_size)`
   combination produces `mini_batch_size > rollout_length`, add a
   validator that either clamps or rejects the combination. Do
   this **before** the genes go live, not after Session 11+1
   produces a silently-broken run.

4. **`PPOTrainer.__init__` already reads both keys** via
   `hp.get(..., default)` — verify this is still true, then just
   extend `config.yaml → search_ranges`. No trainer code changes
   should be needed.

5. **`max_grad_norm` decision, recorded.** Look at the Session 11
   analysis. If gradient-norm variance was flagged as correlated
   with fitness, add it as a third gene in this session (range
   `[0.3, 1.0]`). If not, explicitly record the decision to skip
   it in this session's `progress.md` entry.

**Out of scope:**

- Any change to rollout length itself. That is a separate,
  bigger question.
- Any change to the optimiser or learning-rate schedule (that's
  Session 17).
- Widening the narrow safe ranges "because the narrow ones feel
  arbitrary". If after Session 14+ runs the ranges turn out to be
  too tight, widen them in a follow-up session with data to back
  it up.

## Tests to add

Create `tests/next_steps/test_ppo_risky_knobs.py`:

1. **Gene sampling.** Both (or all three if `max_grad_norm` is in)
   genes present and in range.

2. **Trainer round-trip.** Construct `PPOTrainer` with extreme
   values; assert `trainer.mini_batch_size` and
   `trainer.ppo_epochs` reflect them. This is the "sampled ≠ used"
   assertion from constraint 7.

3. **Mutation stays in the discrete set.** Mutate 200 times;
   assert every value is still in the allowed `int_choice` set.

4. **Rollout-length validator.** If validator was added in scope
   item 3, test it: valid combinations pass, invalid
   combinations are rejected or clamped consistently.

5. **Backward compat.** A checkpoint without these genes loads
   and trains with the old hardcoded defaults (`ppo_epochs=4`,
   `mini_batch_size=64`).

All CPU, all fast. No training loops.

## Manual tests

- **M1 (UI smoke)** — confirm the new choice editors appear in
  the plan editor and match the discrete sets.

## Session exit criteria

- All tests pass.
- Rollout-length sanity check committed before the genes go live.
- `max_grad_norm` decision recorded in `progress.md`.
- `ui_additions.md` Session 13 entries added (scroll down — there
  is already a placeholder).
- `lessons_learnt.md` updated if anything about the rollout-length
  interaction was surprising.
- `master_todo.md` Session 13 ticked.
- Commit.

## Do not

- Do not use free-range integers for these genes. Discrete sets
  only. The whole reason these were deferred is that smooth
  mutation over them is unsafe.
- Do not add `max_grad_norm` without a reason from Session 11
  data.
- Do not bundle this with Session 14 (arch-specific ranges). One
  gene-schema change per session keeps the Session 11+ analysis
  tractable.
- Do not widen to `mini_batch_size: 256` or `512`. The GPU memory
  ceiling on the 3090 used in the shakeout will bite. If a future
  run proves it fits, widen then.
