# Session 2 — Expand PPO hyperparameter schema

## Before you start — read these

- `plans/arch-exploration/purpose.md`
- `plans/arch-exploration/master_todo.md` (you are on Session 2)
- `plans/arch-exploration/testing.md` — **CPU-only, fast feedback,
  test after each feature.**
- `plans/arch-exploration/progress.md` — confirm Session 1 is done.
  This session assumes reward plumbing is fixed.
- `plans/arch-exploration/lessons_learnt.md`
- `plans/arch-exploration/ui_additions.md` — append the new knobs to
  the Session 2 checklist before finishing.
- Repo root `CLAUDE.md`.

## Goal

Promote `gamma`, `gae_lambda`, and `value_loss_coeff` from hardcoded
defaults to mutable genes in the genetic search schema. These are
already read by `PPOTrainer` via `hp.get(..., default)` — this session
is schema-only plus tests.

## Scope

**In scope:**
- Add the three genes to `config.yaml` search_ranges.
- Confirm `sample_hyperparams` handles them without code changes (they
  are all `float` type).
- Confirm `PPOTrainer.__init__` still reads them correctly via existing
  `hp.get` calls.
- Tests.
- Append to `ui_additions.md` Session 2 checklist if anything was
  missed.

**Out of scope:**
- Do NOT add `mini_batch_size`, `ppo_epochs`, or `max_grad_norm` as
  genes. These interact with rollout length and GPU memory — risky.
- Do not touch reward shaping.
- Do not touch architectures.

## Ranges to use

| Gene | Type | Range | Default in config |
|---|---|---|---|
| `gamma` | float | [0.95, 0.999] | 0.99 |
| `gae_lambda` | float | [0.9, 0.98] | 0.95 |
| `value_loss_coeff` | float | [0.25, 1.0] | 0.5 |

Match the YAML shape of existing float entries (`learning_rate`,
`ppo_clip_epsilon`, `entropy_coefficient`). Do not invent a new schema
format.

## Tests to add

Create `tests/arch_exploration/test_ppo_schema.py`:

1. **All three genes are in the sampled dict.** Call
   `sample_hyperparams` with a fixed RNG seed and assert `gamma`,
   `gae_lambda`, and `value_loss_coeff` are present with values in
   their declared ranges.

2. **Extreme-value round-trip.** Construct a `PPOTrainer` with
   `hyperparams={"gamma": 0.951, "gae_lambda": 0.901,
   "value_loss_coeff": 0.26, ...}`. Assert
   `trainer.gamma == 0.951`, etc. This is the "gene is actually read"
   test that `lessons_learnt.md` says every new gene must have.

3. **Mutation stays in range.** Seed an RNG, sample a genome, mutate
   it N times (e.g. 200), and assert the mutated values remain in
   range. Catches off-by-one clamping bugs.

All three tests should run in under one second total. No GPU, no
training loops.

## Session exit criteria

- Tests pass locally.
- `progress.md` Session 2 entry added.
- `lessons_learnt.md` — add only if something surprising happened.
- `ui_additions.md` — Session 2 items already there; tick them off
  once the server-side is live (the UI itself is built in Session 8).
- Commit.

## Do not

- Do not widen the search ranges "while you're there". The proposed
  ranges were chosen in design review. Changing them mid-session
  invalidates the coverage planning in Session 4.
- Do not remove the `hp.get(..., default)` fallbacks in
  `ppo_trainer.py`. They exist so agents loaded from old checkpoints
  without these genes still work.
