# Session 17 — Optimiser and LR schedule work

## Before you start — read these

- `../purpose.md`
- `../hard_constraints.md`
- `../master_todo.md` (Session 17)
- `../progress.md` — this session is **gated on Session 11
  results**. If the analysis did not show transformer agents (or
  any arch) under-training relative to equivalent-capacity
  peers, reconsider whether this session is warranted. An
  optimiser session without a concrete problem to solve is
  speculative work.
- `../lessons_learnt.md`
- `../../arch-exploration/session_6_transformer_arch.md` — the
  session that explicitly excluded optimiser changes ("LR warmup,
  weight decay, any optimiser change. The existing Adam/AdamW and
  single-LR setup stays.")
- `../ui_additions.md`
- `../initial_testing.md`

## Goal

Give the PPO trainer a simple learning-rate schedule — at minimum
linear warmup — and optionally support weight decay as a gene.
Primary motivation is transformer agents: they typically want LR
warmup and the current setup gives them none.

## Scope decision at session start

**Before writing code, decide** which of these are in scope for
this session based on Session 11 evidence. Record the decision in
`design_decisions.md`:

- **(a) Linear LR warmup** — `warmup_steps` applied to every
  architecture, with value 0 defaulting to "no warmup" so existing
  runs are unchanged.
- **(b) Weight decay** — `weight_decay` as a gene, mapped to
  `torch.optim.AdamW`'s `weight_decay` arg. Requires switching
  default optimiser from Adam to AdamW.
- **(c) Cosine / linear decay after warmup** — non-trivial,
  interacts with the `gamma` gene's discounting; probably a
  separate session.

Default recommendation: do **(a)** unconditionally, do **(b)**
only if Session 11 flagged overfitting, defer **(c)** unless
Session 11 showed clear late-training instability. Commit the
decision with rationale.

## Scope

**In scope (assuming default (a) + maybe (b)):**

1. **LR warmup.** Add `warmup_steps` as an `int` gene in
   `config.yaml` with range `[0, 2000]`. Default 0 (no warmup).
   Implementation: a `torch.optim.lr_scheduler.LambdaLR` wrapping
   the optimiser, with a warmup function that ramps LR linearly
   from `0` to the sampled `learning_rate` over `warmup_steps`
   training steps, then holds constant.

2. **Arch-aware default.** If (b) from Session 14 shipped,
   `warmup_steps` should be settable per-architecture so
   transformers can default to a higher warmup than LSTMs. Reuse
   the Session 14 plumbing. If Session 14 is not yet done,
   document the dependency in `progress.md` and ship a
   single-range version now.

3. **Weight decay (optional).** Only if Session 11 evidence
   supports it:
   - Switch optimiser from `Adam` to `AdamW` as default.
   - Add `weight_decay` as a `float_log` gene in
     `[1e-6, 1e-3]`, default 0 (equivalent to Adam behaviour).
   - Test that `weight_decay=0` reproduces the pre-session
     training curves on a CPU smoke test.

4. **Step counting.** The warmup scheduler needs to know the
   current step. `PPOTrainer` already has a rollout-based step
   counter; wire the scheduler to that. Confirm what exactly
   counts as "one step" — per-minibatch gradient step, or per-
   rollout? Document the choice in `design_decisions.md`.

**Out of scope:**

- Cosine / polynomial decay schedules.
- Changing the optimiser beyond AdamW.
- Gradient accumulation.
- Mixed-precision training.
- Any change to `learning_rate` range itself (that's still the
  same gene).

## Tests to add

Create `tests/next_steps/test_optimiser_warmup.py`:

1. **Gene sampling.** `warmup_steps` (and `weight_decay` if in
   scope) present and in range.

2. **Zero-warmup backward compat.** `warmup_steps=0` produces the
   same LR at step 0 as before this session. Important — the
   default must be a no-op.

3. **Warmup ramp.** With `warmup_steps=100` and
   `learning_rate=1e-4`, the optimiser's LR at step 0 is 0, at
   step 50 is ~5e-5, at step 100 is 1e-4, at step 200 is still
   1e-4. Tolerances are tight — these are scheduler outputs, not
   training outputs.

4. **Mutation stays in range.** Standard mutation test.

5. **AdamW equivalence at `weight_decay=0`.** If (b) shipped,
   test that `AdamW(weight_decay=0)` produces bitwise-identical
   gradient updates to `Adam` for a fixed input. If they differ
   (due to numerical differences in the optimiser), document
   the tolerance and why it's acceptable.

6. **Scheduler + rollout interaction.** Given a trainer that
   takes N rollout updates, step through them and assert the
   scheduler's `last_epoch` / step count matches what you'd
   expect from the rollout counter.

All CPU, no GPU. Use a tiny model for the AdamW test so it's
instant.

## Manual tests

- **M1 (UI smoke)** — new gene(s) editable in the plan editor.

## Session exit criteria

- Scope decision committed in `design_decisions.md` before code.
- All tests pass.
- `warmup_steps=0` default means existing runs are byte-identical
  (or documented and justified if they are not).
- `ui_additions.md` Session 17 entries added.
- `lessons_learnt.md` updated — scheduler plumbing tends to
  surface subtle step-counting bugs.
- `master_todo.md` Session 17 ticked.
- Commit.

## Do not

- Do not ship an LR schedule that changes default behaviour
  unasked. `warmup_steps=0` → no warmup → same as before.
- Do not switch to AdamW unless weight decay is actually in
  scope. The optimisers differ numerically even at
  `weight_decay=0` (AdamW decouples weight decay from the
  gradient), so a gratuitous switch is a silent behaviour
  change.
- Do not add a decay schedule "because warmup implies decay".
  Warmup-only is fine and simpler.
- Do not gate this session on Session 16. They are independent.
- Do not gate this session on Session 14 unconditionally — fall
  back to a single-range version if Session 14 hasn't landed.
