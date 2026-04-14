# Session 9 — Auxiliary arb-availability head (optional, gated)

## Before you start — read these

- `plans/arb-improvements/purpose.md`
- `plans/arb-improvements/master_todo.md` — Phase 4, Session 9.
- `plans/arb-improvements/testing.md`
- `plans/arb-improvements/hard_constraints.md` — aux head is
  **gated by default**; one oracle feeds both BC and aux;
  `aux_arb_head=False` means no forward-pass change.
- `plans/arb-improvements/progress.md` — **read the end-of-Phase-3
  decision.** If Phase 1 + Phase 2 + Phase 3 on a short run already
  fixed the collapse and arb-rate is healthy, this session may be
  deferred or skipped. Record the decision in `progress.md`
  explicitly; either "Session 9 shipped" or "Session 9 deferred
  because…".

## Goal (if shipping)

Add a supervised prediction head off the policy's shared trunk
that predicts "will an arb be lockable on runner R within the
next K ticks?" — trained on targets generated from the Session 6
oracle. The supervised signal stabilises the trunk representation
even when PPO updates are noisy. Entirely opt-in; default off.

## Scope (if shipping)

**In scope:**

- `training.aux_arb_head` (bool, default `False`).
- `training.aux_arb_head_weight` (float, default `0.1`) — loss
  coefficient.
- `training.aux_arb_head_horizon` (int ticks, default `5`) — how
  far ahead the oracle looks when building targets.
- New small MLP head in each architecture
  (`ppo_lstm_v1`, `ppo_time_lstm_v1`, `ppo_transformer_v1`) reading
  the shared trunk features. Output shape: `(batch, max_runners)`,
  one binary logit per runner. Constructed lazily — only if
  `aux_arb_head=True`.
- Targets computed from oracle scan: for each training sample, the
  target vector is 1 at the position of any runner with an arb
  available within the horizon, 0 elsewhere.
- Aux loss added to PPO total during training. During BC, aux head
  trains on oracle data too (free — the oracle provides the
  targets anyway).
- Logged as a separate loss curve in the monitor progress event.

**Out of scope:**

- Non-arb auxiliary heads (e.g. "predict LTP velocity"). One head
  at a time.
- Changing the forward-pass interface for callers that don't opt
  in. Default off → forward signature unchanged.

## Exact code path

1. Add the three training config knobs; whitelist them through the
   trainer.
2. In each architecture module, add an optional `aux_head` attr
   constructed in `__init__` only if config says so. `forward()`
   returns a tuple `(action_dist_dict, value, aux_logits | None)`.
   Callers that don't pass the aux flag ignore the third return.
3. In `agents/ppo_trainer.py`, when aux is on: compute BCE loss on
   `aux_logits` vs targets, add `aux_head_weight * bce_loss` to
   the total.
4. Targets: precompute per-sample in the rollout buffer. At each
   step, look up whether any runner has an oracle sample at
   `(tick_index, runner_idx)` within the horizon. This requires
   loading oracle samples per training date; reuse the Session 6
   loader.
5. BC pretrainer (Session 7) also trains the aux head when on —
   targets come from the same oracle that produced the BC
   samples, so the data is free.

## Tests to add (CPU-only, fast)

Create `tests/arb_improvements/test_aux_head.py`:

1. **Forward pass shape.** With aux on, forward returns a 3-tuple;
   `aux_logits.shape == (batch, max_runners)`.

2. **Default off = unchanged forward.** With aux off, forward's
   first two outputs are bit-identical to pre-session.

3. **Gradient flows to trunk.** Compute aux loss; backward;
   assert trunk parameters have non-zero gradients.

4. **Aux loss reduces on synthetic targets.** 20 steps on
   consistent targets → final aux loss < initial aux loss.

5. **All three architectures support aux head.** Parameterised
   test.

6. **Target computation correctness.** Build synthetic oracle
   samples + tick stream; compute aux targets; assert each
   target vector matches hand-computed expectation for the
   horizon.

7. **Default-off check: no memory cost.** Instantiate two policies,
   one with aux on and one off; assert parameter count differs by
   exactly the aux head's size.

## Session exit criteria

- All 7 tests pass.
- Existing tests still pass.
- If aux head is kept: `ui_additions.md` Session 9 tasks confirmed.
- `progress.md` Session 9 entry written. If deferred: explicit
  note with rationale and Phase-3 metrics that justified the
  deferral.
- `lessons_learnt.md` updated.
- If shipped: commit `feat(training): optional auxiliary arb-availability head`.
- If deferred: no commit; simply document in `progress.md`.
- `git push all` (if shipped).

## Do not

- Do not ship this session just because it's on the plan. The
  decision at end-of-Session-8 is explicit: ship it if BC alone
  isn't enough, defer it if Phase-3 already fixed the collapse.
- Do not change forward-pass return types when aux is off.
- Do not add separate oracle scanning — reuse Session 6 loader.
- Do not add GPU tests.
