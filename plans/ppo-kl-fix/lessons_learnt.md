---
plan: ppo-kl-fix
status: complete
landed: 2026-04-24 (Session 01) + 2026-04-25 (Session 02)
---

# Lessons learnt — ppo-kl-fix

## Session 02 (2026-04-25) — per-mini-batch KL check

### What landed

Moved the KL early-stop check from "once per PPO epoch on the full
rollout's log-probs" to "once per mini-batch on this mini-batch's
log-probs". The check now runs immediately after
`optimiser.step()` and breaks BOTH the inner mini-batch loop and
the outer epoch loop as soon as the policy has drifted more than
`kl_early_stop_threshold` from the rollout distribution.

Specifically:
- New per-mini-batch check: `(mb_old_log_probs −
  new_log_probs).mean()` computed under `torch.no_grad()`
  immediately after the gradient step.
- Removed the end-of-epoch full-rollout forward + KL compute
  (it was superseded by the per-mini-batch check).
- Added `n_updates` to `loss_info` so episodes.jsonl surfaces how
  many gradient steps actually ran each update.
- Log line on trip says "skipping X remaining mini-batches across
  Y epoch(s)" — honest about compute saved.

### Why this was needed (Session 01 wasn't enough)

The partial run immediately after Session 01 landed
(`registry/archive_post-kl-fix-partial-20260424T220000Z/`) showed
`approx_kl = 3.94 – 18.87` at the per-epoch check point. Pre-
Session-01 was ~12,740 — a 1,000× reduction — but still 100× the
0.03 threshold. Every update still early-stopped on epoch 0.
Training was still starved.

The cause was structural: a 10k-transition rollout with
`mini_batch_size=64` takes ~156 gradient steps per epoch. Each
step took a small, safe update, but by the end of the epoch the
accumulated drift was > 3.0. The check fired — correctly — but
only after the damage (156 accumulated steps) was done. Moving
the check inside the mini-batch loop catches the drift when it
first breaches, not 156 steps later.

### Regression guard

`test_kl_early_stop_is_per_mini_batch_not_per_epoch` in
`tests/test_ppo_trainer.py::TestRecurrentStateThroughPpoUpdate`.
Construction: set threshold to 1e-12 (guaranteed to trip), run
`_ppo_update`, assert `n_updates < mini_batches_per_epoch`. A
per-epoch check would run an entire epoch (6 mini-batches on
the test construction) before the first KL evaluation; the per-
mini-batch check stops after the first post-gradient mini-batch
(1 update). Any revert to per-epoch granularity fails this test
immediately.

### Expected impact

Post-Session-02 runs should show `n_updates` matching or near
the full `ppo_epochs × mini_batches_per_epoch` budget (~600+ for
production rollouts) when training is healthy — not the
pre-Session-01 value of ~156 (one-epoch-worth). When KL drift
legitimately exceeds threshold the stop is surgical: trim the
remaining mini-batches of the current epoch + skip later epochs
instead of wasting compute on a full epoch sweep before the check.

---

## Meta-lessons

Both the Session 01 bug and the Session 02 residual share a
failure mode worth flagging for future work.

### 1. "Defensive" diagnostics that always fire are a bug, not a defence

`kl_early_stop_threshold=0.03` was added in
`plans/naked-clip-and-stability` Session 02 (2026-04-18) with a
comment framing it as a "runaway guard". From that commit to the
ppo-kl-fix investigation — **nearly a week** — the early-stop
tripped on epoch 0 of every update, every agent, every day.
Nobody investigated because the check was labelled defensive.

Durable rule: **any check that fires on every input is either
mis-targeted or indicates a pathology.** If "defensive" is the
mental label, add an alarm that LOUD-fires the first time the
trip-rate exceeds some threshold (say 10%). A trip-rate of 100%
is signal, not noise.

### 2. Dimensional / granularity bugs hide inside "correct" formulas

Session 01 was "stateful vs stateless forward pass" — a shape/
protocol mismatch that made the KL measurement meaningless.
Session 02 was "per-epoch on 156-step rollout vs per-step" — a
granularity mismatch that made the KL measurement technically
correct but operationally useless. Both formulas were defensible
at the unit-test level; both failed at the integration level.

The 2026-04-18 `naked-clip-and-stability` units-mismatch bug
(episode-sum vs per-step reward centering) was the same shape.
And the 2026-04-24 action-clipping bug (stored clipped action
against un-clipped log-prob) was the same shape.

Durable rule: **integration tests on the actual gradient path
are load-bearing. Any test that mocks the forward pass, the
gradient step, or the rollout can pass while the real path is
broken.** `tests/test_ppo_trainer.py::
TestRecurrentStateThroughPpoUpdate` intentionally mocks nothing.

### 3. The investigation paid for itself before implementation

Spending a session on
`plans/ppo-stability-and-force-close-investigation/` (read-only
evidence + hypothesis ranking) before touching `_ppo_update`
meant:

- H1 (advantage norm off), H2 (reward-centering units), H3 (KL
  formula wrong), H4 (force-close magnitude driving KL), H5
  (alpha saturating) were all falsified by the data BEFORE any
  code change. If we'd jumped straight to "fix H4" (the most
  plausible from the session prompt) we'd have refactored force-
  close sizing without addressing the KL explosion at all.
- The H6 hypothesis (stateful/stateless mismatch) emerged from
  reading the code with the evidence in hand — specifically,
  line 1755's comment "no LSTM state for mini-batch -- treat
  each transition independently during optimisation" was the
  tell. Without the rank-correlation table, that comment would
  have read as intentional.

Durable rule: **for any reproducible-across-all-configs bug,
always start with an investigation plan (read-only) before a
fix plan.** The investigation forces you to enumerate the whole
hypothesis space; the fix plan only commits to one.

### 4. Partial-fix scoreboard data is still data

The Session 01 partial run archived in
`registry/archive_post-kl-fix-partial-20260424T220000Z/` was
killed mid-flight, but its worker.log proved Session 01 alone
wasn't enough and justified Session 02. We almost threw the data
away because it was "only 12 partial agents" — the KL readings
were the evidence that sized the follow-on work.

Durable rule: **a run killed partway still produces data. Don't
delete it reflexively; spend 2 minutes reading the log before
archiving.** The `registry/archive_*` convention makes this free
to preserve.

### 5. When the evidence contradicts the framing, believe the evidence

The session prompt listed H1–H5 as ranked hypotheses. Every one
was refuted by the data. The prompt's meta-framing was "KL is
huge, here are the usual suspects" — and the usual suspects
were all wrong. The real cause was a sixth hypothesis that
wasn't in the prompt because nobody writing the prompt had seen
the stateful-vs-stateless mismatch before.

Durable rule: **when all listed hypotheses fail, don't just
pick the closest one. Re-read the code with the falsifications
in hand and let the data steer you toward the hypothesis nobody
pre-registered.**

---

## Session 01 (2026-04-24) — stateful rollout ↔ stateful update

### What landed

Changes:

- `agents/ppo_trainer.py::Transition` gains
  `hidden_state_in: tuple[np.ndarray, np.ndarray] | None = None`.
- `_collect_rollout` captures the state BEFORE each forward
  (not after — the post-forward state would feed the NEXT
  transition, not THIS one) as CPU numpy and stores it on the
  transition. Also switches action storage to the UN-clipped
  sample so the stored `log_prob` and stored `action` reference
  the same distribution (see pre-existing bug below).
- `_ppo_update` packs per-transition states via
  `policy.pack_hidden_states(...)` once, slices by `mb_idx` via
  `policy.slice_hidden_states(...)` inside the mini-batch loop,
  and passes the result to `self.policy(mb_obs, mb_hidden)`.
  Same treatment for the KL-diagnostics forward.
- `agents/policy_network.py`:
  - `BasePolicy` grows `pack_hidden_states` / `slice_hidden_states`
    defaults (concat / index along dim 0 — correct for the
    transformer's `(batch, ctx, d_model)` / `(batch,)` layout).
  - `PPOLSTMPolicy` and `PPOTimeLSTMPolicy` override both methods
    for the `(num_layers, batch, hidden)` LSTM-family layout with
    batch on dim 1.
- New regression guard tests in
  `tests/test_ppo_trainer.py::TestRecurrentStateThroughPpoUpdate`.
- `CLAUDE.md` — new subsection "Recurrent PPO: hidden-state
  protocol on update (2026-04-24)" under "PPO update stability".

## Evidence

Smoke test across all three architectures, fresh policy, one
`_ppo_update` on a 6-transition rollout:

| Arch | Pre-fix median KL (observed in probe) | Post-fix KL (smoke) |
|---|---|---|
| `ppo_lstm_v1` | 18,471 | 0.0043 |
| `ppo_time_lstm_v1` | 24,871 | −0.0004 |
| `ppo_transformer_v1` | 6,996 | 0.0169 |

All land well under the 0.03 early-stop threshold. PPO will now
run the full 4 epochs per update on a fresh policy instead of
early-stopping on epoch 0.

Full `tests/test_ppo_trainer.py` suite: 62 passed. Adjacent
`test_mark_to_market.py` + `test_integration_policy_network.py`
+ `test_training_worker.py` suite: 33 passed. Reward-centering
units contract test still green (`test_real_ppo_update_feeds_per_
step_mean_to_baseline`).

## Pre-existing bug surfaced and fixed in-session

The regression test with frozen optimiser steps initially
showed `approx_kl = -15` — not the thousands of the pre-fix
observation, but not zero either. Root cause: the rollout
clipped actions to `[-1, 1]` IN PLACE via `np.clip(action_np,
-1.0, 1.0, out=action_np)` AFTER `log_prob` was computed on the
un-clipped sample. The `Transition` stored the clipped action
but the un-clipped log-prob. At update time, `dist.log_prob(
clipped_action)` differed from the stored `log_prob` by ~13
nats on average (33% of action dimensions clipped at the
baseline policy; clipping a sample at +1.5 to +1.0 makes its
log-prob LARGER).

Fix: store the un-clipped action on `Transition`; clip only the
copy passed to `env.step`. The regression test
`test_ppo_update_approx_kl_matches_old_logp_before_any_gradient_step`
catches this signature by construction — same weights + same
obs + same state ⇒ same log-prob, so any non-zero KL with
frozen weights points to either state mismatch or action
mismatch.

This bug was latent for a long time but was not separately
visible: its ~13-nat contribution was dominated by the
thousands-of-nats state-mismatch contribution. The fix for one
surfaced the other.

## Why the original bug wasn't caught earlier

The unit tests for PPO's forward pass mock the policy to
return fixed tensors, so they don't exercise the stateful-vs-
stateless axis at all. The integration tests that DO run the
real trainer never looked at `approx_kl` magnitude — they only
asserted the update runs without error. The `kl_early_stop`
mechanism was added in `plans/naked-clip-and-stability`
Session 02 with a threshold of 0.03, and from day one it was
firing on epoch 0 of every update. The field was labelled a
"defensive trip" in the comments, so the explosive numbers were
dismissed as one-off rollout noise rather than investigated as
a signal.

Don't trust "defensive trips" that fire every time. That's a
bug, not a defence.

## Failure mode to watch for

If a future recurrent architecture lands (GRU variant, more
transformer variants with different state layouts), its hidden
state protocol MUST be:

1. Stateful forward at rollout: `hidden_state = out.hidden_state`
   threaded across ticks.
2. `init_hidden(1)` returns zeros in a format where batch is
   identified on SOME tensor axis (dim 0 for the transformer,
   dim 1 for the LSTMs).
3. Override `pack_hidden_states` / `slice_hidden_states` if the
   batch axis isn't dim 0.
4. No other caller of the update bypasses `_collect_rollout` in
   a way that skips the `hidden_state_in` capture.

The regression guard `test_ppo_update_approx_kl_matches_old_logp_
before_any_gradient_step` will immediately flag any of these
regressing — but only if it actually runs. Add the same test
variant for any new recurrent architecture added to the
registry.

## Scoreboard comparability note

Runs from before 2026-04-24 had PPO doing one mini-batch sweep
per day per agent (every update early-stopped on epoch 0).
Post-fix runs do the intended 4 epochs. Training dynamics will
be VERY different:

- Policy will learn faster (more gradient steps per rollout).
- Reward-centering EMA will warm up on different trajectory
  shapes.
- `alpha` controller will see entropy moving on a different
  time-scale; the controller's effective gain is unchanged but
  the policy it's trying to control is now actually trainable.

Pre-fix and post-fix scoreboard rows are NOT comparable on
`total_reward`, `policy_loss`, `value_loss`, or `entropy`. They
remain comparable on `raw_pnl_reward` (env math untouched) and
on composite PnL metrics. Any GA analysis that spans the fix
should bucket by commit.

## Follow-ons

- `plans/force-close-sizing-review/` was drafted alongside this
  fix. Re-open that plan after at least one full GA cycle
  under the fix to get a trained-policy baseline; designing
  force-close sizing against a BC-frozen policy (which is what
  cohort W's gen-1 measured) is sizing to a confounded surface.
- The `arb-signal-cleanup-probe` plan's Validation should be
  re-done under the fix. Consider resetting the registry for
  gen-0.
