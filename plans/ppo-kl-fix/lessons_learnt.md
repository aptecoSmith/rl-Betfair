---
plan: ppo-kl-fix
status: complete
landed: 2026-04-24
---

# Lessons learnt — ppo-kl-fix

## What landed

Option A from `purpose.md`: store hidden state per transition,
pass it back through the update's forward.

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
