# Session prompt — Phase 2, Session 02: PPO update + DiscretePPOTrainer

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the task, the design decisions
already locked, and the constraints. Do not require any context from
the session that scaffolded this prompt.

---

## The task

Implement the PPO update for the v2 discrete policy. Wrap rollout +
GAE + update into a `DiscretePPOTrainer` class. **Run one PPO update
on a synthetic-day rollout** as the in-session smoke. **No real-day
training** — that's Session 03.

End-of-session bar: fresh policy + Session 01's collector → one
synthetic-day episode → one PPO update → `approx_kl < 1.0`,
value-loss decreased on the first epoch, gradients on every
parameter.

## What you need to read first

1. `plans/rewrite/phase-2-trainer/purpose.md` — locked algorithm
   shape, hyperparameter table, "Per-runner credit assignment".
2. `plans/rewrite/phase-2-trainer/session_prompts/01_rollout_collector_and_gae.md`
   and the resulting code under `training_v2/discrete_ppo/` —
   import these verbatim.
3. `plans/rewrite/phase-1-policy-and-env-wiring/findings.md` — Phase
   1 hand-offs §2, §3 (mask carrying, NOOP-always-legal) and §4
   (LSTM hidden-state batches along dim=1).
4. `CLAUDE.md` §"Recurrent PPO: hidden-state protocol on update" —
   the update-time hidden-state protocol. Pack via
   `policy.pack_hidden_states(list)`, slice via
   `policy.slice_hidden_states(packed, indices)`. **Do not roll your
   own; the policy class already has these.**
5. `CLAUDE.md` §"Per-mini-batch KL check" — measurement granularity.
   v1 ended up with per-mini-batch checks at threshold 0.15. v2's
   default per the purpose.md table is `kl_early_stop_threshold =
   0.15`; the check fires per mini-batch, not per epoch.
6. `agents/ppo_trainer.py::_ppo_update` — v1 reference for the
   surrogate-loss shape. **Read, don't import.** v2 is much simpler:
   no entropy controller, no advantage normalisation, no LR warmup,
   no reward centering. The clip-ratio + value-MSE + entropy math is
   the same.

## What to do

### 1. `training_v2/discrete_ppo/trainer.py::DiscretePPOTrainer` (~120 min)

```python
class DiscretePPOTrainer:
    def __init__(
        self,
        policy: BaseDiscretePolicy,
        shim: DiscreteActionShim,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        max_grad_norm: float = 0.5,
        kl_early_stop_threshold: float = 0.15,
        device: str = "cpu",
    ): ...

    def train_episode(self) -> EpisodeStats: ...

    def _ppo_update(
        self, transitions: list[Transition],
        advantages: np.ndarray, returns: np.ndarray,
    ) -> UpdateLog: ...
```

`EpisodeStats` carries the per-episode summary the trainer logs:
`total_reward`, `n_steps`, `n_updates_run`, `policy_loss_mean`,
`value_loss_mean`, `entropy_mean`, `approx_kl_mean`,
`approx_kl_max`, wall time. Plain dataclass, frozen.

### 2. The PPO update path (~60 min, factored inside the class)

In order:

1. **Bootstrap value.** Run `policy.forward(final_obs, final_hidden,
   final_mask)` on the post-terminal observation; use
   `value_per_runner.detach().cpu().numpy()` as
   `bootstrap_value`. Zero if the episode terminated naturally
   (`done=True`).
2. **GAE.** Stack transitions into `rewards (T, R)`, `values (T,
   R)`, `dones (T,)` arrays; call
   `compute_per_runner_gae(...)`. **No advantage normalisation**
   (rewrite hard constraint §6 — if instability shows up, file a
   finding, don't reach for normalisation).
3. **Build update tensors.** Stack obs, masks, action_idx,
   stake_unit, log_prob_action, log_prob_stake. Pack hidden states
   via `policy.pack_hidden_states(...)`.
4. **Mini-batch loop.** For each PPO epoch, shuffle indices, slice
   into mini-batches of `mini_batch_size`. For each mini-batch:
   - Slice the packed hidden state via
     `policy.slice_hidden_states(packed, mb_indices_tensor)`.
   - Forward pass: `out = policy(mb_obs, mb_hidden, mask=mb_mask)`.
   - **Categorical log-prob** at the chosen action_idx:
     `new_lp_action = out.action_dist.log_prob(mb_action_idx)`.
   - **Beta log-prob** at the stake unit:
     `Beta(out.stake_alpha, out.stake_beta).log_prob(mb_stake_unit)`.
     **Multiply by an `uses_stake` mask** (1 for OPEN_*, 0
     otherwise) so NOOP / CLOSE don't push gradient through the
     stake head from a placeholder log-prob.
   - **Joint log-prob** = `new_lp_action + uses_stake * new_lp_stake`.
   - **Ratio** = `exp(joint_new_lp - joint_old_lp)`.
   - **Chosen-runner advantage**: per the purpose.md, when the
     action is `{OPEN_BACK_i, OPEN_LAY_i, CLOSE_i}`, use
     `advantages[t, i]`; when it's NOOP, use `advantages[t, :].mean()`.
     Build a `(batch,)` chosen-advantage tensor up-front.
   - **Surrogate** = `min(ratio * adv, clip(ratio, 1-c, 1+c) * adv).mean()`.
     Sign convention: this is the policy gradient TARGET, so the
     loss is `-surrogate`.
   - **Value loss** = `value_coeff * mean over (t, i) of
     (out.value_per_runner - mb_returns)^2`. **Per-runner** — sum
     over runners is implicit in the mean.
   - **Entropy** = `out.action_dist.entropy().mean()` (categorical
     entropy only — Beta entropy not added; the stake head's role
     is sizing, not exploration). Loss contribution
     `-entropy_coeff * entropy`.
   - **Total loss** = `-surrogate + value_loss - entropy_coeff *
     entropy`.
   - `optimiser.zero_grad()`, `total_loss.backward()`,
     `clip_grad_norm_(policy.parameters(), max_grad_norm)`,
     `optimiser.step()`.
   - **Per-mini-batch KL check.** After `optimiser.step()`, with
     `torch.no_grad()`, compute
     `approx_kl = (mb_old_lp - new_lp).mean().item()`. If
     `approx_kl > kl_early_stop_threshold`, **break out of BOTH the
     mini-batch loop AND the epoch loop** and log the count of
     skipped mini-batches.

### 3. `training_v2/discrete_ppo/__init__.py` exports (~5 min)

Export `DiscretePPOTrainer`, `Transition`, `RolloutCollector`,
`compute_per_runner_gae`, `EpisodeStats`, `UpdateLog`.

### 4. Tests (~90 min)

`tests/test_discrete_ppo_trainer.py` — slow-marked, skip-if-scorer-
absent:

- `test_one_ppo_update_runs_without_exception` — synthetic day, one
  episode, one update. No assertions on outcomes; just "doesn't
  crash".
- `test_one_update_produces_gradients_on_every_param` — after the
  update, every `requires_grad` param has a non-None `.grad`.
- `test_approx_kl_small_on_first_epoch` — fresh policy, one
  rollout, one update; per-mini-batch `approx_kl < 1.0` on the
  first epoch. Catches the v1-era "state mismatch" bug as a
  regression guard.
- `test_value_loss_decreases_across_epochs` — record the
  per-mini-batch value loss; the **median** across epoch 4 is
  lower than the median across epoch 1. (Per-mini-batch noise is
  high; medians smooth it.) If the test is too flaky, fall back to
  "value loss at the end of training is below the loss at the
  start".
- `test_kl_early_stop_skips_remaining_minibatches` — set
  `kl_early_stop_threshold=1e-12` (guaranteed trip on the first
  mini-batch), assert `n_updates < ppo_epochs *
  mini_batches_per_epoch`. Same load-bearing test v1 has.
- `test_chosen_runner_advantage_used_for_open_actions` — unit
  test on the helper that maps `(action_idx, advantages)` to
  `chosen_advantage`. OPEN_BACK on slot 3 → `advantages[:, 3]`;
  NOOP → `advantages[:, :].mean(axis=1)`.
- `test_uses_stake_mask_blocks_stake_grad_for_noop_actions` —
  with all transitions set to NOOP, `stake_alpha_head.weight.grad`
  is zero (or None). Catches a bug where the placeholder
  `log_prob_stake = 0` accidentally contributes gradient.

## Stop conditions

- All 4 success-bar conditions for THIS session pass → message
  operator "Phase 2 Session 02 GREEN, ready for Session 03", **stop**.
- `approx_kl` on the first PPO update is > 1.0 with default
  hyperparameters → **stop**. Likely cause: hidden-state mismatch
  between rollout and update (the most-warned-about bug in
  CLAUDE.md). Fix the wiring; don't reach for the v1 stabilisers.
- Value loss diverges (NaN, Inf, or order-of-magnitude blow-up) →
  **stop**. Likely cause: per-runner GAE math wrong, or the
  bootstrap value plumbing wrong. Fix the source; don't add
  reward centering.
- Gradients on the stake head appear when the only actions are
  NOOP → **stop and fix the `uses_stake` mask**.

## Hard constraints

- **No entropy controller, no advantage normalisation, no LR
  warmup, no reward centering.** (Rewrite hard constraint §6.) If
  the bar fails, the architecture is wrong; document and stop.
- **No env edits.** Same as Phase 1.
- **No re-import of v1 classes.** Parallel tree.
- **Hidden state on the update side comes from
  `policy.pack/slice_hidden_states`.** Don't reach into the LSTM
  internals.
- **Mask is re-applied at update time.** Same mask the rollout
  saw. The policy's masked categorical handles `-inf` cleanly; if
  the mask drifts, log-probs explode.
- **No multi-agent, no GA, no real-day training.** Session 03 owns
  the real-day run; Phase 3 owns the GA.

## Out of scope

- Real-day training (Session 03).
- GA / cohort (Phase 3).
- Frontend events (Phase 3).
- Performance / wall-time tuning beyond "doesn't crash" (Phase 3).
- Reward shaping (forbidden — rewrite hard constraint §5).
- Auxiliary heads (fill_prob, mature_prob, risk) — those went away
  in the rewrite (replaced by the standalone scorer in Phase 0).

## Useful pointers

- `training_v2/discrete_ppo/transition.py` — Session 01.
- `training_v2/discrete_ppo/rollout.py::RolloutCollector` — Session 01.
- `training_v2/discrete_ppo/gae.py::compute_per_runner_gae` — Session 01.
- `agents_v2/discrete_policy.py::DiscreteLSTMPolicy.pack_hidden_states`
  / `slice_hidden_states` — Phase 1.
- `agents/ppo_trainer.py::_ppo_update` — v1 reference, READ ONLY.
- `tests/test_ppo_trainer.py::TestRecurrentStateThroughPpoUpdate` —
  the load-bearing v1 regression-guard tests for KL stability.
  Adapt the shape of `test_approx_kl_small_on_first_epoch_lstm` to
  v2; it's the strictest signature.

## Estimate

3.5–5 hours.

- 120 min: `DiscretePPOTrainer` skeleton + ctor + `train_episode`
  outer loop.
- 60 min: `_ppo_update` inner loop (KL check, mini-batch shuffle,
  joint log-prob, surrogate, value loss, entropy bonus,
  optimiser step).
- 90 min: tests.
- 30 min: session writeup.

If past 6 hours, stop and check scope. The most likely overrun is
the per-mini-batch KL check + the early-stop bookkeeping (counting
skipped mini-batches across epochs is fiddly). v1's
`agents/ppo_trainer.py::_ppo_update` has a working pattern; copy
the SHAPE without copying the entropy-controller / advantage-norm
/ reward-centering tendrils.
