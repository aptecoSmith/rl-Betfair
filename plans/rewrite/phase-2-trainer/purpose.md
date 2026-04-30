---
plan: rewrite/phase-2-trainer
status: design-locked
opened: 2026-04-27
depends_on: phase-1-policy-and-env-wiring (GREEN, 2026-04-27)
---

# Phase 2 — new PPO trainer with per-runner GAE

## Purpose

Build the v2 PPO trainer that pairs with Phase 1's
`DiscreteLSTMPolicy` + `DiscreteActionShim`. **Train a single agent
for one episode (one day), produce loss curves that look sane.** No
GA, no cohort, no scoreboard wiring — Phase 3's job.

The rewrite's whole architectural premise is that v1's credit-
assignment problem traces back to the 70-dim continuous action space
and the scalar value head (`plans/greenfield_review.md`). Phase 1
fixed the action space; Phase 2 fixes the value head's downstream
consequences:

- **Per-runner value head** → **per-runner GAE** → per-runner
  advantage. The discrete action picks ONE runner; the gradient on
  that runner's logit is driven by **that runner's** advantage, not a
  scalar mixed across all runners.
- The realised P&L per pair is already split by `pair_id` /
  `selection_id` in `BetManager`. The trainer attributes per-step
  reward deltas to the runner that caused them.

Phase 1's smoke proved the policy + env runs end-to-end with a
random init. Phase 2's smoke is: real PPO update on real data
produces a value-loss curve that descends, a policy-loss curve that
stays stable, and `approx_kl` that doesn't explode.

## Why this phase exists separately from Phase 3

The temptation to bundle "build the trainer" with "run a cohort" is
strong. We're splitting them on purpose:

1. **The trainer is the rewrite's load-bearing piece.** v1's trainer
   is what `plans/greenfield_review.md` indicted. If we can't get a
   single agent training cleanly on one day, no amount of cohort
   scaffolding fixes that.
2. **Phase 3 owns the integration surface.** Frontend events,
   websocket schema, GA worker pool, registry wiring, scoreboard
   adapter. Mixing those into Phase 2 is exactly the "while we're at
   it" refactor the rewrite README §4 forbids.
3. **The Phase 2 stop condition is a sanity bar, not a performance
   bar.** "Loss curves look sane" — not "agent beats v1" or "agent
   profitable". Phase 3 measures performance.

## What's locked

### PPO algorithm shape

Plain discrete PPO with two changes from textbook:

1. **Per-runner GAE** instead of a global scalar GAE. Detail in §"Per-
   runner credit assignment" below.
2. **Two action distributions** — the categorical over discrete
   actions and the Beta over stake. Both are PPO-ratio'd; the joint
   log-prob is `log_prob(action_idx) + log_prob(stake_unit)`. The
   stake log-prob ONLY contributes to the surrogate loss when the
   chosen action is `OPEN_BACK_*` or `OPEN_LAY_*` — for `NOOP` /
   `CLOSE` the stake is unused, so we mask its log-prob out (set to
   zero, gradient blocked).

Otherwise: clip-PPO surrogate loss, value loss = MSE(per-runner
return, per-runner value), entropy bonus = `entropy_coeff *
H(masked_categorical)` with **fixed `entropy_coeff`**.

**No entropy controller, no advantage normalisation, no LR warmup,
no reward centering** (rewrite hard constraints §5, §6). If
instability shows up, file as a finding and stop — don't reach for
v1's stabilisers.

### Per-runner credit assignment

`BetManager` exposes per-pair / per-bet P&L. The rollout collector
attributes each step's reward delta to runners as follows:

```
For each step t:
    per_runner_reward[t, i] = sum of P&L deltas attributable to
                              runner i at this step
                              (settled bets, MTM deltas if enabled,
                              shaped contributions tied to runner i)
```

Reward components NOT attributable to a single runner (e.g. terminal
bonuses, episode-level shaping if any) get distributed equally
across `max_runners` slots. That keeps the per-runner GAE consistent
with the scalar PPO total: `sum_i A_t^{(i)} ≈ A_t^{scalar}` to within
floating-point noise.

GAE per-runner:

```
δ_t^{(i)}  = r_t^{(i)} + γ * V_{t+1}^{(i)} - V_t^{(i)}
A_t^{(i)}  = δ_t^{(i)} + γλ * A_{t+1}^{(i)}
return_t^{(i)} = A_t^{(i)} + V_t^{(i)}
```

Standard GAE applied per-runner. The policy gradient uses the
**chosen-runner advantage**: when the discrete action at step t was
`OPEN_BACK_i`, `OPEN_LAY_i`, or `CLOSE_i`, the surrogate loss on
that step uses `A_t^{(i)}`. For `NOOP` we use `mean_i A_t^{(i)}`
(the policy chose nothing, so the gradient signal is the average
across runners — this preserves "doing nothing was a good/bad
choice given everything that was happening").

The value loss sums over runners:
`L_value = mean_t sum_i (V_t^{(i)} - return_t^{(i)})^2`.

### Hidden-state contract

Phase 1's `DiscreteLSTMPolicy.pack_hidden_states` /
`slice_hidden_states` is the load-bearing API. The PPO update path
must:

1. Capture `hidden_state_in` BEFORE each forward pass during
   rollout (NOT the state returned by it — that's the input to the
   NEXT step). First step's state is `init_hidden(batch=1)`.
2. Pack the per-transition states into a batched tuple at update
   time via `policy.pack_hidden_states(list_of_tuples)`.
3. Slice by mini-batch indices via `policy.slice_hidden_states(packed,
   indices)`.
4. Pass the sliced state into `policy(mb_obs, hidden_state=..., mask=...)`
   in BOTH the surrogate-loss forward and the KL-diagnostics forward.

Same protocol v1 ended up at after the `ppo-kl-fix` plan landed
(`CLAUDE.md` §"Recurrent PPO: hidden-state protocol on update"). We
don't need to re-derive it; we just need to wire it through correctly
on the first try.

### Mask carries with the transition

The rollout-time mask MUST be stored alongside `obs` /
`hidden_state_in` / `action_idx` and passed back into the policy at
update time. Skipping this would let the env's mask drift between
rollout and update (e.g. a runner becomes inactive between rollout
and the gradient step), causing `new_log_prob = -inf` and exploding
the importance-sampling ratio.

Phase 1's findings.md §2 flagged this; Phase 2 honours it.

### Hyperparameters (locked, no search)

| Knob | Value | Rationale |
|---|---|---|
| `learning_rate` | `3e-4` | Standard PPO. v1's default. No reason to deviate. |
| `gamma` | `0.99` | Standard. Race-level discounting; per-runner GAE doesn't change this. |
| `gae_lambda` | `0.95` | Standard. |
| `clip_range` | `0.2` | Standard. |
| `entropy_coeff` | `0.01` | Fixed. No controller. |
| `value_coeff` | `0.5` | Standard. |
| `ppo_epochs` | `4` | Standard. |
| `mini_batch_size` | `64` | Same as v1's late-2026 setup. |
| `max_grad_norm` | `0.5` | Standard. Prevents gradient blow-up if a per-runner advantage spike happens. |

These are starting points, not GA genes. Phase 2's stop condition
is "loss curves sane on the chosen settings" — if they aren't,
that's a finding, not a tuning exercise.

## Success bar (Phase 2)

Phase 2 ships iff ALL of:

1. **Trains end-to-end on one day** with no exception.
   `python -m training_v2.discrete_ppo.train --day 2026-04-23`
   completes in under 10 minutes on CPU.
2. **Value-loss curve descends** monotone (with PPO-update-level
   noise) over the 4 PPO epochs of the training run. A flat or
   exploding curve is a fail.
3. **`approx_kl` stays under 0.5** averaged across mini-batches per
   PPO update for the full run. Spikes above 1.0 individually are
   acceptable; a sustained median above 0.5 is a fail.
4. **Per-runner advantage shape is correct.** A unit test asserts
   `advantages.shape == (n_steps, max_runners)` and that the
   chosen-runner advantage matches a hand-computed reference on a
   tiny synthetic rollout.
5. **No env changes.** Same as Phase 1.

If any of 1–4 fails: stop, write findings, decide whether the trainer
design needs revisiting (likely Session 01's GAE math) or the
hyperparameters need a small tweak (only `learning_rate` and
`entropy_coeff` are tweakable in this phase; everything else is
algorithmically locked).

## Deliverables

A new directory `training_v2/discrete_ppo/` with:

- `training_v2/discrete_ppo/__init__.py` — exports.
- `training_v2/discrete_ppo/transition.py` — `Transition` dataclass
  carrying everything the update needs (`obs`, `hidden_state_in`,
  `mask`, `action_idx`, `stake_unit`, `log_prob_action`,
  `log_prob_stake`, `value_per_runner`, `per_runner_reward`, `done`).
- `training_v2/discrete_ppo/rollout.py` — `RolloutCollector`. Drives
  Phase 1's shim + policy through one episode, returns a list of
  `Transition`s. Handles per-runner reward attribution via the
  `BetManager` pair / bet records.
- `training_v2/discrete_ppo/gae.py` — `compute_per_runner_gae(...)`
  pure function. Takes `(rewards, values, dones, gamma, gae_lambda)`
  shape `(n_steps, max_runners)`, returns `(advantages, returns)`
  same shape.
- `training_v2/discrete_ppo/trainer.py` — `DiscretePPOTrainer`.
  Constructs optimiser, runs N episodes, performs PPO update per
  episode, logs losses + KL.
- `training_v2/discrete_ppo/train.py` — CLI entry point.

Tests under `tests/`:

- `tests/test_discrete_ppo_transition.py` — `Transition` round-trip.
- `tests/test_discrete_ppo_rollout.py` — rollout collector produces
  one transition per env step, captures `hidden_state_in` correctly,
  per-runner reward attribution sums to total.
- `tests/test_discrete_ppo_gae.py` — pure GAE math, hand-computed
  reference on a 3-step / 2-runner toy.
- `tests/test_discrete_ppo_trainer.py` — fresh policy + tiny synthetic
  day → 1 PPO update → `approx_kl < 1.0`, value-loss decreased,
  gradients on every parameter.

A short writeup at `plans/rewrite/phase-2-trainer/findings.md`:
loss curves, KL trajectory, per-runner advantage stats, anything
surprising.

## Hard constraints

1. **Don't touch the env.** (Rewrite hard constraint §1.) If the
   trainer needs per-bet P&L exposed differently, file as a Phase −1
   follow-on; do NOT bundle env changes into Phase 2.
2. **Don't touch the data pipeline.** (Rewrite hard constraint §2.)
3. **No re-import of v1 trainer / policy.** Parallel tree —
   everything new lives under `training_v2/discrete_ppo/`. Read
   `agents/ppo_trainer.py` for reference if you need to remember
   how clip-PPO works; do NOT import it. (Rewrite hard constraint
   §3.)
4. **No new shaped rewards.** (Rewrite hard constraint §5.) The
   trainer consumes `info["raw_pnl_reward"] + info["shaped_bonus"]`
   the env already produces; per-runner attribution is just a
   re-organisation, not a new term.
5. **No entropy controller, no advantage normalisation, no LR
   warmup, no reward centering.** (Rewrite hard constraint §6.) If
   training is unstable WITHOUT these, the architecture is doing
   something wrong — file as a finding, don't reach for the v1
   stabilisers.
6. **No GA, no cohort, no breeding.** (Phase 3.) Phase 2 trains ONE
   agent on ONE day, end-to-end.
7. **No frontend wiring.** (Phase 3.) Phase 2 logs to plain text /
   JSONL; no websocket events, no scoreboard rows.
8. **No hyperparameter search.** Locked values above. If they don't
   work, that's a finding; if they do, that's the bar passed.

## Out of scope

- Multi-day training (Phase 3).
- Cohort runs (Phase 3).
- GA mutation / breeding (Phase 3).
- Frontend / websocket events (Phase 3).
- Comparing performance to v1 (Phase 3).
- Removing v1 code (after Phase 3 success per rewrite README).
- BC pretrain (the rewrite removes BC — Phase 0's standalone scorer
  replaces the discriminative half).

## Phase 1 hand-offs that constrain Phase 2

From `plans/rewrite/phase-1-policy-and-env-wiring/findings.md`
(Session 02, GREEN, 2026-04-27):

1. **`obs_dim` source-of-truth is `shim.obs_dim`.** Phase 2's
   trainer must construct the policy with `obs_dim=shim.obs_dim`,
   not `env.observation_space.shape[0]` directly (that misses the
   28-dim scorer extension at `max_runners=14`).
2. **The masked categorical interacts cleanly with PPO's IS ratio**
   — provided the rollout-time mask is stored on the transition and
   re-applied at update time. (Phase 1 findings §2.)
3. **NOOP-always-legal is load-bearing.** PyTorch's
   `Categorical(logits=…)` with all logits at `-inf` produces NaN
   probabilities. `compute_mask` guarantees `mask[0] = True`
   unconditionally; the trainer assertions can lean on this.
4. **LSTM hidden state batches along dim=1.** `pack_hidden_states`
   and `slice_hidden_states` already implement this in Phase 1; the
   trainer just calls them. The transformer (Phase 1 follow-on)
   uses dim=0; Phase 2 only cares about LSTM.
5. **Stake head is `(batch,)`, not `(batch, max_runners)`.** A
   single scalar stake per tick, not per-runner. The Beta log-prob
   is `(batch,)`. (Phase 1 findings §4.)

## Sessions

1. **`01_rollout_collector_and_gae.md`** — `Transition`,
   `RolloutCollector`, `compute_per_runner_gae`. **No PPO update
   yet.** End-of-session check: roll out one episode, GAE produces
   per-runner advantage with the right shape, hand-checked against
   a synthetic toy.
2. **`02_ppo_update_and_trainer.md`** — `DiscretePPOTrainer`.
   Surrogate loss with masked categorical + Beta stake, per-runner
   value loss, entropy bonus, KL diagnostics, optimiser step. End-
   of-session check: one PPO update on one synthetic-day rollout
   produces gradients, `approx_kl < 1.0`.
3. **`03_first_real_train_run.md`** — wire the train CLI, run on
   `2026-04-23`, write findings.md. End-of-session check: success
   bar PASS.

Each session is independently re-runnable. Session 02 imports
Session 01's collector verbatim; Session 03 imports Session 02's
trainer verbatim. Same "if a later session finds a problem in an
earlier session, revisit the earlier session" rule as Phase 1.
