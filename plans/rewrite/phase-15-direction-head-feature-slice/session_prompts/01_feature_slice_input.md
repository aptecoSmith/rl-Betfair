---
plan: rewrite/phase-15-direction-head-feature-slice
session: S01
deliverable: Re-input direction_prob_head to per-runner feature slice
---

# S01 — Feature-slice input for `direction_prob_head`

## Goal

Replace `direction_prob_head`'s input from
`(slot_emb_i, lstm_last)` to the runner's raw `RUNNER_KEYS`
feature slice. Reproduce the supervised probe's input regime
inside `DiscreteLSTMPolicy.forward`.

## File-level changes

### `agents_v2/discrete_policy.py`

1. **`__init__`** — change `direction_prob_head[0]`'s input
   dim from `runner_embed_dim + hidden_size` to `RUNNER_DIM`.
   Import `RUNNER_DIM` from `env.betfair_env`.

   ```python
   from env.betfair_env import RUNNER_DIM  # 125 after phase-14 S02

   self.direction_prob_head = nn.Sequential(
       nn.Linear(RUNNER_DIM, self.actor_mlp_hidden),
       nn.ReLU(),
       nn.Linear(self.actor_mlp_hidden, 2),
   )
   ```

2. **`forward`** — slice the per-runner block out of obs and
   feed it to the head. Pattern from
   `agents/policy_network.py:691-706` (v1's existing per-
   runner extraction):

   ```python
   # Slice per-runner block from obs. market_dim and
   # max_runners are existing fields on the policy.
   runner_start = self.market_dim
   runner_end = runner_start + self.max_runners * RUNNER_DIM
   runners_flat = obs[:, runner_start:runner_end]
   runner_feats_raw = runners_flat.view(
       batch, self.max_runners, RUNNER_DIM,
   )

   # Direction head reads the slice directly. NO concat with
   # lstm_last (hard_constraints §2).
   direction_input_flat = runner_feats_raw.reshape(
       batch * self.max_runners, RUNNER_DIM,
   )
   direction_logits_flat = self.direction_prob_head(
       direction_input_flat,
   )
   direction_logits = direction_logits_flat.view(
       batch, self.max_runners, 2,
   )
   ```

3. **Keep** the existing `runner_embs_b` / `lstm_expanded`
   computation — `actor_head` still uses both. Just don't
   feed them to `direction_prob_head`.

4. **`load_state_dict`** — no explicit version field; PyTorch's
   strict load naturally refuses pre-S01 weights because
   `direction_prob_head[0].weight` shape changed.

### Tests to update / add

In the appropriate `tests/test_*direction*.py` file:

- **Update:** any test asserting
  `direction_prob_head[0].weight.shape ==
  (actor_mlp_hidden, runner_embed + hidden)` →
  assert it equals `(actor_mlp_hidden, RUNNER_DIM)`.

- **Add `test_direction_head_consumes_runner_feature_slice`:**
  build obs, perturb runner i's feature block, assert
  `out.direction_back_logits[:, i]` changes; perturb runner
  j's block, assert runner i's logit unchanged. Catches
  cross-runner mixing.

- **Add `test_direction_head_does_not_depend_on_lstm_last`:**
  with fixed obs, perturb the LSTM hidden state passed in;
  assert `out.direction_back_logits` unchanged. Strict
  bypass guard per hard_constraints §2.

- **Add
  `test_direction_actor_loss_routes_grad_through_direction_head`:**
  same shape as the existing fill-prob / mature-prob gradient
  guards. `out.action_mean.sum().backward()` produces non-None
  `direction_prob_head[0].weight.grad`.

- **Add `test_pre_phase15_weights_fail_to_load`:** old
  state_dict with `direction_prob_head[0].weight` shape
  `(actor_mlp_hidden, runner_embed + hidden)` raises on
  `load_state_dict(strict=True)`.

### Run

`pytest tests/ -k direction` should pass entirely. Run the
broader `pytest tests/` to catch any regression in unrelated
specs that import the policy.

## Done definition

- All updated and new tests pass.
- `git diff` shows changes in `agents_v2/discrete_policy.py`
  and the tests file ONLY.
- No changes in `RUNNER_KEYS`, `feature_engineer.py`,
  trainer, env, or registry.
- Single commit, message:
  `feat(rewrite): phase-15 S01 - direction head reads per-runner feature slice`
