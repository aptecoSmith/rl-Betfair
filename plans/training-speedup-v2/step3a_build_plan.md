# Step 3A — manual batch=N forward: build plan (feasibility DONE 2026-06-01)

Operator decision: pytorch-update path ruled out (vmap-over-LSTM is a
structural gap, won't be unlocked + would break the golden) → **go manual
batch.** This plan is the precise, gated build. Everything below is
verified, not assumed.

## What's proven
- vmap over `nn.LSTM` (`aten::lstm`) is blocked on torch 2.11.0+cu126 and
  structurally across versions (lessons_learnt.md).
- Manual weight-stacking LSTM (`bmm`) matches a per-agent `LSTMCell` loop
  to **1.49e-08** (`C:/tmp/probe_vmap_lstm.py`).
- `DiscreteLSTMPolicy.forward` (agents_v2/discrete_policy.py:853-1109) is
  entirely vmap-able tensor ops EXCEPT: (a) the `self.lstm(...)` call, and
  (b) the `Categorical(logits=...)` + `DiscretePolicyOutput` dataclass
  construction (vmap can't return those). Every python branch in forward
  (`_input_norm_enabled`, `enable_fc_prob_head`, `_runner_block_size`,
  `direction_gate_enabled`, `mature_prob_open_gate_enabled`, the
  `apply_*_gate` defaults) is a CLUSTER-LEVEL flag — identical across all
  agents in a cluster — so vmap traces ONE control-flow path. No
  per-agent data-dependent branch exists. ✔ vmap-safe.

## The approach: vmap + functional_call + manual-LSTM (least code, least drift)
vmap batches every Linear / embedding / head automatically; we only make
the ONE LSTM op vmap-able and stop the forward before the dist.

### Increment 1 — policy refactor (foundation; default-off byte-identical)
1. `__init__`: add `self._manual_lstm_step: bool = False`.
2. New `_lstm_compute(self, proj, hidden) -> (out, (h,c))`: manual LSTM
   matching `nn.LSTM` math (single layer, batch_first), looping seq if
   ctx>1 (in rollout ctx=1). Reads `self.lstm.weight_ih_l0 / weight_hh_l0
   / bias_ih_l0 / bias_hh_l0` so `functional_call` swaps stacked params.
3. New `_forward_tensors(self, obs, hidden, mask, apply_direction_gate,
   apply_mature_prob_gate) -> NamedTuple` containing EVERY tensor field
   forward currently puts in `DiscretePolicyOutput` (logits, masked_logits,
   stake_alpha, stake_beta, value_per_runner, new_hidden h+c, fill_prob,
   mature_prob, risk_mean, risk_log_var, direction_*_prob/logits, fc_prob*)
   — i.e. forward's body lines 861-1089 verbatim, with the LSTM call
   routed: `if self._manual_lstm_step: _lstm_compute else: self.lstm`.
4. `forward` becomes: `t = self._forward_tensors(...); return
   DiscretePolicyOutput(..., action_dist=Categorical(logits=t.masked_logits),
   ...)`. Computation identical → solo byte-identical.
- GATES: golden parity (9 cases) unchanged + `tests/test_agents_v2_discrete_policy.py`
  + `tests/test_policy_network.py` all pass + NEW
  `test_manual_lstm_matches_nn_lstm` (set `_manual_lstm_step=True`, compare
  `_forward_tensors` to nn.LSTM path on real weights, max|Δ| < 1e-4).

### Increment 2 — batched-forward collector
In `batched_rollout.py`, replace the per-agent forward loop with, per tick:
1. Stack active agents' params/buffers via `stack_module_state` (cache &
   restack only when the active set shrinks).
2. Set `_manual_lstm_step=True` on the (shared-arch) policies.
3. `vmap(lambda p,b,o,h,c,m: _forward_tensors_functional(...))(stacked, obs_N,
   h_N, c_N, mask_N)` → batched tensors `(N, …)`.
4. Build ONE `Categorical(logits=masked_logits_N)` + `Beta(alpha_N, beta_N)`;
   sample. **Per-agent RNG independence** must be preserved (HC#2): either
   keep the save/restore window per agent around the batched sample, or
   prove a single batched sample at a cluster seed is acceptable and LOG
   the change. (The current collector saves/restores per agent — a batched
   sample changes RNG ordering; this needs an explicit justification +
   logged decision, or per-agent sampling retained.)
5. Scatter results back to per-agent transition lists (shape unchanged).
- GATES: (a) batched N=1 vs solo `RolloutCollector` at same seed →
  bit-identical (the existing self-parity contract). (b) batched N≥2: each
  agent matches its OWN solo golden within tol (per-agent self-parity —
  the load-bearing guard). (c) the HC#2 no-silent-drop test. (d) measured
  GPU-util + per-tick forward speedup.

### Increment 3 — fold-in feature-parity (operator decision, gated)
Bring `train_cluster_batched` to feature parity with `train_one_agent`
(predictors, feature_cache, input_norm; BC per Step 4). feature_cache is
byte-identical (pure speedup, ~14% wall). Predictors/input_norm change
dynamics but must reproduce the predictors-ON golden (sequential path);
add a batched-path golden test. BC drop logged per HC#2 until Step 4.

## Risks / watch-items
- **RNG ordering** (Increment 2 step 4) is the one place bit-identity may
  legitimately break; handle per HC#2 (preserve or log+justify).
- Manual LSTM vs cuDNN LSTM differ ~1e-6 — within the 1e-4 continuous tol;
  document as float-reordering.
- `stack_module_state` requires identical param KEYS across agents (true
  within a cluster) — validate at collector construction.
- Active-set shrink: restacking params each tick is wasteful; restack only
  on membership change.

## Expected payoff
Forward + sampling ≈ 52% of rollout, kernel-launch-bound at batch=1.
Batching N≈10 agents into one GPU forward should amortise launches hard
(uses the idle GPU on the CPU-bound box). Realistic target: a large cut to
that 52%; measure GPU-util (was ~35%) and per-tick forward wall to size it.
