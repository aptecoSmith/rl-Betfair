---
plan: rewrite/phase-16-ensemble-market-state
session: S01
deliverable: K=5 ensemble of direction predictors with AND-gate at inference
---

# S01 — Ensemble of K=5 direction predictors

## Goal

Replace the single `direction_prob_head` with K=5 independent
predictors. At inference the gate fires only when all K predict
the runner crosses threshold (`min_k(P_back_k) >= T`), giving
implicit consensus-based uncertainty.

## File-level changes

### `agents_v2/discrete_policy.py`

Currently:
```python
self.direction_prob_head = nn.Sequential(
    nn.LayerNorm(RUNNER_DIM),
    nn.Linear(RUNNER_DIM, self.actor_mlp_hidden),
    nn.ReLU(),
    nn.Linear(self.actor_mlp_hidden, 2),
)
```

After S01:
```python
ENSEMBLE_K = 5
self.direction_prob_heads = nn.ModuleList([
    nn.Sequential(
        nn.LayerNorm(RUNNER_DIM),
        nn.Linear(RUNNER_DIM, self.actor_mlp_hidden),
        nn.ReLU(),
        nn.Linear(self.actor_mlp_hidden, 2),
    ) for _ in range(ENSEMBLE_K)
])
```

In `forward`, run all K through the per-runner slice:
```python
direction_logits_k = []
for head in self.direction_prob_heads:
    flat = runner_feats_raw.reshape(batch * R, RUNNER_DIM)
    out_k = head(flat).view(batch, R, 2)
    direction_logits_k.append(out_k)
direction_logits = torch.stack(direction_logits_k, dim=0)  # (K, batch, R, 2)
direction_probs = torch.sigmoid(direction_logits)

# AND-gate: min across K
direction_back_prob = direction_probs[..., 0].min(dim=0).values  # (batch, R)
direction_lay_prob = direction_probs[..., 1].min(dim=0).values

# For backward compat (BC trainer reads logits per K), also expose:
self._last_direction_logits_k = direction_logits  # (K, batch, R, 2)
```

`PolicyOutput.direction_back_logits_per_runner` should be
`direction_logits_k.min(dim=0)` — the gating logit. The K-stack
itself goes on a new field `direction_logits_k_per_runner`
(K, batch, R, 2) for the BC trainer.

### `training_v2/discrete_ppo/bc_pretrain.py`

BC needs to train all K heads simultaneously. Two approaches:

**A. Independent BC for each head (simpler):**
- Initialize K different optimizer states (one per head)
- Each step, sample a batch and compute BCE for each head
- Each head's gradient flows ONLY to its own params (frozen
  rest, including other K-1 heads)
- Sum all K losses for the backward pass

**B. Single optimizer over all K params:**
- Single backward of `sum_k loss_k`
- Adam tracks each parameter separately anyway, so the
  effect is the same as A

Use B (simpler). Code:

```python
target_params = []
for name, p in policy.named_parameters():
    if "direction_prob_heads" in name:
        target_params.append(p)
    elif "actor_head" in name:
        target_params.append(p)
    else:
        frozen.append(p)
```

In the loss compute step, iterate over the K heads and
accumulate per-head BCE:

```python
total_dir_bce = 0.0
for k in range(ENSEMBLE_K):
    bb = direction_logits_k[k, ...]  # (batch, R, 2) for head k
    # ... existing BCE code, indexed by k
    total_dir_bce += bce_back_k + bce_lay_k
loss = oracle_w * oracle_ce + dir_bce_w * total_dir_bce / ENSEMBLE_K
```

The `/K` keeps the loss magnitude comparable to single-head.

### Freeze (worker.py)

Same as phase-15 — freeze all `direction_prob_heads` parameters
post-BC:
```python
for name, p in policy.named_parameters():
    if "direction_prob_heads" in name:
        p.requires_grad_(False)
```

### Gate logic (discrete_policy._apply_direction_gate)

Already operates on `direction_back_prob` / `direction_lay_prob`.
After S01, those are the min-of-K values, so no logic change
needed — just data substitution.

## Tests

- `test_ensemble_has_k_independent_heads`:
  `len(policy.direction_prob_heads) == 5`; each head has independent
  parameters (e.g., perturb one head's weight, others unchanged).
- `test_ensemble_gate_uses_min_of_k`:
  set fake `direction_logits_k` such that head 0 outputs sigmoid >T
  but heads 1-4 output sigmoid < T. Gate should NOT fire.
  Conversely all K above T → gate fires.
- `test_bc_trains_all_k_heads`:
  BC step changes weights of all K heads (snapshot before/after).
- `test_freeze_freezes_all_k_heads`:
  After freeze, all K heads have `requires_grad=False`.
- `test_pre_s01_weights_fail_to_load`:
  pre-phase-16 state_dict has `direction_prob_head.0.weight` (single
  head); post-S01 has `direction_prob_heads.0.0.weight` (modulelist).
  Strict load refuses.

## Smoke

Run `phase15_v8_replicate.sh` shape but with K=5 ensemble in
place:
- 4 agents × 1 gen × 3 train + 1 eval (small)
- BC pretrain 2000 steps, all 5 heads trained
- T=0.85 gate, freeze
- Wall: ~1.5h (5× BC = ~5min × 4 agents = 20min added vs phase-15)

Pass criteria:
- All 4 agents complete (no NaN, no crash)
- Mat rate matches OR EXCEEDS the K=1 baseline
- Bet count is REDUCED vs K=1 baseline (consensus filters
  out borderline decisions)

## Done definition

- All 5 tests pass
- Smoke completes with mat rate >= K=1 baseline
- Single commit: `feat(rewrite): phase-16 S01 - K=5 direction predictor ensemble`
