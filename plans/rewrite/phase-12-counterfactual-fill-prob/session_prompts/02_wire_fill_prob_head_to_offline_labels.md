---
session: phase-12-counterfactual-fill-prob / S02
phase: rewrite/phase-12-counterfactual-fill-prob
parent_purpose: ../purpose.md
depends_on: S01
---

# S02 — wire fill_prob_head to offline labels + widen to per-side

## Context

S01 produced the offline label cache. S02 makes the trainer USE
those labels: replace the existing agent-rollout-derived BCE
target on `fill_prob_head` with the offline label, and widen the
head from `(max_runners,)` to `(max_runners × 2)` so each side
has its own predictor.

Read `purpose.md` and `hard_constraints.md` first. The architecture-
hash break is permitted (§8). `mature_prob_head` is untouched (§6).

## Pre-reqs

- S01 done; `data/fill_labels/{date}/{spread}_ticks.npz` populated
  for the training-window dates.

- Read [agents_v2/discrete_policy.py:284-380](../../../../agents_v2/discrete_policy.py)
  — `DiscreteLSTMPolicy.__init__` and the relevant
  `fill_prob_head` / `mature_prob_head` declarations.

- Read [training_v2/discrete_ppo/trainer.py](../../../../training_v2/discrete_ppo/trainer.py)
  — find the existing fill_prob BCE block. Note where it reads
  the label tensor from. Document the path before touching.

- Read [training_v2/discrete_ppo/aux_labels.py](../../../../training_v2/discrete_ppo/aux_labels.py)
  if it exists — that's likely where the agent-rollout label is
  computed today.

## Decision: replace, don't blend

§5 of hard_constraints.md is mutual exclusion. The agent-rollout
label path is removed once Phase 12 ships. Mixing two label
sources fights on the same head.

Concretely: delete (or stub to no-op + warn) the code that
extracts a per-rollout fill_prob label from the agent's bet
manager. Replace it with a transition-time lookup into the
offline cache: at training time, for each transition's
`(race_market_id, tick_index, runner_idx, side)`, fetch the
corresponding offline label.

## Deliverables

### 1. Widen fill_prob_head to (max_runners × 2)

Per-runner per-side prediction. Output shape becomes
`(batch, max_runners, 2)`. Forward concatenates the per-side
sigmoids into actor_input as separate columns:

```
actor_input = concat([
    runner_embs,
    backbone_expanded,
    fill_prob[..., 0].unsqueeze(-1),   # back-side fill prob
    fill_prob[..., 1].unsqueeze(-1),   # lay-side fill prob
    mature_prob.unsqueeze(-1),
], dim=-1)
```

`actor_head[0].weight.shape[1]` grows from `runner_embed +
backbone + 2` to `runner_embed + backbone + 3` — bump the
architecture-hash and document in lessons_learnt.

Surface the per-side predictions on `DiscretePolicyOutput`:
`fill_prob_back_per_runner` and `fill_prob_lay_per_runner`,
deprecating the single-scalar `fill_prob_per_runner` (keep as
`max(back, lay)` for backward-compat with consumer code that
hasn't migrated yet, OR remove entirely and clean up consumers
in this session).

### 2. Replace the fill_prob BCE target

The trainer's `fill_prob_loss_weight` BCE term currently reads
from per-rollout-derived labels (whatever `aux_labels.py`
produces). Replace with offline-cache lookup:

```python
# At trainer construction (or _build_trainer_hp time) load the
# label cache for every training day this agent will see.
# Pre-flight: assert all required day caches exist.
label_cache = load_offline_labels_for_dates(
    days_to_train,
    data_dir=data_dir,
    arb_spread_ticks=resolved_arb_spread_ticks,
)
# label_cache: dict[(market_id, tick_idx)] -> ndarray(max_runners, 2)
```

At update time, for each transition in the rollout, look up:
```
label[t] = label_cache[market_id_t, tick_idx_t]   # shape (R, 2)
```

Stack across transitions, BCE against the head's per-side
predictions for the chosen runner only. Ignore transitions that
weren't on a priceable open tick (NOOP transitions don't carry
a label).

§11 (class imbalance): compute `pos_weight` from the cache once
at trainer construction:
```
pos_weight = (n_neg / n_pos) for back side
              same for lay side
```
Pass to `F.binary_cross_entropy_with_logits(...,
pos_weight=...)`.

### 3. mature_prob_head label path UNCHANGED

§6. Phase 9's per-transition credit code stays exactly as-is.
Both heads run, both contribute to actor_input, both have BCE
loss terms — but they read from different label sources.

### 4. Tests `tests/test_v2_phase12_fill_label_wiring.py`

Six tests:

1. `test_widened_head_output_shape` — fresh policy; forward
   pass produces `fill_prob_back_per_runner` and
   `fill_prob_lay_per_runner` each shape `(batch, max_runners)`.

2. `test_actor_input_includes_both_fill_prob_columns` — assert
   `actor_head[0].weight.shape[1] == runner_embed + backbone + 3`.

3. `test_grad_through_both_fill_prob_heads` — perturb fill_prob
   weights for either side, confirm `action_mean` changes for
   fixed obs (no detach).

4. `test_pre_phase12_weights_fail_to_load` — old state_dict (single
   `fill_prob_head` of width max_runners) raises on
   `load_state_dict(strict=True)`.

5. `test_offline_label_lookup_replaces_rollout_path` — synthetic
   1-day rollout with a known offline cache; confirm the BCE
   loss tensor matches the cached labels (sample 10 transitions,
   compare).

6. `test_pos_weight_computed_from_cache` — load a synthetic
   cache with known label imbalance (e.g. 80/20); assert
   `pos_weight` set on the loss = `0.8 / 0.2 = 4.0` to within
   floating-point.

### 5. lessons_learnt.md entry

Record:
- The architecture-hash break (precedent: Phase 7 fill_prob_in_actor
  + Phase 7 mature_prob_in_actor).
- Pre-flight check shape: missing labels for any training day
  → hard-fail with a clear "run fill_label_cli first" message.
- pos_weight observed on a real training-day cache.
- Whether removing the rollout-derived label path required
  removing `aux_labels.py` entirely or just dead-coding part of
  it.

## Stop conditions

- **Stop and ask** if `aux_labels.py` (or wherever the
  agent-rollout fill_prob label lives) ALSO computes
  mature_prob_head labels in the same code block. Splitting one
  vs the other may need careful refactoring; surface the design
  before tearing out either.

- **Stop and ask** if the per-side widening conflicts with
  Phase 5 / 7 plumbing in the trainer's `hp` dict (the trainer
  reads `fill_prob_loss_weight` from hp; per-side weighting may
  want two scalars, e.g. `fill_prob_back_loss_weight` and
  `fill_prob_lay_loss_weight`. Decision: start with one shared
  weight applied to both sides, escalate to two if S03 needs it).

## Done when

- All 6 tests pass.
- `pytest tests/test_v2_*.py -q` green for everything else.
- One-agent training smoke (1 day, 1 gen) on a day with a
  populated label cache produces non-zero `fill_prob_bce_mean`
  in episode stats.
- Same training smoke without the label cache hard-fails with
  a clear message.
- Commit: `feat(rewrite): phase-12 S02 - fill_prob_head reads
  offline counterfactual labels; widen to per-side`.
