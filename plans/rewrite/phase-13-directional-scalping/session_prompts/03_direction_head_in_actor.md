---
session: phase-13-directional-scalping / S03
phase: rewrite/phase-13-directional-scalping
parent_purpose: ../purpose.md
---

# S03 — direction_prob_head wired to offline labels, feeds actor_head

## Context

Read `purpose.md`, `hard_constraints.md`, S01 `findings.md`, and
S02's deliverable. This session adds a third aux head alongside
`fill_prob_head` and `mature_prob_head`. The new head:

- Takes the same backbone output as the other two heads
  (`lstm_last` for the LSTM variants; `out_last` for the
  transformer).
- Emits `(batch, max_runners × 2)` — TWO scalars per runner, one
  per side.
- Sigmoid is applied; the output is concat'd into `actor_input` as
  TWO new per-runner columns.
- Trains via BCE on the offline labels from S02.

The pattern is a literal copy of `fill-prob-in-actor` and
`mature-prob-in-actor` (CLAUDE.md). **Do not invent new structure
here.** Re-using the precedent gives us the regression-test
templates and the architecture-hash protocol for free.

This session does NOT do validation cohorts (S06 owns that), does
NOT add the BC pretrain target (S05 owns that), does NOT touch
reward shaping (no plan-level changes — see hard_constraints §12).

## Pre-reqs

Read these before writing code:

- [agents_v2/discrete_policy.py](../../../../agents_v2/
  discrete_policy.py) — the `DiscreteLSTMPolicy` class. Find
  `fill_prob_head`, `mature_prob_head`, see how their sigmoid
  outputs land in `actor_input`. Lines `~258-510` cover the head
  declarations and forward pass (line numbers may have shifted —
  read them).

- [training_v2/discrete_ppo/trainer.py](../../../../training_v2/
  discrete_ppo/trainer.py) — find how `fill_prob_loss_weight` and
  `mature_prob_loss_weight` are read from `hp`, how the BCE loss
  is computed, and how cached labels feed in. Lines `~313-340`
  for the gene reads, `~625-700` for the loss computation.

- [training_v2/discrete_ppo/aux_labels.py](../../../../training_v2/
  discrete_ppo/aux_labels.py) — the existing label-loading
  plumbing. Direction labels load via the same pattern from S02's
  `load_labels`.

- CLAUDE.md sections **"fill_prob feeds actor_head (2026-04-26)"**
  and **"mature_prob_head feeds actor_head (2026-04-26)"** — the
  contract these heads operate under. Direction head joins the
  same contract.

- `tests/test_policy_network.py::TestFillProbInActor` and
  `TestMatureProbInActor` — copy the test patterns. The
  `test_*_pre_*_weights_fail_to_load` test is the load-bearing
  regression guard for the architecture-hash break.

- v1 policy classes if v1 is still in use:
  `agents/policy_network.py::PPOLSTMPolicy`,
  `PPOTimeLSTMPolicy`, `PPOTransformerPolicy`. Same head added to
  each.

## Design decisions resolved here (don't re-litigate)

### D1. Two output dims per runner — `(max_runners × 2)`

`direction_prob_head: nn.Linear(hidden, max_runners * 2)`. The
output is reshaped to `(batch, max_runners, 2)` and sigmoid'd per
element. The two columns mean
`[direction_back_prob, direction_lay_prob]`. They feed actor_input
in this order:

```python
actor_input = torch.cat([
    runner_embs,
    backbone_expanded,
    fill_prob.unsqueeze(-1),        # existing
    mature_prob.unsqueeze(-1),      # existing
    direction_back_prob.unsqueeze(-1),  # NEW
    direction_lay_prob.unsqueeze(-1),   # NEW
], dim=-1)
```

`actor_head[0].weight.shape[1]` becomes
`runner_embed_dim + lstm_hidden + 4` (or `+ d_model + 4` for
transformer). +2 over the post-mature_prob shape.

### D2. BCE loss — same shape as fill_prob / mature_prob

Two scalars per runner-transition, two BCE terms. Sum them, weight
by `direction_prob_loss_weight`, add to the trainer's total loss:

```python
direction_loss = (
    F.binary_cross_entropy(
        direction_back_prob, label_back, reduction="none",
    ) +
    F.binary_cross_entropy(
        direction_lay_prob, label_lay, reduction="none",
    )
).mean()

total_loss = (
    surrogate_loss
    + value_loss * value_loss_weight
    + entropy_term
    + fill_prob_bce * fill_prob_loss_weight
    + mature_prob_bce * mature_prob_loss_weight
    + risk_nll * risk_loss_weight
    + direction_loss * direction_prob_loss_weight  # NEW
)
```

Mask out non-priceable runners (label = NaN sentinel) before BCE
— the same `priceable_mask` plumbing fill_prob already uses.

### D3. Class-balance — symmetric weight

S02 will report per-class density `(positive_back, positive_lay)`.
Mirror phase-12's class-balance approach: a single positive-class
weight per side, computed at trainer init from the cache density:

```python
pos_weight_back = (1 - density_back) / max(density_back, 1e-6)
pos_weight_lay  = (1 - density_lay)  / max(density_lay,  1e-6)
```

Use `F.binary_cross_entropy_with_logits(..., pos_weight=...)`
rather than `binary_cross_entropy` on sigmoid output — cleaner and
numerically more stable. The sigmoid is still applied for the
actor_input column; the loss reads the LOGITS.

### D4. NO gene-gating boolean — head is unconditional

`direction_prob_head` is always present in the network. Its
`actor_input` contribution is always present. The gating is via
`direction_prob_loss_weight`:

- weight `0.0` (default): head exists but has no BCE supervision,
  outputs near `sigmoid(0) ≈ 0.5` constant; the actor_input
  column is benign noise. Same precedent as fill_prob /
  mature_prob default.
- weight `> 0.0`: head trains; actor_input column carries
  meaningful per-runner direction signal.

### D5. Architecture-hash break is the variant identity

`actor_head[0].weight.shape[1] == runner_embed + backbone + 4` is
the structural fingerprint. PyTorch's
`load_state_dict(..., strict=True)` refuses pre-S03 checkpoints
with a shape mismatch on `actor_head.0.weight`. **No new explicit
version field.** This is the same protocol fill-prob-in-actor and
mature-prob-in-actor follow.

## Deliverables

### 1. New head + actor_input plumbing in all policy classes

- `agents_v2/discrete_policy.py::DiscreteLSTMPolicy.__init__`:
  - Add `self.direction_prob_head = nn.Linear(hidden,
    max_runners * 2)` next to `mature_prob_head`.
  - Update `actor_head[0]` input dim from
    `runner_embed + lstm_hidden + 2` to
    `runner_embed + lstm_hidden + 4`.

- `DiscreteLSTMPolicy.forward`:
  - Compute `direction_logits = self.direction_prob_head(lstm_last)`
    of shape `(batch, max_runners * 2)`.
  - Reshape to `(batch, max_runners, 2)`.
  - Sigmoid for the actor_input columns; KEEP the logits on the
    `PolicyOutput` for the trainer to compute BCE-with-logits.
  - Concat the two sigmoid columns into `actor_input` per D1.

- `PolicyOutput` dataclass: add fields
  `direction_back_logits`, `direction_lay_logits` (raw logits) and
  `direction_back_prob`, `direction_lay_prob` (sigmoid). Mirror
  the existing fill/mature naming.

- v1 policy classes if applicable: `PPOLSTMPolicy`,
  `PPOTimeLSTMPolicy`, `PPOTransformerPolicy` get the same
  treatment. Their tests in `tests/test_policy_network.py` already
  cover fill-prob and mature-prob shape; extend to direction.

### 2. Trainer: read gene, compute BCE loss, wire into total_loss

- `training_v2/discrete_ppo/trainer.py::DiscretePPOTrainer.__init__`:
  - Read `self.direction_prob_loss_weight = float(hp.get(
    "direction_prob_loss_weight", 0.0) or 0.0)`. **READ FROM
    `hp` ONLY** — see lessons-learnt v2-specific note about the
    `hp.get(name, config["reward"][name])` precedence trap.
  - On non-zero weight, load the offline label cache for the day's
    training schedule via `direction_label_scan.load_labels`.
    Compute and cache `pos_weight_back`, `pos_weight_lay` from the
    cache density.

- `_compute_aux_losses` (or wherever fill_prob's BCE is computed):
  - Add the direction-back and direction-lay BCE-with-logits
    contributions, masked by `priceable_mask` from the cache.
  - Surface `direction_back_bce_mean`, `direction_lay_bce_mean`,
    `direction_back_pos_rate`, `direction_lay_pos_rate` on
    `loss_info` for episodes.jsonl.

### 3. CohortGenes / config wiring

- `training_v2/cohort/genes.py::CohortGenes`: add
  `direction_prob_loss_weight: float = 0.0`. Range bound
  `[0.0, 1.0]` (matching fill_prob / mature_prob ranges).
- `training_v2/cohort/worker.py::_build_trainer_hp`: ensure
  `--reward-overrides direction_prob_loss_weight=X` pre-merges
  into `hp` before trainer construction. Lessons-learnt v2-
  specific note explicitly calls this out.
- `config.yaml`: add `reward.direction_prob_loss_weight: 0.0`
  default (mirrors fill/mature for v1 reads). `reward_overrides`
  passthrough key list (if it exists) gets the new key.

### 4. Tests — `tests/test_policy_network.py::TestDirectionProbInActor`

Mirror the four-test-per-class pattern (LSTM, TimeLSTM,
Transformer if v1 still in scope; DiscreteLSTM for v2):

1. `test_{policy}_actor_input_includes_direction_prob` —
   `actor_head[0].weight.shape[1] == runner_embed + backbone + 4`.

2. `test_{policy}_action_mean_depends_on_direction_prob_head_weights`
   — perturbing `direction_prob_head.weight` changes `action_mean`
   for fixed obs / hidden_state. Forward-side gradient guard.

3. `test_{policy}_actor_loss_routes_grad_through_direction_prob_head`
   — `out.action_mean.sum().backward()` produces non-None
   `direction_prob_head.weight.grad`. Backward-side gradient
   guard.

4. `test_{policy}_pre_direction_weights_fail_to_load` — old
   state_dict (post-mature, three-narrower `actor_head[0].weight`)
   raises on `load_state_dict(..., strict=True)`.

Cross-update existing `TestMatureProbInActor::
test_*_pre_mature_weights_fail_to_load` to use `old_extra_dim=4`
on the post-direction shape (precedent: that test already got
updated when fill-prob landed on top of pre-fill-prob shape).

### 5. Trainer integration test

In `tests/test_v2_trainer.py` (or wherever
`fill_prob_loss_weight` integration tests live):

- `test_direction_loss_zero_when_weight_zero` — `hp =
  {"direction_prob_loss_weight": 0.0}`; one-rollout dummy run;
  assert direction terms in `loss_info` are present (head still
  forward-passes) but `direction_back_bce_mean × weight` and
  `direction_lay_bce_mean × weight` contribute zero to total_loss.

- `test_direction_loss_nonzero_when_weight_positive` — `hp =
  {"direction_prob_loss_weight": 0.1}`; assert non-zero BCE.

- `test_direction_label_cache_missing_raises` — when
  `direction_prob_loss_weight > 0` and cache missing for a day,
  trainer init raises `FileNotFoundError` with the cache path.
  Hard_constraints §22.

- `test_direction_pos_weight_matches_cache_density` — sanity
  check that `pos_weight_back = (1-d)/d` for the cache's reported
  density.

### 6. `loss_info` / episodes.jsonl additions

Per-episode rows gain (default-tolerant on absence):

- `direction_back_bce_mean` — float, mean BCE on back labels this
  rollout.
- `direction_lay_bce_mean` — float, same for lay.
- `direction_back_pos_rate_in_batch` — float, fraction of priceable
  transitions with `label_back = 1` in this rollout.
- `direction_lay_pos_rate_in_batch` — float, same for lay.
- `direction_prob_loss_weight_active` — float, the value used
  (carries the gene through to the operator panel).

### 7. lessons_learnt.md entry

- Whether the head trains cleanly (BCE trends down across the
  first 3 rollouts) or stalls.
- Calibration check after 3 rollouts: bin the `direction_back_prob`
  outputs into 10 quantile bins; for each bin, compute realised
  `label_back == 1` fraction; report max bin-level deviation
  (predicted vs realised). Healthy: < 0.10.
- Any divergence from the fill_prob / mature_prob templates
  (cache shape mismatches, label-loading edge cases).
- Architecture-hash break test confirmation: pre-S03 checkpoint
  fails to load, post-S03 round-trip works.

## Stop conditions

- **Stop and ask** if the architecture-hash test fails for any
  policy class — the strict-load precedent is load-bearing for all
  three of the in-actor heads, and a regression here breaks the
  contract for fill_prob and mature_prob too.

- **Stop and ask** if BCE loss does NOT trend down across the
  first 3 rollouts at `direction_prob_loss_weight = 0.1` on a real
  day's cache. Either S02's labels are uninformative (revisit
  threshold / horizon) or the trainer wiring has a bug.

- **Stop and ask** if calibration max-bin deviation > 0.20 after
  3 rollouts. Either pos_weight is mis-computed or the head's
  capacity is too thin (head is a single Linear; this would be
  surprising but possible if the backbone hidden is small).

- **Stop and ask** if v1 policy classes are in active use AND
  adding the head to them would require non-trivial refactoring
  (e.g. their `forward` doesn't currently expose
  `lstm_last`-equivalent in a clean way). Decide whether to ship
  v2-only (with a `lessons_learnt.md` note) or to extend the
  scope.

## Done when

- `direction_prob_head` exists in all in-scope policy classes.
- `actor_head[0].weight.shape[1]` reflects the +4 vs pre-fill-prob
  shape (or +2 vs post-mature shape).
- Trainer reads `direction_prob_loss_weight` from `hp`; loss is
  wired in.
- All four `TestDirectionProbInActor` tests per policy class pass.
- All four trainer-integration tests pass.
- Existing fill_prob / mature_prob tests still pass (cross-update
  done).
- `episodes.jsonl` shows the four new direction fields populated
  on a probe run.
- `lessons_learnt.md` updated with calibration finding.
- Commit: `feat(rewrite): phase-13 S03 - direction_prob_head feeds
  actor_head, BCE on offline labels`.
