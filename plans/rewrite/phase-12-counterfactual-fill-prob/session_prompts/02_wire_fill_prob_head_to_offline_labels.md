---
session: phase-12-counterfactual-fill-prob / S02
phase: rewrite/phase-12-counterfactual-fill-prob
parent_purpose: ../purpose.md
depends_on: S01
---

# S02 — wire fill_prob_head to offline labels (per-side widening)

## Context

S01 produced offline labels per `(market_id, tick_index,
runner_idx)` with separate `label_back` and `label_lay` fields.
S02 makes the trainer USE those labels:

1. **Widen** `fill_prob_head` from `(max_runners,)` to
   `(max_runners × 2)` — one prediction per runner per side.
2. **Replace** the BCE target source: trainer reads from offline
   cache instead of the agent's bet-manager rollout state.
3. **Remove** the agent-rollout fill_prob label code path
   entirely (mutual exclusion, hard_constraints §5).
4. **Class-balance** the BCE loss with `pos_weight = N_neg/N_pos`
   computed once at trainer construction.

`mature_prob_head` is left ALONE (hard_constraints §6 — Phase 9's
per-transition credit still owns that head). This session touches
fill_prob only.

## Design decisions resolved here

### D5. Replace, not blend

§5 of hard_constraints. The current
`training_v2/discrete_ppo/aux_labels.py` (or wherever the
agent-rollout label is computed) gets ITS fill-prob path
removed. Mature-prob path stays. Two label sources fighting on
one head produces ambiguous gradients.

### D6. Keep the BCE term per-runner-per-transition

The CURRENT Phase 7 fill_prob BCE contributes loss for every
runner at every transition (the head emits `(batch, max_runners)`
and BCE applies element-wise against a same-shape label tensor).
Phase 12 widens the head and the label tensor in tandem; the
loss STAYS per-runner-per-transition. The actor learns to
predict per-runner, per-side fill prob unconditionally.

This is intentional: the offline cache HAS labels for every
priceable runner at every tick, so we should USE all of them
even though the agent only acted on one runner per transition.
Restricting loss to "the runner the agent chose" would waste 13
of 14 labels per transition.

### D7. Mask non-priceable rows out of the loss

If a runner is not priceable at tick T (no LTP, junk filter
fails), there is no label. Use a per-(transition, runner, side)
boolean mask. BCE contribution is zero where the mask is false.
This applies to BOTH sides independently — a runner may be
priceable for back but not lay (rare but possible).

### D8. Per-side `pos_weight` scalars (two values, one per side)

Class imbalance is different for each side:
- back side: ~0.30 – 0.50 positive density expected
- lay side: same magnitude but possibly different direction

Compute `pos_weight_back = N_neg_back / N_pos_back` and
`pos_weight_lay` separately from the cache. Two scalars, applied
to their respective sub-losses. Total fill_prob loss is the
mean over the two side-losses.

### D9. The trainer's `fill_prob_loss_weight` hp scalar still
applies cohort-wide

The cohort runner's existing `fill_prob_loss_weight` knob (Phase 5
gene + reward_overrides) multiplies the COMBINED per-side loss.
No new gene. The class-balancing `pos_weight` is internal to the
loss function and not operator-tunable.

## Pre-reqs

- S01 done. `data/fill_labels/{date}/spread{N}_fc{M}.npz`
  populated for the cohort's training window.

- Read [agents_v2/discrete_policy.py:284-380](../../../../agents_v2/discrete_policy.py)
  — `DiscreteLSTMPolicy.__init__` declarations of
  `fill_prob_head`, `mature_prob_head`, and the actor_input
  concat in `forward`.

- Read [agents_v2/discrete_policy.py::DiscretePolicyOutput](../../../../agents_v2/discrete_policy.py)
  — current dataclass field names. `fill_prob_per_runner`
  (single scalar per runner) becomes
  `fill_prob_back_per_runner` + `fill_prob_lay_per_runner`.

- Read [training_v2/discrete_ppo/aux_labels.py](../../../../training_v2/discrete_ppo/aux_labels.py)
  end-to-end. Identify the function(s) that produce fill-prob
  labels from agent rollout state. Document their signatures
  and what calls them BEFORE removing.

- Read [training_v2/discrete_ppo/trainer.py](../../../../training_v2/discrete_ppo/trainer.py)
  — find where `fill_prob_bce_mean` is computed in
  `_ppo_update`. Note the existing tensor shape and call site.
  The replacement reads from a transition-time lookup into
  the cache instead.

## Architecture-hash break

The widened `fill_prob_head` plus the extra `actor_input` column
break `load_state_dict(strict=True)` against pre-Phase-12
weights. This is the documented behaviour
(hard_constraints.md §8). Tests must verify the failure mode is
clean (clear error message, no half-loaded state).

`actor_head[0].weight.shape[1]` grows from
`runner_embed + backbone + 2` (post-Phase-7 mature_prob_in_actor)
to `runner_embed + backbone + 3` — one new column for the
back-side fill prob, one for lay-side. Mature-prob column stays.

`fill_prob_head` was `nn.Linear(hidden, max_runners)`; becomes
`nn.Linear(hidden, max_runners * 2)`. In forward, reshape to
`(batch, max_runners, 2)` then sigmoid; split into per-side
tensors `fill_prob_back = (..., 0)`, `fill_prob_lay = (..., 1)`.

## Deliverables

### 1. Widen `fill_prob_head` and `actor_input`

In `agents_v2/discrete_policy.py::DiscreteLSTMPolicy.__init__`:

```python
self.fill_prob_head = nn.Linear(
    hidden_size, action_space.max_runners * 2,
)
```

In `forward`:

```python
fill_logit = self.fill_prob_head(lstm_last)  # (batch, R*2)
fill_logit = fill_logit.view(batch, self.max_runners, 2)
fill_prob = torch.sigmoid(fill_logit)         # (batch, R, 2)
fill_prob_back = fill_prob[..., 0]            # (batch, R)
fill_prob_lay = fill_prob[..., 1]
# mature_prob unchanged
mature_prob = torch.sigmoid(self.mature_prob_head(lstm_last))

actor_input = torch.cat([
    runner_embs,
    lstm_expanded,
    fill_prob_back.unsqueeze(-1),
    fill_prob_lay.unsqueeze(-1),
    mature_prob.unsqueeze(-1),
], dim=-1)  # (batch, R, embed + hidden + 3)
```

`actor_head` first layer's `in_features` bumps to
`runner_embed_dim + hidden_size + 3` (or `+ d_model + 3` for the
transformer variant).

`DiscretePolicyOutput` dataclass:

```python
fill_prob_back_per_runner: torch.Tensor   # (batch, R)
fill_prob_lay_per_runner: torch.Tensor    # (batch, R)
# REMOVE: fill_prob_per_runner (consumers updated below)
```

### 2. Find consumers of `fill_prob_per_runner` and update

`grep -rn "fill_prob_per_runner" training_v2/ agents_v2/ tests/`.
Likely call sites:
- aux_labels.py (about to be modified anyway)
- trainer.py BCE loss (about to be replaced)
- maybe a UI / parquet dump path

Pick max(back, lay) at the deprecation site if a single-scalar
backward-compat is needed downstream; otherwise update consumers
to read both fields.

### 3. Per-side label cache lookup

New module `training_v2/discrete_ppo/fill_label_lookup.py`:

```python
import numpy as np
from pathlib import Path
from training_v2.fill_label_scan import load_labels


class FillLabelLookup:
    """In-memory per-(market_id, tick_index, runner_idx) lookup.

    Holds dense tensors per training day. The trainer queries by
    (market_id, tick_index) and gets back `(R, 2)` arrays
    label[runner, side] where side 0 = back, 1 = lay.

    Non-priceable runners get NaN labels and a False mask entry —
    the BCE loss masks them out.
    """

    def __init__(
        self,
        dates: list[str],
        data_dir: Path,
        *,
        arb_spread_ticks: int,
        force_close_before_off_seconds: float,
        max_runners: int,
    ) -> None:
        self.max_runners = int(max_runners)
        self._labels: dict[tuple[str, int], np.ndarray] = {}  # (market_id, tick_idx) -> (R, 2)
        self._mask: dict[tuple[str, int], np.ndarray] = {}    # (market_id, tick_idx) -> (R, 2) bool
        # Pre-flight: load every day's cache. Hard-fail if missing.
        for d in dates:
            rows = load_labels(
                d, data_dir,
                arb_spread_ticks=arb_spread_ticks,
                force_close_before_off_seconds=force_close_before_off_seconds,
                strict=True,
            )
            self._ingest_day(d, rows)

        # Compute cohort-wide pos_weight per side.
        all_back = np.concatenate(
            [np.ravel(arr[..., 0]) for arr in self._labels.values()]
        )
        all_lay = np.concatenate(
            [np.ravel(arr[..., 1]) for arr in self._labels.values()]
        )
        all_back_mask = np.concatenate(
            [np.ravel(m[..., 0]) for m in self._mask.values()]
        )
        all_lay_mask = np.concatenate(
            [np.ravel(m[..., 1]) for m in self._mask.values()]
        )
        self.pos_weight_back = self._compute_pos_weight(
            all_back, all_back_mask,
        )
        self.pos_weight_lay = self._compute_pos_weight(
            all_lay, all_lay_mask,
        )

    @staticmethod
    def _compute_pos_weight(labels: np.ndarray, mask: np.ndarray) -> float:
        labels = labels[mask]
        n_pos = float(np.sum(labels > 0.5))
        n_neg = float(np.sum(labels <= 0.5))
        if n_pos == 0:
            return 1.0  # degenerate; head can't learn anything anyway
        return n_neg / n_pos

    def lookup(
        self, market_id: str, tick_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (label_array, mask_array), each shape (R, 2).

        Missing keys (e.g. an in-play tick) return all-zero label,
        all-False mask — caller treats as "no contribution".
        """
        key = (market_id, int(tick_index))
        if key not in self._labels:
            zeros = np.zeros((self.max_runners, 2), dtype=np.float32)
            mask = np.zeros((self.max_runners, 2), dtype=bool)
            return zeros, mask
        return self._labels[key], self._mask[key]
```

Need a way to map a `Transition`'s `market_id` and `tick_index`
back from rollout: `RolloutCollector` already stores `market_id`
on the transition (verify) and `tick_index` is the env's
`_tick_idx` — both must surface on `Transition`.

### 4. Replace the trainer's fill_prob BCE block

In `training_v2/discrete_ppo/trainer.py::_ppo_update`:

Current (post-Phase-7):

```python
# Approximately:
fill_prob_pred = out.fill_prob_per_runner   # (B, R)
fill_prob_target = batch.fill_prob_label    # (B, R) from aux_labels.py
fill_loss = F.binary_cross_entropy(fill_prob_pred, fill_prob_target)
total_loss += self.fill_prob_loss_weight * fill_loss
```

Phase 12 replacement:

```python
# Per-side prediction
fp_back = out.fill_prob_back_per_runner   # (B, R) sigmoid output
fp_lay = out.fill_prob_lay_per_runner

# Look up labels + masks for this mini-batch's transitions
labels_back = mb_label_back              # (B, R) float32, prefetched
labels_lay = mb_label_lay                # (B, R)
mask_back = mb_mask_back                 # (B, R) bool
mask_lay = mb_mask_lay

# pos_weight tensors broadcast over the per-side BCE
loss_back = F.binary_cross_entropy_with_logits(
    out.fill_prob_back_logit, labels_back,
    pos_weight=torch.tensor(self.pos_weight_back),
    reduction="none",
) * mask_back.float()
loss_lay = F.binary_cross_entropy_with_logits(
    out.fill_prob_lay_logit, labels_lay,
    pos_weight=torch.tensor(self.pos_weight_lay),
    reduction="none",
) * mask_lay.float()

# Mean over (transitions × runners) where mask is True
n_back = mask_back.sum().clamp(min=1)
n_lay = mask_lay.sum().clamp(min=1)
fill_loss = (loss_back.sum() / n_back + loss_lay.sum() / n_lay) / 2.0

total_loss += self.fill_prob_loss_weight * fill_loss
```

To use `binary_cross_entropy_with_logits`, the policy must surface
the pre-sigmoid logits too — add
`fill_prob_back_logit_per_runner` and
`fill_prob_lay_logit_per_runner` to `DiscretePolicyOutput`. The
post-sigmoid values are still surfaced for actor_input.

### 5. Remove the agent-rollout fill_prob label path

In `training_v2/discrete_ppo/aux_labels.py`: identify the
function(s) that produce a per-runner fill-prob label from the
bet manager. Delete them. If the same module produces mature_prob
labels, leave those alone — split the file if needed.

`grep` for callers of the removed functions and update each.
The trainer's pre-Phase-12 fill_prob block is replaced; any
other consumer (UI, parquet dump) either reads the new
per-side fields or is dropped.

### 6. Wire S01 cache load into trainer construction

In `training_v2/cohort/worker.py::train_one_agent`, after policy
+ trainer are built but BEFORE the day loop:

```python
fill_label_lookup = None
if trainer.fill_prob_loss_weight > 0:
    fill_label_lookup = FillLabelLookup(
        dates=days_to_train,
        data_dir=data_dir,
        arb_spread_ticks=resolved_arb_spread_ticks,
        force_close_before_off_seconds=resolved_fc_seconds,
        max_runners=int(shim.max_runners),
    )
    trainer.set_fill_label_lookup(fill_label_lookup)
```

`resolved_arb_spread_ticks = round(20 * arb_spread_scale_resolved)`
where `arb_spread_scale_resolved` is whatever the agent will
actually use (override OR gene OR default 1.0).

Hard-fail if `fill_prob_loss_weight > 0` and any cache is
missing — clear error message: `"Run python -m
training_v2.fill_label_cli scan --dates ... --arb-spread-ticks
{N} --force-close-before-off-seconds {M} first"`.

### 7. Tests — `tests/test_v2_phase12_fill_label_wiring.py`

Six tests:

1. **`test_widened_head_output_shape`** — fresh policy on
   synthetic obs; forward returns
   `fill_prob_back_per_runner.shape == (batch, max_runners)` and
   same for `_lay_`. Both also surface `_logit_` siblings.

2. **`test_actor_input_includes_three_aux_columns`** —
   `actor_head[0].weight.shape[1] == runner_embed + backbone + 3`
   (back fill, lay fill, mature). Bumped by exactly 1 vs Phase 7
   (which had 2 — fill scalar + mature scalar).

3. **`test_grad_through_both_fill_prob_sides`** — perturb
   `fill_prob_head.weight` and confirm `action_mean` changes for
   fixed obs (the head feeds actor_input, so gradient must flow
   back). No detach.

4. **`test_pre_phase12_weights_fail_to_load`** — old
   state_dict with single-scalar `fill_prob_head` of width
   max_runners raises on `load_state_dict(strict=True)` with a
   clear shape-mismatch error.

5. **`test_offline_label_lookup_replaces_rollout_path`** —
   construct a `FillLabelLookup` with synthetic cache; run a
   trainer step; confirm the BCE labels passed to
   `binary_cross_entropy_with_logits` come from the lookup
   (not from any bet-manager state). Test by spying on the
   lookup's `lookup()` method.

6. **`test_pos_weight_computed_per_side`** — synthetic cache
   with known imbalance per side (e.g. back: 80/20 negative,
   lay: 60/40 negative); construct `FillLabelLookup`; assert
   `pos_weight_back ≈ 4.0` and `pos_weight_lay ≈ 1.5` to fp
   tolerance.

### 8. lessons_learnt.md entry

Record:
- Architecture-hash break (precedent: Phase 7 fill_prob_in_actor,
  Phase 7 mature_prob_in_actor).
- Whether `aux_labels.py` was deletable as a unit or had to be
  split (mature path stays).
- `pos_weight_back` / `pos_weight_lay` observed on the cohort's
  training-window cache.
- Any `Transition` schema changes needed (`market_id` and
  `tick_index` already present? if not, what was added).

## Stop conditions

- **Stop and ask** if `aux_labels.py` produces both fill_prob AND
  mature_prob labels in tightly-coupled code that can't be cleanly
  split. Surface the design before tearing out either.

- **Stop and ask** if `Transition` doesn't already carry
  `market_id` and `tick_index`. Adding new transition fields is
  fine but it touches `RolloutCollector` and the GAE path —
  spell out the diff before doing it.

- **Stop and ask** if the per-cohort cache load (S01 output for
  4 days) takes longer than 10 seconds. The cache is small
  (~250k labels per day → a few MB); slow load means the .npz
  format or the dict ingest is doing something silly.

## Done when

- All 6 tests in `tests/test_v2_phase12_fill_label_wiring.py`
  pass.
- `pytest tests/test_v2_*.py -q` green for everything else.
- One-agent training smoke (1 day, 1 gen) on a day with
  populated label cache, `fill_prob_loss_weight=0.10`, runs
  without error and produces non-zero `fill_prob_bce_mean` in
  episode stats.
- Same smoke without the cache hard-fails with the "run scan"
  message (not a generic FileNotFoundError).
- Smoke with `fill_prob_loss_weight=0` runs without ever loading
  the cache (zero-weight short-circuit).
- Commit: `feat(rewrite): phase-12 S02 - fill_prob_head reads
  offline counterfactual labels (per-side widening)`.
