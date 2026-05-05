---
session: phase-9-per-transition-credit / S02
phase: rewrite/phase-9-per-transition-credit
parent_purpose: ../purpose.md
depends_on: S01
---

# S02 — wire per-transition labels into `DiscretePPOTrainer`

## Context

S01 added `pair_open_records` to the rollout collector output and
`assign_per_transition_labels` to `aux_labels.py`. This session
consumes those in `DiscretePPOTrainer._compute_aux_losses` and
adds the config flag that makes the whole thing an opt-in.

Read `purpose.md` and `hard_constraints.md` (especially §1, §6, §7).
The zero-regression path (`per_transition_credit=false`) must be
byte-identical to Phase 7 — that's the hardest constraint this
session must satisfy.

## Pre-reqs

- S01 done. `assign_per_transition_labels` exists and is tested.
- Read [`training_v2/discrete_ppo/trainer.py`](../../../../training_v2/discrete_ppo/trainer.py)
  — specifically `_compute_aux_losses` (or wherever BCE is applied
  in `_ppo_update`). Understand what per-slot `PerRunnerAuxLabels`
  fields are read and how they reach the mini-batch.
- Read [`training_v2/discrete_ppo/rollout.py`](../../../../training_v2/discrete_ppo/rollout.py)
  — understand the return value of `_collect_rollout` after S01's
  change. Confirm `pair_open_records` is plumbed through.
- Read CLAUDE.md §"v2 stack consumes aux-head loss weights" to
  confirm which fields `DiscretePPOTrainer` currently reads for
  BCE and where `_compute_aux_losses` is called relative to the
  mini-batch loop.

## What changes in the trainer

### Config flag

`DiscretePPOTrainer.__init__` reads:
```python
self.per_transition_credit = bool(
    hp.get("per_transition_credit", False)
)
```

When `False`: existing per-slot path runs unchanged. When `True`:
the per-transition path runs for `mature_prob` BCE only.

### Per-transition path in `_ppo_update`

After calling `_collect_rollout`, if `per_transition_credit=True`:

```python
mature_label_per_step, mature_mask_per_step = assign_per_transition_labels(
    rollout.pair_open_records,
    list(env.all_settled_bets) + list(env.bet_manager.bets),
    n_steps=len(rollout.transitions),
)
```

Store these as tensors alongside the existing rollout data. In the
mini-batch loop, slice by `mb_idx` (same as rewards, advantages,
old_log_probs):

```python
mb_mature_label = mature_label_per_step[mb_idx]   # (mb_size,)
mb_mature_mask  = mature_mask_per_step[mb_idx]    # (mb_size,) bool
```

BCE for mature_prob in the per-transition path:

```python
if mb_mature_mask.any():
    mature_logit_at_open = out.mature_logit[mb_mature_mask, ...]
    # mature_logit shape: (mb_size, max_runners)
    # We need the logit for the specific runner that was opened at
    # each open step. The runner_slot is stored in pair_open_records
    # and must be threaded through to a per-step runner_slot array.
    mature_bce = F.binary_cross_entropy_with_logits(
        mature_logit_at_open[arange, runner_slot_at_step[mb_mature_mask]],
        mb_mature_label[mb_mature_mask],
    )
else:
    mature_bce = torch.zeros((), device=self.device)

mature_prob_loss = mature_bce
```

**`runner_slot_at_step`** is a `(n_steps,) int64` array built from
`pair_open_records` — assign `slot` at `step_index`, leave as 0
elsewhere (the mask ensures only open steps contribute).

### Logging

Per-update log line gains `n_mature_targets=N` when
`per_transition_credit=True`, where N = number of open-step
transitions in this mini-batch that have `mature_mask=True`.
On a healthy run this should be ~2–6 per mini-batch of 64 steps
(~200–500 open steps per 11k-transition rollout / 156 mini-batches).
A value of 0 on every mini-batch means no pairs were opened — worth
warning on.

The per-slot `fill_prob` BCE and `risk_nll` are unchanged — they
continue to use `PerRunnerAuxLabels` regardless of the flag.

### Per-episode JSONL

Add `per_transition_credit_active: bool` to the episode stats row
(§5 of hard_constraints.md). False when the flag is off.

## Tests

Extend `tests/test_v2_per_transition_credit.py`:

8. `test_per_slot_path_byte_identical_when_disabled` — **load-bearing
   regression guard (§6).** Run one real training update with
   `per_transition_credit=False`; capture `policy_loss`,
   `value_loss`, `approx_kl`. Run the same update at the same seed
   with `per_transition_credit=True` disabled. Assert all three are
   bit-for-bit identical. Use `torch.manual_seed` around both runs.
9. `test_n_mature_targets_is_zero_for_naked_only_rollout` — synthetic
   rollout where all pairs go naked; confirm `n_mature_targets=0` in
   the log line and no `NaN` in the loss.
10. `test_n_mature_targets_nonzero_when_pairs_mature` — synthetic
    rollout with 5 matured pairs; confirm `n_mature_targets > 0`.
11. `test_bce_targets_concentrated_not_broadcast` — with
    `per_transition_credit=True`, count how many mini-batch entries
    have `mature_mask=True` across a full update. Assert it is ≪ the
    total transition count (i.e. the label is sparse, not broadcast).
    Specifically: total masked entries across all mini-batches ≤
    `2 × n_pairs_opened` (at most 2 mini-batches could contain the
    same open step when mini-batch boundaries split it — shouldn't
    happen but guard for it).

## Stop conditions

- **Stop and ask** if the mini-batch loop slices transitions by
  a non-integer index (e.g. shuffled or stratified sampling) that
  makes `mature_label_per_step[mb_idx]` ambiguous. The per-step
  label requires that `mb_idx` is the actual rollout step index,
  not a shuffled position. If the trainer shuffles before slicing,
  the label array must be shuffled in the same order.
- **Stop and ask** if `out.mature_logit` shape is `(batch, max_runners)`
  but the batch dimension at update time is per-step (single obs)
  rather than per-runner. Confirm the shape before indexing.
- **Stop and ask** if test #8 (byte-identity) fails by any amount.
  This means the `per_transition_credit=False` path is not truly
  skipped — there's a subtle interaction (e.g. a tensor is created
  even when masked, changing RNG state). Fix before continuing.

## Done when

- All 11 tests in `tests/test_v2_per_transition_credit.py` pass
  (7 from S01 + 4 new).
- Smoke cohort: `python -m training_v2.cohort.runner --n-agents 2
  --generations 1 --days 2 --device cuda --seed 42 --data-dir
  data/processed --per-transition-credit true
  --output-dir registry/_phase9_s02_smoke` completes; per-update
  log shows `n_mature_targets=N` (N > 0 in at least some updates).
- Same smoke with `--per-transition-credit false` shows no
  `n_mature_targets` field and identical statistics to Phase 7
  baseline at same seed.
- Commit: `feat(rewrite): phase-9 S02 - per-transition mature_prob
  BCE in DiscretePPOTrainer`.
