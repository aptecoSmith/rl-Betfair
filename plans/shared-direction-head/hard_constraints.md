# Hard constraints — shared frozen direction head

## §1. Eval / monitor days are off-limits during head training

The shared head's training set is strictly the 16 cohort-training
days listed in `purpose.md`. The 10 eval days and 14 monitor days
MUST be excluded from the head's training corpus AND its
validation-split. The manifest records the day list and any
training script run-on those held-out days fails a pre-flight
check.

**Rationale:** the cohort agents are trained on training days,
evaluated on eval days. If the head sees eval days during its
training, the agents would effectively peek at the answer when
they use the head's output. Same logic as why the
`betfair-predictors` direction model was trained on a separate
corpus from our cohort eval days.

## §2. Held-out validation INSIDE the training days

Within the 16 training days, the head's training script holds out
20 % of (tick, runner) samples for validation. The manifest's
`val_bce_*` numbers reference THIS internal hold-out, not the
cohort's eval days. Acceptance criterion §3 of `purpose.md` is
gated on this number.

## §3. Architecture-hash break on the cohort policy

When the cohort policy loads a pre-trained head, its weights for
`direction_prob_head` come from a SEPARATE file (the manifest's
`weights.pt`), not from the agent's own `.pt`. Old per-agent
weights for `direction_prob_head` in pre-fix `<agent>.pt` files
are now stale / ignored — the policy overwrites them at
construction.

To make this explicit:

* `DiscreteLSTMPolicy.__init__` accepts a new `frozen_direction_head_path:
  Path | None` kwarg.
* When supplied: AFTER the head is constructed (current code), load
  the weights from the file, call `requires_grad_(False)` on every
  head parameter, and assert the head is in eval mode during the
  policy's forward.
* When `None` (default): behaviour is byte-identical to today.

## §4. Inert genes when the head is frozen

When the cohort runs with `--direction-head-manifest <path>`:

* `direction_prob_loss_weight` is FORCED to 0.0 in trainer_hp (no
  supervised loss attempted on a frozen head).
* `bc_direction_target_weight` is FORCED to 0.0 (no BC supervision
  on a frozen head).

If the operator passes `--enable-gene direction_prob_loss_weight`
AND `--direction-head-manifest`, the runner raises with a clear
mutual-exclusion message — the two flags are incompatible.

Per-agent scoreboard rows record `direction_head_manifest_id` (or
"<not_loaded>") so post-hoc analysis can distinguish frozen-head
runs from learned-head runs.

## §5. Manifest schema

`models/direction_head/<exp_id>/manifest.json` carries:

```json
{
  "experiment_id": "<sha or human-readable id>",
  "weights_path": "weights.pt",
  "architecture": {
    "family": "linear_mlp",  // or whatever
    "input_dim": 23,           // RUNNER_KEYS lean obs per runner
    "output_dim": 2,           // (back_prob, lay_prob) per runner
    "hidden_dims": [64]        // example
  },
  "training": {
    "training_dates": [...16 dates...],
    "label_version": "v1_threshold_crossing",
    "direction_horizon_ticks": 60,
    "direction_threshold_ticks": 5,
    "force_close_before_off_seconds": 60.0
  },
  "val_metrics": {
    "val_bce_back": 0.xxx,
    "val_bce_lay": 0.xxx,
    "n_train": ...,
    "n_val": ...,
    "pos_rate_back": 0.18,
    "pos_rate_lay": 0.18
  },
  "obs_schema_version": 9,    // matches env's expected obs schema
  "active_runner_dim": 23,    // lean obs — the head's input dim
  "created_at": "...",
  "commit_sha": "..."
}
```

The cohort runner refuses to load a manifest whose
`obs_schema_version`, `active_runner_dim`, or
`direction_horizon_ticks` don't match the env's current values.
Same fail-fast pattern as the existing cache schema checks.

## §6. Tests

Regression guards at `tests/test_shared_direction_head.py`:

a. Loading a manifest into a fresh policy correctly freezes the
   head's weights (all `param.requires_grad` is False).

b. Forward + backward through the rest of the policy doesn't error
   (gradient is still computed for non-frozen params).

c. Direction-related loss weights are forced to 0 when a manifest
   is loaded — no NaN, no surprise gradient flow into the head.

d. Loading a manifest with mismatched `active_runner_dim` raises
   ValueError with a clear "regenerate head" message.

e. Operator flag mutual-exclusion: `--enable-gene
   direction_prob_loss_weight` + `--direction-head-manifest` →
   ValueError at cohort startup.

## §7. The head's failure-tolerant fallback

If the operator launches a cohort WITHOUT `--direction-head-manifest`,
behaviour is byte-identical to today (per-agent head, trained via
BCE). This plan adds an OPT-IN code path, never a forced
migration. Existing experiments stay reproducible.

## §8. No env / obs-schema changes

The env's obs vector layout is unchanged. The 12 direction obs
columns continue to come from the
betfair-predictors direction model via the existing
`_compute_tick_predictor_outputs`. The shared head is purely
INSIDE the policy — it READS those columns the same way the
per-agent head did.
