---
plan: rewrite/phase-15-direction-head-feature-slice
parent_purpose: ./purpose.md
---

# Hard constraints

## §1 — Architecture-hash break protocol

Same protocol as phase-7 (fill-prob), phase-9 (per-transition
credit), phase-13 (direction head), phase-14 (per-runner
direction MLP). Pre-S01 checkpoints fail strict-load by design;
the variant identity is carried by the existing weight-shape
check in `registry/model_store.py`. No new explicit version
field. Operators expect this; they re-init the cohort.

## §2 — `direction_prob_head` reads ONLY the per-runner slice

```python
direction_input = runner_feats_raw  # (batch, R, RUNNER_DIM)
```

NOT:
```python
direction_input = torch.cat(
    [runner_feats_raw, lstm_expanded], dim=-1,  # FORBIDDEN in S01
)
```

The whole point of phase 15 is to bypass the LSTM-compression
bottleneck. Concatenating `lstm_last` back in defeats it. The
post-S03-validation ablation (open question 1 in purpose.md)
can test the concat variant in a *separate* plan, after the
pure variant ships.

## §3 — Gradient-through invariant

`direction_prob_head` output continues to feed `actor_head`
(the +4 column wiring). The surrogate loss flows back through
the head. Do NOT detach. The head's input is now constant
per-step (raw obs slice has no learnable params) so the
gradient terminates there — that's correct and matches the
probe's regime.

## §4 — Aux BCE loss path unchanged

`DiscretePPOTrainer._compute_aux_losses` reads
`out.direction_back_logits` / `out.direction_lay_logits` and
applies BCE against cached labels. That code path is untouched.
`direction_prob_loss_weight` gene unchanged.

## §5 — Reward magnitudes UNCHANGED

The change is purely on the actor-input pathway and the head's
input pathway. `race_pnl`, `scalping_locked_pnl`,
`scalping_closed_pnl`, `naked_pnl`, all shaped terms — none of
them touch `direction_prob_head`'s input. The
`raw + shaped ≈ total_reward` invariant holds.

Scoreboard rows from phase-14 (and earlier) cohorts remain
comparable to phase-15 rows on `raw_pnl_reward`. They are
NOT comparable on `direction_back_bce_mean` /
`direction_lay_bce_mean` if the head genuinely learns now —
phase-14's flat ~1.04 BCE was the failure mode, not a
calibrated baseline.

## §6 — Regression tests

Inherit phase-14's test layout (`tests/...`/test_direction_*):

- **Updated:** `test_direction_head_input_shape` →
  assert `direction_prob_head[0].weight.shape ==
  (actor_mlp_hidden, RUNNER_DIM)`. Was `(actor_mlp_hidden,
  runner_embed + hidden)` pre-phase-15.
- **New:** `test_direction_head_consumes_runner_feature_slice`
  — perturb `obs[runner_block]` for runner i; assert
  `out.direction_back_logits[:, i]` changes, while perturbing
  `obs[runner_block]` for runner j leaves runner i's logit
  unchanged. Catches accidental cross-runner mixing.
- **New:** `test_direction_head_does_not_depend_on_lstm_last`
  — perturb the LSTM's hidden state; assert
  `out.direction_back_logits` is unchanged. The strict
  bottleneck-bypass guard.
- **Updated:** `test_pre_phase15_weights_fail_to_load` —
  pre-phase-15 state_dict raises on
  `load_state_dict(strict=True)` because
  `direction_prob_head[0].weight` is one shape; phase-15 is
  another.
- **Inherited unchanged:** all gate-mask-capture tests
  (phase-14 S05) and threshold-warmup tests (phase-14 S06).
  The gate path doesn't care WHERE the direction probs came
  from.

## §7 — Slot embedding stays for the actor

`runner_slot_embedding` continues to feed `actor_head` per
slot. The actor still benefits from a learned tag that lets
the per-runner MLP discriminate slot identity above and beyond
the per-runner features. Do NOT remove it from the actor in
this plan.

## §8 — Default OFF semantics

`direction_prob_loss_weight = 0.0` agents still get a working
`direction_prob_head`; the head's outputs feed `actor_head`
unchanged regardless of whether BCE is training it. With
weight 0 the head's column drift is benign (initialised near
sigmoid(0) = 0.5, no supervised gradient pulls it apart).
Same semantics as phase-14.

## §9 — Cache compatibility

OBS_SCHEMA_VERSION is **NOT** bumped. Phase 15 doesn't change
the obs layout — it only changes which *part* of obs the head
reads. Existing oracle and direction-label caches remain
valid. Re-scan NOT required.

## §10 — Probe-cohort consistency

The supervised probe
(`tools/direction_features_probe.py`) ALREADY reads the raw
per-runner slice (~125 dims with phase-14 S02's 10 augmented
features). Phase-15's S01 makes the cohort head's input match
the probe's input. After S01, **the only remaining differences
between the probe and the cohort are**:

- Optimiser (PPO + Adam vs probe's pure Adam on BCE).
- Training-data volume (cohort sees much more).
- Aux-loss-vs-policy-gradient interplay.

If S03 doesn't deliver after this alignment, the residual gap
is not an input-pathway bug — it's a PPO/cohort-scale gap that
needs a different diagnostic.

## §11 — Single-knob plan

Phase 15 changes ONE thing. No simultaneous tweaks to
`actor_mlp_hidden`, `runner_embed_dim`, gate range, label
spec, or feature set. If S03 fails, the residual contributes
to a clean signal: the input pathway alone is not sufficient,
and the next plan can sweep ONE more knob.
