---
plan: rewrite/phase-7-port-aux-heads
opened: 2026-05-04
---

# Phase 7 — lessons learnt

## v1's `hp.get(name, config_fallback)` is unsafe in v2

**Rule.** The v2 `DiscretePPOTrainer` reads
`fill_prob_loss_weight`, `mature_prob_loss_weight`, and
`risk_loss_weight` from the per-agent `hp` dict ONLY. It does NOT
fall back to `config["reward"][...]`. The worker pre-merges any
cohort-level `--reward-overrides <key>=<value>` into the hp dict
BEFORE constructing the trainer.

**Why.** v1's pattern is

```python
self.mature_prob_loss_weight = float(
    hp.get(
        "mature_prob_loss_weight",
        config.get("reward", {}).get("mature_prob_loss_weight", 0.0),
    )
    or 0.0
)
```

This works in v1 because v1's `hp` dict is **sparse**: it only
carries keys for genes the population is actively varying. When
the operator pins a knob cohort-wide via `--reward-overrides`, the
key is ABSENT from `hp`, and the fallback is consulted. Precedence
behaves as intended.

v2's `hp` dict comes from `CohortGenes.to_dict()` — a dataclass
that ALWAYS populates every gene field with its default value.
`hp["mature_prob_loss_weight"]` is `0.0` for an unenabled gene; the
fallback is never consulted; the override is silently swallowed.

This is exactly what produced the byte-identical eval results in
the 2026-05-04 cohort
`v2_phase5_oc1_mpw05_clean5day_1777849498` vs the prior cohort —
the operator pinned `mature_prob_loss_weight=0.5` but the trainer
read `0.0` and the rollouts were identical to a run with the
override absent.

**How to apply.** When porting any v1 trainer-side hp read into
v2, audit whether `hp.get(name, fallback)` would silently swallow
an override. If yes, the worker must pre-merge the override into
`hp` before construction (Path A). If no fallback was ever needed
in the first place, drop it. NEVER copy the v1 pattern verbatim
without thinking through the v2-specific dict semantics.

## Plumbing decision: Path A wins

The two paths to make `--reward-overrides
mature_prob_loss_weight=0.5` reach the trainer:

- **Path A (chosen).** Worker pre-merges `reward_overrides` into
  the per-agent `hp` dict before constructing the trainer. The
  trainer's `hp.get(name, 0.0)` then returns the override value
  because the key carries it. Single source of truth. No new
  trainer constructor surface beyond the existing `hp` dict.
- **Path B (rejected).** Worker passes `reward_overrides` as a
  new kwarg to `DiscretePPOTrainer`; trainer reads via the v1
  precedence pattern `hp.get(name, reward_overrides.get(name,
  0.0))`. Rejected because: (i) two sources of truth inside the
  trainer, (ii) preserves the v1 precedence pattern that has the
  silent-swallow failure mode, (iii) duplicates the worker's
  existing reward-overrides-merge logic across the trainer
  boundary.

Path A is implemented in `training_v2/cohort/worker.py
::_build_trainer_hp`. The frozenset `_PHASE7_TRAINER_HP_KEYS`
documents which keys flow through this path (vs the env-side
`_PHASE5_GENES_VIA_REWARD_OVERRIDES` for env-consumed genes).

## Per-runner aux labels are aggregated across races

The S02 implementation aggregates per-pair labels into per-slot
labels at end-of-rollout, then broadcasts the same per-slot label
to every transition's mini-batch entry. This is a deliberate
simplification vs v1's per-transition pair_to_transition map.

The cost: same slot index can carry different physical runners
across races, so the per-slot label is a noisy aggregate. The
benefit: no per-transition aux_targets array on the rollout
buffer, no per-tick BetManager snapshot, no end-of-rollout
backfill pass. The Phase 7 success bar is "lever is alive", not
"lever is pixel-perfect"; per-transition credit can be
tightened in a follow-on session if the validation cohort shows
the lever signal is too weak.

## Risk NLL must skip when no completed pair in mini-batch

Naked-only rollouts (no pair completed both legs anywhere) have
`risk_mask.sum() == 0`. Computing the NLL term as
`(masked_nll).sum() / 0` would return NaN and propagate into
total_loss. The trainer's `_compute_aux_losses` guards with
`if risk_denom > 0` and returns a fresh
`torch.zeros((), device=...)` otherwise — keeping the loss
expression NaN-free without conditionalising the caller.
