---
plan: rewrite/phase-15-direction-head-feature-slice
parent_purpose: ./purpose.md
---

# Lessons learnt

Append-only journal. Entries land as work completes.

## Inherited lessons (read before any session)

- `plans/rewrite/phase-14-direction-gate/lessons_learnt.md` —
  the full phase-14 journey. The "Architectural lesson —
  per-runner head, NOT shared output" entry is the direct
  precursor to phase 15: phase 14 fixed the head's *output*
  side; phase 15 fixes the *input* side.

- `plans/rewrite/phase-13-directional-scalping/lessons_learnt.md`
  — the phase-13 NULL writeup. The lesson "probe before cohort"
  is doubly load-bearing here: the supervised probe was the
  diagnostic that motivated phase 15.

## Quantitative ground truth (carried over from phase 14)

The phase-14 ground truth applies unchanged. See
`plans/rewrite/phase-14-direction-gate/lessons_learnt.md`
"Quantitative ground truth" section. Key numbers:

- Per-pair P&L: matured +£3.37, force-closed -£1.80,
  break-even mature rate 34.8%.
- OOS top-quintile lift on per-runner-slice probe with 10
  augmented features: 24-94×.
- 3 of 3 OOS days profitable at empirical cost ratio with
  T ∈ [0.90, 0.95].

The 24-94× number is the load-bearing input-pathway evidence.
If phase 15 reproduces even half of that lift inside the
cohort, the success bar (mature rate ≥35%) is reachable.

## Methodological lesson — input pathway matters more than head capacity

Phase 13 fixed the labels (added direction labels). NULL.
Phase 14 fixed the head's output structure (single Linear →
per-runner MLP). Probe lift went up; cohort BCE may still be
flat (TBD when probeAB lands).

The phase-14 sense_check (item 3) called this exact risk
pre-cohort: "the per-runner MLP fed by the SAME shared
`lstm_last` may have the same issue [as phase 13]." Phase 15
exists because that risk was foreseen and pre-staged.

**Lesson:** when a bottleneck is suspected to be an input
pathway, fix the input pathway directly. Don't increase head
capacity, don't increase training time, don't sweep
hyperparameters. The probe already isolated the variable.

## Methodological lesson — single-knob plans

Phase 14 changed three things at once (head architecture,
features, gate). Each was probe-validated independently, but
the cohort outcome can't cleanly attribute residual lift to
any one of them. Phase 15 changes ONE thing: which input the
direction head reads. If S03 delivers, the input pathway is
the load-bearing fix. If S03 doesn't, the residual is a
known-bounded gap to chase in one more plan.

## Saturation from raw obs scales — LayerNorm was needed (S02, 2026-05-08)

The first S02 smoke (cohort
``_phase15_smoke_1778273750``, 2 agents × 1 train + 1 eval
day, gate on T=0.85, direction_prob_loss_weight=0.1) returned
**direction BCE 4-12** — much WORSE than phase-14's 1.04
baseline. The sigmoid saturated against truth: with BCE 12.4,
``p ≈ exp(-12.4) ≈ 4e-6`` — confidently wrong predictions
across the board.

**Root cause:** the head reads the runner's RAW
``RUNNER_KEYS`` slice (~125 dims) directly out of obs. Other
v2 heads (fill / mature / risk / value) read ``lstm_last``,
which is post-``input_proj``'s learned linear scaling, so they
never see the raw heavy-tail features. The direction head
post-S01 saw raw values where ``vol_delta_60`` sits in
[10², 10³] — kaiming-init weights × those magnitudes pushed
pre-activations into the thousands and the sigmoid saturated.

**Fix:** prepend ``nn.LayerNorm(RUNNER_DIM)`` to
``direction_prob_head``. LayerNorm normalises each example to
zero mean / unit std across the feature dim — the same squash
the supervised probe achieved with per-day ``pd.std`` but
without dataset-stats bookkeeping. Sense_check.md item 2 had
flagged feature-scale mismatch as a documented risk; LayerNorm
is the cleanest mitigation.

**Re-smoke (cohort ``_phase15_smoke_1778274494``):** BCE drops
to [1.05, 1.12] — back to the phase-14 baseline, no longer
saturated. KL healthy (0.008-0.014), full PPO budget runs
(n_updates=364), agent 1 emits 375 bets and posts the first
positive eval_day_pnl (+£24.40) seen in this sequence. The
input pathway is now sound; BCE is just back to "head not
clearly learning yet" rather than "head trained against
truth."

**Lesson for any future per-runner head fed raw obs:** LayerNorm
or some equivalent input normalisation is mandatory. Don't rely
on the optimiser to eat the scale mismatch — kaiming-init plus
[10², 10³]-scale inputs saturates the first layer's outputs
faster than gradients can recover.

**Test maintenance:** LayerNorm normalises out uniform shifts,
so test perturbations like ``obs[:, slice] += 1.5`` or
``weight.add_(0.5)`` become INVISIBLE post-normalisation. The
phase-15 regression tests now use multiplicative perturbations
(``weight.mul_(2.0)``) and full-replacement perturbations
(``obs[:, slice] = torch.randn(...)``) so the per-runner-
consumption and gradient-through guards survive LayerNorm.

## BCE flat at baseline after S02 amendment — escalating to weight sweep

Even with LayerNorm and the input-pathway fix, the head's
end-of-day BCE sits at the phase-14 1.04 baseline rather than
the probe's 0.4-0.6 range. With
``direction_prob_loss_weight=0.1`` the cohort sees 0.1 ×
364 PPO updates ≈ 36 effective BCE updates of strength 0.1
— ~17× weaker than the probe's 600 dedicated supervised steps.

Two competing hypotheses for why BCE doesn't drop:
1. **Gradient strength insufficient at weight 0.1.** Bumping
   to 3.0 should give ~10× more effective BCE pull and
   conclusively test this.
2. **PPO surrogate fights BCE.** ``direction_prob_head`` feeds
   ``actor_head`` so policy-loss gradient flows back through
   it; if policy_loss gradient dominates the BCE pull, the
   head drifts to whatever helps actor selection rather than
   to calibrated direction probabilities.

S02 follow-on: re-smoke at ``direction_prob_loss_weight=3.0``.
If BCE drops cleanly → S03 cohort's gene range needs to be
biased toward higher weights. If BCE stays flat → escalate to
investigating PPO/BCE gradient interplay (a separate plan).
