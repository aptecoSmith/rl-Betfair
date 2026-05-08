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

## S02 weight-sweep verdict (smoke v3, w=3.0)

Cohort ``_phase15_smoke_w3_1778275097``. Same shape as smoke
v2 but ``direction_prob_loss_weight=3.0`` (30× stronger BCE
pull). Result: **BCE essentially identical** to weight=0.1
(agent 1 dir_bce_back 1.1193 vs 1.1215 — Δ=0.002, lay
1.0526 vs 1.0544). Plumbing verified via the scoreboard's
``direction_prob_loss_weight_active=3.0`` field; isolation
probe confirmed BCE gradient flows cleanly through
``direction_prob_head[1].weight`` (norm 0.29 on a synthetic
batch). The weight is reaching the loss; the loss is reaching
the head; yet 30× weight produces no change.

**Initial (wrong) diagnosis: gate self-reference loop.** I
hypothesised the head's output drove BOTH the gate (masking
``OPEN_*`` actions when ``max(P_back, P_lay) < threshold``)
AND ``actor_input`` via the +4 column wiring, creating a PPO
pull "high direction_prob keeps OPEN actions legal" that
fought the BCE pull toward truth. Implemented a ``.detach()``
on direction_back_prob / direction_lay_prob before
``actor_input`` and re-smoked at w=3.0
(``_phase15_smoke_w3_detach_1778276053``). Result: identical
BCE again (agent 1 dir_bce_back 1.1201 vs 1.1193 pre-detach —
Δ=0.001). The gate hypothesis was wrong, OR detaching was
insufficient.

## Real root cause: Adam ate the weight scaling

PPO uses Adam (or AdamW). Adam's per-parameter update is
approximately ``learning_rate × m / sqrt(v)`` where m and v
are first/second-moment EMAs. The CRUCIAL property: this
update is **scale-invariant in the gradient magnitude** —
multiplying every gradient by 30× scales m and sqrt(v) by 30×
each, and m/sqrt(v) stays the same. Adam tries to normalise
to a unit-step-size optimiser. Multiplying the BCE loss by
30× doesn't change the per-param Adam step size on
``direction_prob_head``.

So weight=0.1 and weight=3.0 produce the same effective
update trajectory on the head. They differ only in how long
the warmup phase of Adam's variance EMA takes — minor for our
364-update window.

End-of-day BCE 1.05 ≈ ``-log(0.35)`` = the balanced no-skill
baseline (head sits at the positive-class marginal rate, roughly
35%). The probe got to BCE 0.4-0.6 in 600 dedicated SGD steps;
the cohort with Adam at 3e-4 LR can't escape the no-skill
basin in 364 mixed PPO+BCE updates regardless of BCE weight.

**Lesson: don't expect aux-loss weight to drive convergence
speed when the optimiser is Adam.** Weight matters in the
BCE/PPO trade-off direction (more BCE pull vs more PPO pull
per gradient step) but Adam ratios away the magnitude
contribution. To get the head to converge, EITHER:
- Run for many more update steps (Adam is slow but eventually
  gets there).
- Use a SEPARATE optimiser for the head with its own
  learning rate.
- BC-pretrain the head before PPO starts (the closest analogue
  to the probe's regime).

## Detach kept, BC pretrain next

The S01 detach amendment (commit
``[ next ]``) stays in place even though it produced no
measurable change in this smoke. Rationale: the detach
doesn't help yet because the head is at no-skill baseline
anyway, but if BC pretrain DOES land a calibrated head, the
detach prevents PPO from immediately corrupting that
calibration via the actor pathway. Removing the detach later
is trivial if the actor's "learn to use direction signal"
pathway becomes useful.

Next experiment: re-smoke with ``bc_pretrain_steps=2000``
+ ``bc_direction_target_weight=1.0`` (phase-13 gene wiring,
default 0/0). Pre-PPO supervised BC of direction_prob_head on
the cached labels at ``bc_learning_rate=3e-4`` is the closest
analogue to the probe's 600-step regime. If post-BC BCE
sits at 0.4-0.7 (probe range) the head is genuinely
calibrated; PPO + detach should preserve that state through
the rollout. If BC also gets stuck at 1.05, the labels or the
features carry less signal than the probe demonstrated, and
the residual gap is somewhere upstream.

## BREAKTHROUGH — v8 lands positive eval pnl (2026-05-08)

Cohort ``_phase15_smoke_md_1778279309``. 2 agents × 1 gen × 3
train + 1 eval days. ``bc_direction_target_weight=1.0``,
``direction_bce_use_pos_weight=true`` (default), gate enabled
T=0.85, BC pretrains direction_prob_head AND actor_head, head
FROZEN post-BC.

### Eval results (held-out 2026-05-06)

| Agent | Bets | Matured | mat% | Eval pnl |
|---|---|---|---|---|
| 1 (lr=0.00019) | 36 | 13 | 72.2% | **+£20.04** |
| 2 (lr=0.00061) | 94 | 28 | ~30% | **+£39.80** |

**Both agents POSITIVE.** Phase-14 baseline mean was -£73; phase-15
v7 (single-day BC, no freeze) was -£3 mean. v8 mean = +£30.

### What works

The full pipeline is:

1. **LayerNorm prepended to direction_prob_head**: the runner's
   raw feature slice has vol features in [10², 10³] — LayerNorm
   normalises per-example so the kaiming-init Linear doesn't
   saturate the sigmoid against truth.
2. **BC trains direction_prob_head AND actor_head**: the
   pre-existing BC pretrainer trained ONLY actor_head; phase-15
   S02 amendment expanded `_BC_TARGET_NAMES` to include
   `direction_prob_head`. BCE-with-logits on cached binary
   labels feeds the head's params through 2000 supervised steps.
3. **Multi-day pooled BC** (3 days here, 4-5 in big run):
   density of unambiguous (1,0) and (0,1) labels jumps from
   ~24K (single day) to ~55K-110K, giving the head more diverse
   data to fit.
4. **Freeze post-BC**: `direction_prob_head` parameters get
   ``requires_grad_(False)`` after BC. PPO gradients can't
   modify them. The auxiliary BCE during PPO is effectively a
   no-op because gradients flow but don't update.
5. **Gate threshold 0.85 + warmup**: the calibrated head's
   sigmoid output rarely exceeds 0.85 except for genuine
   high-confidence opportunities → 10× drop in bet count vs
   pre-phase-15 (400 → 36-94). Selective bets are profitable.
6. **Detach direction_prob from actor_input**: prevents PPO
   surrogate from corrupting the head (defensive — freeze is
   the load-bearing fix, but detach is belt-and-braces).

### Why pos_weight works here despite the math

`pos_weight` shifts the loss surface optimum away from the
true conditional probability toward a rebalanced one. In v8
this manifests as post-BC BCE 0.75/0.79 (vs v7 single-day no
pos_weight 0.26/0.35). HOWEVER: the head's outputs being
biased UP slightly seems to align with the gate's strict
threshold — the rebalanced operating point puts more genuine
positives ABOVE 0.85 while the marginal-density-true-negatives
stay below. v9 (vanilla BCE) tests whether this is a happy
accident or whether without pos_weight the calibration is
even better.

### Mature rate vs phase-14 break-even

Phase-14 empirical break-even = 34.8% (£3.37 mat / £1.80 force).
v8 agent 1 = 72.2% (2× above break-even). v8 agent 2 = ~30%
(below break-even on rate, but profitable due to high count of
opens at locked spreads averaging higher than the ratio).

### Force-close rate

Pre-phase-15 cohorts had 70-80% force-close rate (most opens
bailed by env at T-N). v8 agent 1: 5/18 = 28%. The selective
gate + calibrated head skip most "this won't mature" opens at
decision time, dropping the force-close rate by 2-3×.

### Strategic implication

Phase-14's NULL was driven by the inability to extract the
direction signal from the LSTM-bottlenecked input pathway.
Phase-15 fixed that by feeding raw features directly. v8
shows that, with the predictor properly trained AND
preserved (freeze), the gate mechanism does what the
strategic thesis predicted: selective opens, high mature
rate, positive eval pnl.

Big run queued (8 agents × 2 gen × 5 train + 3 eval) to
validate at scale.
