# Shared frozen direction head

**Opened:** 2026-05-24 (evening)
**Status:** scaffolded, executing
**Predecessor:** `plans/direction-predictor-label-alignment/`

## Why this plan exists

Today the v2 policy carries a **per-agent** `direction_prob_head`
that's trained inside each cohort agent via BCE supervised loss. 12
agents in a cohort each train their OWN head independently against
the SAME labels from the SAME data. This is structurally wasteful
and produces three documented failure modes:

1. **Slow learning.** Per-agent BCE descends only ~5 % relative
   across 16 days of training. Even at gene loss-weight 1.6 the
   head barely moves below the random-uniform-0.5 floor.

2. **PPO interference.** The supervised BCE gradient flowing into
   the policy's parameters fights the policy/value gradient through
   the SHARED parameters (input_proj projects the same obs the head
   reads, etc.). Each gradient distorts what the other is trying
   to learn.

3. **Inconsistent across agents.** When the GA breeds, two agents
   with different head weights produce comparing-apples-to-oranges
   data — outcome differences attributed to GA may actually be
   head-noise differences.

The pretrained `betfair-predictors` direction model
(`conv1d_k3_s1_d12bfac2b132`) is the precedent for the correct
pattern: **train ONE specialist offline on lots of data, freeze it,
let all agents read its output identically.** The direction head
should follow the same pattern.

## What the head is, after this plan

A small offline-trained NN that lives at:

```
models/direction_head/<exp_id>/
    weights.pt
    manifest.json
```

Mirroring the betfair-predictors layout. Trained on
`(per_runner_obs_slice, direction_label)` pairs from the training
days' caches. Loaded by the cohort policy at construction time,
weights frozen via `requires_grad_(False)`. Every agent in every
cohort sees the SAME calibrated direction signal at the actor_head
column-concat.

## Why we expect this to work better

Per the 2026-05-24 logistic-regression probe on the same input
shape:

| input                                            | val BCE | descent (rel) |
|---|---|---|
| Per-runner obs (23-d), v1 labels                  | 1.06   | +5.7%  back / +12.3% lay |
| Full obs (574-d), v1 labels                       | 0.95   | +16.0% back / +19.7% lay |
| Per-agent head after 16 days PPO (Phase-15 buggy) | 1.14   | ~0%    (was the bug) |
| Per-agent head after 16 days PPO (post-runner-dim-fix, projected) | ~1.10  | ~3-5% (in-progress estimate) |

A single head trained well offline should land at the
**per-runner obs (23-d) logreg level — ~6-12 % descent.** Maybe a
bit better with a slightly deeper architecture, since logreg is
the floor of what's achievable.

That's 2-4× the descent the per-agent head currently produces,
delivered on day 1 of every cohort instead of after 16 days.

## What we lose

* **Per-agent customisation of direction supervision.** The two
  genes `direction_prob_loss_weight` and `bc_direction_target_weight`
  become inert (head is frozen, no loss applies). Optionally
  replace with `direction_head_temperature ∈ [0.1, 2.0]` so each
  agent has a single scalar control over "how confidently to
  listen to the direction head." That preserves some per-agent
  variation without compromising the shared training.

* **Adaptive direction representations.** If the optimal direction
  signal evolves with the actor's policy (it probably doesn't), a
  frozen head can't track that. We accept this — the predictor it
  reads is already calibrated to a fixed forecast objective.

## Hold-out invariants (critical)

The shared head MUST never see eval or monitor day data during its
training. Same as the agents:

* **Train head on:** the 16 cohort-training days
  (2026-04-06, 04-08, 04-09, 04-11, 04-12, 04-13, 04-15, 04-16,
  04-19, 04-20, 04-22, 04-24, 04-26, 05-02, 05-04, 05-05)
* **Never on:** the 10 eval days (2026-04-07, 04-10, 04-14, 04-17,
  04-21, 04-23, 04-25, 05-01, 05-03, 05-06) or the 14 monitor days
  (2026-05-07 ... 2026-05-20)

The manifest records the day list so any future eval-day shuffle
forces a head re-train.

## Success criterion

After landing:

* `models/direction_head/<exp_id>/manifest.json` exists with
  `val_bce_back ≤ 1.05` AND `val_bce_lay ≤ 1.05` on a held-out
  20 % split of the TRAINING days (NOT eval days — head must not
  see eval).

* Cohort probe (2-3 agents × 1 gen × 5 days) launches without
  crashing. Direction obs columns + head output column are both
  populated. Direction-related genes (`direction_prob_loss_weight`,
  `bc_direction_target_weight`) are recorded as `dropped/frozen`
  in scoreboard rows.

* Across the probe, the actor's per-runner `direction_back_prob`
  output (from the frozen head) shows non-trivial variation
  (std ≥ 0.05) and modest correlation with v1 labels
  (|rho| ≥ 0.10 on a held-out probe day).

If any of these fail, this plan didn't address the right thing and
we revisit.

## Out of scope

* Sharing `fill_prob_head`, `mature_prob_head`, `risk_head`. Their
  labels DO depend on the agent's actions (which pair was opened,
  how it filled, etc.), so sharing those is a harder design
  question. Empirically they DO seem to learn at cohort scale
  (fill_bce hit 0.003, mature_bce hit 0.110 on Phase-15), so the
  case for sharing them is weaker.

* Re-architecting the actor_head. The actor still reads the same
  column-concat `[runner_emb, lstm_last, fill_prob, mature_prob,
  direction_back_prob, direction_lay_prob]`. Only the SOURCE of
  the last two columns changes from per-agent to shared.

* Promoting the head's hyperparameters to genes (architecture,
  training schedule). This is a fixed-architecture experiment;
  bigger search is a follow-on if the first try doesn't hit the
  success criterion.
