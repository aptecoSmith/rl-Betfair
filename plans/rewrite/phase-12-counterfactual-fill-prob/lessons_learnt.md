---
plan: rewrite/phase-12-counterfactual-fill-prob
---

# Lessons learnt

_(populated as sessions ship.)_

## Context inherited from Phase 8 / 9 findings (2026-05-06)

The premise for this plan — that natural fill rate is structurally
ceiled at 0.17 – 0.21 across every cohort tested — comes from
[phase-8-oracle-bc-pretrain/findings.md](../phase-8-oracle-bc-pretrain/findings.md).
Five distinct cohorts (Phase 7 S06, S03 × 3 arms, both overnight
arms) all converged to that range despite different mechanism
combinations (per-transition credit on/off, BC on/off, multi-eval
on/off). The conclusion that this is an information problem rather
than a training-signal problem comes from observing that:

1. The fill_prob_head's BCE label was structurally broken — it
   conflated force-closes with maturations, so a well-trained
   head steered the actor TOWARD high-volume opening (cohort F's
   `ρ(fill_prob_loss_weight, fc_rate) = +0.469` — discovered in
   [per-runner-credit/findings.md](../../per-runner-credit/findings.md)).

2. Phase 7's mature_prob_head with the strict label fixed (1) but
   was trained on agent-rollout experience only — it could only
   label what the agent did, not the counterfactual.

3. Phase 9's per-transition credit delivered the gradient at the
   right transition (`n_mature_targets > 0` confirmed) but ρ
   stayed noisy across all cohorts. Confirms the gradient was
   reaching the actor; the actor's *input* didn't carry the
   information needed to discriminate fillable from non-fillable
   opens.

(3) is the key observation that motivates this plan. The actor
knows what it sees (price ladder, time, traded volume) but doesn't
know what it didn't see (the counterfactual fills it could have
produced by opening on a different tick or runner).
