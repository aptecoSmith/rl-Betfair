---
id: 01KTG8PXV1JX49WDDFA2K4SS0S
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-0eda3f]
aliases: [per_pair_reward_at_resolution, credit-assignment miss, settle-lumping, resolution-tick credit]
---

# Per-pair credit assignment (`per_pair_reward_at_resolution`)

The credit-assignment miss the operator named: "how does the model know which trades were good? It can't
learn to make better ones if we can't show it the mistakes." By default the env **lumps all cash P&L
onto the final settle tick**, so PPO sees one number per race and cannot attribute a loss to the
specific open that caused it.

## What it is

`per_pair_reward_at_resolution` defaults OFF. With it off, the open decision is thousands of ticks before
settle and the credit has to survive GAE across that whole gap — it doesn't. PPO sees one number per
race and CANNOT attribute a loss to the specific open that caused it, so it's asked to "open better
trades" while blind to which of its ~200 opens were the bad ones.

Setting `per_pair_reward_at_resolution=True` credits each pair's **realised P&L — including force-close
LOSSES (the mistakes), as a negative — at the tick the pair RESOLVES**, per-pair (env line 5087→3516),
not lumped at settle. With the trainer's per-runner reward streams this attributes "this open → this
outcome" far more directly, and it's a `reward_override` so it carries into the cohort. A possible
further enhancement is to back-attribute to the exact OPEN tick for the cleanest signal.

## Why it matters

Plausibly explains the canary thrash/collapse as much as variance does: an absorbing NOOP collapse
([[noop-absorbing-state]]) is partly *caused* by the agent never seeing which opens were the mistakes.
This is the densification cousin of mark-to-market shaping — both fight the same settle-time credit
smear that GAE can't carry. One of the genes/pins the campaign handed to the GA cohort
([[economic-wall-was-weak-policy-average]]).

## Sources
- `src-0eda3f` findings.md (js_desktop:present)
