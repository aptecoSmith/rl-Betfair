---
id: 01KTG8PXV0VBRF41EB210MZVQ9
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-0eda3f]
aliases: [NOOP collapse, absorbing state, PPO stops trading, no-opens death spiral]
---

# NOOP is an absorbing state (PPO collapse)

Across every BC→PPO canary config, PPO eventually collapses to opening **zero** trades — and the danger
is that NOOP is an **absorbing state**: no opens → no reward signal on opens → the policy can't climb
back out, so the run dies before it can learn which trades are the right ones.

## What it is

The collapse is config-robust: `open_cost` drives it on a knife-edge ([[open-cost-knife-edge]]); the
clean-reward + stop-loss synthesis survived ep0–3 (opened 123–189, days +£14/+£30) but collapsed by ep5
(opened 0, stayed 0 through ep17) with high early KL (0.25–0.30); 10× locked floods then collapses by
ep8. The pre-collapse episodes already posted positive days — the signal was right there before the
policy died. Diagnosed root causes are training-stability, not strategy: `entropy_coeff=0.01` too weak
to hold the policy off deterministic NOOP, no reward-clip (the −£196 variance episodes cause violent
lurches), and far too few episodes (18).

## Why it matters

Reframes "PPO learns not to trade" from an economic verdict into a **stability failure**
([[economic-wall-was-weak-policy-average]]): an absorbing state means the run can't recover to test
better trades, so you never observe the strategy's ceiling. The fixes are the standard PPO-stability
levers — crank entropy to stay stochastic, clip reward variance, run hundreds of episodes — the same
family as the [[ppo-starved-by-kl-early-stop]] and entropy-control work. Likely compounded by the
[[per-pair-credit-assignment]] miss (the agent can't see which opens were mistakes).

## Sources
- `src-0eda3f` findings.md (js_desktop:present)
