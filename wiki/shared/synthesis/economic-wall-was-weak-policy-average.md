---
id: 01KTG8PXV4FJ5Z08RGSPWMRY4T
type: synthesis
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-0eda3f]
links: [{to: toll-to-edge-ratio-wall, type: supersedes}]
aliases: [the wall was wrong, weak-policy average, selection not economics, BC-to-PPO retraction]
---

# The "economic wall" was an average over the wrong trades

The BC→PPO campaign reached "this can't work" **three times**, each time on per-pair arithmetic showing
opening is negative-EV. The operator retracted all three (2026-05-31): people scalp this data profitably
daily, so positive-EV trades demonstrably EXIST — the "−EV" was a sloppy average over the *wrong* trades
the current weak policy opens. The real problem is **selection + learning**, not an economic ceiling.

## What it is

The retraction reframes the whole [[toll-to-edge-ratio-wall]] (which this note supersedes): the pwin /
commission / fill-rate analysis is **descriptive of the CURRENT weak policy, NOT a limit on what a
well-trained one can find**. The actual blocker is a TRAINING-STABILITY failure, not an economic wall —
PPO collapses to NOOP ([[noop-absorbing-state]]), an absorbing state, so the run dies before it can
learn which trades are the right ones. The pre-collapse episodes already posted positive days; the
signal was there.

The campaign then converted every analysis tool into a **gene/pin for a GA cohort** rather than a
single knob to tune: `per_pair_reward_at_resolution` ([[per-pair-credit-assignment]]),
`locked_pnl_reward_weight` (10× locked), `stop_loss_pnl_threshold`
([[stop-loss-fraction-of-stake]]), `reward_clip`, `entropy_coeff`, `mature_prob_open_threshold`,
`open_cost` ([[open-cost-knife-edge]]), `arb_spread_target_lock_pct`. Single-agent configs all land on
collapse or flood; the binding constraint is OPEN VOLUME / selectivity — a SEARCH problem (the GA's
job), not a single-knob fix.

## Why it matters

A repeated methodological trap: a sloppy average over a weak policy's choices masquerades as a strategy
verdict. The right frame is "open the RIGHT trades" — a selection problem the GA addresses, consistent
with the lesson that [[selection-vs-measurement-signal]] / GA selection is for picking survivors while
*reward shaping* is for what each agent learns. The corrected plan: stabilise PPO so it keeps trading
WHILE it learns (crank entropy, tame reward variance, run hundreds of episodes), then GA over the genes
to select the right-trade-opening agents. Bugs that were confounding the canary are catalogued in
[[bc-warmstart-coldstart-bugs]]; the refuted side-probe is [[pwin-not-direction-for-maturation]].

## Sources
- `src-0eda3f` findings.md (js_desktop:present)
