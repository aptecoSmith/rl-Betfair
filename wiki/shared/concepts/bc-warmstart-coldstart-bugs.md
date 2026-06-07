---
id: 01KTG8PXTY1788NZQDFFHDEVHS
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-0eda3f]
aliases: [stake-head cold-start, fixed-stake-unit, device-mismatch eval bug, BC warm-start bugs]
---

# BC warm-start cold-start bugs (device + stake head)

Two cold-start bugs surfaced wiring the BC warm-start into the BC→PPO canary — both about *untrained*
parts of a partially-warm policy behaving badly on the first rollout.

## What it is

1. **Device mismatch.** The trainer parks the policy on `rollout_device` (cpu) after updates; eval must
   read the policy's own device rather than assume one.
2. **Stake-head cold-start.** The env decodes `stake = stake_unit × budget`, and the warm-start's
   UNTRAINED Beta stake head sits at `stake_unit ≈ 0.5` → £50–100 stakes, whose passive-lay liability
   is too big to post → everything goes naked → force-closes → mat% 0. Fixed with `--fixed-stake-unit
   0.1` (pins + freezes the stake head to £10/open at £100 budget, isolating the OPEN-selectivity
   question).

With the stake pinned, the warm-start reproduces the BC result (canary v2, ep0): **opened 278, mat%
13.3%, locked +£16.4** at £10/open — i.e. the BC warm-start opens selectively and locks positive, as
bc-getting-it-right found.

## Why it matters

Isolating the stake head was what let the canary ask the *open-selectivity* question cleanly — without
the pin, the £50–100 cold-start stakes guarantee naked→force-close→mat% 0 and confound every reward
experiment. The reproduced +£16.4 locked is the very number the [[toll-to-edge-ratio-wall]] arithmetic
is built on. A reminder that a partially-frozen warm-start can have a live head sitting at a degenerate
prior.

## Sources
- `src-0eda3f` findings.md (js_desktop:present)
