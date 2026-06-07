---
id: 01KTFQ3AN4A6S73JGE3KKYZHJN
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-032073]
aliases: [degenerate genetic search, gen-0 degeneracy]
---

# Degenerate GA search (gen-0)

A design-review finding: the genetic population search was **degenerate** — most of its declared search
dimensions weren't actually varying, so three generations of evolution contained little signal.

## What it is

The specific degeneracies: reward-shaping genes were dead code (see [[dead-reward-shaping-genes]]);
only 3 PPO knobs varied (`learning_rate`, `ppo_clip_epsilon`, `entropy_coefficient`) while
`gamma`/`gae_lambda`/`value_loss_coeff` were hardcoded; LSTM structural params were hardcoded
(1 layer, no dropout/layernorm); `observation_window_ticks` was sampled but never read (a dead gene);
there was no third architecture and no planning layer to ensure fair gen-0 coverage. Net: gen-0 gave
agents near-identical starting points, so selection had almost nothing to act on.

## Why it matters

The motivating diagnosis for plumbing every gene end-to-end and planning gen-0 deliberately (even
architecture mix + reward/PPO ranges, with a record of what's been tried). A GA only searches the
dimensions that are actually wired and actually vary.

## Sources
- `src-032073` purpose.md (js_desktop:present)
