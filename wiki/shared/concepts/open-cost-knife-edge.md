---
id: 01KTG8PXV1VENDZF7DEFCWA98F
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-0eda3f]
aliases: [open_cost knife-edge, open_cost collapse, selectivity knob]
---

# `open_cost` is a knife-edge selectivity knob

`open_cost` charges a flat per-open toll in the *training* reward to push the policy toward selectivity.
The BC→PPO canary found it has almost no usable range: above a tiny break-even value it collapses the
policy to zero opens; below it, it provides no selectivity pressure at all.

## What it is

With `open_cost=0.1`, PPO went ep0 opened 278 → ep1 opened **1** → ep2 opened **0**: the charge on the
~87% non-matured opens (−£24) swamps the +£16.4 locked, so opening is net-negative in the training
reward and PPO correctly learns "stop opening." The break-even charge is
`locked_per_matured × mat_rate / (1−mat_rate)` ≈ £0.44 × 0.133 / 0.867 ≈ **0.067**. Any `open_cost`
above that collapses the policy; any below gives ~no selectivity pressure — a knife-edge.

## Why it matters

`open_cost` looked like the lever to make PPO open *fewer, better* trades, but its viable window is a
razor's width set by the [[toll-to-edge-ratio-wall]]. This is one of the two collapse mechanisms behind
the canary: the `open_cost` knife-edge here, and the [[noop-absorbing-state]] that makes any collapse
permanent. The eventual move was to demote `open_cost` to one gene among many in a GA search rather than
a single knob to tune ([[economic-wall-was-weak-policy-average]]).

## Sources
- `src-0eda3f` findings.md (js_desktop:present)
