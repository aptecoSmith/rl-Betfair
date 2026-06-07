---
id: 01KTGC8208PXVMC7JXNWV4JZGC
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-106b41]
aliases: [non-binding knob, gate never fired, inert probe, cap set too high, verify the gate bound]
---

# A non-binding knob reads as a null result

A probe-methodology trap: a gate/cap set too lenient **never fires**, so the probe returns "no effect" —
which looks like "the mechanism doesn't work" but actually means "the mechanism was never tested." Two
probes in the campaign hit this.

## What it is

- **E6 (pair-count budget cap=30)** — NO BITE, but the cap never bound: baseline ~178 bets/day ÷ ~3
  races ÷ 2 legs ≈ 30 pairs/race — at the cap, not above it. "This probe tells us nothing about whether
  forcing selectivity via a scarcity gate works"; a real test needs cap=5.
- **R4 (opposite-side depth floor £10)** — IDENTICAL to E3 alone agent-by-agent; R4 at £10 is inert, the
  opposite-side top-level size routinely exceeds it. The £30 variant then over-clipped (locked floor
  crashed £36/d). The bandwidth where it helps is narrow.

The tell is that a refusal/activity counter reads ~0 when the knob should be active — the same diagnostic
that catches the Path-A launch-wiring foot-gun.

## Why it matters

Before concluding a mechanism failed, confirm its gate actually **bound** (check the refusal counter / the
implied vs natural rate). This is the necessary companion to the [[probe-before-cohort-budget]] discipline:
a probe's "no signal" is only informative if the lever was in its active range. The inverse error — a knob
silently dropped by a forked code path — is [[batched-path-silent-drops]].

## Sources
- `src-106b41` EXPERIMENTS.md (js_desktop:present)
