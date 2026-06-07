---
id: 01KTGP9R7HDWMGHM0N6KM5TM8G
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-1b344c]
aliases: [continuous head inits mid-range, arb_frac 0.5, agents herd near default, stronger signal breaks the herd]
---

# A continuous action head initialises at its range centre

An exploration observation: a continuous (Beta/Gaussian) action head starts near the **middle of its
range**, so before it gets a strong gradient the whole population clusters at the default value regardless
of per-agent multipliers.

## What it is

With `arb_raw` defaulting to N(0, σ), `arb_frac` starts around 0.5 — i.e. 8 ticks (mid-range). "The agents
explored a little but mostly stayed near 8," which is why fill rates cluster similarly across agents even
though `arb_spread_scale` varies 0.5–1.5. The conclusion: "stronger signal to that head should break the
herd" — without a sharp gradient, a continuous head's prior dominates and per-agent genetic variation
barely moves the effective behaviour.

## Why it matters

Explains why a continuous knob can look inert across a cohort even when it's genuinely wired: the head sits
at its init centre and the distal reward can't pull it off. The remedy is the same as
[[weak-gradient-needs-aux-supervision]] — a direct supervised signal breaks the herd. Also a caution when
reading cohort variation: clustered behaviour at a head's mid-range may be init inertia, not a discovered
optimum (cf. [[gen0-pnl-is-meaningless]]).

## Sources
- `src-1b344c` lessons_learnt.md (js_desktop:present)
