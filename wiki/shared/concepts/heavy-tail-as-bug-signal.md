---
id: 01KTFBWWTM3XVF8M4SQBTY42HW
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-0604e6]
links: [{to: value-domain-feature-assertions, type: see-also}]
aliases: [90-sigma is a bug, heavy-tail bug signal, suspect clip not pattern]
---

# A lone ~90σ z-score is more likely a bug than a fat tail

When a single dimension in a z-scored feature set shows max ≈ +90, the
prior should be **feature-construction bug**, not "financial data is
heavy-tailed, clip it later." Investigate first; clipping would *hide* the
bug.

## What it is

In the trajectory-retrieval probe, `delta_vol_short_z` max came back at
+87.7. The instinct was to flag a heavy tail for later clipping. The real
diagnosis: the feature was reading the wrong pre-decision tick (see
[[value-domain-feature-assertions]]). After fixing the read, the max
dropped to +3.92 — a normal heavy-but-not-pathological tail.

## Why it matters

The naive "clip and normalise" response hides feature-construction bugs
behind a plausible-looking normalised distribution. The corrected rule:
when a z-scored feature exhibits a single value in the high tens of σ,
investigate the **construction** before deciding it's a real distribution
to flatten. Clipping after-the-fact is fine for *real* heavy tails; it is
the wrong tool for catching bugs.

## Sources
- `src-0604e6` lessons_learnt.md (js_desktop:present)
