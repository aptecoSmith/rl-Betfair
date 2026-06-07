---
id: 01KTGNYHY8Z7RD060WTB69NDF1
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-19293f]
aliases: [re-test dropped levers, lever helps on strong base, weak-baseline regression revisited, composability]
---

# Re-test dropped levers on a strong base

A methodology point: a lever that **regressed on a weak baseline** isn't permanently dead — on a strong
base it may finally bite, because the two mechanisms can compose. Round 5 deliberately re-tests gates that
failed in Round 1.

## What it is

C3 (lay-side pwin) and C4 (race confidence) regressed in Round 1 on the weak baseline; with Round 5's
strong (BC-augmented) base, "the lay-side gates may finally bite — the 'remove bad lays' mechanism could
compose with BC's selection lift." The intuition: an env-side "remove bad opens/lays" gate has little to
add when the policy is already opening mostly junk (everything regresses together), but on a base that
already opens decent trades, removing the residual bad ones is pure gain. So dropped levers get a second
evaluation once the base improves, rather than being written off from their first-pass result.

## Why it matters

Guards against premature lever abandonment: "X didn't help" is conditional on the baseline X was tested
against. Whenever the base recipe materially improves, the dropped-lever list is worth re-running — the
composability with the new base is the thing being tested, not the lever in isolation. The flip side of
[[probe-to-cohort-regression]] (a lever that helps at probe scale can fail at cohort scale); here a lever
that failed on a weak base can help on a strong one. Same env-side "remove bad decisions" family as
[[remove-decisions-beats-teaching]].

## Sources
- `src-19293f` purpose.md (js_desktop:present)
