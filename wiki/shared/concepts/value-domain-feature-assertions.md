---
id: 01KTFBWWTPGG22V1KR39BBD0NN
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-0604e6]
links: []
aliases: [value-domain check, named-runner head(30), feature engineering value assertion]
---

# Value-domain feature assertions (the only thing that caught the bug)

When a feature-engineering loop indexes into sorted views, ship a
per-feature **value-domain assertion** alongside the usual shape/z-score
sanity report: "for one named runner at one named tick, the feature must
equal X to within float epsilon."

## What it is

The trajectory-retrieval probe went through two wrong "value at decision-
tick D" implementations before landing on the correct one. Three
independent fences failed to catch the second wrong version (v1):

- **No-lookahead smoke test** — perturbs post-D ticks; doesn't catch a
  WRONG pre-D tick being read.
- **z-score sanity table** — caught v0's std=0 degeneracy but cleared v1
  (values from the wrong tick are still varied, just wrong).
- **Visual review of feature magnitudes** — `target_log_return` |mean|=15%
  "felt about right" but was 5× too high (should be ~3%) without a
  declared expectation.

The catch came from `ticks.head(30)` on a single named outlier — reading
the actual timestamps spelled out the time-inversion.

The pandas gotcha that motivates it: `time_to_off_s` counts **backward**
(large = far from off), so after `sort_values(ascending=True)`, `iloc[0]`
is the smallest `time_to_off_s` = the **most recent in chronological time**.
Easy to invert.

## Why it matters

Shape and z-score diagnostics are blind to "right shape, wrong tick" bugs.
A value-domain assertion encodes the domain expectation explicitly, fails
loudly on the first run, and costs about three lines to write. Generalises
to any feature pipeline whose inputs are sorted views with non-trivial
sort-key semantics. See also: [[heavy-tail-as-bug-signal]] — a single
~90σ z-score is more likely a bug than a real fat tail.

## Sources
- `src-0604e6` lessons_learnt.md (js_desktop:present)
