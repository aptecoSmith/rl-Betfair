---
id: 01KTFTM1P9WK9Q2EW75G5TTJ25
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-0604e6]
aliases: [value-domain check, feature value assertion, time_to_off backward]
---

# Value-domain feature assertions

When a feature-engineering loop indexes into sorted views, ship a per-feature **value-domain
assertion** ("for one named runner at one named tick, is the feature reading the value I expect?"),
not just shape/z-score checks — that is the only kind of check that catches reading-the-wrong-tick bugs.

## What it is

A tick-direction feature went through two wrong versions; **three independent fences failed to catch
v1**: the no-lookahead smoke test (perturbs post-D ticks, blind to a wrong *pre*-D read), the z-score
sanity table (caught v0's std=0 but reported clean stats for v1 — wrong-tick values are still varied),
and visual review (a 15% mean "felt right" but was 5x too high). The catch came from spot-checking one
outlier with `ticks.head(30)` and seeing the timestamps spell out the inversion. The pandas trap:
`time_to_off_s` **counts BACKWARD** (large = far from off), so "most recent" = smallest `time_to_off_s`
= `iloc[0]` under ascending sort — easy to invert if you think in chronological time.

## Why it matters

Shape/z-score/no-lookahead checks all pass on data read from the *wrong* tick; only a value-domain
assertion ("feature X for runner Y at tick D equals the value computed directly from the closest tick")
catches it — cheap to write, would have caught both wrong versions immediately. See the sibling signal
[[heavy-tail-as-bug-signal]].

## Sources
- `src-0604e6` lessons_learnt.md (js_desktop:present)
