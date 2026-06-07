---
id: 01KTGJS2NDKMGAX5AGKP48APC9
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-184d90]
aliases: [coverage includes discarded models, did we sample this corner, float_log buckets, clamp out-of-range]
---

# Coverage measures exploration, not quality

A measurement-design lesson: a search-space *coverage* metric must count **discarded and garaged models**,
not just `status='active'` ones — the question is "did we ever sample this corner?", not "did we like the
result?".

## What it is

The first draft filtered coverage history to active models on the reasonable-sounding basis that
"discarded models are bad examples" — wrong framing: a discarded `ppo_lstm_v1` with `gamma=0.998` is still
evidence that bucket was explored. Two companion bucketing fixes: `float_log` genes need **log-space**
bucket edges (linear deciles put ~99% of log-distributed `learning_rate` samples in the bottom bucket,
falsely reading as "unexplored"), matching how `sample_hyperparams` actually draws; and out-of-range
historical samples (after a range is tightened) are **clamped to the end bucket, not dropped** — dropping
would understate coverage in exactly the buckets that did have samples.

## Why it matters

A general metric-semantics trap: a metric named for one thing (coverage = exploration breadth) quietly
filtered by another (quality) measures neither. Match the bucketing to the *sampling distribution*
(log-space for log-sampled genes) and decide deliberately how to treat out-of-domain history (clamp, not
drop) so a config change doesn't silently rewrite past coverage. Kin to
[[selector-blind-to-incentivised-cost]] — define the metric by the question it must answer.

## Sources
- `src-184d90` lessons_learnt.md (js_desktop:present)
