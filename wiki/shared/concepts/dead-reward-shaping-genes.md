---
id: 01KTFBKMQ1YX07KXGR8F6CREC0
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-032073]
links: [{to: raw-vs-shaped-reward, type: see-also}]
aliases: [dead genes, silently-ignored genes, plumbing bugs]
---

# Dead reward-shaping genes (early GA plumbing bug)

A class of bug — and a recurring lesson — where the genetic search **samples
a gene that no downstream consumer reads**, so every agent trains with the
default and the population is degenerate on that axis.

## What it is

In the pre-`e76ac98` cohort code, `reward_early_pick_bonus`,
`reward_efficiency_penalty`, and `reward_precision_bonus` were sampled
per-agent in `population_manager.py:220` but `env/betfair_env.py` read the
values straight from `config.yaml`, so the per-agent values were silently
ignored. `observation_window_ticks` was sampled but never read at all. The
GA looked alive (population members had distinct genomes) while actually
being degenerate.

The diagnostic is mechanical: every gene must be plumbed through to the
object that uses it, and that wiring must be exercised by a test — sampling
without reading is the bug. This is the same shape of failure later
captured by [[audit-launch-wiring-foot-gun]] (the Path-A `.get(key, default)`
foot gun in cohort worker code) — a cohort-flag/CLI knob that the worker
reads with a fallback never sees the launcher's value.

## Why it matters

The whole point of a GA is exploration of the search dimensions. A dead
gene is **invisible no-op exploration** — the cohort runs, costs compute,
and produces what looks like signal but is just noise around a single
configuration. The lesson generalised: a new knob needs a refusal/activity
counter and an end-to-end test before any cohort campaign uses it.

## Sources
- `src-032073` purpose.md (js_desktop:present)
