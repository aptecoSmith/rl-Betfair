---
id: 01KTGC820A2QCKX51EEVGQ56YT
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-106b41]
aliases: [locked_per_std blind to fc, selector incentivises hidden cost, fc substitutable for variance, day_pnl_per_std]
---

# A selector blind to the cost it incentivises (tnv2/tnv3)

The failure mode that sank the `locked_per_std` GA selector: it rewards a high locked floor and tight
variance, both of which the heavy-force-close phenotype achieves — so the metric **breeds exactly the
cost it can't see**.

## What it is

tnv2 trained at fc=120 with `locked_per_std` and REGRESSED across all 4 cells on the layq null
(fc=120 newwindow −£177/d, 0/10 profitable). The locked floor was actually *higher* than layq (+£198 vs
+£122) but force-close cost climbed **3.6× (−£251/d vs −£69/d)** because training at fc=120 with a
locked-rewarding selector produced volume-of-opens agents. Root cause: `locked_per_std` is blind to the
fc cost it incentivises. Worse, **fc cost is partially substitutable for naked variance in the
selector** — two agents with identical day_pnl can score 3.8× differently if one converts variance into
bounded fc cost, because the denominator's non-linear σ penalty rewards heavy fc use when the numerator
is held constant. The fix (tnv3) switched the numerator to day_pnl directly so the GA can't game the
locked floor without paying for fc.

## Why it matters

A general selector-design trap: if your fitness numerator rewards a proxy (locked) and your denominator
penalises one cost (naked variance) but ignores another (fc cost), the GA finds the phenotype that maxes
the proxy and hides the cost in the ignored channel. The deeper lesson is [[selection-vs-measurement-signal]]
(selection can't fix what PPO learns) — but even as a pure measurement, a selector must price *every*
cost it can trade against. Part of the [[scalping-cohort-lineage]] tnv regression arc.

## Sources
- `src-106b41` EXPERIMENTS.md (js_desktop:present)
