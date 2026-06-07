---
id: 01KTG18MW3A94BCDNXRJY9SKMS
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-0c84fe]
aliases: [LSTM compression bottleneck, shared-summary bottleneck]
---

# LSTM-compression bottleneck (per-runner signal)

Per-runner direction signal cannot be cleanly recovered from the LSTM's shared 128-dim summary: the
cohort direction head reads `(slot_emb, lstm_last)`, but a supervised probe reading the runner's **raw
per-runner feature slice** got **24-94× top-quintile lift** on identical data.

## What it is

The LSTM squeezes 14 runners + market state into 128 numbers each tick. To predict runner 5's direction
the head had to dig runner 5's signal back out of that compressed shared summary — and there isn't
enough room in 128 dims to carry per-runner direction info for 14 runners cleanly, even with a learned
slot embedding to tag "which runner". Phase-14's sense-check pre-staged this: if direction BCE stays
flat on the gate-off arm, the bottleneck is the **LSTM-compression pathway, not the head architecture**.

## Why it matters

A representation-bottleneck distinct from the [[direction-head-data-ceiling]] (a signal ceiling):
here the signal exists in the raw inputs but is destroyed by shared compression. The fix is
[[direction-head-feature-slice]] (feed the head the raw slice). General lesson: a shared compressed
backbone can starve a per-entity auxiliary head even when the entity's signal is present in the inputs.

## Sources
- `src-0c84fe` purpose.md (js_desktop:present)
