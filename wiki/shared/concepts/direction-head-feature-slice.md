---
id: 01KTG18MW2ZC0AN1VTMCJC5DPG
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research]
sources: [src-0c84fe]
aliases: [phase-15 feature slice, direction head per-runner slice]
---

# Direction-head per-runner feature slice (phase 15)

The phase-15 fix for the [[lstm-compression-bottleneck]]: rewire `direction_prob_head`'s input from
`(slot_emb, lstm_last)` to the **per-runner RUNNER_KEYS slice** read directly from obs.

## What it is

**ONE structural change** — hand the head the runner's raw feature slice directly (the same block v1
extracts for `runner_feats_raw`, RUNNER_DIM=125); its first Linear shrinks from
`Linear(runner_embed+hidden, 64)` to `Linear(RUNNER_DIM, 64)`, output side unchanged. Everything else
is untouched: `actor_head` still reads `(slot_emb, lstm_last, fill/mature/direction probs)` (it still
benefits from the LSTM's cross-runner context), the other heads still read `lstm_last`, the gate, the
aux BCE loss, and reward magnitudes are unchanged (phase-14 rows stay comparable on `raw_pnl_reward`;
pre-phase-15 weights cannot cross-load — an arch-hash break by design). The **slot embedding is dropped
from the direction head** — it is unnecessary when each per-runner slice already differs by
construction (actor keeps it). Per §2, no backbone concat — adding `lstm_last` back would reintroduce
the bottleneck the plan exists to fix.

## Why it matters

A minimal-surgery way to bypass a shared-backbone bottleneck for one head while keeping the rest of the
network intact; its success is read off the [[bce-trajectory-load-bearing-diagnostic]].

## Sources
- `src-0c84fe` purpose.md (js_desktop:present)
