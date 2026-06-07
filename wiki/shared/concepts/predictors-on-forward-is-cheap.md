---
id: 01KTGC1SK8PP6DR2FEC33XYNG4
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-106b41]
aliases: [GPU forward lane NO-GO, predictor floor, forward is 3-14%, CPU-core-bound, big-model-threads]
---

# Predictors-ON, the forward is cheap (GPU-lane NO-GO)

A measure-and-decide gate that killed a multi-day build before it started. Hypothesis: heavy archs
(transformers, wide LSTMs) are FLOP-bound and starved on the CPU 1-thread multiprocess path → route them
to the idle GPU. **NO-GO** — under the real predictors-ON config the policy forward is only **3–14% of
the agent-day**; the LightGBM predictor + scorer floor is the other ~86–97%.

## What it is

Per-arch microbench (batch=1 forwards, full obs 2254-d): transformers ~4% of the agent-day, h1024 LSTM
14% @1t / 8% @ the real 6-thread baseline. GPU batch=1 *loses* for every arch except the FLOP-bound h1024
LSTM (the rest are launch-bound; transformers run 1.5–2× slower on CUDA). A GPU forward lane would save
~5% on the costliest agents while slowing most of the population → net-negative. The original premise was
wrong because the COMPUTE NOTE it came from was a predictors-OFF lean smoke; predictors-ON dominates every
arch and erases the transformer's forward edge.

The addendum: `--big-model-threads 6` is itself net-negative at cohort scale (gen-0 ~60 min vs ~45 min
single-threaded — 16×6=96 threads oversubscribe 20 cores). The box is **CPU-core-bound at N=16**; nothing
that still needs cores (threads, or a GPU lane whose env runs on CPU) accelerates the per-tick work
without contending for the saturated pool. Fix: revert to `--big-model-threads 1` (pure multiprocess,
byte-identical).

## Why it matters

The decisive reason the cohort fast path is CPU multiprocess, not GPU ([[cpu-rollout-beats-gpu-batch1]]).
The real lever is the predictor/scorer floor — already attacked by the R2 scorer cache and the
[[shared-memory-day-cache]] memmap — not the forward. A 1h measure-first gate prevented a net-negative
multi-day build: always profile the *real* (predictors-ON) config before optimising a lane.

## Sources
- `src-106b41` EXPERIMENTS.md (js_desktop:present)
