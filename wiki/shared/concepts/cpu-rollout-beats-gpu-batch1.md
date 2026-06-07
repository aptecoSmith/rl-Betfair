---
id: 01KTGC1SK4Y6MSRYA5481JWN7D
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-106b41]
aliases: [CPU rollout, batch=1 kernel-launch overhead, bit-identical speedup, rollout_device split]
---

# CPU rollout beats GPU batch=1 (the bit-identical speedup lever)

The bit-identical training speedup turned out to be **CPU rollout**, not the GPU "batch=N forward" the
plan assumed. The batched forward CANNOT be bit-identical (it reorders float reductions ~1e-6 and flips
rare near-tie `Categorical.sample()` actions vs the CPU golden), so the safe lever is to run the rollout
on CPU — which *also* sidesteps the batch=1 CUDA kernel-launch overhead.

## What it is

Landed via the trainer's `rollout_device` split (rollout on CPU, PPO update on CUDA) plus a feature_cache
re-wire (it had been silently dropped — 11× redundant `engineer_day` per cluster-day). Measured A/B on a
real day, extrapolated to an 11-agent cluster: cluster-day wall **1130s → 789s (30.2% faster, 1.43×,
bit-identical)**; CPU rollout 74.0 → 56.9 s/agent (−23%), feature_cache 22.7 → 7.5s on hit. Gated by
`test_batched_path_matches_golden_fixture`.

The GPU batch=N forward was later built anyway as a *sanctioned non-bit-identical* lever (manual vs fused
LSTM float-reorder; actions/bets/value/P&L still match exactly): forward 4.8× (vmap) but env/obs dominate
at the cluster so it lands ~2.0× cluster-day; +R2 cross-agent scorer cache (bit-identical) reached ~2.5×.

## Why it matters

Two reusable rules: (1) batching a forward breaks discrete bit-identity
([[gpu-batch-breaks-discrete-parity]]), so a discrete-action golden harness must either run on CPU or
accept documented tolerances; (2) at batch=1 the GPU loses to CPU on kernel-launch overhead, so "move it
to the GPU" is not automatically faster. The multiprocess CPU path is the production fast lane
([[predictors-on-forward-is-cheap]] explains why a GPU forward lane is net-negative under predictors-ON).

## Sources
- `src-106b41` EXPERIMENTS.md (js_desktop:present)
