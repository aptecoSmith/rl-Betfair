---
id: 01KTGC1SKAH5WQC9AVFW64N6VD
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-106b41]
aliases: [shared-memory day cache, static_obs memmap, predictors-ON OOM fix, DayStaticObs]
---

# Shared-memory day cache (predictors-ON OOM fix)

The infra change that restored the predictors-ON multiprocess path to ~9× after it OOM'd a 128 GB box at
N=8 (band-aided to a crippling N=4). Each worker had held its own ~1 GB/day private copy of the
engineered features on top of the master's ~47 GB. The fix: store each day's features **once, shared
read-only across processes** via memory-mapping — a pure memory change, bit-identical.

## What it is

Step 0 corrected the plan's premise: `engineer_day` returns nested Python **dicts** (~1 GB/day, not
memmappable), but the arrays the env actually reads are the downstream `env._static_obs` (~93 MB/day
full-obs, ~10–20× smaller). So the master bakes each day's `static_obs` float32 arrays (predictors baked
in) as a per-day `.npy` + a `meta_{day}.pkl` sidecar (gate caches + obs-contract manifest); workers
`np.load(mmap_mode='r')` so the OS page cache holds ONE physical copy. Gated by
`test_static_obs_cache_path_matches_from_scratch`.

Measured RAM (127.7 GB box, predictors-ON, 8 days): N=4 → 20 GB; **N=8 → 30 GB** (was OOM at 128 GB; ~4×
cut); N=12 → 36 GB. Scales `≈ baseline + N×2.4 GB`. RAM no longer caps N — the K≈12–20 throughput plateau
does. Recalibrated speedup: K=16 **9.1×**, K=20 9.4× (peak). Band-aids retired (`--parallel-agents 4→16`).

## Why it matters

The enabler for predictors-ON cohorts at full concurrency — the intended production config. It composes
with [[cpu-rollout-beats-gpu-batch1]] (CPU workers ⇒ no per-worker CUDA context, so per-worker overhead
is far below the first projection). The recurring discipline: a "per-day cache" is a silent RAM multiplier
under multiprocess; memmap the smallest array the consumer actually reads, not the upstream dict.

## Sources
- `src-106b41` EXPERIMENTS.md (js_desktop:present)
