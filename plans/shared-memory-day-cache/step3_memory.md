# Step 3 — memory + correctness validation (RAM-vs-N)

**Status: COMPLETE. GATE PASSED.** Measured on the real Windows box
(127.7 GB physical), predictors-ON + FULL obs, 8 training-pool days, one full
wave of N workers per run, with the PowerShell auto-kill RAM watch
(`_measure/ram_watch.ps1`, threshold 12 GB Available). CSVs:
`_measure/ram{4,8,12}.csv`.

Metric notes: **Available** = `\Memory\Available MBytes` (free + reclaimable
standby/file-cache — the correct OOM headroom metric; the shared `static_obs`
pages live in the reclaimable file cache). **Commit** = `\Memory\Committed
Bytes` (the true OOM driver). **py-private** = Σ `PrivateMemorySize64` over
all python procs (master + N workers) — file-backed memmap pages are NOT
private, so this is the per-process *real* footprint and shows the
day-sharing win directly.

## RAM-vs-N (peak / plateau through the training-start spike)

| run | min Available | peak Commit | peak py-private | physical used¹ | per-worker priv² |
|---|---:|---:|---:|---:|---:|
| idle baseline | 116.0 GB | 24.7 GB | 0 | 11.7 GB | — |
| **N=4** | 107.3 GB | 36.0 GB | 11.2 GB | 20.4 GB | ~1.7 GB |
| **N=8** | **97.5 GB** | 48.1 GB | 23.2 GB | **30.2 GB** | ~2.3 GB |
| **N=12** | 91.3 GB | 56.0 GB | 31.3 GB | 36.4 GB | ~2.2 GB |

¹ `physical used = total(127.7) − Available`. ² `(peak py-private −
master ~4.5 GB) / N`.

**Run-added physical** (used − 11.7 idle): N=4 **+8.7**, N=8 **+18.5**,
N=12 **+24.7 GB**. Available reduction ≈ **2.3–2.5 GB per worker**; commit
≈ 3 GB/worker. Slightly *sub*-linear because the 746 MB 8-day shared cache
is counted **once** regardless of N (the whole point).

### The gate

- **N=8 plateaus at 97.5 GB Available (30.2 GB used) — vs the 110 GB gate
  and the 127.7 GB box. PASS with ~80 GB of headroom.** The old private-copy
  path OOM'd this exact config at **128 GB**; the shared-memmap path uses
  **~30 GB**. A ~**4× reduction** at N=8.
- **N=12 — the config the plan called "impossible" — fits at 91.3 GB
  Available (36 GB used).** Scaling confirmed `≈ baseline + N × ~2.4 GB`.
- The auto-kill watch never tripped (min Available 91.3 GB ≫ 12 GB
  threshold) — it was a safety net, not a brake.

### Why even better than the Step-0 projection (~51 GB)

Step 0 assumed ~6 GB/worker fixed (torch + **CUDA context** + bundle).
Measured per-worker is **~2.2 GB**, because multiprocess workers run
`device="cpu"` (spec-forced) → **no per-worker CUDA context**. The residual
per-worker is torch-CPU + the predictor bundle + rollout buffers. So N is now
**CPU-throughput-bound, not RAM-bound**.

## Projection to the real 32-day config

The per-worker private footprint is **day-count-independent** (days are
memmapped, not copied), so the only delta vs the real `--days 32` config is
the shared page-cache: 32 × ~97 MB ≈ **3.1 GB** (vs 0.75 GB at 8 days) →
**+2.3 GB reclaimable**. And `--n-agents 30 --parallel-agents N` runs
`ceil(30/N)` waves on the **same** N warm workers, so the peak = one wave =
exactly what was measured. Therefore the real-config plateaus project as:

| N | projected Available (32 days) | projected Commit | verdict |
|---|---:|---:|---|
| 8 | ~95 GB | ~48 GB | ✓ comfortable |
| 12 | ~89 GB | ~56 GB | ✓ comfortable |
| 16 | ~78 GB | ~66 GB | ✓ fits (CPU-bound first) |
| 18 (cpu−2) | ~73 GB | ~71 GB | ✓ fits |

RAM no longer constrains N up to the core count; the throughput plateau
(K≈12–20, `tools/measure_optimal_n.py`) is the binding limit again.

## Cleanup (HC#7) — RAM returns to baseline

| event | python procs after | Available after |
|---|---:|---:|
| N=2 wiring run **completes** normally | 0 | 116 GB |
| N=8 **hard kill** (`taskkill /F /IM python.exe /T`) | 0 | 115.9 GB |
| N=4 **hard kill** | 0 | 116.0 GB |
| N=12 **hard kill** | 0 | 116.0 GB |

Every termination — normal completion **and** hard kill — returned to the
~116 GB idle baseline with **zero orphaned processes**. The kill protocol
(`taskkill /F /IM python.exe /T`, by image name + `/T` tree) cascaded
cleanly every time. The memmap mechanism uses **files**, so there are **no
`SharedMemory` blocks to orphan**; the per-run cache files persist on disk
(intended — reusable across generations) and their page-cache is reclaimable
(Available returns to baseline ⇒ reclaimed). NB (Windows): an open memmap
holds a file lock on the `.npy`, so cache files can only be deleted after the
worker pool drains — cleanup is post-run, never mid-run.

## Correctness

Golden parity re-confirmed alongside the memory run
(`test_static_obs_cache_path_matches_from_scratch` + GATE(a)
`test_current_env_matches_golden_fixture`) — the shared path stays
bit-identical to the from-scratch build (Steps 1–2 gates). The N=8/4/12 runs
themselves dispatched workers that consumed the cache (0 predictor-inference
calls post-spawn in the N=2 e2e run; here, 0 `baked` lines on N=4/N=12 also
confirmed the idempotent cache-reuse skip).

## GATE: PASS

N=8 plateaus < 110 GB (97.5 GB Available / 30 GB used) with ~80 GB margin;
scaling `≈ baseline + N×2.4 GB` confirmed at N=4/8/12; golden parity holds;
RAM returns to baseline post-completion and post-kill. **Cleared to raise
`--parallel-agents` (Step 4).**
