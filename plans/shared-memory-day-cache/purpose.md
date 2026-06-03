# shared-memory-day-cache — purpose

## Problem (the 2026-06-02 four-OOM firefight)

The multiprocess training path (`--parallel-agents N`,
`training_v2/cohort/multiproc_worker.py`) is **memory-bound to a uselessly
low N at predictors-ON**, because it **duplicates the engineered day
features across every process**:

- The **master** prebuilds and holds all training+eval days in RAM
  (measured **~47 GB for 32 days** ≈ ~1.5 GB/day).
- **Each worker** then holds its *own private* LRU copy of the days it
  touches (`_WORKER_DAY_CACHE`, a per-process dict filled by `pickle.load`).

So one day's features can live in **master + N workers = up to N+1 copies**.
Measured per-worker footprint ≈ **~6 GB fixed** (per-process torch+CUDA
context + its own copy of the predictor bundle) **+ ~1 GB × LRU_size** of
duplicated day features. At N=8 the training-start spike (all workers
materialising their day copies at once) hit **128 GB on a 128 GB box** and
OOM'd. We were forced down to N=4 / cache=4 just to fit — crippling the
throughput the multiprocess path exists to provide.

## Insight

The engineered day features are **policy-independent** — pure function of
`(day, cohort-fixed params)`, *identical* across every agent and worker
(the warm-pool comment already states "reuse is bit-identical"). Anything
identical across processes should be stored **once and shared read-only**,
not copied N+1 times. This is the same "compute-once, share" principle as
`plans/training-speedup-v2/` (Step 0 found the predictor/feature work is
policy-independent); here we apply it to *cross-process* memory.

## The fix

Store each engineered day's large arrays in a form that all processes
**reference one physical copy of**, read-only:

- **Recommended mechanism: `np.memmap` / `.npy` per day.** The master
  writes each day's big arrays to disk as memory-mappable `.npy`; workers
  `np.load(path, mmap_mode='r')` instead of `pickle.load`. The **OS page
  cache holds one physical copy** of those pages and shares them across
  every process that maps the file — automatic, robust, no explicit
  shared-memory lifecycle to leak. The prebuild **already writes per-day
  cache files** (`mp_day_{day}.pkl`); this is an evolution of that path
  (pickle → memmap), not a new subsystem.
- **Alternative: `multiprocessing.shared_memory.SharedMemory`** (explicit,
  RAM-only, no disk) — faster but more lifecycle/cleanup risk on Windows
  (spawn semantics, orphaned-block leaks). Fall back to this only if Step 0
  shows the disk/page-cache path is too slow.

## Expected win

Total day-storage collapses from `47 GB (master) + N × (LRU × ~1 GB)` to
**~47 GB once (shared) + N × ~6 GB (per-worker fixed)**:

| config | today (private copies) | with shared days |
|---|---|---|
| N=4 | ~87 GB (cap'd to fit) | ~47 + 4×6 = **~71 GB** |
| N=8 | **128 GB → OOM** | ~47 + 8×6 = **~95 GB** ✓ |
| N=12 | impossible | ~47 + 12×6 = **~119 GB** ✓ |

So predictors-ON multiprocess goes from "max useful N≈4" back to **N=8-12**,
restoring the ~8-9× the path was built for. The remaining ~6 GB/worker
(per-process torch/CUDA + predictor bundle) is a harder, separate problem
(Step 5, optional) — the day-feature duplication is the big, clean win.

## Non-negotiable: bit-identical

This is a **memory optimization, not a dynamics change**. The bytes a worker
reads from shared/memmapped day features MUST be identical to what it gets
today via `pickle.load`. Gated by the **`tests/test_env_golden_parity.py`
harness already built in `plans/training-speedup-v2/`** — a cohort on the
shared path must reproduce the pickle path bit-for-bit. Read-only is enforced
(memmap opened `'r'`); a worker that mutated a shared array would corrupt its
siblings.

See `hard_constraints.md` for the rules and `master_todo.md` for the steps.
Motivating evidence: the firefight trace in this session + the per-worker
decomposition (~6 GB fixed + ~1 GB/day × LRU).
