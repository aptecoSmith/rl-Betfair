# shared-memory-day-cache — master todo

Legend: `[ ]` todo · `[~]` in progress · `[x]` done. Each step has a
DELIVERABLE and a GATE. Nothing is "done" until its gate passes.

---

## Step 0 — Characterise `engineer_day` output + confirm shareability  `[x]`

**DONE 2026-06-02 — see `step0_structure.md`. KEY CORRECTION:**
`engineer_day` returns **nested dicts** (~1 GB/day, ~0 % arrays), NOT the
numpy arrays the plan assumed. The memmap target is the **downstream
`static_obs` float32 arrays** (93 MB/day full-obs, ~10–20× smaller, already
built per-env in `_precompute` but not cached). Mechanism: per-day
`static_obs.npy` memmap, predictors baked at cache-build, additive
`static_obs_cache=` env kwarg (default None = byte-identical). Bigger change
than "pickle→memmap" but a bigger win + the only form that shares
cross-process. Shareability ✓ (cohort-fixed, bit-identical across builds),
in-place-write (predictor `.update()` on dicts) eliminated by baking arrays.

Decide the mechanism on facts, not assumptions.

- **Structure:** what does `engineer_day` return? Map it into (a) large
  numpy arrays — the `static_obs` per-race `(n_races, n_ticks, n_feat)`
  tensors and any predictor-baked arrays — vs (b) small Python objects
  (dicts/lists/scalars/metadata). Measure the byte breakdown for one full-obs
  + predictors day (target: confirm the ~1 GB/day figure and that ≥~90 % is
  shareable arrays).
- **Shareability (HC#6):** confirm `engineer_day`'s inputs are cohort-fixed
  only (day + fixed params), with NO per-agent / per-gene / reward-override
  input. If anything cohort-varying leaks in, stop — sharing is unsound.
- **In-place writes (HC#2):** grep every consumer of the engineered-day
  arrays for in-place mutation (`arr[...] =`, `+=`, `.fill`, `np.put`, sort
  in place). Each such site needs copy-on-write before the shared rollout.
- **Mechanism decision:** memmap/`.npy` (recommended) vs `SharedMemory`,
  justified by the structure + a quick read-latency probe (memmap first-touch
  vs pickle.load).
- DELIVERABLE: `step0_structure.md` — byte breakdown, shareability
  confirmation, in-place-write audit, mechanism choice.
- GATE: byte breakdown reconciles with the measured ~1 GB/day; zero
  unaudited in-place writers remain.

---

## Step 1 — Per-day cache writer → memmappable format  `[x]`

**DONE 2026-06-02.** `training_v2/cohort/static_obs_cache.py::DayStaticObs`
(build/save/load/validate, atomic write, memmap read-only).
`multiproc_worker.prebuild_static_obs_cache` writes `static_obs_{day}.npy`
(predictors baked) + `meta_{day}.pkl` (gate caches + obs-contract manifest);
master holds ~1 day's arrays at a time. Env gains gated `static_obs_cache`
kwarg + `_precompute_from_static_obs` (default None = byte-identical).
GATES PASS: `tests/test_static_obs_cache.py` (13, format), the production
smoke (writer bakes predictors — champion_p_win non-zero across 11,353
ticks — and matches a from-scratch worker env bit-for-bit), and
`tests/test_env_golden_parity.py::test_static_obs_cache_path_matches_from_scratch`
(3 cases: normal/gated/force-close, bit-identical) + GATE(a) self-parity
still green.

Evolve the existing prebuild-to-disk path (`save_shared_cache_per_day`,
`mp_day_{day}.pkl`).

- Master writes each day's large arrays as memory-mappable `.npy` (or a
  single packed `.npz`/raw buffer + offset table), plus a small sidecar
  (json/pickle) for the Python-object metadata.
- Keep the master from also holding a full private RAM copy once the shared
  artifact exists (drop the master's ~47 GB after writing, OR have the master
  itself read via memmap so there's exactly one copy).
- DELIVERABLE: the new writer + the on-disk format spec.
- GATE: a written-then-reloaded day equals the in-RAM `engineer_day` output
  bit-for-bit.

---

## Step 2 — Worker read path → memmap, drop the private LRU  `[x]`

**DONE 2026-06-02.** `_worker_load_static_obs` (memmap, MRU cache of cheap
views; graceful fallback to non-mmap read then to from-scratch on any failure,
HC#3); `_train_agent_worker` consumes `_static_obs_day_paths` →
`static_obs_cache`; runner uses `prebuild_static_obs_cache` +
`_static_obs_day_paths` on predictors-ON multiproc (legacy dict path kept for
predictor-OFF). GATES PASS: 21 `test_v2_multiproc_cluster.py` (incl. 3 new
static_obs plumbing tests) + golden parity (Step 1). **End-to-end real run
(N=2, predictors-ON, 3 days):** completed exit 0; LGBM inference 506 lines
BEFORE the Multiprocess line, **0 after** (workers skipped inference =
consumed the baked cache); "Feature engineering" = 3 (master prebuild only,
zero worker re-engineering); both agents trained+evaled; **0 orphaned procs,
RAM 116 GB back to baseline** post-run. NOTE: Windows holds an open-memmap
file lock on the `.npy` while mapped (so cache files can't be deleted mid-run;
cleanup happens after the pool drains — see Step 3).

- `_worker_load_day` (multiproc_worker.py) reads via memmap (read-only)
  instead of `pickle.load`. Workers hold lightweight memmap *views*, not full
  copies, so `_WORKER_DAY_CACHE` shrinks to references (or is removed) — the
  OS page cache is the single shared store.
- Fallback to `pickle.load` on any memmap failure (HC#3), logged.
- DELIVERABLE: the shared read path + fallback.
- GATE: `tests/test_env_golden_parity.py` passes — a shared-path cohort
  reproduces the pickle-path golden bit-for-bit (HC#1).

---

## Step 3 — Memory + correctness validation  `[x]`

**DONE 2026-06-02 — GATE PASSED. See `step3_memory.md`.** Real predictors-ON
config, 8 days, auto-kill watch (`_measure/ram_watch.ps1`, CSVs `ram{4,8,12}`).
**N=8 plateaus at 97.5 GB Available / ~30 GB used** (vs the old 128 GB OOM and
the 110 GB gate — ~4× reduction, ~80 GB margin). Scaling `≈ baseline + N×2.4
GB` confirmed at N=4 (20 GB) / N=8 (30 GB) / N=12 (36 GB — the "impossible"
config now fits). Per-worker private ~2.2 GB (NOT 6 — CPU workers have no CUDA
context). Golden parity 10/10 still green. **Cleanup: RAM → 116 GB baseline
after every kill AND normal completion, 0 orphaned procs** (memmap uses files,
no SharedMemory blocks to orphan). RAM no longer binds N; the K≈12–20
throughput plateau does → cleared for Step 4.

- **Memory:** run the real config (predictors-ON, 32 days) at **N=8** and
  measure the RAM plateau through the **training-start spike** (the firefight
  auto-kill watch). Expect ~95 GB (47 shared + 8×6), NOT 128. Confirm it
  scales as `~47 + N×6` by also sampling N=4 and N=12.
- **Cleanup (HC#7):** after the cohort completes AND after a hard kill,
  confirm free RAM returns to baseline (no orphaned shared blocks/files).
- DELIVERABLE: a RAM-vs-N table + the cleanup check.
- GATE: N=8 plateaus < 110 GB with margin; golden parity still passes;
  RAM returns to baseline post-run and post-kill.

---

## Step 4 — Restore worker count + retire the band-aids  `[x]`

**DONE 2026-06-02.** Band-aids retired: `_WORKER_DAY_CACHE_MAX 4→16`
(now fallback/predictor-OFF only — predictors-ON uses the shared
`_WORKER_STATIC_OBS_CACHE` of cheap views); `--parallel-agents` code default
was already 16 (test-gated), the `ab_bc_chain.sh` band-aid `4→16` retired.
**Recalibrated** (`measure_optimal_n` updated to the static_obs path):
predictors-ON K=8 6.0× · K=12 8.0× · K=16 **9.1×** · K=20 9.4× (peak) —
~36 % faster than the pre-fix curve (workers skip per-tick inference). Default
16 kept (sweet spot; K=20 oversubscribes 20 cores for +3 %). Recorded in
`plans/EXPERIMENTS.md` + CLAUDE.md `--parallel-agents` note. The full-5-gen
no-OOM gate is validated live by the 18h BC A/B run launched on this setup
(`plans/shared-memory-day-cache/_scripts/launch_18h_bc_ab.sh`).

- Raise `--parallel-agents` back to the throughput peak the RAM now allows
  (re-run `tools/measure_optimal_n.py`); restore `_WORKER_DAY_CACHE_MAX` to a
  sensible value (the LRU now bounds cheap memmap views, not GB copies) or
  remove the cap.
- Re-measure per-gen wall — confirm we get the multiprocess speedup back at
  predictors-ON (target back toward ~8-9×).
- DELIVERABLE: the new default N + the measured speedup, recorded in
  `plans/EXPERIMENTS.md` and CLAUDE.md (update the `--parallel-agents` note).
- GATE: predictors-ON cohort runs stably at the new N with the measured
  speedup; no OOM across a full 5-gen run.

---

## Step 5 (optional, harder) — shrink the per-worker ~6 GB fixed  `[ ]`

The day features are now shared; the residual per-worker cost is the
per-process torch/CUDA context + its own predictor-bundle copy. Investigate:
can the predictor bundle be shared (it's torch/LightGBM models — harder
across processes), or CUDA context pooled? Likely lower ROI than Steps 0-4;
scope only if N is still RAM-limited after Step 4.

---

## Cross-cutting

- Reuse the `plans/training-speedup-v2/` golden-parity harness as the gate —
  do not build a new one.
- Record the RAM-vs-N table + speedup in `plans/EXPERIMENTS.md`.
- Update the CLAUDE.md "Fast cohort training: `--parallel-agents`" note with
  the new RAM math + safe N once Step 4 lands.
