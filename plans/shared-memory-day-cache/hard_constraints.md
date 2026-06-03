# shared-memory-day-cache — hard constraints

1. **Bit-identical (the spine).** The day-feature bytes a worker reads via
   shared memory / memmap MUST equal what it reads today via `pickle.load`.
   Gate every change with `tests/test_env_golden_parity.py` (from
   `plans/training-speedup-v2/`): a cohort on the shared path reproduces the
   pickle path bit-for-bit (discrete exact, continuous within the declared
   per-quantity tolerance). This is a memory change, never a dynamics change.

2. **Read-only sharing.** Shared/memmapped day arrays are opened read-only
   (`np.load(..., mmap_mode='r')` → arrays with `WRITEABLE=False`). No worker
   may mutate them — a mutation corrupts every sibling sharing the page. If
   any consumer currently writes into the engineered-day arrays in place,
   that site must copy-on-write first (and that copy is then private/small).
   Audit `engineer_day` consumers for in-place writes in Step 0.

3. **Graceful fallback, never crash.** If the shared/memmap path fails
   (platform, permissions, missing file, mmap error), fall back to the
   existing `pickle.load` private-copy path with a logged warning. Shared
   memory is an optimisation; its failure must degrade to "slower + more
   RAM", not "cohort dies".

4. **Windows spawn semantics.** This box is Windows (multiprocessing =
   *spawn*, not fork). Shared state is NOT inherited — workers must attach by
   path/name. `np.memmap` of a file passed by path is spawn-safe; explicit
   `SharedMemory` blocks must be passed by name and **explicitly
   closed/unlinked** at cohort end or they leak. Test on the real Windows box,
   not by reasoning about POSIX fork behaviour.

5. **No silent feature drops (inherited HC).** The shared path must carry
   every feature the pickle path does — predictors, input_norm inputs, the
   full obs. A regression test asserts the shared-path env build produces the
   same feature set. (This plan exists because the `--batched` path silently
   dropped predictors; do not reintroduce that class of bug.)

6. **Preserve warm-pool correctness.** The days are shared *because* they are
   policy-independent (pure fn of day + cohort-fixed params). If any
   cohort-varying input (a gene, a reward override, a per-agent flag) leaks
   into `engineer_day`, sharing one copy across agents is WRONG. Step 0 must
   confirm `engineer_day` takes only cohort-fixed inputs before sharing is
   sound.

7. **Cleanup is mandatory.** Whatever is allocated (memmap files, SharedMemory
   blocks) is released at cohort end AND on crash/kill (best-effort
   `atexit` / `finally`). The firefight showed orphaned workers re-consume
   RAM; orphaned shared blocks would do the same. Verify free RAM returns to
   baseline after a cohort completes and after a kill.

8. **Measure, don't estimate (the firefight lesson).** Before raising N on the
   strength of this fix, MEASURE the actual RAM plateau through the
   training-start spike at the target N (the auto-kill validation watch from
   the firefight). Four OOMs came from committing compute on memory estimates;
   the per-worker footprint and the post-fix plateau are both to be measured,
   not extrapolated.
