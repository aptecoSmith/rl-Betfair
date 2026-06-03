# Session prompt — execute `shared-memory-day-cache`

You're implementing shared-memory day features so the multiprocess training
path stops duplicating ~1 GB/day per worker (which OOM'd a 128 GB box at N=8
on 2026-06-02 and forced a crippling drop to N=4).

## Read first
1. `plans/shared-memory-day-cache/purpose.md` — the why + the measured
   memory decomposition (~6 GB fixed/worker + ~1 GB/day × LRU, duplicated
   master + N workers).
2. `plans/shared-memory-day-cache/hard_constraints.md` — **inviolable.**
   Bit-identical gate, read-only sharing, graceful fallback, Windows spawn,
   cleanup, measure-don't-estimate.
3. `plans/shared-memory-day-cache/master_todo.md` — steps 0-4 + gates.
4. `training_v2/cohort/multiproc_worker.py` — `_worker_load_day`,
   `_WORKER_DAY_CACHE`, `save_shared_cache_per_day` (the per-day cache flow
   you're evolving from pickle → memmap).
5. `plans/training-speedup-v2/` — reuse its `tests/test_env_golden_parity.py`
   harness as the gate; same "compute-once, share" principle.

## Discipline (this plan was born from 4 OOMs)
- **Measure, never estimate** memory. Before raising N, run the real config
  and watch the RAM plateau through the *training-start spike* with an
  auto-kill watch (free RAM < 12 GB → `taskkill /F /IM python.exe /T`). Four
  OOMs came from committing compute on memory guesses.
- **Kill protocol:** multiprocess pools orphan their workers — `Stop-Process`
  on the parent does NOT cascade. Use `taskkill /F /IM python.exe /T` and
  verify `Get-Process python` is 0 and RAM returns to baseline.
- **Bit-identical or it didn't happen:** every change gated by the golden
  harness. This is a memory change, not a dynamics change.

## Order + stop-points
- **Step 0 (characterise)** is pure investigation — do it fully before any
  code. Confirm shareability (cohort-fixed inputs only) and audit in-place
  writes. **Report the structure + mechanism choice before writing the new
  format.**
- Steps 1-2 (writer → memmap, worker read path) — each gated by golden parity.
- **Step 3 (memory validation)** — the load-bearing proof. Report the RAM-vs-N
  table (expect N=8 ≈ 95 GB) before raising the production default.
- Step 4 (restore N + retire band-aids: `_WORKER_DAY_CACHE_MAX`, the low
  `--parallel-agents`). Record in EXPERIMENTS.md + CLAUDE.md.

## Stop and report when
- Step 0 structure/mechanism decided · Step 3 RAM-vs-N measured · any golden
  parity failure · before changing the production `--parallel-agents` default.

## Done means
N=8 predictors-ON runs a full 5-gen cohort with no OOM, RAM plateaus
< 110 GB, golden parity holds, RAM returns to baseline after run and after
kill, and the measured speedup is back toward ~8-9× — recorded.
