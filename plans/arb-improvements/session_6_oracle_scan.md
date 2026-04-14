# Session 6 — Arb oracle scan

## Before you start — read these

- `plans/arb-improvements/purpose.md`
- `plans/arb-improvements/master_todo.md` — Phase 3, Session 6.
- `plans/arb-improvements/testing.md`
- `plans/arb-improvements/hard_constraints.md` — **oracle runs
  offline only; targets must be env-reachable; output is
  deterministic.**
- `plans/arb-improvements/progress.md` — read Phase 2 completion.
- `env/exchange_matcher.py` — the filter rules the oracle must
  mirror (LTP-aware junk filter, hard price cap after filter).
- `CLAUDE.md` "Order matching" — same rules, stated in prose.

## Goal

For every day of training data, produce a deterministic on-disk
dataset of `(obs_vector, runner_idx, oracle_arb_spread_action)`
samples marking every real arb opportunity. Consumed later by BC
pretraining (Session 7). Training never runs the scan — it reads.

## Scope

**In scope:**

- New module `training/arb_oracle.py` with:
  - `OracleSample` dataclass: `tick_index: int, runner_idx: int,
    obs: np.ndarray, arb_spread_ticks: int, expected_locked_pnl:
    float`.
  - `scan_day(date: str, data_dir: Path, config: dict) ->
    list[OracleSample]`: walk every tick of every race on `date`,
    detect moments where an arb is lockable after commission and
    reachable by the env (filters applied), emit samples.
  - The arb-detection uses the same pure functions from Session 4
    so feature semantics are consistent end-to-end.
  - The `arb_spread_ticks` in each sample is the optimal placement
    (the exact number of ticks between best back and best lay at
    that moment, clamped to `[MIN_ARB_TICKS, MAX_ARB_TICKS]`).
- Output: compressed `.npz` at
  `data/oracle_cache/{date}/oracle_samples.npz` containing stacked
  `obs` (float32), `runner_idx` (int16), `arb_spread_ticks`
  (int8), `expected_locked_pnl` (float32), `tick_index` (int32).
- CLI entrypoint: `python -m training.arb_oracle scan --date
  2026-04-06 [--dates 2026-04-06,2026-04-07]`. Prints per-day sample
  count, total runtime, and a density metric
  (`samples / total_ticks`).
- Load helper: `load_samples(date, data_dir) -> list[OracleSample]`.
  Missing file → empty list (not an error).

**Out of scope:**

- BC training (Session 7).
- Wiring the oracle into the training loop (Session 7).
- UI work — append to `ui_additions.md`, Session 8 consolidates.

## Exact code path

1. Create `training/arb_oracle.py`.
2. Import `env/features.py` pure functions and the env's
   `ExchangeMatcher` filter predicate (export it as a callable from
   `env/exchange_matcher.py` if it isn't already exposed — do not
   re-implement the filter rules).
3. For each race on `date`:
   - Load the episode (same path the training loop uses — reuse
     existing data-loading utilities; don't invent a new one).
   - Walk tick-by-tick. At each tick, for each runner:
     - Compute `arb_lock_profit_pct` and `arb_spread_ticks` using
       the pure functions.
     - If `arb_lock_profit_pct > 0` and the paired order would
       pass the matcher filters (price cap, LTP-aware junk, etc.):
       build the obs vector for this tick (reuse the env's
       `_build_observation` path or a lightweight equivalent),
       emit one `OracleSample`.
4. Write the `.npz` atomically (write to `.tmp`, rename).
5. Log per-day density and warn if density < 0.001 (fewer than
   1 arb moment per 1000 ticks).

## Tests to add (all CPU-only, fast)

Create `tests/arb_improvements/test_arb_oracle.py`:

1. **Synthetic day with one injected arb moment → one sample.**
   Build a fake episode where exactly one tick has a crossed
   post-commission book that also passes the filters. Run
   `scan_day`; assert exactly one `OracleSample`, with the right
   `runner_idx` and `tick_index`.

2. **Filter compliance.** Build a fake episode where a crossed
   book exists but at a price that the matcher's price cap would
   refuse. Assert zero samples emitted.

3. **Filter compliance — junk filter.** Build a fake episode
   where the crossed book is junk (far from LTP). Assert zero
   samples.

4. **Empty day → empty dataset, no crash.** A day with zero arb
   moments produces an empty list; the `.npz` still writes
   successfully (even if empty).

5. **Determinism.** Scan the same synthetic day twice; assert the
   two `.npz` files are byte-identical.

6. **Round-trip.** Save samples, load via `load_samples`, assert
   equality of every field.

7. **Density metric.** The CLI output contains
   `samples=X ticks=Y density=X/Y`, assertable from captured stdout.

8. **Obs dim matches env.** On a synthetic tick, assert
   `oracle_sample.obs.shape[0]` equals
   `env.observation_space.shape[0]` (so BC can feed samples to the
   policy without shape errors).

## Session exit criteria

- All 8 tests pass: `pytest tests/arb_improvements/test_arb_oracle.py -x`.
- Existing tests still pass.
- CLI works end-to-end on at least one of the 90fcb25f training
  dates (2026-04-06 / 07 / 08). Record sample count per day in
  `progress.md`.
- `ui_additions.md` Session 6 tasks appended.
- `lessons_learnt.md` updated if density is surprisingly low on any
  real day.
- Commit: `feat(training): arb oracle scan + on-disk sample cache`.
- `git push all`.

## Do not

- Do not run the scan inside the training loop. Ever. It's
  offline-only by contract.
- Do not emit samples for moments the env would reject. The whole
  point is that BC targets are reachable.
- Do not write non-deterministic output. Same input, same bytes.
- Do not expand scope to "also precompute features for BC". BC
  calls the policy's forward on the obs vector — no extra
  precomputation needed, and mixing concerns makes the cache
  fragile.
- Do not add GPU tests.
