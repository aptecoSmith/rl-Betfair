---
plan: gauntlet-pipeline
status: proposed — design agreed, not built
---

# Master todo — gauntlet-pipeline

Phased build. Each phase is independently useful and low-risk; the executor +
ledger + breeder land behind a flag with the old `--breeding lockstep` left
intact, and nothing cuts over until the A/B (Phase 6) passes. Log results into
`findings.md` (create on first result; entry shape Intention / Implementation /
Result).

## Phase 0 — diagnostics (no training)
- [x] **Memory profile** — DONE 2026-06-13. `tools/profile_worker_memory.py`:
      per-worker peak **1.93 GB RSS / 1.61 GB USS** (predictors ON, full obs).
      Hypothesis confirmed — ~10 GB WS is RSS double-counting the shared memmap;
      predictor bundle only ~0.25 GB private/worker. Shared cache NOT regressed.
      Sharing predictor = LOW priority (worker-count cost, not day-growth); cheap
      alt = lazy-load bundle on cache-miss only. See `findings.md`.

## Phase 1 — gene register v1 (read-only coverage map)
- [x] DONE 2026-06-13. `tools/gene_register.py` + `tests/test_gene_register.py`
      (10 green). Loads scoreboard.jsonl / model_register.csv / reeval+holdout
      jsonl across eras; per-gene bins, visited cells, held-out outcome per cell,
      blank/promising-thin regions. Report `registry/gene_register.{txt,json}`.
- [x] Sanity: every persisted gene appears; cross-era de-dup by config hash
      (2865 models → 1214 distinct configs; legacy-7 all present). Tested.
- [x] Foundation for Phase 7 + next-Tock seed bands. See `findings.md`.

## Phase 2 — executor primitive (`run_tranche`)
- [x] DONE 2026-06-13. `training_v2/cohort/executor.py` + `tests/test_v2_executor.py`
      (9 green). `run_tranche(agents, tranche_K, train_days_for_K, validation_days,
      cfg)` → `list[TrancheResult]` (weights_path, fc=0 validation locked/naked/
      day_pnl). **No selection inside.** Builds the SAME `train_one_agent` spec
      the runner's multiprocess path builds (drift-guarded by a spec-key-vs-
      signature test); reuses predictors + shared static_obs cache + BC verbatim.
- [x] Recipe-purity assert: structural (K==1 ⟺ no weights) + a `.genehash`
      sidecar written next to each checkpoint and re-checked on warm-start
      (config_hash + lineage_id must match — blocks cross-recipe chimera).
- [x] Uniform-cost: a run is always `batch × 1 tranche` (single tranche's
      `train_days_for_K`; no catch-up replay — climbers warm-start own K-1).

## Phase 3 — ledger / queues (persistent state)
- [x] DONE 2026-06-13. `training_v2/cohort/ledger.py` + `tests/test_v2_ledger.py`
      (11 green). `GauntletLedger` (one JSONL, last-snapshot-per-lineage wins,
      atomic `compact()`): per-lineage `recipe/lineage_id/origin/weights_path/
      tranches_completed/validation_score[K]`. Derives `needs(K)` (completed==K-1)
      + `frontier()` (same-depth pool). Resumable via `load()`; a 2nd process can
      read it to grab a batch.
- [x] Leakage asserts: `DaySplit` + `assert_day_split_disjoint`
      (validation ∩ train == ∅; final_test ∩ (train ∪ validation) == ∅).

## Phase 4 — breeder (separate selection + mutation)
- [x] DONE 2026-06-13. `training_v2/cohort/breeder.py` + `tests/test_v2_breeder.py`
      (9 green). `breed_frontier(ledger, rng, cfg)`: reads `ledger.frontier()`
      (same-depth), truncation-selects on fc=0 composite (σ_leg ceiling via a
      pluggable `sigma_leg_fn` pre-filter, never deadlocks), keeps top fraction
      (stay active), culls rest, emits mutants (`make_offspring`) + fresh blood
      (`sample_fresh_blood_genes`) into `needs-T1`. **Full fair shot** proven by
      a test: a shallower-depth climber with the worst score is never culled.
- [ ] One-source-of-truth + OR-semantics wiring for any new flags. (Phase 5.)

## Phase 5 — orchestrator + flag
- [x] DONE 2026-06-13. `training_v2/cohort/gauntlet.py` (orchestrator:
      `seed_population` → `climb_to_frontier` uniform per-tranche batches →
      `breed_frontier` → repeat; resumable) + `tests/test_v2_gauntlet.py` (7).
      `--breeding gauntlet` wired in `runner.py`: argparse choice, validation,
      early dispatch to `_run_gauntlet_breeding` (builds DaySplit from
      `make_rotations` chronological + fixed fc=0 validation; reuses model_store
      + warm pool). Old ga/pbt/lockstep paths byte-untouched (early return).
- [x] Agent-1/tranche-1 audit: `tests/test_v2_gauntlet_runner.py` (2) drives the
      REAL executor+ledger+breeder via the runner dispatch (stub trainer) —
      queues populated, fixed validation reached every eval, recipe-purity
      sidecar chain real across tranches. Live (real-training) audit → Phase 6.
- [x] One-source-of-truth wiring: validation-warning + `_pbt_config` now include
      gauntlet; full suite (102 incl. existing cohort/lockstep) green, no regress.

## Phase 5b — real-training smoke (added 2026-06-13)
- [x] Small real run validated the LIVE path: shared static_obs cache bakes +
      consumed, multiprocess workers train, **tranche-2 warm-start passes recipe
      purity on REAL weights**, ledger advances. Surfaced + fixed 3 wiring gaps:
      `--seed-gene` now accepts gauntlet; the gauntlet path now WRITES
      `scoreboard.jsonl` rows (the Phase 6 judge + gene_register read them);
      `--era-type` is tick/tock only (omit for gauntlet). See `findings.md`.

## Phase 6 — A/B validation (gate to cutover)
- [ ] **READY TO LAUNCH.** Scripts: `tick-tock/launch_gauntlet_ab_gauntlet.sh`
      + `…_lockstep.sh` (matched: same seed/data/predictors/tranche-size, only
      `--breeding` differs). Runbook: `plans/gauntlet-pipeline/ab_runbook.md`.
      Run `--breeding gauntlet` vs `--breeding lockstep` on the SAME data + seed
      budget. Judge on the sealed-7 held-out (fc=0 select + fc=120 deploy): the
      new pipeline's held-out **locked/σ must match-or-beat** the old loop, via
      `tools.cross_era_holdout_board`. Record Intention/Implementation/Result.
      MULTI-HOUR compute — launch detached+logged.
- [ ] Confirm the wins: uniform per-run wall (no heavy tail), flat per-run
      memory, machine evenly fed.
- [ ] DECISION: cut over only if held-out quality holds. Else iterate.

## Phase 7 — gene register v2 (gap-targeted fresh blood)
- [ ] Replace uniform `sample_fresh_blood_genes` with register-driven draws from
      under-explored / promising-thin cells (low-discrepancy coverage). Gate
      behind a flag; A/B coverage + held-out outcome vs random fresh blood.

---

## Sequencing
1. **Now (no compute):** Phase 0 memory profile (when box free) + **Phase 1 gene
   register v1**. Both cheap, both immediately useful.
2. **Build core:** Phases 2→3→4→5 behind a flag (old path intact).
3. **Gate:** Phase 6 A/B — the only thing that authorises cutover.
4. **Later:** Phase 7 once the register has enough cross-era coverage to target.

## Open design details to pin during the build (not blockers)
- Operational cadence: does a "run to depth N" conclude and pause for new data,
  or breed continuously at the frontier? (Affects when selection fires.)
- Quorum rule for frontier selection (how full before trimming to carry-size).
- Multi-machine claim/lease protocol on the ledger (if/when we farm it out).
