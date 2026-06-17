# Findings — gauntlet-pipeline

Build log. Entry shape: **Intention / Implementation / Result.**

---

## Phase 6 — A/B validation (the cutover gate)  [2026-06-16]  — PASS

**Intention.** Run `--breeding gauntlet` vs `--breeding lockstep` on the SAME
data + seed, judge both on the sealed-7 held-out (locked/σ_naked_leg). Cut over
only if gauntlet matches-or-beats.

**Implementation.** `tick-tock/run_gauntlet_ab_full.sh` — both arms identical
except `--breeding`: 16 agents, full 43-day non-sealed pool → 4 fixed tranches,
`--validation-holdout-recent 5` (fc=0 select), `--holdout-recent 7` (sealed),
full `--enable-all-genes` + `--gpu-policy-lane` (transformers on CUDA),
predictors ON. Gauntlet `--generations 4` (3 breed rounds) ≈ lockstep's 4
tranches (3 selection boundaries) — matched selection rounds. Judged via
`tools.cross_era_holdout_board --rank-by locked_over_sigma`. Wall: arm A ~24h,
arm B ~19h, judge ~5.5h (~2.5 days total — full-obs/transformer agents +
full-fair-shot re-climbs).

**Result — PASS. Board `registry/gauntlet_ab_board.txt`:**
- **Gauntlet champion da8ce989 (transformer): held-out locked £19.32 @ σ_leg
  £22.8 (locked/σ 0.85) — the #1 model on the board.**
- Lockstep champion 7b449d99: £15.83 @ £22.7 (locked/σ 0.70).
- Gauntlet's best beats lockstep's best by ≈+22% on locked/σ; both σ_leg inside
  the £30 ceiling. Gauntlet holds 7 of the top 12 slots.
- Honest caveats: (1) single seed, modest margin — lockstep takes ranks 2-3, so
  it's "gauntlet's best > lockstep's best", not a sweep. (2) Both arms' absolute
  quality is still poor (large negative naked ho_nkd −250…−470, maturation 7-12%)
  — that's the known unsolved SELECTIVITY problem (maturation-raising), NOT an
  orchestration difference. The A/B only asks "is the gauntlet orchestration at
  least as good?" → yes.

**DECISION: gate cleared → proceed to cutover** (flip default to gauntlet +
repoint tick/tock bats + Phase C/D removals), per `feedback_improvements_become_
default`. Operational wins also confirmed: uniform per-tranche wall (no heavy
tail), recipe purity held across all breed rounds on real weights.

---

## Phase 5b — real-training smoke of `--breeding gauntlet`  [2026-06-13]

**Intention.** Before the Phase 6 A/B, validate the LIVE training path
(predictors ON, shared static_obs cache, multiprocess workers) end-to-end — the
real agent-1/day-1 audit the stub tests can't cover — and surface any wiring
gaps cheaply.

**Implementation.** Small real run: `--breeding gauntlet --n-agents 4
--parallel-agents 4 --use-race-outcome-predictor --predictor-bundle-manifests …
--seed-gene architecture=lstm --seed-gene hidden_size=64 --seed-gene
predictor_lean_obs=true --holdout-recent 2 --validation-holdout-recent 2`
(lean obs h64 for speed). Monitored the log; inspected the ledger + sidecars +
scoreboard live.

**Result — the real path works; three wiring gaps found + fixed:**
- ✅ **Shared static_obs cache** bakes per day on the real path (lean variant,
  ~7–19 MB/day) and workers consume it — Phase-0 design intact end-to-end.
- ✅ **Recipe purity on REAL weights:** tranche-1 trained 4 agents (multiprocess),
  wrote real weights + `.genehash` sidecars, ledger advanced to depth 1, and
  **tranche-2 warm-started each agent's OWN K-1 weights with NO
  `RecipePurityError`** — the load-bearing invariant holds on real artifacts.
- ✅ Ledger live state correct: 4 fresh lineages @depth 1, fixed validation set
  `[2026-06-02, 2026-06-03]`, frontier_depth 1, 4 sidecars.
- 🔧 **Gap 1 (fixed): `--seed-gene` rejected gauntlet.** `_resolve_seed_bands`
  guarded `breeding in ("pbt","lockstep")`; added `"gauntlet"` (the gauntlet
  seeds fresh blood via `sample_fresh_blood_genes`, the same funnel).
- 🔧 **Gap 2 (fixed): no `scoreboard.jsonl` on the gauntlet path.** The worker
  writes weights/models.db/bet_logs but the per-agent scoreboard row is the
  generation loop's job, which the gauntlet bypasses — so the Phase 6 judge
  (`cross_era_holdout_board`) + `gene_register` would see nothing. Fixed: the
  dispatch's capturing wrapper now appends a `_agent_result_to_scoreboard_row`
  per agent per tranche (+ `lineage_id`/`origin`/`tranche_K`/era tags).
  Unit-tested in `test_v2_gauntlet_runner.py`.
- 🔧 **Gap 3 (cosmetic): `--era-type` choices are tick/tock only** — not a code
  bug; just omit it for gauntlet eras (era_id still tags rows).

Net: `--breeding gauntlet` runs the real predictors-ON multiprocess path with
recipe purity intact and now writes scoreboard rows the A/B needs.

**Confirmed on a second real run (per "just finish the smoke"):** after a real
tranche-1 completed (4 LSTM-h64 lean agents, multiprocess), `scoreboard.jsonl`
held 4 well-formed `v2_cohort_scoreboard` rows — full `hyperparameters`,
`composite_score`, `tranche_K`, `lineage_id`, `origin`, `era_id`,
`architecture_name=v2_discrete_ppo_lstm_h64` (confirming the seeded
architecture=lstm / hidden_size=64 took). `tools.gene_register
--registry registry/gauntlet_smoke` read those rows cleanly (4 configs,
hidden_size all 64) — the downstream-tool loop is closed. Stopped before the
full 4-tranche climb + breed round (breeding is exhaustively unit-tested;
grinding the hour adds no validation). **GPU-policy-lane wiring verified intact**
through the gauntlet path (env+LSTM/small-tx CPU; ctx≥128 transformers → CUDA
under `--gpu-policy-lane`); A/B scripts updated to enable it on both arms.

Phase 5b DONE. The gauntlet pipeline is built, unit-tested (48 new tests), and
real-path-validated. Phase 6 (the A/B cutover gate) is ready to launch but
HELD per operator ("just finish the smoke") — scripts + runbook in place.

---

## Phase 5 — orchestrator + `--breeding gauntlet` flag  [2026-06-13]

**Intention.** Wire executor + ledger + breeder into a resumable loop driven by
uniform per-tranche batches, exposed behind `--breeding gauntlet`, with the old
ga/pbt/lockstep paths untouched.

**Implementation.**
- `training_v2/cohort/gauntlet.py` (+ `tests/test_v2_gauntlet.py`, 7 green):
  `run_gauntlet` = seed N fresh recipes → `climb_to_frontier` (each iteration
  runs the shallowest non-empty `needs-T(K)` queue through ONE tranche — uniform
  cost) → `breed_frontier` → repeat for `max_breed_rounds`. `_entry_to_agent`
  threads each lineage's OWN K-1 weights (None at K==1) — recipe purity. Stable
  per-(lineage,tranche) seeds (md5, not salted `hash`) for reproducible resume.
- Runner wiring: `--breeding gauntlet` argparse choice; `run_cohort` validates it
  and **early-dispatches** to `_run_gauntlet_breeding` BEFORE any ga/pbt/lockstep
  setup (early return ⇒ those paths byte-untouched). The helper builds the
  `DaySplit` from `make_rotations` (chronological, eval days folded into train
  since selection is on the FIXED fc=0 validation set), assembles
  `TrancheExecConfig`/`GauntletConfig`, runs the gauntlet on a warm pool, and
  returns the frontier recipes' AgentResults (best-locked first). `_pbt_config`
  + the validation-holdout warning now include gauntlet (one-source-of-truth).

**Result.** `--breeding gauntlet` is a live mode (parser + dispatch verified).
`tests/test_v2_gauntlet_runner.py` (2) drives the REAL executor+ledger+breeder
through the runner dispatch with a stub trainer: ledger written, frontier at full
depth, fixed validation reached every eval, recipe-purity sidecar chain real
across tranches, results returned best-locked-first. Full suite **102 green**
(incl. existing cohort_runner + lockstep) — no regression. Requires
`--validation-holdout-recent > 0`. Next: a small REAL-training smoke (predictors
ON, shared cache) as the live agent-1/day-1 audit, then the Phase 6 A/B.

---

## Phase 4 — breeder (selection + mutation)  [2026-06-13]

**Intention.** The ONLY selection stage. Read the same-depth frontier,
truncation-select on the fc=0 composite (with the σ_naked_leg ≤ £30 ceiling),
keep the top fraction, cull the rest, emit mutant + fresh replacements into
needs-T1 — never culling a climber mid-climb (full fair shot).

**Implementation.** `training_v2/cohort/breeder.py` (+ `tests/test_v2_breeder.py`,
9 green). `breed_frontier(ledger, rng, *, cfg, score_fn=None, sigma_leg_fn=None)`:
- Operates on `ledger.frontier()` only (same depth ⇒ same-gauntlet comparison).
  Below `min_quorum` → no-op. Ranks on the fc=0 composite (default = the
  validation score the orchestrator records as locked).
- σ_leg ceiling is a **pluggable pre-filter** (`sigma_leg_fn`, per-leg σ from bet
  logs, wired by the orchestrator) — filters before ranking; falls back to
  unfiltered if it would empty the pool (never deadlocks the gauntlet).
- Survivors stay active (advance when the next tranche appends); culled →
  status "culled"; one replacement per culled slot via `make_offspring`
  (mutant, carries parent lineage) or `sample_fresh_blood_genes` (fresh,
  register-driven later), appended at depth 0 (needs-T1). Deterministic given rng.

**Result.** 9/9 tests: quorum skip, top-fraction keep/cull, replacements land in
needs-T1 at depth 0, mutants carry parent lineage, fresh blood when
mutant_fraction=0, σ_leg ceiling filters the blown-variance leader,
legacy-key-tolerant `genes_from_dict`, determinism, and the **full-fair-shot**
guard (shallow climber untouched). Old lockstep `select_lockstep` untouched.

---

## Phase 3 — ledger / queues (persistent state)  [2026-06-13]

**Intention.** A durable, resumable record of every lineage climbing the
gauntlet, from which the `needs-T(K)` queues + the frontier pool are derived.
The ledger IS the resume state (and what a second machine reads to grab work).

**Implementation.** `training_v2/cohort/ledger.py` (+ `tests/test_v2_ledger.py`,
11 green).
- `GauntletLedger`: one JSONL file, append-a-full-snapshot per update,
  last-per-`lineage_id` wins on `load()`; `compact()` atomically rewrites
  (tmp + `os.replace`). Crash-safe: a half-written trailing line is skipped.
- A **lineage is the climber identity** — survivors keep their lineage across
  tranches (genes fixed = recipe purity); each `record_tranche(lineage, K, …)`
  updates the same entry and enforces K == completed+1 (no skipping).
- Queues are derived, not stored: `needs(K)` = active lineages with
  `tranches_completed == K-1`; `frontier()` = active lineages at the deepest
  depth (the breeder's same-depth input). `add_recipe()` lands a new recipe in
  needs-T1; `set_status("culled")` drops it from active queues. NO selection here.
- `DaySplit` (ordered tranches + fixed validation + sealed final-test) +
  `assert_day_split_disjoint` carry the `holdout-selection.md` leakage guard.

**Result.** Ledger lands as a standalone module; nothing existing imports it yet
(wired in Phase 5). 11/11 tests: queue advance on tranche completion, skip
rejection, cull leaves active queues, frontier is same-depth-only, resume
round-trip, compaction preserves state + shrinks the log, leakage asserts.

---

## Phase 2 — executor primitive (`run_tranche`)  [2026-06-13]

**Intention.** A pure batch executor — train a batch through ONE tranche, eval
on the fixed validation set at fc=0, return scores — with NO selection inside
(selection lives only in the breeder). Uniform per-run cost; recipe purity
enforced at the boundary.

**Implementation.** `training_v2/cohort/executor.py` (+ `tests/test_v2_executor.py`,
9 green).
- `run_tranche(agents, *, tranche_K, train_days_for_K, validation_days, cfg)` →
  `list[TrancheResult]`. `RecipeAgent` carries (genes, lineage_id, origin,
  init_weights_path, seed); `TrancheExecConfig` carries the cohort-wide context
  (predictors, device, reward/gene wiring, BC overrides, gate flags). Defaults =
  predictors-ON fast path (`--parallel-agents 16 --device cpu`, never `--batched`).
- **Gauntlet model (vs lockstep):** all agents in a batch are at the same depth
  K, train on ONE tranche's days (same set for all), and warm-start their OWN
  K-1 weights — **no catch-up replay**. So every run is `batch × 1 tranche`,
  uniform regardless of depth. Mutants/fresh enter at K==1 from scratch and climb.
- **Recipe purity** asserted BEFORE training: structural (K==1 ⟺ no weights) +
  a `.genehash` sidecar (`config_hash` + `lineage_id`) written next to every
  saved checkpoint and re-checked on warm-start — a cross-recipe inherit raises
  `RecipePurityError` (the chimera `purpose.md` forbids).
- Reuses the worker verbatim: `_build_spec` emits the SAME `train_one_agent`
  spec dict the runner's multiprocess path builds (shared static_obs cache,
  predictor-by-manifest, memory-budget guard, size-aware threading). A
  spec-key-vs-`train_one_agent`-signature test guards drift. Sequential path
  (parallel_agents=0) for tests + single-box debugging.

**Result.** Executor lands behind the new module; old `--breeding lockstep`
untouched (not imported, not modified). 9/9 tests green: structural + sidecar
recipe-purity, leakage refusal, no-selection ordering (best-locked agent stays
in input order, not ranked), validation extraction, spec-key drift guard. Not
yet wired into a runner flag — that's Phase 5 (orchestrator).

---

## Phase 0 — single-worker memory profile  [2026-06-13]

**Intention.** Put real numbers on the ~10 GB per-worker working set and decide
whether the gauntlet's per-run footprint is shared (cheap — one physical copy,
day-growth scales it once) or private (scales ×workers). Confirm the shared
static_obs memmap cache is not regressed. Hypothesis: most of the WS is shared
memmap pages RSS double-counts + the per-worker predictor bundle.

**Implementation.** `tools/profile_worker_memory.py` — faithfully replays the
worker's load sequence (thread-pin → torch import → `PredictorBundle.from_
manifests` → memmapped `DayStaticObs` per day → real `train_one_agent` on 1 train
+ 1 eval day with the predictors-ON `static_obs_cache` injected). Measures
**RSS vs USS** at each stage (USS = private pages = the true marginal cost of one
more worker; RSS counts shared file-backed/library pages every worker shares).
Peak sampled in a background thread; Python allocs via tracemalloc. Run against
the real `tt_tock_004` full-obs cache (2050-d, both predictors ON, max_runners
14). Report: `plans/gauntlet-pipeline/phase0_mem_profile.txt`.

**Result.** Per-worker peak **1.93 GB RSS / 1.61 GB USS** — an order of magnitude
under the ~10 GB hypothesis. Stage USS deltas:

| stage | RSS GB | USS GB | ΔUSS |
|---|---|---|---|
| baseline | 0.02 | 0.01 | — |
| + torch/numpy/worker import | 0.75 | 0.55 | +0.54 |
| + predictor bundle | 1.03 | 0.80 | **+0.25** |
| + mmap 2 days (81 MB disk) | 1.03 | 0.80 | ~0 |
| + train_one_agent (1058 s) | 1.68 | 1.36 | +0.56 |
| peak during train | 1.93 | 1.61 | — |

Conclusions:
- **Hypothesis confirmed.** Private (USS) per-worker cost is ~1.6 GB; the
  predictor bundle is only **~0.25 GB** of it. The "~10 GB WS" is RSS
  **double-counting the shared memmap + shared libs** across workers — the
  RSS−USS gap. True multi-worker RAM ≈ N×USS + ONE shared cache copy, which
  matches the recorded "N=8 ≈ 30 GB" (the shared-memory-day-cache memory note),
  not N×10 GB.
- **Shared day cache NOT regressed.** mmap-loading 2 days added ~0 USS — pages
  are file-backed and shared (one physical copy), exactly the design intent.
  Day-growth scales the SHARED copy once, never ×workers.
- **Caveat on the absolute memmap number:** this run mmapped only 2 days and
  touched the first row, so the *resident* shared-memmap pages are understated
  vs a real cohort that faults in all training-day ticks (still file-backed,
  still one shared copy — USS-invariant, so the per-worker conclusion holds).
- **Decision — sharing the predictor across workers: LOW priority.** At ~0.25
  GB/worker it's a worker-COUNT cost (~4 GB at N=16), not a day-growth cost, and
  the box is no longer RAM-bound since the shared day cache landed. The cheaper,
  lower-risk win if RAM ever bites at high N: **lazy-load the bundle only on a
  cache MISS** — the static_obs cache already bakes the predictor columns in, so
  fully-cached days never call inference and don't need the bundle resident. A
  cross-process shared-inference server is not worth the IPC complexity for ~4 GB.
- Side note: 1058 s for 1 train + 1 eval day single-threaded full-obs (2050-d,
  h256) — that's the expected per-agent compute cost, not a memory issue.

---

## Phase 1 — gene register v1 (read-only coverage map)  [2026-06-13]

**Intention.** Build the read-only foundation for Phase 7's gap-targeted
fresh-blood sampler: load every persisted gene config across all eras and print
a per-gene coverage map (visited cells + held-out outcome, blank + thin
regions). Useful immediately for picking the next Tock's seed bands.

**Implementation.** `tools/gene_register.py` (+ `tests/test_gene_register.py`,
10 tests green). Read-only; writes only its own report via `--output`.
- Sources scanned under `registry/`: `**/scoreboard.jsonl`
  (`v2_cohort_scoreboard` — full gene config in `hyperparameters` + in-sample
  `eval_locked_pnl`/`eval_naked_pnl`/`composite_score`); `**/model_register.csv`
  (`gene_*` cols reconstruct a config + in-sample `naked_std`);
  `**/*reeval*.jsonl` + `**/tt_*_fc*.jsonl` (`v2_cohort_reeval` — HELD-OUT
  `reeval_locked_pnl`/`reeval_naked_pnl`/maturation); and
  `registry/cross_era_holdout_board.jsonl` (per-leg `ho_sigma_leg`, joined by
  8-char model prefix).
- The unit of a "visited cell" is a distinct **gene config** (canonical hash of
  the full gene dict, floats rounded to 9 s.f.), NOT a model_id — survivors
  carry the same genes across tranches, so model-count would over-count.
- Gene ranges/choices imported from `training_v2.cohort.genes` (stays in sync
  with the sampler); genes with no declared spec fall back to observed-range
  binning and are flagged `[OBSERVED-RANGE]` (this surfaces dead legacy genes
  like `arb_spread_scale` from old eras).
- Per gene: log/linear/choice/bool bins; per bin: #configs, #with-holdout, mean
  held-out locked, mean held-out σ_leg, mean in-sample locked. Plus a BLANK-cell
  list and a PROMISING-BUT-THIN list. Machine-readable `.json` companion.

**Result.** Loaded **2641 scoreboard + 72 reeval + 692 csv rows + 16 cross-era
σ joins → 2865 distinct models → 1214 distinct gene configs**, 53 genes mapped.
Report at `registry/gene_register.{txt,json}`. Sanity (master_todo): all legacy-7
genes present; config-hash de-dup strictly reduces 2865 models → 1214 configs.

Signal worth carrying into Phase 7 / next-Tock seeding:
- **Three direction-label genes are completely unexplored off their default.**
  `direction_horizon_ticks` (only 60 seen), `direction_threshold_ticks` (only
  5), `direction_force_close_seconds` (only ~60) are single-valued across all
  1214 configs — every other bin BLANK. These were promoted to genes 2026-06-06
  but never enabled (they also need a pre-scanned direction-label cache per
  distinct triple, so sampling them isn't free).
- **Held-out σ_leg coverage is sparse** — only the 16-row cross-era board
  carries it, so `ho_sig` ≈ 22 nearly everywhere (one era's models). Widening
  σ_leg coverage needs more `tools/cross_era_holdout_board.py` runs; flagged as
  a v2 input gap, not a tool bug.
- **`bc_pretrain_steps=500` and `direction_gate_enabled=True`** both show higher
  mean held-out locked than their off counterparts (21.3 vs 10.6; 22.4 vs 13.2),
  though confounded with era. In-sample locked is dominated by a few large-magnitude
  old-era rows (the ~88–127 means) — read held-out columns, not in-sample, for
  selection signal.

---
