# Session prompt — execute `pbt-breeding`

You're replacing the cohort GA's gene-only breeding with a **Population-Based-
Training-style promotion ladder** so a champion's learned *identity* (weights),
not just its hyperparameters, is heritable. The mechanism is already DESIGNED
and operator-signed-off — your job is to build it, not re-litigate it.

## Read first (in order)
1. `purpose.md` — the why + the measured failure (gene-only GA re-trains every
   agent from scratch each gen with a fresh seed; champions don't reproduce,
   selection is ~half noise — evidence in `registry/smdc_bcon_1780436088`).
2. `design.md` — **THE settled mechanism.** Every decision is made here. Treat
   it as the spec.
3. `hard_constraints.md` — inviolable.
4. `master_todo.md` — Step 0 is DONE; start at **Step 1**.

## The mechanism in one breath
Promotion ladder: fresh blood enters **rotation 1** (a pure rookie division);
**winning earns the next unseen rotation** (3 random equal day-folds, 6 train /
4 eval each); each promoted tier = **50% preserved-elite + 50% offspring**
(offspring = copy ONE winner's weights + perturb its recipe ±20%); **R3 winners
FREEZE** to a hall-of-fame leaderboard and thaw when a rotation 4 (new data)
arrives. **No diversity cap** — measure lineage share as an observable, don't
intervene; eval-reliability (4 eval days/rotation) is the protector. Execution:
the EXISTING multiprocess pool runs the **whole mixed-rotation population every
gen in parallel** (no within-gen barrier — each agent warm-starts from last
gen's on-disk weights); per-tier ranking + promotion happen between gens.

**Gene space + architecture tournament (design.md):** fresh blood samples the
FULL gene space (NO `enabled_set` restriction) INCLUDING an **architecture
gene** (LSTM vs transformer, sizes, transformer depth/heads/ctx). The
hall-of-fame records each champion's architecture → the gauntlet *reports which
architecture wins*. **Structural genes (arch + sizing) freeze within a
lineage** (warm-start can't cross weight shapes); only non-structural genes are
inherited + perturbed. NEW BUILD (Step 1b): port v1 `PPOTransformerPolicy` →
v2 `DiscreteTransformerPolicy`, add arch genes, add a single
`build_policy(genes)` factory used by BOTH worker and reeval tool (HC#11).

## State of the repo (important)
- Builds ON the shared-memory-day-cache infra (multiprocess pool +
  `static_obs` memmap cache + warm pool). Those changes — plus the
  `tools/reevaluate_cohort.py` input_norm/mature-prob fixes — are **uncommitted
  in the working tree**. pbt-breeding sits on top of them. (Confirm with
  `git status` before starting; consider committing the shared-memory work
  first if the operator agrees.)

## Codebase touchpoints
- `training_v2/cohort/runner.py::_breed_next_generation` — fork a
  `--breeding pbt` path: per-tier rank → promote winners (next unseen rotation)
  → spawn offspring → refill R1 fresh blood → freeze R3 winners. Default path
  untouched (HC#1).
- `training_v2/cohort/worker.py::train_one_agent` — add `init_weights_path`
  warm-start (load parent/own weights before BC+PPO). Registry `weights/`
  already stores per-agent weights.
- `training_v2/cohort/genes.py` — pluggable `_make_offspring` = copy-one +
  perturb ±20% (so the future (b) two-winner recipe-crossover slots in behind a
  flag).
- `training_v2/cohort/multiproc_worker.py::prebuild_static_obs_cache` — bake
  each rotation's days once; that rotation's agents share its day memmaps.
- `registry/model_store.py` — per-agent weights + lineage/leaderboard state
  (track each lineage's `rotations_seen`).

## Discipline / gates
- **A/B vs the current gene-only GA**, same cohort seed + same day pool, judged
  ONLY on the sealed **May 20–29** days via `tools/reevaluate_cohort.py`
  (now input_norm-aware; pass `--mature-prob-open-threshold` to match training).
- `--breeding pbt` default OFF = byte-identical (gate it).
- Step-1 gate: a warm-started child's gen-0 forward == the parent's final
  forward on a fixed obs BEFORE any new training (HC#5 — inheritance is real).
- Measure + report honestly (HC#6/#7): heritability (do champions reproduce?),
  selection spread÷signal, lineage diversity, fresh-blood survival rate,
  per-gen wall-clock parity. Don't declare success on held-out cash alone.

## Stop-and-report
After Step 1's forward-match gate; before Step 5's A/B burns real compute;
on any byte-identity (default-off) failure.

## Done means
A/B shows PBT champions reproduce across gens, selection spread shrinks vs
signal, held-out `locked_per_std` beats gene-only, diversity doesn't collapse,
wall-clock at parity — recorded in `findings.md` + `plans/EXPERIMENTS.md`, with
the CLAUDE.md GA note updated if it lands.
