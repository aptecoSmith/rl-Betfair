# pbt-breeding — master todo

Legend: `[ ]` todo · `[~]` in progress · `[x]` done. Each step has a GATE.

---

## Step 0 — Design session: settle the open decisions  `[x]`
**DONE 2026-06-03 — see `design.md` (operator-signed-off mechanism).**
Resolved: **promotion-ladder** (compete within rotation tier, NOT single
leaderboard); per-tier composition R1=100% fresh blood, R2+=50% graduated
elites + 50% offspring; **rotation gauntlet** (winning earns the lowest unseen
rotation; R3 winners freeze to the hall-of-fame leaderboard, thaw on rotation
4); offspring=(a) copy-one-winner + perturb ±20% (pluggable for (b) later);
**no monoculture cap** (open system + equal-training-within-tier self-corrects;
measure lineage share as an observable; eval-reliability is the real lever);
day split 3×10 (6 train / 4 eval), random i.i.d. splits, ~12 spare days reserved
for rotation 4; **execution = the existing multiprocess pool running the full
mixed-rotation population each gen (no within-gen barrier) + static_obs cache +
warm-start load.** Remaining knobs (episodes/gen under warm-start, per-gene
perturbation form) are Step-1 implementation details, not blockers.

## Step 1 — Weight-threading infrastructure  `[x]`
**DONE 2026-06-03 — see `findings.md` "Step 1".** Worker WARM-START load
landed: `train_one_agent(init_weights_path=...)` + the single
`load_warm_start_weights(policy, path)` load path (HC#11); warm-started
agents skip BC; threads through the multiprocess pool automatically (it's
a picklable spec-dict kwarg). GATE PASSED: `tests/test_v2_pbt_warm_start.py`
(5 tests) proves a warm-started child's gen-0 forward is BIT-identical to
the parent's on a fixed obs before any new training (HC#5), incl. the
`input_norm` buffers; strict load raises on a structural mismatch (HC#10);
default-off byte-identity holds (HC#1, 38 worker+genes tests still pass).
TAIL (parent→child weight COPY + checkpoint weight-pointers) folds into
Step 2/3 — it's just pointing the offspring's `init_weights_path` at the
parent's existing `registry/weights/<model_id>.pt` (no physical copy) and
riding the lineage/rotation state. STOPPED here for operator review (the
brief's first mandatory stop-point).

## Step 1b — Architecture genes + v2 transformer + policy factory  `[x]`
**DONE 2026-06-03 — see `findings.md` "Step 1b".** 4 structural genes
(`architecture` + transformer `depth`/`heads`/`ctx_ticks`) on `CohortGenes`,
frozen within a lineage; `sample_fresh_blood_genes` draws them (HC#9), base
`sample_genes`/crossover/mutate keep them at the LSTM default with NO rng draw
(HC#1 byte-identity). `DiscreteTransformerPolicy` ports v1's encoder onto
`BaseDiscretePolicy`, subclassing the LSTM to reuse its head stack (LSTM left
untouched). `agents_v2/policy_factory.py::build_policy` is the single
constructor; worker + reeval both call it (reeval now also threads
`runner_dim`). GATE PASSED: factory builds + forward for every arch/sizing;
transformer checkpoint round-trips strict; **transformer trains end-to-end
through the real v2 PPO trainer** (buffer hidden state survives); warm-start
composes with both arches; LSTM factory path byte-identical; fresh blood
covers both arches. 31 new tests + 152 existing v2 tests pass.

## Step 2 — 3-way breed step  `[x]`
**DONE 2026-06-03 — `training_v2/cohort/pbt.py` + runner `--breeding pbt`.**
Promotion-ladder (the design's refinement of the purpose.md 3-way split):
`init_pbt_population` (gen-0 fresh blood) + `breed_pbt` (R1 fresh; R2+ = 50%
promoted elites with weights intact + 50% offspring bred from them via
`make_offspring` = copy-one + perturb non-structural ±20%, structural frozen
HC#10; R3 winners FREEZE to a hall-of-fame; R1 absorbs transient slack so each
gen = exactly n_agents). Behind `--breeding pbt`; GA path byte-identical
(HC#1). GATE PASSED: `test_v2_pbt_ladder.py` (partition/counts/freeze/lineage/
parent-links ×13) + `test_v2_pbt_runner.py` (stub-driven run_cohort wiring) +
28 GA runner tests still pass.

## Step 3 — Day rotation  `[x]`
**DONE 2026-06-03 — `pbt.make_rotations` + runner per-tier day threading.**
3 random equal i.i.d. folds (6 train / 4 eval), deterministic in cohort_seed
(paired A/B). Each agent trains on ITS TIER's rotation; `gen_days` = union over
tiers. GATE PASSED: disjoint folds, deterministic across re-run, no sealed leak
(`test_v2_pbt_ladder.py::TestMakeRotations`); the runner test confirms R2
agents train on rotation 2, disjoint from R1's rotation 1.

## Step 4 — Heritability + diversity instrumentation  `[x]`
**DONE 2026-06-03 — per-gen `pbt_lineage.jsonl` + `tools/analyze_pbt.py`.**
The runner writes per-agent lineage/tier/role/score rows each gen; the tool
computes heritability (lineage score gen→gen+1 ρ), selection spread÷signal,
lineage diversity (monoculture observable), fresh-blood survival, and the
architecture leaderboard, with an optional `--ga scoreboard.jsonl` side-by-side.
GATE PASSED: the runner test asserts the lineage rows; the tool runs on a run's
JSONL.

## Step 5 — A/B run + held-out validation  `[~]`
**RUNNING 2026-06-03 (operator pivot to a long autonomous campaign).** Instead
of the short paired A/B, the operator (away ~18-20h) asked for a CONTINUOUS PBT
run accumulating a rich R3 leaderboard + a per-model register. Launched
`plans/pbt-breeding/_scripts/run_pbt_long.ps1` → `registry/pbt_long/`: wrapper
loops to a ~20h deadline, 16 agents × 25 gens/run, relaunch-on-exit with a new
seed (resets pool memory + explores new fresh blood), one dir so the
hall-of-fame + register accumulate. Live `leaderboard.txt` (R3 champions,
usual columns + `frozen_at`) + `model_register.csv` regenerate every gen. A
persistent Monitor heartbeats + alerts on stall/OOM. On return: sealed-day
re-eval of the top champions + `analyze_pbt` → `findings.md` +
`plans/EXPERIMENTS.md`. The paired A/B harness (`run_ab.ps1`) remains for a
later head-to-head verdict.
DELIVERABLE: leaderboard.txt + model_register.csv + (on return) held-out
locked verdict, heritability, selection-noise, diversity.

---

## Cross-cutting
- Reuse the held-out re-eval (`tools/reevaluate_cohort.py`, now input_norm-fixed)
  for the sealed-day verdict.
- Record in `plans/EXPERIMENTS.md`; update the GA section of CLAUDE.md if it lands.
