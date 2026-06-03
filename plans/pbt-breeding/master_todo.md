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

## Step 2 — 3-way breed step  `[ ]`
New `_breed_next_generation_pbt`: elites preserve (weights+recipe, continue
training), offspring exploit (inherit top brain + ±20% recipe), immigrants
explore (inherit-brain + bold recipe, plus the protected from-scratch quota
with K-gen cull-immunity). Behind `--breeding pbt`; default path untouched.
GATE: unit tests on the partition (counts, immunity bookkeeping, parent links)
+ default-off byte-identity (HC#1).

## Step 3 — Day rotation  `[ ]`
Per-gen training-day sampling from a larger pool + rotating iteration-eval,
sealed test excluded. Deterministic per (cohort_seed, gen) so the A/B is paired.
GATE: no sealed-test leak (assert); paired determinism across a re-run.

## Step 4 — Heritability + diversity instrumentation  `[ ]`
Per-gen metrics: champion-performance reproduction across gens, selection
spread÷signal, recipe + behavioural diversity, immigrant survival rate. Written
to the scoreboard/an analysis JSONL.
GATE: the metrics exist and are readable from a short run.

## Step 5 — A/B run + held-out validation  `[ ]`
PBT vs gene-only GA, same seed + day pool, both judged on the sealed days.
DELIVERABLE: `findings.md` — heritability, selection-noise, held-out
locked_per_std, diversity, wall-clock parity.
GATE: purpose.md success criteria (a)-(d) evaluated honestly (HC#7).

---

## Cross-cutting
- Reuse the held-out re-eval (`tools/reevaluate_cohort.py`, now input_norm-fixed)
  for the sealed-day verdict.
- Record in `plans/EXPERIMENTS.md`; update the GA section of CLAUDE.md if it lands.
