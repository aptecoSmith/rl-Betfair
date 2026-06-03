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

## Step 1 — Weight-threading infrastructure  `[ ]`
Save each agent's final weights per gen (registry `weights/` already does);
add a parent→child weight COPY and a worker WARM-START load (train_one_agent
gains an optional `init_weights_path`). Extend the resume/checkpoint to carry
the weight pointers (it is gene-only today — the biggest change).
GATE: a child built with `init_weights_path=parent` reproduces the parent's
forward on a fixed obs BEFORE any new training (HC#5).

## Step 1b — Architecture genes + v2 transformer + policy factory  `[ ]`
Add the structural genes (`architecture` ∈ {lstm, transformer}; transformer
`depth`/`n_heads`/`ctx_ticks`; keep `hidden_size`) to `CohortGenes`. Port v1's
`PPOTransformerPolicy` → a v2 `DiscreteTransformerPolicy` on
`BaseDiscretePolicy`. Add a **policy factory** `build_policy(genes)` that the
worker + reeval tool both call (one source of truth for genome→architecture).
Structural genes are sampled ONLY at fresh-blood birth and frozen within a
lineage (so warm-start weight inheritance always sees matching shapes).
GATE: factory builds + a forward runs for every architecture/sizing in range;
a transformer checkpoint round-trips (save→load, strict); fresh-blood sampling
covers the full gene space (no `enabled_set` restriction).

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
