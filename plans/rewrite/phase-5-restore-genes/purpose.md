---
plan: rewrite/phase-5-restore-genes
status: GREEN
opened: 2026-05-03
session_03_completed: 2026-05-03
depends_on: rewrite/phase-3-followups/force-close-architecture
            (5-stack mechanism layer GREEN, Bar 6c open),
            rewrite/phase-4-restore-speed (per-tick perf restored)
---

# Phase 5 — restore-genes: wire the per-agent search dimensions the design always intended

## Purpose

The v2 cohort GA evolves only **7 genes**:

```
learning_rate, entropy_coeff, clip_range, gae_lambda,
value_coeff, mini_batch_size, hidden_size
```

Across the rewrite's plans, **11 additional knobs were explicitly
designed as per-agent genes** but never landed in `CohortGenes`.
Several of them have plan-level master_todo entries that say
"promote to gene" and were skipped or deferred. The result is a GA
that's been searching only the PPO/architecture subspace while
all the reward-shape, mechanism-strength, and aux-head loss
weights are fixed cohort-wide.

The 5-stack `force-close-architecture` cohort exposed why this
matters: with `open_cost = 1.0` applied uniformly to all 12
agents, GA had no way to find per-agent working points. Bet count
barely budged (4% drop), and 0/9 agents cleared positive eval
P&L. **A cohort-wide flag answers "is this value good for the
average agent?" — never "what's the right value for THIS gene
combination?"**

## What this phase does

1. **Promote 11 cohort-level knobs to per-agent `CohortGenes`
   fields** with documented ranges and sampling distributions:

   - **Tier 1** (designed-as-gene, Bar-6c-relevant):
     `open_cost`, `matured_arb_bonus_weight`,
     `mark_to_market_weight`
   - **Tier 2** (proven-relevant for scalping):
     `naked_loss_scale`, `stop_loss_pnl_threshold`,
     `arb_spread_scale`
   - **Tier 3** (aux head loss intensities, designed-as-gene):
     `fill_prob_loss_weight`, `mature_prob_loss_weight`,
     `risk_loss_weight`
   - **Tier 4** (PPO/training stability, designed-as-gene):
     `alpha_lr`, `reward_clip`

2. **Add a `--enable-gene NAME` repeatable CLI flag** so the
   operator decides which genes evolve per cohort. Disabled genes
   stay frozen at a sensible default for the whole cohort
   (matching the existing cohort-wide behaviour byte-identically).

3. **Plumb per-agent gene values through to the env / trainer**
   so each agent's reward_overrides reflect its own gene draws.

The result: `CohortGenes` grows 7 → 18 fields. With per-gene
switches, an operator can:

- Run today's 5-stack cohort verbatim (no `--enable-gene` flags
  → all new genes disabled → byte-identical to pre-plan).
- Add one new gene per run (`--enable-gene open_cost`) to
  measure incremental contribution.
- Run a wider exploration (e.g. 20 agents × 6 generations,
  all 11 new genes enabled) to let GA find the working point
  across the full space.

## Why this is its own phase

`phase-3-followups/force-close-architecture` ships GREEN on the
mechanism layer (Bar 6a cleared). What remains is the
**policy-shape question**: does the rewrite's stack converge to
v1-equivalent quality given equivalent per-agent search? That's
not a mechanism question — it's a search-dimension question.
The fix lives in `training_v2/cohort/genes.py` and the runner /
worker, not in any of the mechanic plans.

`phase-4-restore-speed` makes wider-and-longer cohort runs
affordable (per-tick wall down toward v1's 3 ms territory). The
genes need to exist for that compute budget to mean anything.

Together, Phase 4 (per-tick speed) + Phase 5 (per-agent search
dims) are the two structural prerequisites for re-asking the v1
catch-up question on a level playing field.

## Why not just promote everything?

The env exposes ~25 reward_overrides keys. Most of them aren't
gene-promotion candidates because they're either:

- Cohort-level operator policy (`force_close_before_off_seconds`,
  `min_seconds_before_off`, `lay_only_naked_price_threshold`)
- Architectural booleans (`target_pnl_pair_sizing_enabled`)
- Fixed external constraints (`commission`)
- Directional-mode-only and inert in scalping
  (`early_pick_bonus_min/max`, `efficiency_penalty`,
  `precision_bonus`, `drawdown_shaping_weight`,
  `spread_cost_weight`, `inactivity_penalty`,
  `terminal_bonus_weight`, `early_lock_bonus_weight`,
  `naked_penalty_weight`)

The 11 promoted in this plan are the ones whose own plan
documents called them genes, OR where per-agent variation is
clearly principled (different agents have different optimal
risk tolerance / scalping aggressiveness / etc.).

If a future plan exposes a tunable that fits the gene criterion
(per-agent variation makes principled sense, mechanism is
independent across agents), promotion goes the same way: a row
in `_sample_field`, a range constant, an `--enable-gene` switch
becomes available automatically.

## What's locked

### Backwards compatibility for runs without `--enable-gene` flags

A cohort launched without any `--enable-gene` flags must produce
**byte-identical** results to a pre-plan run with the same seed.
The 11 new genes are fields on `CohortGenes` but, when their
switch is off, they get a fixed default value (matching today's
cohort-wide reward_overrides default — typically `0.0` or the
plan's pre-existing baseline). Sample / mutate / crossover skip
disabled genes entirely.

This is the load-bearing correctness invariant: existing
cohort runners must not silently change behaviour when this
phase lands.

### CUDA↔CUDA self-parity holds

Two CUDA runs at the same seed with the same `--enable-gene`
flags produce bit-identical `total_reward` and `value_loss_mean`
per agent — same shape as Phase 3 Session 01b's load-bearing
parity bar. Different `--enable-gene` flags produce DIFFERENT
results (that's the whole point), but at fixed flag-set the
result is reproducible.

### Same `--seed 42` for all phase-bar cohorts

Per-gene sampling is seeded from the agent's per-agent seed
(`seed × 1_000_003 + generation × 10_000 + i`) so the GA's
draws are deterministic. Comparing cohorts across plans
requires holding seed AND `--enable-gene` flag set constant.

### Schema growth, not break

`CohortGenes` adds 11 fields. Existing serialised genes (registry
rows, scoreboard JSONL) don't have these fields; readers must
default-tolerate. Forward path: future cohorts always carry the
new fields; legacy rows read with defaults. No `ALTER TABLE`
needed because `CohortGenes` is serialised as a JSON dict in
the `hyperparameters` column.

### No env edits

The env (`env/betfair_env.py`, `env/bet_manager.py`,
`env/exchange_matcher.py`) is untouched in this phase. All
plumbing happens in `training_v2/cohort/` and `agents_v2/`. The
new genes already exist as reward_overrides keys; this phase
just promotes them from cohort-wide flags to per-agent values
flowing through the same passthrough.

### No new mechanics, no new shaping terms

This phase ships zero new env behaviours. Every gene maps to a
reward_overrides / scalping_overrides key that already exists
and already has a meaningful effect when set. The phase changes
**who decides the value** (operator-cohort-wide vs GA-per-agent)
— nothing else.

## Success bar

The plan ships GREEN iff:

1. **All 11 genes wired in** with sensible ranges, sampling,
   mutation, and crossover. `CohortGenes` has 18 fields.
2. **`--enable-gene NAME` CLI flag works** for any of the 11
   new genes; repeatable; can combine with `--reward-overrides`
   for the cohort-level flags that are NOT gene candidates.
3. **Mutual exclusion guard.** Setting `--reward-overrides
   open_cost=X` AND `--enable-gene open_cost` errors at startup
   ("the operator must pick one source of truth per knob per
   run") rather than silently picking one.
4. **Byte-identity for legacy launches.** A cohort launched
   without `--enable-gene` flags produces results identical to
   a pre-plan run at the same seed.
5. **End-to-end gene flow validated.** A small validation
   cohort (1 agent × 1 day) with each new gene flipped on
   confirms that the agent's per-agent gene value reaches the
   env / trainer correctly.
6. **Test coverage.** Unit tests for sampling, mutation,
   crossover, range bounds, default-tolerance, and CLI parsing.
   Integration test for the gene → env reward_overrides flow.
7. **CUDA self-parity holds** at fixed seed and fixed
   `--enable-gene` flags.

## Sessions

### Session 01 — schema extension + sampling/mutation/crossover

Extend `CohortGenes` with all 11 new fields. Add range constants
to `genes.py`. Extend `_sample_field`, `mutate`, `crossover`,
`assert_in_range`, `to_dict`, `sample_genes` to handle the new
fields conditionally on an `enabled_set: frozenset[str]`
parameter. Disabled genes get a fixed default (matching
pre-plan cohort-wide behaviour); enabled genes are sampled /
mutated / crossed normally.

End-of-session bar:

- `CohortGenes` has 18 fields.
- `sample_genes(rng, enabled_set)` returns a `CohortGenes` whose
  enabled fields are sampled and disabled fields are at default.
- `mutate(genes, rng, mutation_rate, enabled_set)` only
  re-samples enabled fields.
- `crossover(parent_a, parent_b, rng, enabled_set)` only
  inherits enabled fields from parents; disabled fields stay at
  default.
- `assert_in_range` accepts the new fields and validates ranges.
- All existing `test_v2_cohort_genes.py` tests pass unchanged
  (legacy `enabled_set = {original 7}` is the default).
- New tests: per-gene range validation, disabled-gene defaults,
  enabled_set respected by mutate / crossover.

Session prompt: `session_prompts/01_gene_schema_and_breeding.md`.

### Session 02 — CLI plumbing + worker passthrough

Add `--enable-gene NAME` repeatable flag to
`training_v2/cohort/runner.py`. Build the `enabled_set` from
flags. Add mutual-exclusion guard with `--reward-overrides`.
Plumb `enabled_set` through `run_cohort` to `sample_genes` /
`mutate` / `crossover`.

In `training_v2/cohort/worker.py`, extend `train_one_agent` to
convert each agent's gene values into a `reward_overrides` dict
that the env/trainer reads. Genes whose names match
reward_overrides keys flow through directly. Genes that map to
trainer hyperparameters (`alpha_lr`, `reward_clip`) flow
through the trainer's hyperparameter dict.

End-of-session bar:

- `python -m training_v2.cohort.runner --enable-gene open_cost
  ...` runs without error.
- A cohort launched WITHOUT any `--enable-gene` flags produces
  byte-identical results to a pre-plan run at the same seed.
- Per-agent gene values reach the env (verified by reading the
  env's `_open_cost`, `_mark_to_market_weight`, etc. mid-rollout).
- Per-agent gene values reach the trainer where applicable
  (`alpha_lr`, `reward_clip`).
- New tests: CLI parsing, mutual-exclusion guard, enabled_set
  flowing into per-agent reward_overrides.

Session prompt: `session_prompts/02_cli_and_worker_plumbing.md`.

### Session 03 — validation cohort + writeup

A small dry-run cohort to validate end-to-end gene flow.
12 agents × 1 generation × 1 day, each agent's gene values
sampled freely from all 11 new genes' ranges. Inspect the
resulting `scoreboard.jsonl` rows to confirm:

- Each agent's gene draws are reflected in the row.
- The agent's actual env behaviour is consistent with its gene
  values (e.g. agents with high `open_cost` show fewer / smaller
  bets; agents with high `matured_arb_bonus_weight` show
  positive shaped contribution).

Update `findings.md` with:

- Schema-extension table (11 new genes + ranges).
- Validation cohort scoreboard summary.
- CLI usage examples for incremental and everything-on
  cohorts.
- Notes on dimension-explosion: 18-gene GA needs more agents
  per generation OR more generations to explore meaningfully.

End-of-session bar: validation run ships, findings.md
populated, plan marked **status: GREEN** in the frontmatter.

Session prompt: `session_prompts/03_validation_and_writeup.md`.

## Hard constraints

In addition to all rewrite hard constraints
(`plans/rewrite/README.md`), phase-3-cohort hard constraints,
and inherited from phase-3-followups:

1. **No env edits.** All plumbing in `training_v2/cohort/` +
   `agents_v2/`. `env/betfair_env.py`, `env/bet_manager.py`,
   `env/exchange_matcher.py` are off-limits.
2. **No new mechanics, no new shaping terms.** Genes promoted
   from existing reward_overrides keys; nothing new at the env
   layer.
3. **No removal of existing genes.** The 7 PPO/arch genes stay
   evolvable; no field renames; no shape changes.
4. **Backwards compatibility for legacy launches.** A run
   without any `--enable-gene` flags is byte-identical at fixed
   seed to a pre-plan run.
5. **Mutual exclusion between CLI override and gene switch.**
   Setting `--reward-overrides X=Y` AND `--enable-gene X`
   errors at startup. Operator picks one source of truth per
   knob per run.
6. **CUDA↔CUDA self-parity is the load-bearing guard.** Two
   CUDA runs at the same seed and same `--enable-gene` flags
   produce bit-identical training metrics.
7. **Schema is forward-only.** `CohortGenes` adds fields,
   never removes or renames. Existing serialised genes
   (registry rows, scoreboard JSONL) are read with default-
   tolerance for missing new fields.
8. **Same `--seed 42`** for any cross-cohort comparison cohort.
9. **NEW output dirs for every cohort run.** Don't overwrite
   AMBER v2, force-close-architecture, or any baseline.
10. **No GA gene additions outside the 11 named.** If a future
    plan wants to promote a 12th gene, that's a separate plan.
    This phase ships exactly what's listed.

## Out of scope

- Multi-GPU coordination / distributed cohorts (Phase 6+).
- Frontend / UI changes (the v1 UI continues to read what it
  reads; new gene fields surface in the scoreboard's
  `hyperparameters` JSON dict).
- New mechanic plans (force-close-architecture's three
  follow-up directions — stronger open_cost, multi-gen,
  matured_arb_bonus — become testable AFTER this plan ships).
- 66-agent scale-up (gated on rewrite-overall verdict).
- v1 deletion (gated on rewrite-overall PASS).
- Reward-shape iteration. Genes promoted are existing knobs;
  no new shaping logic is added.
- BC pretrain. Existing BC genes (`bc_pretrain_steps`,
  `bc_learning_rate`, `bc_target_entropy_warmup_eps`) stay as
  they are.
- Throughput / speed work — `phase-4-restore-speed` owns that.

## Useful pointers

- Current `CohortGenes`:
  [`training_v2/cohort/genes.py`](../../../training_v2/cohort/genes.py).
- Cohort runner CLI:
  [`training_v2/cohort/runner.py`](../../../training_v2/cohort/runner.py).
- Worker (where gene → reward_overrides happens):
  [`training_v2/cohort/worker.py`](../../../training_v2/cohort/worker.py).
- `_REWARD_OVERRIDE_KEYS` whitelist:
  [`env/betfair_env.py`](../../../env/betfair_env.py) (search
  for the constant).
- Existing gene tests:
  [`tests/test_v2_cohort_genes.py`](../../../tests/test_v2_cohort_genes.py).
- Existing runner tests:
  [`tests/test_v2_cohort_runner.py`](../../../tests/test_v2_cohort_runner.py).
- Plans that called specific knobs "genes" but never promoted:
  - `plans/selective-open-shaping/master_todo.md` (open_cost)
  - `plans/reward-densification/` (mark_to_market_weight)
  - `plans/scalping-active-management/` (fill_prob_loss_weight,
    mature_prob_loss_weight, risk_loss_weight)
  - `plans/per-runner-credit/` (mature_prob_loss_weight)
  - `plans/arb-signal-cleanup/` (alpha_lr)

## Estimate

Per session:

- Session 01 (schema + breeding): ~1.5 h (1 h code + 30 min
  tests).
- Session 02 (CLI + plumbing): ~1.5 h (45 min CLI + 30 min
  worker plumbing + 15 min tests).
- Session 03 (validation + writeup): ~1 h (20 min validation
  cohort + 40 min findings.md).

Total: ~4 h. No GPU cohort wall in any session — Session 03's
validation is 1 agent × 1 day (~5 min).

If past 5 h on Session 01 or 02 excluding tests, stop and check
scope — the work is largely repetitive (each new gene adds the
same shape of code in 4-5 places); a long Session 01 means
something is wrong with the abstractions chosen.
