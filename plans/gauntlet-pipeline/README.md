# gauntlet-pipeline — design

**Date:** 2026-06-13. **Status: design agreed (this conversation), not built.**
Read `purpose.md` first, then `hard_constraints.md`, then `master_todo.md`.

This is the design trail. CLAUDE.md stays slim; the detail lives here.

---

## 1. The gauntlet model (the science — unchanged)

- **Tranches are FIXED-SIZE** chronological blocks (~10 days; current:
  `--pbt-train-per-rotation 6 --pbt-eval-per-rotation 4`, with held-out
  selection the tranche-eval days fold into train). The **gauntlet** = the
  ordered sequence T1, T2, …, TN.
- **The gauntlet GROWS by appending tranches** as data accumulates (T(N+1) when
  enough new days bank). Tranches are **never resized** — so every model that
  ever ran faced the *identical* T1, T2, … gauntlet → fair comparison. A longer
  gauntlet = more training + more selection rounds (operator: a feature).
- **Every model is recipe-pure:** one gene config, cooked from T1, warm-starting
  only its OWN weights between tranches. (Survivors already do this; fresh blood
  and mutants enter at T1 and climb.)
- **Full fair shot (operator decision 2026-06-13):** a newly-bred recipe climbs
  the ENTIRE current gauntlet T1→TN **uninterrupted — never culled mid-climb.**
  It is judged only once it has completed the same gauntlet as the incumbents,
  in a same-depth comparable pool. We do **not** early-cull climbers (the
  cheaper "halving-flavored" option was considered and rejected: we don't trust
  a recipe judged on a partial cook). This costs more compute — accepted.
- **Selection** happens among agents that completed the **same depth**, scored
  on the **fixed validation set at fc=0** (the held-out-selection regime —
  `plans/maturation-raising/holdout-selection.md`). Keep top fraction → they
  advance; cull the rest → breeding emits replacements at T1.
- **Breeding** (separate stage): culled slots → new recipes — **mutants**
  (perturbations of survivors, `make_offspring`) + **fresh blood** (gene-space
  samples; later register-driven). All enter the **needs-T1** queue.

Net: "survivor" vs "mutant" **dissolves at execution time** — everyone is just
*an agent climbing the gauntlet under one fixed recipe, warm-starting its own
weights*. Selection removes; breeding adds at T1; the climb is automatic
recipe-pure catch-up.

---

## 2. The execution architecture (the new part)

**Decouple execution from breeding.** Three components:

### a. Executor — a pure primitive (no selection inside)
```
run_tranche(batch, K) -> results
  batch[i] = (recipe, init_weights | None, train_days_for_K, validation_days)
  trains each agent on tranche K (fresh init iff K==1 / no weights; else
  warm-start the agent's OWN K-1 weights), evals on the fixed validation set
  at fc=0, returns (weights_path, validation_scores, bet_logs). NO culling.
```
This is ~`train_one_agent` today (it already takes config + init_weights +
train_days + eval_days). The work is to expose it as a clean batch primitive and
strip any in-loop selection coupling. Per-run cost is **uniform**: `batch × one
tranche`.

### b. Ledger / queues — the persistent state
A durable record of every agent: `recipe`, `lineage_id`, `origin`,
`weights_path`, `tranches_completed`, `validation_score_per_tranche`. Derived
**needs-T(K)** queues (an agent in needs-T(K) has completed 1..K-1). Fresh blood
+ mutants land in needs-T1. The ledger is the resume state and the thing a
second machine reads to grab work.

### c. Breeder — separate selection + mutation
Reads the **frontier pool** (agents that completed the current deepest tranche),
applies truncation selection on the fc=0 composite, promotes survivors, and
emits replacement recipes (mutant + register-driven fresh) into needs-T1. Never
touches weights of a different recipe (recipe purity).

### Orchestrator
Drives `run_tranche` over batches pulled from the queues, advancing agents
through the gauntlet, keeping the box busy on uniform runs. Distributable (any
worker grabs a batch) and resumable (ledger = state).

---

## 3. Data flow (worked example, batch=16, keep-top-8)

```
needs-T1: [16 fresh]                  -> run_tranche(.,1) -> 16 done@T1
needs-T1: [next 16 fresh/mutants]     -> run_tranche(.,1) -> 16 done@T1
... agents accumulate at each depth; a recipe climbs T1->T2->... on its OWN
    weights, never culled mid-climb (full fair shot) ...
frontier pool (all @TN) reaches quorum -> BREEDER: keep top 8 (advance/hold),
    cull 8 -> emit 8 mutants + (fresh) into needs-T1, which then climb T1..TN.
new data banks T(N+1) -> everyone @TN advances to needs-T(N+1); breeder keeps
    perturbing; new mutants chase the frontier from T1.
```

Population is bounded by **frontier selection** (trim to carry-size); the
in-flight climbers are the accepted extra compute.

---

## 4. Gene register (replaces random fresh blood, eventually)

- **v1 (read-only coverage map):** load every `scoreboard.jsonl` /
  `model_register.csv` across eras (every agent's full gene config is ALREADY
  persisted), bin each gene's range, mark visited cells + their held-out
  outcomes. Answers "where have we looked, where's blank, where's
  promising-but-thin." Cheap, immediately useful, no training.
- **v2 (gap-targeting sampler):** replace `sample_fresh_blood_genes`'s uniform
  roll with draws from under-explored / promising-thin cells (low-discrepancy /
  Latin-hypercube-style coverage). Turns fresh blood from random into a
  systematic fill of the gene space.

---

## 5. What this is NOT

- **Not warm-starting mutants** (recipe purity — see purpose.md). The compute
  saving that idea offered is deliberately forgone.
- **Not fixed-N / growing-tranche-size** (rejected: would break the
  same-gauntlet-for-all comparability). Tranche size fixed, count grows.
- **Not successive-halving** (considered, rejected): halving never injects
  mid-march mutants and assumes a fixed dataset — it conflicts with
  grow-with-data + mutate-toward-winners. Its one transferable idea (cull cheap
  + early) was also rejected here via the "full fair shot" decision.
- **Not a reward/env/selection-metric change** — those carry over.

---

## 6. Validation before cutover

The new path is a re-architecture, NOT byte-identical. Gate it behind a flag and
**A/B against the current `--breeding lockstep`** on the same data: held-out
locked / σ_naked_leg on the sealed-7 must not regress. Only cut over once the
new pipeline matches-or-beats the old loop on held-out quality. Keep the old
lockstep path intact until then.
