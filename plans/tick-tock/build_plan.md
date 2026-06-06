# Build plan — Tick-Tock v1 (settled 2026-06-06)

The design session agreed the decisions below. This is the actionable spec the
build works from. Read `purpose.md` for *why*, `current_state.md` for the bricks
that exist, `design_decisions.md` for the pitfalls (now resolved — see its
RESOLVED section).

---

## Settled decisions

| Topic | Decision |
|---|---|
| **Terminology** | **era** = one R1→R3 run (one seed, 5 gens), appends to the shared leaderboard. **cohort** = the agents of an era. |
| **Engine** | **Two file-coupled loops.** The proven python era-loop stays the compute engine; Claude is a scheduled science worker. They talk through marker files. |
| **Autonomy** | **Fully autonomous in steady state.** ONE exception: the operator reviews the *first* analysis + hypothesis before the first tock runs (de-risking, not a standing gate). |
| **Held-out** | **sealed-7** (only 7 of the 10 named sealed dates exist on disk) as the cross-era judge. Score **locked_pnl + σ_naked_leg** (never day_pnl). Report **fc=0 AND fc=120**. Log **every peek** to a ledger. A clean final-test set is held back for the eventual deploy candidate (defined when one emerges). |
| **Pinning** | **band-seed + drift.** A tock enables the driver gene, seeds fresh blood within `[lo,hi]`, and lets breeding drift ±20% from the seed. Structural drivers (`use_direction_predictor`, `direction_gate_enabled`, `bc_pretrain_steps`, architecture) are pinned era-wide by nature (frozen per lineage, re-seeded on every fresh draw). |
| **Tagging** | **One shared leaderboard.** Every row stamped `era_id` / `era_type` (tick\|tock) / `hypothesis_id`. The work folder holds the hypothesis + summary mds. |
| **Analysis** | Phenotype correlations run on **tick rows only** (a tock pins its drivers → ~0 variance → would corrupt discovery). Pooling is automatic in the shared dir. |
| **Brain** | **Multi-candidate + self-critique.** Claude drafts several candidate recipes, critiques them against prior falsified hypotheses, the compositional-rate trap, and gene-dependency consistency, then picks one and records the rejected ones + why. |
| **Schedule** | Strict alternation tick → tock. |
| **Build order** | Phase 1 = A–D + one manual cycle (operator reviews analysis+hypothesis before the first tock). Phase 2 = E–F (autonomy). |

---

## The loop + marker protocol

```
python era-loop  (bat-started, runs forever, the compute engine):
  run TICK era  → tag rows tick → drop NEEDS_ANALYSIS → BLOCK until TOCK_READY
  run TOCK era  → tag rows tock + hypothesis_id (seed from seeds/seed_args_NNN.txt)
                → drop NEEDS_SUMMARY (non-blocking) → loop to next TICK

Claude worker  (scheduled routine, wakes ~20–30 min, idempotent, autonomous):
  NEEDS_ANALYSIS → phenotype(tick-only) + read ALL prior hypotheses/summaries
                 → draft N candidate recipes → self-critique → pick one
                 → write hypotheses/hypothesis_NNN.md + seeds/seed_args_NNN.txt
                 → drop TOCK_READY, clear NEEDS_ANALYSIS
  NEEDS_SUMMARY  → held-out compare on sealed-7 (tick champs vs tock champs)
                 → write hypotheses/hypothesis_NNN_summary.md → append peek_ledger.jsonl
                 → clear NEEDS_SUMMARY
```

- **Analysis blocks the next tock** (never run an un-hypothesised tock). **Summary is non-blocking** — it only feeds the *next* hypothesis, and ticks don't need it, so the engine keeps cranking.
- **Watchdog:** if `TOCK_READY` doesn't appear within a timeout, the engine runs another *tick* instead of wedging — a stalled/absent worker degrades to "keep exploring," never to a hang.
- **Durable state:** everything lives on disk → any fresh Claude reconstructs the loop position from `loop_state.json` + the work folder. This is what makes "start from a bat and walk away" safe.

### Work-folder layout (`plans/tick-tock/work/`)
```
loop_state.json            # cycle #, last era_id, era_type, pending marker
markers/                   # NEEDS_ANALYSIS | TOCK_READY | NEEDS_SUMMARY | STOP (touch-files)
analysis/                  # phenotype_analysis_*.md + corr csv per cycle
hypotheses/hypothesis_NNN.md          # chosen recipe: target, seeded genes+bands, rationale, prediction
hypotheses/hypothesis_NNN_summary.md  # did it hold out? locked/σ_naked deltas, verdict
hypotheses/candidates_NNN.md          # the N drafts + critique + why this one won (transparency)
seeds/seed_args_NNN.txt    # the exact --seed-gene args the tock consumes
peek_ledger.jsonl          # one row per held-out evaluation (erosion audit trail)
```

Campaign registry: a **new** dir (e.g. `registry/tick_tock_v1`) so tagging is clean from row 1. The existing `registry/pbt_genes_v2` register is a full-width tick — pool it in as bonus tick data for the first analysis.

---

## Build pieces

### A. Fresh-blood band-seed — the keystone
- `training_v2/cohort/genes.py`: `sample_fresh_blood_genes(rng, enabled_set, seed_bands: dict[str, tuple] | None)`. For each gene in `seed_bands`, draw within `[lo,hi]` respecting type (int / log-uniform / categorical / bool); `lo==hi` ⇒ point pin. Must respect the `use_direction_predictor ⇄ direction_gate_enabled` coupling (seeding gate=True requires predictor=True).
- `training_v2/cohort/pbt.py`: thread `seed_bands` through `_fresh` → `init_pbt_population` + `breed_pbt`'s R1 refill. `make_offspring` is **unchanged** — an enabled gene already perturbs from the seeded parent value (drift), and a structural gene is inherited verbatim (era-wide pin).
- `training_v2/cohort/runner.py`: `--seed-gene NAME=LO:HI` (repeatable). Parses to `seed_bands`; **auto-adds non-structural seeded genes to `enabled_set`** (so they breed + record correctly); collision-guards vs `--reward-overrides` for the same name; validates each band ⊆ the gene's declared range. Plus `--era-type {tick,tock}` and `--hypothesis-id ID` for tagging.
- **Tests:** band draw stays in range AND is *recorded* in the row's `hyperparameters`; structural seed holds across all gens/agents of the era; offspring drift from the seed (not from the gene default); **a no-seed tick is byte-identical to today** at the same seed.

### B. Row tagging
- Stamp `era_id` / `era_type` / `hypothesis_id` at the scoreboard + `model_register` writer. Tolerant readers (missing ⇒ untagged/legacy).

### C. Phenotype `--tick-only`
- `tools/phenotype_analysis.py`: read the `era_type` tag; `--tick-only` filters discovery correlations to tick rows. Pooling across eras is already automatic (it reads the whole accumulating register).

### D. Held-out compare harness
- New thin wrapper over `tools/reevaluate_cohort.py`: given the shared cohort dir + two era selectors (tick vs tock champions, by `era_id`/`hypothesis_id`), reeval both on sealed-7 with identical flags at **fc=0 and fc=120**, report **locked_pnl + σ_naked_leg + paired delta** side-by-side, and append a `peek_ledger.jsonl` row (which sealed days, which eras, when). Output is what the hypothesis summary reads.

### E. File-handshake era-loop
- Modified `run_*_campaign.ps1`: alternate tick/tock; tag eras (pass `--era-type`/`--hypothesis-id`); before a tock, **block on `TOCK_READY`** and load `seeds/seed_args_NNN.txt`; drop `NEEDS_ANALYSIS` / `NEEDS_SUMMARY` between eras; watchdog timeout → run a tick. Maintain `loop_state.json`. `start_tick_tock.bat` / `stop_tick_tock.bat` (stop = today's killer + a `STOP` touch-file the worker checks).

### F. Hypothesis brain + scheduled worker
- The Claude routine (scheduled wake) + its prompt/skill. **Multi-candidate + self-critique** spec:
  1. Run phenotype (tick-only), read all `hypothesis_*` + `*_summary` mds.
  2. Draft **N candidate recipes**, each: target behaviour as a **locked-P&L / naked-variance OUTCOME** (rates are diagnostics only), genes + **bands** (not points), rationale citing correlations.
  3. **Self-critique** each against: prior **falsified** hypotheses (don't re-propose), the **compositional-rate trap**, **marginal≠joint**, and **gene-dependency consistency** (is every seeded gene active given the others? — the bc_learning_rate/bc_pretrain_steps catch below).
  4. Pick one; write `hypothesis_NNN.md` + `candidates_NNN.md` (rejected + why) + `seed_args_NNN.txt`.
  5. Flag low-confidence hypotheses in the md (the summary will catch misses regardless).

---

## Sequence + gates

- **Phase 1 (de-risk):** build A → D. Then run **one hand-driven cycle**: manually launch the direction-machinery tock via `--seed-gene`, run the held-out compare vs the current tick. **Operator reviews the analysis + the hypothesis md before the tock runs.** Confirm the mechanism (seed lands, records, drifts) and the metrics (locked/σ_naked sane) before automating.
- **Phase 2 (automate):** build E → F, start the autonomous loop.

---

## First concrete tock — direction recipe → seed args

From `current_state.md §2`, mapped to the seed mechanism:

| Driver | Type | Seed | Note |
|---|---|---|---|
| `use_direction_predictor` | structural | `=true` | era-wide pin; satisfies the gate coupling |
| `direction_gate_enabled` | structural | `=true` | coupled to the above |
| `direction_gate_threshold` | non-structural | band e.g. `0.25:0.40` | the gate must actually filter |
| `stop_loss_pnl_threshold` | non-structural | band ~`0.18:0.26` | drifts from the seed |
| ~~`bc_learning_rate` high~~ + ~~`bc_pretrain_steps=0`~~ | — | **RESOLVE FIRST** | **Contradiction:** at `bc_pretrain_steps=0` BC never runs, so `bc_learning_rate` is inert; its +0.52 maturation correlation is a co-inheritance confound. Pick ONE: BC **on** (steps 500) with seeded high `bc_learning_rate`, OR BC **off** and drop the LR seed. Decide at the first-hypothesis review. |

Everything else: `--enable-all-genes` full-sample (the seeded genes are auto-added to `enabled_set`). The label triple stays pinned 60/5/60 (covered by the existing pre-scan; the recipe doesn't touch it).

**Build-time wiring checks:** (1) the gene `use_direction_predictor=true` is consistent with the cohort flag `--use-direction-predictor` (both on); (2) the seed parser enforces the gate↔predictor coupling; (3) auto-enable of seeded genes doesn't trip the `--reward-overrides` collision guard.

---

## Deferred (not v1)
- Explicit narrow-range *sampling distributions* beyond a uniform band (band+drift suffices).
- Auto-hypothesis surrogate (Bayesian-opt over gene→outcome) — keep Claude-in-the-loop until the loop is shown to converge.
- Adaptive scheduling (tock-again-if-validated) — start strict.
- Final-test-set definition — pin when a real deploy candidate emerges (likely new future days, or a held-back sealed slice).
