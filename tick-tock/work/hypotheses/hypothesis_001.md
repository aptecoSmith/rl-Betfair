# Hypothesis 001 — direction machinery + BC-driven maturation

**Era type:** tock (exploit) | **Tests against:** the first Tick
(`registry/pbt_genes_v2`, full-width `--enable-all-genes`, n=80, gens 0–5,
completed 2026-06-06 21:47). | **Confidence:** MODERATE (see caveats).

Analysis source: `tick-tock/work/analysis/phenotype_analysis_20260606_2301.md`
(`--tick-only`; the first Tick is untagged ⇒ all 80 rows are tick).

---

## Target (an OUTCOME, not a rate)

Raise **held-out locked_pnl** while **holding σ_naked_leg flat-or-lower** — i.e.
beat the cold Tick on **locked_pnl AND locked/σ_naked_leg** on the sealed-7, at
matched force_close. Rates (maturation / close / naked / stop) are
**diagnostics**, never the objective (compositional-rate trap: the four rates
sum to ~1, so chasing one is an accounting artefact).

Why this target is the right one for THIS Tick: the first Tick's reward redesign
already killed the spray-and-bail pathology (force_close_rate = 0, was 74%), but
the remaining problem is plainly visible in the behaviour summary —
**maturation_rate = 1.7%**, **naked_rate = 58%**, **mean locked_pnl = £1.73**,
**mean naked_sd = £224**. Resolution is low and naked variance dominates. The
deployment goal is *more resolved P&L per unit of naked-leg variance*.

## The mechanism (why these genes)

Two coherent, mechanistically-plausible clusters dominate the n=80 correlations:

**1. The direction machinery — converts naked legs into resolved closes.**
| gene | close_rate | naked_rate | locked_pnl | note |
|---|---|---|---|---|
| `direction_gate_enabled` | **+0.61** | **−0.68** | +0.22 | the dominant lever |
| `use_direction_predictor` | +0.31 | **−0.49** | +0.23 | required by the gate (coupling) |
| `direction_gate_warmup_eps` | — | −0.41 | **+0.31** | the TOP locked driver |

The gate refuses opens its direction-confidence can't support, so fewer pairs
are left naked (−0.68) and more are actively closed (+0.61). `warmup_eps` is the
single strongest locked_pnl correlate (+0.31) — anneal the gate in slowly rather
than slamming it on at episode 0.

**2. BC-driven maturation — the only lever that moves the 1.7% maturation floor.**
| gene | maturation_rate | locked_pnl |
|---|---|---|
| `bc_learning_rate` | **+0.41** (top) | +0.30 (Pearson **+0.47**) |
| `bc_pretrain_steps` | +0.28 | — |

The maturation-conditioned BC oracle is *designed* to teach "open pairs that will
mature." A higher BC learning rate is its strongest correlate with both
maturation and locked.

`stop_loss_pnl_threshold` is included as a **moderate** bail control, NOT
maximised: it cuts stop-bails (stop_close_rate ρ=−0.93) but RAISES naked_rate
(ρ=+0.61) — a genuine trade-off. We seed it mid-range (0.18–0.26) and let PBT
drift, rather than pushing it to the recipe-synth's ~0.28.

## The recipe → seed bands (`seeds/seed_args_001.txt`)

| driver | type | seed | drift? | rationale |
|---|---|---|---|---|
| `use_direction_predictor` | structural bool | `=true` | era-wide pin | satisfies gate coupling; naked −0.49 |
| `direction_gate_enabled` | structural bool | `=true` | era-wide pin | close +0.61, naked −0.68 (dominant) |
| `direction_gate_threshold` | non-struct float | `0.25:0.40` | ✓ | gate must actually filter (range [0.20,0.50]) |
| `direction_gate_warmup_eps` | non-struct int | `8:16` | ✓ | TOP locked driver +0.31 (range [0,20]) |
| `stop_loss_pnl_threshold` | non-struct float | `0.18:0.26` | ✓ | moderate bail (range [0,0.30]) |
| `bc_pretrain_steps` | structural int | `=500` | era-wide pin | turns BC ON — see contradiction below |
| `predictor_lean_obs` | structural bool | `=false` | era-wide pin | full obs — the only layout with a BC oracle (see below) |
| `bc_learning_rate` | non-struct log-float | `5e-4:1e-3` | ✓ | top maturation driver +0.41 (range [1e-5,1e-3]) |

Everything else: `--enable-all-genes` full-sample (the 4 non-structural seeds
are auto-added to `enabled_set`, so they breed/record + drift). The
direction-LABEL triple stays pinned (60/5/60) — the one pre-scanned cache; the
analysis appendix confirms it had zero variance in the Tick.

## RESOLVED — the bc_learning_rate / bc_pretrain_steps contradiction

The original first-recipe (`current_state.md §2`) seeded `bc_learning_rate` high
**and** `bc_pretrain_steps→0`. **At 0 steps BC never runs, so `bc_learning_rate`
is inert** — its +0.41 maturation correlation among 0-step agents would be a
pure PBT co-inheritance confound (block-inherited dead gene), not a lever.

**Decision: BC ON.** Seed `bc_pretrain_steps=500` (structural, era-wide) so the
BC step actually runs on fresh blood, and seed `bc_learning_rate=5e-4:1e-3`
(high) so it bites. Rationale:
- maturation (1.7%) is the weakest behaviour and `bc_learning_rate` is its top
  correlate — abandoning BC (the "BC-off, drop the LR seed" branch) would
  discard the only maturation lever the analysis surfaces.
- gene-dependency-consistency is now satisfied: every seeded gene is *active*
  given the others.
- PBT interaction is sound: BC runs on fresh blood only (warm-started
  elites/offspring skip it — `project_pbt_breeding`), so the seeded high-LR BC
  trains each fresh-blood lineage's maturation behaviour, and PBT carries the
  trained weights forward. The R1 refill re-seeds BC fresh blood every gen.

## Why predictor_lean_obs is pinned full (BC ⇄ obs-layout coupling)

Caught during the first launch (gen-0 log): with `predictor_lean_obs` left free,
~half the fresh blood drew the LEAN obs layout (obs_dim ≈ 574) and **skipped BC**
— "BC oracle obs_dim mismatch (cache 2254 vs caller 574)". BC clones a
**layout-specific** oracle, and only the FULL-obs oracle (obs_dim 2254) has been
scanned. So "BC on era-wide" is only true if every agent uses the full-obs
layout. `bc_pretrain_steps=500` makes BC *requested* era-wide; pinning
`predictor_lean_obs=false` makes it *runnable* era-wide. This isn't hiding a bug
— BC is inherently coupled to an obs layout, so a BC-on hypothesis is implicitly
a full-obs hypothesis.

Note the analysis's "full obs > lean" (locked ρ=−0.30, maturation ρ=−0.26) is
itself **confounded** by exactly this: in the Tick only full-obs agents ever ran
BC. So pinning full loses no clean signal. The genuine lean-vs-full question
(BC on BOTH layouts) is deferred to a future hypothesis once the **per-layout
oracle cache** lands (queued as a spawned task: key `oracle_cache_v2` by obs
layout so lean + full coexist).

## Day structure — chronological 4-tier folds + sliding holdout (revised)

After two false starts (full-obs, then a stale-window crash), the operator
re-architected the day structure (rotation-rework) — so **this tock is an
intentional "oranges" one-off**: the first era on the new structure, a valid
comparison ANCHOR for future eras (the old Tick used random folds on the
05-19 pool and is not comparable).

- **Sliding holdout = newest 7 racing days** (`--holdout-recent 7`; currently
  2026-05-29..06-05). All older cached days are training (currently 48,
  2026-04-06..05-28). As data lands the holdout slides forward.
- **Chronological, old-anchored folds** (`--pbt-rotation-mode chronological`):
  R1..R(n-1) are FIXED as data accumulates; the top tier trains on the
  freshest data. Cross-era comparability comes from the fixed folds; the
  sliding holdout is the un-overfittable judge.
- **4-tier ladder (R4 live)**: `--pbt-tier-sizes 6,4,3` +
  `--pbt-promote-counts 3,2,2` + `--pbt-freeze-top 2` (R1 absorbs slack = 3).
  48 training days = 4 × 12-day rotations.
- **12-day rotations, 6 train / 6 eval** (was 6/4) — more held-out per tier
  for a lower-variance, less-overfit-prone selection signal under fixed folds.
- All 48 training days are direction-label + oracle cached (scanned
  2026-06-07); the newest-7 holdout needs no caches (eval uses the live
  predictor bundle).

The held-out compare judges on the SAME sliding newest-7
(`tools/tick_tock_compare --holdout-recent 7`).

## Prediction (falsifiable, judged on held-out sealed-7)

At **fc=0** (deploy default), the tock's R3 champions will show:
1. **higher mean locked_pnl** than the Tick's R3 champions, and
2. **locked/σ_naked_leg ≥** the Tick's (a higher locked bought with a
   proportionally higher σ_naked_leg is NOT a win),

with σ_naked_leg ideally ≤ ~£30/leg. Also report **fc=120** (overdraft upper
bound) for completeness. If locked does not rise, or rises only by inflating
σ_naked_leg, the joint recipe is **falsified** and the next hypothesis should
isolate which cluster (direction vs BC) carried the marginal signal.

## Caveats (load-bearing — this is a candidate to test, not proof)

1. **Correlation ≠ causation**, n=80, p-values uncorrected for dozens of tests.
   The Tock is the interventional A/B that settles it.
2. **Architecture / co-inheritance confound.** `use_direction_predictor` +
   `direction_gate_enabled` are STRUCTURAL and PBT-inherited in blocks, so their
   marginal correlations partly proxy "the lineages that happened to carry the
   winning block." The Tock seeds them deliberately, breaking that confound.
3. **Marginal ≠ joint.** These are one-gene-at-a-time correlations; the Tock
   tests the JOINT recipe empirically. A wrong joint fails its held-out test and
   we learn something.
4. **σ_naked_leg unknown for the Tick yet** — the held-out compare computes it
   for both eras; the prediction is therefore relative (tock vs tick), not
   against an absolute £30 line.

## Launch (after operator sign-off — see the STOP gate)

Tock = a fresh PBT era seeded from `seed_args_001.txt`, tagged
`--era-type tock --hypothesis-id hypothesis_001`, launched with the SAME
predictor bundle + flags as the first Tick (the gate needs the live predictor
obs: `--use-race-outcome-predictor --predictor-bundle-manifests <champ> <rank>
<dir> --use-direction-predictor`). Then run the held-out compare:
`tools/tick_tock_compare.py` (tick = `registry/pbt_genes_v2` all-rows;
tock = the new era by `--tock-hypothesis-id hypothesis_001`).
