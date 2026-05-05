---
plan: rewrite/phase-11-bc-gene-exploration
status: SKETCH
opened: 2026-05-05
depends_on: rewrite/phase-8-oracle-bc-pretrain
---

# Phase 11 — BC gene exploration (sketch)

## Why this plan exists

Phase 8 ports the v1 BC pretrain pipeline to v2 (oracle scan in S01,
discrete BC + entropy-warmup handshake in S02, validation cohort in
S03). The Phase 8 design **deliberately pins** two BC knobs at
v1-derived defaults rather than letting the GA evolve them:

- `bc_learning_rate = 3e-4`
- `bc_target_entropy_warmup_eps = 5`

The reasoning is in the Phase 8 lessons-learnt entry "Why we pinned BC
gene values for S03": the S03 success bar ("BC half mr ≥ no-BC half mr
+ 1 pp") is a **mechanism test**, not a tuning test. Letting the GA
vary lr/warmup alongside the seven legacy + eleven Phase 5 genes
introduces confounders that make any specific surviving lineage's
outcome impossible to attribute cleanly to BC vs. lr vs. warmup.

Phase 11 picks up the tuning question once Phase 8 establishes the
mechanism works.

## Pre-requisite

Phase 8 S03 passes its gate (BC arm beats no-BC arm by ≥ 1 pp on
gen-1 maturation rate). If S03 fails, Phase 11 may still run as a
diagnostic — "did v1 defaults bury a working mechanism?" — but the
plan's framing changes from "tune" to "rescue".

## Scope

Two BC knobs, both Phase 8 genes already plumbed end-to-end:

- `bc_learning_rate` — Adam learning rate for the BC pretrainer.
  v1 range [1e-5, 1e-3], log-uniform.
- `bc_target_entropy_warmup_eps` — entropy-controller warmup window
  in PPO rollout episodes after BC. v1 range [0, 20].

Out of scope for this plan: `bc_pretrain_steps` (how long to BC).
That's a different question — about marginal returns at the
pretrain-time end of the curve — and probably wants its own plan
(or a fixed sweep at 100, 250, 500, 1000, 2000 once lr/warmup are
settled).

## Two candidate session structures

(Decide between these when fleshing out the plan; each addresses the
"what's responsible for the improvement?" attribution problem
differently.)

### Option A — Manual sweep over a small grid

Run N cohorts back-to-back, each with a different
`(bc_learning_rate, bc_target_entropy_warmup_eps)` combination set
via the existing
[`--bc-learning-rate`](../../../training_v2/cohort/runner.py) and
[`--bc-target-entropy-warmup-eps`](../../../training_v2/cohort/runner.py)
CLI flags (added in Phase 8 S02 as escape hatches for exactly this
plan). Same seed across cohorts so the only variable is the BC knob.
Compare gen-1 maturation rate / day_pnl across cohorts.

Pros: clean attribution per-cohort. No GA confounders. Deterministic
under fixed seed.

Cons: O(N) compute. The grid is operator-chosen so it can miss the
right neighbourhood entirely if v1 defaults are far off.

### Option B — Promote BC knobs to Phase 5-style enable-set genes

Extend `_sample_field` in `training_v2/cohort/genes.py` with the two
BC ranges, add the names to `PHASE5_GENE_DEFAULTS` and `_PHASE5_RANGES`,
and run a single 24-agent / 4-gen cohort with
`--enable-gene bc_learning_rate --enable-gene bc_target_entropy_warmup_eps`.
The GA selection pressure picks the winning lineage's BC settings.

Pros: native GA-driven tuning. Single cohort. Selection pressure does
the work.

Cons: confounders — a winning lineage's `bc_lr` value is correlated
with whatever else its genome has. Phase 5's
`fill_prob_loss_weight` story is the cautionary tale (the GA voted
the gene to 0 every cohort because the head's label was broken; the
selection pressure couldn't fix a broken mechanism). Mitigation is to
run AFTER S03 confirms BC works.

## Open questions for the operator

1. Do we trust the v1 defaults enough to skip Phase 11 entirely if S03
   passes its gate at +1 pp? (i.e. "good enough is good enough")
2. If running Option A, how many grid points are worth the compute?
   A 3 × 3 grid = 9 cohorts × ~hours of GPU time — non-trivial.
3. If running Option B, do we lock other Phase 5 genes (no
   `--enable-gene` for non-BC knobs) so the GA's only freedom is
   over BC + the seven legacy genes?

## Deliverables (TBD)

To be sketched once Phase 8 S03 has shipped and we know whether v1
defaults are good enough to leave alone, marginal, or wrong.
