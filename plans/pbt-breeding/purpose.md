# pbt-breeding — purpose

## Problem

The cohort GA (`training_v2/cohort/runner.py::_breed_next_generation`) evolves
the **genotype** (hyperparameters: lr, entropy, the Phase-5 reward knobs,
hidden_size) but **discards the phenotype** (the trained policy weights) every
generation. Top-half elites carry their *genes* verbatim, but every agent —
elites included — **re-trains from scratch with a fresh per-generation seed**
(`per_agent_seeds = f(cohort_seed, generation_idx)`). Consequences, measured
on the 2026-06-03 BC-ON cohort (`registry/smdc_bcon_1780436088`):

- **A champion's performance does not reproduce.** Gen-4 best composite
  +0.151 → gen-5 best +0.132, and the winning `hidden_size` flipped 64→256.
  The agent's *identity* (its learned weights) was thrown away; only the recipe
  survived, and a re-train under a new seed landed somewhere else.
- **Selection is ~half noise.** Within-gen composite spread (best−worst ≈ 0.09)
  is about the size of the signal (≈ 0.13). The "best" each gen is partly the
  luckiest from-scratch draw, so good recipes are evicted when their re-train is
  unlucky. Some recipe genes still converge (open_cost→0.76, lr→0.00041), so
  the GA isn't dead — but heritability of *what actually trades* is near zero.

## Insight

The thing that trades is the **weights**, not the recipe. To make identity
heritable you must carry the weights forward **and keep training them**
(warm-start), not re-roll from scratch. That is **Population-Based Training**
(PBT): the population trains continuously; periodically weak members *exploit*
(copy a strong member's weights+recipe) and *explore* (perturb the recipe),
while strong members keep their weights and keep training.

Two things fall out for free: warm-starting is fine-tuning rather than a fresh
dice roll, so **the seed-noise that made selection meaningless collapses**; and
**crossover dies** — you can't meaningfully cross two independently-trained
networks' weights (no neuron alignment), so reproduction becomes asexual
copy+perturb, not 50/50 gene crossover.

## The two design tensions (operator-raised 2026-06-03 — load-bearing)

**1. Warm-start ⇒ overfit, so day-rotation becomes MANDATORY.** A surviving
brain that keeps taking gradient steps on the *same* days will memorise them.
The current from-scratch GA had accidental regularisation (every agent a fresh
draw); warm-start removes it. So this plan couples weight-inheritance with:
rotate the **training** days every generation (sample a different slice from a
larger pool), keep selecting on **rotating held-out** days, keep the **sealed
final test** (May 20–29) untouched. Overfit guard and warm-start are ONE
decision.

**2. Two-way PBT kills novelty ⇒ 3-way split with protected fresh blood.**
Survive + perturb-the-survivors collapses the population to descendants of
gen-1's winners (diversity death → local optimum). Fix = a third tier. But
"fresh blood" is two different things:
  - *Recipe novelty* (new hyperparameters) — cheap, and it's what the GA
    optimises. An inherited brain + a boldly resampled recipe stays competitive.
  - *Weight novelty* (a from-scratch brain in a new basin) — the true fresh
    blood, BUT a from-scratch immigrant has 1 gen of training vs the veterans'
    N, so it is almost always culled before it matures. Fresh blood drowns
    unless it is **protected** (cull-immunity for K generations).

## Proposed mechanism (the starting point, not the settled answer)

Per generation, 30 agents, 3-way split:

| tier | ~share | brain (weights) | recipe (genes) | trains? |
|---|---|---|---|---|
| **Elites (preserve)** | 50% | keep own | keep own | yes, continue on rotated days |
| **Offspring (exploit)** | 30% | inherit a top elite's | perturb ±20% | yes, warm-start |
| **Immigrants (explore)** | 20% | mostly inherit a (random) elite's; small **protected** quota from-scratch | boldly resampled (full range) | yes; from-scratch quota gets K-gen cull-immunity |

## Open decisions (for `master_todo` / a design session to settle)

- **Split ratios** (50/30/20?) and the protected-from-scratch sub-quota size.
- **Explore perturbation:** offspring ±20% vs immigrants full-resample — which
  genes, multiplicative vs resample, per-gene independent prob.
- **Protected-youth window K** (2 gens?) and how immunity interacts with the
  elite count (do protected juveniles displace an elite slot?).
- **Day-rotation scheme:** pool size, per-gen sample size, train/iteration-eval
  split, determinism (so A/B is paired).
- **Checkpoint format:** weights must now be threaded across gens + copied to
  children. The current resume/checkpoint is gene-only by design — this is the
  biggest code change. Where do per-agent weights live between gens (registry
  `weights/` already stores them; need a parent→child copy + warm-start load).
- **Compute:** warm-start trains everyone every gen (no saving vs from-scratch),
  but each gen's training can be SHORTER (fewer PPO episodes) since it builds on
  prior learning — tune episodes/gen down to keep wall-clock flat.

## Validation

A/B vs the current gene-only GA on the **same cohort seed + same day pool**,
both evaluated on the **sealed** May 20–29 test. Success = (a) champion
performance reproduces across gens (heritability), (b) selection spread shrinks
relative to signal, (c) held-out locked_per_std beats the gene-only GA, (d)
population diversity (recipe + behavioural) does not collapse — measured, with
the immigrant tier demonstrably contributing survivors.

See `hard_constraints.md` and `master_todo.md`.
