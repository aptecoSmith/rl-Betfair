# Candidates 001 — drafts + self-critique (why hypothesis_001 won)

Multi-candidate + self-critique over the n=80 first-Tick analysis. Each target
is framed as a **locked-P&L / naked-variance OUTCOME** (rates are diagnostics).
Self-critique gates: prior falsified hypotheses (none yet — #1), the
**compositional-rate trap**, **marginal ≠ joint**, and **gene-dependency
consistency**.

## C1 — Direction machinery + BC-on maturation ✅ CHOSEN
Seed the direction trio (predictor+gate+warmup) + moderate stop-loss + BC-on
with high bc_learning_rate. See `hypothesis_001.md`.
- **For:** combines the two strongest, mechanistically-distinct clusters — the
  gate (naked −0.68 / close +0.61 / locked +0.22) cuts naked variance, BC
  (maturation +0.41 / locked +0.47 Pearson) lifts the 1.7% maturation floor.
  Gene-dependency-consistent (BC on ⇒ bc_learning_rate is live).
- **Against:** 7 seeds is the upper end of "lean (c)"; BC adds fresh-blood
  compute. Accepted — the trio is one coupled cluster, and BC targets the
  binding constraint.

## C2 — Direction machinery ONLY (BC off) — rejected (kept as the fallback)
Seed predictor+gate+threshold+warmup+stop-loss; drop both BC genes.
- **For:** the cleanest isolation of the direction cluster; cheapest; no BC
  variance.
- **Against / why rejected:** leaves maturation at its 1.7% floor — it discards
  the only lever (`bc_learning_rate`) the analysis ties to maturation. Since the
  Tick's binding problem is *low resolution*, a recipe that can't move
  maturation is a weaker bet.
- **Disposition:** this is the **designated next hypothesis** if C1 is falsified
  by BC *adding σ_naked_leg without buying locked* — it cleanly attributes the
  marginal signal to the direction cluster alone.

## C3 — Maximise stop_loss + gate (the recipe-synth's literal top picks) — rejected
Seed `stop_loss_pnl_threshold→~0.28`, gate on, `bc_target_entropy_warmup_eps→18`.
- **Why rejected — compositional-rate trap.** The synth ranks `stop_loss` #1
  only because it dominates `stop_close_rate` (ρ=−0.93) — but stop_close_rate is
  a *diagnostic*, not the objective, and the SAME gene RAISES `naked_rate`
  (ρ=+0.61). Maximising it directly fights the σ_naked_leg goal. C1 instead
  seeds stop_loss mid-range and lets it drift.

## C4 — Kitchen sink (seed all 16 synth candidates) — rejected
- **Why rejected — over-pinning + marginal≠joint.** Pinning 16 genes makes the
  Tock a near-point test, closing off the un-hypothesised space the warm-start
  is supposed to refine (violates "pin 3–5 drivers, full-sample the rest").
  Most of the 16 are weak (|ρ|<0.25), conflicted, or predictor-coupled. A
  designed point in 16-D is exactly what marginal correlations can't justify.

## Gene-dependency consistency note (the catch that shaped C1 vs C2)
`bc_learning_rate` is inert unless `bc_pretrain_steps>0`. The original recipe
seeded it high *with* steps=0 — a dead seed. C1 resolves it (steps=500); C2
honestly drops it. Any future hypothesis seeding `bc_learning_rate`,
`bc_target_entropy_warmup_eps`, or `bc_direction_target_weight` MUST also seed
`bc_pretrain_steps>0`, else those seeds are confound-only.
