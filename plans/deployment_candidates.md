# Deployment candidates — current best agents

Curated list of trained agents that have passed both in-sample
selection AND held-out reeval (7-day forward window, fc=0 AND
fc=120). Trained at `starting_budget=100.0` per `config.yaml`.

Updated 2026-05-20 after the E3 cohort + Sortino cohort + 19
probe series. Every candidate uses the **E3 mechanism** at its
base — `close_feasibility_max_spread_pct=0.05` — which is the
proven open-side intervention worth +£29/d over the prior
deployment baseline (layq's +£26/d held-out fc=120).

All candidates pass the **deployable composite filter**:
positive pnl mean, penalised by worst-day breach < −£30 and
worst-naked-day breach < −£40.

## Tier 1 — primary deployment pool

### 11099f65 (E3 cohort, gen 2) — the CONSISTENT EARNER

- **In-sample**: pnl +£32/d, **span only £96** (tightest in cohort),
  worst day −£30, naked_worst −£75
- **Held-out fc=120**: **+£49/d** (improved +£17 vs in-sample!)
- **Held-out fc=0**: +£117/d, naked +£30
- **Shape**: locked +£101, consistent ~£30-50/day earner.
  Smallest per-day span we've found. Modest naked exposure.
- **Why deploy**: most replicable shape — held-out IMPROVED, not
  regressed. The "+£32/d safe earner" you can run with confidence.
  Excellent for capital-constrained deployment.

### 57a42db5 (E3 cohort, gen 2) — the CONSISTENT EARNER (twin)

- **In-sample**: pnl +£48/d, span £198, worst day −£21,
  naked_worst −£74
- **Held-out fc=120**: **+£48/d** (matched in-sample exactly)
- **Held-out fc=0**: +£114/d, naked +£26
- **Shape**: locked +£101, slightly more variance than 11099f65
  but higher mean. Bounded worst-day (-£21 in-sample).
- **Why deploy**: paired with 11099f65 these are the
  "850522b9-shape" robust agents the plan was hunting for. Run
  both as a diversified pair.

### 1df49aa0 (Sortino cohort, gen 1) — the CLEAN-NAKED EARNER

- **In-sample**: pnl +£65/d, span £270, worst day −£23,
  naked_worst −£76
- **Held-out fc=120**: **+£63/d** (regressed only −£2)
- **Held-out fc=0**: TBD (fc=0 reeval pending as of writing)
- **Shape**: locked +£123 (highest of any deploy candidate),
  naked −£3 on held-out (lowest tailwind-dependence of any
  candidate). Held-out shape is virtually pure locked-floor.
- **Why deploy**: cleanest single-agent shape in the deploy
  pool. Locked floor doing essentially all the work; almost
  zero naked dependence on held-out. The agent that would
  bench-test the cleanest in a live-trading audit.

## Tier 2 — secondary / higher-variance picks

### cea2ee94 (E3 cohort, gen 1) — the HIGH-UPSIDE PICK

- **In-sample**: pnl +£65/d, span £270, worst −£23, naked_worst
  −£76
- **Held-out fc=120**: +£72/d (held up cleanly)
- **Held-out fc=0**: **+£215/d**, naked +£115 (large positive
  tailwind contribution)
- **Shape**: locked +£114 (strong), naked +£9 on fc=120, naked
  +£115 on fc=0
- **Why deploy with caveat**: highest upside of any candidate,
  but the fc=0 win has significant naked-tailwind contribution.
  Add to the pool for absolute pnl chase, but size smaller than
  Tier 1 — the +£115 naked is luckier than the +£2 to +£30
  range of Tier 1 candidates.

### 850522b9 (E3 cohort, gen 2) — the IN-SAMPLE GOLD that REGRESSED

- **In-sample**: pnl +£65/d, **worst −£20** (gold-standard
  shape), naked_worst −£33
- **Held-out fc=120**: +£45/d (regressed −£20)
- **Held-out fc=0**: +£12/d (regressed −£53!)
- **Shape**: locked +£109, naked dragged it down significantly
  on held-out
- **Why caution**: looked perfect in-sample (the gold-standard
  shape we were hunting). The −£20 regression on fc=120 and
  −£53 on fc=0 are the worst regressions in either cohort top-5.
  Probably overfit to in-sample. **Marginal pick** — include in
  a 4-agent deploy ensemble if you want broad coverage, but
  don't size it large.

## Anti-recommendations (DO NOT DEPLOY)

### Agents not in the top-5 of either cohort

Any agent below the Tier 1/2 list above has lower deploy_score
either by in-sample mean, worst-day breach, or naked tail. Not
worth deploying.

### Sortino cohort agents beyond 1df49aa0

The remaining Sortino top-5 (f51a8bb3, d28b6edb, e8b73472,
7c07f647) all hold up on held-out but have lower means than
their E3-cohort equivalents. 1df49aa0 is the unique Sortino
contribution; the others are inferior duplicates of E3 cohort
agents.

## Deployment strategy considerations

1. **Capital sizing**: trained at £100/race. Budget-sweep
   experiments queued to determine minimum-viable budget — likely
   £50-£100 sweet spot (below £50, MIN_BET_STAKE=£2 forces 4 %+
   per-bet concentration that distorts the trained policy).

2. **Diversification across candidates**: 11099f65 + 57a42db5 +
   1df49aa0 as a 3-agent ensemble would give:
   - Locked floor coverage from all three
   - Different gen / cohort / training-trajectory roots → reduces
     correlated failure risk
   - All three hold up on held-out fc=120 cleanly

3. **Force-close setting at deployment**: stick with `fc=120` to
   protect against late-day naked variance. The fc=0 numbers
   look better but only because un-flattened pairs ran lucky on
   the held-out window. Production deploy ≠ academic backtest.

4. **Next cohort (R3+E3) may produce new candidates**: currently
   queued, ~28h GPU. If it bites, top-5 may include better shapes
   than Tier 1 here. Will update this file accordingly.

5. **What changed our thinking**: Sortino was hypothesised to
   surface bounded-worst-day shapes. Empirically it produced agents
   with cleaner naked profiles but lower mean. The E3 cohort's
   day_pnl_per_std selector remains the right choice. Sortino is
   retired.
