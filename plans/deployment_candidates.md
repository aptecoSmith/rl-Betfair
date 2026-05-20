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

### ⭐ cea2ee94 / 1df49aa0 (gen 1 — SAME AGENT, two IDs) ⭐

**CRITICAL: these are the same agent under different uuids.**
Deep compare (2026-05-20) confirmed:
- Identical eval stats to 12 decimal places (eval_day_pnl
  65.11172469282474 exactly equal in both cohorts' scoreboards)
- Identical hyperparameters/genes
- Different md5 hashes ONLY because the .pt file stores
  agent_id metadata inline

The reason: both cohorts used `--seed 42`, identical gates,
identical training data. Gen 1 agents were bred from gen 0
top performers; the two selectors (day_pnl_per_std and
sortino) ranked gen 0 the SAME WAY at this particular sample
of agents → bred identical parents → produced identical
gen 1 offspring. Selection only meaningfully diverges from
gen 2 onwards.

**Held-out reeval numbers differ slightly** (cea2ee94 +£72 vs
1df49aa0 +£63 on fc=120) but only because reeval uses
stochastic action sampling (Categorical/Beta heads), and the
per-agent RNG seed is derived from the agent uuid. Same
weights + different RNG = different bet sequences = different
P&L. **The shape is the same; the noise is sampling-only.**

**Deploy ONE of them** (recommend cea2ee94 — higher reeval
numbers from happier RNG; or use `--argmax-eval` for a
deterministic comparison). Deploying both is duplicate
exposure, NOT diversification.

Combined held-out shape (averaging across both RNG samples):
- fc=120: ~+£68/d, naked ~+£3
- fc=0:   ~+£168/d, naked ~+£63 (high-upside; partly tailwind)

- **In-sample**: pnl +£65/d, span £270, worst day −£23,
  naked_worst −£76
- **Held-out fc=120**: **+£63/d** (regressed only −£2),
  locked +£123, naked **−£3**
- **Held-out fc=0**: **+£121/d**, locked +£119, naked **+£11**
- **Shape**: locked +£123 (highest of any deploy candidate),
  naked near-zero on both fc settings. Held-out is essentially
  pure locked-floor.
- **Why deploy first**: cleanest tailwind-independent shape in
  the entire deploy pool. Locked floor doing virtually all the
  work; the +£63 on fc=120 holds up to a live audit because the
  naked is bench-flat. The agent you'd put in front of a
  compliance reviewer.

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

## Tier 2 — secondary / higher-variance picks

### cea2ee94 — see ⭐ entry above (merged with 1df49aa0)

This was previously listed separately. Subsequent deep compare
(2026-05-20) showed cea2ee94 ≡ 1df49aa0 are the same agent
with different uuids — see the Tier 1 ⭐ entry.

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
