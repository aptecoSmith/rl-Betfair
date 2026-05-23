# Deployment candidates — current best agents

Curated list of trained agents that have passed both in-sample
selection AND held-out reeval (7-day forward window:
2026-05-07..05-13). Trained at `starting_budget=100.0` per
`config.yaml`. All candidates use the **E3 mechanism** at their
base — `close_feasibility_max_spread_pct=0.05` — which is the
proven open-side intervention worth +£29/d over the prior
deployment baseline (layq's +£26/d held-out fc=120).

**Updated 2026-05-21** after R3+E3 full cohort reeval and the
argmax-eval probe. The framing changed from "deploy at fc=120 for
safety" to **deploy at fc=0 for higher mean + simpler ops** per
operator decision; the trade-off is higher per-week variance and
a need for £300–£500 bank to absorb naked-leg swings.

## Operator deployment decision: fc=0

**Why fc=0 over fc=120 (2026-05-21):**

- **Higher mean.** Across all candidates, fc=0 mean is +£97/d
  (E3 top-5) and +£84/d (R3+E3 top-5), vs +£55 and +£47 at fc=120.
- **Simpler ops.** No force-close mechanic at T−120s = no
  overdraft = peak open liability stays at £100/race (env-enforced
  cap). Bank requirement falls naturally out of "1–2 concurrent
  races × £100 × safety margin" = **£300–£500**.
- **No deployment-economics gap.** The env's fc=120 mechanism
  bypasses the per-race budget gate at flatten time, spiking peak
  liability to £310–£410 (well above what the env normally
  reserves). Real Betfair would refuse those over-budget flattens;
  un-flattened pairs would ride naked to settle → fc=0-like
  variance anyway. fc=0 deploy just admits this from the start.

**Trade-off accepted at fc=0:** per-week variance is bigger.
fc=0 across the top-5 spans +£12 to +£215/d on the held-out
window. One unlucky week could see big swings. Mitigation =
size positions for that variance (bank ≥ £300; don't size up
on first-week wins).

## Tier 1 — primary deployment pool

The Tier 1 list is reordered for **fc=0 deployment** with naked
dependence as the secondary sort key (less naked exposure = more
robust to tailwind reversal).

### ⭐ 1df49aa0 (≡ cea2ee94, gen 1) — THE TOP DEPLOY PICK ⭐

**Same agent under two uuids — confirmed twice.**

Stochastic reeval (2026-05-20): both produced eval_day_pnl
65.11172469282474 to 12 decimal places. Argmax reeval
(2026-05-21): both produced identical day_pnl, locked, naked,
bet count, and maturation rate at BOTH fc=0 and fc=120 settings.
This is byte-identical weights — gen 1 of the E3 cohort and gen
1 of the Sortino cohort bred the same offspring because both
selectors ranked gen 0 the same way at this sample. **Deploy
ONE.** Recommended id: **1df49aa0** (cleaner naked exposure on
fc=0 reeval than the cea2ee94 sample).

| Metric | fc=0 stochastic | fc=120 stochastic | fc=0 argmax | fc=120 argmax |
|---|---:|---:|---:|---:|
| day_pnl | +£121/d | +£63/d | −£7/d | +£14/d |
| locked | +£119/d | +£123/d | +£18/d | +£19/d |
| naked | **+£11/d** | **−£3/d** | −£26/d | +£2/d |
| bets/d | ~50 | ~50 | 14 | 17 |
| mat rate | — | — | 0.565 | 0.565 |

- **In-sample**: pnl +£65/d, span £270, worst day −£23, naked_worst −£76
- **Shape**: locked +£119/d on fc=0 — highest of any candidate,
  and held-out naked is essentially zero (+£11). Pure locked-floor
  performance, near-zero tailwind dependence.
- **Why first**: the cleanest tailwind-independent shape in the
  pool. Even if naked goes against you on a week, locked floor
  keeps you at +£90 to +£120/d. Compliance-reviewer-ready.
- **Caveat on argmax**: if live deployment uses argmax (most-
  likely action, deterministic), expect ~3× lower bet rate and a
  sign-flip on net P&L (−£7/d). The +£121 stochastic number is
  contingent on action-sampling at deploy time. **Resolve action
  mode with ai-betfair integration before sizing capital.**

### 11099f65 (E3 cohort, gen 2) — THE CONSISTENT EARNER

- **In-sample**: pnl +£32/d, **span only £96** (tightest in cohort),
  worst day −£30, naked_worst −£75
- **Held-out fc=0**: +£117/d (locked +£101, naked +£30)
- **Held-out fc=120**: +£49/d
- **Shape**: smallest per-day variance we've found. Modest naked
  exposure. Held-out IMPROVED relative to in-sample — most
  replicable agent in the pool.
- **Why deploy**: the "+£32/d safe earner" you can run with
  confidence. Excellent for capital-constrained deployment.

### 57a42db5 (E3 cohort, gen 2) — THE CONSISTENT EARNER (twin)

- **In-sample**: pnl +£48/d, span £198, worst day −£21, naked_worst −£74
- **Held-out fc=0**: +£114/d (locked +£101, naked +£26)
- **Held-out fc=120**: +£48/d (matched in-sample exactly)
- **Shape**: paired with 11099f65 these are robust agents.
  Slightly more variance than 11099f65 but higher mean. Bounded
  worst-day (−£21).
- **Why deploy**: run alongside 11099f65 as a diversified pair.

### ba15deda (R3+E3 cohort, gen 0) — NEW, CLEAN NAKED

- **Held-out fc=0**: **+£221/d** (locked +£116, naked +£113)
- **Held-out fc=120**: +£70/d (locked +£119, naked **+£6**)
- **Shape on fc=120 is the cleanest in the R3+E3 cohort**: locked
  +£119 (highest of any candidate), naked near-zero. The fc=0
  +£221 is impressive but +£113 of it is naked tailwind, so
  expect mean-reversion on weeks without that wind.
- **Why deploy**: locked-floor characteristics on fc=120 say this
  is a genuine arbing agent (not naked-luck). Bonus upside on
  fc=0 weeks when the wind blows your way.

### b4439dd4 (R3+E3 cohort, gen 0) — NEW

- **Held-out fc=0**: +£48/d (locked +£105, naked −£50)
- **Held-out fc=120**: +£68/d (locked +£113, naked +£10)
- **Shape**: locked +£113 on fc=120, naked tightly bounded.
  fc=0 numbers worse than fc=120 because of one bad naked week
  in this window — still positive overall.
- **Why deploy**: clean locked-floor on fc=120 + bounded naked.

### 5862366d (R3+E3 cohort, gen 4) — NEW

- **Held-out fc=0**: +£134/d (locked +£101, naked +£42)
- **Held-out fc=120**: +£60/d (locked +£110, naked +£3)
- **Shape**: locked +£110, naked near-zero on fc=120. Moderate
  upside on fc=0 from naked.
- **Why deploy**: third clean R3+E3 candidate. Forms a 3-agent
  cluster (ba15deda + b4439dd4 + 5862366d) with shape
  characteristics distinct from the E3 cluster (1df49aa0 +
  11099f65 + 57a42db5) — diversification across breeding
  trajectories.

## Tier 2 — secondary / marginal picks

### 850522b9 (E3 cohort, gen 2) — IN-SAMPLE GOLD, HELD-OUT REGRESSED

- **In-sample**: pnl +£65/d, **worst −£20** (gold-standard shape),
  naked_worst −£33
- **Held-out fc=120**: +£45/d (regressed −£20)
- **Held-out fc=0**: +£12/d (regressed −£53)
- **Why caution**: looked perfect in-sample. The held-out
  regression (especially −£53 on fc=0) suggests overfit to
  in-sample. **Marginal pick** — include in a 4+ agent ensemble
  for breadth, don't size large.

## Anti-recommendations (DO NOT DEPLOY)

### Agents outside Tier 1/2

Any agent below this list has lower deploy_score, or fails the
deployable composite filter (worst-day breach < −£30 or
worst-naked-day breach < −£40).

### Sortino cohort agents beyond 1df49aa0

The remaining Sortino top-5 (f51a8bb3, d28b6edb, e8b73472, 7c07f647)
all hold up on held-out but have lower means than their E3-cohort
equivalents. **1df49aa0 is the unique Sortino contribution** (and
is byte-identical to E3's cea2ee94 anyway).

### R3+E3 cohort agents beyond the 3 listed

The remaining R3+E3 top-5 (5ed9504f +£7/d fc=120, 05b5d4bb −£62/d
fc=0) failed the deployable composite filter — the R3+E3 cohort
as a recipe is RETIRED, and only individual agents with
independently-clean shapes survive.

## Deployment strategy

1. **Capital sizing.** Trained at £100/race. fc=0 deployment
   keeps peak open liability at £100/race (env-enforced). With
   1–2 concurrent races + drawdown buffer + safety margin:
   **£300–£500 bank**. Higher than fc=120 would have required,
   but mean is also higher.

2. **Diversification ensemble** (recommended starting allocation):
   - Tier 1 (run all six):
     - **E3 cluster**: 1df49aa0, 11099f65, 57a42db5
     - **R3+E3 cluster**: ba15deda, b4439dd4, 5862366d
   - Different gen / cohort / training-trajectory roots; the
     E3 cluster is more locked-floor-pure; the R3+E3 cluster
     captures more naked upside. Together they cover most regimes.

3. **Action mode is unresolved** (CRITICAL for live deploy):
   - The argmax-eval probe on cea2ee94/1df49aa0 showed the
     deterministic (argmax) policy trades ~3× less than the
     stochastic (sampled) policy and produces a sign-flip on net
     P&L for that one agent under fc=0.
   - **If ai-betfair integration uses argmax** (likely for
     auditability + repeatability), the held-out stochastic
     numbers above OVERSTATE expected live performance — paper-
     trade first to calibrate.
   - **If it uses sampled actions** (matches reeval mode), held-
     out numbers are direct.
   - **Argmax-fc=0 numbers for all 3 E3 candidates (2026-05-21):**

     | Agent | argmax fc=0 | stochastic fc=0 | argmax bets/d |
     |---|---:|---:|---:|
     | 1df49aa0/cea2ee94 | −£7/d | +£121/d | 14 |
     | 11099f65 | +£3/d | +£117/d | 7 |
     | 57a42db5 | +£18/d | +£114/d | 4 |

     Cohort argmax mean is **+£4/d** vs stochastic **+£117/d** —
     a 26× collapse. Argmax bet counts are 7–13× lower than
     stochastic; 57a42db5 places only 4 bets/day = essentially
     silent. The mean argmax P&L is noise around zero. **Every
     fc=0 number above is contingent on live trading using
     sampled actions, not argmax.**

4. **Force-close exposure check** (paper-trade requirement):
   The env's fc=120 mechanism deliberately overdrafts at T−120s,
   which real Betfair would refuse on agents with >£100 open
   liability. We're deploying at fc=0 to sidestep this, but the
   trained policies still post passive opens that can ride to
   settle naked. Paper-trade to measure the refusal rate on
   in-flight exposure (Betfair refuses bets that would push
   open exposure above account balance).

5. **What's NOT in the pool anymore**:
   - **Sortino selector** — retired 2026-05-20 (cohort mean
     lower than E3 selector across both fc settings).
   - **R3 (β=0.01) recipe** — retired 2026-05-21 (cohort mean
     regression vs E3 cohort at both fc=120 and fc=0).
   - Both retire decisions share the same lesson: naked-
     variance interventions look promising at probe scale (5
     agents, 5 gens) but don't compound under full-cohort
     (12×8gen) breeding. Future experiments should focus on
     env priors that REMOVE bad decisions (E3-style) or
     curriculum-side changes — not naked-side reward shaping.
