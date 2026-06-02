# Recipe sensitivity sweep

## What

A wide random sweep across the cohort recipe's knob space, designed
to map which knobs move outcomes — not to find a winning agent.

## Why

The current frozen-head cohort was launched with a recipe inherited
from probe2, an integration test. probe2 had env-side priors
(`open_cost`, `direction_gate_enabled`) deliberately OFF so the
aux-head wiring could be observed in isolation. That "everything off"
baseline was carried forward into a 13h cohort, so the GA was being
asked to evolve a `direction_gate_threshold` gene whose master switch
was disabled, while `open_cost = 0` left selectivity entirely
unsupervised.

The operator's instinct: we have 20-ish knobs, we don't have a
principled basis for which are levers and which are noise, and we
keep burning compute on configurations that turned out to omit
load-bearing pieces. A sensitivity sweep produces that basis.

## Design

- **60 agents, 1 generation, no GA.** The point isn't selection; it's
  a Spearman ρ matrix between every knob and every outcome metric.
- **12 training days, 5 eval days.** 12 days is enough for PPO to
  move past the initial transient and for aux heads to actually
  train. 5 eval days gives reasonable per-agent metric stability.
- **Uniform random sampling** per gene per agent. At N=60 across
  15 knobs this gives readable univariate ρ for any effect size
  ≥ 0.25. (LHS would buy ~2× better marginal stratification but
  requires new code; uniform is acceptable at this N.)
- **BC pretrain DISABLED** (`--bc-pretrain-steps 0`). BC clamps every
  agent to the same aggressive starting point, absorbing variance
  that should be attributable to the swept knobs. Production recipe
  re-adds BC after we know what to keep.
- **Frozen C11 direction head** loaded as a fixed component (not
  under test). This matches what production will use.
- **Direction gate ENABLED** so `direction_gate_threshold` is no
  longer dead code.
- 14 Phase-5 knobs enabled for evolution, plus the 7 Phase-3 knobs
  that always evolve. Excluded: `direction_prob_loss_weight` and
  `bc_direction_target_weight` (forced 0 by frozen head); the 4
  value-mode knobs (no-ops in scalping).

## Knobs swept (15 active)

| group | knobs | range |
|---|---|---|
| PPO hyperparams (Phase 3, auto) | learning_rate | [1e-5, 1e-3] log |
| | entropy_coeff | [1e-4, 1e-1] log |
| | clip_range | [0.1, 0.3] |
| | gae_lambda | [0.9, 0.99] |
| | value_coeff | [0.25, 1.0] |
| | mini_batch_size | {32, 64, 128} |
| | hidden_size | {64, 128, 256} |
| Reward shaping | open_cost | [0, 2] |
| | matured_arb_bonus_weight | [0, 5] |
| | mark_to_market_weight | [0, 0.10] |
| | naked_loss_scale | [0, 1] |
| | stop_loss_pnl_threshold | [0, 0.30] |
| | naked_variance_penalty_beta | [0, 0.10] |
| | reward_clip | [1, 10] |
| Aux heads | fill_prob_loss_weight | [0, 0.30] |
| | mature_prob_loss_weight | [1, 5] |
| | risk_loss_weight | [0, 0.30] |
| Env priors | arb_spread_target_lock_pct | [0.005, 0.05] |
| | direction_gate_threshold | [0.5, 0.95] (with `--direction-gate-enabled`) |
| Other | alpha_lr | [1e-2, 1e-1] log |
| | predictor_feature_gain | [0, 1] |

## What we'll write up

After completion, `findings.md` contains:

### Outcome-side analysis (selection metrics)

- Spearman ρ heatmap (15 swept knobs × 6 metrics: locked, σ_naked_leg,
  fc%, mat%, bets, raw_pnl).
- Univariate scatter for top |ρ| knobs — direction of effect + shape.
- Pareto frontier on (locked, σ_naked_leg).

### Behavioural-divergence analysis (the "do knobs reach the action distribution?" question)

The operator's working hypothesis (2026-05-24): the cohort recipe may
have many weak levers that don't actually change agent behaviour
because the gradient effect gets absorbed by PPO advantage
normalisation / advantaged by frozen-head shared signal / drowned by
market structural cost. Worth testing directly.

For each pair of agents (i, j) with very different gene values on knob
K (top quartile vs bottom quartile of K), compute behavioural distance:

- **Bet-rate distance:** |bets_i - bets_j| / mean(bets_i + bets_j)
- **Side-mix distance:** |back_ratio_i - back_ratio_j|
- **Price-band distance:** JS-divergence of opened-price histograms
  binned [1.0-2.0, 2.0-4.0, 4.0-8.0, 8.0-15.0, 15.0-30.0, 30.0+]
- **Drift-at-open distance:** Wasserstein distance over the per-pair
  `drift_at_open_ticks` distribution

For each knob K, plot (gene-space distance, behaviour-space distance)
across all (i, j) pairs. Knobs that DON'T cluster behavioural
distance with gene distance are weak levers — they change internal
representations but not actions. These should be pruned from
production cohorts and replaced with env-side priors that DO act on
behaviour.

Knobs that DO show monotone (gene-space, behaviour-space) clustering
are real levers worth keeping in the GA.

### Recommendation

Concrete recipe for the next production cohort:
- Which knobs to evolve (real levers)
- Which to pin (one good value identified by the sweep)
- Which to retire (weak levers — no behavioural impact)

## Out of scope

- No GA selection pressure.
- No multi-generation breeding.
- Frozen direction head is the fixed component — head architecture
  is not under test here (that was the c0-c20 sweep).
- BC pretrain mechanics not tested (re-added at production time).

## Budget

- ~12 min/agent (12 train × 40s + 5 eval × 50s + startup)
- 60 agents × 12 min ≈ 12h wall.
- Operator's budget: 12-14h.

## Day stratification

Training days mix small/medium/large by tick volume:
- Small (~4-7k ticks): 2026-04-19, 2026-04-26
- Medium (~9-11k ticks): 2026-04-06, 2026-04-13, 2026-04-15, 2026-04-20, 2026-04-22
- Large (~11-13k ticks): 2026-04-08, 2026-04-09, 2026-04-11, 2026-04-12, 2026-05-02

Eval days (5 from held-out 10-day pool, monitor pool untouched):
2026-04-10, 2026-04-17, 2026-04-21, 2026-05-03, 2026-05-06.

## Hard constraints

- Held-out invariant preserved: no eval day in training set.
- Frozen head not retrained per agent (would defeat the point).
- No GA — this is a sensitivity sweep, not a fitness search.
