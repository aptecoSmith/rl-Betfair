# Recipe sensitivity sweep — findings

Cohort: `_recipe_sensitivity_sweep_1779662659`
Agents: 43 (1 generation, no GA)
Training: 12 days × 5 eval days per agent, BC pretrain disabled,
frozen C11 direction head, direction gate enabled (policy-side).

## Cohort-wide summary (per-agent means averaged over 5 eval days)

| metric | mean | median | min | max |
|---|---|---|---|---|
| mean_locked_pnl | +2.4 | +1.7 | +0.0 | +6.5 |
| mean_naked_pnl | +15.6 | +11.5 | -51.5 | +98.2 |
| mean_force_closed_pnl | -25.2 | -29.8 | -48.5 | +0.0 |
| mean_day_pnl | -70.6 | -72.7 | -144.8 | +0.0 |
| mean_bet_count | +131.5 | +137.0 | +0.0 | +177.0 |

Profitable agents (mean P&L > 0 across 5 eval days): **0/43**.

## Spearman ρ matrix — swept knobs × outcome metrics

Per-agent rank correlation between gene value and per-agent aggregate
(mean across 5 eval days). |ρ| ≥ 0.30 is the cutoff for "real"
lever; |ρ| < 0.15 is likely noise at N=43.

| knob | locked | σ(naked) | fc_pnl | day_pnl | bets | mat% | cls% | fc% |
|---|---|---|---|---|---|---|---|---|
| stop_loss_pnl_threshold | +0.05 | -0.21 | **-0.91** | -0.20 | +0.15 | -0.02 | **+0.52** | **+0.89** |
| predictor_feature_gain | **+0.52** | -0.12 | +0.06 | **-0.31** | +0.13 | +0.09 | +0.04 | +0.05 |
| arb_spread_target_lock_pct | -0.13 | -0.05 | -0.06 | +0.13 | +0.08 | **-0.46** | +0.08 | -0.06 |
| reward_clip | +0.24 | -0.05 | -0.12 | -0.08 | -0.20 | +0.11 | **+0.43** | +0.08 |
| direction_gate_threshold | -0.19 | +0.02 | **+0.40** | -0.02 | -0.01 | -0.14 | -0.03 | **-0.41** |
| entropy_coeff | -0.01 | +0.15 | -0.25 | **-0.38** | **+0.38** | -0.20 | +0.18 | +0.19 |
| gae_lambda | +0.04 | +0.06 | +0.21 | +0.04 | -0.21 | **+0.41** | -0.11 | -0.22 |
| open_cost | -0.03 | -0.08 | -0.06 | -0.13 | -0.05 | -0.17 | **+0.37** | +0.03 |
| mark_to_market_weight | +0.02 | -0.20 | -0.18 | +0.11 | **+0.35** | -0.10 | -0.02 | +0.28 |
| hidden_size | -0.19 | **-0.34** | -0.17 | +0.20 | -0.14 | -0.11 | **+0.35** | +0.12 |
| naked_loss_scale | **-0.34** | +0.15 | -0.08 | +0.02 | +0.09 | -0.22 | -0.10 | -0.01 |
| naked_variance_penalty_beta | +0.18 | +0.06 | -0.01 | -0.06 | +0.22 | +0.01 | -0.30 | +0.03 |
| alpha_lr | +0.30 | -0.15 | -0.09 | -0.18 | -0.10 | +0.14 | -0.20 | +0.19 |
| clip_range | +0.20 | +0.14 | +0.09 | -0.17 | -0.24 | -0.05 | -0.09 | -0.04 |
| value_coeff | -0.09 | +0.11 | -0.16 | -0.15 | -0.07 | -0.23 | +0.28 | +0.12 |
| fill_prob_loss_weight | +0.00 | +0.04 | -0.02 | -0.20 | +0.10 | -0.26 | +0.05 | -0.07 |
| matured_arb_bonus_weight | -0.09 | -0.15 | +0.20 | +0.08 | +0.09 | -0.08 | +0.01 | -0.21 |
| learning_rate | +0.10 | -0.20 | +0.18 | +0.23 | -0.14 | +0.21 | -0.10 | -0.09 |
| mini_batch_size | +0.01 | -0.08 | -0.02 | +0.21 | -0.01 | -0.17 | -0.18 | +0.05 |
| risk_loss_weight | -0.00 | -0.05 | +0.17 | +0.18 | -0.21 | -0.08 | +0.21 | -0.18 |
| mature_prob_loss_weight | +0.16 | -0.14 | -0.10 | -0.04 | -0.13 | -0.04 | +0.18 | +0.14 |

Bold values: |ρ| ≥ 0.30 (real lever at N=43).

## Top 20 strongest correlations

| rank | knob | metric | ρ | direction |
|---|---|---|---|---|
| 1 | `stop_loss_pnl_threshold` | `mean_force_closed_pnl` | **-0.911** | ↑ knob → ↓ metric |
| 2 | `stop_loss_pnl_threshold` | `fc_pct` | **+0.892** | ↑ knob → ↑ metric |
| 3 | `stop_loss_pnl_threshold` | `mean_arbs_force_closed` | **+0.848** | ↑ knob → ↑ metric |
| 4 | `stop_loss_pnl_threshold` | `mean_closed_pnl` | **-0.712** | ↑ knob → ↓ metric |
| 5 | `stop_loss_pnl_threshold` | `mean_arbs_closed` | **+0.627** | ↑ knob → ↑ metric |
| 6 | `stop_loss_pnl_threshold` | `cls_pct` | **+0.521** | ↑ knob → ↑ metric |
| 7 | `predictor_feature_gain` | `mean_locked_pnl` | **+0.516** | ↑ knob → ↑ metric |
| 8 | `stop_loss_pnl_threshold` | `mean_pairs_opened` | **+0.490** | ↑ knob → ↑ metric |
| 9 | `arb_spread_target_lock_pct` | `mat_pct` | **-0.459** | ↑ knob → ↓ metric |
| 10 | `arb_spread_target_lock_pct` | `mean_arbs_completed` | **-0.456** | ↑ knob → ↓ metric |
| 11 | `reward_clip` | `cls_pct` | **+0.428** | ↑ knob → ↑ metric |
| 12 | `direction_gate_threshold` | `fc_pct` | **-0.412** | ↑ knob → ↓ metric |
| 13 | `entropy_coeff` | `mean_arbs_closed` | **+0.408** | ↑ knob → ↑ metric |
| 14 | `gae_lambda` | `mat_pct` | **+0.405** | ↑ knob → ↑ metric |
| 15 | `direction_gate_threshold` | `mean_force_closed_pnl` | **+0.398** | ↑ knob → ↑ metric |
| 16 | `entropy_coeff` | `mean_bet_count` | **+0.384** | ↑ knob → ↑ metric |
| 17 | `gae_lambda` | `mean_arbs_completed` | **+0.381** | ↑ knob → ↑ metric |
| 18 | `entropy_coeff` | `mean_day_pnl` | **-0.376** | ↑ knob → ↓ metric |
| 19 | `entropy_coeff` | `mean_pairs_opened` | **+0.369** | ↑ knob → ↑ metric |
| 20 | `open_cost` | `cls_pct` | **+0.369** | ↑ knob → ↑ metric |

## Pareto frontier on (mean_locked, σ_naked)

Pareto-non-dominated agents: maximise mean_locked AND minimise
σ_naked (cross-day leg-variance proxy). Smaller σ_naked = more
deployment-stable; larger mean_locked = more spread captured.

| agent | mean_locked | σ(naked) | open_cost | dir_gate | arb_lock | naked_scale | mature_prob_w |
|---|---|---|---|---|---|---|---|
| `4bf112e1` | +6.50 | 60.1 | 0.13 | 0.22 | 0.017 | 0.06 | 3.49 |
| `f4f32e03` | +4.57 | 56.9 | 1.51 | 0.47 | 0.014 | 0.21 | 4.91 |
| `a9de8248` | +1.79 | 43.2 | 0.66 | 0.28 | 0.048 | 0.11 | 2.78 |
| `51462a4c` | +1.79 | 30.7 | 1.14 | 0.35 | 0.007 | 0.27 | 4.48 |
| `d6261655` | +0.00 | 0.0 | 0.05 | 0.39 | 0.024 | 0.61 | 4.78 |

## Interpretation guide

- **Real levers** (|ρ| ≥ 0.30): worth keeping in the GA's evolvable set.
- **Weak levers** (0.15 ≤ |ρ| < 0.30): may matter in combination,
  or may be noise — flag for follow-up.
- **Inert knobs** (|ρ| < 0.15): consider pinning at a single value
  to free up the GA's variance budget for the real levers.

## Production-cohort recipe recommendation

### Keep evolving (real levers)
- `stop_loss_pnl_threshold` (max |ρ| = 0.91)
- `predictor_feature_gain` (max |ρ| = 0.52)
- `arb_spread_target_lock_pct` (max |ρ| = 0.46)
- `reward_clip` (max |ρ| = 0.43)
- `direction_gate_threshold` (max |ρ| = 0.41)
- `entropy_coeff` (max |ρ| = 0.41)
- `gae_lambda` (max |ρ| = 0.41)
- `open_cost` (max |ρ| = 0.37)
- `mark_to_market_weight` (max |ρ| = 0.35)
- `hidden_size` (max |ρ| = 0.35)
- `naked_loss_scale` (max |ρ| = 0.34)

### Consider keeping (weak — investigate combination)
- `naked_variance_penalty_beta` (max |ρ| = 0.30)
- `alpha_lr` (max |ρ| = 0.30)
- `clip_range` (max |ρ| = 0.28)
- `value_coeff` (max |ρ| = 0.28)
- `fill_prob_loss_weight` (max |ρ| = 0.28)
- `matured_arb_bonus_weight` (max |ρ| = 0.23)
- `learning_rate` (max |ρ| = 0.23)
- `mini_batch_size` (max |ρ| = 0.21)
- `risk_loss_weight` (max |ρ| = 0.21)
- `mature_prob_loss_weight` (max |ρ| = 0.18)

### Pin (inert at N=43)

---

Generated by `plans/recipe-sensitivity-sweep/analyze.py`.
Raw data in `plans/recipe-sensitivity-sweep/findings.json`.
