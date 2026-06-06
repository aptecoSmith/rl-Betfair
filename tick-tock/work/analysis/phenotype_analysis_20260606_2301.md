# Phenotype analysis — pbt_genes_v2

_Generated 2026-06-06 23:01 by `tools/phenotype_analysis.py`._

- **Source:** `registry\pbt_genes_v2\model_register.csv` (model_register.csv)
- **Discovery scope:** `--tick-only` — no `era_type` column; all rows treated as tick (legacy full-width campaign).
- **Agents (n):** 80
- **Generations:** gen0=16, gen1=16, gen2=16, gen3=16, gen4=16
- **Architectures:** transformer=29, lstm_h256=18, lstm_h64=13, lstm_h1024=11, lstm_h512=6, lstm_h128=3
- **Varying genes analysed:** 46 (of 50 total gene columns)

> **Correlation is not causation.** PBT genes co-vary (elite agents carry whole gene vectors forward; offspring inherit blocks), and architecture is itself a gene — so a gene's apparent effect may be a proxy for the architecture or for a co-inherited gene. Treat every driver below as a hypothesis to A/B, not a proven lever.

## Behaviour summary (per-agent rates / values)

| behaviour | n | mean | std | min | max | caveat |
|---|---|---|---|---|---|---|
| maturation_rate | 80 | 0.0173 | 0.05271 | 0 | 0.2713 |  |
| close_rate | 80 | 0.2887 | 0.1665 | 0 | 0.75 |  |
| force_close_rate | 80 | 0 | 0 | 0 | 0 | ZERO-VARIANCE (constant = 0) — no gene can correlate; behaviour is pinned/inactive in this cohort |
| naked_rate | 80 | 0.5767 | 0.1806 | 0.0625 | 0.7838 |  |
| stop_close_rate | 80 | 0.1155 | 0.1051 | 0.01408 | 0.5625 |  |
| locked_pnl | 80 | 1.73 | 5.651 | 0 | 32.05 |  |
| naked_sd | 80 | 224.6 | 92.02 | 47.63 | 431.9 |  |

## Gene drivers per behaviour

Top 6 genes by |Spearman| for each behaviour. ρ = Spearman (rank, robust to outliers/nonlinearity); r = Pearson (linear). Sign shows direction: + means the gene and the behaviour rise together.

### maturation_rate

| gene | ρ (Spearman) | p(ρ) | r (Pearson) | p(r) | n | interpretation |
|---|---|---|---|---|---|---|
| bc_learning_rate | +0.414 | <.001 | +0.477 | <.001 | 80 | moderate: higher `bc_learning_rate` => higher maturation_rate |
| bc_pretrain_steps | +0.284 | 0.011 | +0.241 | 0.032 | 80 | weak: higher `bc_pretrain_steps` => higher maturation_rate |
| direction_gate_warmup_eps | +0.284 | 0.011 | +0.307 | 0.006 | 80 | weak: higher `direction_gate_warmup_eps` => higher maturation_rate |
| predictor_lean_obs | -0.264 | 0.018 | -0.317 | 0.004 | 80 | weak: lower `predictor_lean_obs` => lower maturation_rate |
| open_cost | +0.260 | 0.020 | +0.144 | 0.202 | 80 | weak: higher `open_cost` => higher maturation_rate |
| risk_loss_weight | +0.252 | 0.024 | +0.242 | 0.031 | 80 | weak: higher `risk_loss_weight` => higher maturation_rate |

### close_rate

| gene | ρ (Spearman) | p(ρ) | r (Pearson) | p(r) | n | interpretation |
|---|---|---|---|---|---|---|
| direction_gate_enabled | +0.609 | <.001 | +0.800 | <.001 | 80 | strong: higher `direction_gate_enabled` => higher close_rate |
| use_direction_predictor | +0.311 | 0.005 | +0.373 | <.001 | 80 | moderate: higher `use_direction_predictor` => higher close_rate |
| value_edge_threshold | +0.238 | 0.033 | +0.228 | 0.042 | 80 | weak: higher `value_edge_threshold` => higher close_rate |
| learning_rate | -0.219 | 0.051 | -0.213 | 0.058 | 80 | weak: lower `learning_rate` => lower close_rate (not significant) |
| lay_price_max | -0.209 | 0.063 | -0.306 | 0.006 | 80 | weak: lower `lay_price_max` => lower close_rate (not significant) |
| mature_prob_open_threshold | -0.187 | 0.097 | -0.225 | 0.045 | 80 | weak: lower `mature_prob_open_threshold` => lower close_rate (not significant) |

### force_close_rate

_ZERO-VARIANCE (constant = 0) — no gene can correlate; behaviour is pinned/inactive in this cohort. No gene correlations are defined — skipping._

### naked_rate

| gene | ρ (Spearman) | p(ρ) | r (Pearson) | p(r) | n | interpretation |
|---|---|---|---|---|---|---|
| direction_gate_enabled | -0.678 | <.001 | -0.871 | <.001 | 80 | strong: lower `direction_gate_enabled` => lower naked_rate |
| stop_loss_pnl_threshold | +0.606 | <.001 | +0.423 | <.001 | 80 | strong: higher `stop_loss_pnl_threshold` => higher naked_rate |
| lay_price_max | +0.596 | <.001 | +0.519 | <.001 | 80 | strong: higher `lay_price_max` => higher naked_rate |
| use_direction_predictor | -0.490 | <.001 | -0.545 | <.001 | 80 | moderate: lower `use_direction_predictor` => lower naked_rate |
| race_confidence_threshold | -0.407 | <.001 | -0.334 | 0.002 | 80 | moderate: lower `race_confidence_threshold` => lower naked_rate |
| direction_gate_warmup_eps | -0.406 | <.001 | -0.360 | 0.001 | 80 | moderate: lower `direction_gate_warmup_eps` => lower naked_rate |

### stop_close_rate

| gene | ρ (Spearman) | p(ρ) | r (Pearson) | p(r) | n | interpretation |
|---|---|---|---|---|---|---|
| stop_loss_pnl_threshold | -0.931 | <.001 | -0.822 | <.001 | 80 | strong: lower `stop_loss_pnl_threshold` => lower stop_close_rate |
| lay_price_max | -0.495 | <.001 | -0.357 | 0.001 | 80 | moderate: lower `lay_price_max` => lower stop_close_rate |
| bc_target_entropy_warmup_eps | -0.376 | <.001 | -0.308 | 0.005 | 80 | moderate: lower `bc_target_entropy_warmup_eps` => lower stop_close_rate |
| bc_direction_target_weight | +0.351 | 0.001 | +0.153 | 0.176 | 80 | moderate: higher `bc_direction_target_weight` => higher stop_close_rate |
| direction_gate_warmup_eps | +0.344 | 0.002 | +0.276 | 0.013 | 80 | moderate: higher `direction_gate_warmup_eps` => higher stop_close_rate |
| transformer_heads | -0.339 | 0.002 | -0.245 | 0.029 | 80 | moderate: lower `transformer_heads` => lower stop_close_rate |

### locked_pnl

| gene | ρ (Spearman) | p(ρ) | r (Pearson) | p(r) | n | interpretation |
|---|---|---|---|---|---|---|
| direction_gate_warmup_eps | +0.306 | 0.006 | +0.277 | 0.013 | 80 | moderate: higher `direction_gate_warmup_eps` => higher locked_pnl |
| bc_learning_rate | +0.296 | 0.008 | +0.474 | <.001 | 80 | weak: higher `bc_learning_rate` => higher locked_pnl |
| predictor_lean_obs | -0.258 | 0.021 | -0.299 | 0.007 | 80 | weak: lower `predictor_lean_obs` => lower locked_pnl |
| use_direction_predictor | +0.231 | 0.039 | +0.201 | 0.074 | 80 | weak: higher `use_direction_predictor` => higher locked_pnl |
| direction_gate_enabled | +0.217 | 0.054 | +0.063 | 0.581 | 80 | weak: higher `direction_gate_enabled` => higher locked_pnl (not significant) |
| direction_prob_loss_weight | +0.214 | 0.057 | +0.435 | <.001 | 80 | weak: higher `direction_prob_loss_weight` => higher locked_pnl (not significant) |

### naked_sd

| gene | ρ (Spearman) | p(ρ) | r (Pearson) | p(r) | n | interpretation |
|---|---|---|---|---|---|---|
| mark_to_market_weight | -0.302 | 0.006 | -0.252 | 0.024 | 80 | moderate: lower `mark_to_market_weight` => lower naked_sd |
| open_cost | +0.216 | 0.055 | +0.220 | 0.050 | 80 | weak: higher `open_cost` => higher naked_sd (not significant) |
| learning_rate | -0.207 | 0.066 | -0.260 | 0.020 | 80 | weak: lower `learning_rate` => lower naked_sd (not significant) |
| mature_prob_loss_weight | -0.195 | 0.083 | -0.159 | 0.159 | 80 | weak: lower `mature_prob_loss_weight` => lower naked_sd (not significant) |
| transformer_depth | -0.185 | 0.100 | -0.168 | 0.137 | 80 | weak: lower `transformer_depth` => lower naked_sd (not significant) |
| bc_learning_rate | +0.168 | 0.137 | +0.124 | 0.273 | 80 | weak: higher `bc_learning_rate` => higher naked_sd (not significant) |

## Combined-recipe suggestion

Goal: jointly **raise maturation_rate + close_rate** while **lowering force_close_rate** (and stop_close_rate where it is the active bail channel). The genes below are those whose Spearman direction helps at least one of these targets at |ρ|≥0.20, p≤0.10. `direction` is which way to move the gene; `approx value` is the median value the *already-good* agents used (descriptive, not a causal optimum).

**Top picks** (strongest direction-coherence first — set these toward the stated value/direction):

- **increase `stop_loss_pnl_threshold`** (toward ~0.277) — helps 1 target(s): stop_close_rate-: increase gene (rho=-0.93, p=0.000)
- **increase `direction_gate_enabled`** (toward ~1) — helps 1 target(s): close_rate+: increase gene (rho=+0.61, p=0.000)
- **increase `bc_target_entropy_warmup_eps`** (toward ~18) — helps 1 target(s): stop_close_rate-: increase gene (rho=-0.38, p=0.001)
- **decrease `bc_direction_target_weight`** (toward ~0.2232) — helps 1 target(s): stop_close_rate-: decrease gene (rho=+0.35, p=0.001)
- **increase `bc_pretrain_steps`** (toward ~0) — helps 1 target(s): maturation_rate+: increase gene (rho=+0.28, p=0.011)
- **decrease `predictor_lean_obs`** (toward ~1) — helps 1 target(s): maturation_rate+: decrease gene (rho=-0.26, p=0.018)
- **increase `open_cost`** (toward ~1.889) — helps 1 target(s): maturation_rate+: increase gene (rho=+0.26, p=0.020)
- **increase `risk_loss_weight`** (toward ~0.1138) — helps 1 target(s): maturation_rate+: increase gene (rho=+0.25, p=0.024)

_7 gene(s) marked ⚠ show a **direction conflict** — they move one way to help one target and the opposite way to help another. Where the helping association is much stronger than the opposing one (e.g. `direction_gate_enabled` close_rate ρ=+0.63 vs stop_close ρ=+0.25), follow the dominant direction but expect the secondary trade-off._

Full candidate table (top 16 by direction coherence; `conflict?` = gene must move both ways across targets):

| gene | suggested direction | approx value | targets helped | conflict? | evidence |
|---|---|---|---|---|---|
| stop_loss_pnl_threshold | **increase** | 0.277 | 1 | no | stop_close_rate-: increase gene (rho=-0.93, p=0.000) |
| direction_gate_enabled | **increase** | 1 | 1 | no | close_rate+: increase gene (rho=+0.61, p=0.000) |
| bc_target_entropy_warmup_eps | **increase** | 18 | 1 | no | stop_close_rate-: increase gene (rho=-0.38, p=0.001) |
| bc_direction_target_weight | **decrease** | 0.2232 | 1 | no | stop_close_rate-: decrease gene (rho=+0.35, p=0.001) |
| bc_pretrain_steps | **increase** | 0 | 1 | no | maturation_rate+: increase gene (rho=+0.28, p=0.011) |
| predictor_lean_obs | **decrease** | 1 | 1 | no | maturation_rate+: decrease gene (rho=-0.26, p=0.018) |
| open_cost | **increase** | 1.889 | 1 | no | maturation_rate+: increase gene (rho=+0.26, p=0.020) |
| risk_loss_weight | **increase** | 0.1138 | 1 | no | maturation_rate+: increase gene (rho=+0.25, p=0.024) |
| matured_arb_bonus_weight | **decrease** | 1.054 | 1 | no | stop_close_rate-: decrease gene (rho=+0.25, p=0.027) |
| predictor_feature_gain | **increase** | 0.7905 | 1 | no | stop_close_rate-: increase gene (rho=-0.24, p=0.029) |
| value_edge_threshold | **increase** | 0.06079 | 1 | no | close_rate+: increase gene (rho=+0.24, p=0.033) |
| predictor_p_win_back_threshold | **decrease** | 0.2315 | 1 | no | stop_close_rate-: decrease gene (rho=+0.23, p=0.041) |
| predictor_p_win_lay_threshold | **decrease** | 0.2154 | 1 | no | stop_close_rate-: decrease gene (rho=+0.22, p=0.046) |
| naked_variance_penalty_beta | **decrease** | 0.03691 | 1 | no | stop_close_rate-: decrease gene (rho=+0.22, p=0.050) |
| learning_rate | **decrease** | 9.957e-05 | 1 | no | close_rate+: decrease gene (rho=-0.22, p=0.051) |
| direction_gate_threshold | **decrease** | 0.3535 | 1 | no | maturation_rate+: decrease gene (rho=-0.21, p=0.067) |

### Recipe caveats

1. **Correlation ≠ causation.** These are observational associations across a co-evolving population, not interventional effects. Confirm any gene by an A/B that pins it while holding the rest at cohort defaults.
2. **Sample size n=80.** p-values are uncorrected for multiple comparisons (dozens of gene×behaviour tests); a p≈0.05 here is weak evidence.
3. **Architecture confound.** `architecture` (and the hidden_size / transformer_* structural genes) is itself an evolved gene. If it appears as a driver, the 'recipe' may really be 'use that architecture', and other genes may be proxies for the architecture mix. Stratify by architecture before trusting a non-structural gene.
4. **Behaviour coupling.** maturation/close/naked/force_close rates share the pairs_opened denominator and sum to ~1, so raising one rate mechanically tends to lower others. A gene that raises close_rate may lower maturation_rate as an accounting side-effect, not a real trade-off in the policy.
5. **force_close_rate is identically 0 in this cohort** (force_close_before_off_seconds pinned to 0 during training — the standard 'keep naked-variance signal' setup). The recipe's 'lower force_close' target therefore contributed nothing here; `stop_close_rate` is the live bail channel and is used as the proxy. Re-evaluate force-close behaviour on a held-out run with force_close enabled.

## Appendix — pinned / constant genes (excluded)

4 gene columns showed no variation across the cohort and were excluded from correlation (a constant has no defined correlation):

- `direction_force_close_seconds` = 60.0
- `direction_horizon_ticks` = 60
- `direction_threshold_ticks` = 5
- `force_close_before_off_seconds` = 0.0
