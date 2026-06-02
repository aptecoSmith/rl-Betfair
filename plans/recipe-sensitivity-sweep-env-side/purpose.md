# Env-side sensitivity sweep

## What

A short multi-cohort fan-out testing env-side cohort-wide knobs that
the main gradient sweep cannot reach. Operator's working hypothesis
(`feedback_remove_decisions_beats_teaching` memory): env priors that
*refuse* bad decisions tend to work; reward-shaping gradients that
*teach* the agent away from bad decisions tend not to. The main
gradient sweep tests the gradient hypothesis. This sweep tests the
env-prior hypothesis.

## Why a separate sweep

The env-side knobs in scope here are **cohort-wide**, not per-agent
genes — they're set once at env-construction and apply to every
agent. So per-agent random sampling can't test them. The only way to
test them is a fan-out: launch N cohorts, each at a different env-knob
combination.

## What's being tested

Six cells, each a small cohort (4 agents × 3 train days × 5 eval
days). All agents within a cell share the env-side config but have
random Phase-3 PPO genes so we average over PPO-hyperparam noise.

| cell | hypothesis | knob delta from C0 baseline |
|---|---|---|
| **C0 baseline** | reference point | `force_close_before_off_seconds=120` only |
| **C1 tight FC** | shorter naked-variance window helps | `force_close_before_off_seconds=60` |
| **C2 back pwin** | refuse low-win-prob backs | `--predictor-p-win-back-threshold 0.20` |
| **C3 lay pwin** | refuse high-win-prob lays | `--predictor-p-win-lay-threshold 0.50` |
| **C4 race conf** | skip low-confidence races | `--race-confidence-threshold 0.35` |
| **C5 dir gate 0.30** | moderate direction gate refusal (~40-50 % blocked) | `direction_gate_threshold=0.30` |
| **C6 dir gate 0.45** | aggressive direction gate refusal (~80-85 % blocked) | `direction_gate_threshold=0.45` |
| **C7 ALL ON** | stack the priors | C1 + C2 + C3 + C4 + `--lay-price-max 20` + `direction_gate_threshold=0.35` |

**Threshold calibration**: the C11 frozen head was trained unweighted
(positive class rate ~18 %) so its output probabilities cluster low —
observed `max(direction_back_prob, direction_lay_prob)` per-runner
distribution has mean ~0.32 and max ~0.84 on placed bets. Threshold
0.30 ≈ refuse ~40-50 % of opens (moderate); 0.45 ≈ refuse ~80-85 %
(strict). Compare to the old policy-side gene range (0.5, 0.95)
which corresponded to "starve PPO entirely" under the C11 head — the
range was recalibrated to (0.20, 0.50) in genes.py on 2026-05-25.

C0 is the same config as the main gradient sweep's baseline (so the
gradient sweep's results are directly comparable to C0).

Phase-5 genes are FIXED to the cohort-wide pre-Phase-5 default within
each cell (so the env-side knob is the only variable). Specifically:

- `open_cost = 0.0`, `matured_arb_bonus_weight = 0.0`,
  `naked_loss_scale = 1.0`, `stop_loss_pnl_threshold = 0.0`,
  `naked_variance_penalty_beta = 0.0`,
  `direction_gate_threshold = 0.5` (no-op floor),
  `predictor_feature_gain = 1.0`.
- Aux-head loss weights pinned at moderate values:
  `fill_prob_loss_weight = 0.1`, `mature_prob_loss_weight = 2.0`,
  `risk_loss_weight = 0.1`.
- Other Phase-5: `mark_to_market_weight = 0.05`, `alpha_lr = 1e-2`,
  `reward_clip = 10`, `arb_spread_target_lock_pct = 0.02`.

Phase-3 PPO genes (`learning_rate`, `entropy_coeff`, etc.) evolve
randomly across the 4 agents per cell so the cell-mean is robust to
per-agent PPO noise.

## What we'll write up

`findings.md` with:

- Per-cell summary: mean ± std of `locked_pnl`, `naked_pnl`, `bets`,
  `fc%`, `mat%`, `raw_pnl` across 4 agents × 5 eval days = 20
  datapoints per cell.
- Within-cell behavioural distribution (price band, side mix,
  drift-at-open) — same metrics as the gradient sweep's behavioural
  analysis.
- Effect-size table: each cell vs C0, signed delta and significance
  (Welch's t).
- Recommendation for the next production cohort: which env-side
  priors to turn on, at what values.

## Budget

- Per agent: ~4 min (3 train × 40s + 5 eval × 10s + startup).
- Per cell (4 agents): ~16 min.
- 8 cells × 17 min (with overhead) ≈ **136 min ≈ 2.3h**.
- Operator's budget: 2h (slightly over — acceptable, this is overnight).

## Hard constraints

- All cells use the same training days as the gradient sweep
  (12 days: 2026-04-06, 04-08, 04-09, 04-11, 04-12, 04-13, 04-15,
  04-19, 04-20, 04-22, 04-26, 05-02) — but we only use the first 3
  (`2026-04-06, 2026-04-08, 2026-04-09`) to stay within budget.
- Same 5 eval days as the gradient sweep (held-out invariant
  preserved).
- BC pretrain DISABLED (same reason as gradient sweep).
- Frozen C11 direction head (same fixed component).
- `--use-race-outcome-predictor` and `--use-direction-predictor` on
  for ALL cells (needed for the gate flags to mean anything).
- `--direction-gate-enabled` ON for ALL cells with
  `direction_gate_threshold = 0.5` (no-op floor) so the gate is the
  same in C0 as in C2-C5.
