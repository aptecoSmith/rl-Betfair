# Master TODO

## Activated genes (six)

| Gene | Range (locked, Phase 5) | What it does |
|---|---|---|
| `stop_loss_pnl_threshold` | [0.0, 0.30] | Auto-close a pair when its MTM PnL falls below this fraction of starting budget |
| `open_cost` | [0.0, 2.0] | Per-pair-open shaped charge, refunded on natural-mature / agent-close. Pushes selectivity at open |
| `matured_arb_bonus_weight` | [0.0, 5.0] | Positive shaped reward per matured/closed pair (skill-of-closing) |
| `naked_loss_scale` | [0.0, 1.0] | Anneals naked-loss contribution to raw PnL (loss side undercounted, win side untouched). Bootstraps past the naked valley |
| `mature_prob_loss_weight` | [1.0, 5.0] | BCE auxiliary head trained on strict-mature labels; output feeds actor_head |
| `fill_prob_loss_weight` | [0.0, 0.30] | BCE auxiliary head trained on fill labels; output feeds actor_head |

## Steps

1. Scaffold plan folder. ✅
2. Launch cohort: 12 agents × 8 generations × 5 training days,
   eval=2026-05-06, scalping mode, predictor lean obs, all six
   genes enabled.
3. Wait for completion (~12h GPU). Monitor for crashes / GA
   collapse / KL blowups via tail of log.
4. After cohort completes, re-eval top-5 via
   `tools/reevaluate_cohort.py` against held-out
   `2026-05-04 / 05 / 06` with the same flags as predecessor.
5. Compare to predecessor cohort:
   - Composite scoreboard (in-sample)
   - 3-day held-out distribution (profitability rate)
   - Per-gene correlation with held-out PnL (which gene
     mattered?)
6. Write `findings.md`. Verdict on the success bar (≥3/5
   profitable on held-out).
7. Operator decision after findings: productionise (b/c),
   widen sweep, or stop.

## Operator decisions

- **After step 4 (reeval):** Default to "continue to findings"
  unless the cohort produced ≥5 profitable agents OR ≤1
  (clear pass / clear fail).
- **After step 6 (findings):** Stop the autonomous loop;
  productionization is an operator-scope decision.

## Wall-clock budget

- Predecessor (12 × 5 = 60 evals): 7h37m.
- This run (12 × 8 = 96 evals): estimate ~12h.
- Reeval: 20 min.
