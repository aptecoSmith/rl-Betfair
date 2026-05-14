# Hard constraints

## 1. Do NOT launch until lay-quality-gate findings.md commits

The lay-quality-gate cohort
(`_predictor_SCALPING_layq_1778712871`) was running mid-flight when
this plan was scaffolded. Its held-out reeval verdict is what
calibrates this plan's success bar. If it lands no-improvement or
worse, the priority shifts back to the gate, not this plan.

Check `plans/scalping-lay-quality-gate/findings.md` exists and
commits a verdict before promoting this plan to in-flight.

## 2. Default-off byte-identical for both changes

- Selection score: when no override flag, default to
  `eval_total_reward` (preserves byte-identity with pre-plan cohort
  runs). The new score is opt-in via flag.
- Obs feature: when no open pair on the runner, the new column
  reports exactly 0.0. Same byte-identical-when-no-position
  guarantee as Phase 2b's leverage features.

## 3. Architecture-hash break is expected and correct

The new obs column widens `lstm_input_proj.0.weight` by
`max_runners` columns. Pre-plan weights cannot cross-load via
`strict=True`. Regression test enforces.

## 4. Same gate config as lay-quality-gate

This plan is orthogonal to the gate. Do not tune gate parameters
in this plan even if the held-out verdict suggests they want
tuning — that's a different plan (donut filter, threshold
adjustment, etc.).

## 5. Reward shape unchanged

`raw + shaped` accumulators are identical to lay-quality-gate.
The selection-score change is on the GA's *composite_score* —
which is fed into selection / mutation operators, NOT into PPO
reward feedback. PPO sees the same per-episode reward as before.

## 6. Held-out window locked

2026-04-28/29/30. Same as every predecessor.

## 7. Both fc=0 and fc=120 reevals reported

Same dual-reeval discipline as lay-quality-gate. The verdict reads
the fc=120 number for deployment realism but the fc=0 number for
A/B comparison with predecessors.

## 8. No new genes

The selection-score formula uses fixed weights. No promotion of
the 0.25 weight to a gene. Genes carry over from lay-quality-gate
unchanged.

## 9. Score formula is locked

```
composite_score = locked_pnl + 0.25 * naked_pnl
```

The 0.25 weight is a single calibration based on the cross-agent
naked/locked variance ratio observed in lay-quality-gate
(σ_naked / σ_locked ≈ 4-5x). It is not subject to mid-flight
tuning. If the held-out result is poor, the next plan can probe
the weight; this plan locks it.

## 10. Loop ends only on these conditions

1. Verdict written.
2. Stop condition: pre-flight smoke fails, OR a constraint above is
   about to be violated, OR three consecutive iterations make no
   progress.
3. Crash recovery needed: cohort crashes mid-run.
