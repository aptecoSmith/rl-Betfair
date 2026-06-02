# [SUPERSEDED 2026-05-28] BREAKTHROUGH: force_close=0 unlocks positive day_pnl

> ## ⚠️ THIS FINDING WAS WRONG. KEPT AS A WARNING.
>
> The +£287 day_pnl was **eval-window overfitting**. Held-out validation
> on 14 unseen days (2026-05-07..05-20) collapsed it to **-£155 to -£195
> per agent, every agent negative.** See
> `plans/EXPLORATIONS.md` 2026-05-28 entry "CORRECTION — the naked-EV
> 'edge' was eval-window overfitting" and `EXPERIMENTS.md` "Held-out
> validation of fc=0 winner (HV cells)".
>
> **Corrected understanding (do NOT relitigate):**
> - Naked P&L is zero-EV directional variance. It looked +EV on 5
>   April eval days by luck; on 14 May days it lost.
> - `force_close=120` is a SAFETY RAIL bounding naked variance, not
>   a cost to remove. Keep it ON.
> - The only structural edge is LOCKED P&L from matured scalps
>   (~+£3-6/matured pair). mat% is the bottleneck.
>
> The body of this doc is preserved verbatim for the audit trail
> of how the mistake happened — repeated eval on the same 5 days
> across 60+ probe cells with no held-out check.

**Date:** 2026-05-27 ~01:10 BST (Round 6.5 completion)

## The hypothesis

Across 50+ probe cells the policy's *naked* P&L was consistently
positive (+£44 to +£95/day, repeatable across 7 different recipes).
This suggested the back-leg selection was structurally EV-positive,
but the env's force-close-at-T-120s was crossing the spread on
pairs whose passive lay never filled — eating most of the naked EV.

Hypothesis: **disable env force-close → nakeds settle out → day_pnl
turns positive.**

## Result

**Round 6.5 confirms the hypothesis decisively.**

Every cell with `force_close_before_off_seconds=0`:

| cell                          | day_pnl (mean) | range across 4 agents | passes |
|-------------------------------|---------------:|-----------------------|-------:|
| K1_e7_fc_off                  |     **+£217.2** | +£125 to +£325         | 4/5    |
| K2_e7_lay_max_fc_off          |     **+£201.9** | +£85  to +£341         | 4/5    |
| K3_e7_bc1000_fc_off           |     **+£215.7** | +£71  to +£370         | 4/5    |
| K5_e7_lay_max_bc1000_fc_off   |     **+£232.7** | +£171 to +£308         | 4/5    |
| K6_e7_seed43_fc_off           |     **+£206.9** | +£131 to +£257         | 4/5    |

**20 out of 20 agents POSITIVE day_pnl.** Range +£71 to +£370.
Mean ~+£200. Every cell, every agent.

The non-fc=0 cell:
| K4_e7_fc30 (force-close at T-30s) | -£81.7 | (negative) | 4/5 |

So partial disable doesn't capture it — must be fully off.

## Mechanism

With `force_close_before_off_seconds=0`:
- Pairs that previously force-closed at T-120s (60-70% of opens)
  now settle naked at race start.
- Force-closed P&L: WAS -£123/day → NOW £0/day (force-close disabled).
- Naked P&L: WAS +£77/day (small) → NOW +£250-£280/day (large
  because more pairs settle naked).
- Locked P&L: DROPS to +£4-£7 (fewer pairs mature because fewer get
  a chance — but the naked side dominates anyway).
- Closed P&L: -£50/day (agent still uses close_signal as stop-loss).

Net effect: **the +£250 naked term swamps the -£50 closed term, giving
+£200 day_pnl.**

## The winning recipe

```bash
python -m training_v2.cohort.runner \
    --n-agents 4 \
    --generations 1 \
    --device cuda \
    --seed 42 \
    --strategy-mode arb \
    --training-days-explicit 2026-04-06 2026-04-08 2026-04-09 \
    --cohort-eval-days 2026-04-10 2026-04-17 2026-04-21 2026-05-03 2026-05-06 \
    --rotating-eval-sample 0 \
    --direction-head-manifest models/direction_head/sweep_c11 \
    --predictor-lean-obs \
    --use-race-outcome-predictor \
    --use-direction-predictor \
    --predictor-bundle-manifests ... \
    --reward-overrides force_close_before_off_seconds=0 \
    --reward-overrides close_feasibility_max_spread_pct=0.05 \
    --reward-overrides matured_arb_expected_random=0.0 \
    --bc-pretrain-steps 500 \
    --predictor-p-win-back-threshold 0.20
```

Optional additions (don't appear to change the headline materially):
- `--lay-price-max 20` (small lift on day_pnl, drops 1 agent's variance)
- `--bc-pretrain-steps 1000` (slightly higher mat%, slightly higher day_pnl)

## Caveats

1. **L/σ_naked ratio is near zero** (0.04-0.10). Locked P&L is +£4-£7
   while naked σ across agents is large. The day_pnl is almost entirely
   naked. So the recipe's profitability depends on the naked side
   continuing to be EV-positive on the held-out eval days.

2. **Eval days fixed.** All cells share the same 5 eval days
   (2026-04-10, -17, -21, 2026-05-03, -06). If those days drift in
   a direction that favours the policy's back-leg selection, the
   result is luck-of-the-eval-window. Need to test on different
   eval days to confirm robustness.

3. **Deployment safety.** A live trader with `force_close=0` is
   carrying naked positions into in-play (the race actually running).
   That's where real money is at risk. The £200/day result assumes
   settlement is fair — but in live trading there's slippage and
   adverse selection. The training environment may overstate naked
   EV.

## What's queued

Rounds 9-11 (~36 cells, ~17 hours) test:
- Round 9: lay_price_max sweep at fc=0; pwin_back sweep at fc=0;
  4 more seed replicates of the winning recipe.
- Round 10: 8-agent scale-up; multi-gen training; BC dose at fc=0;
  more seeds.
- Round 11: reward-shape variations (naked_loss_scale, close_bonus,
  matured_arb_bonus) on the fc=0 base.

These should produce a robust deploy candidate cluster, not just
one lucky cell.

## Open questions for the next sessions

1. **Does the +£200 hold on different eval days?** Need a held-out
   eval-window probe (run same recipe but eval on April 11/12/13).
2. **Is the per-agent variance bounded enough for live deployment?**
   Min agent was +£71, max +£370. The min is what matters for
   worst-case PnL. Across more seeds the floor might drop.
3. **Does the recipe survive 8-agent scale-up?** Round 10 tests this.
4. **Does multi-gen PPO improve or destabilize?** Round 10 also.
5. **Can `close_signal_bonus > 0` reduce the -£50 closed drag?**
   Round 11 tests this.
