# Hard Constraints — Scalping Active Management

Non-negotiables. Violation of any of these is rejected at review.

## Reward integrity

1. **`raw + shaped ≈ total_reward` invariant still holds.** New
   auxiliary losses (fill probability, risk) live outside the
   PPO reward. They train auxiliary heads via separate loss
   terms added to the total objective. They do **not** modify
   the raw or shaped reward accumulators.

2. **Auxiliary losses use their own coefficient knobs.** Each
   new head (`fill_prob`, `risk`) gets a `*_loss_weight` config
   key defaulting to `0.0`. Session 02 / 03 land plumbing off,
   then enable with small values and watch for interference.

3. **Auxiliary gradient must not swamp the PPO gradient.** If
   enabling the fill-probability head at weight 1.0 degrades
   the main policy's mean reward by > 20 % over a training run,
   either the weight is too high or the head architecture
   shares too much. Treat degradation as a failure, not a
   feature.

## Action-space & matcher discipline

4. **Re-quote never walks the ladder.** The re-quote mechanic
   cancels the existing passive and re-places via the same
   `PassiveOrderBook.place` path that the initial placement
   uses. Same junk filter, same one-price-only matching, same
   clamp to best-post-filter-size. No new matcher code paths.
   See CLAUDE.md "Order matching: single-price, no walking".

5. **Re-quote is a no-op without an open passive to manage.**
   If there's no outstanding passive on a runner, the
   `requote_signal` action is silently ignored. Never opens a
   new naked position.

6. **Re-quote budget accounting.** Cancelling a passive returns
   its reserved liability/stake to `bm.available_budget` before
   the new passive is reserved. Net budget change for a re-quote
   should be small (difference in liability between old and new
   price). Any bug that double-reserves or leaks budget is
   rejected.

7. **Action-space additions are additive.** The new
   `requote_signal` dimension adds one per-runner slot at the
   end of the action vector. Pre-existing action indices do not
   move. Pre-Session-04 checkpoints load with the new head
   freshly initialised.

## Network architecture

8. **Auxiliary heads share the backbone, not the policy head.**
   Both fill-probability and risk heads take the shared LSTM /
   transformer output as input. They do NOT receive the sampled
   action as input (that's not available at train-time in PPO;
   it would require reparameterisation). The prediction is
   conditioned on *state*, not on *action + state*.

9. **Backward compatibility for checkpoints.** Models trained
   before Session 02 / 03 load cleanly with the new heads
   freshly initialised (fresh weights, identity-ish output for
   the first forward pass). Existing tests that load Gen 0
   checkpoints must continue to pass.

10. **Per-bet predictions are captured, not recomputed.** The
    fill-probability and risk predictions produced at placement
    time are recorded on the `Bet` object so the evaluator /
    bet log has the *decision-time* prediction, not one
    recomputed from the post-hoc network. Recomputing would
    defeat the calibration-plot use case.

## Persistence

11. **Parquet schema additions are optional columns.** New
    per-bet prediction columns (`fill_prob_at_placement`,
    `predicted_locked_pnl`, `predicted_locked_stddev`) default
    to NULL on read if the column is absent. Pre-Session-04
    parquet files load without breaking.

12. **No breaking changes to the `/bets` or `/scoreboard` API
    responses.** New fields are optional (`... | None = None`
    on the Pydantic model). Existing frontend code that
    doesn't know about them keeps working.

## Evaluation honesty

13. **Calibration metrics are reported on held-out test days
    only.** Training-day fills train the head; only eval-day
    fills are used to judge how calibrated the agent is. Mixing
    them is rejected — a head can trivially memorise its
    training data and look perfectly calibrated on it.

14. **The "is this a good scalper?" headline metric is
    unchanged.** The scoreboard still ranks by
    L/N ratio > composite. Adding a calibration column is
    diagnostic; it does not feed the composite score. Rank on
    realised outcome, not on self-reported confidence.

## Scope discipline

15. **`ExchangeMatcher` is untouched.** No ladder walking, no
    dropping the LTP requirement, no gating price caps on the
    unfiltered top-of-book. Three independent regressions last
    time — any PR touching the matcher is rejected. See
    CLAUDE.md.

16. **No observation features beyond the two specified in
    Session 01** (`seconds_since_passive_placed`,
    `passive_price_vs_current_ltp_ticks`). Tempting to throw
    more features at the problem; resist. Adds of-scope
    features go in a follow-on plan.

17. **No changes to the existing `arb_spread` action dim or
    `arb_spread_scale` gene.** This plan adds NEW action
    dimensions and NEW genes; it does not re-tune the existing
    ones.

## Documentation

18. **Every session updates `progress.md`.**
19. **Every surprising finding goes in `lessons_learnt.md`.**
20. **Reward-scale breaks are called out in commit messages
    AND in the session progress entry.** Enabling auxiliary
    loss weights doesn't break reward scale (they're separate
    losses) but any change that does must be flagged loudly so
    operators don't chase ghosts comparing pre-change and
    post-change model P&L.
