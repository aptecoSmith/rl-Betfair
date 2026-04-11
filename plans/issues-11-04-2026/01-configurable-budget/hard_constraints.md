# Hard Constraints — Configurable Budget

Non-negotiables. Violation of any of these is rejected at review.

## Budget semantics

1. **Budget is per-race, not per-day.** Each race starts with a
   fresh `starting_budget`. Day P&L = sum of per-race P&Ls. This
   rule is unchanged by this plan.

2. **Composite score must remain budget-independent.** The existing
   normalisation (`pnl_norm = mean_pnl / starting_budget`) already
   handles this. Any new scoring code must use the recorded
   `starting_budget` from the evaluation, not the global config.

3. **Global config is the fallback, never the override.** If a
   training plan specifies `starting_budget`, that value wins.
   The global `config.yaml` value is the default for plans that
   don't specify one.

4. **Existing models default to budget=100.** Any migration must
   backfill `starting_budget=100.0` for existing `evaluation_days`
   rows, since that's what they were evaluated with.

## Display

5. **Raw P&L stays visible.** Percentage return is an *addition*,
   not a replacement. The operator should see both.

6. **Percentage return uses the recorded budget, not the current
   global config.** A model trained at £10 always shows % based on
   £10, even if the global config later changes to £50.

## Scope

7. **No changes to the reward function.** The terminal bonus
   already normalises by `starting_budget`. Shaped components use
   the original stake. No reward changes needed.

8. **No changes to the observation space or action space.** The
   agent does not see the budget as a feature (it's implicit in
   the stake sizing). This plan is config + display only.

9. **No changes to genetic selection.** Tournament selection uses
   `composite_score` which is already normalised. No breeding
   logic changes.

## Documentation

10. **Every session updates `progress.md`.**
11. **Every surprising finding goes in `lessons_learnt.md`.**
