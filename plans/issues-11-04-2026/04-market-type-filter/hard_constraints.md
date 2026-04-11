# Hard Constraints — Market Type Filter

Non-negotiables. Violation of any of these is rejected at review.

## Schema stability

1. **No observation schema change.** The `market_type_win` and
   `market_type_each_way` features remain in the obs vector at the
   same positions. A WIN-only model's obs will always have
   `[1.0, 0.0]` for these — that's fine, they're just constant.
   OBS_SCHEMA_VERSION stays at current value.

2. **No action schema change.** The filter controls which races are
   presented, not how actions work within a race.
   ACTION_SCHEMA_VERSION stays at current value.

3. **Weight compatibility across filter values.** A model trained
   with filter=WIN can load weights from a filter=BOTH parent (same
   obs/action dims). The genetic crossover depends on this.

## Backward compatibility

4. **Existing models default to BOTH.** Any model without
   `market_type_filter` in its hyperparameters is treated as BOTH.
   No migration needed for existing registry rows.

5. **Global config is not affected.** The filter is per-model (a
   gene), not a global training setting. Different models in the
   same population can have different filters.

## Correctness

6. **Eval filter must match training filter.** A model trained on
   WIN-only must be evaluated on WIN-only races. Evaluating a
   WIN-only model on all races would produce misleading scores
   (it would skip EW races where it has no training signal).

7. **Zero-race days must not crash.** If the filter eliminates all
   races in a day, the episode completes with zero reward and zero
   bets. The evaluator records this as a valid 0-pnl day.

8. **Filter applies before the episode loop, not mid-episode.**
   Races are filtered in `reset()`, not conditionally skipped
   during `step()`. The agent never sees a filtered-out race.

## Scope

9. **No new observation features.** The existing market type
   features are sufficient.
10. **No reward changes.** The filter is a data selection mechanism,
    not a reward shaping term.
11. **No population-level constraints.** The genetic algorithm can
    freely evolve the filter. There's no requirement for "at least
    N models must be BOTH" or similar.

## Documentation

12. **Every session updates `progress.md`.**
13. **Every surprising finding goes in `lessons_learnt.md`.**
