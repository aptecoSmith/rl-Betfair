# Hard Constraints — Managed Hyperparameter Search

Non-negotiables. Violation of any of these is rejected at review.

## Backward compatibility

1. **"Random" strategy is the default.** Existing workflows
   (no strategy specified) behave exactly as today. The managed
   search is opt-in.

2. **Existing models and evaluations are untouched.** No migration
   of existing model records. Historical models are read-only inputs
   to the coverage analysis.

3. **Training plans without a strategy field default to "random".**
   Old plans in the registry continue to work.

## Genetic algorithm integrity

4. **The seed point only controls initialisation.** Once the
   population is created, the genetic algorithm (crossover, mutation,
   selection) runs unchanged. The seed biases the starting region,
   not the evolutionary dynamics.

5. **Perturbation around the seed must stay within valid bounds.**
   No gene value outside its defined min/max or outside its choice
   set.

6. **Composite scoring is unaffected.** Models from different
   exploration strategies are ranked on the same scoreboard with
   the same normalised composite score.

## Exploration integrity

7. **Every seeded run is logged.** The exploration_runs table must
   record the seed point, strategy, and coverage snapshot for
   every non-random training run. This is the audit trail for
   the coverage map.

8. **Sobol points are deterministic given the same dimensionality.**
   Two systems with the same gene specs and the same skip value
   must produce identical seed points. This enables reproducibility.

9. **Coverage analysis reads from the model registry, not from the
   exploration log.** The log records *intent* (where we seeded);
   the registry records *reality* (where models actually ended up
   after evolution). The coverage map uses reality.

## Scope

10. **Phase 2 (focused exploitation) is out of scope for this plan.**
    This plan builds the exploration infrastructure. Focused
    exploitation is a follow-up plan once we have coverage data.
11. **No changes to observation space, action space, or reward.**
12. **No changes to the scorer, discard policy, or genetic operators.**

## Documentation

13. **Every session updates `progress.md`.**
14. **Every surprising finding goes in `lessons_learnt.md`.**
