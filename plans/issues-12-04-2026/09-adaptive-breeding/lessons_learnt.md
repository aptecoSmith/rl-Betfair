# Lessons Learnt — Adaptive Breeding

## From discussion

- The mutation rate (0.3) is fixed in config.yaml and passed straight
  through to breed(). No way to tune per run, no adaptive behaviour.
  The wizard shows it as a read-only value.

- There's no concept of "this generation was bad" in the system. The
  discard policy checks individual models against thresholds, but
  there's no aggregate quality check at the generation level.

- The user's intuition maps to a well-known pattern in evolutionary
  algorithms: increase mutation pressure when the population is
  stagnating. In biology, environmental stress increases mutation
  rates — the same principle applies to hyperparameter search.

- Three natural responses to a bad generation:
  1. Do nothing (current) — hope random crossover fixes it
  2. Boost mutation — shake up the search space more aggressively
  3. Inject proven parents — bring in garaged top performers to
     contribute good hyperparameters via crossover

- Options 2 and 3 can combine: inject top performers AND boost
  mutation on the children bred from them. This is the most
  aggressive recovery strategy.

- Issue 08 (breeding pool scope) and this issue overlap on the
  "inject top performers" mechanism. If 08 lands first, this issue
  can reuse that plumbing. If not, implement it inline.
