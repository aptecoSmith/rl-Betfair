# 10 — Directed Mutation (Learning from Bad Mutations)

## What

1. After each generation, compare child scores vs parent scores. For
   each mutated gene, record whether the delta direction improved or
   worsened the model.
2. Build a per-gene mutation history that accumulates across
   generations: "increasing learning_rate tends to help" or "increasing
   entropy_coefficient tends to hurt".
3. Bias future mutations toward historically successful directions
   rather than sampling uniformly. Genes with consistent directional
   signal get stronger bias; genes with noisy outcomes stay random.
4. Optionally queue "inverse mutation" experiments: if a specific
   mutation made a good model worse, automatically schedule a child
   with the opposite mutation to test whether it improves.

## Why

- The system already records detailed per-gene mutation deltas in
  `BreedingRecord` and `GeneticEventRecord`, and logs parent scores
  alongside. All the data exists — it's just not used.
- Currently mutation is purely random: same rate, same Gaussian
  distribution, every generation. If mutating learning_rate upward
  consistently produces worse models, the system doesn't notice.
- This is a well-studied technique in evolutionary computation
  (related to CMA-ES, estimation of distribution algorithms, and
  self-adaptive mutation). The key insight is that the search space
  isn't isotropic — some directions are consistently better, and
  the algorithm should learn that.
- The "inverse mutation" idea is a form of hypothesis testing: "this
  mutation hurt — would the opposite help?" Rather than waiting for
  the genetic algorithm to randomly try the opposite, queue it
  explicitly.
