# Long-Term Ideas

Features that are worth doing eventually but aren't urgent. These build
on top of nearer-term work and should wait until prerequisites land.

---

## Directed Mutation — Learning from Bad Mutations

**Prerequisite:** Issue 11 (mutation count cap) should land first.
Directional signals are meaningless when 9 genes change simultaneously.
Also benefits from running 10+ generations to accumulate enough data.

### What

After each generation, compare child scores vs parent scores. For each
mutated gene, record whether the delta direction improved or worsened
the model. Build a per-gene directional signal that biases future
mutations toward historically successful directions.

### How

1. **Mutation outcome tracking** — for each child, compute
   `score_delta = child_score - mean(parent_a_score, parent_b_score)`.
   For each mutated gene, record `(gene, mutation_delta, score_delta)`.

2. **Directional signal** — partition outcomes by sign of mutation_delta
   (increased vs decreased). Compute mean score_delta per direction.
   Signal = difference in means. Confidence = min(n_samples_per_direction).

3. **Biased mutation** — in `mutate()`, shift Gaussian noise from
   `gauss(0, sigma)` to `gauss(bias, sigma)` where bias is proportional
   to the directional signal. Only act when signal has >= 5 samples per
   direction.

4. **Inverse mutation experiments** — when a specific mutation makes a
   good model significantly worse, queue a "try the opposite" experiment
   in the next generation. Reserve 1-3 breeding slots for these
   controlled single-gene experiments.

### Why wait

- Needs mutation count cap (issue 11) to make per-gene attribution
  reliable.
- Needs multiple generations (10+) to accumulate enough mutation
  outcomes for confident directional signals.
- The immediate wins (more generations + mutation cap) give 80% of the
  benefit with 20% of the complexity.

### Related

- Issue 09 (adaptive breeding) handles generation-level "things are
  bad, crank up mutation". Directed mutation handles gene-level "this
  direction tends to help".
- CMA-ES and estimation of distribution algorithms are the academic
  cousins of this approach.

### Full plan

A detailed plan was drafted in `plans/issues-12-04-2026/10-directed-mutation/`
(2 sessions). That folder can be used as the starting point when this
work is picked up.
