# Master TODO — Directed Mutation

## Session 1: Mutation outcome analysis + directional bias

### Mutation outcome tracking

- [ ] After each generation's evaluation, compute per-child score
      deltas: `child_score - mean(parent_a_score, parent_b_score)`
- [ ] For each mutated gene in that child (where `delta != None` in
      BreedingRecord), record a `MutationOutcome`:
      ```python
      @dataclass
      class MutationOutcome:
          gene_name: str
          mutation_delta: float     # +0.02 (increased) or -0.02 (decreased)
          score_delta: float        # +0.1 (child better) or -0.3 (child worse)
          generation: int
          child_model_id: str
          parent_mean_score: float
      ```
- [ ] Store outcomes in a `MutationHistory` object that accumulates
      across generations within a run
- [ ] Persist to `logs/training/mutation_history.jsonl` for post-run
      analysis

### Directional signal computation

- [ ] For each gene, compute a directional signal from accumulated
      outcomes:
      - Partition outcomes by sign of mutation_delta (increased vs decreased)
      - For each direction: mean score_delta
      - `direction_signal = mean_score_delta_when_increased - mean_score_delta_when_decreased`
      - Positive signal → increasing this gene tends to help
      - Negative signal → decreasing tends to help
      - Near zero → no consistent directional preference
- [ ] Compute a confidence measure: how many outcomes back the signal?
      With <5 outcomes per direction, the signal is noise — don't act
      on it
- [ ] Update signals after each generation as new outcomes arrive

### Directed mutation implementation

- [ ] Add `directed_mutation: bool` option to config.yaml (default: false)
- [ ] When enabled, modify `population_manager.mutate()`:
      - For genes with a confident directional signal (>= N outcomes,
        signal magnitude above threshold):
        bias the Gaussian mutation toward the successful direction
      - For genes without confident signal: standard random mutation
      - Bias strength scales with signal confidence (more data → stronger)
- [ ] Implementation: instead of `delta = rng.gauss(0, sigma)`, use
      `delta = rng.gauss(bias, sigma)` where bias = signal * bias_strength
- [ ] `bias_strength` gene or config parameter (0.0-1.0, default 0.3)
      controls how aggressively to follow the signal
- [ ] Always retain some randomness — pure exploitation kills exploration

### Add to wizard / training plan

- [ ] Add "Directed mutation" toggle to wizard genetics step
- [ ] Help text: "When enabled, the system learns which direction of
      change tends to improve models for each hyperparameter. Future
      mutations are biased toward historically successful directions.
      Requires at least 2 generations to accumulate enough data."
- [ ] Add `directed_mutation` and `bias_strength` to training plan model

### Logging / visibility

- [ ] Log per-gene directional signals after each generation:
      "learning_rate: ↑ tends to help (+0.05 avg, 12 samples)"
- [ ] Activity log entry in training monitor when directed bias activates
- [ ] Log when a gene's signal flips direction (was positive, now negative)
      — this indicates the optimum has been passed

### Tests

- [ ] Test: MutationOutcome correctly computed from BreedingRecord + scores
- [ ] Test: directional signal positive when increases consistently help
- [ ] Test: directional signal near zero with random outcomes
- [ ] Test: mutate() applies bias when directed_mutation=true and signal
      is confident
- [ ] Test: mutate() falls back to random when signal has <N outcomes
- [ ] Test: directed_mutation=false produces identical behaviour to current
- [ ] Test: mutation history persisted to JSONL

### Verify

- [ ] `python -m pytest tests/ --timeout=120 -q` — all green
- [ ] `cd frontend && ng build` — clean

---

## Session 2: Inverse mutation experiments

### Identify candidates for inverse mutation

- [ ] After each generation, scan for "high-value regressions":
      child with significantly worse score than both parents, where
      a single gene was mutated with a large delta
- [ ] Threshold: `score_delta < -0.1` AND `mutation_delta` is the
      largest contributor (single-gene attribution when possible)
- [ ] These are candidates for "try the opposite"

### Queue inverse mutation experiments

- [ ] Create an `InverseMutationQueue` that holds pending experiments:
      ```python
      @dataclass
      class InverseMutationExperiment:
          base_model_id: str        # The good parent
          gene_name: str            # Which gene to reverse-mutate
          original_delta: float     # What the bad mutation did
          inverse_delta: float      # Opposite direction, same magnitude
          priority: float           # How confident we are this is worth trying
      ```
- [ ] When breeding the next generation, reserve N slots (configurable,
      default: 2) for inverse mutation experiments instead of random
      crossover
- [ ] The experiment child: starts from the good parent's HP, applies
      only the inverse mutation (no crossover, no other mutations)
- [ ] This is a controlled experiment: one variable changed

### Track experiment outcomes

- [ ] After evaluation, compare experiment child vs the good parent
      and vs the bad child:
      - If experiment child > good parent → the inverse direction
        genuinely helps. Strengthen the directional signal.
      - If experiment child ≈ good parent → the mutation doesn't
        matter much. Weaken the signal.
      - If experiment child < good parent → both directions hurt.
        The gene may be at its optimum for this context.
- [ ] Log experiment results clearly in genetics log and activity log
- [ ] Feed outcomes back into MutationHistory to improve directional
      signals

### Experiment slots in breeding

- [ ] Add `inverse_mutation_slots: int` to config (default: 0)
- [ ] In `breed()`: reserve first N slots for queued experiments,
      fill remaining with normal crossover + mutation
- [ ] If queue is empty, all slots are normal breeding
- [ ] Experiments get `selection_reason: "inverse_mutation_experiment"`
      in GeneticEventRecord

### Wizard / training plan UI

- [ ] Add "Inverse mutation experiments" toggle
- [ ] Slots input (how many per generation to reserve)
- [ ] Help text: "When a mutation makes a good model worse, the system
      automatically tests the opposite change in the next generation.
      This is a controlled experiment — only one gene changes. Reserve
      1-3 slots per generation for these experiments."

### Tests

- [ ] Test: high-value regressions correctly identified
- [ ] Test: inverse mutation queued with correct delta
- [ ] Test: experiment child has only the one inverse mutation applied
- [ ] Test: experiment outcome feeds back into MutationHistory
- [ ] Test: breeding reserves correct number of experiment slots
- [ ] Test: empty queue → all slots are normal breeding

### Verify

- [ ] `python -m pytest tests/ --timeout=120 -q` — all green
- [ ] `cd frontend && ng build` — clean
