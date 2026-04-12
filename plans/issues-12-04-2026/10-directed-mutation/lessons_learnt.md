# Lessons Learnt — Directed Mutation

## From discussion

- All the data needed for directed mutation already exists in the
  codebase. BreedingRecord captures per-gene mutation deltas,
  GeneticEventRecord persists them to SQLite, and the genetics log
  includes parent scores. The missing piece is a feedback loop that
  uses this data to inform future mutations.

- The key insight is that `rng.gauss(0, sigma)` is always centred at
  zero. Directed mutation shifts the centre to `rng.gauss(bias, sigma)`
  where bias is learned from historical outcomes. This is the minimal
  change needed — the same Gaussian distribution, just with a shifted
  mean.

- Confidence gating is essential. With 2-3 outcomes per direction, the
  signal is noise. Requiring >= 5 samples per direction before acting
  means directed mutation only activates after generation 2-3 at
  earliest, which is appropriate.

- The "inverse mutation" concept (session 2) is a controlled
  experiment: take a good parent, apply one specific mutation in the
  opposite direction, see what happens. This is more targeted than
  normal breeding (which applies crossover + multiple mutations
  simultaneously). Reserving 1-3 slots per generation for these
  experiments balances exploration with controlled hypothesis testing.

- Architecture parameters (lstm_hidden_size, transformer_depth etc.)
  are already evolvable — the user wasn't aware of this. No changes
  needed for architecture variation. The directed mutation system
  should work equally well on architecture parameters as on learning
  rate or reward shaping genes.
