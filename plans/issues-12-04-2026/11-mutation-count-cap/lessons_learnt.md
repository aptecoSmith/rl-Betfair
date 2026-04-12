# Lessons Learnt — Mutation Count Cap

## From discussion

- With mutation_rate=0.3 and ~30 genes, each child gets ~9 simultaneous
  mutations on average. This makes attribution impossible — you can't
  tell which change caused improvement or regression.

- Capping to 1-3 mutations per child is the single most impactful
  change for making the GA's search interpretable, especially at low
  generation counts (3-10).

- This is a prerequisite for directed mutation (long-term/issue 10) to
  work — directional signals are meaningless when 9 genes change at
  once.

- The user noted that 3 generations is too few for a 30-dimensional
  search space. Running more generations (10-15) is the simplest
  improvement and requires no code changes — just a different number
  in the wizard. Combined with a mutation cap, even 5 generations
  should show clearer convergence patterns.
