# Lessons Learnt — Breeding Pool Scope

## From investigation

- The breeding pool IS correctly scoped to the current run's agents
  (`run_training.py:688-692`). The "61 models" in the user's log was
  likely from the scoreboard's full ranking output, not the breeding
  pool itself.

- The "0 children bred" anomaly happens when `len(survivors) >=
  population_size`. This can occur if:
  - A shakeout script uses a non-standard population size
  - The selection top_pct is too generous relative to population_size
  - The run started with more agents than population_size (e.g.
    session_9_shakeout with 21 agents but a different pop config)

- Garaged models are re-evaluated AFTER all generations complete, not
  during breeding. They never enter the breeding pool currently. But
  the user's intuition is correct — mixing proven garaged models into
  the breeding pool could accelerate convergence by giving new
  generations access to optimised hyperparameters from previous runs.

- The key design question is whether external parents occupy survivor
  slots (competitive, biologically accurate) or are parent-only
  (predictable, safer). Parent-only is probably better to start with.
