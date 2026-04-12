# Hard Constraints

- `directed_mutation=false` (default) must produce identical behaviour
  to current code. No regression when disabled.
- Directional bias must not eliminate exploration. The Gaussian noise
  must always be present — the bias shifts the mean of the
  distribution, it doesn't replace it with a deterministic step.
- Signals based on fewer than 5 samples per direction must not be
  acted on. Below this threshold, use standard unbiased mutation.
- MutationHistory accumulates within a run only. It does not persist
  across separate training runs — the search space context (data,
  population, architecture mix) changes between runs.
- Inverse mutation experiments (session 2) must be clearly labelled
  in the genetics log and GeneticEventRecord so they can be
  distinguished from normal breeding.
- All tests pass: `python -m pytest tests/ --timeout=120 -q`.
- Frontend builds clean: `ng build`.
