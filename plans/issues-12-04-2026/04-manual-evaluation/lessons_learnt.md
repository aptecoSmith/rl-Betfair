# Lessons Learnt — Manual Evaluation

## From discussion

- The Evaluator class is well-factored — it takes explicit inputs and
  writes to the registry. No modifications needed to support standalone
  use. The coupling is at the *call site* (training loop), not in the
  evaluator itself.
- The main challenge for standalone evaluation is reconstructing a
  policy from registry metadata + saved weights. During training the
  policy already exists in memory. For manual eval, we need a factory
  that goes: architecture name + hyperparameters → policy instance →
  load state_dict. This factory logic exists somewhere in the training
  code but may need to be extracted.
- Garaged model re-evaluation during training is the closest precedent.
  It already loads models from the registry and evaluates them. The
  difference is it runs inside the training loop and uses the same
  test days as the current run.
