# Hard Constraints

- Default behaviour must be identical to current code: persist policy,
  no adaptive mutation, 0.3 base rate. All new features are opt-in.
- Adaptive mutation state (consecutive bad generation count, effective
  rate) is per-run only — not persisted to disk or across runs.
- Mutation rate must stay within [0.0, 1.0]. The adaptive cap prevents
  runaway mutation.
- inject_top policy must not re-train injected models — they are
  parent-only contributors to crossover.
- Wizard overrides take precedence over config.yaml values.
- All tests pass: `python -m pytest tests/ --timeout=120 -q`.
- Frontend builds clean: `ng build`.
