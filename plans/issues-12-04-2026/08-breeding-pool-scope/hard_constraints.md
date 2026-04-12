# Hard Constraints

- `run_only` mode (default) must produce identical behaviour to
  current code — no regression.
- Garaged/registry models used as parents must NOT be re-trained.
  They contribute hyperparameters to crossover only.
- External parents must not silently reduce the number of children
  bred. The user should see clearly how many run-agents survived
  vs how many external parents contributed.
- Add a warning log when `n_children <= 0` — this is always
  surprising and should be visible.
- All tests pass: `python -m pytest tests/ --timeout=120 -q`.
- Frontend builds clean: `ng build`.
