# Hard Constraints

- Empty stud list must produce identical behaviour to current code.
- Studs are parent-only — NOT trained, NOT evaluated, NOT counted
  as survivors. n_children is unaffected by stud count.
- All stud IDs validated at run start — reject immediately if any
  ID doesn't exist or lacks hyperparameters.
- Max 5 studs — enforced in both UI and API validation.
- Each stud must be parent of at least one child per generation
  (when breeding slots are available).
- Stud usage must be logged in the genetics log and GeneticEventRecord
  with `selection_reason: "stud"`.
- All tests pass: `python -m pytest tests/ --timeout=120 -q`.
- Frontend builds clean: `ng build`.
