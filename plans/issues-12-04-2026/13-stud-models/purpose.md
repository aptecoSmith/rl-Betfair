# 13 — Stud Models (Guaranteed Breeding Parents)

## What

Add a training option where the user hand-picks specific models that
are guaranteed to be used as breeding parents from generation 1 onward.
These "studs" contribute their hyperparameters to children via crossover
every generation, regardless of whether they'd survive normal selection.

## Why

- Issue 08 (breeding pool scope) lets you include garaged models in
  the breeding pool — but they still have to survive selection to
  become parents. A weak-scoring garaged model with one brilliant
  hyperparameter combination won't make the cut.
- Studs bypass selection entirely. The user says "I know this model
  has something good — force it into every generation's breeding".
  This is like a stud book in horse breeding: proven sires breed
  regardless of recent form.
- Use cases:
  - "Model X has the best learning rate I've ever seen — breed it
    into everything"
  - "I want to cross-pollinate hyperparameters from a model trained
    on different data"
  - "This model scores poorly overall but has a uniquely good
    architecture config — preserve those genes"

## Relationship to other issues

- **Issue 08 (breeding pool scope)**: studs are a stronger version of
  `include_garaged`. Issue 08 adds models to the pool; studs force
  them to be parents. Can coexist — studs are always parents, pool
  models compete for parent slots.
- **Issue 09 (adaptive breeding)**: `inject_top` policy is automatic;
  studs are manual. Both can coexist.
