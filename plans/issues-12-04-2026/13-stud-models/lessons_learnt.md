# Lessons Learnt — Stud Models

## From discussion

- Three breeding-related issues overlap but serve different purposes:
  - Issue 08 (breeding pool scope): adds models to the selection pool
    — they must still compete to become parents
  - Issue 09 (adaptive breeding): automatically injects top performers
    when a generation is bad — reactive, not pre-planned
  - Issue 13 (studs): user hand-picks models that are ALWAYS parents
    — proactive, bypasses selection entirely

- The analogy to horse breeding stud books is apt: proven sires breed
  regardless of their most recent race. In RL terms: a model with a
  uniquely good hyperparameter combination (e.g. learning rate,
  architecture config) contributes those genes even if its overall
  composite score isn't competitive.

- Parent-only semantics are important: studs don't take survivor slots,
  don't get trained, don't get evaluated. They're gene donors. This
  keeps the population dynamics predictable — you still get the full
  population_size worth of new models, just with some forced parentage.
