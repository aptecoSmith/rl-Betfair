"""v2 GA cohort — Phase 3, Session 03.

Parallel-tree replacement for v1's ``training/run_training.py`` +
``training/worker.py`` cohort scaffolding. Composes Sessions 01 + 02
(GPU pathway, multi-day train) into a worker-pool runner that trains
N agents, evaluates on a held-out day, breeds the next generation, and
writes the same registry shape v1 writes.

Hard constraints honoured here (rewrite README §3, phase-3 purpose
§"Hard constraints", session 03 prompt §"Hard constraints"):

- No env edits.
- No re-import of v1 trainer / worker / runner.
- Locked Phase 3 gene schema (no additions, no removals).
- Registry write shape matches v1 field-for-field; ``arch_name``
  discriminates v2 weights from v1 weights.
"""
