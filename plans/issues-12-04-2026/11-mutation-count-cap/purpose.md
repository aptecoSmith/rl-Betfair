# 11 — Mutation Count Cap

## What

Add a `max_mutations_per_child` config option that caps how many genes
can mutate simultaneously when breeding a child. Instead of each gene
independently flipping a 30% coin (~9 mutations on average from 30
genes), the system mutates at most N genes per child.

## Why

- With ~9 simultaneous mutations per child, it's impossible to
  attribute improvement or regression to any specific change. This
  is the classic confounding variable problem.
- Capping to 1-3 mutations per child makes attribution much cleaner:
  if the child is worse, it's probably one of the 1-3 things that
  changed.
- This is a prerequisite for directed mutation (issue 10 / long-term)
  to be effective — directional signals are meaningless when 9 genes
  change simultaneously.
- GAs with fewer simultaneous mutations converge more reliably in
  low-generation runs (3-10 gens). The trade-off is slower
  exploration of the full space, but with 50 agents per generation,
  population diversity handles breadth.
