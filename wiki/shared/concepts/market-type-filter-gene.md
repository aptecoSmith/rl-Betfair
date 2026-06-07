---
id: 01KTJ06D5S53TCVWVFC0VPFW4S
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [work, research]
sources: [src-039ca3]
aliases: [market-type-filter, market_type_filter]
---

# Market-type filter gene

Per-agent gene with values `{WIN, EW, BOTH}` that filters which Betfair markets a model trains on — implemented by skipping races whose market type doesn't match, NOT by altering the obs/action schema.

## What it is

Added 2026-04-11 as a `str_choice` entry under `config.yaml:hyperparameters.search_ranges`. The [[populationmanager]] auto-handles sampling, crossover, and mutation for `str_choice` types — no Python code needed for the genetic operators, just the YAML entry.

The filter does NOT change observation or action schema. A [[win]]-only model has the same obs vector shape as a BOTH model; the `market_type_win` / `market_type_each_way` features just happen to always be `[1, 0]` for a WIN-only agent. **Weights are therefore cross-compatible** — a child can inherit weights from a BOTH parent and be trained as WIN-only.

## Why it matters

- Bet counts and PnL magnitude scale with how many races/day match the filter (some days are 100% WIN, others heavily [[ew]]), but the budget normalisation handles that.
- Adding the gene was trivially cheap — the cost was the design decision to keep schema stable, which preserves weight portability across the filter axis. This is the canonical pattern for adding a population-search dimension without breaking warm-starts.

## Links
- [[populationmanager]] — auto-handles `str_choice` gene plumbing.
- [[shared/index|hub]]
