# Lessons Learnt — Market Type Filter

Append-only. Date each entry.

---

## 2026-04-11 — Initial analysis

Adding a new gene to the population is trivially simple — just a
YAML entry in `config.yaml:hyperparameters.search_ranges`. The
`PopulationManager` automatically handles sampling, crossover, and
mutation for `str_choice` types. No Python code needed for the
genetic operators.

The key design decision is that the filter does NOT change the
observation or action schema. A WIN-only model has the same obs
vector shape as a BOTH model — the `market_type_win` /
`market_type_each_way` features just happen to always be `[1, 0]`.
This means weights are cross-compatible: a child can inherit weights
from a BOTH parent and be trained as WIN-only. The schema version
stays stable.

Market type is currently observed as a feature but never used as a
filter. The env plays every race. Some days may be 100% WIN (small
fields with no EW terms) or heavily EW. A WIN-only model will see
fewer races per day on average, which affects bet counts and P&L
magnitude — but the scoreboard's normalisation by budget handles
this.
