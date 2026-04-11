# Issue Priority Order

Recommended execution order based on dependencies, impact, and effort.

## Also in flight (separate plan folders)

These two plans are already partially complete and should finish first:
- `plans/ew-metadata-pipeline/` — all 4 sessions done
- `plans/bet-explorer-redesign/` — UI work (depends on ew-metadata)

---

## Issues — recommended order

| # | Issue | Sessions | Why this order |
|---|---|---|---|
| 1 | **05 — Fix failing tests** | 1 | Housekeeping. Get the suite green before adding more code. Quick win, builds trust in CI. |
| 2 | **03 — Training log autoscroll** | 1 | Trivial quality-of-life. One session, immediate payoff. Do it between bigger tasks. |
| 3 | **07 — Anti-passivity** | 1 | Small change, directly improves training quality. One session. Should land before the next training run. |
| 4 | **01 — Configurable budget** | 5 | Enables £10/race training (operator's original request). Unblocks experimentation. Medium effort but high value. |
| 5 | **04 — Market type filter** | 3 | New gene for WIN/EW/BOTH/FREE_CHOICE. Enables specialised models. Benefits from anti-passivity landing first (EW-only models need engagement nudge). |
| 6 | **02 — Training stop options** | 4 | Quality-of-life for the training workflow. Not blocking anything — current stop/finish works, just lacks granularity. |
| 7 | **06 — Managed HP search** | 7 | Largest piece of work. Strategic, not urgent. Benefits from having more trained models in the registry first (coverage analysis needs data). Do this after several training runs. |

## Suggested batching

**Sprint 1 (quick wins):** 05 + 03 + 07 = 3 sessions
Get the test suite green, add autoscroll, add inactivity penalty.
Run a training session after this to validate.

**Sprint 2 (training config):** 01 + 04 = 8 sessions
Configurable budget + market type filter. Run training at £10 with
WIN-only and EW-only models to validate.

**Sprint 3 (workflow):** 02 = 4 sessions
Training stop dialog with granularity options.

**Sprint 4 (strategic):** 06 = 7 sessions
Managed hyperparameter search. By this point there should be enough
models in the registry for the coverage analysis to be meaningful.

## Notes

- Issues 03, 05, 07 are independent and can run in parallel sessions.
- Issue 04 benefits from 07 (anti-passivity) landing first but doesn't
  strictly depend on it.
- Issue 06 benefits from 01 (configurable budget) and 04 (market type
  filter) landing first — more diverse models = better coverage data.
- All issues are independent of the ew-metadata and bet-explorer plans.
