# Lessons Learnt — Managed Hyperparameter Search

Append-only. Date each entry.

---

## 2026-04-11 — Discovery: existing coverage infrastructure

The codebase already has ~170 lines of coverage analysis and biased
sampling in `training/training_plan.py` (`compute_coverage()`,
`bias_sampler()`, `sample_with_bias()`). These were built but never
wired into `initialise_population()`. The coverage API endpoint
(`GET /api/training-plans/coverage`) also exists and works.

This means the "coverage-biased" strategy is ~80% implemented. The
main work is:
1. The Sobol generator (new code).
2. Seed-to-population perturbation (new code in pop manager).
3. The exploration log DB table (new schema).
4. Wiring it all into the orchestrator (glue code).
5. Frontend (new UI).

The existing `CoverageReport` buckets numeric genes into 10 bins and
flags architectures with < 15 samples. The bias is modest (1.5x for
empty buckets vs 1.0x) �� this is sensible for a first pass but may
need tuning after we see real coverage data.

Key design choice: the seed point defines the *centre* of the
initial population, not a hard constraint. Individual models are
perturbed around it (±10% of range). This ensures the genetic
algorithm has diversity to work with while staying in the target
region.
