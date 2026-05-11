# Hard constraints

1. **No new gene fields.** Six existing Phase 5 genes are
   activated via `--enable-gene`. No edits to
   `training_v2/cohort/genes.py` to add new gene fields.
2. **No new shaping terms in env or trainer.** The genes
   activated here all already have wired-in reward / loss
   pathways. Do not add a new shaping term to make a gene
   "work better."
3. **Predictor bundle is frozen at production champions** —
   same three manifests as the predecessor cohort.
4. **Lean obs schema preserved.** `--predictor-lean-obs` set;
   `obs_dim=504`. Cross-loading with the predecessor cohort's
   weights is NOT a goal (different gene distributions →
   different policies — fresh start is correct).
5. **3-day held-out reeval is the verdict.** Same eval days as
   the predecessor (`2026-05-04 / 05 / 06`). Don't change the
   eval window mid-flight to flatter the result.
6. **Success bar is concrete.** ≥3 of top-5 profitable on the
   3-day held-out window. Anything less is a failure verdict
   and triggers a stop, not a quiet pivot to a different
   metric.
7. **The gene activation list is locked.** Six genes (see
   `master_todo.md`). Adding a seventh mid-run after seeing
   results is reward-hacking on the GA's noise.
