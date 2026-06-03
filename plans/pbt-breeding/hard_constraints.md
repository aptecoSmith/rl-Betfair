# pbt-breeding — hard constraints

1. **Opt-in, default byte-identical.** The new breeding path is behind a flag
   (e.g. `--breeding pbt`). With the flag off, `_breed_next_generation` is the
   current gene-only GA, byte-identical. No silent behaviour change to existing
   launches.

2. **Day-rotation is mandatory under warm-start (coupled, not optional).**
   Warm-start without rotating the training days memorises them. The PBT path
   MUST rotate training days per generation from a larger pool, select on
   rotating held-out days, and keep the sealed final test untouched. Shipping
   warm-start with fixed training days is a rejected design.

3. **Sealed final test stays sealed.** The held-out test days (e.g. May 20–29)
   are never in training OR selection for EITHER arm of the A/B. All verdicts
   are on those sealed days (`feedback_always_eval_holdout`).

4. **Paired A/B or it doesn't count.** PBT vs gene-only GA run on the SAME
   cohort seed, SAME day pool, SAME agent count/gens — the only difference is
   the breeding mechanism. Determinism preserved so the comparison is paired
   and the run is resumable.

5. **Identity inheritance must be real, and measured.** A child that "inherits"
   a brain must load the parent's actual weights (verified: child's gen-0
   forward == parent's final forward on a fixed obs, before any new training).
   Heritability is a reported metric, not an assumption.

6. **Diversity is a first-class metric, not theatre.** The immigrant tier must
   demonstrably contribute survivors; recipe + behavioural diversity is tracked
   per generation. If immigrants never survive, the protection/quota is wrong —
   surface it, don't bury it.

7. **No selection on a noisy single draw silently.** If warm-start does NOT
   collapse the selection-spread-vs-signal ratio (purpose.md success (b)), the
   plan has not delivered its core claim — report it honestly rather than
   declaring success on held-out cash alone.

8. **Compute budget stays bounded.** Warm-start trains everyone every gen;
   keep wall-clock at parity with the current GA by tuning episodes/gen down
   (warm policies need fewer). Measure per-gen wall; don't let the population
   silently balloon training time.

9. **Fresh blood gets the FULL gene space.** Rookies sample EVERY gene across
   its full range/choices — no `enabled_set` disbarring, including the
   architecture genes. Maximal exploration is the point; selection + held-out
   do the filtering, not a restricted search space.

10. **Structural genes freeze within a lineage.** Architecture type + sizing
    (`architecture`, `hidden_size`/`d_model`, transformer `depth`/`n_heads`/
    `ctx_ticks`, lstm `num_layers`) are set only at fresh-blood birth and never
    mutated for that lineage — warm-start weight inheritance requires matching
    weight shapes. Offspring inherit the parent's architecture + weights and
    perturb only non-structural genes. A breed step that changes a structural
    gene on an inheriting child is a bug (it silently can't load the weights).

11. **One policy factory, two callers.** The worker AND
    `tools/reevaluate_cohort.py` build policies from a genome through the SAME
    `build_policy(genes)` factory. (The held-out re-eval already bit us by
    rebuilding the policy differently from training — input_norm. With multiple
    architectures in play, a single factory is mandatory, not optional.)
