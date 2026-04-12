# Master TODO — Mutation Count Cap

## Session 1: Implement cap + wizard control

### Backend — mutation cap in mutate()

- [ ] Add `max_mutations_per_child: int | null` to config.yaml under
      `population` (default: null = unlimited, current behaviour)
- [ ] In `population_manager.py::mutate()`: when cap is set, change
      the mutation strategy:
      1. Collect all eligible genes (not on cooldown, not backfill-only)
      2. Randomly select min(cap, len(eligible)) genes to mutate
      3. Only mutate those genes — skip the rest
      4. Within selected genes, apply the same Gaussian/choice mutation
         as today
- [ ] When cap is null, keep current per-gene coin-flip behaviour
      (backward compatible)
- [ ] Log: "Mutating 2/30 genes: learning_rate, entropy_coefficient"

### Wizard UI

- [ ] Add "Max mutations per child" input to wizard step 4 (genetics)
      or step 6 (training parameters)
- [ ] Default: blank (unlimited). Range: 1 to number of genes.
- [ ] Help text: "Limits how many hyperparameters change at once when
      breeding. Lower values (1-3) make it easier to understand what
      made a model better or worse, but explore the search space more
      slowly. Leave blank for the default behaviour where each gene
      has an independent 30% chance of mutating (~9 changes at once
      with 30 genes)."
- [ ] Pass through StartTrainingRequest → worker → orchestrator →
      breed() → mutate()

### Training plan support

- [ ] Add `max_mutations_per_child` to TrainingPlan model
- [ ] Add to training plans editor with same help text

### Tests

- [ ] Test: cap=2 → exactly 2 genes mutated (or fewer if only 1 eligible)
- [ ] Test: cap=1 → single gene changed, rest unchanged
- [ ] Test: cap=null → current coin-flip behaviour (backward compatible)
- [ ] Test: cap > eligible genes → mutates all eligible (no error)
- [ ] Test: architecture cooldown still respected when cap is active
- [ ] Test: mutation deltas correctly recorded for capped mutations

### Verify

- [ ] `python -m pytest tests/ --timeout=120 -q` — all green
- [ ] `cd frontend && ng build` — clean
