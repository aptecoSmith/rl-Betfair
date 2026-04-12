# Master TODO — Stud Models

## Session 1: Backend + wizard UI

### Backend — stud model config

- [ ] Add `stud_model_ids: list[str]` to `StartTrainingRequest` in
      `api/schemas.py` (default: empty list)
- [ ] Add `stud_model_ids` to `TrainingPlan` model
- [ ] Validate on start: all stud IDs exist in registry, have saved
      weights and hyperparameters. Reject with clear error if not.
- [ ] Pass through worker → orchestrator

### Backend — studs in breeding

- [ ] In `run_training.py::_run_generation()`, after selection and
      before breeding:
      1. Load stud model hyperparameters from ModelStore
      2. Add stud model IDs to the survivor list as parent-only
         contributors (they don't take a survivor slot — children
         count is still `population_size - run_survivors`)
      3. When breeding, guarantee each stud is selected as parent_a
         or parent_b for at least one child per generation
- [ ] Implementation option: reserve N breeding slots (1 per stud)
      where one parent is always the stud. The other parent is a
      random survivor. Remaining slots are normal breeding.
- [ ] Studs are NOT trained or evaluated — only their HP dict is used
- [ ] Log clearly: "Stud model abc123 used as parent for 2 children"

### Backend — stud guarantee

- [ ] If there are more studs than breeding slots (unlikely but
      possible), rotate: each stud gets at least one child across
      generations, not necessarily every generation
- [ ] If there are 0 children to breed (survivors >= population),
      studs still can't force additional children. Log a warning:
      "Studs configured but no breeding slots available"

### Wizard UI — stud picker

- [ ] Add "Stud models" section to wizard step 4 (genetics) or step 6
- [ ] Model picker: searchable dropdown of all garaged + active models
      from the scoreboard. Show model ID (short), architecture,
      composite score
- [ ] Selected studs shown as chips with remove button
- [ ] Help text: "Select models whose hyperparameters you want bred
      into every generation. Studs are guaranteed to be parents —
      they contribute their configuration to children via crossover,
      regardless of how they'd score in selection. Use this to
      preserve specific traits you know are valuable. Studs are not
      re-trained — only their hyperparameters are used."
- [ ] Limit: max 5 studs (more than this and you're replacing most
      of the breeding with forced parents)

### Training plan UI

- [ ] Add stud picker to training plans editor
- [ ] Same component/pattern as wizard

### Activity log

- [ ] Log stud usage per generation: which studs were parents, which
      children they produced
- [ ] Show in training monitor activity log

### Tests

- [ ] Test: stud model HP loaded and used as parent
- [ ] Test: each stud is parent of at least one child per generation
- [ ] Test: studs don't take survivor slots (n_children unchanged)
- [ ] Test: invalid stud ID rejected at start
- [ ] Test: empty stud list = current behaviour (backward compatible)
- [ ] Test: more studs than breeding slots logs warning

### Verify

- [ ] `python -m pytest tests/ --timeout=120 -q` — all green
- [ ] `cd frontend && ng build` — clean
