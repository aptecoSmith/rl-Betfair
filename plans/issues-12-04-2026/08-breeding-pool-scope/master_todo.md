# Master TODO — Breeding Pool Scope

## Session 1: Configurable breeding pool + debug 0-children

### Debug the "0 children" anomaly

- [ ] Trace the exact scenario: if a shakeout script or non-standard
      run sets population_size to a value ≤ number of survivors, breed
      produces 0 children. Find where this happened (check
      `scripts/session_9_shakeout.py` and any non-standard runs)
- [ ] Add a warning log when `n_children <= 0` in
      `population_manager.py::breed()` — this is surprising and should
      be visible
- [ ] Consider whether `n_children = 0` should be an error or just a
      warning. It's valid if the user intentionally ran a large
      population and selected aggressively, but it defeats the purpose
      of the genetic algorithm

### Add breeding_pool config option

- [ ] Add `breeding_pool` to `config.yaml` under `population`:
      `breeding_pool: run_only | include_garaged | full_registry`
      (default: `run_only` — preserves current behaviour)
- [ ] Add to wizard UI (step 4 genetics or step 6 parameters) with
      help text explaining each option
- [ ] Pass through `StartTrainingRequest` → worker → orchestrator

### Implement pool expansion in run_training.py

- [ ] `run_only` (default): keep current scoping at lines 688-692
- [ ] `include_garaged`: after filtering to `run_ids`, also include
      scores for garaged models. Need to:
      - Load garaged model IDs from ModelStore
      - Include their scores from the scoreboard
      - Add them to `run_scores` before passing to `select()`
      - Garaged models that survive selection become parents but
        are NOT re-trained — only their hyperparameters are used
- [ ] `full_registry`: use the unfiltered scoreboard scores
      (all active models). Same caveat — non-run models are parents
      only, not re-trained

### Handle "parent-only" models in breeding

- [ ] When a garaged/registry model survives selection, it contributes
      hyperparameters to children but does NOT count toward the
      survivor slots that reduce `n_children`. Otherwise you'd get
      fewer children than intended.
- [ ] Alternative: count them as survivors but don't re-train them.
      They carry forward their existing evaluation scores.
- [ ] Decision needed: does a garaged parent take a slot in the next
      generation (reducing children), or is it parent-only (children
      are always `population_size - run_survivors`)?

### Tests

- [ ] Test: `run_only` mode produces identical behaviour to current code
- [ ] Test: `include_garaged` mode adds garaged model scores to pool
- [ ] Test: `full_registry` mode uses all active scores
- [ ] Test: garaged parents contribute hyperparameters to children
- [ ] Test: warning logged when `n_children <= 0`

### Verify

- [ ] `python -m pytest tests/ --timeout=120 -q` — all green
- [ ] `cd frontend && ng build` — clean
