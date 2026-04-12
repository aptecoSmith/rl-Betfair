# Master TODO — Adaptive Breeding & Mutation Controls

## Session 1: Detection + response config + mutation controls

### Define "bad generation" threshold

- [ ] Add config options under `population`:
      ```yaml
      population:
        bad_generation_threshold: 0.2   # composite score floor
        bad_generation_policy: persist   # persist | boost_mutation | inject_top
        adaptive_mutation: false
        adaptive_mutation_increment: 0.1
        adaptive_mutation_cap: 0.8
      ```
- [ ] A generation is "bad" when the best composite score across all
      agents is below `bad_generation_threshold`
- [ ] Could also use: mean composite score, or percentage of agents
      meeting discard criteria. Start with best-score — simplest and
      most meaningful

### Implement bad generation detection

- [ ] In `run_training.py::_run_generation()`, after scoring and before
      selection: check if `max(score.composite_score for score in run_scores)`
      is below threshold
- [ ] Log clearly: "Generation N underperformed (best: 0.12, threshold: 0.20)"
- [ ] Publish a WebSocket event so the UI can show it

### Implement response policies

- [ ] **persist** (default): do nothing different. Current behaviour.
      Breed from survivors as normal. Log a warning but continue.
- [ ] **boost_mutation**: increase mutation_rate for this generation's
      breeding by `adaptive_mutation_increment`. Pass the boosted rate
      to `breed()`. Log: "Boosting mutation rate: 0.3 → 0.4"
- [ ] **inject_top**: load top N garaged/scoreboard models and add
      them as parent-only contributors to the breeding pool (same
      mechanism as issue 08 `include_garaged`). Log which models
      were injected and why.
      Note: this depends on issue 08 landing first. If not yet done,
      implement the parent injection inline.

### Adaptive mutation (automatic ramp)

- [ ] When `adaptive_mutation: true`, track consecutive bad generations
- [ ] Each consecutive bad gen: mutation_rate += increment (capped)
- [ ] When a generation is good (above threshold): reset mutation rate
      back to base
- [ ] Log the current effective mutation rate each generation
- [ ] Store the ramp state in the orchestrator (resets per run)

### Wizard — mutation controls (step 4 or new step)

- [ ] Add mutation rate override to wizard:
      - Base mutation rate slider/input (default: 0.3, range: 0.05-1.0)
      - Help text: "Probability of mutating each hyperparameter when
        breeding. Higher = more exploration, lower = more exploitation
        of what's already working. 0.3 is a good starting point."
- [ ] Add adaptive mutation toggle:
      - Checkbox: "Increase mutation when generations underperform"
      - When checked, show increment and cap inputs
      - Help text: "If every model in a generation scores below the
        threshold, the mutation rate increases to shake up the search.
        Resets when a generation performs well."
- [ ] Add bad generation policy selector:
      - Radio: persist / boost mutation / inject top performers
      - Help text per option explaining what happens
- [ ] Add bad generation threshold input:
      - "Minimum best composite score to consider a generation
        successful"
      - Help text: "If the best model in a generation scores below
        this, the bad-generation policy kicks in. Set to 0 to disable."

### Pass through API

- [ ] Add mutation_rate, adaptive_mutation, bad_generation_policy,
      bad_generation_threshold to `StartTrainingRequest`
- [ ] Worker passes these to the orchestrator
- [ ] Orchestrator reads them alongside config values (run override
      takes precedence over config.yaml)

### Training monitor — visibility

- [ ] Show effective mutation rate in the phase banner or detail line
      when it differs from base
- [ ] Show "Generation N underperformed — [policy applied]" in the
      activity log
- [ ] If inject_top: show which models were injected

### Tests

- [ ] Test: bad generation detected when best score < threshold
- [ ] Test: persist policy — breeds normally, logs warning
- [ ] Test: boost_mutation — mutation_rate increased for that breed
- [ ] Test: inject_top — top garaged models added as parents
- [ ] Test: adaptive mutation ramps up over consecutive bad gens
- [ ] Test: adaptive mutation resets on a good generation
- [ ] Test: mutation rate capped at adaptive_mutation_cap
- [ ] Test: wizard override takes precedence over config.yaml

### Verify

- [ ] `python -m pytest tests/ --timeout=120 -q` — all green
- [ ] `cd frontend && ng build` — clean
