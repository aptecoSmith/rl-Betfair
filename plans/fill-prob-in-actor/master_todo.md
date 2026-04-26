---
plan: fill-prob-in-actor
status: draft
---

# Master todo — fill-prob-in-actor

Two sessions:

- **Session 01** — architectural change + tests (no probe).
- **Session 02** — probe scaffolding + run + analysis.

## Session 01 — implement + test

### Read first

- [ ] `plans/fill-prob-in-actor/purpose.md` and `hard_constraints.md`.
- [ ] `plans/selective-open-shaping/lessons_learnt.md` — Sessions
      03 + 04, the predecessor analysis.
- [ ] `agents/policy_network.py` — three policy classes:
  - `PPOLSTMPolicy` (line ~440 onwards; actor_head at ~565,
    fill_prob_head at ~590, forward at ~720).
  - `PPOTimeLSTMPolicy` (line ~900 onwards; fill_prob_head at
    ~1053, forward at ~1170).
  - `PPOTransformerPolicy` (line ~1370 onwards; fill_prob_head
    at ~1479, forward at ~1580).
- [ ] CLAUDE.md "Recurrent PPO: hidden-state protocol on update"
      and "Transformer context window" — pattern for
      architecture-hash-distinct variants.
- [ ] `registry/model_store.py` — the arch-hash check.

### Implement (in order)

- [ ] **PPOLSTMPolicy**:
  - Update `actor_input_dim = runner_embed_dim + lstm_hidden + 1`
    at construction. Hard_constraints §3.
  - In `forward()`: move fill_prob computation BEFORE actor.
    Concat fill_prob.unsqueeze(-1) into actor_input.
    Hard_constraints §2.
  - Keep risk_head computation order as-is (it's a separate
    head, doesn't feed actor).

- [ ] **PPOTimeLSTMPolicy**: same change. Same line ordering.

- [ ] **PPOTransformerPolicy**: same change. Note Transformer's
      backbone state is `out_last` (not `lstm_last`); the concat
      pattern is identical otherwise.

- [ ] Verify `migrate_*` helpers in `policy_network.py` (lines
      ~50–300) don't need to change — they migrate state_dicts
      between intra-architecture versions; this is an inter-
      architecture change so the arch-hash break is the
      correct-by-default behaviour.

### Regression tests

- [ ] `tests/test_policy_network.py` (or new file) gains a
      `TestFillProbInActor` class with one test per policy class:
  - `test_lstm_actor_input_includes_fill_prob`
  - `test_time_lstm_actor_input_includes_fill_prob`
  - `test_transformer_actor_input_includes_fill_prob`
  Each asserts `actor_head[0].weight.shape[1] ==
  runner_embed_dim + hidden + 1`, runs a forward pass, and
  checks `output.action_mean` shape.

- [ ] `test_action_mean_depends_on_fill_prob_head_weights` (one
      per class). Hard_constraints §10 — gradient-through check.
      Build policy, capture action_mean, perturb
      `fill_prob_head.weight`, re-run forward, assert action_mean
      changed. This catches accidental detach.

- [ ] `test_pre_plan_weights_fail_to_load` — Hard_constraints §4.
      Build a state_dict with the OLD `actor_head[0].weight` shape
      (`runner_embed_dim + lstm_hidden`, no +1). Try to load into
      the new policy. Assert failure with a shape-mismatch error
      mentioning `actor_head`.

### Verify existing guards still pass

- [ ] `pytest tests/test_ppo_trainer.py -x` (full suite, esp.
      `TestRecurrentStateThroughPpoUpdate`).
- [ ] `pytest tests/test_forced_arbitrage.py -x` (esp.
      `TestSelectiveOpenShaping` — the open-cost math is
      unchanged).
- [ ] `pytest tests/test_mark_to_market.py -x`.
- [ ] `pytest tests/test_population_manager.py -x`.
- [ ] `pytest tests/test_policy_network.py -x` (full file).

If any of those fail, debug — don't paper over by editing the
existing tests. Hard_constraints §11.

### Smoke check (still no probe)

- [ ] In a Python REPL or one-shot script, instantiate each of the
      three policy classes with default config; run a forward pass
      on a synthetic obs batch; verify no exceptions and shapes
      match expectations.

### CLAUDE.md

- [ ] Add subsection per Hard_constraints §12. Match the style of
      "Recurrent PPO: hidden-state protocol on update" — concrete,
      load-bearing, names the binding tests.

### Lessons learnt

- [ ] Write Session 01 entry in `lessons_learnt.md`. Cover:
  - What landed (the three forward-pass changes + tests).
  - Whether anything in `migrate_*` needed to change.
  - Whether any pre-existing test surfaced an unexpected
    coupling (e.g., a test that mocked actor_head with the old
    input shape — those need updating, document why).
  - Any architectural ambiguity resolved (e.g., where exactly to
    place fill_prob computation in the Transformer's forward).

---

## Session 02 — probe scaffolding + run + analysis

### Scaffold the probe

- [ ] Create training plan JSON
      `selective-open-shaping-plus-fill-prob-probe`. Cohort label
      `F`. Seed 8403.
- [ ] Population 6, single arch (`ppo_time_lstm_v1`), n_epochs 2,
      n_generations 1, auto_continue false.
- [ ] hp_ranges per Hard_constraints §13:
  - `open_cost` [0.0, 1.0]
  - `matured_arb_bonus_weight` [5.0, 20.0]
  - `fill_prob_loss_weight` [0.0, 0.3]
  - All other genes cloned from cohort-O.
- [ ] notes section: state hypothesis, baseline (cohort-O ρ=+0.055,
      cohort-O2 ρ=+0.314), success threshold (ρ ≤ −0.5),
      pivot/close criteria.
- [ ] status `draft` until operator launches.

### Run

- [ ] Operator launches via the usual worker flow. NOT auto-
      starting from the plan.
- [ ] Monitor first 2–3 episodes for `bet_count = 0` collapse
      under any active gene.

### Analysis

- [ ] When all 6 agents complete: compute Spearman correlations
      following the cohort-O/O2 pattern:
  - ρ(open_cost, fc_rate) — primary metric.
  - ρ(open_cost, pairs_opened).
  - ρ(open_cost, arbs_closed).
  - ρ(fill_prob_loss_weight, fc_rate) — secondary, confirms
    actor uses the new input.
  - ρ(open_cost, oc_shaped) — sanity (expect ≈ −0.95 still).
- [ ] Compare same-day per-episode pairs_opened against cohort-O
      cohort-O2 same-day baselines.
- [ ] Write Session 02 entry in `lessons_learnt.md` per the
      decision matrix in Hard_constraints §13.

### Decision

- [ ] If ρ(open_cost, fc_rate) ≤ −0.5: promote architecture +
      promote `open_cost` to the production gene set. Update the
      operator's canonical training-plan template. Plan status
      `complete`.
- [ ] If between −0.5 and −0.2: scaffold a 12-agent follow-on
      probe with the same gene ranges. Plan status `partial`.
- [ ] If within ±0.2: write the negative-result entry. Plan
      status `complete` with note that selective-open-shaping
      stays closed and the selectivity problem is elsewhere.

---

## Close-out

- [ ] CLAUDE.md updated.
- [ ] `lessons_learnt.md` has Session 01 + Session 02 entries.
- [ ] If promoted: update `selective-open-shaping/master_todo.md`
      with a closing note that the dead-end was unblocked here,
      and re-promote the open_cost gene with the new architecture
      as a prerequisite.
- [ ] If discarded: the architectural change stays in (it's not
      gene-gated, no revert). The probe ruled out one more cause
      and that's a useful negative result on its own.
