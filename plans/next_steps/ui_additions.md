# UI Additions — Next Steps

Running list of UI work required by sessions promoted from
`next_steps.md`. Same contract as
`arch-exploration/ui_additions.md`: every new configurable value
introduced by a session in this folder must be appended here before
that session is marked complete, and a dedicated UI session consumes
the list and wires everything up.

**Rule:** a session is not complete until any new configurable
values it introduces have been added to the list below.

No items yet — this file is populated as sessions in this folder are
scoped and landed.

---

## Items by backlog entry

Entries correspond to the numbered items in `next_steps.md`. Only
populated once the item is promoted into a session.

### 1. Multi-generation exploration run

TBD. The run itself is triggered from the existing training-plan UI
built in Session 4 + wired in Session 8 of the previous phase, so
most of the UI is already in place. Items to confirm rather than
build:

- [ ] Confirm the training-plan editor supports the full set of
      genes we want to search over (PPO + reward + structural +
      transformer).
- [ ] Confirm the coverage page correctly reflects historical
      samples from the Session 9 shakeout once the real run is
      under way (i.e. coverage grows, poorly-covered genes move
      toward well-covered).
- [ ] Any new "stop criteria" or "generations planned" fields on
      the plan editor? If the session prompt introduces these,
      append the UI tasks here.

### 2. Hold-cost reward term

- [ ] Gene editor entry for whatever the final design-pass gene
      name is. (Append after design pass lands.)
- [ ] Tooltip explaining the zero-mean property so users know the
      opt-out (setting the gene to 0) is clean.

### 3. Risky PPO knobs

- [ ] `mini_batch_size` (choice editor — do NOT use a free-form
      integer widget; we want discrete safe values only).
- [ ] `ppo_epochs` (choice editor, small discrete set).
- [ ] *(Probably not)* `max_grad_norm`. Only add if the session
      actually promotes it.

### 4. Arch-specific ranges beyond `learning_rate`

- [ ] Extend the `TrainingPlan.arch_lr_ranges` editor to cover
      whichever additional genes the session promotes
      (`entropy_coefficient`, `ppo_clip_epsilon`, structural knobs
      that only apply to one arch).
- [ ] UI must hide structural knobs that don't apply to the
      selected architecture (e.g. `lstm_dropout` is meaningless
      for `ppo_transformer_v1`). This was noted but not done in
      the previous phase — worth catching here.

### 5. Fourth architecture candidates

- [ ] Update the architecture-choice widget to include the new
      arch name.
- [ ] Any arch-specific hyperparameters get their own gene
      editors.

### 6. Pre-existing `obs_dim` test fix

No UI work. Pure test maintenance.

### 7. Coverage math upgrades

- [ ] Probably no UI work in the first pass — upgrades happen in
      the backend and surface through the existing coverage page.
      Re-check when the session is scoped.

### 8. LSTM `num_layers > 2`

- [ ] If promoted, widen the existing `lstm_num_layers` choice
      editor. One-line change.

### 9. Market / runner encoder changes

TBD — depends entirely on what the session proposes. Likely new
choice editors for encoder type / depth / width.

### 10. Optimiser / schedule work

- [ ] LR warmup steps (int editor).
- [ ] Weight decay (float editor).
- [ ] Optimiser choice if we want to vary Adam vs AdamW.

### 11. Housekeeping

- [x] Confirm the Session 8 "schema inspector" read-only view is
      actually live. **Session 10:** confirmed — lives at
      `frontend/src/app/schema-inspector/` and reads from the
      `get_hyperparameter_schema` endpoint which projects over
      `config.yaml → hyperparameters.search_ranges` directly. No
      hardcoded gene list. Human-click verification deferred to
      the next session that brings up the stack.
- [x] Confirm no UI references to `observation_window_ticks` or
      the scalar `reward_early_pick_bonus` remain. **Session 10:**
      grep of `frontend/` and `api/` returns zero hits for either
      name.
