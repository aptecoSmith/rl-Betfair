# Session 14 — Arch-specific ranges beyond `learning_rate`

## Before you start — read these

- `../purpose.md`
- `../hard_constraints.md` — constraint 7 (sampled ≠ used) applies
- `../master_todo.md` (Session 14)
- `../progress.md` — ideally Session 11 flagged either transformer
  under-training or LSTM under-exploration as evidence that the
  single shared range was too wide. If no such signal exists,
  reconsider whether this session is warranted.
- `../lessons_learnt.md`
- `../../arch-exploration/session_6_transformer_arch.md` — the
  session that introduced `TrainingPlan.arch_lr_ranges` and
  scoped it to `learning_rate` only. Read the "Out of scope"
  block.
- `../ui_additions.md`
- `../initial_testing.md`

## Goal

Widen `TrainingPlan.arch_lr_ranges` (or rename it) so per-
architecture overrides can apply to more than just `learning_rate`.
Expected use cases:

- Different `entropy_coefficient` ranges for LSTMs vs transformers
- Different `ppo_clip_epsilon` ranges if stability differs by arch
- Structural knobs that only make sense for one arch (hide
  `lstm_*` when selecting `ppo_transformer_v1` and vice versa)

## Scope

**In scope:**

1. **Generalise the override field.** Rename
   `TrainingPlan.arch_lr_ranges` to `TrainingPlan.arch_hp_ranges`
   (or similar — pick a name, document the choice in
   `design_decisions.md`). Data shape:
   ```
   {
     "ppo_lstm_v1": {
       "entropy_coefficient": {"type": "float", "min": ..., "max": ...},
       ...
     },
     "ppo_transformer_v1": {...}
   }
   ```
   Any gene not listed for an arch falls back to the global
   `hp_ranges` block.

2. **Backward compat.** Plans persisted under the old
   `arch_lr_ranges` key must still load. Either:
   - Keep the old field name, treat it as a reserved alias, and
     map it to the new field at load time, **or**
   - Add a migration step in `TrainingPlan.from_dict` that
     promotes old-shape values into the new shape.
   Document the choice in `design_decisions.md`.

3. **Gene scoping by architecture.** Structural genes only
   relevant to one arch (`lstm_num_layers`, `lstm_dropout`,
   `lstm_layer_norm`, `transformer_heads`, `transformer_depth`,
   `transformer_ctx_ticks`) must be excluded from sampling when
   the sampled `architecture_name` is the other arch. Today these
   get sampled for every agent regardless. Implementation: either
   mark each spec with an `applicable_to: [arch1, arch2]` list, or
   filter in the population manager at sampling time. Pick one;
   document in `design_decisions.md`.

4. **`PopulationManager.initialise_population` honours the new
   shape.** When building Gen 0 from a plan, merge the arch-
   specific overrides on top of the global `hp_ranges` for each
   agent based on its sampled architecture. The merge order is:
   arch-specific wins over global. Global wins over `config.yaml`.

5. **UI support.** The plan editor must be able to edit per-arch
   overrides. Make this additive: the existing single-range
   editor stays, and an "add arch-specific override" button opens
   a collapsible panel per architecture.

**Out of scope:**

- Inventing new genes. This session is a shape change only.
- Mutation logic for the new shape. Mutation continues to use
  whichever range an agent's arch resolved to at Gen 0. An agent
  whose arch changes mid-lineage (after the cooldown expires)
  resolves to the new arch's range at that point.
- A schema-registry overhaul. If the current specs are hard to
  annotate with `applicable_to`, do the minimum viable thing and
  document the debt.

## Tests to add

Create `tests/next_steps/test_arch_specific_ranges.py`:

1. **Plan round-trip under old shape.** Load a plan with the old
   `arch_lr_ranges` field and assert it loads into the new
   structure without data loss.

2. **Plan round-trip under new shape.** Save a plan with per-arch
   overrides for multiple genes, reload, assert equality.

3. **Merge priority.** Given global `entropy_coefficient` range
   [0.001, 0.05] and a transformer-specific override [0.005, 0.02],
   a transformer agent samples from [0.005, 0.02] and an LSTM
   agent samples from [0.001, 0.05]. Test via a fixed-seed
   population.

4. **Structural gene scoping.** An LSTM agent has `lstm_*` keys
   and no `transformer_*` keys. A transformer agent has
   `transformer_*` keys and no `lstm_*` keys. This is the most
   important test — it's where the current behaviour is
   silently wrong.

5. **Default plan still works.** A plan with no arch-specific
   overrides produces the same samples as before the rename.
   Guard against silent regressions for config-only users.

All CPU, all fast.

## Manual tests

- **M1 (UI smoke)** — confirm the per-arch override editor
  renders, is editable, and persists.

## Session exit criteria

- All tests pass.
- Old-shape plans still load.
- `design_decisions.md` updated with the rename decision AND the
  structural-gene scoping decision.
- `ui_additions.md` Session 14 entries added.
- `lessons_learnt.md` updated — structural gene scoping will
  probably surface surprises.
- `master_todo.md` Session 14 ticked.
- Commit.

## Do not

- Do not widen every gene to per-arch by default. The plan
  editor should default to shared ranges; per-arch overrides are
  opt-in.
- Do not silently drop the old `arch_lr_ranges` field. Migration
  or alias, pick one, don't lose data.
- Do not try to mutate between per-arch override shapes. Once
  resolved at init time, the range is fixed for that
  (agent, arch) pair.
