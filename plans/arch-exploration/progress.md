# Progress — Architecture & Hyperparameter Exploration

Update this file at the end of each session. One short entry per
session: what shipped, what files changed, what tests were added, what
didn't ship and why.

Don't use this file for running thoughts — that's what
`lessons_learnt.md` is for. Keep entries factual.

---

## Session 0 — Design review & planning (2026-04-06)

**Shipped:**
- Created `plans/arch-exploration/` folder with planning docs.
- Design review: traced the reward-gene-plumbing bug, audited every
  hyperparameter in the current schema, confirmed `ppo_transformer_v1`
  is the named third architecture in `PLAN.md`.

**Files added:**
- `plans/arch-exploration/purpose.md`
- `plans/arch-exploration/master_todo.md`
- `plans/arch-exploration/progress.md` (this file)
- `plans/arch-exploration/lessons_learnt.md`
- `plans/arch-exploration/testing.md`
- `plans/arch-exploration/ui_additions.md`
- `plans/arch-exploration/session_1_reward_plumbing.md`
- `plans/arch-exploration/session_2_ppo_schema.md`
- `plans/arch-exploration/session_3_reward_schema.md`
- `plans/arch-exploration/session_4_training_plan.md`
- `plans/arch-exploration/session_5_lstm_structural.md`
- `plans/arch-exploration/session_6_transformer_arch.md`
- `plans/arch-exploration/session_7_drawdown_shaping.md`
- `plans/arch-exploration/session_8_ui_additions.md`
- `plans/arch-exploration/session_9_gpu_shakeout.md`

**Not shipped:**
- No code changes. This session was planning only.

**Next:** Session 1 — plumb reward genes through to env.

---

## Session 1 — Plumb reward genes through to env (2026-04-06)

**Shipped:**
- `BetfairEnv.__init__` accepts a new `reward_overrides: dict | None`
  kwarg. It overlays the overrides on top of a per-instance copy of
  `config["reward"]` (shared config is never mutated). Unknown keys are
  silently dropped after a one-shot `logger.debug` so a typoed gene
  name can't crash a multi-day run.
- `PPOTrainer.__init__` now caches `self.hyperparams` and builds a
  `self.reward_overrides` dict via the new `_reward_overrides_from_hp`
  helper. `_collect_rollout` passes that dict into every `BetfairEnv`
  it constructs — closing the dead path from `population_manager.py:220`
  to `env/betfair_env.py:226`.
- Gene-name → reward-config-key mapping lives in `_REWARD_GENE_MAP`
  inside `agents/ppo_trainer.py`. `reward_early_pick_bonus` pins both
  `early_pick_bonus_min` and `early_pick_bonus_max` to the same value
  (min == max → constant multiplier, no reward-formula change).
  Session 3 will split it into proper min/max genes.
- Retired `observation_window_ticks`: removed from `config.yaml`
  `search_ranges` and from the assertion list in `test_config.py`. No
  downstream code reads it, so no migration shim was needed.
- Added `gpu` and `slow` pytest markers in `pyproject.toml` and
  deselected them in the default `addopts`, matching the rules in
  `plans/arch-exploration/testing.md`.
- New test file `tests/arch_exploration/test_reward_plumbing.py` with
  8 CPU-only tests (covers all 6 session-plan items plus a
  shared-config-mutation guard and a direct unit test of
  `_reward_overrides_from_hp`). Runs in ~2.5 s.

**Files changed:**
- `env/betfair_env.py` — added `reward_overrides` kwarg,
  `_REWARD_OVERRIDE_KEYS` class attribute, merge logic.
- `agents/ppo_trainer.py` — added `_REWARD_GENE_MAP`,
  `_reward_overrides_from_hp`, `self.hyperparams`,
  `self.reward_overrides`, and updated the `BetfairEnv(...)`
  construction site in `_collect_rollout`.
- `config.yaml` — removed `observation_window_ticks` from
  `hyperparameters.search_ranges`.
- `pyproject.toml` — registered `gpu`/`slow` markers, updated
  `addopts` to deselect them by default.
- `tests/test_config.py` — drop `observation_window_ticks` from the
  expected-params list.
- `tests/arch_exploration/__init__.py` — new empty package marker.
- `tests/arch_exploration/test_reward_plumbing.py` — new 8-test file.
- `plans/arch-exploration/progress.md`, `lessons_learnt.md`,
  `ui_additions.md` — session notes.

**Tests:**
- `pytest tests/arch_exploration/` → 8 passed (~2.5 s).
- `pytest tests/test_betfair_env.py tests/test_config.py
  tests/test_population_manager.py tests/test_genetic_operators.py
  tests/test_orchestrator.py tests/test_genetic_selection.py
  tests/arch_exploration/` → 227 passed, 1 pre-existing failure
  deselected (`test_obs_dim_matches_env` hardcodes 1630 but actual is
  1636; confirmed stale on plain `master` via `git stash` → not
  caused by this session).

**Not shipped:**
- Did not touch the UI; Session 8 will consume `ui_additions.md`.
- Did not add `gamma`/`gae_lambda`/`value_loss_coeff` genes — that's
  Session 2.

**Next:** Session 2 — expand PPO hyperparameter schema.

---

## Session 2 — Expand PPO hyperparameter schema (2026-04-06)

**Shipped:**
- Added `gamma`, `gae_lambda`, and `value_loss_coeff` as float genes in
  `config.yaml` `hyperparameters.search_ranges`, matching the shape of
  the existing `ppo_clip_epsilon` / `entropy_coefficient` entries. Ranges
  are the ones specified in the session plan:
  - `gamma`: float [0.95, 0.999]
  - `gae_lambda`: float [0.9, 0.98]
  - `value_loss_coeff`: float [0.25, 1.0]
- No code changes in `sample_hyperparams` or `PPOTrainer.__init__`
  were needed — the sampler already handles `type: float`, and the
  trainer already reads all three via `hp.get(..., default)`. The
  `hp.get(..., default)` fallbacks were kept to protect old checkpoints
  loaded without these genes (explicit regression test added).
- New test file `tests/arch_exploration/test_ppo_schema.py` with 4
  CPU-only tests:
  1. `test_sampler_produces_ppo_genes_in_range` — sampler emits all
     three genes inside their declared ranges (50-iteration loop).
  2. `test_trainer_reads_ppo_genes_from_hyperparams` — extreme-value
     round-trip: construct `PPOTrainer` with values pinned near range
     edges and assert `trainer.gamma` / `.gae_lambda` / `.value_loss_coeff`
     match. This is the "gene is actually read" test mandated by
     `lessons_learnt.md`.
  3. `test_trainer_defaults_survive_when_genes_missing` — hardening
     around the `hp.get` fallbacks so a future refactor can't silently
     break old-checkpoint loading.
  4. `test_mutation_keeps_ppo_genes_in_range` — seeded RNG, 200
     mutation rounds at `mutation_rate=1.0`, assert every mutated
     value remains clamped inside `[spec.min, spec.max]`.
- `tests/test_config.py::test_hyperparameter_search_ranges_present`
  extended to require the three new keys.
- `ui_additions.md` Session 2 checklist already contained the three
  items; left as pending-UI (Session 8 consumes them — the server side
  is now live so they're eligible to be ticked there).

**Files changed:**
- `config.yaml` — added three `float` entries under
  `hyperparameters.search_ranges`.
- `tests/test_config.py` — added `gamma`, `gae_lambda`, `value_loss_coeff`
  to the required-params assertion list.
- `tests/arch_exploration/test_ppo_schema.py` — new 4-test file.
- `plans/arch-exploration/progress.md` — this entry.

**Tests:**
- `pytest tests/arch_exploration/test_ppo_schema.py tests/test_config.py`
  → 17 passed (~4.5 s).
- `pytest tests/arch_exploration/ tests/test_population_manager.py
  tests/test_genetic_operators.py` → 91 passed, 1 pre-existing failure
  (`test_obs_dim_matches_env` stale `1630` vs actual `1636` — documented
  in Session 1 progress/lessons as unrelated drive-by; out of scope).

**Not shipped:**
- `mini_batch_size`, `ppo_epochs`, `max_grad_norm` — explicitly out of
  scope per the session plan (GPU-memory / rollout-length interactions).
- Nothing touched in the UI; Session 8 consumes `ui_additions.md`.

**Next:** Session 3 — expand reward hyperparameter schema.

---

## Session 3 — Expand reward hyperparameter schema (2026-04-06)

**Shipped:**
- Added the four reward genes called for by the session plan to
  `config.yaml` `hyperparameters.search_ranges`:
  - `early_pick_bonus_min`: float [1.0, 1.3]
  - `early_pick_bonus_max`: float [1.1, 1.8]
  - `early_pick_min_seconds`: int [120, 900]
  - `terminal_bonus_weight`: float [0.5, 3.0]
- Removed the Session 1 stopgap `reward_early_pick_bonus` scalar gene
  (and the `("early_pick_bonus_min", "early_pick_bonus_max")` fan-out
  in `_REWARD_GENE_MAP`). The new genes use 1:1 passthrough mappings
  so the trainer's reward-overrides path stays mechanical.
- Added `reward.terminal_bonus_weight: 1.0` to the `config.yaml`
  reward block as the new default. The env reads it via
  `reward_cfg.get("terminal_bonus_weight", 1.0)` so existing
  test-config fixtures (which don't set it) keep working unchanged.
- Plumbed `terminal_bonus_weight` into `BetfairEnv` and applied the
  multiplier inside `step()` exactly where the existing terminal
  bonus is computed:
  `terminal_bonus = self._terminal_bonus_weight * day_pnl / starting_budget`.
  The result still lands in `_cum_raw_reward` because `day_pnl` is
  real cash, so scaling it does NOT break the zero-mean shaping
  invariant.
- Repair step (swap, not reject): if
  `early_pick_bonus_max < early_pick_bonus_min` after sampling /
  mutation / direct env construction, the two values are swapped.
  Implemented in **two** places (belt-and-braces):
  1. `agents/population_manager.py::_repair_reward_gene_pairs`,
     called at the end of `sample_hyperparams()` and `mutate()`. This
     ensures every persisted/logged genome shows the repaired values.
  2. `BetfairEnv.__init__` immediately after merging overrides, so a
     directly-constructed env with bad overrides still works.
- New test file `tests/arch_exploration/test_reward_schema.py` with
  6 CPU-only tests covering all six items from the session plan:
  1. Sampler emits all four genes within range AND post-sample
     repair holds (`max >= min`, int seconds stays int).
  2. Env picks up extreme overrides (env attrs match).
  3. Inverted interval is repaired (swap), env attrs reflect the
     swapped order.
  4. Symmetry: equal-magnitude winning + losing back bets at the
     same placement time produce zero net early-pick bonus, even
     with the widened multiplier range.
  5. `terminal_bonus_weight=2.0` lands in `info["raw_pnl_reward"]`
     (not `shaped_bonus`), and `raw + shaped ≈ total` still holds.
     Uses `unittest.mock.patch.object(BetManager, "settle_race", ...)`
     to inject a known race P&L so the weight has a non-zero base.
  6. 200 seeded mutations: every gene stays in range AND
     `early_pick_bonus_max >= early_pick_bonus_min` after each round.
- Updated existing tests that referenced the now-removed
  `reward_early_pick_bonus` gene (`test_config.py`,
  `test_reward_plumbing.py`).

**Files changed:**
- `config.yaml` — new reward gene under `reward.`, four new
  `search_ranges` entries, removed `reward_early_pick_bonus`.
- `agents/population_manager.py` — added `_repair_reward_gene_pairs`
  helper, called from end of `sample_hyperparams()` and end of
  `mutate()`.
- `agents/ppo_trainer.py` — `_REWARD_GENE_MAP` now contains the four
  new 1:1 passthrough entries; the `reward_early_pick_bonus` entry
  is gone.
- `env/betfair_env.py` — `_REWARD_OVERRIDE_KEYS` gains
  `terminal_bonus_weight`, `__init__` performs the repair swap and
  reads `_terminal_bonus_weight`, `step()` applies the weight to the
  terminal bonus.
- `tests/test_config.py` — expected-params list updated.
- `tests/arch_exploration/test_reward_plumbing.py` — three tests
  updated to drop the removed `reward_early_pick_bonus` references.
- `tests/arch_exploration/test_reward_schema.py` — new 6-test file.
- `plans/arch-exploration/progress.md`, `lessons_learnt.md` — this
  entry and a note on the patched-`settle_race` testing technique.

**Tests:**
- `pytest tests/arch_exploration/ tests/test_config.py` →
  31 passed (~14 s).
- `pytest tests/test_betfair_env.py tests/test_population_manager.py
  tests/test_genetic_operators.py` → 138 passed, 1 pre-existing
  failure (`test_obs_dim_matches_env` stale `1630` vs actual `1636`,
  documented in Session 1 progress as unrelated).

**Not shipped:**
- No new reward formulas. The `terminal_bonus_weight` change is a
  *coefficient* on an existing raw term, not a new shaping term.
  Drawdown / hold-cost / other new terms land in Session 7 after a
  design pass, per the plan.
- No UI work. `ui_additions.md` Session 3 already lists the four
  new range editors and the `max >= min` validator widget; Session 8
  will consume them.
- `commission` was deliberately NOT promoted to a gene (per the
  session plan's "Do not" list).

**Next:** Session 4 — training plan / Gen-0 coverage tracker.

---

## Session 4 — Training plan / Gen-0 coverage tracker (2026-04-06)

**Shipped:**
- New module `training/training_plan.py` containing the full Session 4
  surface area:
  - `TrainingPlan` dataclass (plan_id, name, created_at, population_size,
    architectures, hp_ranges, seed, arch_mix, min_arch_samples, notes,
    outcomes) with `to_dict` / `from_dict` / `new` helpers.
  - `GenerationOutcome` dataclass — appended to a plan after each
    generation completes (best/mean fitness, alive/died architectures,
    n_agents, recorded_at, notes).
  - `PlanRegistry` — JSON-on-disk store at
    `registry/training_plans/{plan_id}.json`. Save / load / list /
    delete / `record_outcome`. `_path_for` blocks `..`/slashes so a
    malicious POST can't escape `self.root`.
  - `validate_plan` returning a list of `ValidationIssue`s; the
    pre-flight check refuses pop_size < `min_arch_samples × n_arch`
    (default 5 per arch). `is_launchable(issues)` is the bool helper.
  - `compute_coverage` — per-arch counts + per-gene decile bucket
    counts. Numeric genes only; `float_log` is bucketed in log space.
    Out-of-range historical samples are clamped, not dropped (so an
    old agent surviving a range tightening still counts).
  - `bias_sampler` — returns `BiasedSpec` wrappers attaching
    bucket-weight nudges to poorly-covered numeric genes (empty
    buckets get `1.5×` weight, populated buckets `1.0×`).
  - `sample_with_bias` — opt-in helper that picks a bucket by weight
    then uniform-samples within it. Mirrors `sample_hyperparams`'
    branching for unbiased specs.
  - `historical_agents_from_model_store` — adapter that reads every
    model record (active *and* discarded — coverage is about "did we
    sample this corner", not "did we like the result") and projects to
    `HistoricalAgent`.
- `agents/population_manager.py::initialise_population` now accepts an
  optional `plan: TrainingPlan | None = None`. When supplied the plan
  overrides `population_size`, parses its own `hp_ranges` to specs
  (when non-empty), restricts the architecture choice list, and -- if
  `arch_mix` is set -- builds a deterministic per-slot architecture
  assignment by shuffling the expanded mix. Default `plan=None` keeps
  the legacy `config.yaml`-driven path bit-identical so
  `start_training.sh` / `start_training.bat` keep working unchanged
  (Session 4 invariant).
- `training/run_training.py::TrainingOrchestrator` accepts new optional
  `training_plan` and `plan_registry` constructor kwargs, threads the
  plan into `initialise_population`, tracks the architectures present
  on Gen 0, and at the end of every `_run_generation` calls a new
  `_record_plan_outcome` helper that appends a `GenerationOutcome` to
  the plan. The helper is a silent no-op when either dependency is
  None, so config-only launches are unaffected.
- New router `api/routers/training_plans.py` with the four endpoints
  the session plan calls for:
  - `GET /api/training-plans` — list all plans (sorted desc).
  - `GET /api/training-plans/coverage` — coverage stats + biased gene
    list. Reads history from `app.state.store` (production) or
    `app.state.coverage_history` (test seam).
  - `GET /api/training-plans/{id}` — full plan + validation issues.
  - `POST /api/training-plans` — create + validate (does NOT launch).
    Returns 422 with the issue list if any error-severity issues are
    present, and the plan is *not* persisted in that case.
- `api/main.py` lifespan now constructs a `PlanRegistry` rooted at
  `config.paths.training_plans` (defaulting to
  `<registry_db_dir>/training_plans/`) and stashes it on
  `app.state.plan_registry`. The new router is mounted alongside the
  existing routers.
- New test file `tests/arch_exploration/test_training_plan.py` with
  10 CPU-only tests (covers all 7 items from the session plan):
  1. Plan round-trip via `PlanRegistry.save`/`load`/`list`.
  2. Validate rejects pop=6 with 3 architectures + min=5.
  2b. Validate accepts pop=15 (just-right floor case).
  3. Coverage with empty history flags every numeric gene as
     poorly-covered and both architectures as under-covered.
  4. Coverage with synthetic 50-agent history matches hand-counted
     bucket assignments and arch counts.
  5. `bias_sampler` nudges empty `gamma` buckets — populated bucket
     weight is strictly less than the empty buckets, every empty
     bucket is at least as heavy as the populated one, and at least
     one empty bucket is strictly heavier.
  6. Outcome round-trip via `record_outcome` → `load`.
  7. API: POST happy path → list → get → coverage all 200; POST with
     undersized plan → 422 with `population_too_small` in issues; GET
     unknown id → 404.
- Smoke-checked `api/main.create_app()` builds with all 4 new routes.

**Files changed:**
- `training/training_plan.py` — new (~500 lines).
- `agents/population_manager.py` — `initialise_population` accepts
  `plan`; new `TYPE_CHECKING` import for `TrainingPlan`.
- `training/run_training.py` — `TrainingOrchestrator.__init__` gains
  `training_plan` / `plan_registry`; `initialise_population` call
  threads `plan=`; new `_record_plan_outcome` helper called at the
  end of `_run_generation`; `datetime`/`timezone` imports added.
- `api/routers/training_plans.py` — new (~150 lines).
- `api/main.py` — imports the new router + `PlanRegistry`, builds the
  registry in `lifespan`, stashes it on `app.state.plan_registry`,
  mounts the router.
- `tests/arch_exploration/test_training_plan.py` — new 10-test file.
- `plans/arch-exploration/progress.md`, `lessons_learnt.md`,
  `ui_additions.md` — session notes.

**Tests:**
- `pytest tests/arch_exploration/test_training_plan.py` → 10 passed
  (~3.6 s).
- `pytest tests/arch_exploration/ tests/test_population_manager.py
  tests/test_orchestrator.py tests/test_genetic_operators.py
  tests/test_genetic_selection.py tests/test_config.py` → 188 passed,
  1 skipped, 1 pre-existing failure (`test_obs_dim_matches_env`,
  documented as stale on plain `master` in Session 1 progress).
- `tests/test_api_training.py` had 2 pre-existing
  `worker_disconnected` failures — confirmed via `git stash` they
  exist on plain `master`, unrelated to this session.

**Not shipped:**
- No UI work — Session 8 consumes `ui_additions.md`. The new backend
  fields are appended there.
- No actual Gen-0 training run under the planner — that's Session 9.
- `ppo_transformer_v1` is not yet in the architecture registry, so
  the planner only knows about `ppo_lstm_v1` and `ppo_time_lstm_v1`
  in practice. Session 6 will add the third arch as a schema
  extension; the planner already accepts it as a string, no rewrite
  needed.
- Bias sampler is intentionally simple: empty-bucket × 1.5 weight,
  populated × 1.0. Latin hypercube / Bayesian bandit deferred per
  the session plan; documented in `lessons_learnt.md`.

**Next:** Session 5 — LSTM structural knobs.

---

## Session 5 — LSTM structural knobs (2026-04-07)

**Shipped:**
- Promoted three new structural genes to `config.yaml`
  `hyperparameters.search_ranges`:
  - `lstm_num_layers`: `int_choice` {1, 2}
  - `lstm_dropout`: `float` [0.0, 0.3]
  - `lstm_layer_norm`: `int_choice` {0, 1} (0/1 rather than
    a dedicated `bool_choice` type — the existing sampler has no
    bool branch, and bool-as-int round-trips cleanly through JSON
    checkpoints and the `bool(...)` cast in the policy ctors).
- `PPOLSTMPolicy.__init__` now reads all three keys via
  `hyperparams.get(...)` with pre-Session-5 defaults
  (`num_layers=1`, `dropout=0.0`, `layer_norm=False`). Dropout is
  only forwarded to `nn.LSTM(...)` when `num_layers > 1` — PyTorch
  emits a warning and ignores the value otherwise. Documented inline.
- `PPOLSTMPolicy` layer-norm location: applied to the LSTM *output*
  (post-recurrence, pre-head). One fixed location, recorded in a
  comment. `nn.Identity` when disabled so the forward path is
  branch-free.
- `PPOLSTMPolicy.init_hidden` shape changed from `(1, batch, hidden)`
  to `(num_layers, batch, hidden)` to match `nn.LSTM`'s stacked
  hidden-state layout. `PPOTrainer` passes the hidden state opaquely
  through the policy, so no trainer change was needed.
- `PPOTimeLSTMPolicy` converts the single `time_lstm_cell` attribute
  into `time_lstm_cells: nn.ModuleList[TimeLSTMCell]`. Layer 0 takes
  the fused market+runner input (`lstm_input_dim`), subsequent layers
  take the previous layer's hidden state as input (`lstm_hidden`).
  `forward` keeps per-layer `h_layers` / `c_layers` as Python lists
  so per-timestep assignment doesn't fight autograd, then stacks them
  back to `(num_layers, batch, hidden)` for the outgoing hidden state.
- Inter-layer dropout in the stacked Time-LSTM: `F.dropout` is
  applied to each layer's hidden output *before it feeds the next
  layer*, only when `lstm_dropout > 0` and there IS a next layer.
  `training` is read from `self.training` so `.eval()` disables it
  automatically. This matches `nn.LSTM`'s "dropout between layers
  only" semantics.
- `PPOTimeLSTMPolicy` layer norm: same location as the stock-LSTM
  variant — applied to the top-layer hidden state (`lstm_out`) after
  the per-timestep loop completes. `nn.Identity` when disabled.
- `PPOTimeLSTMPolicy.init_hidden` now returns
  `(num_layers, batch, hidden)` to match the list-of-layers forward.
- Backward-compat: both policy constructors default every Session 5
  gene to its pre-Session-5 value, so a checkpoint that was saved
  without these keys instantiates into an identical single-layer,
  no-dropout, no-layer-norm policy. Covered by
  `test_policy_defaults_without_new_keys`.
- `tests/test_config.py` expected-params list gets the three new
  keys.
- New test file `tests/arch_exploration/test_lstm_structural.py`
  with 22 CPU-only cases covering all 5 items on the session plan:
  1. Gene sampling + range checks + "both ends seen" for choice
     genes (200 seeded iterations).
  2. Policy instantiation grid — 16 combos × 2 architectures = 32
     instantiation + forward-pass cases (wired as 16 parametrized
     cases per arch). Each asserts output shapes, structural
     attributes, and hidden-state shape.
  3. `init_hidden()` shape matches `num_layers=2` on both arches.
  4. Stacked `TimeLSTMCell` unit test: two-timestep sequence,
     ModuleList length check, eval-mode determinism, train-mode
     divergence under heavy dropout.
  5. Backward-compat: both arches default cleanly when none of
     the Session-5 keys are supplied.
- Added `torch.nn.functional as F` import (needed for the stacked
  Time-LSTM's inter-layer dropout).

**Files changed:**
- `config.yaml` — three new `search_ranges` entries under
  `hyperparameters`.
- `agents/policy_network.py` — `F` import, new structural-gene
  reads in both policy ctors, `nn.LSTM` stacking + dropout + layer
  norm in `PPOLSTMPolicy`, `ModuleList` of `TimeLSTMCell` +
  per-layer loop + inter-layer `F.dropout` + layer norm in
  `PPOTimeLSTMPolicy`, updated `init_hidden` shapes on both.
- `tests/test_config.py` — expected-params list updated.
- `tests/arch_exploration/test_lstm_structural.py` — new 22-test
  file.
- `plans/arch-exploration/progress.md`, `lessons_learnt.md`,
  `ui_additions.md` — session notes (ui_additions Session 5
  checklist reads as server-side complete; Session 8 will consume
  the UI items).

**Tests:**
- `pytest tests/arch_exploration/test_lstm_structural.py` → 22
  passed (~3.1 s).
- `pytest tests/arch_exploration/ tests/test_config.py
  tests/test_population_manager.py tests/test_genetic_operators.py`
  → 142 passed, 1 pre-existing failure
  (`test_population_manager.py::test_obs_dim_matches_env`, stale
  `1630` vs actual `1636`; confirmed unrelated in Session 1 notes).

**Not shipped:**
- No transformer work. That's Session 6.
- No tuning of default values — the new genes start from the legacy
  `num_layers=1, dropout=0, layer_norm=False` defaults so any
  checkpoint or call-site that doesn't know about them loads
  bit-identically.
- No UI work. `ui_additions.md` Session 5 entries are server-side
  complete but the widget work itself lands in Session 8.

**Next:** Session 6 — `ppo_transformer_v1` architecture.

---

## Session 6 — `ppo_transformer_v1` architecture (2026-04-07)

**Shipped:**
- New `PPOTransformerPolicy` in `agents/policy_network.py` registered
  via the existing `@register_architecture` decorator as
  `ppo_transformer_v1`. Shares the market encoder, per-runner shared
  MLP encoder (mean + max pool), actor head, and critic head with the
  two LSTM variants — only the sequence model changes. The transformer
  block is a stock `nn.TransformerEncoder` with `batch_first=True`,
  `norm_first=True`, GELU activation, and
  `enable_nested_tensor=False` (the latter silences a spurious user
  warning that fires under `norm_first=True`).
- Three new genes added to `config.yaml` `hyperparameters.search_ranges`:
  - `transformer_heads`: `int_choice` {2, 4, 8}
  - `transformer_depth`: `int_choice` {1, 2, 3}
  - `transformer_ctx_ticks`: `int_choice` {32, 64, 128}
- `ppo_transformer_v1` appended to the `architecture_name`
  `str_choice` so the existing sampler / mutator can discover and
  evolve transformer agents without any spec change.
- Rolling tick-context buffer repurposes the `BasePolicy` hidden-state
  slot as a 2-tuple `(buffer, valid_count)`:
  - `buffer`: `(batch, ctx_ticks, d_model)` float tensor of fused
    embeddings. Unfilled slots stay zero and the transformer learns
    to ignore the warmup window.
  - `valid_count`: `(batch,)` long tensor, clamped at `ctx_ticks`.
  - `init_hidden` returns both elements as tensors so `PPOTrainer`'s
    existing `h[0].to(device), h[1].to(device)` idiom still works
    unchanged — no trainer edit needed. `BasePolicy`'s docstring now
    documents the "opaque 2-tuple of tensors" contract explicitly.
- `forward()` handles both 2-D `(batch, obs_dim)` (single-step
  rollout) and 3-D `(batch, seq_len, obs_dim)` (sequence / test)
  inputs. For each tick in the provided sequence it shift-left-
  appends the fused embedding onto the rolling buffer. The
  transformer is then run once on the final buffer with learned
  positional embeddings and an upper-triangular causal mask
  (registered as a non-persistent buffer so `.to(device)` moves it
  with the module). The critic reads `encoded[:, -1, :]`; the actor
  concatenates `encoded[:, -1, :]` with each runner's current
  embedding, exactly mirroring the LSTM architectures.
- `d_model` is aliased to `lstm_hidden_size` so the existing
  hyperparameter spec still drives the transformer's width. All
  shipped `lstm_hidden_size` choices (64, 128, 256, 512, 1024, 2048)
  divide evenly by every allowed head count (2, 4, 8); a defensive
  guard in `__init__` still raises on hand-crafted combos that
  violate divisibility.
- New `encode_sequence(obs)` helper returns the per-position encoder
  output so the causal-masking test can inspect non-final positions
  directly. Used only by tests and diagnostics.
- **Arch cooldown.** `PopulationManager.mutate` now reads
  `hp["arch_change_cooldown"]` (metadata-only, not a sampled gene,
  defaults to 0). When the counter is positive, the
  `architecture_name` mutation branch is skipped for that generation
  and the counter is decremented toward zero. When a mutate() call
  actually changes the architecture, the counter is re-armed to 1 so
  the next generation cannot flip again. The field lives on the hp
  dict so it travels with the agent through `crossover` and the
  registry without a new schema field.
- **Arch-specific learning-rate override.** `TrainingPlan` gains a new
  optional `arch_lr_ranges: dict[str, dict] | None` field that maps
  architecture name → `HyperparamSpec`-shaped dict. When the planner
  drives Gen-0 via `initialise_population(plan=...)`, agents of the
  overridden architecture re-sample their `learning_rate` from the
  per-arch range immediately after the normal sample. Scope is
  deliberately narrow to `learning_rate` (that is the gene the plan
  explicitly calls out); widening to a full per-arch hp range table
  is additive and can land in a later session.
- New test file `tests/arch_exploration/test_transformer_arch.py`
  with 14 CPU-only tests covering every item from the session plan:
  1. Registry: `ppo_transformer_v1` is in `REGISTRY`, has the right
     class, non-empty description, and `create_policy` returns a
     `PPOTransformerPolicy` instance.
  2. Sampler: 300 seeded samples from the real `config.yaml` specs
     exercise every value of all three new genes and stay in range.
  3. Instantiation grid: parametrised
     `{heads: [2, 4]} × {depth: [1, 2]} × {ctx_ticks: [32, 64]}` —
     8 combinations; each does two sequential `forward()` calls to
     exercise the rolling buffer, and asserts structural attributes,
     transformer layer count, output shapes, and hidden-state
     shapes / valid_count increments.
  4. Causal masking: two 4-tick sequences identical at ticks 0..2
     and differing only at tick 3. `encode_sequence` is called on
     both; output at buffer position `-2` (= tick 2) must be
     bit-for-bit identical, position `-1` (= tick 3) must differ.
     This catches any regression that drops the upper-triangular
     mask from the encoder.
  5. Rolling-buffer overflow: feed `ctx_ticks + 5 = 37` known
     fingerprint obs; assert no crash, `valid_count` clamps at
     `ctx_ticks`, the first 5 fingerprints have rolled off (verified
     by confirming none of the 32 buffer slots matches
     `_encode_ticks(obs_list[0])`), and each of the remaining 32
     slots matches the freshly-encoded fingerprint of its expected
     original obs.
  6. Arch cooldown: construct a minimal `PopulationManager` with
     three architectures, seed `architecture_name = "ppo_time_lstm_v1"`
     (middle of the choice list so every ±1 mutation is a real
     flip), `arch_change_cooldown = 1`, `mutation_rate = 1.0`.
     Assert the first `mutate` does NOT change arch and decrements
     cooldown to 0; the second `mutate` DOES change arch and re-arms
     cooldown to 1; the third `mutate` is blocked again. Full round
     trip through the cooldown state machine.
  7. Planner arch-specific LR: `TrainingPlan.new(...,
     arch_lr_ranges={"ppo_transformer_v1": {...}})` with
     `global_lr.max < override_lr.min`, 10 LSTM + 10 transformer
     slots via `arch_mix`. Build Gen-0 and assert every LSTM agent's
     `learning_rate` is in the global range, every transformer
     agent's `learning_rate` is in the override range, and
     `max(lstm_lrs) < min(xf_lrs)` (no cross-contamination).

**Files changed:**
- `agents/policy_network.py` — added `math` / `Any` imports,
  clarified the `BasePolicy` hidden-state protocol docstring,
  appended `PPOTransformerPolicy` (class + forward + rolling-buffer
  helper + `encode_sequence` + `init_hidden` + `get_action_distribution`).
- `agents/population_manager.py` — arch cooldown state machine in
  `mutate`; `initialise_population` applies `plan.arch_lr_ranges`
  after resolving the per-slot architecture.
- `training/training_plan.py` — new `arch_lr_ranges` field on
  `TrainingPlan`, threaded through `to_dict`/`from_dict`/`new`.
- `config.yaml` — three new `search_ranges` entries, appended
  `ppo_transformer_v1` to the `architecture_name` choice list.
- `tests/test_config.py` — expected-params list gains the three new
  transformer genes.
- `tests/arch_exploration/test_transformer_arch.py` — new 14-test
  file.
- `plans/arch-exploration/progress.md`, `lessons_learnt.md`,
  `ui_additions.md` — session notes.

**Tests:**
- `pytest tests/arch_exploration/test_transformer_arch.py` →
  14 passed (~3.1 s).
- `pytest tests/arch_exploration/ tests/test_config.py` →
  77 passed (~5.7 s).
- `pytest tests/test_population_manager.py tests/test_genetic_operators.py
  tests/test_genetic_selection.py tests/test_orchestrator.py` →
  147 passed, 1 skipped, 1 pre-existing failure
  (`test_obs_dim_matches_env` stale 1630 vs actual 1636, documented
  in Session 1 progress / lessons as unrelated drive-by).
- `pytest tests/test_policy_network.py tests/test_worker_run_overrides.py
  tests/test_session_2_8.py` → 123 passed, 1 pre-existing failure
  (`test_gradients_flow` references the now-renamed `time_lstm_cell`
  attribute, which Session 5 turned into `time_lstm_cells`;
  confirmed failing on plain `master` via `git stash`).

**Not shipped:**
- No GPU tests, no training loops — Session 9 consumes those.
- No hierarchical runner-attention architecture (mentioned in the
  design review as "a different architecture" — explicitly out of
  scope per the session plan).
- No LR-warmup, weight-decay, or optimiser changes. Adam / AdamW
  single-LR stays.
- No UI work — `ui_additions.md` Session 6 entries remain the
  pending list for Session 8.
- The per-arch override in `TrainingPlan` is narrow to
  `learning_rate` only. A full per-arch hp-range table is additive
  and can land when a concrete need shows up.
- `observation_window_ticks` was retired cleanly in Session 1; the
  new `transformer_ctx_ticks` is a fresh gene with its own name, no
  silent aliasing or migration shim.

**Next:** Session 7 — drawdown-aware shaping (DESIGN PASS FIRST).

---

## Session 7 — Drawdown-aware shaping (2026-04-07)

**Design pass first.** Committed separately as `design pass for
Session 7` before any implementation code was touched — the
commit ordering is what the session plan's "Do not skip the
design pass" rule is asking for.

**Chosen formulation (Option D — reflection-symmetric range
position).** None of the A/B/C options in the session plan gave a
closed-form zero-mean term without either a hindsight bootstrap
(Option A), a shadow policy doubling episode cost (Option B), or
duplicating `efficiency_penalty` (Option C). Option D, added in
the design pass:

```
peak_t   = max(peak_{t-1},  day_pnl_t)
trough_t = min(trough_{t-1}, day_pnl_t)
shaped  += weight × (2·day_pnl_t − peak_t − trough_t) / starting_budget
```

with `peak_0 = trough_0 = 0`. Under reflection `X → −X`,
`peak ↔ −trough` so the per-race term flips sign exactly and
paths + reflections cancel pairwise. See the design pass for the
worked examples and the full proof.

**Shipped:**
- `env/betfair_env.py` — added `drawdown_shaping_weight` to
  `_REWARD_OVERRIDE_KEYS`, plumbed via the Session-1 override
  path. `__init__` reads it into `self._drawdown_shaping_weight`
  (default `0.0`). `reset()` initialises `_day_pnl_peak` and
  `_day_pnl_trough` to `0.0` — both the initial values are load-
  bearing for the reflection proof, so they have a comment that
  says so. New helper `_update_drawdown_shaping` contains the
  whole formula (running-max/min + normalised range-position)
  and is called from `_settle_current_race` after
  `self._day_pnl += race_pnl`. Zero-weight is a clean no-op
  early-exit so existing runs are byte-identical. The returned
  term is added to `shaped`, which accumulates into
  `self._cum_shaped_reward` — NOT `_cum_raw_reward`. Factoring
  the formula into its own helper was purely for testability:
  the unit tests drive it directly without having to run a full
  bet-matching pipeline to produce race_pnl trajectories.
- `agents/ppo_trainer.py` — one new row in `_REWARD_GENE_MAP`:
  `"reward_drawdown_shaping": ("drawdown_shaping_weight",)`. Name
  follows the existing `reward_efficiency_penalty` /
  `reward_precision_bonus` convention (genes use `reward_`
  prefix, env reward-config keys do not).
- `config.yaml` — `reward.drawdown_shaping_weight: 0.0` default in
  the reward block, plus a new `reward_drawdown_shaping` entry in
  `hyperparameters.search_ranges` (float, `[0.0, 0.2]`). Upper
  bound picked so the maximum accumulated episode contribution
  (`≈ weight × N_races`) is comparable in scale to the existing
  `precision_bonus` and `terminal_bonus_weight` maxima.
- `tests/arch_exploration/test_drawdown_shaping.py` — new 9-test
  file covering every item in the session plan and a couple of
  additions:
  1. Gene is present in `config.yaml`, spec is `[0.0, 0.2]`, and
     50 seeded samples all stay in range.
  2. `_reward_overrides_from_hp` maps `reward_drawdown_shaping`
     → `drawdown_shaping_weight` (not the raw gene name).
  3. **Zero-mean for random walks.** 1000 seeded symmetric walks
     (uniform `[-10, +10]`, 25 races each) driven through
     `_update_drawdown_shaping`; asserts the mean total
     contribution is within 2 standard errors of zero. Runs in
     well under a second — the sub-second budget holds because
     the tests bypass the full env pipeline and drive the helper
     directly.
  4. **Reflection pairs cancel exactly.** Stronger than (3):
     for 50 random walks, `forward + reflected` is asserted to
     be algebraically `0` (within `1e-9`). Guards against
     statistical false positives in test (3) and catches any
     regression that introduces an asymmetric bias.
  5. Drawdown-avoiding policy `[+2, +4, +6, +8]` → numerator
     `+20`, total `+0.010` with `weight=0.05, budget=100`.
     Replicates worked example 2 from the design pass exactly.
  6. Drawdown-amplifying policy `[-10, +5, -10, +5]` → numerator
     `-30`. Replicates worked example 3 exactly.
  7. **`raw + shaped ≈ total_reward` invariant** with a non-zero
     drawdown weight on a real 3-race `_make_day` episode — the
     CLAUDE.md invariant must still hold with the new term live.
  8. **Bucketing.** The term accumulates into
     `_cum_shaped_reward` (visible via `info["shaped_bonus"]`),
     never into `_cum_raw_reward` (`info["raw_pnl_reward"]`).
     Drives a deterministic non-zero drawdown trajectory and
     asserts raw stays pinned while shaped moves by exactly the
     helper's return value.
  9. Zero-weight early-exit is a clean no-op: returns `0.0` for
     every step and leaves `_day_pnl_peak` / `_day_pnl_trough`
     both at `0.0`. Guards against a future refactor that
     accidentally advances peak/trough before the early-exit
     check.
- `tests/test_config.py` — `expected_params` in the schema-
  presence test gains `reward_drawdown_shaping`.

**Files changed:**
- `env/betfair_env.py` — `_REWARD_OVERRIDE_KEYS`, `__init__`,
  `reset`, `_settle_current_race`, new helper
  `_update_drawdown_shaping`.
- `agents/ppo_trainer.py` — `_REWARD_GENE_MAP` row.
- `config.yaml` — reward default + search_ranges entry.
- `tests/arch_exploration/test_drawdown_shaping.py` — new.
- `tests/test_config.py` — expected_params list.
- `plans/arch-exploration/session_7_drawdown_shaping.md` — design
  pass (committed separately).
- `plans/arch-exploration/progress.md`, `lessons_learnt.md`,
  `ui_additions.md` — session notes.

**Tests:**
- `pytest tests/arch_exploration/test_drawdown_shaping.py tests/test_config.py`
  → 22 passed (~3.1 s).
- `pytest tests/arch_exploration/ tests/test_betfair_env.py tests/test_config.py`
  → 145 passed (~6.5 s). No regressions in the existing reward
  plumbing, reward schema, env, or config suites.

**Not shipped:**
- No per-tick drawdown shaping. Race-settlement schedule only —
  the zero-mean proof assumes discrete settlement and the rest
  of the shaping stack lives there.
- No `info["day_pnl_peak"]` / `info["day_pnl_trough"]` exposure.
  Nothing downstream needs it today; easy to add if a diagnostic
  test later asks for it.
- No hold-cost-per-open-liability-tick term. Explicitly out of
  scope per the session plan — same asymmetric pitfalls,
  deserves its own design pass.
- No UI work. `ui_additions.md` Session 7 entries land in
  Session 8 alongside the rest.

**Next:** Session 8 — UI wiring for new knobs.

