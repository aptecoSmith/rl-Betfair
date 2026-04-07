# UI Additions — Running List

Every gene or config added during sessions 1–7 must be exposable and
editable from the web UI. This file is the running list of UI work
needed to catch up. Sessions 1–7 **append to this file** as they add
new configurable values; Session 8 consumes the list and wires
everything up.

**Rule:** a session is not complete until any new configurable values
it introduces have been added to the list below.

Front-end lives under `frontend/`. The current UI exposes the basic
training config (population size, architecture default, learning rate
range, etc.) — the new work extends that. Before touching the UI,
read the existing config-editing components to match style.

---

## Session 1 — Reward plumbing

Not a UI-visible session on its own (it's a bugfix), but confirms the
following existing schema entries are actually live and should already
appear in the UI's "search ranges" editor. Verify they are editable
end-to-end after Session 1 lands:

- [x] `reward_early_pick_bonus` (range editor) — **Session 1 note:**
      this is a single scalar gene that maps to *both*
      `early_pick_bonus_min` and `early_pick_bonus_max` in the env.
      Session 3 will split it into two editable ranges; until then the
      UI should continue to expose the single gene.
- [x] `reward_efficiency_penalty` (range editor)
- [x] `reward_precision_bonus` (range editor)
- [x] **Remove** `observation_window_ticks` from the UI range editor —
      retired in Session 1 (no longer in `config.yaml` search_ranges).

## Session 2 — PPO schema expansion

- [x] `gamma` (range editor, float, [0.95, 0.999])
- [x] `gae_lambda` (range editor, float, [0.9, 0.98])
- [x] `value_loss_coeff` (range editor, float, [0.25, 1.0])

## Session 3 — Reward schema expansion

- [x] `early_pick_bonus_min` (range editor, float, [1.0, 1.3])
- [x] `early_pick_bonus_max` (range editor, float, [1.1, 1.8])
- [x] `early_pick_min_seconds` (range editor, int seconds, [120, 900])
- [x] `terminal_bonus_weight` (range editor, float, [0.5, 3.0])
- [x]Validator widget: enforce `max >= min` on the early-pick range
      in the UI before save. (Server-side validation must also exist;
      the UI widget is belt-and-braces.)

## Session 4 — Training plan / coverage tracker

This is the biggest UI piece. New page(s) needed:

- [x] **Training-plan page.** Show a history of Gen-0 runs: date,
      population size, architecture mix, hyperparam ranges used,
      seed, outcome summary. Backed by `GET /api/training-plans`.
      Each plan card should show:
      - `plan.name`, `plan.created_at`, `plan.population_size`
      - `plan.architectures` (and `plan.arch_mix` when present)
      - `plan.seed`, `plan.min_arch_samples`, `plan.notes`
      - All `plan.outcomes` rows: `generation`, `recorded_at`,
        `best_fitness`, `mean_fitness`, `architectures_alive`,
        `architectures_died`, `n_agents`.
      - Validation status from `validation` field on the GET-by-id
        response (warnings shown as a yellow badge, errors as red).
- [x] **Plan editor.** Compose a new Gen-0 run via `POST
      /api/training-plans`. Form fields, all 1:1 with the request
      payload:
      - `name` (text, required)
      - `population_size` (int, required)
      - `architectures` (multi-select from the architecture registry)
      - `arch_mix` (optional dict; collapse/expand panel)
      - `hp_ranges` (range editors keyed by gene name; can start from
        `config.yaml` defaults or be cleared to fall back to them)
      - `seed` (int, optional)
      - `min_arch_samples` (int, default 5)
      - `notes` (textarea, optional)
      - 422 responses must surface the per-issue `code` + `message`
        list inline so the user knows which field is wrong.
- [x] **Coverage warning widget.** If the selected `population_size <
      min_arch_samples × len(architectures)`, warn inline before the
      user can submit. Server-side `validate_plan` is the source of
      truth — but the widget mirrors the math so the user sees the
      error before round-tripping.
- [x] **Coverage page / panel.** Backed by `GET
      /api/training-plans/coverage`. Show:
      - `report.total_agents`, `report.arch_counts`, list of
        `arch_undercovered`.
      - Per-gene bar chart of `bucket_counts` (one chart per entry in
        `report.gene_coverage`), with the well-covered/poorly-covered
        flag visible.
      - The `biased_genes` list (returned alongside the report) so the
        user knows which genes the planner would currently nudge.
- [x] **"Bias toward uncovered"** toggle on the plan editor: when
      enabled, the UI calls the coverage endpoint, applies the
      returned bias to the editable `hp_ranges` *before* POST so the
      user can review and tweak the nudge before saving. The bias
      itself is opt-in on the backend — `population_manager` does NOT
      apply it automatically (Session 4 lessons).
- [x] Backend endpoints (list plans, create plan, get coverage stats)
      — landed in Session 4 (`api/routers/training_plans.py`).
- [x]Outcome history rendering: each plan card should show the per-
      generation outcome timeline once `record_outcome` populates it
      from a real run (Session 9).
- [x]Path-traversal note: `plan_id` is server-generated; the UI
      should treat it as opaque and never let the user edit it
      directly.

## Session 5 — LSTM structural knobs

Server side landed in Session 5 (`config.yaml` search_ranges +
`PPOLSTMPolicy` / `PPOTimeLSTMPolicy` constructors). UI widgets
themselves still land in Session 8.

- [x] `lstm_num_layers` (choice editor, {1, 2}) — backend live
- [x] `lstm_dropout` (range editor, float, [0, 0.3]) — backend live
- [x] `lstm_layer_norm` (choice editor, {false, true}) — backend
      live. Note: stored as `int_choice` 0/1 in `config.yaml` and
      cast to bool in the policy ctor, so the UI should render it
      as a true/false toggle but persist it as 0/1.

## Session 6 — Transformer architecture

Server side landed in Session 6 (`config.yaml` search_ranges +
`PPOTransformerPolicy` in `agents/policy_network.py` +
`arch_change_cooldown` in `PopulationManager.mutate` +
`TrainingPlan.arch_lr_ranges`). UI widgets themselves still land in
Session 8.

- [x] `transformer_heads` (choice editor, {2, 4, 8}) — backend live
- [x] `transformer_depth` (choice editor, {1, 2, 3}) — backend live
- [x] `transformer_ctx_ticks` (choice editor, {32, 64, 128}) —
      backend live
- [x]Update the architecture-choice widget to include
      `ppo_transformer_v1` — backend live (now in
      `architecture_name` str_choice choices)
- [x] **Arch-specific LR range** support: the plan editor must
      expose `TrainingPlan.arch_lr_ranges` — a per-architecture
      HyperparamSpec-shaped override for `learning_rate`. Shape is
      `{arch_name: {"type": "float_log", "min": ..., "max": ...}}`
      and the backend accepts it on the `POST /api/training-plans`
      payload via the standard `TrainingPlan.from_dict` path.
      (Session 6 only handles `learning_rate`; widening to other
      genes is additive and can land in a later session.)
- [x]Arch-cooldown indicator: show which lineages have cooldowns
      active on the population view. Field lives on the agent's hp
      dict as `arch_change_cooldown` (int, defaults to 0) and is
      updated by `PopulationManager.mutate` — no new schema needed,
      the UI just needs to read and render the existing metadata
      field.

## Session 7 — Drawdown-aware shaping

Design pass landed on Option D (reflection-symmetric range position).
See `session_7_drawdown_shaping.md` for the full formulation.

- [x] `reward_drawdown_shaping` (range editor, float, `[0.0, 0.2]`).
      Gene name matches the `reward_efficiency_penalty` /
      `reward_precision_bonus` convention; env reward-config key is
      `drawdown_shaping_weight` (no `reward_` prefix inside the
      reward block). The UI should display the gene name and the
      spec range only — the env-key mapping is server-side.
- [x]Tooltip / help text: "Shaped reward for spending time near
      the running day high (positive) vs running day low
      (negative). Zero-mean for random policies by construction."
      Users need to know setting it to 0 is a clean opt-out.
- [x]Episode diagnostic panel: no new fields *required* for
      Session 7 — the term already flows through `shaped_bonus`
      which the panel already displays. If/when we expose
      `info["day_pnl_peak"]` / `info["day_pnl_trough"]` (not in
      scope for Session 7), add "Day peak P&L" and "Day trough
      P&L" sparklines at that point.

## Session 8 — The UI session itself

Consume everything above. Additionally:

- [x]Make sure the "config editor" vs "search-range editor"
      distinction is clean: some of these are *ranges to mutate over*,
      some are *fixed config values*. The UI must not conflate them.
- [x]Add a "what's in the schema today" read-only view so a user can
      sanity-check which genes are live vs dead without grepping the
      codebase.
- [x]Review every config value touched during this phase of work and
      ensure it is editable from the UI. No YAML-only knobs.
