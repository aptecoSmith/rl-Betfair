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

- [ ] `reward_early_pick_bonus` (range editor) — **Session 1 note:**
      this is a single scalar gene that maps to *both*
      `early_pick_bonus_min` and `early_pick_bonus_max` in the env.
      Session 3 will split it into two editable ranges; until then the
      UI should continue to expose the single gene.
- [ ] `reward_efficiency_penalty` (range editor)
- [ ] `reward_precision_bonus` (range editor)
- [ ] **Remove** `observation_window_ticks` from the UI range editor —
      retired in Session 1 (no longer in `config.yaml` search_ranges).

## Session 2 — PPO schema expansion

- [ ] `gamma` (range editor, float, [0.95, 0.999])
- [ ] `gae_lambda` (range editor, float, [0.9, 0.98])
- [ ] `value_loss_coeff` (range editor, float, [0.25, 1.0])

## Session 3 — Reward schema expansion

- [ ] `early_pick_bonus_min` (range editor, float, [1.0, 1.3])
- [ ] `early_pick_bonus_max` (range editor, float, [1.1, 1.8])
- [ ] `early_pick_min_seconds` (range editor, int seconds, [120, 900])
- [ ] `terminal_bonus_weight` (range editor, float, [0.5, 3.0])
- [ ] Validator widget: enforce `max >= min` on the early-pick range
      in the UI before save. (Server-side validation must also exist;
      the UI widget is belt-and-braces.)

## Session 4 — Training plan / coverage tracker

This is the biggest UI piece. New page(s) needed:

- [ ] **Training-plan page.** Show a history of Gen-0 runs: date,
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
- [ ] **Plan editor.** Compose a new Gen-0 run via `POST
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
- [ ] **Coverage warning widget.** If the selected `population_size <
      min_arch_samples × len(architectures)`, warn inline before the
      user can submit. Server-side `validate_plan` is the source of
      truth — but the widget mirrors the math so the user sees the
      error before round-tripping.
- [ ] **Coverage page / panel.** Backed by `GET
      /api/training-plans/coverage`. Show:
      - `report.total_agents`, `report.arch_counts`, list of
        `arch_undercovered`.
      - Per-gene bar chart of `bucket_counts` (one chart per entry in
        `report.gene_coverage`), with the well-covered/poorly-covered
        flag visible.
      - The `biased_genes` list (returned alongside the report) so the
        user knows which genes the planner would currently nudge.
- [ ] **"Bias toward uncovered"** toggle on the plan editor: when
      enabled, the UI calls the coverage endpoint, applies the
      returned bias to the editable `hp_ranges` *before* POST so the
      user can review and tweak the nudge before saving. The bias
      itself is opt-in on the backend — `population_manager` does NOT
      apply it automatically (Session 4 lessons).
- [x] Backend endpoints (list plans, create plan, get coverage stats)
      — landed in Session 4 (`api/routers/training_plans.py`).
- [ ] Outcome history rendering: each plan card should show the per-
      generation outcome timeline once `record_outcome` populates it
      from a real run (Session 9).
- [ ] Path-traversal note: `plan_id` is server-generated; the UI
      should treat it as opaque and never let the user edit it
      directly.

## Session 5 — LSTM structural knobs

- [ ] `lstm_num_layers` (choice editor, {1, 2})
- [ ] `lstm_dropout` (range editor, float, [0, 0.3])
- [ ] `lstm_layer_norm` (choice editor, {false, true})

## Session 6 — Transformer architecture

- [ ] `transformer_heads` (choice editor, {2, 4, 8})
- [ ] `transformer_depth` (choice editor, {1, 2, 3})
- [ ] `transformer_ctx_ticks` (choice editor, {32, 64, 128})
- [ ] Update the architecture-choice widget to include
      `ppo_transformer_v1`.
- [ ] **Arch-specific LR range** support: the range editor for
      `learning_rate` may need to accept per-architecture overrides
      so transformers can use a different distribution than LSTMs.
      (Confirm schema shape with Session 4's planner.)
- [ ] Arch-cooldown indicator: show which lineages have cooldowns
      active on the population view.

## Session 7 — Drawdown-aware shaping

(If design pass approves the feature.)

- [ ] `drawdown_penalty` (range editor, float, range TBD in design
      pass)
- [ ] Any new tracked stats (peak_day_pnl, running drawdown) need to
      appear on the episode/diagnostic panels.

## Session 8 — The UI session itself

Consume everything above. Additionally:

- [ ] Make sure the "config editor" vs "search-range editor"
      distinction is clean: some of these are *ranges to mutate over*,
      some are *fixed config values*. The UI must not conflate them.
- [ ] Add a "what's in the schema today" read-only view so a user can
      sanity-check which genes are live vs dead without grepping the
      codebase.
- [ ] Review every config value touched during this phase of work and
      ensure it is editable from the UI. No YAML-only knobs.
