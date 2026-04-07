# Progress — Next Steps

Update this file at the end of each session. One short entry per
session: what shipped, what files changed, what tests were added,
what didn't ship and why.

Factual only. Running thoughts go in `lessons_learnt.md`.

---

## Session 0a — Backlog creation (2026-04-07)

**Shipped:**
- Created `plans/next_steps/` folder.
- Captured every deferred / skipped item from the
  `arch-exploration` phase into `next_steps.md` with status tags.
- Created the standard scaffolding files to match the
  `arch-exploration/` layout.

**Files added:**
- `plans/next_steps/next_steps.md` (the backlog itself)
- `plans/next_steps/purpose.md`
- `plans/next_steps/progress.md` (this file)
- `plans/next_steps/lessons_learnt.md`
- `plans/next_steps/testing.md` *(superseded in Session 0b)*
- `plans/next_steps/ui_additions.md`

## Session 0b — Adopt jarvis plan-folder layout (2026-04-07)

**Shipped:**
- Restructured `plans/next_steps/` to match the `jarvis/plan/`
  scaffold used in a sibling repo. Rationale and tradeoffs
  recorded in `design_decisions.md`.
- Moved hard constraints out of `purpose.md` into a dedicated
  `hard_constraints.md`. Rewrote `purpose.md` to document the
  new folder layout.
- Added `master_todo.md` as the ordered, tickable execution list
  (distinct from `next_steps.md` which stays as the raw backlog).
- Added `design_decisions.md` for load-bearing decisions with
  rationale and revisit triggers.
- Split the single `testing.md` into three contract-specific
  docs: `initial_testing.md` (fast CPU feedback loop),
  `integration_testing.md` (opt-in GPU / slow tests),
  `manual_testing_plan.md` (human-in-the-loop verification).
- Created `sessions/` subfolder with a `README.md` documenting
  the naming convention and prompt structure. Future session
  prompts land there; numbering continues from arch-exploration,
  so the first will be `session_10_*.md`.
- Deleted the superseded `testing.md`.

**Files added:**
- `plans/next_steps/hard_constraints.md`
- `plans/next_steps/master_todo.md`
- `plans/next_steps/design_decisions.md`
- `plans/next_steps/initial_testing.md`
- `plans/next_steps/integration_testing.md`
- `plans/next_steps/manual_testing_plan.md`
- `plans/next_steps/sessions/README.md`

**Files modified:**
- `plans/next_steps/purpose.md` — hard constraints moved out,
  folder layout section added.
- `plans/next_steps/progress.md` — this entry.

**Files removed:**
- `plans/next_steps/testing.md` — content split across the three
  new testing docs.

**Not shipped:**
- No code changes.
- No session prompts yet. Session 10 (housekeeping) and
  Session 11 (real run) are named in `master_todo.md` but their
  prompt files are not written.

## Session 0c — Draft all Phase 1-3 session prompts (2026-04-07)

**Shipped:**
- Wrote eight session prompts under `sessions/`:
  - `session_10_housekeeping.md` — debt clearance (obs_dim, retired
    names, schema inspector audit, TODO triage)
  - `session_11_real_run.md` — first real multi-generation run,
    gated behind Session 10, with reprioritisation pass at the end
  - `session_12_hold_cost.md` — hold-cost reward, design pass first
    following the Option D template from Session 7
  - `session_13_ppo_risky_knobs.md` — `mini_batch_size` and
    `ppo_epochs` as discrete-choice genes
  - `session_14_arch_specific_ranges.md` — generalise
    `arch_lr_ranges`, add structural-gene scoping
  - `session_15_feedforward_baseline.md` — `ppo_feedforward_v1`
    non-recurrent baseline
  - `session_16_hierarchical_transformer.md` — `ppo_hierarchical_v1`
    with per-tick runner attention, outer LSTM (recommended),
    decision committed to `design_decisions.md` before code
  - `session_17_optimiser_warmup.md` — LR warmup (default 0 = no-op),
    weight decay + AdamW optional, gated on Session 11 evidence
- Wrote `not_doing.md` explaining why items #7 (coverage math
  upgrades), #8 (LSTM num_layers > 2), and #9 (encoder changes)
  are parked, with promotion criteria for each.
- Updated `master_todo.md`:
  - Promoted all eight sessions into the ordered execution list.
  - Phase 3 sessions are explicitly marked independent and
    unordered — Session 11 evidence drives the order.
  - Phase 4 parking lot now points at `not_doing.md`.
  - Phase 3 prompts each re-check Session 11 evidence at their
    own preamble so they can be skipped individually.

**Not shipped:**
- Still no code changes.
- Session 11 post-run reprioritisation has not happened yet
  (obviously — the run doesn't exist).

**Next:** Session 10 when ready to start clearing debt.

## Session 10 — Housekeeping sweep (2026-04-07)

**Shipped:**

1. **`obs_dim` assertion unstaled.**
   `tests/test_population_manager.py::test_obs_dim_matches_env` no
   longer hardcodes a fresh integer — it now asserts only against the
   dynamically computed value derived from the `DIM` constants in
   `env/betfair_env.py`, with an inline comment pointing at where to
   update the layout. The previous `assert pm.obs_dim == 1630` was
   stale (current value is 1636, +6 agent state expansion in the
   Session 4.10 work), and a fresh integer would go stale the next
   time the layout changes.

2. **Retired names swept from live test fixtures and root docs.**
   Replaced `observation_window_ticks` (retired in arch-exploration
   Session 1) and the scalar `reward_early_pick_bonus` (split into
   `early_pick_bonus_min` / `early_pick_bonus_max` in arch-exploration
   Session 3) across:
   - `tests/test_population_manager.py` (search_ranges fixture,
     `test_int_in_range`, `test_reward_params_in_range`)
   - `tests/test_genetic_operators.py` (search_ranges, PARENT_A_HP,
     PARENT_B_HP)
   - `tests/test_genetic_selection.py` (search_ranges)
   - `tests/test_orchestrator.py` (search_ranges)
   - `tests/test_e2e_training.py` (search_ranges)
   - `PLAN.md` (hyperparameter table + retirement note)
   - `TODO.md` (Session 2.1 gene schema line)

   Deliberate references kept:
   - `tests/arch_exploration/test_reward_plumbing.py` —
     `test_observation_window_ticks_retired` is an intentional
     regression guard. `test_sampler_produces_reward_genes_in_range`
     already has an inline comment explaining the Session 3 split.
     Neither is stale.
   - `agents/ppo_trainer.py:54` — explanatory comment on the
     `early_pick_bonus_min`/`_max` data class fields. Legitimate
     historical marker, not a stale reference.
   - `plans/arch-exploration/**` — frozen phase log (out of scope
     per session prompt).
   - `plans/next_steps/**` — current session scaffolding, describes
     the work we are doing.
   - `.claude/worktrees/**` — disposable scratch, not part of the
     live codebase.

3. **Schema inspector view confirmed shipped.**
   `frontend/src/app/schema-inspector/` exists and renders a
   read-only table of every gene returned by
   `api/routers/training.py::get_hyperparameter_schema`, which reads
   directly from `config.yaml → hyperparameters.search_ranges`. By
   construction it auto-includes every gene introduced in
   arch-exploration Sessions 3, 5, 6 and 7 — there is no hard-coded
   gene list. No new work required.

4. **TODO/FIXME triage (commits `e76ac98..HEAD`).**
   `git diff e76ac98..HEAD` shows zero new `TODO` / `FIXME` / `XXX`
   / `HACK` comments in code. The only matches are doc headings
   (`# Master TODO — ...`) and cross-refs to the legacy root
   `TODO.md`, neither of which is a code-level debt marker. Nothing
   to fix, promote, or delete.

5. **Integration suite pass/fail state unchanged.**
   - **Start of session:** `pytest tests/ -m "gpu or slow"` →
     4 passed, 0 failed (GPU smoke tests for `ppo_lstm_v1`,
     `ppo_time_lstm_v1`, `ppo_transformer_v1` + `test_cuda_is_available`).
   - **End of session:** same command → 4 passed, 0 failed. No
     regression.

**Fast-suite state (context, not a session exit criterion):**
Running the fixture-touched test files in isolation —
`tests/test_population_manager.py`, `tests/test_genetic_operators.py`,
`tests/test_genetic_selection.py`, `tests/test_orchestrator.py` —
gives 148 passed, 1 skipped. The full default-marker suite has 18
pre-existing failures + 5 errors in `test_api_training.py`,
`test_integration_session_2_7b.py`, `test_integration_session_2_8.py`,
`test_real_extraction.py`, `test_session_2_8.py`, `test_session_4_9.py`,
`test_training_worker.py`, and `test_e2e_training.py`. Verified
pre-existing by stashing the Session 10 diff and re-running those
files — same failures, same errors. These are environmental issues
(port 18002 held by a stale worker, MySQL / real-data dependencies,
API/worker IPC) and are out of scope for Session 10 per the "do not
fix integration tests that were already broken" rule. Filed as a
follow-up item in `next_steps.md`.

**Files modified:**
- `tests/test_population_manager.py`
- `tests/test_genetic_operators.py`
- `tests/test_genetic_selection.py`
- `tests/test_orchestrator.py`
- `tests/test_e2e_training.py`
- `PLAN.md`
- `TODO.md`
- `plans/next_steps/master_todo.md` (tick Session 10)
- `plans/next_steps/progress.md` (this entry)
- `plans/next_steps/lessons_learnt.md`
- `plans/next_steps/next_steps.md` (pre-existing fast-suite failures
  follow-up)

**Manual tests run:** M1 (UI smoke) — partial.
  - Automated surrogate: grep of `frontend/` and `api/` for the two
    retired names returns zero hits, so no stale UI label/tooltip/
    error string can leak the retired gene names. Pass.
  - Human-click portion (visit schema inspector, confirm every
    live gene renders; open the plan editor; confirm no console
    errors) **not executed in this session** — the session was
    CPU-only and headless. Queued for the next session that spins
    up the stack.

**Not shipped / deferred:**
- Human-driven portion of M1 UI smoke — see above.
- Fixes to the 18 pre-existing fast-suite failures — out of scope
  per "do not fix integration tests that were already broken".

---

## Cross-project sequencing — DO NOT IGNORE

This folder is **not the only thing in flight**. ai-betfair has its
own plan folder at `c:/Users/jsmit/source/repos/ai-betfair/plans/rl-compat/`
which has work that must interleave with this one. The audit
recorded in `ai-betfair/plans/rl-compat/progress.md` Session 0
identified critical bugs in ai-betfair that fire **the moment**
this folder's Session 11 produces a high-scoring transformer
checkpoint.

**The required ordering is:**

1. `ai-betfair/plans/rl-compat/sessions/session_01_critical_compat.md`
   — fixes ai-betfair's hidden-state plumbing for transformers,
   adds an architecture allowlist. **Must land before this folder's
   Session 11**, otherwise ai-betfair will silently route the new
   transformer checkpoint into broken plumbing.
2. `ai-betfair/plans/rl-compat/sessions/session_02_high_compat.md`
   — startup assertions, error message fixes, live-feed key
   validation. **Must land before any non-LSTM model is loaded
   against live data.** Not strictly blocking Session 11 here, but
   blocking any later live-deploy.
3. **This folder's Session 10** (housekeeping).
4. **This folder's Session 11** (real multi-gen run). After it
   lands, transformer checkpoints exist in
   `rl-betfair/registry/models.db`.
5. `ai-betfair/plans/rl-compat/sessions/session_04_*.md` — live
   shakeout against one of those new checkpoints. Prompt is
   intentionally not yet written; it gets scoped after Session 11
   produces a concrete artefact to point at.
6. `ai-betfair/plans/rl-compat/sessions/session_03_medium_low_cleanup.md`
   — can slot in anywhere after rl-compat Session 02; not blocking
   this folder.
7. Any of this folder's Phase 3 sessions (12–17), prioritised
   based on Session 11 findings.

**Rule:** before starting any session in this folder, check
`ai-betfair/plans/rl-compat/master_todo.md` for blocking items.
Before starting any session in `ai-betfair/plans/rl-compat/`,
check this folder's `master_todo.md` for upstream changes that
might invalidate it.

If you find yourself wanting to skip ahead to a session that the
ordering above puts later, stop and re-read this section. The
ordering exists because at least one upstream session must land
first or downstream work proceeds against silently-broken
infrastructure.
