# Session 8 — Wizard UI, evaluator & UI consolidation

## Before you start — read these

- `plans/arb-improvements/purpose.md`
- `plans/arb-improvements/master_todo.md` — Phase 3, Session 8.
- `plans/arb-improvements/testing.md`
- `plans/arb-improvements/ui_additions.md` — **every unticked item
  in this file gets implemented this session**.
- `plans/arb-improvements/progress.md` — read sessions 1–7.
- `plans/issues-12-04-2026/05-forced-arbitrage/master_todo.md` —
  the existing scalping toggle and evaluator awareness work is the
  pattern this session extends.

## Goal

Make everything the operator needs for this plan editable from the
wizard and visible in the training monitor. Consolidate UI tasks
accumulated across sessions 1–7. Teach the evaluator to record
`bc_pretrain_steps` and warn on stale oracle cache.

## Scope

**In scope:**

- **Wizard (Angular):** new inputs on the appropriate steps —
  - Reward clipping (Session 1): `reward_clip`, `advantage_clip`,
    `value_loss_clip`.
  - Entropy floor (Session 2): `entropy_floor`,
    `entropy_floor_window`, `entropy_boost_max`.
  - Signal-bias warmup (Session 3): `signal_bias_warmup`,
    `signal_bias_magnitude`.
  - BC pretrain (Session 7): `bc_pretrain_steps`, `bc_learning_rate`.
    Greyed out when `scalping_mode` is off.
  - Help text per input. Defaults match session definitions.

- **Training monitor (Angular):**
  - Per-episode `clipped_reward_total` badge when clipping active
    (Session 1).
  - Per-head entropy panel with "collapsing" warning (Session 2).
  - Bet-rate / arb-rate sparklines per agent (Session 3).
  - BC warmup phase indicator + BC loss curve (Session 7).

- **Admin action (Angular):** "Scan days for arb oracle samples"
  — date-range picker, runs `training/arb_oracle.py scan` via an
  existing admin endpoint pattern, shows sample counts per day
  (Session 6).

- **Evaluator (Python):**
  - Record `bc_pretrain_steps` (and `bc_learning_rate` if non-
    default) in the model record so the scoreboard / detail page
    can show it.
  - On model load: compare oracle-cache mtime per training date
    against the episode file mtime; if cache is older, emit a
    warning event (evaluation still proceeds — BC only runs at
    training time).
  - Surface `oracle_density` per training date used by the model
    so low-density days are visible.

**Out of scope:**

- Aux head UI (Session 9).
- New verification visualisations (Session 10).

## Exact code path

Frontend paths (typical Angular structure — confirm during session):

- Wizard: `frontend/src/app/training-wizard/` — new inputs go on
  the parameters / constraints step that already hosts the
  scalping toggle from the forced-arb plan.
- Training monitor: `frontend/src/app/training-monitor/` —
  episode row template + agent detail panel.
- Admin oracle scan: if an admin area exists (check the frontend
  tree), extend it; otherwise add a simple dialog behind a dev
  flag and document for a later polish pass.

Backend paths:

- `api/routers/models.py` — record and expose `bc_pretrain_steps`.
- `training/evaluator.py` — oracle-cache staleness check.
- New endpoint for oracle scan: `api/routers/oracle.py` (or
  extend an existing admin router). Endpoint runs `scan_day` for
  each date in the request and returns densities.

## Tests to add (CPU-only, fast)

Create `tests/arb_improvements/test_ui_integration.py`:

1. **Evaluator records `bc_pretrain_steps`.** Save a model record
   with `bc_pretrain_steps=500`; load via the evaluator's read
   path; assert the value is present.

2. **Staleness warning.** Mock file mtimes: oracle cache older
   than episode file → evaluator emits a warning event. Cache
   newer → no warning.

3. **Oracle scan endpoint.** POST to the new endpoint with a
   synthetic date; assert response includes density and sample
   count fields.

4. **Wizard schema round-trip.** Submit a wizard config with every
   new knob set to a non-default; load it back; assert all values
   are preserved through the config serialiser.

Manual UI verification checklist (document in `progress.md` Session
8 entry):

- Wizard: flip every new knob, submit, start a training run, open
  the run record, confirm each knob's value is present.
- Training monitor: entropy panel shows five heads; bet-rate
  sparkline updates per episode.
- BC warmup phase indicator appears before PPO when
  `bc_pretrain_steps > 0`.
- Oracle scan admin action completes and shows density per day.

## Session exit criteria

- All backend tests pass: `pytest tests/arb_improvements/ -x`.
- Existing tests still pass.
- Frontend builds clean: `cd frontend && ng build`.
- Manual verification checklist executed; results logged in
  `progress.md`.
- Every task in `ui_additions.md` for sessions 1–7 is ticked.
- `progress.md` Session 8 entry written. End-of-Phase-3 note:
  decide whether Phase 4 (aux head) is needed — based on whether
  the Phase-1 smoke test + Phase-2 features + Phase-3 BC on a
  short run is enough to fix the 90fcb25f failure mode. If yes,
  defer Session 9 and jump to Session 10.
- `lessons_learnt.md` updated.
- Commit: `feat(ui): wizard + monitor + evaluator integration for arb improvements`.
- `git push all`.

## Do not

- Do not leave any ticked-off-in-code-but-not-in-UI knobs. This
  session exists because the UI wiring is a frequent miss. Every
  config knob is editable or the session isn't done.
- Do not rebuild frontend components from scratch. Extend existing
  ones (training wizard, training monitor, model detail page).
- Do not land this session with a broken `ng build`.
- Do not add GPU tests.
