# Session 10 — Housekeeping sweep

## Before you start — read these

- `../purpose.md`
- `../hard_constraints.md`
- `../master_todo.md` (you are Session 10)
- `../progress.md` — read the last entry
- `../lessons_learnt.md`
- `../design_decisions.md`
- `../ui_additions.md`
- `../initial_testing.md` — CPU-only, fast feedback, test after each fix
- `../integration_testing.md` — run the existing integration suite
  **once at the start** to confirm nothing is already broken before
  we touch anything
- `../manual_testing_plan.md` — M1 (UI smoke) is required at the end
- repo-root `CLAUDE.md`

## Goal

Clear the debt that accumulated during the arch-exploration phase so
that Session 11 (real multi-generation run) starts from a clean,
verified baseline. No new features; no speculative refactors.

## Scope

**In scope:**

1. **Fix the stale `obs_dim` assertion.**
   `tests/test_population_manager.py:408` hardcodes `obs_dim == 1630`;
   the current value is 1636 (flagged in
   `arch-exploration/lessons_learnt.md` Session 1 entry, never fixed).
   Update the assertion to match the current observation layout and
   add a comment pointing at where the value is computed so the next
   drift is easier to diagnose.

2. **Audit retired names.** Grep the whole repo (including `frontend/`,
   `api/`, `scripts/`, `docs/`) for:
   - `observation_window_ticks` (retired in Session 1 of
     arch-exploration)
   - `reward_early_pick_bonus` as a **scalar** gene name (split into
     `_min` / `_max` in Session 3)
   Any stale reference — in code, tests, YAML, JSON, markdown, or UI
   strings — gets removed or updated. If a test still mentions the
   retired name deliberately (e.g. a regression guard), add an inline
   comment explaining why.

3. **Confirm the schema inspector view landed.** Session 8 of
   arch-exploration proposed a read-only "schema inspector" page
   that lists every gene currently in the search ranges with its
   type, range, and the file that consumes it. Check whether it
   actually shipped. If yes, verify it renders every gene including
   the ones introduced in Sessions 3, 5, 6, and 7. If no, **do not
   build it in this session** — add a TODO entry in `next_steps.md`
   and note it in `progress.md`. This session is for clearing known
   debt, not adding features.

4. **Triage stray `TODO` / `FIXME` comments.** Grep for TODO/FIXME
   comments added between commits `e76ac98` (phantom-profit fix) and
   the current HEAD. For each, decide one of three outcomes:
   - **Fix now** if it's a one-line correction.
   - **Add to `next_steps.md`** with a proper backlog entry.
   - **Delete the comment** if the thing is actually done and the
     comment is lying.
   Document the triage outcomes in `progress.md` so future sessions
   can see what was cleared vs what was promoted.

5. **Run the existing integration suite once** (`pytest tests/ -m "gpu
   or slow"`) before making any changes. Record the starting pass/fail
   state in `progress.md`. After all fixes are in, run it again and
   confirm the pass/fail state has not regressed. This is the only
   time integration tests run during Session 10.

**Out of scope:**

- Any new feature, gene, reward term, or architecture.
- Refactoring code "while I'm here". Rename spree, formatter runs,
  import sorting — all deferred to a dedicated refactor session if
  ever needed.
- Fixing any integration test that was **already broken** at session
  start. Log the failure, file a follow-up entry in `next_steps.md`,
  but do not attempt a fix in Session 10 unless it is directly blocked
  by one of the in-scope items.
- Touching the existing `testing.md` / old docs in `arch-exploration/`.
  That folder is frozen.

## Tests to add

For item 1 (obs_dim fix):
- The updated assertion must match the live `BetfairEnv` output.
  Prefer asserting against a dynamically computed expected value
  rather than a fresh hardcoded integer — this is the whole reason
  the original test went stale.

For items 2 / 3 / 4:
- No new tests required unless a grep hit reveals a behavioural bug
  rather than a pure-text leftover. If it does, add a minimal
  regression test in `tests/next_steps/` and document it in the
  session's `progress.md` entry.

Entire Session 10 test addition should be small. If you find yourself
writing more than ~100 lines of new test code, stop — the scope has
probably drifted.

## Manual tests

- **M1 (UI smoke)** from `manual_testing_plan.md`. Required.
  Particularly check that retired-gene names do not leak into any UI
  label, tooltip, or error message.

## Session exit criteria

- `obs_dim` test passes without hardcoding a fresh integer.
- Grep for `observation_window_ticks` returns zero hits outside of
  `arch-exploration/lessons_learnt.md` (which is a frozen historical
  log and may legitimately mention it).
- Grep for scalar `reward_early_pick_bonus` returns zero hits outside
  frozen historical docs.
- Schema inspector status documented in `progress.md` (shipped /
  not-shipped / partially-shipped + action taken).
- TODO/FIXME triage outcomes recorded in `progress.md`.
- `initial_testing` fast suite passes.
- `integration_testing` suite pass/fail state is no worse than the
  recorded starting state.
- M1 manual test recorded in `progress.md`.
- Entry added to `progress.md`; `lessons_learnt.md` updated with
  anything surprising; `master_todo.md` Session 10 ticked.
- Commit (do not push).

## Do not

- Do not widen the scope. A housekeeping sweep that grows into a
  refactor is how one-hour sessions become one-week sessions.
- Do not fix integration tests that were already broken. They become
  a follow-up backlog item.
- Do not build the schema inspector in this session. If it didn't
  ship in Session 8, promote it to its own session.
- Do not touch `plans/arch-exploration/` — it is frozen context.
