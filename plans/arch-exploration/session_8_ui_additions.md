# Session 8 — UI wiring for new knobs

## Before you start — read these

- `plans/arch-exploration/purpose.md`
- `plans/arch-exploration/master_todo.md` (Session 8)
- `plans/arch-exploration/testing.md` — UI tests run in the frontend
  test runner. Still no GPU.
- `plans/arch-exploration/progress.md` — confirm Sessions 1-7 done.
- `plans/arch-exploration/lessons_learnt.md`
- **`plans/arch-exploration/ui_additions.md`** — this is the spec
  for this session. Read it in full. Every checklist item must end
  this session ticked off.
- `frontend/` — read the existing config-editor components to match
  style. Do NOT invent new patterns; reuse what's there.

## Goal

Expose every new configurable value added in Sessions 1–7 in the web
UI. No developer-only YAML knobs remain. Build the training-plan page
from Session 4's backend.

## Scope

**In scope:**
- Every item in `ui_additions.md`. That file is authoritative — if
  something is on the list, it must ship this session. If something
  was forgotten on the list, add it and then do it.
- The training-plan page (Session 4 backend). Biggest chunk of work.
- The arch-specific LR range editor (Session 6).
- The min/max validator for the early-pick range (Session 3).
- A read-only "schema inspector" view that lists every gene
  currently in the search ranges with its type, range, and a link
  to the file that reads it. This makes future dead-gene bugs
  visible immediately.

**Out of scope:**
- Adding new genes. Everything in this session is UI wiring for
  genes that already exist as of Session 7.
- Backend work. All backend endpoints should already exist from
  Sessions 1-7. If you discover a missing endpoint, add it as a
  tiny side-quest, document in `lessons_learnt.md`, and continue.

## UI conventions to follow

- **Distinguish config from search-range.** Some values are "the
  current config" (e.g. the default architecture, population size)
  and some are "the range to mutate over" (e.g. `gamma ∈ [0.95,
  0.999]`). These must not share a widget. Build or reuse a distinct
  "range editor" component and only use it for search ranges.
- **Server-side validation is authoritative.** UI validation widgets
  (like the min ≤ max check) are belt-and-braces. The backend must
  always re-validate, and the UI must surface server errors cleanly.
- **Label every knob with the gene name** exactly as it appears in
  `config.yaml`. Developers and users need to map UI fields to grep
  targets. No friendly renames that hide the real key.

## Tests to add

- Component tests for each new widget (range editor variants, choice
  editor, min/max validator).
- Page test for the training-plan list / detail / editor flow. Mock
  the backend API.
- An end-to-end smoke test that renders every new field on the
  config page and asserts they are present. This is the
  "nothing-dropped" check — it catches the case where a gene is
  added to `ui_additions.md` but never actually rendered.

All tests run in the frontend test runner, no backend required, no
GPU.

## Session exit criteria

- Every item in `ui_additions.md` is ticked off.
- All frontend tests pass.
- Manual sanity check: spin up the UI (`npm run dev` or equivalent
  under `frontend/`), visit the config page, visit the training-plan
  page, confirm every new gene is visible and editable.
- `progress.md` Session 8 entry.
- `lessons_learnt.md` — capture anything about the existing
  frontend patterns that surprised you.
- Commit.

## Do not

- Do not run a GPU training loop from the UI as part of verification.
  The UI's "launch training" button still dispatches to the existing
  backend, which runs on GPU — that's fine, but don't click it from
  this session. Session 9 handles the first GPU shakeout.
- Do not refactor the existing config editor "just to make it
  cleaner". Additive work only. Refactors are a separate session.
- Do not hardcode the list of genes in the UI. Fetch the schema from
  the backend so that adding a gene in future does not require a UI
  change.
