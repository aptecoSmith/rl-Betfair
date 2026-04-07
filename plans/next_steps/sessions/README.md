# sessions/

One file per session prompt. Numbering continues from the
`arch-exploration` phase, so the first session in this folder is
`session_10_*.md`, not `session_01_*.md`.

## Naming convention

`session_NN_short_slug.md` where `NN` is two digits and the slug
is 2–4 words joined with underscores. Examples:

- `session_10_housekeeping.md`
- `session_11_real_run.md`
- `session_12_hold_cost_design.md`

## Structure of a session prompt

Every prompt starts with a "Before you start — read these" block
pointing at the shared planning docs:

- `../purpose.md`
- `../hard_constraints.md`
- `../master_todo.md`
- `../progress.md` (last entry)
- `../lessons_learnt.md`
- `../design_decisions.md`
- `../ui_additions.md`
- `../initial_testing.md` / `../integration_testing.md` /
  `../manual_testing_plan.md` as relevant
- repo-root `CLAUDE.md`

Then **Goal**, **Scope (in / out)**, **Tests to add**, **Session
exit criteria**, and **Do not**. Match the shape of the
arch-exploration session prompts — they worked.

## Tick-list lives elsewhere

This folder does not track which sessions are done. That is in
`../master_todo.md`. Keep it single-source-of-truth.
