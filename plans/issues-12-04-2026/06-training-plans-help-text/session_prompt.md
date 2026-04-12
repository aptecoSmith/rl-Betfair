# Session: Training Plans Help Text

Read `CLAUDE.md` and `plans/issues-12-04-2026/06-training-plans-help-text/`
before starting. Follow `master_todo.md`. Mark items done as you go and
update `progress.md` at the end.

## Context

The training plans editor (`frontend/src/app/training-plans/`) has bare
technical labels with minimal explanation. The training wizard
(`frontend/src/app/training-monitor/`) already has rich help text using
`.help-text` paragraphs and `.field-help` spans — use the same pattern.

The user is learning reinforcement learning. Every field should explain:
1. What it controls (in plain English)
2. What good values look like (concrete recommendations)
3. What happens if you get it wrong (consequences)

## What to do

Add help text to every field and section in the editor. See
`master_todo.md` for the full list of fields and suggested copy.

### Layout

Use inline help text consistent with the wizard. Each field gets a
`<span class="field-help">` below the input. Sections get a
`<p class="help-text">` paragraph. The existing `.hint` class in the
training plans SCSS is already used for some hints — either reuse it
or switch to the wizard's classes for consistency.

Check `training-monitor.html` and `training-monitor.scss` for the
existing help text patterns:
- `.help-text` — block-level explanatory paragraph
- `.field-help` — inline hint below an input

Also check `training-plans.scss` for the existing `.hint` class and
decide whether to keep it or align with the wizard.

### Gene tooltips

The `app-gene-editor` component receives a `spec` object that includes
a `description` field from the schema. Check if it already renders
this description. If not, add it as a subtitle or tooltip.

## Key files

| File | What to change |
|------|----------------|
| `frontend/src/app/training-plans/training-plans.html` | Add help text to every field and section |
| `frontend/src/app/training-plans/training-plans.scss` | Style help text consistently with wizard |
| `frontend/src/app/training-monitor/training-monitor.scss` | Reference for existing `.help-text` / `.field-help` styles |
| `frontend/src/app/gene-editor/` (if exists) | Check/add gene description display |

## Constraints

- No backend changes.
- Help text should be accurate — don't guess about what a field does.
  If unsure, read the backend code that consumes the value.
- Keep the same visual style as the wizard's help text.
- `cd frontend && ng build` must be clean.

## Commit

Single commit: `feat: add explanatory help text to training plans editor`
Push: `git push all`
