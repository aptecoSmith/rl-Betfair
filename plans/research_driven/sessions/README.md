# Sessions — Research-Driven

One file per session, named `session_NN_<short_slug>.md`.

Numbering continues from `next_steps/master_todo.md`. Pick the next
free number when promoting an item — do not start at 1.

A session prompt is a self-contained instruction set for a future
chat. Anyone (or any agent) opening it should be able to start work
without re-reading the rest of the planning folder, although they
*should* still skim `hard_constraints.md` and the most recent entry
of `progress.md` before touching code.

## Required sections in every session prompt

```
# Session NN — Title

## Goal
One-paragraph statement of what shipping this session looks like.

## Inputs
- Files to read first (with paths)
- Constraints to obey (cite hard_constraints.md item numbers)
- Open questions answered for this session (cite open_questions.md
  Q-numbers and the operator decision)

## Steps
Numbered, concrete. Each step is a single action with a verifiable
outcome. No "consider whether..." steps — the planning happens
before the session opens, not during.

## Acceptance criteria
Copy from proposals.md, then make them more specific. The session
is done when every box is ticked.

## Tests added
Which initial_testing.md and integration_testing.md items this
session adds, and why.

## What this session does NOT do
Explicit list. Anything tempting that is *not* in scope.

## Cross-repo follow-ups
ai-betfair items owed (cite downstream_knockon.md sections).
```

## When a session is finished

1. Tick its box in `master_todo.md`.
2. Append to `progress.md` using the format in that file.
3. Append any surprising findings to `lessons_learnt.md`.
4. Append any new UI work owed to `ui_additions.md`.
5. If the session changed obs schema, action space, or matcher,
   update `downstream_knockon.md` to reflect what `ai-betfair` now
   needs to do.
6. Leave the session prompt file in place — it is the historical
   record of what was meant to ship vs what did.

This README is intentionally short. The convention is in the
top-level files of `research_driven/`; this directory exists to
hold the prompts, not to re-document the convention.
