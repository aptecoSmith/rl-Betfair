# Progress — Research-Driven

One entry per completed session, newest at the top. Each entry
records the factual outcome — what shipped, what files changed,
what tests were added, what did *not* ship and why.

This file is the source of truth for "what state did the last
session leave the repo in?". A new session starts by reading the
most recent entry. If the entry doesn't tell you, the previous
session under-documented and the next session is allowed to push
back.

Format:

```
## YYYY-MM-DD — Session NN — Title

**Shipped:**
- bullet of file/area changes

**Tests added:**
- bullet of test file + what it asserts

**Did not ship:**
- bullet of what was scoped but cut, with reason

**Notes for next session:**
- anything load-bearing the next reader needs

**Cross-repo follow-ups:**
- bullet for `ai-betfair` items owed (link to downstream_knockon.md
  section)
```

---

## (no sessions completed yet)

This folder was created on 2026-04-07 as a follow-on to
`next_steps/`, and no research-driven session has shipped yet.

The first entry will be added when the first item from
`master_todo.md` lands. Until then, treat the planning files
(`purpose.md`, `analysis.md`, `proposals.md`, `open_questions.md`,
`downstream_knockon.md`, `hard_constraints.md`,
`design_decisions.md`, `not_doing.md`) as the current state.

The first session that lands here is **not** session 11. Numbering
continues from `next_steps/master_todo.md` — pick the next free
number when promoting an item, do not start over.
