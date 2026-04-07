# Design Decisions — Next Steps

Load-bearing decisions made during this phase, with rationale.
Append one entry per decision. Each entry answers three questions:

1. **What** was decided
2. **Why** (the tradeoff — what we gave up to get it)
3. **When to revisit** (concrete trigger, not "someday")

This file is not a changelog. It is the place a future session
looks when it's about to re-litigate a decision, to check whether
the original reasoning still applies. If the reasoning *does* still
apply, don't re-litigate; if it doesn't, record the reversal here
as a new entry rather than editing the old one.

For frozen context from the previous phase, read
`arch-exploration/lessons_learnt.md` — several entries there are
effectively design decisions that deserve citing rather than
rewriting.

---

## 2026-04-07 — Adopt the jarvis plan-folder layout

**What:** Restructured `plans/next_steps/` to match the scaffold
used in the `jarvis/plan/` folder in a sibling repo. Added
`hard_constraints.md`, `master_todo.md`, `design_decisions.md`,
split `testing.md` into `initial_testing.md` /
`integration_testing.md` / `manual_testing_plan.md`, and moved
future session prompts under a `sessions/` subfolder.

**Why:** The arch-exploration layout worked but left three soft
spots. (1) Hard constraints were buried inside `purpose.md` which
a session prompt reader was unlikely to open. (2) The single
`testing.md` conflated "fast CPU feedback" with "dedicated
integration runs" and "human-in-the-loop verification", which are
three distinct contracts. (3) Session prompts lived at the top
level of the folder, interleaved with planning docs, making it
hard to see what was a living reference vs a one-shot prompt.
The jarvis layout solves all three with cheap file-moves and
adds `design_decisions.md` (this file) which we didn't have
before. Cost: a dozen files instead of six, and a migration step
that breaks any hypothetical external link into the old
structure — but nothing links in yet, so the cost is notional.

**When to revisit:** If a third planning folder is created
(beyond `arch-exploration` and `next_steps`), re-evaluate whether
the jarvis layout is the right default. If either `initial_*` or
`integration_*` testing rules are being routinely ignored because
the split is confusing, merge them back.
