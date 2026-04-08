# Purpose — Research-Driven Work

## Why this folder exists

`research/research.txt` is a set of microstructure notes the operator
gathered before session 10 — covering money pressure, queue position,
and crossing the spread on Betfair Exchange. Reading it surfaced two
things at once:

1. The simulator already crosses the spread on every bet (it has no
   other mode), and that is what the operator's screenshots have
   been showing all along — not a UI bug, just an under-modelled
   regime.
2. There is a real, observable bug in `ai-betfair` where the live
   wrapper declares fills for trades that demonstrably had no
   liquidity to match against. That one is pre-existing and
   unrelated to any new feature work.

This folder is where the work that flows from those observations
gets planned, triaged, and broken into sessions. It sits **after**
the `next_steps/` programme — nothing here should land until session
10 and the immediately-following debt-clearing work in
`next_steps/master_todo.md` are done. It is *not* a fork of
`next_steps/`; it is its successor for a specific class of work.

## What success looks like

- **Every research-driven proposal has an explicit status**: in
  flight, parked, or done. The `proposals.md` menu doesn't rot into
  a "maybe later" pile.
- **The `ai-betfair` knock-on is tracked, not assumed.** Every item
  here that requires changes in the live wrapper has those changes
  catalogued in `downstream_knockon.md`, with cost estimates, before
  it lands in `master_todo.md`.
- **The phantom-fill bug is fixed before any new feature work.**
  See `downstream_knockon.md` §0 — this is a hard prerequisite, not
  a co-task.
- **No speculative work.** Items are only promoted from `proposals.md`
  into `master_todo.md` once the open questions in
  `open_questions.md` have operator decisions on them.

## Relationship to `next_steps/`

`next_steps/` is the active session backlog leading up to and through
session 10 — debt-clearing, training-run housekeeping, and the first
real multi-generation run. It does not concern itself with simulator
realism beyond the bugs already on its docket.

This folder picks up the *next* class of question: "is the simulator
optimistic in ways that will hurt us in live?". It is allowed to
propose changes that break checkpoint compatibility and require
re-trains, because by the time it starts those costs will be paid
once already by the session-10 work.

Treat `next_steps/` as the prerequisite. Treat this folder as the
follow-on.

## Folder layout

```
plans/research_driven/
  purpose.md              ← this file: why we're here, what done looks like
  analysis.md             ← current sim measured against the research
  proposals.md            ← P1–P5 menu, ordered by cost-to-value
  open_questions.md       ← decisions needed before sizing
  downstream_knockon.md   ← what the proposals require in ai-betfair
  hard_constraints.md     ← non-negotiables (CLAUDE.md + research-derived)
  design_decisions.md     ← load-bearing decisions with rationale
  bugs.md                 ← bugs surfaced by research-driven planning (R-prefix)
  not_doing.md            ← deliberately parked items, with promotion triggers
  master_todo.md          ← ordered session list, tick boxes (empty until promoted)
  progress.md             ← one entry per completed session
  lessons_learnt.md       ← surprising findings, append-only
  ui_additions.md         ← running list of UI work owed
  initial_testing.md      ← fast CPU-only tests done during every session
  integration_testing.md  ← slow tests (GPU, full runs) — opt-in
  manual_testing_plan.md  ← human-in-the-loop verification steps
  sessions/               ← one prompt per session, numbered
    README.md
    session_NN_*.md
```

Read `hard_constraints.md` before starting any session, the same
rule as `next_steps/`. Read the most recent entry of `progress.md`
to know what state the repo was left in. Read `lessons_learnt.md`
if the session involves anything similar to previous work — that is
where gotchas go.

For *first-time* readers of this folder (no session has run yet),
read in this order: `purpose.md` (here) → `analysis.md` (what the
simulator currently does and why proposals exist) → `proposals.md`
(the menu) → `open_questions.md` (what the operator needs to
decide) → `downstream_knockon.md` (the `ai-betfair` cost) →
`hard_constraints.md` and `design_decisions.md` (the rules).
`master_todo.md`, `progress.md`, `lessons_learnt.md`, `bugs.md`,
and `ui_additions.md` start mostly empty and fill up as sessions
land.
