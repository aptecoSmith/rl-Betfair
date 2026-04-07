# Purpose — Next Steps

## Why this folder exists

The `arch-exploration` phase delivered the infrastructure needed to
search honestly over reward shaping, PPO hyperparameters, LSTM
structural knobs, and architecture choice. Nine sessions landed, the
GPU shakeout passed all invariants, and we now have a genetic
population system that is no longer silently ignoring its own genes.

But the shakeout was infrastructure verification, not an actual
search. And along the way we consciously deferred a pile of work —
reward-shaping ideas, PPO knobs, fourth architecture candidates,
optimiser tweaks, coverage-math upgrades, and a handful of
housekeeping items. None of those were urgent at the time; several
will become urgent the moment we run a real multi-generation
exploration and look at the results.

This folder is where that follow-on work gets planned, triaged, and
broken into sessions.

## What success looks like

- **Session 10 (the first real multi-generation run) happens** and
  its results are logged, analysed, and used to reprioritise
  everything else in this folder.
- **Every deferred item from `arch-exploration` has an explicit
  status**: done, parking-lot, or promoted into a session prompt.
  Nothing rots silently in a "maybe later" pile.
- **Housekeeping debt stays small.** Drive-by fixes (stale tests,
  retired-gene leftovers) get swept regularly rather than piling up
  across future work.
- **No speculative work.** Parking-lot items (coverage math upgrades,
  deeper LSTMs, encoder changes) are only promoted when a real run
  gives us a concrete reason.

## Relationship to `arch-exploration/`

Treat `arch-exploration/` as historical context, not active work.
Its `master_todo.md`, `progress.md`, and `lessons_learnt.md` are
frozen references that explain *why* the current state of the code
looks the way it does. Items identified at the end of that phase
live in this folder's `next_steps.md` backlog and graduate into
`sessions/session_NN_*.md` prompts when promoted.

## Folder layout

```
plans/next_steps/
  purpose.md              ← this file: why we're here, what done looks like
  hard_constraints.md     ← non-negotiables (CLAUDE.md-derived)
  master_todo.md          ← ordered session list, tick boxes
  next_steps.md           ← raw backlog (review candidates, triage notes)
  progress.md             ← one entry per completed session
  design_decisions.md     ← load-bearing decisions with rationale
  lessons_learnt.md       ← surprising findings, append-only
  ui_additions.md         ← running list of UI work owed
  initial_testing.md      ← fast CPU-only tests done during every session
  integration_testing.md  ← slow tests (GPU, full runs) — opt-in
  manual_testing_plan.md  ← human-in-the-loop verification steps
  sessions/               ← one prompt per session, numbered
    session_NN_*.md
```

Read `hard_constraints.md` before starting any session. Read the
last entry of `progress.md` to know what state the repo was left in.
Read `lessons_learnt.md` if the session involves anything similar
to previous work — that is where gotchas go.
