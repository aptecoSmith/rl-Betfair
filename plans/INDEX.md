# Plans index

Chronological order of work-tracking plans under `plans/`. Folder
names stay symbolic (referenced from CLAUDE.md, config, code, other
plan files); this index is the canonical "which plan came when /
what's the latest" source.

To add a new plan: append a row at the bottom with the next number.
Don't renumber on top — numbers are append-only history.

| # | Plan folder | First commit | One-line |
|---|---|---|---|
| 01 | [arch-exploration](arch-exploration/) | 2026-04-06 | Architecture + reward-schema + training-plan plumbing |
| 02 | [next_steps](next_steps/) | 2026-04-07 | Backlog of small follow-ups; not a single plan |
| 03 | [research_driven](research_driven/) | 2026-04-08 | Research-mode workflow notes |
| 04 | [bet-explorer-redesign](bet-explorer-redesign/) | 2026-04-11 | Bet-explorer page rewrite |
| 05 | [ew-metadata-pipeline](ew-metadata-pipeline/) | 2026-04-11 | Each-way market metadata extraction pipeline |
| 06 | [ew-settlement](ew-settlement/) | 2026-04-11 | Each-way settlement accounting |
| 07 | [issues-11-04-2026](issues-11-04-2026/) | 2026-04-11 | Sprint backlog (config budget, stop options, market filter, …) |
| 08 | [issues-12-04-2026](issues-12-04-2026/) | 2026-04-12 | Sprint backlog (log consolidation, ETA overhaul, training-plans integration, …) |
| 09 | [arb-improvements](arb-improvements/) | 2026-04-14 | Reward clipping, entropy floor, signal bias, BC pretrainer, aux head |
| 10 | [scalping-asymmetric-hedging](scalping-asymmetric-hedging/) | 2026-04-15 | Asymmetric naked-loss raw reward + freed-budget reservation |
| 11 | [scalping-active-management](scalping-active-management/) | 2026-04-16 | Re-quote + fill-prob + risk aux heads + UI surfaces |
| 12 | [scalping-close-signal](scalping-close-signal/) | 2026-04-17 | "Take the red" close-at-loss action (Session 01 landed) |
| 13 | [scalping-naked-asymmetry](scalping-naked-asymmetry/) | 2026-04-18 | Per-pair naked P&L penalty — fix luck-cancellation in raw reward |
| 14 | [scalping-equal-profit-sizing](scalping-equal-profit-sizing/) | 2026-04-18 | Correct passive-leg sizing — equalises P&L (not exposure) after commission |
| 15 | [policy-startup-stability](policy-startup-stability/) | 2026-04-18 | **(latest)** PPO advantage normalisation — prevents first-update spike that saturates `close_signal` head |

## Conventions

- Each plan folder follows the layout established in
  `scalping-active-management/`: `purpose.md`, `hard_constraints.md`,
  `master_todo.md`, `progress.md`, `lessons_learnt.md`,
  `session_prompt.md`, `session_prompts/`.
- The "latest" plan is the bottom row of the table. When referencing
  a plan in commits or code, use the folder name (stable) not the
  number (positional, but unchanging once assigned).
- Top-level plan files (`discussion-session-prompt.md`,
  `long-term.md`, `project-brief.md`) sit alongside this index and
  are not numbered — they're meta, not work-tracking.
