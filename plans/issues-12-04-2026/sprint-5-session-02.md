# Sprint 5, Session 2: Forced Arbitrage — Reward + Settlement (Issue 05, Session 2)

Read `CLAUDE.md` and `plans/issues-12-04-2026/05-forced-arbitrage/`
before starting. Follow session 2 of `master_todo.md`.

**Prerequisite:** Sprint 5 session 1 (environment mechanics) must be done.

## Scope

Build the scalping reward function and settlement:

1. Completed arb reward: locked_pnl after commission (always positive)
2. Naked exposure penalty: proportional to exposure at the off
3. Early lock bonus: time-proportional reward for fast second-leg fills
4. Do NOT use precision_bonus or early_pick_bonus — meaningless for scalping
5. Settlement: cancel unfilled passive legs at race-off, naked bets
   settle directionally, completed arbs settle both legs
6. Episode stats: arbs_completed, arbs_naked, locked_pnl, naked_pnl
7. Reward accounting: scalping PnL is raw, verify invariant

See session 2 of `master_todo.md` for the full task list.

## Commit

`feat: scalping reward function + paired settlement`
Push: `git push all`
