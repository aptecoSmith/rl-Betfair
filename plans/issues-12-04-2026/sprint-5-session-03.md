# Sprint 5, Session 3: Forced Arbitrage — Training Integration + UI (Issue 05, Session 3)

Read `CLAUDE.md` and `plans/issues-12-04-2026/05-forced-arbitrage/`
before starting. Follow session 3 of `master_todo.md`.

**Prerequisite:** Sprint 5 sessions 1+2 (mechanics + reward) must be done.

## Scope

Wire scalping into training and the UI:

1. Scalping genes: scalping_mode, arb_spread_scale, naked_penalty_weight
2. Wizard: "Scalping mode" toggle with help text
3. Evaluator: respect scalping_mode, add scalping metrics
4. Scoreboard / model detail: show scalping metrics for scalping models
5. Training monitor: arb events in activity log, arb stats in episode display

See session 3 of `master_todo.md` for the full task list.

## After this session

Run a scalping training session (small population, 3-5 gens) to
validate the full loop: paired orders generate, some fill, reward
tracks locked PnL vs naked exposure, the GA evolves arb_spread.

## Commit

`feat: scalping training integration — genes, wizard, evaluator, metrics`
Push: `git push all`
