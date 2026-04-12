# Sprint 5, Session 1: Forced Arbitrage — Environment Mechanics (Issue 05, Session 1)

Read `CLAUDE.md` and `plans/issues-12-04-2026/05-forced-arbitrage/`
before starting. Follow session 1 of `master_todo.md`.

## Scope

Build the scalping environment mechanics:

1. Betfair tick ladder utility (`tick_offset` function)
2. `arb_spread` as 5th action dimension per runner
3. `scalping_mode` gene — when on, aggressive fills auto-generate
   passive counter-orders
4. Pair tracking: `pair_id` on Bet and PassiveOrder
5. BetManager helpers: get_paired_positions, get_naked_exposure
6. New observation features: has_open_arb, passive_fill_proximity,
   locked_pnl_frac, naked_exposure_frac

See `plans/issues-12-04-2026/05-forced-arbitrage/session_prompt.md`
for full details including tick ladder increments and commission math.

## Commit

`feat: forced arbitrage mechanics — paired orders + arb_spread action`
Push: `git push all`
