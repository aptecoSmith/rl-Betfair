# Lessons Learnt — Forced Arbitrage

## From discussion

- The agent has never discovered arbitrage despite having back, lay,
  and passive order capability. The action space is too large and
  directional reward shaping (precision bonus, early pick bonus)
  actively discourages hedging — a completed arb always has one
  "losing" bet which hurts precision metrics.

- Forcing the paired structure is the right approach. Rather than
  hoping the agent stumbles into arb behaviour, make it structural:
  every aggressive fill generates a passive counter-order. The agent's
  skill becomes *when* to bet and *how far apart* to set the legs.

- Commission is the key constraint. Betfair takes ~5% of net market
  profit. At low prices (2.0-3.0) tick increments are tiny (£0.02)
  and you need many ticks of spread to cover commission. At high
  prices (10.0+) one tick may be enough. The agent must learn this
  price-dependent breakeven — it's not just "pick a number of ticks".

- Reward structure is fundamentally different from directional betting:
  - Completed arbs: always positive, one leg "losing" is the plan
  - Naked exposure at the off: the main risk — penalise this
  - Win rate / precision: meaningless, do not use
  - The reward should incentivise filling second legs quickly and
    minimising exposure at race start

- This is a genuine live strategy. Scalping produces smaller returns
  than correct directional bets but with far lower risk. A scalping
  model could run live with much smaller drawdown variance.

- The user explicitly sees this as same-horse, same-race, same-market
  arbitrage (back high / lay low on one runner), not cross-market
  WIN vs EACH_WAY arb. Cross-market arb is a separate, more complex
  idea.

## Two "it's fine, the tests pass" bugs (2026-04-15)

Sessions 1–2 landed the scalping mechanics with both formulas
wrong but internally consistent:

1. `_maybe_place_paired` auto-sized the passive leg with
   **equal** stake to the aggressive leg.
2. `get_paired_positions.locked_pnl` used `stake × spread ×
   (1 − commission)` — which equals the MAX-outcome P&L of an
   equal-stake pair, not the guaranteed floor.

The bugs reinforced each other. Equal-stake pairs reported
non-zero locked_pnl when the runner won, so the reward path
credited them. The agent learned to place more pairs. The tests
passed because they asserted the (buggy) locked_pnl formula.

Only the Gen 0 Bet Explorer screenshot evidence — equal-stake
back+lay pairs with a `+£1,379` headline sitting next to a 31.5%
precision rate — made it obvious that the "profit" was
directional luck in a scalping costume.

Fix landed in `plans/scalping-asymmetric-hedging/`:
- passive sizing: `S_passive = S_agg × P_agg / P_passive`.
- `locked_pnl = max(0, min(win_pnl, lose_pnl))`.

**Discipline for future reward terms:** when the reward depends
on (stake, price, outcome), write the test as "min over
outcomes" from the start. A closed-form formula that happens to
equal the correct answer in one case is an algebraic coincidence,
not a proof of correctness.
