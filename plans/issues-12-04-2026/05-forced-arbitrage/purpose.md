# 05 — Forced Arbitrage (Scalping Mode)

## What

Add a training mode where every aggressive bet automatically generates
a paired passive order on the opposite side of the same runner. The
agent learns to scalp — picking spots where the market will move to
fill the second leg, locking in a small profit before the race starts.

New action dimension: `arb_spread` — how many ticks away from the
fill price to place the second leg. The agent must learn the tension
between spread (profit) and distance (fill probability), accounting
for Betfair's ~5% commission on net market profit.

New reward path: scalping rewards completion of arb pairs, penalises
naked exposure at the off, and ignores directional win/loss metrics
that are meaningless for market-making.

## Why

- No model has yet attempted arbitrage despite having both back and
  lay capability. The action space doesn't naturally encourage it
  because directional reward shaping (precision bonus, early pick
  bonus) rewards winning bets, not hedged positions.
- Scalping is a fundamentally different strategy to directional
  betting — lower returns per trade but far safer. A model trained
  to scalp could run live with much lower risk than a directional
  model.
- Forcing the paired structure (rather than hoping the agent discovers
  it) is the right approach because the action space is too large for
  the agent to stumble into arb behaviour spontaneously.

## How it works

1. Agent places aggressive back on Runner X at 5.0 (fills immediately)
2. System automatically places passive lay on Runner X at 5.0 minus
   N ticks (where N = agent's arb_spread output)
3. If market drops and the lay fills at 4.6 → locked profit on both
   legs regardless of race outcome. Net ≈ £0.40/£10 minus 5%
   commission = £0.38
4. If lay doesn't fill → naked back exposure. Cancels at the off.
   Bet settles directionally (normal win/loss).

The same works in reverse: aggressive lay → passive back further out.

## Commission constraint

Betfair charges ~5% on net market profit. The second leg must be
enough ticks away that the locked spread exceeds commission. At low
prices (2.0-3.0) a single tick is ~£0.02/£10 — need several ticks.
At high prices (10.0+) a single tick is ~£0.50/£10 — one tick may
suffice. The agent must learn this price-dependent breakeven.
