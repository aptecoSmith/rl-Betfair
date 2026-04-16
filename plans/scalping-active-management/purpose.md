# Purpose — Scalping Active Management

## Why this work exists

The reward-signal fix (commit `c218bfb`) and scoreboard tabs
(commit `7a2baaa`) made it possible to see what the scalping
agent is actually doing. The Gen 1 training run (pop=16,
gens=4, 2026-04-16) answered the question and the answer is
uncomfortable:

- **Top scalper `ef453cd9` completes only 14.5 % of its pair
  attempts.** 29 pairs done, 171 pairs where the passive leg
  never filled before race-off. Those 171 "arbs naked" are not
  deliberate unhedged gambles — they're **failed scalp
  attempts that devolved into directional bets by accident**,
  because nobody crossed the spread to hit the passive before
  race-off.
- **The agent chooses `arb_spread` per-runner per-tick already**
  (5th action dimension). But the only gradient signal reaching
  that output is a long credit chain: place passive → passive
  fills or doesn't → locked_pnl or naked_pnl → reward. Weak.
- **No active management.** Once the passive is placed, it sits
  until it fills or race-off cancels it. A real scalper would
  re-quote — if the passive hasn't filled with 60 seconds to
  go, cancel and move closer to market to avoid naked exposure.
- **No confidence or risk awareness.** The agent can't say "I
  don't think this pair will complete" or "this trade has wide
  predicted variance". Every decision carries identical
  implicit confidence.

## The strategy we want the agent to be able to express

A real human scalper's decision loop, per runner per tick:

1. **Place:** "Here's a good moment; lay 6 ticks above fill."
2. **Wait:** "Price is moving my way; fill looks likely."
3. **Re-quote:** "Price stalled; cancel and re-post 3 ticks
   closer with 90 seconds left."
4. **Close or accept:** "Re-quote still hasn't filled with 30
   seconds; hit the book aggressively to close rather than go
   naked into the race."

Our current agent can only do step 1 — and it does step 1
blindly, without a sense of fill likelihood. This plan gives it
steps 2–4 plus the self-awareness to use them well.

## Four changes, in dependency order

### 1. Active re-quote mechanic

Add a per-runner `requote_signal` action dim. When raised for a
runner with an outstanding passive that hasn't filled:

- Cancel the existing passive (uses existing
  `PassiveOrderBook.cancel` path).
- Re-place at current-tick `arb_ticks` from current LTP (not
  the original fill price).

Observation features added: `seconds_since_passive_placed`,
`passive_price_vs_current_ltp_ticks` (how far the passive is
from where the market is now). Lets the network notice
"I posted 8 ticks away but price hasn't moved; the passive is
now effectively 12 ticks from market because the whole book
drifted".

Hard constraint: re-quoting **never** walks the ladder. Same
one-price-only matching rule as the aggressive placement path.

### 2. Fill-probability head

Auxiliary supervised head: for each pair placed, predict
`P(passive fills before race-off | state, arb_ticks)`. Trained
with binary cross-entropy against the realised outcome. Shares
the network backbone with policy + value.

Three gains:

- **Direct gradient to the `arb_spread` output.** Currently the
  arb-ticks head only learns through the long credit chain.
  The fill-probability head trains on the same fills and
  back-propagates through the shared backbone, teaching it to
  pick tick counts it can predict accurately — which means
  learning when fills actually happen.
- **Input to the re-quote decision.** The agent can learn
  policies like "if my own fill-probability prediction drops
  below 0.3, trigger re-quote."
- **UI transparency.** Every bet carries the model's confidence
  at placement time. Calibration plots become possible — does
  the agent's 70 % mean 70 % observed fill rate?

### 3. Risk / predicted-variance head

Second auxiliary head: for each pair placed, predict the
**locked-P&L distribution** as mean + log-variance. Gaussian
NLL loss on realised locked_pnl.

Gains:

- **Variance-aware sizing.** Stake scales inversely with
  predicted variance — bigger when confident, smaller when
  uncertain. Policy can learn to use this or not; the option
  exists.
- **Risk badges in the UI.** "Predicted locked +£5 ± £15" vs
  "+£1 ± £1.50" tells the operator at a glance which trades
  the agent is confident about.
- **Future: CVaR / risk-averse utility.** Once a distribution
  is predicted, swapping the expected-value objective for a
  tail-aware one is a coefficient change, not a rewrite.

### 4. UI surfaces

Once (2) and (3) are persisted per-bet:

- **Bet Explorer:** confidence badge (green > 70 %, amber
  40–70 %, red < 40 %) and a risk indicator next to the pair
  classification badge.
- **Model-detail page:** calibration plot (predicted vs
  observed fill probability in buckets), and a
  risk-vs-realised scatter.
- **Scoreboard:** add a "Calibration" column — mean absolute
  calibration error, so operators can see which models are
  actually self-aware vs just guessing.

## What success looks like

- Active re-quote lifts the Gen 1 fill rate from 14.5 % toward
  something like 50 %+ on the best models.
- Fill-probability predictions are calibrated within ±5 % of
  observed rates in each of four buckets (<25 %, 25–50 %,
  50–75 %, >75 %).
- Risk predictions correlate with realised locked-P&L variance
  (Spearman ρ > 0.3 within a run's bets).
- Top model's `arbs_naked` count trends *down* over
  generations, not because the agent stops trying but because
  it re-quotes intelligently.
- Bet Explorer calibration plots look like diagonals, not
  scatter clouds.

## What this folder does NOT cover

- **Cross-market arbitrage** (Win vs Place market). Separate
  plan. This work is single-market, single-runner only.
- **Partial-fill aware sizing.** If a passive fills 50 %, we
  currently treat the pair as completed for accounting.
  Refining that is out of scope here.
- **`ExchangeMatcher` changes.** Load-bearing — see CLAUDE.md
  "Order matching: single-price, no walking". Any change
  touching the matcher is rejected.

## Folder layout

```
plans/scalping-active-management/
  purpose.md              <- this file
  hard_constraints.md     <- non-negotiables
  master_todo.md          <- ordered session list
  progress.md             <- one entry per completed session
  lessons_learnt.md       <- append-only
  session_prompt.md       <- brief for the next session
```
