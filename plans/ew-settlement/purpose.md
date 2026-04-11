# Purpose — Each-Way Settlement Fix

## Why this work exists

The simulator does not model each-way (EW) betting correctly. EW
markets are treated identically to Win markets throughout the entire
pipeline: stake calculation, settlement, and reward signal. This
produces incorrect training rewards for every EW race, teaching the
agent a distorted view of EW market economics.

Three things are wrong:

1. **Stake is not doubled.** A real EW bet at stake S places two
   equal sub-bets: S/2 on the win leg and S/2 on the place leg.
   The system treats S as a single bet. The agent therefore
   underestimates the capital at risk on EW markets by 2x.

2. **Place leg pays at full win odds instead of fractional odds.**
   When a runner *places but does not win*, the place leg should
   pay at `(win_odds - 1) / divisor + 1`. The system currently
   pays placed runners at full win odds — the same as a winner.

3. **Winner does not collect the place leg.** When a runner *wins*
   an EW race, the bettor collects on *both* legs: full win odds
   on the win leg, plus place fraction on the place leg. The system
   currently pays the winner once at win odds only.

The net effect: the reward signal for EW races is wrong in both
direction and magnitude, depending on whether the bet's selection
won, placed, or was unplaced.

## How Betfair EW markets work

The BetfairPoller captures odds from the **Win market** for all
market types. The captured prices are always win odds. EW terms
(divisor, number of places) are captured separately and stored per
market.

An EW bet of stake S at decimal win odds W with divisor D:

| Outcome   | Win leg (stake S/2)              | Place leg (stake S/2)                       | Net P&L                                                  |
|-----------|----------------------------------|---------------------------------------------|----------------------------------------------------------|
| Wins      | +(S/2) x (W - 1) x (1 - comm)   | +(S/2) x ((W-1)/D) x (1 - comm)            | Profit on both legs, minus commission                    |
| Places    | -(S/2)                           | +(S/2) x ((W-1)/D) x (1 - comm)            | Lose win leg, profit on place leg                        |
| Unplaced  | -(S/2)                           | -(S/2)                                      | Lose both legs = -S                                      |

Place odds (decimal) = `(W - 1) / D + 1`

Example: £10 EW at 10.0, 1/4 odds, horse places but doesn't win:
- Win leg: lose £5
- Place leg: profit = £5 x ((10-1)/4) x 0.95 = £5 x 2.25 x 0.95 = £10.69
- Net: +£5.69

Same example if horse wins:
- Win leg: profit = £5 x 9 x 0.95 = £42.75
- Place leg: profit = £5 x 2.25 x 0.95 = £10.69
- Net: +£53.44

## What the system currently does (wrong)

Using the same example (£10 EW at 10.0, horse places):
- Treats it as a single £10 bet
- Pays at full odds: £10 x (10-1) x 0.95 = £85.50

That's a +£85.50 reward instead of +£5.69. Off by 15x.

If the horse wins:
- Pays once at full odds: £85.50
- Should be: +£53.44 (both legs)

Even the winner case is wrong, and in the opposite direction.

## What the incorrect comment says

`episode_builder.py` and `feature_engineer.py` contain comments
stating "Betfair EACH_WAY markets already quote the place-adjusted
price." This is **wrong**. The BetfairPoller queries
`RUNNER_EXCHANGE_PRICES_BEST` which returns Win market prices for
all market types. The EW divisor is captured separately via
`EachWayHelper` in the poller. The odds are win odds; the place
fraction must be derived.

## What success looks like

- `BetManager.settle_race()` produces correct P&L for EW markets:
  doubled stake, place fraction for placed runners, both legs paid
  for winners.
- Training reward on EW races reflects the corrected settlement.
- The agent can learn different strategies for EW vs Win markets
  because the reward signal is now honest.
- Incorrect comments in `episode_builder.py` and
  `feature_engineer.py` are corrected.
- Existing Win-market settlement is completely unaffected.

## What this folder does NOT cover

- Live inference changes in `ai-betfair` (see
  `ai-betfair/plans/ew-settlement/`).
- Changes to the BetfairPoller or data capture pipeline.
- Adding a separate Place market data feed (out of scope — we
  derive place odds from win odds + divisor, which is standard
  UK EW practice).

## Folder layout

```
plans/ew-settlement/
  purpose.md              <- this file
  hard_constraints.md     <- non-negotiables
  master_todo.md          <- ordered session list
  progress.md             <- one entry per completed session
  lessons_learnt.md       <- surprising findings, append-only
```
