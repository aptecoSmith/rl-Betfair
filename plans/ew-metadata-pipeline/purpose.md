# EW Metadata Pipeline

## Problem

When each-way bets settle, the `Bet` dataclass only stores the final P&L and
a binary WON/LOST outcome.  It doesn't record:

- Whether the bet was each-way or straight win
- The EW divisor (1/4, 1/5) or number of places paid
- Which legs settled (win+place, place-only, neither)
- The effective place odds the place leg settled at

This means downstream consumers (evaluator bet logs, bet explorer UI, any
future analytics) can't distinguish a straight win from an EW place-only
settlement.  A user looking at "WON @ 4.30" has no way to know that only the
place leg paid at fractional odds.

### Alpha Capture example (2026-04-07 Pontefract 13:57)

Alpha Capture placed 3rd.  Three BACK bets showed:
- Lost: -44.72 (full stake lost — correct for a non-placed leg)
- Won: +40.19 @ 4.30  
- Won: +10.59 @ 4.40

Quick maths: `12.82 * (4.30 - 1) * 0.95 = 40.19` — that's **straight WIN
settlement**, not EW.  Root cause: `each_way_divisor` was `None` for all
April 7th data (the `PolledMarketSnapshots` table didn't have the column in
that day's backup).  Data has been deleted.

But even with correct divisors, the bet record wouldn't show the settlement
type or effective place odds.  That's what this plan fixes.

## Scope

1. **Bet dataclass** — add EW metadata fields
2. **settle_race()** — populate those fields during settlement
3. **EvaluationBetRecord** — propagate EW fields to evaluator output
4. **API layer** — include EW fields in the bet explorer endpoint response
5. **Tests** — verify EW metadata is set correctly for all settlement paths

## Layers touched

```
Bet (env/bet_manager.py)
  → settle_race() populates new fields
  → evaluator reads them into EvaluationBetRecord (training/evaluator.py)
  → API maps to ExplorerBet (api/routers/replay.py, api/models/)
  → Frontend consumes (bet-explorer-redesign plan, separate)
```

## Data model changes

### Bet dataclass (env/bet_manager.py)

```python
# New fields:
is_each_way: bool = False
each_way_divisor: float | None = None        # e.g. 4.0 for 1/4 odds
number_of_places: int | None = None          # e.g. 3
settlement_type: str = "standard"            # "standard" | "ew_winner" | "ew_placed" | "ew_unplaced"
effective_place_odds: float | None = None    # (price-1)/divisor + 1, for display
```

### EvaluationBetRecord (training/evaluator.py)

Same fields added.  The evaluator copies them from `Bet` when building
records in `_evaluate_day()`.

### ExplorerBet (API response model)

Same fields surfaced in the JSON response.  The bet-explorer-redesign plan
consumes these.

## settle_race() changes

Inside the EW branch (`if ew and in_winners:`), after computing P&L:

```python
bet.is_each_way = True
bet.each_way_divisor = each_way_divisor
bet.number_of_places = number_of_places  # needs new param
bet.effective_place_odds = (price - 1.0) / each_way_divisor + 1.0

if is_winner:
    bet.settlement_type = "ew_winner"
elif is_placed:
    bet.settlement_type = "ew_placed"
```

For the unplaced EW branch (falls into the non-EW else block currently):
```python
if ew and not in_winners:
    bet.is_each_way = True
    bet.each_way_divisor = each_way_divisor
    bet.number_of_places = number_of_places
    bet.settlement_type = "ew_unplaced"
    # P&L = -matched_stake (both legs lose), same as current
```

## settle_race() signature

Add `number_of_places: int | None = None` parameter.  The caller in
`betfair_env.py` already has `race.number_of_each_way_places` available.

## Test plan

1. Existing `TestEachWaySettlementCorrected` — extend to assert new fields
2. New test: verify `settlement_type` for winner, placed, unplaced
3. New test: verify `effective_place_odds` calculation
4. New test: verify non-EW bets have `is_each_way=False`, `settlement_type="standard"`
5. Integration test: full episode with EW race, check evaluator bet record has fields
