# EW Metadata Pipeline — Knock-ons

## ai-betfair: `BetEvent` in `ai_engine/bet_store.py`

`BetEvent` is ai-betfair's own serialisation of settled bets into the
JSONL dry-run log.  It already carries `each_way_divisor` but is
missing the four new fields added to `Bet` in this pipeline:

- `is_each_way`
- `number_of_places`
- `settlement_type`
- `effective_place_odds`

The `append_bet` and `load_bets` serialisation paths will also need
updating to round-trip these fields.

Without this change the dry-run log can't distinguish a straight-win
settlement from an EW place-only settlement — exactly the problem this
pipeline was created to solve in rl-betfair.
