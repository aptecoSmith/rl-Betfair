# Downstream Knock-on — `ai-betfair` Changes

This file audits what `ai-betfair` needs to do in response to the
market type filter work in rl-betfair. The actual implementation
lives in the `ai-betfair` repo.

---

## 1. Enforce market type filter at inference time

### What changes in rl-betfair

Models now carry a `market_type_filter` gene in their hyperparameters:
`WIN`, `EACH_WAY`, `BOTH`, or `FREE_CHOICE`.

### What `ai-betfair` must do

- **Read `market_type_filter` from the loaded model's
  `record.hyperparameters`** during model loading.
- **Filter live markets accordingly:**
  - `WIN` → only subscribe to / bid on WIN markets.
  - `EACH_WAY` → only subscribe to / bid on EACH_WAY markets.
  - `BOTH` → subscribe to both but present all to the model
    (current behaviour, equivalent to FREE_CHOICE at inference).
  - `FREE_CHOICE` → same as BOTH at inference (model was trained
    on everything and learned to self-select).
- **Display the filter on the recommendations page** so the operator
  knows which markets the model will engage with.

### Operator override: radio buttons on go-live page

The operator should be able to override the model's trained filter
at inference time. Use case: a BOTH-trained model that the operator
wants to restrict to WIN-only for a particular session.

- **Radio buttons on the go-live / model selection page:**
  - `WIN only`
  - `EW only`
  - `Both (all markets)`
  - `Model default` (uses the trained filter) ← **default selection**
- **Show the model's trained filter** next to the radio buttons so
  the operator can see what the default is.
- **Warning if overriding:** If the operator selects a filter
  different from the model's training filter, show a subtle warning:
  "This model was trained on [X] markets. Running on [Y] may
  produce unexpected results."

### Cost

Small. One config read + market filter in the subscription logic +
radio buttons on the go-live page. ~1 session.

---

## 2. Dashboard display

### What changes in rl-betfair

Scoreboard shows a market type badge per model (WIN/EW/BOTH/FREE).

### What `ai-betfair` should do

- Show the active market type filter on the live dashboard header:
  "Running: Model abc123 (WIN only)" or "Running: Model abc123
  (all markets, trained as FREE_CHOICE)".
- If the operator overrode the filter, show both: "Trained: BOTH,
  Running: WIN only".

### Cost

Trivial. Display only.

---

## Summary

| rl-betfair change | `ai-betfair` impact | Cost |
|---|---|---|
| `market_type_filter` gene | Filter live market subscriptions | Small (~1 session) |
| Filter badge on scoreboard | Dashboard display of active filter | Trivial |
| — | Go-live radio buttons for override | Part of the session above |
