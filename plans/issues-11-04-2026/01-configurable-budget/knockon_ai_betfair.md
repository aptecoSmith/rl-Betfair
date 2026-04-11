# Downstream Knock-on — `ai-betfair` Changes

This file audits what `ai-betfair` needs to do in response to the
configurable budget work in rl-betfair. The actual implementation
lives in the `ai-betfair` repo.

---

## 1. Checkpoint metadata should carry training budget

### What changes in rl-betfair

Training plans and evaluation records now include `starting_budget`.
Models trained at £10 vs £100 are distinguishable.

### What `ai-betfair` must do

- **Read training budget from checkpoint metadata** when loading a
  model for live inference.
- **Display "trained at £X/race"** on the live dashboard alongside
  the live budget setting, so the operator can see at a glance
  whether there's a mismatch.
- **Log the ratio `training_budget / live_budget`** as a sanity
  metric. A ratio far from 1.0 (e.g. trained at £100, running at
  £5) is a distribution-shift risk — the policy learned stake
  fractions in a different absolute-£ regime.

### Structural impact

None. The policy's forward pass is budget-agnostic — actions are
stake fractions relative to available budget, not absolute amounts.
The live wrapper already applies its own budget independently.

### Cost

Trivial. Display + logging only. No inference-path changes.

---

## 2. Live dashboard P&L display

### What changes in rl-betfair

Scoreboard and model detail now show percentage return alongside
raw P&L.

### What `ai-betfair` should consider

- **Show live session P&L as both raw £ and % of budget** on the
  recommendations page and day summary.
- This is a display-only change. The live budget is already known
  to `ai-betfair`'s own config.

### Cost

Trivial. One computed field in the dashboard.

---

## Summary

| rl-betfair change | `ai-betfair` impact | Cost |
|---|---|---|
| Per-plan budget in checkpoint | Read + display training budget | Trivial |
| % return display | Mirror on live dashboard | Trivial |

Total: effectively zero structural work. Both items are display
enhancements that can ride along with any other `ai-betfair` session.
