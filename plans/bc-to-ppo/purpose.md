# BC → PPO — purpose

**Created 2026-05-31.** Successor to `plans/bc-getting-it-right/` (which
PASSED: BC produces a selective, maturation-aware policy). This plan turns
that BC warm-start into a **profitable, deployable** scalping policy via
reward-aware fine-tuning. Read this, then `hard_constraints.md` (locked
decisions), then `master_todo.md` (the steps + status).

## One-paragraph thesis

BC works — the policy's `mature_prob_head` reaches **holdout maturation AUC
0.745** (≈ the LightGBM 0.759 ceiling) and a threshold-gated rollout opens
selectively at **14–15% mat% (3.5× the prior BC), locked positive**. But BC
only *imitates* the ~breakeven oracle, so day_pnl is still negative. The
profit comes from **selectivity beyond the oracle**: open only the
high-fill-probability pairs so the locked edge exceeds the force-close toll.
The signal to do this is present (the 0.745-AUC head). PPO with a maturation
reward + an open-cost toll is the mechanism that converts "ranks maturation
well" into "opens selectively for profit."

## The decisive insight (steers the whole plan)

The deployment problem is the **~79% force-close rate**, NOT capital.
Diagnostics (`plans/bc-getting-it-right/findings.md` Step E + spend probe):
- Removing the per-race budget made P&L *worse* (−£428 vs −£182/3d) — it just
  opens more pairs that mostly force-close. Budget is not the lever.
- The lever is **selectivity**: open fewer, higher-`mature_prob` pairs so
  fc% falls, the locked edge clears the toll, and capital-at-risk shrinks.
- The mature_prob head (0.745 AUC) is exactly the tool to drive this; PPO is
  how the policy learns to act on it under a real budget.

## Where we are at plan creation

- **BC policy:** `plans/bc-getting-it-right/_scripts/stepB_alltrain_wd3e-3.pt`
  (holdout maturation AUC 0.745, input-norm ON, full obs 2254-d).
- **Safety barrier landed + tested:** `force_close_max_deviation_pct` (config
  default 0.50) caps how far past LTP a force-close may cross; no-LTP closes
  are refused. The relaxed-matcher "fill at any price" hole (with
  `max_lay_price: null`) is closed. This MUST be on during PPO (else the
  policy learns to exploit the free overdraft-at-any-price).
- **Rollout mechanics proven:** greedy-by-mature_prob rollout, fc=120 +
  close_walk=10, deployment economics characterised, deployment-honest now
  that the barrier is on.

## The plan in one line

**Step 1** map mat%-vs-budget (pick the deployment operating point) →
**Step 2** build the BC→PPO warm-start (actor opens + mature head selects) →
**Step 3** BC→PPO canary (maturation reward + open_cost, select on LOCKED;
the profit step) → **Step 4** scale to a GA cohort + honest holdout +
dry-run (OPERATOR-GATED — come back before starting).

## Success bar (what "profitable BC→PPO" means)

A single-config canary, warm-started from the BC policy, fully-hedged
(fc=120, close_walk=10, barrier on), at the deployment budget, drives the
**holdout LOCKED P&L cleanly positive** with mat% materially above the BC's
14–15% — by being MORE selective than the oracle (lower fc%), not by opening
more. Only then is the GA-cohort scale-up (Step 4) warranted.
