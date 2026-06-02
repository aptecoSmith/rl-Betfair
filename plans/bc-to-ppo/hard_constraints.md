# BC → PPO — hard constraints (locked decisions)

Settled. A fresh session inherits these; don't relitigate.

## §1 — The metric to OPTIMISE is held-out LOCKED P&L, fully hedged.
- Select / judge on **LOCKED** P&L, never day_pnl. Naked is ~zero-EV
  variance; day_pnl-top surfaces naked-lucky agents that don't generalise
  (memory `feedback_sort_top_by_locked_not_total`).
- Secondary deployment-risk metric: naked variance per leg (`σ_naked_leg`)
  and the force-close rate. Lower fc% IS the goal (the toll, the capital,
  and the deployment risk all scale with it).

## §2 — Data split (inherited; no leak).
- Train = 40 days Apr 6 → May 17; BC-val = May 18/19; Holdout = the reserved
  7 (May 20,21,22,25,27,28,29). NEVER train/select/threshold-tune on holdout.
- Eval the holdout ONCE at the end, fully hedged, no cherry-picking
  (memory `feedback_always_eval_holdout`).

## §3 — The force-close safety barrier MUST be ON in training + eval.
- `force_close_max_deviation_pct` (config 0.50) is on. PPO trains against it
  too — otherwise the policy learns to exploit the free overdraft-at-any-
  price (now fixed) and we'd retrain to deploy. No-LTP closes are refused.

## §4 — Deployment pins.
- `force_close_before_off_seconds=120`, `close_walk_ticks=10`,
  `input_norm=True`, the force-close barrier, and the deployment budget
  chosen in Step 1. These are pinned across the canary and the cohort.

## §5 — Warm-start, don't start cold.
- PPO initialises from the BC policy weights
  (`plans/bc-getting-it-right/_scripts/stepB_alltrain_wd3e-3.pt`). The
  `mature_prob_head` is already supervised (AUC 0.745); BC the actor too
  (Step 2) so the warm-start opens selectively from step 0.

## §6 — Selectivity is the lever, not capital.
- Train at the DEPLOYMENT budget (Step 1's pick). The reward shape must push
  the policy to open FEWER, higher-fill-prob pairs (lower fc%), not more.
  `maturation_reward_mode` (pays only matured/profit-closed) + `open_cost`
  toll are the wired mechanisms. Do NOT chase higher mat% by raising budget.

## §7 — Train/deploy asymmetry: held-out is an UPPER BOUND.
- The env's force-close overdraft assumes capital is available AND Betfair
  fills the close in the thin near-off book. Live P&L depends on the real
  near-off refusal rate. Report BOTH the optimistic (overdraft-allowed) and
  a conservative number (memory `project_force_close_train_vs_deploy`).

## §8 — Iterate standalone; wire the cohort only once the recipe is proven.
- Drive `DiscretePPOTrainer` standalone for the canary (like the
  `bc-getting-it-right/_scripts`). The GA cohort (Step 4) needs `input_norm`
  wired through `cohort/worker.py` — GREP every callsite (worker + policy +
  env) to confirm the flag reaches construction (memory
  `feedback_audit_launch_wiring` — a half-wired flag burned 3 restarts).

## §9 — Step 4 (GA cohort + dry-run) is OPERATOR-GATED.
- The canary (Steps 1–3) runs autonomously. Do NOT start the GA cohort
  scale-up or any dry-run/live connection without operator sign-off
  (operator directive 2026-05-31).

## §10 — Honesty / discipline.
- Negative results are first-class: if the canary can't drive LOCKED
  positive, that's a real finding (the oracle's edge may be too thin at
  deployment fill rates → richer data / deeper book), not a reason to keep
  tuning. Report against the holdout split + LOCKED P&L.
