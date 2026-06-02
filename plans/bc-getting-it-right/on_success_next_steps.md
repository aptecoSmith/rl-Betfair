# BC getting-it-right — what we do next IF it works

Read this only after the success bar (`hard_constraints.md §8`) is met:
the policy's held-out maturation AUC ≈ 0.70+ (approaching LightGBM's 0.76)
AND, at some confidence threshold, a fully-hedged holdout rollout shows
mat% well above the ~1% baseline with locked positive — a clearly better
warm-start than the imitation-first BC (4% mat%, −£1513/7d).

A working BC gives us a policy that **opens the right trades** but only
**imitates** the hindsight oracle — so its ceiling is ~breakeven. The next
steps turn that into a profitable, deployable strategy.

## Step 1 — BC→PPO (reward-aware fine-tune). The profit step.

This is `plans/imitation-first/` Step 2 proper. Both prerequisites are
already built + tested.

- **Warm-start** PPO from the working BC weights (don't start cold).
- Turn on the **maturation reward** (`maturation_reward_mode`, wired) — it
  pays the policy ONLY for trades that actually mature (or are
  closed at a profit), nothing for the ones that get stuck and bail. Plus
  the **`open_cost` toll** (existing) so every speculative open that
  doesn't pay off costs something. Plus MTM densification.
- **Why it beats BC alone:** BC can only copy the oracle (~breakeven).
  The reward pushes the policy to be MORE selective than the oracle — open
  only the high-confidence maturations, decline the rest — which is where
  net-positive comes from. Step 1 (LightGBM 0.76, 2.45× top-decile lift)
  says the signal to do this is present.
- **Pins (deploy honesty):** `force_close_before_off_seconds=120`,
  `close_walk_ticks=10`. **Select on LOCKED P&L**, never day_pnl (naked is
  zero-EV variance — memory `feedback_sort_top_by_locked_not_total`).
- **Start with a single-config canary** (standalone `DiscretePPOTrainer`,
  like `_step1/bc_fullnet_canary.py`) before any multi-agent horde.

## Step 2 — Scale to a GA cohort (if the canary promises).

- **Wire `input_norm` through the cohort runner/worker** — the standalone
  scripts don't scale to a GA. Do it CAREFULLY: grep the flag through
  `worker.py` + policy construction + env to confirm it reaches every
  place (memory `feedback_audit_launch_wiring` — a half-wired flag burned
  3 cohort restarts before).
- Run the `plans/ga-recipe-search/` vehicle: evolve the genes
  (`arb_spread_target_lock_pct`, `open_cost`, `mature_prob_open_threshold`,
  `mature_prob_loss_weight`, …), **select on `locked_per_std`**.
- Eval on the reserved 7 holdout days, ONCE, fully-hedged.

## Step 3 — Honest held-out validation (the deploy-reality check).

- Report the held-out number fully-hedged (`close_walk` ON), once, no
  cherry-picking (memory `feedback_always_eval_holdout`).
- **Train-vs-deploy asymmetry** (memory
  `project_force_close_train_vs_deploy`): the env's force-close overdraft
  means a held-out +£X/day is an UPPER BOUND — live P&L depends on
  Betfair's actual order-refusal rate near the off. Report both the
  optimistic (overdraft-allowed) and the conservative number.

## Step 4 — Deploy to dry-run (live, no money).

- Connect the trained policy to `ai-betfair` (the live-inference project —
  memory `project_ai_betfair`) for dry-run recommendations on live Betfair
  data. The matcher (`env/exchange_matcher.py`) is kept vendorable into
  that project for exactly this.
- Watch live behaviour vs backtest (fill rates, refusal rates, the
  near-off thin-book reality). Only after dry-run matches expectations
  does real money get discussed.

## Parallel / queued ideas (don't block the above)

- **Two-cohort diversification** (memory
  `project_two_cohort_diversification`): scalping strategies that trade
  different price regions, deployed together.
- **Race-outcome + price-mover two-model architecture** (memory
  `project_form_data_rl_integration`): feed a form-based win/place
  predictor alongside the price-direction predictor into the RL obs.

## If BC does NOT meet the bar

Do NOT proceed to PPO (it can't fix a signal the policy can't represent).
The likely finding is one of: (a) the pooled-LSTM actor bottleneck loses
signal a tree keeps → feed the per-runner feature slice to the head
directly (as the direction-head redesign did); or (b) the features are
insufficient → the unlock is richer market data (deeper book / the
StreamRecorder 10-level work). Write it up; that's a real, cheap negative.
