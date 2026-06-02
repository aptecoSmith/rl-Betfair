# Imitation-first — hard constraints (locked decisions)

These are settled. Don't relitigate; a fresh session inherits them.

## §1 — Data split (deployment-realistic, no leak)
- **Train = 42 days, Apr 6 → May 19** (every processed day in range).
- **Holdout = latest 7: 2026-05-20, 05-21, 05-22, 05-25, 05-27, 05-28,
  05-29** (verified 75-91 markets each, ~570 races). NEVER train on these.
- **Explicit day lists only** (`--training-days-explicit` /
  `--cohort-eval-days`); NEVER `select_days(n)` (the n_days≥14 Apr-30 leak
  foot-gun). Assert train ∩ holdout = ∅ at every run.
- Train strictly before holdout = mimics live (learn past, trade future).
- We have 49 processed days total (`data/processed/YYYY-MM-DD.parquet`).

## §2 — Full obs (un-starve the input)
- **Drop `--predictor-lean-obs`** → 143-feature-per-runner full obs (was
  23). KEEP the predictor bundle + `--direction-head-manifest
  models/direction_head/sweep_c11` (free features).
- Normalization handled (feature_engineer log-norms volumes/sizes/depths;
  policy per-runner LayerNorm) — but run the value-domain audit (master_todo
  Step 1a) before trusting it: a stale/leaky/outlier dim lean-obs never
  exposed is the residual risk (memory `feedback_feature_engineering_diagnostics`).
- Full obs = new input dim → fresh init, no cross-load from lean weights.
- **Oracle + feature caches MUST be rebuilt at full-obs dim across all 42
  train days + 7 holdout days** before any run (preflight enforces it).

## §3 — Held-out discipline (the E7 trap)
- Select / judge on **LOCKED P&L** (via `--composite-score-mode
  locked_per_std` = locked ÷ naked-σ for any GA/selection), NEVER day_pnl
  (naked is zero-EV variance) and NEVER in-sample. The DEFAULT
  `total_reward` mode is the trap — pin it explicitly.
- mat% is a SECONDARY diagnostic, not the selector.
- Holdout number reported ONCE, fully-hedged (close_walk ON), no
  cherry-picking best-of-N.
- Monitor tripwire on the holdout each gen for any iterative method.

## §4 — Reward shape (Step 2 only; operator-defined 2026-05-30)
- Positive raw channel = **naturally-matured locked P&L + agent-closes
  AT A PROFIT** ("matured naturally and agent-closed-at-a-profit on the
  same side"). Agent-close-at-loss, force-close, stop-close, naked → 0 in
  the positive channel. Pure helper `maturation_only_reward` (built+tested).
- **`open_cost` toll** is mandatory alongside it — without it the
  matured-only reward degenerates to "open maximally" (non-maturing opens
  are free lottery tickets). The toll makes a non-maturing open net-negative.
- **`per_pair_reward_at_resolution` ON** — pay the matured pair's locked
  P&L at the maturation tick (densify; outcome-independent once hedged).
- **MTM shaping ON** (`mark_to_market_weight`) — per-tick density between
  open and maturation.
- BC stays the open-decision signal. Densification stack = BC (open) +
  per-pair-at-resolution (maturation payoff) + MTM (in between).

## §5 — Pins (never evolve these)
- `force_close_before_off_seconds = 120` (safety rail; if it's a gene the
  GA rediscovers the fc=0 naked-luck trap: +£287 in-sample → −£175 held-out).
- `close_walk_ticks = 10` (deploy-honest fully-hedged number; Round W
  showed it's a variance tool, keep it on for honesty).
- BC stays PER-AGENT, never shared (sharing collapses GA diversity).

## §6 — Architecture
- Default LSTM h256. Transformer + TimeLSTM are built in v2 but UNVERIFIED
  on held-out P&L — second-order vs data/signal/reward. Optional GA gene
  later; do NOT lead with an architecture change.

## §7 — The hindsight caveat (read before interpreting any result)
The oracle labels using the FUTURE. The policy only sees decision-time
features. So:
- **Step 0** (oracle's own P&L) is an UPPER BOUND, not achievable by any
  causal policy — it tells us the opportunity exists, not that it's reachable.
- **Step 1** (BC) can only imitate what's PREDICTABLE from the obs. A flat
  Step 1 means "not learnable from current features," which — combined
  with the direction head's 0.70 AUC — would point back at richer data
  (deeper book) as the real unlock. That's a legitimate, cheap negative.

## §8b — CRITICAL (2026-05-30): the current oracle labels "spread placeable", NOT "will mature"
`scan_day` (verified, lines 252-321) labels a (tick, runner) as a
profitable scalp purely on current-tick prices: a back price + a passive
lay `min_ticks` away whose `expected_locked_pnl > 0` **if both fill**. It
does **NOT forward-walk to confirm the passive actually fills.** So the
oracle teaches "open whenever a profitable SPREAD exists," not "open a
pair that will MATURE." Since ~81% of opens force-close (passive never
fills), this is almost certainly the core campaign flaw — BC/PPO
imitating this oracle learns to over-open exactly as our agents do.

**Implication for the plan:**
- **Step 0 diagnoses it:** run the CURRENT oracle's labels through the env
  on holdout. If it force-closes / bleeds → the oracle itself is the
  problem, not the policy.
- **Likely prerequisite (Step 0.5): redesign the oracle to be maturation-
  conditioned** — forward-walk each candidate open, label positive ONLY
  if the passive fills before fc=120 under the real future path + the env
  matcher (the close/hold augmentation already forward-walks, so the
  machinery exists). This is a true "will-mature" hindsight oracle.
- Until then, "the oracle proves scalps exist" must be read narrowly: it
  proves profitable SPREADS are placeable — necessary, not sufficient.

## §8 — The oracle is DETERMINISTIC, not trained (don't "retrain" it)
`arb_oracle.py::scan_day` is a deterministic hindsight scan: it loads ONE
day, walks the pre-race ticks, and labels (tick, runner, side) where a
pair WOULD have matured profitably using fixed math (commission,
`min_arb_ticks_for_profit`, the actual forward price path). No weights,
no learning, no cross-day fitting. So:
- **"More data" = scan more DAYS**, not retrain. Generate labels for all
  42 train days (only a few were ever scanned). The benefit flows to the
  IMITATOR (BC gets more examples), not the oracle.
- **Holdout-safe by construction:** each day is labeled independently, so
  scanning the 7 holdout days (Step 0, eval only) cannot contaminate the
  42 train-day labels BC learns from (Step 1). No leak mechanism exists.
- **Oracle PARAMETERS** (commission, profit floor / `min_arb_ticks`,
  reachability window) define "what counts as a profitable scalp." If
  ever tuned: **train days only, NEVER against holdout outcomes** (that
  would leak the labeler), and DEFER — run Step 0/1 on the existing
  definition first; only revisit if the oracle's edge proves too thin.
- **Scan must inject predictor outputs** (`use_race_outcome_predictor`,
  `use_direction_predictor`) so cached obs match what BC + the policy see;
  else the `dir_*` columns are zero placeholders (oracle's 2026-05-24 note).
