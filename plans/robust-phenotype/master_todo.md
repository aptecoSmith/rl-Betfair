# robust-phenotype — master todo

Sequential by phase. Each phase ends in a verdict that determines
whether the next phase runs.

## Phase 0 — wait for prerequisites

Blocked on:

- E3 full cohort completes (or is stopped at gen 3) and writes
  verdict to `plans/EXPERIMENTS.md`
- E3+E4 combo probe completes (auto-fires after the cohort, ~2h)
  and writes verdict

Both verdicts inform the R1+R3+R4 baseline (what we're trying to
beat).

## Phase 1 — R1+R3+R4 probe (5×7d, ~2h GPU)

**Implementation (~3-4h scaffolding wall):**

1. R1 (Sortino selector):
   - `training_v2/cohort/runner.py::_score_one_agent` — new
     ``composite_score_mode=sortino`` branch
   - Formula: ``score = mean(day_pnl) - λ × downside_deviation``
     where λ defaults to 1.0 (additive form per
     hard_constraints §4)
   - CLI flag accepts the new mode value; existing modes unchanged

2. R3 (quadratic naked-loss):
   - `env/betfair_env.py::_compute_scalping_reward_terms` — new
     parameter ``naked_loss_quadratic_beta``, contribution
     ``-β × sum(min(0, p)² for p in naked_per_pair)``
   - Whitelist key in ``_REWARD_OVERRIDE_KEYS``
   - Init reads from ``reward_cfg.get(..., 0.0)`` — default
     byte-identical (hard_constraints §1)
   - Goes into the SHAPED channel (hard_constraints §2)

3. R4 (liquidity-floor open gate):
   - `env/betfair_env.py::_process_action` aggressive-path —
     after E3's spread-check, add depth check
   - New env kwarg ``opposite_side_depth_floor`` (default None)
   - Read post-filter ladder depth via the matcher's existing
     accessors (hard_constraints §5)
   - Refusal increments
     ``opens_refused_liquidity_floor`` counter (info dict)

4. Tests — TestSortinoSelector, TestQuadraticNakedLoss,
   TestLiquidityFloorGate. Each has default-off byte-id +
   mechanism + telemetry tests (hard_constraints §7).

5. Probe launcher
   `C:\tmp\probe_r1r3r4_robust_phenotype.ps1`. Same 5×7d shape as
   E1-E6 (raceconf gate, fc=120 in training, exclude same days,
   --reward-overrides combined). Cohort tag prefix
   `_predictor_SCALPING_probe_r1r3r4_*`.

**Verdict:** Per `purpose.md` success criteria —
- Cohort top-3 has worst-day ≥ −£30 AND mean pnl ≥ +£30/d, OR
- At least one agent matches the 850522b9 shape (worst ≥ −£20 at
  mean ≥ +£60).

## Phase 2 — Ablation probes (only if Phase 1 bites)

Per hard_constraints §9, before scaling:

- **2a.** R1 alone — same probe shape, just `composite_score_mode=
  sortino`.
- **2b.** R3 alone — `naked_loss_quadratic_beta=<from-1>`.
- **2c.** R4 alone — `opposite_side_depth_floor=<from-1>`.

If at least one ablation also bites at probe scale, mechanism is
clean enough for full cohort. Otherwise the bite at 1 is
emergent on the combination — interpret carefully and consider
whether the combo cohort is still the right move.

## Phase 3 — Full cohort (~28h GPU)

Recipe (matching the E3 full-cohort lineage):

- 12 agents × 8 generations × 13 train / 10 in-sample-eval days
- raceconf gate (race_confidence_threshold=0.50, pwin gate
  0.20/0.40, lay_price_max=20)
- fc=120 in training (deployment-realistic per
  `project_force_close_train_vs_deploy.md` memory)
- `composite_score_mode=sortino` (R1 active)
- `--early-stop-patience 3 --early-stop-min-gens 4`
- `--reward-overrides force_close_before_off_seconds=120
  naked_loss_quadratic_beta=<from-2> opposite_side_depth_floor=<from-2>
  close_feasibility_max_spread_pct=0.05`
  (stacks E3's open gate with R3+R4)

Held-out reeval on top-10 by `mean(day_pnl) - 1.0 × downside_dev`
(matches the Sortino selector). 7-day forward window, fc=0 AND
fc=120 (per the train-vs-deploy memory).

## Phase 4 — Verdict and follow-ons

Bands per `purpose.md`:

- **Strong:** mean ≥ +£50/d held-out, worst ≥ −£40, ≥ 4/5 prof,
  fc=120 + fc=0 both clearing → DEPLOY candidate
- **Modest:** ≥ +£20/d held-out, worst ≥ −£60 → next-iteration
  refinement
- **No improvement:** ≈ E3 full cohort numbers (this plan's null)
  → R2 + R5 follow-ons:
  - R2 (worst-day floor selector) replaces R1
  - R5 (velocity mask) layers on top of R4

## Out of scope

- Predictor retraining (orthogonal)
- Gate-config sweeps (race_confidence_threshold, pwin thresholds)
  — handled by separate predictor-tuning work
- Action-space changes — E4 already explored, deferred
