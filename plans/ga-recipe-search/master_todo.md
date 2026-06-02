# GA recipe search — master todo

Status legend: [x] done · [~] partial/needs-GPU · [ ] pending · [GATE] blocked on Round T

## Prerequisites (do before launch — no GPU needed except where noted)

### §C Resume / checkpoint — [x] CODE DONE, [~] integration test pending GPU
- [x] `_write_resume_state` / `_load_resume_state` / `_truncate_scoreboard_at_generation`
  helpers in `training_v2/cohort/runner.py` (after `_make_genetic_event`).
- [x] `run_cohort(resume_from=...)` param; gen-0-vs-resume branch; loop runs
  `range(start_generation, n_generations)`; checkpoint written at the START of
  each gen (atomic tmp+replace); scoreboard truncated to drop the interrupted
  gen's stale rows then appended; monitor_metrics preserved on resume.
- [x] `--resume-from DIR` CLI flag + main `run_cohort(...)` passthrough.
- [x] Unit tests: `tests/test_v2_ga_resume.py` (8 passing) — round-trip,
  deterministic RNG restoration, scoreboard truncation edge cases.
- [~] **Integration test (needs GPU, run when free):** `--n-agents 2
  --generations 2 --device cuda` minimal cell; kill after gen 1; `--resume-from`
  → confirm it skips gen 0-1, re-runs gen 1 from checkpointed cohort with gen-0
  elites as parents (not fresh-init), no duplicate scoreboard rows. Round T has
  the GPU now; run this in the gap before the GA launch.

### §R Reward shape — REQUIRES ENV CODE (corrected 2026-05-30; my earlier "config-only" claim was WRONG)
**Code-read correction:** the raw reward is `race_pnl` =
`scalping_locked_pnl + scalping_closed_pnl + scalping_force_closed_pnl +
sum(naked_per_pair)` (env `_compute_scalping_reward_terms`, line 324) — the
WHOLE-race cashflow across EVERY outcome, plus a terminal `day_pnl` bonus, plus
shaping. Naked *winners* get a 95% shaped clip; naked *losses*, force-close
cost, and closed cash are all in raw at full value. NO existing knob isolates
"matured only" — `open_cost`/`matured_arb_bonus` only ADD shaped terms;
`naked_loss_scale` only scales the naked-loss side. So the operator's reward
needs a real env change, NOT config.

**Operator's reward definition (confirmed 2026-05-30):** positive raw channel =
naturally-matured locked P&L + agent-closes-AT-A-PROFIT ("matured naturally and
agent-closed-as-a-profit on the same side"); agent-close-at-loss, force-close,
stop-close, naked all → 0 in raw. The `open_cost` toll (shaped) prevents the
spam degeneracy (non-maturing opens earn 0 raw, so without a toll the agent
opens maximally).

- [x] Pure helper `env/betfair_env.py::maturation_only_reward(pair_outcomes)` +
  7 unit tests (`tests/test_maturation_reward.py`, green). matured→locked,
  agent_closed→max(0,covered_cash), force/stop/naked→0.
- [ ] **`maturation_reward_mode` flag** — plumb like `close_walk_ticks` (read
  from reward_overrides/constraints, add to `_REWARD_OVERRIDE_KEYS`). Default
  OFF = byte-identical.
- [ ] **Settle wiring** — in `_settle_current_race`, gather `pair_outcomes`
  (one per pair, classified matured/agent_closed/force/stop/naked with its
  `locked` floor = min(win,lose) and `covered_cash`). When the mode is on, set
  `race_reward_pnl = maturation_only_reward(pair_outcomes)` (override at the
  line ~5740 `race_reward_pnl = ...` assignment). Self-contained accumulator —
  do NOT disturb the existing `scalping_locked_pnl`/`closed_pnl` accounting or
  the `raw + shaped ≈ total` invariant. Shaped channel (open_cost toll,
  matured_arb_bonus) unchanged → still applies on top.
- [ ] **Integration test (needs GPU):** a scalping cell with
  `maturation_reward_mode` ON — assert raw reward == sum of matured locked +
  agent-close profits, and that force/naked races contribute 0 raw. Run when
  GPU frees (gated on Round T anyway).
- Still evolve `open_cost` + `matured_arb_bonus_weight`; pin
  `matured_arb_expected_random` ≈ 0.006 matured/race (M6: 5% mat%, ~4.5
  matured/agent over ~700 eval races) — recalibrate at launch.

## Staged rollout — validate → smoke → CANARY → GA horde

Do NOT jump to the multi-agent GA. Each stage de-risks the next; a setup
bug caught at step 1 costs minutes, caught at the GA costs a week.

### Step 1 — Validate (hours, CPU/IO-bound, no GPU contention)
- [ ] **§V value-domain audit of FULL obs** — build one env at full obs
  for a training day; for a named runner, `head(30)` the per-runner obs
  block + a z-score min/max table across all 143 dims. Catch any
  unnormalized / leaky / degenerate dim lean-obs never exposed (the
  memory lesson: shape checks pass, `head(30)` catches the ~90σ bug).
- [ ] **Confirm 42/7 split wiring** — explicit `--training-days-explicit`
  (Apr 6→May 19) + `--cohort-eval-days` (May 20,21,22,25,27,28,29);
  assert train ∩ holdout = ∅; NO `select_days(n)`.
- [ ] **Rebuild oracle + feature caches at FULL-obs dim** across all 42
  training days (the preflight check requires it; pre-build to avoid a
  30s-in crash). This is the long pole of step 1.

### Step 2 — Tiny smoke (minutes-1h GPU)
- [ ] 2-4 agents × 2 gens, full obs, 42/7 split, BC on, SHORT. Goal:
  does it RUN end-to-end — dims right, cache hits, no NaN/scale blowup,
  scoreboard + monitor rows written, resume checkpoint written. Not a
  performance read; a plumbing read.
- [ ] Also the §R `maturation_reward_mode` integration assertion here
  (raw == matured-locked + agent-close-profit; force/naked → 0 raw).

### Step 3 — CANARY (operator-requested 2026-05-30): ONE config, full scale
**Before letting loose on the horde, run ONE promising config at full
obs + full 42-day train, to real length, on the 7-day holdout.** Purpose:
surface compute/scale/reward-attribution/overfit issues on a single
config, not multiplied across 64 agents.
- [ ] Config: one sensible arch (default LSTM h256), full obs, BC ON +
  substantial (sparsity bridge, §S), maturation reward, MTM on, fc=120 +
  close_walk=10 pinned, pwin band (N4), `locked_per_std` selection.
  1 config, optionally 2-3 seeds for variance. Multi-gen to real length.
- [ ] **Measures:** (a) wall-time per agent at this scale → the GA
  compute budget; (b) does held-out locked/mat% move off the ~5% floor
  (does it LEARN); (c) reward-attribution sanity (gradient non-zero,
  value_loss bounded — the §S sparsity check at scale); (d) overfit
  signature (in-sample vs holdout gap, gen-over-gen).
- [ ] **GATE (= hard_constraints §L):** canary learns on holdout →
  launch GA. Canary flat/blowup → STOP, don't scale; that's the strong
  negative (full data + obs + oracle BC still can't reach it).

### Step 4 — GA horde (only if the canary passes)
- [ ] `run_ga_search.sh`: ~64 agents × N gens, `--device cuda`,
  `--batched`, full obs, 42/7 split, BC, maturation reward.
- [ ] **Compute control for 42 days:** a rollout is one day, so an epoch
  = 42 episodes/agent; use `--rotating-eval-sample` for the holdout-eval
  cost, and consider a rotating TRAINING-day sample per episode (train on
  K-of-42, not all 42 every epoch) if wall-time from the canary is too
  high. Size N gens / agents to the canary's measured per-agent time.
- [ ] **Selection:** pin `--composite-score-mode locked_per_std` (NOT the
  default `total_reward` = E7 trap). Confirm formula at `_composite_score`
  ~line 237.
- [ ] **PINS (CLI, not genes):** `force_close_before_off_seconds=120`,
  `close_walk_ticks=10`, `matured_arb_expected_random=<base ≈0.006/race,
  recalibrate>`, pwin band, full predictor stack + `--direction-head-
  manifest sweep_c11`. **Run FULL obs — OMIT `--predictor-lean-obs`**
  (its presence in all 18 prior recipes was the lean-obs bottleneck, §O).
- [ ] **EVOLVE (`--enable-gene`):** `open_cost`, `matured_arb_bonus_weight`,
  `arb_spread_target_lock_pct`, `mature_prob_loss_weight`, `bc_pretrain_steps`
  (incl. 0, mixes BC/no-BC; BC stays per-agent §P.3), + core PPO genes.
  Do NOT enable `force_close_*` / `close_walk_ticks` (§P).
- [ ] **Auto-resume wrapper** (§C is built): `until grep -q "Cohort
  complete" <log>; do python -m training_v2.cohort.runner ... --resume-from
  <out>; done` (first pass no checkpoint → fresh; later → resume).

### [ ] Gene ranges review (before step 4)
`open_cost` hard-bound [0,2] (>2 → bets=0 collapse); confirm
`matured_arb_bonus_weight` range gives room without collapse. From
genes.py defaults; widen only if the population clusters at a bound.

## Per-run verification (every stage)
- [ ] First-gen log: `train_mean_mature_prob_bce > 0` (if gene on),
  `pairs_opened > 0` (collapse guard §R.4 / §S.4), monitor row written,
  obs_dim == full (not 23-d lean).
- [ ] train ∩ holdout = ∅ (no leak).
- [ ] Holdout number reported once, fully-hedged (close_walk ON).
