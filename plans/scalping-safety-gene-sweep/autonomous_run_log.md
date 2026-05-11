# Scalping safety-gene sweep — autonomous run log

Per-iteration progress through the plan. Newest entries at the
bottom.

## 2026-05-11 09:50 — Plan opened, cohort launching

**State entering iteration:** Predecessor cohort
`_predictor_SCALPING_overnight_1778458751` shipped with all
safety/shaping genes pinned to defaults. Plan scaffolded;
launch command being prepared.

**Work done:**
- Created `README.md`, `hard_constraints.md`, `master_todo.md`.
- Verified gene infrastructure: `training_v2/cohort/genes.py`
  Phase 5 has all six target genes registered with
  `--enable-gene NAME` activation. No code changes needed.

**Next iteration:** Launch the cohort, then schedule a
~30-min wakeup to confirm the first generation is rolling
cleanly (no KL blowup / GA collapse).

## 2026-05-11 17:58 — Cohort launched

**Tag:** `_predictor_SCALPING_safety_1778518690`
**Log:** `registry/_predictor_SCALPING_safety_1778518690.log`

**Configuration:**
- 12 agents × 8 generations, 3 training days, 3 eval days
  (eval=2026-05-04..2026-05-06).
- CUDA, seed 42, mutation_rate 0.2 (raised from predecessor's
  default 0.1 — wider gene exploration on a new gene pool).
- All 6 target genes active (confirmed in log):
  `fill_prob_loss_weight, mature_prob_loss_weight,
  matured_arb_bonus_weight, naked_loss_scale, open_cost,
  stop_loss_pnl_threshold`.
- Predictor bundle: same three production champions
  (`1c15250ee90d1b65`, `b23018bf5c8bcc70`,
  `conv1d_k3_s1_9659e9e9c3fb`).
- Lean obs (`--predictor-lean-obs`), scalping mode.

**Gene draw spot-check (Generation 1):**
- agent 1: open_cost=1.78, naked_loss_scale=0.42,
  stop_loss=0.009, matured_arb_bonus=0.43, fpw=0.066,
  mpw=3.02. **High open_cost, mid naked_scale, low stop_loss.**
- agent 4: open_cost=0.46, naked_loss_scale=0.08,
  stop_loss=0.07, matured_arb_bonus=1.45, fpw=0.030,
  mpw=2.11. **Heavy naked-anneal, light open-cost.**
- agent 12: open_cost=1.06, naked_loss_scale=0.96,
  stop_loss=0.28, matured_arb_bonus=3.03, fpw=0.227,
  mpw=3.76. **Aggressive everything — interesting test case.**

**Anomaly noted:** With `--days 6` the runner produced 3
train + 3 eval (predecessor `--days 6` produced 5 train + 1
eval). The GA will now select on the same 3-day window I was
treating as held-out. The final verdict will need a reeval
against a fresh window outside this 6-day span. Will resolve
when the cohort completes — note it but don't restart, the
gene exploration is what matters.

**Next iteration:** Schedule ~30 min wakeup to confirm
Generation 1 is progressing (no KL blowup / GA crash).
After that, ~2h heartbeats until completion (~12h ETA).
