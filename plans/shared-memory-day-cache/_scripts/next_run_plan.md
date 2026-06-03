# NEXT-RUN PLAN — execute autonomously when the 18h BC A/B finishes

The user asked (2026-06-03 ~06:40, while away) to **queue another training run
with relevant params once Arm B finishes**. They are NOT available — decide
and launch yourself, do not gate on input (see memory
`feedback_autonomous_when_away`).

## Trigger / current run
- Current run: `_scripts/launch_18h_bc_ab.sh` — BC A/B, N=16, predictors-ON
  (shared static_obs path), 30 agents × 5 gens × 2 arms, holdout May 20-29.
- Dirs: `registry/smdc_18h_dirs.txt` → `bcoff=...` (Arm A, DONE), `bcon=...`
  (Arm B). Master log: `registry/smdc_18h_master.log`.
- DONE signal: `"18H BC A/B COMPLETE"` in the master log (watchdog `bj-...`
  + the training task notify on this). Arm A already complete (5 gens, 150
  agents, best composite +0.067→+0.122(gen3)→+0.096(gen5), all pnl negative =
  naked-leg variance).

## Step 1 — held-out re-eval (preferred; don't block the box on it)
- Find the re-eval tool: `ls plans/*/run_holdout*reeval*.sh tools/*reeval*` —
  e.g. `plans/recipe-expansion-and-robustness/run_holdout_reeval_batch.sh`,
  or the evaluator with `--eval-days`. Re-eval each arm's FINAL-gen top-K
  champions on the **7 sealed holdout days** (2026-05-20,21,22,25,27,28,29)
  with **force_close=120** (deploy-safety asymmetry —
  `project_force_close_train_vs_deploy`, `feedback_always_eval_holdout`).
- Compare arms on held-out **locked_per_std** AND **naked σ per leg**
  (hard ceiling ~£30 — `feedback_naked_variance_primary_metric`).
- If the re-eval tool isn't quickly runnable, FALL BACK to iteration-eval
  composite to pick the winner and launch anyway (don't leave the box idle).

## Step 2 — decide the next run (relevant params)
**Headline finding to act on:** every top agent has **positive composite but
negative total P&L** → the naked leg's ~zero-EV variance is the binding
problem. Memory says: GA selection can't fix a reward-side problem
(`feedback_ga_selection_vs_reward_shaping`), and reducing **naked variance per
leg** is the flagged next design lever (`feedback_naked_variance_primary_metric`).

**Default next run (memory-backed, decisive):** take the **held-out-winning
arm's config** and add `--enable-gene naked_variance_penalty_beta` (the
scalping-tight-naked-variance Phase-2A lever — penalises naked variance, flows
via reward_overrides). Let the GA find the per-agent penalty strength. This
attacks exactly what the A/B exposed. Keep everything else: N=16, predictors-ON,
`--exclude-days $HOLDOUT`, `--composite-score-mode locked_per_std`,
`--argmax-eval`, `--mature-prob-open-threshold 0.30`, the 4 existing
`--enable-gene` knobs (open_cost, stop_loss_pnl_threshold,
mature_prob_loss_weight, arb_spread_target_lock_pct), the reward-overrides
(`per_pair_reward_at_resolution=true`, `locked_pnl_reward_weight=9.0`).
- If BC-ON won the held-out: include `--bc-pretrain-steps 500` (BC-ON arm).
- If BC-OFF won / wash: drop BC (the simpler arm), it isn't the lever.
- 30 agents × 5 gens (≈6.7h/arm at N=16). One arm (not an A/B) unless a second
  knob is clearly worth pairing.

## Step 3 — launch + protect
- Copy `_scripts/launch_18h_bc_ab.sh` → `_scripts/launch_next_naked_var.sh`,
  edit to a SINGLE arm with the decided params + `naked_variance_penalty_beta`.
  Output dir `registry/smdc_nakedvar_<ts>`.
- Launch via Bash `run_in_background` (survives session-idle — the 18h run
  proved bash-bg persists; PowerShell-tool bg does NOT).
- Re-arm: a survivable RAM watch (`powershell -File ram_watch.ps1 ...` via
  Bash bg) + a completion watchdog (bash bg polling for the new
  `Cohort complete`).
- Append an EXPERIMENTS.md entry (intention = attack naked-leg variance on the
  A/B winner; implementation; result-pending).

## Hard rules
- N=16 (RAM is fine — `step3_memory.md`). NO `--batched` (drops
  predictors/BC). System python (`python`, C:\Python314 — validated).
- Don't launch while the previous run still holds the box (wait for
  `taskkill`-clean or natural pool drain; check `Get-Process python`).
