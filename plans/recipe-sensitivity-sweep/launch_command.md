# Launch command — recipe sensitivity sweep

Run from `C:/Users/jsmit/source/repos/rl-betfair/`.

```bash
python -m training_v2.cohort.runner \
  --n-agents 60 \
  --generations 1 \
  --device cuda \
  --seed 42 \
  --output-dir registry/_recipe_sensitivity_sweep_1779649265 \
  --strategy-mode arb \
  --training-days-explicit \
      2026-04-06 2026-04-08 2026-04-09 2026-04-11 2026-04-12 2026-04-13 \
      2026-04-15 2026-04-19 2026-04-20 2026-04-22 2026-04-26 2026-05-02 \
  --cohort-eval-days \
      2026-04-10 2026-04-17 2026-04-21 2026-05-03 2026-05-06 \
  --rotating-eval-sample 0 \
  --bc-pretrain-steps 0 \
  --direction-head-manifest models/direction_head/sweep_c11 \
  --direction-gate-enabled \
  --enable-gene open_cost \
  --enable-gene matured_arb_bonus_weight \
  --enable-gene mark_to_market_weight \
  --enable-gene naked_loss_scale \
  --enable-gene stop_loss_pnl_threshold \
  --enable-gene arb_spread_target_lock_pct \
  --enable-gene fill_prob_loss_weight \
  --enable-gene mature_prob_loss_weight \
  --enable-gene risk_loss_weight \
  --enable-gene alpha_lr \
  --enable-gene reward_clip \
  --enable-gene naked_variance_penalty_beta \
  --enable-gene direction_gate_threshold \
  --enable-gene predictor_feature_gain \
  --reward-overrides \
      force_close_before_off_seconds=120 \
      close_feasibility_max_spread_pct=0.05 \
      matured_arb_expected_random=0.0
```

Notes:
- `--n-agents 60 --generations 1` = 60 unique random gene draws, no GA.
- `--rotating-eval-sample 0` disables rotation (single gen anyway).
- `--bc-pretrain-steps 0` overrides any per-agent gene to disable BC.
- `--direction-head-manifest` loads the frozen C11 head; mutex check
  refuses combination with `direction_prob_loss_weight` /
  `bc_direction_target_weight` --enable-gene flags (correctly).
- `matured_arb_bonus_weight` is now in `--enable-gene`, NOT in
  `--reward-overrides` (would conflict).
- 14 Phase-5 genes evolve + 7 Phase-3 genes auto-evolve = 21 total
  varying knobs; 15 of those are "actively swept" (the
  Phase-3 PPO-hyperparams 7 + the 8 Phase-5 ones we expect to matter).
