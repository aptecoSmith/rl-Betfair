# Predictor model card: gbm_t300_s0_a90ce4ee4d77

- session: smoke
- seed: 0
- architecture: gbm (t300)
- arch_kwargs: `{"n_trees": 200, "max_depth": 5, "learning_rate": 0.05}`
- feature variant: V1
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.05, batch=1024, max_epochs=1

## Run extras
```json
{
  "param_count": 997,
  "train_seconds": 23.3,
  "infer_us_per_row": 3.9872247725725174,
  "device": "cpu",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\gbm_t300_s0_a90ce4ee4d77.joblib"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.526734,
  "pinball_3m_q50": 0.916998,
  "pinball_3m_q90": 0.513759,
  "mae_3m": 1.833995,
  "coverage_3m": 0.7863,
  "calibration_gap_3m": 0.0137,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 1.0,
  "pinball_7m_q10": 0.777144,
  "pinball_7m_q50": 1.445871,
  "pinball_7m_q90": 0.713044,
  "mae_7m": 2.891742,
  "coverage_7m": 0.771,
  "calibration_gap_7m": 0.029,
  "dir_acc_k5_7m": null,
  "dir_fires_k5_7m": 0,
  "dir_fire_rate_k5_7m": 0.0,
  "backtest_pnl_k5_7m": 0.0,
  "backtest_winrate_k5_7m": null,
  "lag1_autocorr_q50_7m": 1.0
}
```