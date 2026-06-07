# Predictor model card: gbm_t300d5_s2_ffde0d6d1c94

- session: S04
- seed: 2
- architecture: gbm (t300d5)
- arch_kwargs: `{"n_trees": 300, "max_depth": 5, "learning_rate": 0.05}`
- feature variant: V5
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.05, batch=1024, max_epochs=1

## Run extras
```json
{
  "param_count": 1669,
  "train_seconds": 41.1,
  "infer_us_per_row": 6.05359673500061,
  "device": "cpu",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\gbm_t300d5_s2_ffde0d6d1c94.joblib"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.517713,
  "pinball_3m_q50": 0.916985,
  "pinball_3m_q90": 0.506332,
  "mae_3m": 1.83397,
  "coverage_3m": 0.7862,
  "calibration_gap_3m": 0.0138,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 1.0,
  "pinball_7m_q10": 0.781889,
  "pinball_7m_q50": 1.446807,
  "pinball_7m_q90": 0.710418,
  "mae_7m": 2.893614,
  "coverage_7m": 0.7702,
  "calibration_gap_7m": 0.0298,
  "dir_acc_k5_7m": null,
  "dir_fires_k5_7m": 0,
  "dir_fire_rate_k5_7m": 0.0,
  "backtest_pnl_k5_7m": 0.0,
  "backtest_winrate_k5_7m": null,
  "lag1_autocorr_q50_7m": 1.0,
  "pinball_15m_q10": 1.168678,
  "pinball_15m_q50": 2.194925,
  "pinball_15m_q90": 1.030252,
  "mae_15m": 4.389849,
  "coverage_15m": 0.7483,
  "calibration_gap_15m": 0.0517,
  "dir_acc_k5_15m": null,
  "dir_fires_k5_15m": 0,
  "dir_fire_rate_k5_15m": 0.0,
  "backtest_pnl_k5_15m": 0.0,
  "backtest_winrate_k5_15m": null,
  "lag1_autocorr_q50_15m": 0.9278
}
```