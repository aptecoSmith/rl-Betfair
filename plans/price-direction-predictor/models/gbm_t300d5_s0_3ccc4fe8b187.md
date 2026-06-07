# Predictor model card: gbm_t300d5_s0_3ccc4fe8b187

- session: S04
- seed: 0
- architecture: gbm (t300d5)
- arch_kwargs: `{"n_trees": 300, "max_depth": 5, "learning_rate": 0.05}`
- feature variant: V3
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.05, batch=1024, max_epochs=1

## Run extras
```json
{
  "param_count": 1385,
  "train_seconds": 30.6,
  "infer_us_per_row": 5.21843321621418,
  "device": "cpu",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\gbm_t300d5_s0_3ccc4fe8b187.joblib"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.520307,
  "pinball_3m_q50": 0.91699,
  "pinball_3m_q90": 0.510295,
  "mae_3m": 1.833981,
  "coverage_3m": 0.7889,
  "calibration_gap_3m": 0.0111,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 1.0,
  "pinball_7m_q10": 0.77738,
  "pinball_7m_q50": 1.447517,
  "pinball_7m_q90": 0.710905,
  "mae_7m": 2.895033,
  "coverage_7m": 0.7732,
  "calibration_gap_7m": 0.0268,
  "dir_acc_k5_7m": null,
  "dir_fires_k5_7m": 0,
  "dir_fire_rate_k5_7m": 0.0,
  "backtest_pnl_k5_7m": 0.0,
  "backtest_winrate_k5_7m": null,
  "lag1_autocorr_q50_7m": 1.0,
  "pinball_15m_q10": 1.170406,
  "pinball_15m_q50": 2.189531,
  "pinball_15m_q90": 1.021309,
  "mae_15m": 4.379062,
  "coverage_15m": 0.7535,
  "calibration_gap_15m": 0.0465,
  "dir_acc_k5_15m": null,
  "dir_fires_k5_15m": 0,
  "dir_fire_rate_k5_15m": 0.0,
  "backtest_pnl_k5_15m": 0.0,
  "backtest_winrate_k5_15m": null,
  "lag1_autocorr_q50_15m": 0.8863
}
```