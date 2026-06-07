# Predictor model card: gbm_t300d5_s2_00a11defb52e

- session: S04
- seed: 2
- architecture: gbm (t300d5)
- arch_kwargs: `{"n_trees": 300, "max_depth": 5, "learning_rate": 0.05}`
- feature variant: V4
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.05, batch=1024, max_epochs=1

## Run extras
```json
{
  "param_count": 767,
  "train_seconds": 5.3,
  "infer_us_per_row": 3.3515971153974533,
  "device": "cpu",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\gbm_t300d5_s2_00a11defb52e.joblib"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.536585,
  "pinball_3m_q50": 0.917052,
  "pinball_3m_q90": 0.522689,
  "mae_3m": 1.834104,
  "coverage_3m": 0.7739,
  "calibration_gap_3m": 0.0261,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 1.0,
  "pinball_7m_q10": 0.810301,
  "pinball_7m_q50": 1.447738,
  "pinball_7m_q90": 0.766053,
  "mae_7m": 2.895475,
  "coverage_7m": 0.7625,
  "calibration_gap_7m": 0.0375,
  "dir_acc_k5_7m": null,
  "dir_fires_k5_7m": 0,
  "dir_fire_rate_k5_7m": 0.0,
  "backtest_pnl_k5_7m": 0.0,
  "backtest_winrate_k5_7m": null,
  "lag1_autocorr_q50_7m": 1.0,
  "pinball_15m_q10": 1.231052,
  "pinball_15m_q50": 2.194118,
  "pinball_15m_q90": 1.075873,
  "mae_15m": 4.388236,
  "coverage_15m": 0.7389,
  "calibration_gap_15m": 0.0611,
  "dir_acc_k5_15m": null,
  "dir_fires_k5_15m": 0,
  "dir_fire_rate_k5_15m": 0.0,
  "backtest_pnl_k5_15m": 0.0,
  "backtest_winrate_k5_15m": null,
  "lag1_autocorr_q50_15m": 0.8654
}
```