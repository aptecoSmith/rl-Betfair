# Predictor model card: gbm_t100d4_s1_e3ae684d7b8b

- session: S04
- seed: 1
- architecture: gbm (t100d4)
- arch_kwargs: `{"n_trees": 100, "max_depth": 4, "learning_rate": 0.05}`
- feature variant: V3
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.05, batch=1024, max_epochs=1

## Run extras
```json
{
  "param_count": 496,
  "train_seconds": 3.8,
  "infer_us_per_row": 2.210726961493492,
  "device": "cpu",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\gbm_t100d4_s1_e3ae684d7b8b.joblib"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.540701,
  "pinball_3m_q50": 0.917114,
  "pinball_3m_q90": 0.540886,
  "mae_3m": 1.834228,
  "coverage_3m": 0.7836,
  "calibration_gap_3m": 0.0164,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 1.0,
  "pinball_7m_q10": 0.806201,
  "pinball_7m_q50": 1.447736,
  "pinball_7m_q90": 0.759345,
  "mae_7m": 2.895473,
  "coverage_7m": 0.7695,
  "calibration_gap_7m": 0.0305,
  "dir_acc_k5_7m": null,
  "dir_fires_k5_7m": 0,
  "dir_fire_rate_k5_7m": 0.0,
  "backtest_pnl_k5_7m": 0.0,
  "backtest_winrate_k5_7m": null,
  "lag1_autocorr_q50_7m": 0.9739,
  "pinball_15m_q10": 1.224182,
  "pinball_15m_q50": 2.192345,
  "pinball_15m_q90": 1.068616,
  "mae_15m": 4.38469,
  "coverage_15m": 0.7461,
  "calibration_gap_15m": 0.0539,
  "dir_acc_k5_15m": null,
  "dir_fires_k5_15m": 0,
  "dir_fire_rate_k5_15m": 0.0,
  "backtest_pnl_k5_15m": 0.0,
  "backtest_winrate_k5_15m": null,
  "lag1_autocorr_q50_15m": 0.8739
}
```