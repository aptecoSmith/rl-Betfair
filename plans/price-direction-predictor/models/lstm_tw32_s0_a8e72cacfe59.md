# Predictor model card: lstm_tw32_s0_a8e72cacfe59

- session: S04_neural
- seed: 0
- architecture: lstm (tw32)
- arch_kwargs: `{"time_window": 32, "hidden": 64, "layers": 2, "dropout": 0.1}`
- feature variant: V2
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 61577,
  "train_seconds": 377.2,
  "infer_us_per_row": 5.063600838184357,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\lstm_tw32_s0_a8e72cacfe59.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.535945,
  "pinball_3m_q50": 0.918018,
  "pinball_3m_q90": 0.52486,
  "mae_3m": 1.836035,
  "coverage_3m": 0.7914,
  "calibration_gap_3m": 0.0086,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.9756,
  "pinball_7m_q10": 0.794759,
  "pinball_7m_q50": 1.433607,
  "pinball_7m_q90": 0.733756,
  "mae_7m": 2.867214,
  "coverage_7m": 0.7762,
  "calibration_gap_7m": 0.0238,
  "dir_acc_k5_7m": 0.8407,
  "dir_fires_k5_7m": 113,
  "dir_fire_rate_k5_7m": 0.0007,
  "backtest_pnl_k5_7m": 91.7984,
  "backtest_winrate_k5_7m": 0.8407,
  "lag1_autocorr_q50_7m": 0.9462,
  "pinball_15m_q10": 1.192248,
  "pinball_15m_q50": 2.151119,
  "pinball_15m_q90": 1.039715,
  "mae_15m": 4.302239,
  "coverage_15m": 0.7506,
  "calibration_gap_15m": 0.0494,
  "dir_acc_k5_15m": 0.8863,
  "dir_fires_k5_15m": 255,
  "dir_fire_rate_k5_15m": 0.0033,
  "backtest_pnl_k5_15m": 231.7349,
  "backtest_winrate_k5_15m": 0.8863,
  "lag1_autocorr_q50_15m": 0.9383
}
```