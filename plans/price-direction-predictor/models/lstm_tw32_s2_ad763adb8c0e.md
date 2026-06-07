# Predictor model card: lstm_tw32_s2_ad763adb8c0e

- session: S04_neural
- seed: 2
- architecture: lstm (tw32)
- arch_kwargs: `{"time_window": 32, "hidden": 64, "layers": 2, "dropout": 0.1}`
- feature variant: V3
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 63625,
  "train_seconds": 234.6,
  "infer_us_per_row": 4.340894520282745,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\lstm_tw32_s2_ad763adb8c0e.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.547691,
  "pinball_3m_q50": 0.917733,
  "pinball_3m_q90": 0.544264,
  "mae_3m": 1.835465,
  "coverage_3m": 0.7769,
  "calibration_gap_3m": 0.0231,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.9038,
  "pinball_7m_q10": 0.805498,
  "pinball_7m_q50": 1.448338,
  "pinball_7m_q90": 0.757704,
  "mae_7m": 2.896676,
  "coverage_7m": 0.7696,
  "calibration_gap_7m": 0.0304,
  "dir_acc_k5_7m": 0.7704,
  "dir_fires_k5_7m": 135,
  "dir_fire_rate_k5_7m": 0.0009,
  "backtest_pnl_k5_7m": 184.8104,
  "backtest_winrate_k5_7m": 0.7704,
  "lag1_autocorr_q50_7m": 0.8607,
  "pinball_15m_q10": 1.194657,
  "pinball_15m_q50": 2.184822,
  "pinball_15m_q90": 1.049657,
  "mae_15m": 4.369644,
  "coverage_15m": 0.7646,
  "calibration_gap_15m": 0.0354,
  "dir_acc_k5_15m": 0.8106,
  "dir_fires_k5_15m": 301,
  "dir_fire_rate_k5_15m": 0.0039,
  "backtest_pnl_k5_15m": 317.9288,
  "backtest_winrate_k5_15m": 0.8106,
  "lag1_autocorr_q50_15m": 0.9485
}
```