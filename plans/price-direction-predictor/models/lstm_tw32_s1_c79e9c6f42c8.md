# Predictor model card: lstm_tw32_s1_c79e9c6f42c8

- session: S04_neural
- seed: 1
- architecture: lstm (tw32)
- arch_kwargs: `{"time_window": 32, "hidden": 64, "layers": 2, "dropout": 0.1}`
- feature variant: V5
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 65929,
  "train_seconds": 383.4,
  "infer_us_per_row": 4.543224349617958,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\lstm_tw32_s1_c79e9c6f42c8.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.568696,
  "pinball_3m_q50": 0.917593,
  "pinball_3m_q90": 0.549795,
  "mae_3m": 1.835186,
  "coverage_3m": 0.7569,
  "calibration_gap_3m": 0.0431,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.9689,
  "pinball_7m_q10": 0.829995,
  "pinball_7m_q50": 1.442129,
  "pinball_7m_q90": 0.769963,
  "mae_7m": 2.884257,
  "coverage_7m": 0.7334,
  "calibration_gap_7m": 0.0666,
  "dir_acc_k5_7m": 0.6822,
  "dir_fires_k5_7m": 365,
  "dir_fire_rate_k5_7m": 0.0024,
  "backtest_pnl_k5_7m": 289.8391,
  "backtest_winrate_k5_7m": 0.6822,
  "lag1_autocorr_q50_7m": 0.9489,
  "pinball_15m_q10": 1.230467,
  "pinball_15m_q50": 2.180504,
  "pinball_15m_q90": 1.088664,
  "mae_15m": 4.361008,
  "coverage_15m": 0.7096,
  "calibration_gap_15m": 0.0904,
  "dir_acc_k5_15m": 0.8211,
  "dir_fires_k5_15m": 710,
  "dir_fire_rate_k5_15m": 0.0091,
  "backtest_pnl_k5_15m": 579.0795,
  "backtest_winrate_k5_15m": 0.8211,
  "lag1_autocorr_q50_15m": 0.9752
}
```