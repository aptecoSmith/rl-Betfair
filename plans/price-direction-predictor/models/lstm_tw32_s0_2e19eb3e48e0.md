# Predictor model card: lstm_tw32_s0_2e19eb3e48e0

- session: S04_neural
- seed: 0
- architecture: lstm (tw32)
- arch_kwargs: `{"time_window": 32, "hidden": 64, "layers": 2, "dropout": 0.1}`
- feature variant: V3
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 63625,
  "train_seconds": 42.3,
  "infer_us_per_row": 4.410045221447945,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\lstm_tw32_s0_2e19eb3e48e0.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.562161,
  "pinball_3m_q50": 0.91801,
  "pinball_3m_q90": 0.558448,
  "mae_3m": 1.836021,
  "coverage_3m": 0.8003,
  "calibration_gap_3m": 0.0003,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.8287,
  "pinball_7m_q10": 0.822194,
  "pinball_7m_q50": 1.465709,
  "pinball_7m_q90": 0.787067,
  "mae_7m": 2.931417,
  "coverage_7m": 0.779,
  "calibration_gap_7m": 0.021,
  "dir_acc_k5_7m": null,
  "dir_fires_k5_7m": 0,
  "dir_fire_rate_k5_7m": 0.0,
  "backtest_pnl_k5_7m": 0.0,
  "backtest_winrate_k5_7m": null,
  "lag1_autocorr_q50_7m": 0.925,
  "pinball_15m_q10": 1.231735,
  "pinball_15m_q50": 2.227207,
  "pinball_15m_q90": 1.109581,
  "mae_15m": 4.454413,
  "coverage_15m": 0.7523,
  "calibration_gap_15m": 0.0477,
  "dir_acc_k5_15m": null,
  "dir_fires_k5_15m": 0,
  "dir_fire_rate_k5_15m": 0.0,
  "backtest_pnl_k5_15m": 0.0,
  "backtest_winrate_k5_15m": null,
  "lag1_autocorr_q50_15m": 0.9528
}
```