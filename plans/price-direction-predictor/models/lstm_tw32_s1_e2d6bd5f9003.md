# Predictor model card: lstm_tw32_s1_e2d6bd5f9003

- session: S04_neural
- seed: 1
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
  "train_seconds": 31.9,
  "infer_us_per_row": 4.364410415291786,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\lstm_tw32_s1_e2d6bd5f9003.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.570526,
  "pinball_3m_q50": 0.917569,
  "pinball_3m_q90": 0.564034,
  "mae_3m": 1.835139,
  "coverage_3m": 0.7725,
  "calibration_gap_3m": 0.0275,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.8786,
  "pinball_7m_q10": 0.836358,
  "pinball_7m_q50": 1.449244,
  "pinball_7m_q90": 0.787237,
  "mae_7m": 2.898488,
  "coverage_7m": 0.771,
  "calibration_gap_7m": 0.029,
  "dir_acc_k5_7m": null,
  "dir_fires_k5_7m": 0,
  "dir_fire_rate_k5_7m": 0.0,
  "backtest_pnl_k5_7m": 0.0,
  "backtest_winrate_k5_7m": null,
  "lag1_autocorr_q50_7m": 0.9153,
  "pinball_15m_q10": 1.244601,
  "pinball_15m_q50": 2.207204,
  "pinball_15m_q90": 1.112216,
  "mae_15m": 4.414407,
  "coverage_15m": 0.7452,
  "calibration_gap_15m": 0.0548,
  "dir_acc_k5_15m": null,
  "dir_fires_k5_15m": 0,
  "dir_fire_rate_k5_15m": 0.0,
  "backtest_pnl_k5_15m": 0.0,
  "backtest_winrate_k5_15m": null,
  "lag1_autocorr_q50_15m": 0.9789
}
```