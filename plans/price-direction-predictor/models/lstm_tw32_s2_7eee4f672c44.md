# Predictor model card: lstm_tw32_s2_7eee4f672c44

- session: S04_neural
- seed: 2
- architecture: lstm (tw32)
- arch_kwargs: `{"time_window": 32, "hidden": 64, "layers": 2, "dropout": 0.1}`
- feature variant: V1
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 59017,
  "train_seconds": 273.2,
  "infer_us_per_row": 4.957662895321846,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\lstm_tw32_s2_7eee4f672c44.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.536618,
  "pinball_3m_q50": 0.920008,
  "pinball_3m_q90": 0.527604,
  "mae_3m": 1.840016,
  "coverage_3m": 0.806,
  "calibration_gap_3m": 0.006,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.946,
  "pinball_7m_q10": 0.793891,
  "pinball_7m_q50": 1.440906,
  "pinball_7m_q90": 0.736039,
  "mae_7m": 2.881812,
  "coverage_7m": 0.7716,
  "calibration_gap_7m": 0.0284,
  "dir_acc_k5_7m": 0.7383,
  "dir_fires_k5_7m": 428,
  "dir_fire_rate_k5_7m": 0.0028,
  "backtest_pnl_k5_7m": 378.4693,
  "backtest_winrate_k5_7m": 0.7383,
  "lag1_autocorr_q50_7m": 0.9353,
  "pinball_15m_q10": 1.205696,
  "pinball_15m_q50": 2.160207,
  "pinball_15m_q90": 1.046302,
  "mae_15m": 4.320413,
  "coverage_15m": 0.7601,
  "calibration_gap_15m": 0.0399,
  "dir_acc_k5_15m": 0.8819,
  "dir_fires_k5_15m": 601,
  "dir_fire_rate_k5_15m": 0.0077,
  "backtest_pnl_k5_15m": 617.0377,
  "backtest_winrate_k5_15m": 0.8819,
  "lag1_autocorr_q50_15m": 0.9715
}
```