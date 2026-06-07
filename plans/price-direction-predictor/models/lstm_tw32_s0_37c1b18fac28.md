# Predictor model card: lstm_tw32_s0_37c1b18fac28

- session: S04_neural
- seed: 0
- architecture: lstm (tw32)
- arch_kwargs: `{"time_window": 32, "hidden": 64, "layers": 2, "dropout": 0.1}`
- feature variant: V4
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 64905,
  "train_seconds": 37.6,
  "infer_us_per_row": 4.535075277090073,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\lstm_tw32_s0_37c1b18fac28.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.56333,
  "pinball_3m_q50": 0.918569,
  "pinball_3m_q90": 0.560806,
  "mae_3m": 1.837138,
  "coverage_3m": 0.8078,
  "calibration_gap_3m": 0.0078,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.8047,
  "pinball_7m_q10": 0.825526,
  "pinball_7m_q50": 1.460406,
  "pinball_7m_q90": 0.789862,
  "mae_7m": 2.920813,
  "coverage_7m": 0.794,
  "calibration_gap_7m": 0.006,
  "dir_acc_k5_7m": null,
  "dir_fires_k5_7m": 0,
  "dir_fire_rate_k5_7m": 0.0,
  "backtest_pnl_k5_7m": 0.0,
  "backtest_winrate_k5_7m": null,
  "lag1_autocorr_q50_7m": 0.8914,
  "pinball_15m_q10": 1.22298,
  "pinball_15m_q50": 2.220423,
  "pinball_15m_q90": 1.117791,
  "mae_15m": 4.440847,
  "coverage_15m": 0.78,
  "calibration_gap_15m": 0.02,
  "dir_acc_k5_15m": null,
  "dir_fires_k5_15m": 0,
  "dir_fire_rate_k5_15m": 0.0,
  "backtest_pnl_k5_15m": 0.0,
  "backtest_winrate_k5_15m": null,
  "lag1_autocorr_q50_15m": 0.9641
}
```