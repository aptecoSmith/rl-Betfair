# Predictor model card: lstm_tw64_s1_ec88bfa8d823

- session: S03
- seed: 1
- architecture: lstm (tw64)
- arch_kwargs: `{"time_window": 64, "hidden": 64, "layers": 2, "dropout": 0.1}`
- feature variant: V3
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 63625,
  "train_seconds": 62.4,
  "infer_us_per_row": 2.037733793258667,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\lstm_tw64_s1_ec88bfa8d823.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.562372,
  "pinball_3m_q50": 0.917909,
  "pinball_3m_q90": 0.560803,
  "mae_3m": 1.835818,
  "coverage_3m": 0.7746,
  "calibration_gap_3m": 0.0254,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.9207,
  "pinball_7m_q10": 0.824065,
  "pinball_7m_q50": 1.454705,
  "pinball_7m_q90": 0.793564,
  "mae_7m": 2.90941,
  "coverage_7m": 0.7545,
  "calibration_gap_7m": 0.0455,
  "dir_acc_k5_7m": null,
  "dir_fires_k5_7m": 0,
  "dir_fire_rate_k5_7m": 0.0,
  "backtest_pnl_k5_7m": 0.0,
  "backtest_winrate_k5_7m": null,
  "lag1_autocorr_q50_7m": 0.9534,
  "pinball_15m_q10": 1.224427,
  "pinball_15m_q50": 2.206852,
  "pinball_15m_q90": 1.117186,
  "mae_15m": 4.413704,
  "coverage_15m": 0.7267,
  "calibration_gap_15m": 0.0733,
  "dir_acc_k5_15m": null,
  "dir_fires_k5_15m": 0,
  "dir_fire_rate_k5_15m": 0.0,
  "backtest_pnl_k5_15m": 0.0,
  "backtest_winrate_k5_15m": null,
  "lag1_autocorr_q50_15m": 0.9447
}
```