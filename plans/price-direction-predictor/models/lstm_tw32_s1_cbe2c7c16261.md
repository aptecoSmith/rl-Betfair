# Predictor model card: lstm_tw32_s1_cbe2c7c16261

- session: S04_neural
- seed: 1
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
  "train_seconds": 416.9,
  "infer_us_per_row": 4.278263077139854,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\lstm_tw32_s1_cbe2c7c16261.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.530632,
  "pinball_3m_q50": 0.917823,
  "pinball_3m_q90": 0.526336,
  "mae_3m": 1.835646,
  "coverage_3m": 0.7774,
  "calibration_gap_3m": 0.0226,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.9385,
  "pinball_7m_q10": 0.78904,
  "pinball_7m_q50": 1.438877,
  "pinball_7m_q90": 0.740085,
  "mae_7m": 2.877753,
  "coverage_7m": 0.7648,
  "calibration_gap_7m": 0.0352,
  "dir_acc_k5_7m": 0.7229,
  "dir_fires_k5_7m": 433,
  "dir_fire_rate_k5_7m": 0.0028,
  "backtest_pnl_k5_7m": 311.1325,
  "backtest_winrate_k5_7m": 0.7229,
  "lag1_autocorr_q50_7m": 0.9344,
  "pinball_15m_q10": 1.206792,
  "pinball_15m_q50": 2.161393,
  "pinball_15m_q90": 1.062759,
  "mae_15m": 4.322787,
  "coverage_15m": 0.7338,
  "calibration_gap_15m": 0.0662,
  "dir_acc_k5_15m": 0.8519,
  "dir_fires_k5_15m": 918,
  "dir_fire_rate_k5_15m": 0.0118,
  "backtest_pnl_k5_15m": 746.9599,
  "backtest_winrate_k5_15m": 0.8519,
  "lag1_autocorr_q50_15m": 0.9441
}
```