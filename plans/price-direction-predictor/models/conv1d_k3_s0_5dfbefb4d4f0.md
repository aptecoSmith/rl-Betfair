# Predictor model card: conv1d_k3_s0_5dfbefb4d4f0

- session: S06_neural
- seed: 0
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V2
- train corpus: tvl_mask_29d
- horizons: ['1m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 46857,
  "train_seconds": 402.3,
  "infer_us_per_row": 3.7529971450567245,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s0_5dfbefb4d4f0.pt"
}
```

## Val metrics
```json
{
  "pinball_1m_q10": 0.353237,
  "pinball_1m_q50": 0.565556,
  "pinball_1m_q90": 0.357411,
  "mae_1m": 1.131111,
  "coverage_1m": 0.8006,
  "calibration_gap_1m": 0.0006,
  "dir_acc_k5_1m": 0.7297,
  "dir_fires_k5_1m": 37,
  "dir_fire_rate_k5_1m": 0.0002,
  "backtest_pnl_k5_1m": 36.1443,
  "backtest_winrate_k5_1m": 0.7297,
  "lag1_autocorr_q50_1m": 0.7771,
  "pinball_7m_q10": 0.772128,
  "pinball_7m_q50": 1.41822,
  "pinball_7m_q90": 0.724628,
  "mae_7m": 2.83644,
  "coverage_7m": 0.7811,
  "calibration_gap_7m": 0.0189,
  "dir_acc_k5_7m": 0.7738,
  "dir_fires_k5_7m": 442,
  "dir_fire_rate_k5_7m": 0.0029,
  "backtest_pnl_k5_7m": 396.4798,
  "backtest_winrate_k5_7m": 0.7738,
  "lag1_autocorr_q50_7m": 0.6989,
  "pinball_15m_q10": 1.167096,
  "pinball_15m_q50": 2.128993,
  "pinball_15m_q90": 1.035177,
  "mae_15m": 4.257985,
  "coverage_15m": 0.7717,
  "calibration_gap_15m": 0.0283,
  "dir_acc_k5_15m": 0.9221,
  "dir_fires_k5_15m": 244,
  "dir_fire_rate_k5_15m": 0.0031,
  "backtest_pnl_k5_15m": 356.6908,
  "backtest_winrate_k5_15m": 0.9221,
  "lag1_autocorr_q50_15m": 0.7308
}
```