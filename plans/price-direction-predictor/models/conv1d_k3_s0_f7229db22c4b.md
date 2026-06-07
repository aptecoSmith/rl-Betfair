# Predictor model card: conv1d_k3_s0_f7229db22c4b

- session: S04_neural
- seed: 0
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V3
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 48393,
  "train_seconds": 251.2,
  "infer_us_per_row": 3.7481077015399933,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s0_f7229db22c4b.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.532941,
  "pinball_3m_q50": 0.913042,
  "pinball_3m_q90": 0.519074,
  "mae_3m": 1.826084,
  "coverage_3m": 0.7654,
  "calibration_gap_3m": 0.0346,
  "dir_acc_k5_3m": 0.3684,
  "dir_fires_k5_3m": 76,
  "dir_fire_rate_k5_3m": 0.0004,
  "backtest_pnl_k5_3m": 15.3858,
  "backtest_winrate_k5_3m": 0.3684,
  "lag1_autocorr_q50_3m": 0.0659,
  "pinball_7m_q10": 0.803311,
  "pinball_7m_q50": 1.428246,
  "pinball_7m_q90": 0.721222,
  "mae_7m": 2.856491,
  "coverage_7m": 0.7571,
  "calibration_gap_7m": 0.0429,
  "dir_acc_k5_7m": 0.6821,
  "dir_fires_k5_7m": 302,
  "dir_fire_rate_k5_7m": 0.002,
  "backtest_pnl_k5_7m": 177.6427,
  "backtest_winrate_k5_7m": 0.6821,
  "lag1_autocorr_q50_7m": 0.3426,
  "pinball_15m_q10": 1.19882,
  "pinball_15m_q50": 2.147172,
  "pinball_15m_q90": 1.022657,
  "mae_15m": 4.294344,
  "coverage_15m": 0.7529,
  "calibration_gap_15m": 0.0471,
  "dir_acc_k5_15m": 0.7644,
  "dir_fires_k5_15m": 225,
  "dir_fire_rate_k5_15m": 0.0029,
  "backtest_pnl_k5_15m": 327.4016,
  "backtest_winrate_k5_15m": 0.7644,
  "lag1_autocorr_q50_15m": 0.3947
}
```