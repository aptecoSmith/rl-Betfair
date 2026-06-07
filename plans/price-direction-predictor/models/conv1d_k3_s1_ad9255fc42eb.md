# Predictor model card: conv1d_k3_s1_ad9255fc42eb

- session: S06_neural
- seed: 1
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V4
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m', '15m']
- output: pinball5 (quantiles [0.1, 0.3, 0.5, 0.7, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 49743,
  "train_seconds": 367.9,
  "infer_us_per_row": 2.7443747967481613,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s1_ad9255fc42eb.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.526474,
  "pinball_3m_q30": 0.83845,
  "pinball_3m_q50": 0.909064,
  "pinball_3m_q70": 0.843625,
  "pinball_3m_q90": 0.511349,
  "mae_3m": 1.818127,
  "coverage_3m": 0.7995,
  "calibration_gap_3m": 0.0005,
  "dir_acc_k5_3m": 0.6667,
  "dir_fires_k5_3m": 15,
  "dir_fire_rate_k5_3m": 0.0001,
  "backtest_pnl_k5_3m": 23.7102,
  "backtest_winrate_k5_3m": 0.6667,
  "lag1_autocorr_q50_3m": 0.0525,
  "pinball_7m_q10": 0.787492,
  "pinball_7m_q30": 1.307352,
  "pinball_7m_q50": 1.424006,
  "pinball_7m_q70": 1.26884,
  "pinball_7m_q90": 0.720709,
  "mae_7m": 2.848011,
  "coverage_7m": 0.7835,
  "calibration_gap_7m": 0.0165,
  "dir_acc_k5_7m": 0.7804,
  "dir_fires_k5_7m": 601,
  "dir_fire_rate_k5_7m": 0.0039,
  "backtest_pnl_k5_7m": 556.9643,
  "backtest_winrate_k5_7m": 0.7804,
  "lag1_autocorr_q50_7m": 0.1509,
  "pinball_15m_q10": 1.18034,
  "pinball_15m_q30": 1.975719,
  "pinball_15m_q50": 2.138921,
  "pinball_15m_q70": 1.872761,
  "pinball_15m_q90": 1.028363,
  "mae_15m": 4.277841,
  "coverage_15m": 0.7837,
  "calibration_gap_15m": 0.0163,
  "dir_acc_k5_15m": 0.8892,
  "dir_fires_k5_15m": 388,
  "dir_fire_rate_k5_15m": 0.005,
  "backtest_pnl_k5_15m": 531.0542,
  "backtest_winrate_k5_15m": 0.8892,
  "lag1_autocorr_q50_15m": 0.2971
}
```