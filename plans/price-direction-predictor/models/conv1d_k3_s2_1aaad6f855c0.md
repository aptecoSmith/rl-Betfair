# Predictor model card: conv1d_k3_s2_1aaad6f855c0

- session: S06_neural
- seed: 2
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
  "train_seconds": 351.7,
  "infer_us_per_row": 3.070337697863579,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s2_1aaad6f855c0.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.523548,
  "pinball_3m_q30": 0.835842,
  "pinball_3m_q50": 0.914412,
  "pinball_3m_q70": 0.847332,
  "pinball_3m_q90": 0.510145,
  "mae_3m": 1.828823,
  "coverage_3m": 0.7934,
  "calibration_gap_3m": 0.0066,
  "dir_acc_k5_3m": 0.3556,
  "dir_fires_k5_3m": 45,
  "dir_fire_rate_k5_3m": 0.0002,
  "backtest_pnl_k5_3m": 25.7448,
  "backtest_winrate_k5_3m": 0.3556,
  "lag1_autocorr_q50_3m": 0.2204,
  "pinball_7m_q10": 0.784361,
  "pinball_7m_q30": 1.305211,
  "pinball_7m_q50": 1.425776,
  "pinball_7m_q70": 1.274716,
  "pinball_7m_q90": 0.719341,
  "mae_7m": 2.851552,
  "coverage_7m": 0.7841,
  "calibration_gap_7m": 0.0159,
  "dir_acc_k5_7m": 0.7347,
  "dir_fires_k5_7m": 505,
  "dir_fire_rate_k5_7m": 0.0033,
  "backtest_pnl_k5_7m": 354.6579,
  "backtest_winrate_k5_7m": 0.7347,
  "lag1_autocorr_q50_7m": 0.2662,
  "pinball_15m_q10": 1.180345,
  "pinball_15m_q30": 1.97136,
  "pinball_15m_q50": 2.136309,
  "pinball_15m_q70": 1.872506,
  "pinball_15m_q90": 1.03026,
  "mae_15m": 4.272618,
  "coverage_15m": 0.7791,
  "calibration_gap_15m": 0.0209,
  "dir_acc_k5_15m": 0.845,
  "dir_fires_k5_15m": 529,
  "dir_fire_rate_k5_15m": 0.0068,
  "backtest_pnl_k5_15m": 647.1289,
  "backtest_winrate_k5_15m": 0.845,
  "lag1_autocorr_q50_15m": 0.3939
}
```