# Predictor model card: conv1d_k3_s0_d6eabe2b26e8

- session: S04_neural
- seed: 0
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V4
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 49353,
  "train_seconds": 316.5,
  "infer_us_per_row": 3.8386788219213486,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s0_d6eabe2b26e8.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.518748,
  "pinball_3m_q50": 0.910563,
  "pinball_3m_q90": 0.505731,
  "mae_3m": 1.821127,
  "coverage_3m": 0.7991,
  "calibration_gap_3m": 0.0009,
  "dir_acc_k5_3m": 0.6364,
  "dir_fires_k5_3m": 11,
  "dir_fire_rate_k5_3m": 0.0001,
  "backtest_pnl_k5_3m": 15.6218,
  "backtest_winrate_k5_3m": 0.6364,
  "lag1_autocorr_q50_3m": 0.1595,
  "pinball_7m_q10": 0.781185,
  "pinball_7m_q50": 1.419343,
  "pinball_7m_q90": 0.712435,
  "mae_7m": 2.838687,
  "coverage_7m": 0.769,
  "calibration_gap_7m": 0.031,
  "dir_acc_k5_7m": 0.7262,
  "dir_fires_k5_7m": 504,
  "dir_fire_rate_k5_7m": 0.0033,
  "backtest_pnl_k5_7m": 413.982,
  "backtest_winrate_k5_7m": 0.7262,
  "lag1_autocorr_q50_7m": 0.3104,
  "pinball_15m_q10": 1.179151,
  "pinball_15m_q50": 2.124916,
  "pinball_15m_q90": 1.014744,
  "mae_15m": 4.249832,
  "coverage_15m": 0.7649,
  "calibration_gap_15m": 0.0351,
  "dir_acc_k5_15m": 0.8914,
  "dir_fires_k5_15m": 221,
  "dir_fire_rate_k5_15m": 0.0028,
  "backtest_pnl_k5_15m": 370.3793,
  "backtest_winrate_k5_15m": 0.8914,
  "lag1_autocorr_q50_15m": 0.4018
}
```