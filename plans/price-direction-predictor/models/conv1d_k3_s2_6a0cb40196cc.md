# Predictor model card: conv1d_k3_s2_6a0cb40196cc

- session: S04_neural
- seed: 2
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
  "train_seconds": 211.0,
  "infer_us_per_row": 3.759516403079033,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s2_6a0cb40196cc.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.523455,
  "pinball_3m_q50": 0.914838,
  "pinball_3m_q90": 0.512164,
  "mae_3m": 1.829677,
  "coverage_3m": 0.786,
  "calibration_gap_3m": 0.014,
  "dir_acc_k5_3m": 0.431,
  "dir_fires_k5_3m": 58,
  "dir_fire_rate_k5_3m": 0.0003,
  "backtest_pnl_k5_3m": 44.7665,
  "backtest_winrate_k5_3m": 0.431,
  "lag1_autocorr_q50_3m": 0.2609,
  "pinball_7m_q10": 0.783812,
  "pinball_7m_q50": 1.423767,
  "pinball_7m_q90": 0.717375,
  "mae_7m": 2.847534,
  "coverage_7m": 0.7667,
  "calibration_gap_7m": 0.0333,
  "dir_acc_k5_7m": 0.7407,
  "dir_fires_k5_7m": 729,
  "dir_fire_rate_k5_7m": 0.0047,
  "backtest_pnl_k5_7m": 631.2677,
  "backtest_winrate_k5_7m": 0.7407,
  "lag1_autocorr_q50_7m": 0.3382,
  "pinball_15m_q10": 1.180351,
  "pinball_15m_q50": 2.132807,
  "pinball_15m_q90": 1.021421,
  "mae_15m": 4.265615,
  "coverage_15m": 0.7504,
  "calibration_gap_15m": 0.0496,
  "dir_acc_k5_15m": 0.8656,
  "dir_fires_k5_15m": 588,
  "dir_fire_rate_k5_15m": 0.0075,
  "backtest_pnl_k5_15m": 706.0682,
  "backtest_winrate_k5_15m": 0.8656,
  "lag1_autocorr_q50_15m": 0.4303
}
```