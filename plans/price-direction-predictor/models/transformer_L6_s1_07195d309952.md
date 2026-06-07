# Predictor model card: transformer_L6_s1_07195d309952

- session: S03
- seed: 1
- architecture: transformer (L6)
- arch_kwargs: `{"depth": 6, "d_model": 64, "heads": 4, "ctx_ticks": 32, "dropout": 0.1}`
- feature variant: V3
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.0005, batch=512, max_epochs=20

## Run extras
```json
{
  "param_count": 304777,
  "train_seconds": 203.3,
  "infer_us_per_row": 22.714026272296906,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\transformer_L6_s1_07195d309952.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.810195,
  "pinball_3m_q50": 1.061021,
  "pinball_3m_q90": 0.696705,
  "mae_3m": 2.122041,
  "coverage_3m": 0.7525,
  "calibration_gap_3m": 0.0475,
  "dir_acc_k5_3m": 0.4149,
  "dir_fires_k5_3m": 94,
  "dir_fire_rate_k5_3m": 0.0005,
  "backtest_pnl_k5_3m": 6.5252,
  "backtest_winrate_k5_3m": 0.4149,
  "lag1_autocorr_q50_3m": 0.9781,
  "pinball_7m_q10": 0.986672,
  "pinball_7m_q50": 1.788779,
  "pinball_7m_q90": 1.065264,
  "mae_7m": 3.577557,
  "coverage_7m": 0.8284,
  "calibration_gap_7m": 0.0284,
  "dir_acc_k5_7m": 0.3474,
  "dir_fires_k5_7m": 521,
  "dir_fire_rate_k5_7m": 0.0034,
  "backtest_pnl_k5_7m": 488.4672,
  "backtest_winrate_k5_7m": 0.3474,
  "lag1_autocorr_q50_7m": 0.9813,
  "pinball_15m_q10": 1.443767,
  "pinball_15m_q50": 3.076556,
  "pinball_15m_q90": 1.240088,
  "mae_15m": 6.153112,
  "coverage_15m": 0.7784,
  "calibration_gap_15m": 0.0216,
  "dir_acc_k5_15m": 0.7419,
  "dir_fires_k5_15m": 31,
  "dir_fire_rate_k5_15m": 0.0004,
  "backtest_pnl_k5_15m": 43.8592,
  "backtest_winrate_k5_15m": 0.7419,
  "lag1_autocorr_q50_15m": 0.9989
}
```