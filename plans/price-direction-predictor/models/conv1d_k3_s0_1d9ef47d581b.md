# Predictor model card: conv1d_k3_s0_1d9ef47d581b

- session: S04_neural
- seed: 0
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V1
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 44937,
  "train_seconds": 428.0,
  "infer_us_per_row": 3.6121346056461334,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s0_1d9ef47d581b.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.523601,
  "pinball_3m_q50": 0.915617,
  "pinball_3m_q90": 0.514048,
  "mae_3m": 1.831234,
  "coverage_3m": 0.8033,
  "calibration_gap_3m": 0.0033,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.6526,
  "pinball_7m_q10": 0.777485,
  "pinball_7m_q50": 1.421028,
  "pinball_7m_q90": 0.72613,
  "mae_7m": 2.842056,
  "coverage_7m": 0.7946,
  "calibration_gap_7m": 0.0054,
  "dir_acc_k5_7m": 0.6765,
  "dir_fires_k5_7m": 238,
  "dir_fire_rate_k5_7m": 0.0015,
  "backtest_pnl_k5_7m": 215.621,
  "backtest_winrate_k5_7m": 0.6765,
  "lag1_autocorr_q50_7m": 0.7379,
  "pinball_15m_q10": 1.17292,
  "pinball_15m_q50": 2.123787,
  "pinball_15m_q90": 1.030455,
  "mae_15m": 4.247574,
  "coverage_15m": 0.7951,
  "calibration_gap_15m": 0.0049,
  "dir_acc_k5_15m": 0.8509,
  "dir_fires_k5_15m": 825,
  "dir_fire_rate_k5_15m": 0.0106,
  "backtest_pnl_k5_15m": 920.4999,
  "backtest_winrate_k5_15m": 0.8509,
  "lag1_autocorr_q50_15m": 0.7392
}
```