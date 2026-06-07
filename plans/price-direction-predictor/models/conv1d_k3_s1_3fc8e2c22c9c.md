# Predictor model card: conv1d_k3_s1_3fc8e2c22c9c

- session: S06_neural
- seed: 1
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V4
- train corpus: tvl_mask_29d
- horizons: ['1m', '7m', '15m']
- output: pinball5 (quantiles [0.1, 0.3, 0.5, 0.7, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 49743,
  "train_seconds": 621.9,
  "infer_us_per_row": 2.6975758373737335,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s1_3fc8e2c22c9c.pt"
}
```

## Val metrics
```json
{
  "pinball_1m_q10": 0.339911,
  "pinball_1m_q30": 0.510709,
  "pinball_1m_q50": 0.554796,
  "pinball_1m_q70": 0.518257,
  "pinball_1m_q90": 0.343745,
  "mae_1m": 1.109591,
  "coverage_1m": 0.8205,
  "calibration_gap_1m": 0.0205,
  "dir_acc_k5_1m": 0.7185,
  "dir_fires_k5_1m": 135,
  "dir_fire_rate_k5_1m": 0.0006,
  "backtest_pnl_k5_1m": 127.2698,
  "backtest_winrate_k5_1m": 0.7185,
  "lag1_autocorr_q50_1m": -0.1045,
  "pinball_7m_q10": 0.787093,
  "pinball_7m_q30": 1.299745,
  "pinball_7m_q50": 1.41873,
  "pinball_7m_q70": 1.262701,
  "pinball_7m_q90": 0.714926,
  "mae_7m": 2.837461,
  "coverage_7m": 0.7709,
  "calibration_gap_7m": 0.0291,
  "dir_acc_k5_7m": 0.788,
  "dir_fires_k5_7m": 764,
  "dir_fire_rate_k5_7m": 0.005,
  "backtest_pnl_k5_7m": 755.9861,
  "backtest_winrate_k5_7m": 0.788,
  "lag1_autocorr_q50_7m": 0.0398,
  "pinball_15m_q10": 1.190475,
  "pinball_15m_q30": 1.968151,
  "pinball_15m_q50": 2.131213,
  "pinball_15m_q70": 1.870501,
  "pinball_15m_q90": 1.02332,
  "mae_15m": 4.262427,
  "coverage_15m": 0.7633,
  "calibration_gap_15m": 0.0367,
  "dir_acc_k5_15m": 0.8374,
  "dir_fires_k5_15m": 1076,
  "dir_fire_rate_k5_15m": 0.0138,
  "backtest_pnl_k5_15m": 1018.6762,
  "backtest_winrate_k5_15m": 0.8374,
  "lag1_autocorr_q50_15m": 0.2326
}
```