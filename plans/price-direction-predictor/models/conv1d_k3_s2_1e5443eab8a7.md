# Predictor model card: conv1d_k3_s2_1e5443eab8a7

- session: S06_neural
- seed: 2
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V2
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 46857,
  "train_seconds": 419.6,
  "infer_us_per_row": 3.695022314786911,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s2_1e5443eab8a7.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.521892,
  "pinball_3m_q50": 0.912131,
  "pinball_3m_q90": 0.510427,
  "mae_3m": 1.824263,
  "coverage_3m": 0.7779,
  "calibration_gap_3m": 0.0221,
  "dir_acc_k5_3m": 0.7541,
  "dir_fires_k5_3m": 61,
  "dir_fire_rate_k5_3m": 0.0003,
  "backtest_pnl_k5_3m": 54.2958,
  "backtest_winrate_k5_3m": 0.7541,
  "lag1_autocorr_q50_3m": 0.7681,
  "pinball_7m_q10": 0.779465,
  "pinball_7m_q50": 1.430921,
  "pinball_7m_q90": 0.714174,
  "mae_7m": 2.861842,
  "coverage_7m": 0.7725,
  "calibration_gap_7m": 0.0275,
  "dir_acc_k5_7m": 0.6692,
  "dir_fires_k5_7m": 1064,
  "dir_fire_rate_k5_7m": 0.0069,
  "backtest_pnl_k5_7m": 645.6632,
  "backtest_winrate_k5_7m": 0.6692,
  "lag1_autocorr_q50_7m": 0.8951,
  "pinball_15m_q10": 1.185404,
  "pinball_15m_q50": 2.132215,
  "pinball_15m_q90": 1.028103,
  "mae_15m": 4.264429,
  "coverage_15m": 0.7615,
  "calibration_gap_15m": 0.0385,
  "dir_acc_k5_15m": 0.8339,
  "dir_fires_k5_15m": 939,
  "dir_fire_rate_k5_15m": 0.0121,
  "backtest_pnl_k5_15m": 840.3482,
  "backtest_winrate_k5_15m": 0.8339,
  "lag1_autocorr_q50_15m": 0.8309
}
```