# Predictor model card: conv1d_k3_s0_c0cebb87e424

- session: S05_neural
- seed: 0
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V2
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m', '15m']
- output: pinball5 (quantiles [0.1, 0.3, 0.5, 0.7, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 47247,
  "train_seconds": 336.8,
  "infer_us_per_row": 3.6745332181453705,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s0_c0cebb87e424.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.523779,
  "pinball_3m_q30": 0.837665,
  "pinball_3m_q50": 0.912243,
  "pinball_3m_q70": 0.845716,
  "pinball_3m_q90": 0.508652,
  "mae_3m": 1.824486,
  "coverage_3m": 0.7789,
  "calibration_gap_3m": 0.0211,
  "dir_acc_k5_3m": 0.5339,
  "dir_fires_k5_3m": 118,
  "dir_fire_rate_k5_3m": 0.0006,
  "backtest_pnl_k5_3m": 72.3556,
  "backtest_winrate_k5_3m": 0.5339,
  "lag1_autocorr_q50_3m": 0.7045,
  "pinball_7m_q10": 0.785354,
  "pinball_7m_q30": 1.304158,
  "pinball_7m_q50": 1.42434,
  "pinball_7m_q70": 1.268569,
  "pinball_7m_q90": 0.712968,
  "mae_7m": 2.848679,
  "coverage_7m": 0.7784,
  "calibration_gap_7m": 0.0216,
  "dir_acc_k5_7m": 0.7325,
  "dir_fires_k5_7m": 658,
  "dir_fire_rate_k5_7m": 0.0043,
  "backtest_pnl_k5_7m": 528.6176,
  "backtest_winrate_k5_7m": 0.7325,
  "lag1_autocorr_q50_7m": 0.7775,
  "pinball_15m_q10": 1.191341,
  "pinball_15m_q30": 1.972012,
  "pinball_15m_q50": 2.136139,
  "pinball_15m_q70": 1.875078,
  "pinball_15m_q90": 1.026673,
  "mae_15m": 4.272278,
  "coverage_15m": 0.7597,
  "calibration_gap_15m": 0.0403,
  "dir_acc_k5_15m": 0.8553,
  "dir_fires_k5_15m": 615,
  "dir_fire_rate_k5_15m": 0.0079,
  "backtest_pnl_k5_15m": 762.3609,
  "backtest_winrate_k5_15m": 0.8553,
  "lag1_autocorr_q50_15m": 0.7666
}
```