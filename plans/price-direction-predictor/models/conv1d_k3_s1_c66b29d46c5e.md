# Predictor model card: conv1d_k3_s1_c66b29d46c5e

- session: S03
- seed: 1
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V3
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 48393,
  "train_seconds": 40.0,
  "infer_us_per_row": 3.723660483956337,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s1_c66b29d46c5e.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.545846,
  "pinball_3m_q50": 0.916374,
  "pinball_3m_q90": 0.541905,
  "mae_3m": 1.832749,
  "coverage_3m": 0.7919,
  "calibration_gap_3m": 0.0081,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.204,
  "pinball_7m_q10": 0.806995,
  "pinball_7m_q50": 1.448977,
  "pinball_7m_q90": 0.765322,
  "mae_7m": 2.897954,
  "coverage_7m": 0.7771,
  "calibration_gap_7m": 0.0229,
  "dir_acc_k5_7m": 0.6,
  "dir_fires_k5_7m": 60,
  "dir_fire_rate_k5_7m": 0.0004,
  "backtest_pnl_k5_7m": 92.2349,
  "backtest_winrate_k5_7m": 0.6,
  "lag1_autocorr_q50_7m": 0.4767,
  "pinball_15m_q10": 1.214063,
  "pinball_15m_q50": 2.211514,
  "pinball_15m_q90": 1.096713,
  "mae_15m": 4.423028,
  "coverage_15m": 0.7482,
  "calibration_gap_15m": 0.0518,
  "dir_acc_k5_15m": 0.7607,
  "dir_fires_k5_15m": 305,
  "dir_fire_rate_k5_15m": 0.0039,
  "backtest_pnl_k5_15m": 476.7447,
  "backtest_winrate_k5_15m": 0.7607,
  "lag1_autocorr_q50_15m": 0.5652
}
```