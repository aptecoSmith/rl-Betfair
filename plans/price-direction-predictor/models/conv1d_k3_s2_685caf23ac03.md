# Predictor model card: conv1d_k3_s2_685caf23ac03

- session: S04_neural
- seed: 2
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V5
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 50121,
  "train_seconds": 47.0,
  "infer_us_per_row": 3.7993304431438446,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s2_685caf23ac03.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.560427,
  "pinball_3m_q50": 0.919031,
  "pinball_3m_q90": 0.551495,
  "mae_3m": 1.838062,
  "coverage_3m": 0.773,
  "calibration_gap_3m": 0.027,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.573,
  "pinball_7m_q10": 0.818896,
  "pinball_7m_q50": 1.452997,
  "pinball_7m_q90": 0.775299,
  "mae_7m": 2.905993,
  "coverage_7m": 0.7479,
  "calibration_gap_7m": 0.0521,
  "dir_acc_k5_7m": 0.5043,
  "dir_fires_k5_7m": 470,
  "dir_fire_rate_k5_7m": 0.0031,
  "backtest_pnl_k5_7m": 693.0563,
  "backtest_winrate_k5_7m": 0.5043,
  "lag1_autocorr_q50_7m": 0.6662,
  "pinball_15m_q10": 1.218397,
  "pinball_15m_q50": 2.207189,
  "pinball_15m_q90": 1.119454,
  "mae_15m": 4.414377,
  "coverage_15m": 0.7097,
  "calibration_gap_15m": 0.0903,
  "dir_acc_k5_15m": 0.5402,
  "dir_fires_k5_15m": 659,
  "dir_fire_rate_k5_15m": 0.0085,
  "backtest_pnl_k5_15m": 900.5599,
  "backtest_winrate_k5_15m": 0.5402,
  "lag1_autocorr_q50_15m": 0.6401
}
```