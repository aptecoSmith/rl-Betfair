# Predictor model card: conv1d_k3_s0_bedb0bbf6219

- session: S06_neural
- seed: 0
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
  "train_seconds": 186.9,
  "infer_us_per_row": 1.305481418967247,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s0_bedb0bbf6219.pt"
}
```

## Val metrics
```json
{
  "pinball_1m_q10": 0.351825,
  "pinball_1m_q30": 0.520656,
  "pinball_1m_q50": 0.561602,
  "pinball_1m_q70": 0.530363,
  "pinball_1m_q90": 0.35956,
  "mae_1m": 1.123205,
  "coverage_1m": 0.8087,
  "calibration_gap_1m": 0.0087,
  "dir_acc_k5_1m": 1.0,
  "dir_fires_k5_1m": 1,
  "dir_fire_rate_k5_1m": 0.0,
  "backtest_pnl_k5_1m": 0.2938,
  "backtest_winrate_k5_1m": 1.0,
  "lag1_autocorr_q50_1m": -0.111,
  "pinball_7m_q10": 0.800362,
  "pinball_7m_q30": 1.310753,
  "pinball_7m_q50": 1.428411,
  "pinball_7m_q70": 1.287798,
  "pinball_7m_q90": 0.732138,
  "mae_7m": 2.856822,
  "coverage_7m": 0.7756,
  "calibration_gap_7m": 0.0244,
  "dir_acc_k5_7m": 0.375,
  "dir_fires_k5_7m": 48,
  "dir_fire_rate_k5_7m": 0.0003,
  "backtest_pnl_k5_7m": 39.7945,
  "backtest_winrate_k5_7m": 0.375,
  "lag1_autocorr_q50_7m": 0.0635,
  "pinball_15m_q10": 1.196555,
  "pinball_15m_q30": 1.986161,
  "pinball_15m_q50": 2.158979,
  "pinball_15m_q70": 1.895306,
  "pinball_15m_q90": 1.039163,
  "mae_15m": 4.317959,
  "coverage_15m": 0.7638,
  "calibration_gap_15m": 0.0362,
  "dir_acc_k5_15m": 0.895,
  "dir_fires_k5_15m": 181,
  "dir_fire_rate_k5_15m": 0.0023,
  "backtest_pnl_k5_15m": 305.1591,
  "backtest_winrate_k5_15m": 0.895,
  "lag1_autocorr_q50_15m": 0.352
}
```