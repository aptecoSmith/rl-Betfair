# Predictor model card: conv1d_k3_s0_f3c2f5eea3a1

- session: S03
- seed: 0
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
  "train_seconds": 70.0,
  "infer_us_per_row": 3.754161298274994,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s0_f3c2f5eea3a1.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.531054,
  "pinball_3m_q50": 0.91344,
  "pinball_3m_q90": 0.539634,
  "mae_3m": 1.826881,
  "coverage_3m": 0.7756,
  "calibration_gap_3m": 0.0244,
  "dir_acc_k5_3m": 0.75,
  "dir_fires_k5_3m": 4,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 7.6991,
  "backtest_winrate_k5_3m": 0.75,
  "lag1_autocorr_q50_3m": 0.157,
  "pinball_7m_q10": 0.80041,
  "pinball_7m_q50": 1.440676,
  "pinball_7m_q90": 0.759702,
  "mae_7m": 2.881353,
  "coverage_7m": 0.7516,
  "calibration_gap_7m": 0.0484,
  "dir_acc_k5_7m": 0.5563,
  "dir_fires_k5_7m": 773,
  "dir_fire_rate_k5_7m": 0.005,
  "backtest_pnl_k5_7m": 971.6011,
  "backtest_winrate_k5_7m": 0.5563,
  "lag1_autocorr_q50_7m": 0.4398,
  "pinball_15m_q10": 1.211445,
  "pinball_15m_q50": 2.176979,
  "pinball_15m_q90": 1.077908,
  "mae_15m": 4.353957,
  "coverage_15m": 0.7361,
  "calibration_gap_15m": 0.0639,
  "dir_acc_k5_15m": 0.7146,
  "dir_fires_k5_15m": 953,
  "dir_fire_rate_k5_15m": 0.0122,
  "backtest_pnl_k5_15m": 1231.6379,
  "backtest_winrate_k5_15m": 0.7146,
  "lag1_autocorr_q50_15m": 0.547
}
```