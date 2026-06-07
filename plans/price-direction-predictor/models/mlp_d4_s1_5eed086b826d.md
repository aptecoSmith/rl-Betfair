# Predictor model card: mlp_d4_s1_5eed086b826d

- session: S03
- seed: 1
- architecture: mlp (d4)
- arch_kwargs: `{"depth": 4, "hidden": 128, "dropout": 0.1}`
- feature variant: V3
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 55177,
  "train_seconds": 70.6,
  "infer_us_per_row": 1.034000888466835,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\mlp_d4_s1_5eed086b826d.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.545445,
  "pinball_3m_q50": 0.919125,
  "pinball_3m_q90": 0.543575,
  "mae_3m": 1.838251,
  "coverage_3m": 0.7898,
  "calibration_gap_3m": 0.0102,
  "dir_acc_k5_3m": 1.0,
  "dir_fires_k5_3m": 1,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 2.463,
  "backtest_winrate_k5_3m": 1.0,
  "lag1_autocorr_q50_3m": 0.5891,
  "pinball_7m_q10": 0.816403,
  "pinball_7m_q50": 1.463841,
  "pinball_7m_q90": 0.761371,
  "mae_7m": 2.927682,
  "coverage_7m": 0.7662,
  "calibration_gap_7m": 0.0338,
  "dir_acc_k5_7m": 0.8182,
  "dir_fires_k5_7m": 132,
  "dir_fire_rate_k5_7m": 0.0009,
  "backtest_pnl_k5_7m": 181.2061,
  "backtest_winrate_k5_7m": 0.8182,
  "lag1_autocorr_q50_7m": 0.6155,
  "pinball_15m_q10": 1.236178,
  "pinball_15m_q50": 2.197075,
  "pinball_15m_q90": 1.069291,
  "mae_15m": 4.39415,
  "coverage_15m": 0.7425,
  "calibration_gap_15m": 0.0575,
  "dir_acc_k5_15m": 0.7564,
  "dir_fires_k5_15m": 628,
  "dir_fire_rate_k5_15m": 0.0081,
  "backtest_pnl_k5_15m": 978.0411,
  "backtest_winrate_k5_15m": 0.7564,
  "lag1_autocorr_q50_15m": 0.5657
}
```