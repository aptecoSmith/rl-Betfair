# Predictor model card: conv1d_k3_s1_8fac17f7b5a3

- session: S04_neural
- seed: 1
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V4
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 49353,
  "train_seconds": 57.6,
  "infer_us_per_row": 3.7404242902994156,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s1_8fac17f7b5a3.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.542959,
  "pinball_3m_q50": 0.91667,
  "pinball_3m_q90": 0.542079,
  "mae_3m": 1.83334,
  "coverage_3m": 0.768,
  "calibration_gap_3m": 0.032,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.2056,
  "pinball_7m_q10": 0.808546,
  "pinball_7m_q50": 1.44799,
  "pinball_7m_q90": 0.762097,
  "mae_7m": 2.89598,
  "coverage_7m": 0.7516,
  "calibration_gap_7m": 0.0484,
  "dir_acc_k5_7m": 0.625,
  "dir_fires_k5_7m": 64,
  "dir_fire_rate_k5_7m": 0.0004,
  "backtest_pnl_k5_7m": 131.7545,
  "backtest_winrate_k5_7m": 0.625,
  "lag1_autocorr_q50_7m": 0.4859,
  "pinball_15m_q10": 1.222428,
  "pinball_15m_q50": 2.212796,
  "pinball_15m_q90": 1.087802,
  "mae_15m": 4.425593,
  "coverage_15m": 0.7252,
  "calibration_gap_15m": 0.0748,
  "dir_acc_k5_15m": 0.6473,
  "dir_fires_k5_15m": 465,
  "dir_fire_rate_k5_15m": 0.006,
  "backtest_pnl_k5_15m": 889.0727,
  "backtest_winrate_k5_15m": 0.6473,
  "lag1_autocorr_q50_15m": 0.5695
}
```