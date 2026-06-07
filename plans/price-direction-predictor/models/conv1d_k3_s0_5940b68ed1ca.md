# Predictor model card: conv1d_k3_s0_5940b68ed1ca

- session: S04_neural
- seed: 0
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
  "train_seconds": 415.6,
  "infer_us_per_row": 3.670342266559601,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s0_5940b68ed1ca.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.52005,
  "pinball_3m_q50": 0.910019,
  "pinball_3m_q90": 0.513509,
  "mae_3m": 1.820038,
  "coverage_3m": 0.7739,
  "calibration_gap_3m": 0.0261,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.8457,
  "pinball_7m_q10": 0.772061,
  "pinball_7m_q50": 1.417062,
  "pinball_7m_q90": 0.718647,
  "mae_7m": 2.834125,
  "coverage_7m": 0.7598,
  "calibration_gap_7m": 0.0402,
  "dir_acc_k5_7m": 0.75,
  "dir_fires_k5_7m": 24,
  "dir_fire_rate_k5_7m": 0.0002,
  "backtest_pnl_k5_7m": 23.546,
  "backtest_winrate_k5_7m": 0.75,
  "lag1_autocorr_q50_7m": 0.7807,
  "pinball_15m_q10": 1.168986,
  "pinball_15m_q50": 2.132195,
  "pinball_15m_q90": 1.036155,
  "mae_15m": 4.264391,
  "coverage_15m": 0.7545,
  "calibration_gap_15m": 0.0455,
  "dir_acc_k5_15m": 0.8988,
  "dir_fires_k5_15m": 168,
  "dir_fire_rate_k5_15m": 0.0022,
  "backtest_pnl_k5_15m": 516.3066,
  "backtest_winrate_k5_15m": 0.8988,
  "lag1_autocorr_q50_15m": 0.8111
}
```