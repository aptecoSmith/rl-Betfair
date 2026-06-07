# Predictor model card: mlp_d2_s2_ab134b31c75f

- session: S03
- seed: 2
- architecture: mlp (d2)
- arch_kwargs: `{"depth": 2, "hidden": 128, "dropout": 0.1}`
- feature variant: V3
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 22153,
  "train_seconds": 44.5,
  "infer_us_per_row": 0.9867362678050995,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\mlp_d2_s2_ab134b31c75f.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.558359,
  "pinball_3m_q50": 0.919215,
  "pinball_3m_q90": 0.555076,
  "mae_3m": 1.838429,
  "coverage_3m": 0.7989,
  "calibration_gap_3m": 0.0011,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.7558,
  "pinball_7m_q10": 0.825467,
  "pinball_7m_q50": 1.456079,
  "pinball_7m_q90": 0.773997,
  "mae_7m": 2.912158,
  "coverage_7m": 0.7767,
  "calibration_gap_7m": 0.0233,
  "dir_acc_k5_7m": 0.8929,
  "dir_fires_k5_7m": 28,
  "dir_fire_rate_k5_7m": 0.0002,
  "backtest_pnl_k5_7m": 68.8173,
  "backtest_winrate_k5_7m": 0.8929,
  "lag1_autocorr_q50_7m": 0.6902,
  "pinball_15m_q10": 1.2323,
  "pinball_15m_q50": 2.203342,
  "pinball_15m_q90": 1.106485,
  "mae_15m": 4.406684,
  "coverage_15m": 0.7482,
  "calibration_gap_15m": 0.0518,
  "dir_acc_k5_15m": 0.9231,
  "dir_fires_k5_15m": 78,
  "dir_fire_rate_k5_15m": 0.001,
  "backtest_pnl_k5_15m": 155.8145,
  "backtest_winrate_k5_15m": 0.9231,
  "lag1_autocorr_q50_15m": 0.6457
}
```