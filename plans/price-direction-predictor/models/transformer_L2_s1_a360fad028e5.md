# Predictor model card: transformer_L2_s1_a360fad028e5

- session: S03
- seed: 1
- architecture: transformer (L2)
- arch_kwargs: `{"depth": 2, "d_model": 64, "heads": 4, "ctx_ticks": 32, "dropout": 0.1}`
- feature variant: V3
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.0005, batch=512, max_epochs=20

## Run extras
```json
{
  "param_count": 104841,
  "train_seconds": 99.5,
  "infer_us_per_row": 5.521811544895172,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\transformer_L2_s1_a360fad028e5.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.617757,
  "pinball_3m_q50": 1.751422,
  "pinball_3m_q90": 0.594719,
  "mae_3m": 3.502843,
  "coverage_3m": 0.7764,
  "calibration_gap_3m": 0.0236,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.9958,
  "pinball_7m_q10": 1.150918,
  "pinball_7m_q50": 2.030717,
  "pinball_7m_q90": 1.233417,
  "mae_7m": 4.061434,
  "coverage_7m": 0.8462,
  "calibration_gap_7m": 0.0462,
  "dir_acc_k5_7m": 0.587,
  "dir_fires_k5_7m": 184,
  "dir_fire_rate_k5_7m": 0.0012,
  "backtest_pnl_k5_7m": 371.3485,
  "backtest_winrate_k5_7m": 0.587,
  "lag1_autocorr_q50_7m": 0.9874,
  "pinball_15m_q10": 1.243489,
  "pinball_15m_q50": 2.48756,
  "pinball_15m_q90": 1.294202,
  "mae_15m": 4.975119,
  "coverage_15m": 0.7861,
  "calibration_gap_15m": 0.0139,
  "dir_acc_k5_15m": 0.7576,
  "dir_fires_k5_15m": 33,
  "dir_fire_rate_k5_15m": 0.0004,
  "backtest_pnl_k5_15m": 45.7025,
  "backtest_winrate_k5_15m": 0.7576,
  "lag1_autocorr_q50_15m": 0.9888
}
```