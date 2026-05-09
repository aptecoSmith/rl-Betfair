# Predictor model card: conv1d_k3_s1_9659e9e9c3fb

- session: S06_neural
- seed: 1
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V2
- train corpus: tvl_mask_29d
- horizons: ['1m', '3m', '7m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 46857,
  "train_seconds": 365.7,
  "infer_us_per_row": 3.6996789276599884,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s1_9659e9e9c3fb.pt"
}
```

## Val metrics
```json
{
  "pinball_1m_q10": 0.355415,
  "pinball_1m_q50": 0.563904,
  "pinball_1m_q90": 0.355083,
  "mae_1m": 1.127808,
  "coverage_1m": 0.7956,
  "calibration_gap_1m": 0.0044,
  "dir_acc_k5_1m": 0.7215,
  "dir_fires_k5_1m": 79,
  "dir_fire_rate_k5_1m": 0.0004,
  "backtest_pnl_k5_1m": 65.3809,
  "backtest_winrate_k5_1m": 0.7215,
  "lag1_autocorr_q50_1m": 0.7314,
  "pinball_3m_q10": 0.515504,
  "pinball_3m_q50": 0.906996,
  "pinball_3m_q90": 0.498315,
  "mae_3m": 1.813991,
  "coverage_3m": 0.7728,
  "calibration_gap_3m": 0.0272,
  "dir_acc_k5_3m": 0.8772,
  "dir_fires_k5_3m": 57,
  "dir_fire_rate_k5_3m": 0.0003,
  "backtest_pnl_k5_3m": 37.8449,
  "backtest_winrate_k5_3m": 0.8772,
  "lag1_autocorr_q50_3m": 0.6197,
  "pinball_7m_q10": 0.774076,
  "pinball_7m_q50": 1.419182,
  "pinball_7m_q90": 0.708912,
  "mae_7m": 2.838364,
  "coverage_7m": 0.7677,
  "calibration_gap_7m": 0.0323,
  "dir_acc_k5_7m": 0.8009,
  "dir_fires_k5_7m": 648,
  "dir_fire_rate_k5_7m": 0.0042,
  "backtest_pnl_k5_7m": 651.7265,
  "backtest_winrate_k5_7m": 0.8009,
  "lag1_autocorr_q50_7m": 0.7788
}
```
## S09 test metrics (sealed, May 4–6)
```json
{
  "test_dir_acc_k5_7m": 0.7875,
  "test_dir_fires_k5_7m": 753,
  "test_backtest_pnl_k5_7m": 675.82,
  "test_rows": 214836,
  "test_dates": "2026-05-04 to 2026-05-06"
}
```

**Champion model.** Selected 2026-05-09. See `registry/predictor/production/manifest.json`.
