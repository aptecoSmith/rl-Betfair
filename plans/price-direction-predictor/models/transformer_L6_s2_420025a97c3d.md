# Predictor model card: transformer_L6_s2_420025a97c3d

- session: S03
- seed: 2
- architecture: transformer (L6)
- arch_kwargs: `{"depth": 6, "d_model": 64, "heads": 4, "ctx_ticks": 32, "dropout": 0.1}`
- feature variant: V3
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.0005, batch=512, max_epochs=20

## Run extras
```json
{
  "param_count": 304777,
  "train_seconds": 287.7,
  "infer_us_per_row": 25.451648980379105,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\transformer_L6_s2_420025a97c3d.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.773912,
  "pinball_3m_q50": 0.9907,
  "pinball_3m_q90": 1.048274,
  "mae_3m": 1.9814,
  "coverage_3m": 0.7334,
  "calibration_gap_3m": 0.0666,
  "dir_acc_k5_3m": 0.0,
  "dir_fires_k5_3m": 4,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": -0.3051,
  "backtest_winrate_k5_3m": 0.0,
  "lag1_autocorr_q50_3m": 0.9843,
  "pinball_7m_q10": 1.078907,
  "pinball_7m_q50": 1.85937,
  "pinball_7m_q90": 0.889649,
  "mae_7m": 3.71874,
  "coverage_7m": 0.8103,
  "calibration_gap_7m": 0.0103,
  "dir_acc_k5_7m": 0.617,
  "dir_fires_k5_7m": 47,
  "dir_fire_rate_k5_7m": 0.0003,
  "backtest_pnl_k5_7m": 57.4344,
  "backtest_winrate_k5_7m": 0.617,
  "lag1_autocorr_q50_7m": 0.9964,
  "pinball_15m_q10": 1.522272,
  "pinball_15m_q50": 2.29521,
  "pinball_15m_q90": 1.259075,
  "mae_15m": 4.590421,
  "coverage_15m": 0.779,
  "calibration_gap_15m": 0.021,
  "dir_acc_k5_15m": 0.419,
  "dir_fires_k5_15m": 105,
  "dir_fire_rate_k5_15m": 0.0013,
  "backtest_pnl_k5_15m": 58.0167,
  "backtest_winrate_k5_15m": 0.419,
  "lag1_autocorr_q50_15m": 0.9857
}
```