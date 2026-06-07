# Predictor model card: transformer_L6_s0_283b51190938

- session: S03
- seed: 0
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
  "train_seconds": 306.5,
  "infer_us_per_row": 17.29000359773636,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\transformer_L6_s0_283b51190938.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.618499,
  "pinball_3m_q50": 1.295328,
  "pinball_3m_q90": 0.645341,
  "mae_3m": 2.590657,
  "coverage_3m": 0.7595,
  "calibration_gap_3m": 0.0405,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.9927,
  "pinball_7m_q10": 0.922619,
  "pinball_7m_q50": 1.582437,
  "pinball_7m_q90": 1.195323,
  "mae_7m": 3.164874,
  "coverage_7m": 0.6568,
  "calibration_gap_7m": 0.1432,
  "dir_acc_k5_7m": 0.5227,
  "dir_fires_k5_7m": 88,
  "dir_fire_rate_k5_7m": 0.0006,
  "backtest_pnl_k5_7m": 56.5582,
  "backtest_winrate_k5_7m": 0.5227,
  "lag1_autocorr_q50_7m": 0.9829,
  "pinball_15m_q10": 1.607311,
  "pinball_15m_q50": 2.31349,
  "pinball_15m_q90": 1.192596,
  "mae_15m": 4.626981,
  "coverage_15m": 0.7767,
  "calibration_gap_15m": 0.0233,
  "dir_acc_k5_15m": 0.5,
  "dir_fires_k5_15m": 164,
  "dir_fire_rate_k5_15m": 0.0021,
  "backtest_pnl_k5_15m": 75.7459,
  "backtest_winrate_k5_15m": 0.5,
  "lag1_autocorr_q50_15m": 0.9756
}
```