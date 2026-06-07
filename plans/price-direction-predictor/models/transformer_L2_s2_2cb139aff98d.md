# Predictor model card: transformer_L2_s2_2cb139aff98d

- session: S03
- seed: 2
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
  "train_seconds": 110.9,
  "infer_us_per_row": 7.543247193098068,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\transformer_L2_s2_2cb139aff98d.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.853634,
  "pinball_3m_q50": 1.030743,
  "pinball_3m_q90": 0.935999,
  "mae_3m": 2.061486,
  "coverage_3m": 0.8472,
  "calibration_gap_3m": 0.0472,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.9901,
  "pinball_7m_q10": 1.140195,
  "pinball_7m_q50": 1.661152,
  "pinball_7m_q90": 0.832448,
  "mae_7m": 3.322303,
  "coverage_7m": 0.7805,
  "calibration_gap_7m": 0.0195,
  "dir_acc_k5_7m": 0.6875,
  "dir_fires_k5_7m": 16,
  "dir_fire_rate_k5_7m": 0.0001,
  "backtest_pnl_k5_7m": 37.3108,
  "backtest_winrate_k5_7m": 0.6875,
  "lag1_autocorr_q50_7m": 0.9949,
  "pinball_15m_q10": 1.432674,
  "pinball_15m_q50": 2.991049,
  "pinball_15m_q90": 1.247535,
  "mae_15m": 5.982098,
  "coverage_15m": 0.7431,
  "calibration_gap_15m": 0.0569,
  "dir_acc_k5_15m": 0.6286,
  "dir_fires_k5_15m": 35,
  "dir_fire_rate_k5_15m": 0.0004,
  "backtest_pnl_k5_15m": 47.4546,
  "backtest_winrate_k5_15m": 0.6286,
  "lag1_autocorr_q50_15m": 0.9975
}
```