# Predictor model card: transformer_L4_s0_28bfba517c29

- session: smoke
- seed: 0
- architecture: transformer (L4)
- arch_kwargs: `{"depth": 4, "d_model": 64, "heads": 4, "ctx_ticks": 32, "dropout": 0.1}`
- feature variant: V1
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=3

## Run extras
```json
{
  "param_count": 203462,
  "train_seconds": 141.5,
  "infer_us_per_row": 6.1157625168561935,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\transformer_L4_s0_28bfba517c29.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 1.829422,
  "pinball_3m_q50": 1.240157,
  "pinball_3m_q90": 0.749862,
  "mae_3m": 2.480314,
  "coverage_3m": 0.5311,
  "calibration_gap_3m": 0.2689,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.9936,
  "pinball_7m_q10": 1.187,
  "pinball_7m_q50": 1.890418,
  "pinball_7m_q90": 1.052289,
  "mae_7m": 3.780837,
  "coverage_7m": 0.6689,
  "calibration_gap_7m": 0.1311,
  "dir_acc_k5_7m": null,
  "dir_fires_k5_7m": 0,
  "dir_fire_rate_k5_7m": 0.0,
  "backtest_pnl_k5_7m": 0.0,
  "backtest_winrate_k5_7m": null,
  "lag1_autocorr_q50_7m": 0.9918
}
```