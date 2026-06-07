# Predictor model card: conv1d_k5_s0_ae30c96da60f

- session: smoke
- seed: 0
- architecture: conv1d (k5)
- arch_kwargs: `{"kernel": 5, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V1
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=3

## Run extras
```json
{
  "param_count": 71366,
  "train_seconds": 64.6,
  "infer_us_per_row": 2.4177134037017822,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k5_s0_ae30c96da60f.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.548331,
  "pinball_3m_q50": 0.91713,
  "pinball_3m_q90": 0.549966,
  "mae_3m": 1.834259,
  "coverage_3m": 0.8231,
  "calibration_gap_3m": 0.0231,
  "dir_acc_k5_3m": 0.0,
  "dir_fires_k5_3m": 1,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": 0.0,
  "lag1_autocorr_q50_3m": 0.8933,
  "pinball_7m_q10": 0.813121,
  "pinball_7m_q50": 1.444079,
  "pinball_7m_q90": 0.770519,
  "mae_7m": 2.888157,
  "coverage_7m": 0.812,
  "calibration_gap_7m": 0.012,
  "dir_acc_k5_7m": 0.809,
  "dir_fires_k5_7m": 89,
  "dir_fire_rate_k5_7m": 0.0006,
  "backtest_pnl_k5_7m": 180.4265,
  "backtest_winrate_k5_7m": 0.809,
  "lag1_autocorr_q50_7m": 0.8888
}
```