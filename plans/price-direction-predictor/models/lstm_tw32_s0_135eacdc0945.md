# Predictor model card: lstm_tw32_s0_135eacdc0945

- session: smoke
- seed: 0
- architecture: lstm (tw32)
- arch_kwargs: `{"time_window": 32, "hidden": 64, "layers": 2, "dropout": 0.1}`
- feature variant: V1
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=3

## Run extras
```json
{
  "param_count": 58822,
  "train_seconds": 67.0,
  "infer_us_per_row": 4.707369953393936,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\lstm_tw32_s0_135eacdc0945.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.559228,
  "pinball_3m_q50": 0.917603,
  "pinball_3m_q90": 0.558696,
  "mae_3m": 1.835205,
  "coverage_3m": 0.7851,
  "calibration_gap_3m": 0.0149,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.9497,
  "pinball_7m_q10": 0.826652,
  "pinball_7m_q50": 1.448345,
  "pinball_7m_q90": 0.786915,
  "mae_7m": 2.89669,
  "coverage_7m": 0.7734,
  "calibration_gap_7m": 0.0266,
  "dir_acc_k5_7m": null,
  "dir_fires_k5_7m": 0,
  "dir_fire_rate_k5_7m": 0.0,
  "backtest_pnl_k5_7m": 0.0,
  "backtest_winrate_k5_7m": null,
  "lag1_autocorr_q50_7m": 0.9397
}
```