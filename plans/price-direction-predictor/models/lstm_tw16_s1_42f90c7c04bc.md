# Predictor model card: lstm_tw16_s1_42f90c7c04bc

- session: S03
- seed: 1
- architecture: lstm (tw16)
- arch_kwargs: `{"time_window": 16, "hidden": 64, "layers": 2, "dropout": 0.1}`
- feature variant: V3
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 63625,
  "train_seconds": 53.6,
  "infer_us_per_row": 4.809116944670677,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\lstm_tw16_s1_42f90c7c04bc.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.563645,
  "pinball_3m_q50": 0.917656,
  "pinball_3m_q90": 0.554416,
  "mae_3m": 1.835313,
  "coverage_3m": 0.7869,
  "calibration_gap_3m": 0.0131,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.809,
  "pinball_7m_q10": 0.828835,
  "pinball_7m_q50": 1.4555,
  "pinball_7m_q90": 0.778279,
  "mae_7m": 2.911,
  "coverage_7m": 0.769,
  "calibration_gap_7m": 0.031,
  "dir_acc_k5_7m": null,
  "dir_fires_k5_7m": 0,
  "dir_fire_rate_k5_7m": 0.0,
  "backtest_pnl_k5_7m": 0.0,
  "backtest_winrate_k5_7m": null,
  "lag1_autocorr_q50_7m": 0.8874,
  "pinball_15m_q10": 1.216058,
  "pinball_15m_q50": 2.204512,
  "pinball_15m_q90": 1.108386,
  "mae_15m": 4.409024,
  "coverage_15m": 0.7464,
  "calibration_gap_15m": 0.0536,
  "dir_acc_k5_15m": null,
  "dir_fires_k5_15m": 0,
  "dir_fire_rate_k5_15m": 0.0,
  "backtest_pnl_k5_15m": 0.0,
  "backtest_winrate_k5_15m": null,
  "lag1_autocorr_q50_15m": 0.9182
}
```