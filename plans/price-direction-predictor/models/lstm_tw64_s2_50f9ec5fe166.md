# Predictor model card: lstm_tw64_s2_50f9ec5fe166

- session: S03
- seed: 2
- architecture: lstm (tw64)
- arch_kwargs: `{"time_window": 64, "hidden": 64, "layers": 2, "dropout": 0.1}`
- feature variant: V3
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 63625,
  "train_seconds": 48.7,
  "infer_us_per_row": 1.4528632164001465,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\lstm_tw64_s2_50f9ec5fe166.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.565683,
  "pinball_3m_q50": 0.917889,
  "pinball_3m_q90": 0.563057,
  "mae_3m": 1.835777,
  "coverage_3m": 0.7938,
  "calibration_gap_3m": 0.0062,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.8377,
  "pinball_7m_q10": 0.827684,
  "pinball_7m_q50": 1.451158,
  "pinball_7m_q90": 0.791048,
  "mae_7m": 2.902315,
  "coverage_7m": 0.7748,
  "calibration_gap_7m": 0.0252,
  "dir_acc_k5_7m": null,
  "dir_fires_k5_7m": 0,
  "dir_fire_rate_k5_7m": 0.0,
  "backtest_pnl_k5_7m": 0.0,
  "backtest_winrate_k5_7m": null,
  "lag1_autocorr_q50_7m": 0.9556,
  "pinball_15m_q10": 1.236751,
  "pinball_15m_q50": 2.212049,
  "pinball_15m_q90": 1.122031,
  "mae_15m": 4.424098,
  "coverage_15m": 0.7417,
  "calibration_gap_15m": 0.0583,
  "dir_acc_k5_15m": null,
  "dir_fires_k5_15m": 0,
  "dir_fire_rate_k5_15m": 0.0,
  "backtest_pnl_k5_15m": 0.0,
  "backtest_winrate_k5_15m": null,
  "lag1_autocorr_q50_15m": 0.9749
}
```