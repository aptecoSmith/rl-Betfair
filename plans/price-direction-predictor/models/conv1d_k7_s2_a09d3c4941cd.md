# Predictor model card: conv1d_k7_s2_a09d3c4941cd

- session: S03
- seed: 2
- architecture: conv1d (k7)
- arch_kwargs: `{"kernel": 7, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V3
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 106249,
  "train_seconds": 53.8,
  "infer_us_per_row": 1.6114208847284317,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k7_s2_a09d3c4941cd.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.548803,
  "pinball_3m_q50": 0.917886,
  "pinball_3m_q90": 0.548901,
  "mae_3m": 1.835773,
  "coverage_3m": 0.7572,
  "calibration_gap_3m": 0.0428,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.2976,
  "pinball_7m_q10": 0.81876,
  "pinball_7m_q50": 1.453786,
  "pinball_7m_q90": 0.764581,
  "mae_7m": 2.907572,
  "coverage_7m": 0.7464,
  "calibration_gap_7m": 0.0536,
  "dir_acc_k5_7m": 0.5029,
  "dir_fires_k5_7m": 340,
  "dir_fire_rate_k5_7m": 0.0022,
  "backtest_pnl_k5_7m": 347.5698,
  "backtest_winrate_k5_7m": 0.5029,
  "lag1_autocorr_q50_7m": 0.6401,
  "pinball_15m_q10": 1.209381,
  "pinball_15m_q50": 2.216524,
  "pinball_15m_q90": 1.093601,
  "mae_15m": 4.433048,
  "coverage_15m": 0.7575,
  "calibration_gap_15m": 0.0425,
  "dir_acc_k5_15m": 0.6954,
  "dir_fires_k5_15m": 581,
  "dir_fire_rate_k5_15m": 0.0075,
  "backtest_pnl_k5_15m": 959.472,
  "backtest_winrate_k5_15m": 0.6954,
  "lag1_autocorr_q50_15m": 0.7282
}
```