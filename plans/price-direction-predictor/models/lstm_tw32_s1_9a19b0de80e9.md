# Predictor model card: lstm_tw32_s1_9a19b0de80e9

- session: S04_neural
- seed: 1
- architecture: lstm (tw32)
- arch_kwargs: `{"time_window": 32, "hidden": 64, "layers": 2, "dropout": 0.1}`
- feature variant: V5
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 65929,
  "train_seconds": 36.9,
  "infer_us_per_row": 4.470581188797951,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\lstm_tw32_s1_9a19b0de80e9.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.577045,
  "pinball_3m_q50": 0.917701,
  "pinball_3m_q90": 0.576787,
  "mae_3m": 1.835401,
  "coverage_3m": 0.8147,
  "calibration_gap_3m": 0.0147,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.8829,
  "pinball_7m_q10": 0.844507,
  "pinball_7m_q50": 1.449006,
  "pinball_7m_q90": 0.801492,
  "mae_7m": 2.898012,
  "coverage_7m": 0.7855,
  "calibration_gap_7m": 0.0145,
  "dir_acc_k5_7m": null,
  "dir_fires_k5_7m": 0,
  "dir_fire_rate_k5_7m": 0.0,
  "backtest_pnl_k5_7m": 0.0,
  "backtest_winrate_k5_7m": null,
  "lag1_autocorr_q50_7m": 0.973,
  "pinball_15m_q10": 1.244423,
  "pinball_15m_q50": 2.204019,
  "pinball_15m_q90": 1.12667,
  "mae_15m": 4.408037,
  "coverage_15m": 0.7622,
  "calibration_gap_15m": 0.0378,
  "dir_acc_k5_15m": null,
  "dir_fires_k5_15m": 0,
  "dir_fire_rate_k5_15m": 0.0,
  "backtest_pnl_k5_15m": 0.0,
  "backtest_winrate_k5_15m": null,
  "lag1_autocorr_q50_15m": 0.992
}
```