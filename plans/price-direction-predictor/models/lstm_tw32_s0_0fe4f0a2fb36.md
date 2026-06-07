# Predictor model card: lstm_tw32_s0_0fe4f0a2fb36

- session: S04_neural
- seed: 0
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
  "train_seconds": 53.6,
  "infer_us_per_row": 4.573492333292961,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\lstm_tw32_s0_0fe4f0a2fb36.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.574354,
  "pinball_3m_q50": 0.917716,
  "pinball_3m_q90": 0.569242,
  "mae_3m": 1.835431,
  "coverage_3m": 0.7891,
  "calibration_gap_3m": 0.0109,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.9242,
  "pinball_7m_q10": 0.840773,
  "pinball_7m_q50": 1.452133,
  "pinball_7m_q90": 0.800147,
  "mae_7m": 2.904266,
  "coverage_7m": 0.7737,
  "calibration_gap_7m": 0.0263,
  "dir_acc_k5_7m": null,
  "dir_fires_k5_7m": 0,
  "dir_fire_rate_k5_7m": 0.0,
  "backtest_pnl_k5_7m": 0.0,
  "backtest_winrate_k5_7m": null,
  "lag1_autocorr_q50_7m": 0.9752,
  "pinball_15m_q10": 1.236832,
  "pinball_15m_q50": 2.215473,
  "pinball_15m_q90": 1.122518,
  "mae_15m": 4.430945,
  "coverage_15m": 0.7491,
  "calibration_gap_15m": 0.0509,
  "dir_acc_k5_15m": null,
  "dir_fires_k5_15m": 0,
  "dir_fire_rate_k5_15m": 0.0,
  "backtest_pnl_k5_15m": 0.0,
  "backtest_winrate_k5_15m": null,
  "lag1_autocorr_q50_15m": 0.9872
}
```