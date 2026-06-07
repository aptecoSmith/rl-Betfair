# Predictor model card: lstm_tw32_s2_dcb28e172fc9

- session: S04_neural
- seed: 2
- architecture: lstm (tw32)
- arch_kwargs: `{"time_window": 32, "hidden": 64, "layers": 2, "dropout": 0.1}`
- feature variant: V2
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 61577,
  "train_seconds": 441.1,
  "infer_us_per_row": 4.334608092904091,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\lstm_tw32_s2_dcb28e172fc9.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.535138,
  "pinball_3m_q50": 0.917237,
  "pinball_3m_q90": 0.520857,
  "mae_3m": 1.834475,
  "coverage_3m": 0.786,
  "calibration_gap_3m": 0.014,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.9379,
  "pinball_7m_q10": 0.789298,
  "pinball_7m_q50": 1.428687,
  "pinball_7m_q90": 0.724641,
  "mae_7m": 2.857374,
  "coverage_7m": 0.7777,
  "calibration_gap_7m": 0.0223,
  "dir_acc_k5_7m": 0.871,
  "dir_fires_k5_7m": 62,
  "dir_fire_rate_k5_7m": 0.0004,
  "backtest_pnl_k5_7m": 80.3419,
  "backtest_winrate_k5_7m": 0.871,
  "lag1_autocorr_q50_7m": 0.9491,
  "pinball_15m_q10": 1.178331,
  "pinball_15m_q50": 2.13661,
  "pinball_15m_q90": 1.032154,
  "mae_15m": 4.27322,
  "coverage_15m": 0.7502,
  "calibration_gap_15m": 0.0498,
  "dir_acc_k5_15m": 0.9098,
  "dir_fires_k5_15m": 687,
  "dir_fire_rate_k5_15m": 0.0088,
  "backtest_pnl_k5_15m": 686.2453,
  "backtest_winrate_k5_15m": 0.9098,
  "lag1_autocorr_q50_15m": 0.9469
}
```