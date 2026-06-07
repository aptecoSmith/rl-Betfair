# Predictor model card: lstm_tw32_s0_8df3a93f714c

- session: S04_neural
- seed: 0
- architecture: lstm (tw32)
- arch_kwargs: `{"time_window": 32, "hidden": 64, "layers": 2, "dropout": 0.1}`
- feature variant: V1
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 59017,
  "train_seconds": 455.7,
  "infer_us_per_row": 4.9211084842681885,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\lstm_tw32_s0_8df3a93f714c.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.535275,
  "pinball_3m_q50": 0.918652,
  "pinball_3m_q90": 0.52502,
  "mae_3m": 1.837305,
  "coverage_3m": 0.778,
  "calibration_gap_3m": 0.022,
  "dir_acc_k5_3m": 0.381,
  "dir_fires_k5_3m": 21,
  "dir_fire_rate_k5_3m": 0.0001,
  "backtest_pnl_k5_3m": 8.0878,
  "backtest_winrate_k5_3m": 0.381,
  "lag1_autocorr_q50_3m": 0.8865,
  "pinball_7m_q10": 0.795237,
  "pinball_7m_q50": 1.437345,
  "pinball_7m_q90": 0.741262,
  "mae_7m": 2.874689,
  "coverage_7m": 0.7725,
  "calibration_gap_7m": 0.0275,
  "dir_acc_k5_7m": 0.8272,
  "dir_fires_k5_7m": 162,
  "dir_fire_rate_k5_7m": 0.0011,
  "backtest_pnl_k5_7m": 170.7228,
  "backtest_winrate_k5_7m": 0.8272,
  "lag1_autocorr_q50_7m": 0.9001,
  "pinball_15m_q10": 1.198694,
  "pinball_15m_q50": 2.16906,
  "pinball_15m_q90": 1.058784,
  "mae_15m": 4.338119,
  "coverage_15m": 0.7568,
  "calibration_gap_15m": 0.0432,
  "dir_acc_k5_15m": 0.8585,
  "dir_fires_k5_15m": 544,
  "dir_fire_rate_k5_15m": 0.007,
  "backtest_pnl_k5_15m": 706.0706,
  "backtest_winrate_k5_15m": 0.8585,
  "lag1_autocorr_q50_15m": 0.9376
}
```