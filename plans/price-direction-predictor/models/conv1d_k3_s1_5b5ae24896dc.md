# Predictor model card: conv1d_k3_s1_5b5ae24896dc

- session: S04_neural
- seed: 1
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V3
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 48393,
  "train_seconds": 70.3,
  "infer_us_per_row": 3.694789484143257,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s1_5b5ae24896dc.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.533946,
  "pinball_3m_q50": 0.917138,
  "pinball_3m_q90": 0.534454,
  "mae_3m": 1.834277,
  "coverage_3m": 0.7872,
  "calibration_gap_3m": 0.0128,
  "dir_acc_k5_3m": 0.4167,
  "dir_fires_k5_3m": 48,
  "dir_fire_rate_k5_3m": 0.0002,
  "backtest_pnl_k5_3m": 10.3275,
  "backtest_winrate_k5_3m": 0.4167,
  "lag1_autocorr_q50_3m": 0.1376,
  "pinball_7m_q10": 0.797518,
  "pinball_7m_q50": 1.468872,
  "pinball_7m_q90": 0.754173,
  "mae_7m": 2.937744,
  "coverage_7m": 0.7683,
  "calibration_gap_7m": 0.0317,
  "dir_acc_k5_7m": 0.6063,
  "dir_fires_k5_7m": 569,
  "dir_fire_rate_k5_7m": 0.0037,
  "backtest_pnl_k5_7m": 692.7649,
  "backtest_winrate_k5_7m": 0.6063,
  "lag1_autocorr_q50_7m": 0.4012,
  "pinball_15m_q10": 1.229354,
  "pinball_15m_q50": 2.214125,
  "pinball_15m_q90": 1.090781,
  "mae_15m": 4.428251,
  "coverage_15m": 0.7164,
  "calibration_gap_15m": 0.0836,
  "dir_acc_k5_15m": 0.709,
  "dir_fires_k5_15m": 1553,
  "dir_fire_rate_k5_15m": 0.0199,
  "backtest_pnl_k5_15m": 1430.858,
  "backtest_winrate_k5_15m": 0.709,
  "lag1_autocorr_q50_15m": 0.5173
}
```