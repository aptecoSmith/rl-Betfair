# Predictor model card: conv1d_k3_s2_ce18f784e13d

- session: S03
- seed: 2
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
  "train_seconds": 49.4,
  "infer_us_per_row": 3.7481077015399933,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s2_ce18f784e13d.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.542685,
  "pinball_3m_q50": 0.918297,
  "pinball_3m_q90": 0.540814,
  "mae_3m": 1.836594,
  "coverage_3m": 0.8072,
  "calibration_gap_3m": 0.0072,
  "dir_acc_k5_3m": 0.5,
  "dir_fires_k5_3m": 2,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 4.4735,
  "backtest_winrate_k5_3m": 0.5,
  "lag1_autocorr_q50_3m": 0.1401,
  "pinball_7m_q10": 0.807728,
  "pinball_7m_q50": 1.454533,
  "pinball_7m_q90": 0.768412,
  "mae_7m": 2.909065,
  "coverage_7m": 0.7991,
  "calibration_gap_7m": 0.0009,
  "dir_acc_k5_7m": 0.6,
  "dir_fires_k5_7m": 105,
  "dir_fire_rate_k5_7m": 0.0007,
  "backtest_pnl_k5_7m": 251.1718,
  "backtest_winrate_k5_7m": 0.6,
  "lag1_autocorr_q50_7m": 0.4973,
  "pinball_15m_q10": 1.199505,
  "pinball_15m_q50": 2.215278,
  "pinball_15m_q90": 1.100951,
  "mae_15m": 4.430557,
  "coverage_15m": 0.7718,
  "calibration_gap_15m": 0.0282,
  "dir_acc_k5_15m": 0.6794,
  "dir_fires_k5_15m": 627,
  "dir_fire_rate_k5_15m": 0.008,
  "backtest_pnl_k5_15m": 989.3933,
  "backtest_winrate_k5_15m": 0.6794,
  "lag1_autocorr_q50_15m": 0.4868
}
```