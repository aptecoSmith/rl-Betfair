# Predictor model card: conv1d_k3_s0_358ebe0d2ec7

- session: S04_neural
- seed: 0
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
  "train_seconds": 40.6,
  "infer_us_per_row": 3.6812853068113327,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s0_358ebe0d2ec7.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.548958,
  "pinball_3m_q50": 0.917416,
  "pinball_3m_q90": 0.546705,
  "mae_3m": 1.834831,
  "coverage_3m": 0.7632,
  "calibration_gap_3m": 0.0368,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.2533,
  "pinball_7m_q10": 0.826315,
  "pinball_7m_q50": 1.44164,
  "pinball_7m_q90": 0.766113,
  "mae_7m": 2.883279,
  "coverage_7m": 0.7365,
  "calibration_gap_7m": 0.0635,
  "dir_acc_k5_7m": 0.5405,
  "dir_fires_k5_7m": 111,
  "dir_fire_rate_k5_7m": 0.0007,
  "backtest_pnl_k5_7m": 261.1923,
  "backtest_winrate_k5_7m": 0.5405,
  "lag1_autocorr_q50_7m": 0.4731,
  "pinball_15m_q10": 1.24045,
  "pinball_15m_q50": 2.187607,
  "pinball_15m_q90": 1.104172,
  "mae_15m": 4.375214,
  "coverage_15m": 0.7137,
  "calibration_gap_15m": 0.0863,
  "dir_acc_k5_15m": 0.6873,
  "dir_fires_k5_15m": 742,
  "dir_fire_rate_k5_15m": 0.0095,
  "backtest_pnl_k5_15m": 1058.0319,
  "backtest_winrate_k5_15m": 0.6873,
  "lag1_autocorr_q50_15m": 0.5518
}
```