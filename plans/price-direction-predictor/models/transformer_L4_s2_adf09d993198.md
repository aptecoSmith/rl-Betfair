# Predictor model card: transformer_L4_s2_adf09d993198

- session: S03
- seed: 2
- architecture: transformer (L4)
- arch_kwargs: `{"depth": 4, "d_model": 64, "heads": 4, "ctx_ticks": 32, "dropout": 0.1}`
- feature variant: V3
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.0005, batch=512, max_epochs=20

## Run extras
```json
{
  "param_count": 204809,
  "train_seconds": 94.7,
  "infer_us_per_row": 16.03923738002777,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\transformer_L4_s2_adf09d993198.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.876855,
  "pinball_3m_q50": 1.403767,
  "pinball_3m_q90": 0.911593,
  "mae_3m": 2.807534,
  "coverage_3m": 0.8507,
  "calibration_gap_3m": 0.0507,
  "dir_acc_k5_3m": 1.0,
  "dir_fires_k5_3m": 1,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 2.0854,
  "backtest_winrate_k5_3m": 1.0,
  "lag1_autocorr_q50_3m": 0.9933,
  "pinball_7m_q10": 0.958896,
  "pinball_7m_q50": 1.693406,
  "pinball_7m_q90": 0.853313,
  "mae_7m": 3.386812,
  "coverage_7m": 0.7817,
  "calibration_gap_7m": 0.0183,
  "dir_acc_k5_7m": 0.4578,
  "dir_fires_k5_7m": 249,
  "dir_fire_rate_k5_7m": 0.0016,
  "backtest_pnl_k5_7m": 65.9153,
  "backtest_winrate_k5_7m": 0.4578,
  "lag1_autocorr_q50_7m": 0.9939,
  "pinball_15m_q10": 1.73298,
  "pinball_15m_q50": 2.353215,
  "pinball_15m_q90": 1.319829,
  "mae_15m": 4.70643,
  "coverage_15m": 0.8008,
  "calibration_gap_15m": 0.0008,
  "dir_acc_k5_15m": 0.3492,
  "dir_fires_k5_15m": 189,
  "dir_fire_rate_k5_15m": 0.0024,
  "backtest_pnl_k5_15m": -59.4434,
  "backtest_winrate_k5_15m": 0.3492,
  "lag1_autocorr_q50_15m": 0.9942
}
```