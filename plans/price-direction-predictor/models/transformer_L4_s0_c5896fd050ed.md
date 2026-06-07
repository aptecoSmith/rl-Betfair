# Predictor model card: transformer_L4_s0_c5896fd050ed

- session: S03
- seed: 0
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
  "train_seconds": 83.2,
  "infer_us_per_row": 16.48860052227974,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\transformer_L4_s0_c5896fd050ed.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.623126,
  "pinball_3m_q50": 1.378598,
  "pinball_3m_q90": 0.72861,
  "mae_3m": 2.757196,
  "coverage_3m": 0.7974,
  "calibration_gap_3m": 0.0026,
  "dir_acc_k5_3m": 0.0476,
  "dir_fires_k5_3m": 42,
  "dir_fire_rate_k5_3m": 0.0002,
  "backtest_pnl_k5_3m": -3.3978,
  "backtest_winrate_k5_3m": 0.0476,
  "lag1_autocorr_q50_3m": 0.9959,
  "pinball_7m_q10": 1.201214,
  "pinball_7m_q50": 2.239551,
  "pinball_7m_q90": 1.105412,
  "mae_7m": 4.479103,
  "coverage_7m": 0.7812,
  "calibration_gap_7m": 0.0188,
  "dir_acc_k5_7m": 0.1336,
  "dir_fires_k5_7m": 217,
  "dir_fire_rate_k5_7m": 0.0014,
  "backtest_pnl_k5_7m": -66.9222,
  "backtest_winrate_k5_7m": 0.1336,
  "lag1_autocorr_q50_7m": 0.9975,
  "pinball_15m_q10": 1.377329,
  "pinball_15m_q50": 2.42087,
  "pinball_15m_q90": 1.334914,
  "mae_15m": 4.841739,
  "coverage_15m": 0.7219,
  "calibration_gap_15m": 0.0781,
  "dir_acc_k5_15m": 0.425,
  "dir_fires_k5_15m": 40,
  "dir_fire_rate_k5_15m": 0.0005,
  "backtest_pnl_k5_15m": 7.5879,
  "backtest_winrate_k5_15m": 0.425,
  "lag1_autocorr_q50_15m": 0.9937
}
```