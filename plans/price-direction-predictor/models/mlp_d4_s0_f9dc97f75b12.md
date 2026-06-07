# Predictor model card: mlp_d4_s0_f9dc97f75b12

- session: S03
- seed: 0
- architecture: mlp (d4)
- arch_kwargs: `{"depth": 4, "hidden": 128, "dropout": 0.1}`
- feature variant: V3
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 55177,
  "train_seconds": 57.3,
  "infer_us_per_row": 0.9837094694375992,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\mlp_d4_s0_f9dc97f75b12.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.545943,
  "pinball_3m_q50": 0.919513,
  "pinball_3m_q90": 0.546384,
  "mae_3m": 1.839025,
  "coverage_3m": 0.7893,
  "calibration_gap_3m": 0.0107,
  "dir_acc_k5_3m": 0.8333,
  "dir_fires_k5_3m": 6,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 6.1595,
  "backtest_winrate_k5_3m": 0.8333,
  "lag1_autocorr_q50_3m": 0.6795,
  "pinball_7m_q10": 0.81104,
  "pinball_7m_q50": 1.462836,
  "pinball_7m_q90": 0.768216,
  "mae_7m": 2.925672,
  "coverage_7m": 0.7756,
  "calibration_gap_7m": 0.0244,
  "dir_acc_k5_7m": 0.575,
  "dir_fires_k5_7m": 520,
  "dir_fire_rate_k5_7m": 0.0034,
  "backtest_pnl_k5_7m": 652.4216,
  "backtest_winrate_k5_7m": 0.575,
  "lag1_autocorr_q50_7m": 0.6681,
  "pinball_15m_q10": 1.21172,
  "pinball_15m_q50": 2.205733,
  "pinball_15m_q90": 1.085602,
  "mae_15m": 4.411467,
  "coverage_15m": 0.754,
  "calibration_gap_15m": 0.046,
  "dir_acc_k5_15m": 0.6801,
  "dir_fires_k5_15m": 497,
  "dir_fire_rate_k5_15m": 0.0064,
  "backtest_pnl_k5_15m": 745.2396,
  "backtest_winrate_k5_15m": 0.6801,
  "lag1_autocorr_q50_15m": 0.5703
}
```