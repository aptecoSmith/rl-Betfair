# Predictor model card: conv1d_k3_s0_b035e7f93cf4

- session: S04_neural
- seed: 0
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V5
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 50121,
  "train_seconds": 36.5,
  "infer_us_per_row": 3.7695281207561493,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s0_b035e7f93cf4.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.565802,
  "pinball_3m_q50": 0.918611,
  "pinball_3m_q90": 0.554889,
  "mae_3m": 1.837222,
  "coverage_3m": 0.7591,
  "calibration_gap_3m": 0.0409,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.7453,
  "pinball_7m_q10": 0.838064,
  "pinball_7m_q50": 1.449945,
  "pinball_7m_q90": 0.781068,
  "mae_7m": 2.89989,
  "coverage_7m": 0.7362,
  "calibration_gap_7m": 0.0638,
  "dir_acc_k5_7m": 0.7778,
  "dir_fires_k5_7m": 45,
  "dir_fire_rate_k5_7m": 0.0003,
  "backtest_pnl_k5_7m": 80.4848,
  "backtest_winrate_k5_7m": 0.7778,
  "lag1_autocorr_q50_7m": 0.7639,
  "pinball_15m_q10": 1.240895,
  "pinball_15m_q50": 2.199573,
  "pinball_15m_q90": 1.101449,
  "mae_15m": 4.399146,
  "coverage_15m": 0.7218,
  "calibration_gap_15m": 0.0782,
  "dir_acc_k5_15m": 0.9122,
  "dir_fires_k5_15m": 148,
  "dir_fire_rate_k5_15m": 0.0019,
  "backtest_pnl_k5_15m": 215.5181,
  "backtest_winrate_k5_15m": 0.9122,
  "lag1_autocorr_q50_15m": 0.7677
}
```