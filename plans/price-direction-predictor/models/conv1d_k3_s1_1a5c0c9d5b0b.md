# Predictor model card: conv1d_k3_s1_1a5c0c9d5b0b

- session: S04_neural
- seed: 1
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V1
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 44937,
  "train_seconds": 154.7,
  "infer_us_per_row": 3.5779085010290146,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s1_1a5c0c9d5b0b.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.542255,
  "pinball_3m_q50": 0.916216,
  "pinball_3m_q90": 0.540279,
  "mae_3m": 1.832432,
  "coverage_3m": 0.8051,
  "calibration_gap_3m": 0.0051,
  "dir_acc_k5_3m": 0.25,
  "dir_fires_k5_3m": 8,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 14.3333,
  "backtest_winrate_k5_3m": 0.25,
  "lag1_autocorr_q50_3m": 0.8916,
  "pinball_7m_q10": 0.807438,
  "pinball_7m_q50": 1.43893,
  "pinball_7m_q90": 0.751053,
  "mae_7m": 2.87786,
  "coverage_7m": 0.8012,
  "calibration_gap_7m": 0.0012,
  "dir_acc_k5_7m": 0.8167,
  "dir_fires_k5_7m": 60,
  "dir_fire_rate_k5_7m": 0.0004,
  "backtest_pnl_k5_7m": 166.7626,
  "backtest_winrate_k5_7m": 0.8167,
  "lag1_autocorr_q50_7m": 0.7577,
  "pinball_15m_q10": 1.189174,
  "pinball_15m_q50": 2.16529,
  "pinball_15m_q90": 1.052352,
  "mae_15m": 4.330579,
  "coverage_15m": 0.7986,
  "calibration_gap_15m": 0.0014,
  "dir_acc_k5_15m": 0.9282,
  "dir_fires_k5_15m": 209,
  "dir_fire_rate_k5_15m": 0.0027,
  "backtest_pnl_k5_15m": 392.3854,
  "backtest_winrate_k5_15m": 0.9282,
  "lag1_autocorr_q50_15m": 0.7934
}
```