# Predictor model card: conv1d_k5_s1_baf481c5c57d

- session: S03
- seed: 1
- architecture: conv1d (k5)
- arch_kwargs: `{"kernel": 5, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V3
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 77321,
  "train_seconds": 59.8,
  "infer_us_per_row": 4.933448508381844,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k5_s1_baf481c5c57d.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.545474,
  "pinball_3m_q50": 0.91912,
  "pinball_3m_q90": 0.541613,
  "mae_3m": 1.83824,
  "coverage_3m": 0.7608,
  "calibration_gap_3m": 0.0392,
  "dir_acc_k5_3m": 1.0,
  "dir_fires_k5_3m": 2,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 1.8829,
  "backtest_winrate_k5_3m": 1.0,
  "lag1_autocorr_q50_3m": 0.3121,
  "pinball_7m_q10": 0.813017,
  "pinball_7m_q50": 1.467209,
  "pinball_7m_q90": 0.767908,
  "mae_7m": 2.934417,
  "coverage_7m": 0.7446,
  "calibration_gap_7m": 0.0554,
  "dir_acc_k5_7m": 0.5357,
  "dir_fires_k5_7m": 1036,
  "dir_fire_rate_k5_7m": 0.0067,
  "backtest_pnl_k5_7m": 978.4599,
  "backtest_winrate_k5_7m": 0.5357,
  "lag1_autocorr_q50_7m": 0.5765,
  "pinball_15m_q10": 1.216128,
  "pinball_15m_q50": 2.25411,
  "pinball_15m_q90": 1.104697,
  "mae_15m": 4.50822,
  "coverage_15m": 0.7456,
  "calibration_gap_15m": 0.0544,
  "dir_acc_k5_15m": 0.65,
  "dir_fires_k5_15m": 920,
  "dir_fire_rate_k5_15m": 0.0118,
  "backtest_pnl_k5_15m": 1146.3905,
  "backtest_winrate_k5_15m": 0.65,
  "lag1_autocorr_q50_15m": 0.6469
}
```