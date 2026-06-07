# Predictor model card: mlp_d3_s1_9a05a4bc65fe

- session: S03
- seed: 1
- architecture: mlp (d3)
- arch_kwargs: `{"depth": 3, "hidden": 128, "dropout": 0.1}`
- feature variant: V3
- train corpus: tvl_required_10d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 38665,
  "train_seconds": 50.8,
  "infer_us_per_row": 0.9024515748023987,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\mlp_d3_s1_9a05a4bc65fe.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.55478,
  "pinball_3m_q50": 0.919514,
  "pinball_3m_q90": 0.544519,
  "mae_3m": 1.839027,
  "coverage_3m": 0.7675,
  "calibration_gap_3m": 0.0325,
  "dir_acc_k5_3m": 0.7778,
  "dir_fires_k5_3m": 18,
  "dir_fire_rate_k5_3m": 0.0001,
  "backtest_pnl_k5_3m": 18.7964,
  "backtest_winrate_k5_3m": 0.7778,
  "lag1_autocorr_q50_3m": 0.6914,
  "pinball_7m_q10": 0.823651,
  "pinball_7m_q50": 1.451636,
  "pinball_7m_q90": 0.762805,
  "mae_7m": 2.903271,
  "coverage_7m": 0.7499,
  "calibration_gap_7m": 0.0501,
  "dir_acc_k5_7m": 0.6485,
  "dir_fires_k5_7m": 202,
  "dir_fire_rate_k5_7m": 0.0013,
  "backtest_pnl_k5_7m": 180.0008,
  "backtest_winrate_k5_7m": 0.6485,
  "lag1_autocorr_q50_7m": 0.7013,
  "pinball_15m_q10": 1.245299,
  "pinball_15m_q50": 2.212903,
  "pinball_15m_q90": 1.085333,
  "mae_15m": 4.425805,
  "coverage_15m": 0.7308,
  "calibration_gap_15m": 0.0692,
  "dir_acc_k5_15m": 0.8077,
  "dir_fires_k5_15m": 260,
  "dir_fire_rate_k5_15m": 0.0033,
  "backtest_pnl_k5_15m": 292.945,
  "backtest_winrate_k5_15m": 0.8077,
  "lag1_autocorr_q50_15m": 0.6162
}
```