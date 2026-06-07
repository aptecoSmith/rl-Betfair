# Predictor model card: conv1d_k3_s0_ed716bb3b1a2

- session: S06_neural
- seed: 0
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V2
- train corpus: tvl_mask_29d
- horizons: ['1m', '3m', '7m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 46857,
  "train_seconds": 263.9,
  "infer_us_per_row": 7.690861821174622,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s0_ed716bb3b1a2.pt"
}
```

## Val metrics
```json
{
  "pinball_1m_q10": 0.352758,
  "pinball_1m_q50": 0.563692,
  "pinball_1m_q90": 0.359328,
  "mae_1m": 1.127385,
  "coverage_1m": 0.7684,
  "calibration_gap_1m": 0.0316,
  "dir_acc_k5_1m": 1.0,
  "dir_fires_k5_1m": 4,
  "dir_fire_rate_k5_1m": 0.0,
  "backtest_pnl_k5_1m": 10.9147,
  "backtest_winrate_k5_1m": 1.0,
  "lag1_autocorr_q50_1m": 0.8277,
  "pinball_3m_q10": 0.516656,
  "pinball_3m_q50": 0.907872,
  "pinball_3m_q90": 0.504467,
  "mae_3m": 1.815745,
  "coverage_3m": 0.7822,
  "calibration_gap_3m": 0.0178,
  "dir_acc_k5_3m": 1.0,
  "dir_fires_k5_3m": 16,
  "dir_fire_rate_k5_3m": 0.0001,
  "backtest_pnl_k5_3m": 24.6924,
  "backtest_winrate_k5_3m": 1.0,
  "lag1_autocorr_q50_3m": 0.5806,
  "pinball_7m_q10": 0.779671,
  "pinball_7m_q50": 1.418167,
  "pinball_7m_q90": 0.71204,
  "mae_7m": 2.836334,
  "coverage_7m": 0.7604,
  "calibration_gap_7m": 0.0396,
  "dir_acc_k5_7m": 0.75,
  "dir_fires_k5_7m": 32,
  "dir_fire_rate_k5_7m": 0.0002,
  "backtest_pnl_k5_7m": 27.944,
  "backtest_winrate_k5_7m": 0.75,
  "lag1_autocorr_q50_7m": 0.8422
}
```