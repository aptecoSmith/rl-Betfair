# Predictor model card: conv1d_k3_s0_9ff5d4a2aa72

- session: S06_neural
- seed: 0
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V4
- train corpus: tvl_mask_29d
- horizons: ['7m']
- output: pinball5 (quantiles [0.1, 0.3, 0.5, 0.7, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 49093,
  "train_seconds": 257.7,
  "infer_us_per_row": 3.0819792300462723,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s0_9ff5d4a2aa72.pt"
}
```

## Val metrics
```json
{
  "pinball_7m_q10": 0.788763,
  "pinball_7m_q30": 1.308964,
  "pinball_7m_q50": 1.430079,
  "pinball_7m_q70": 1.282472,
  "pinball_7m_q90": 0.729847,
  "mae_7m": 2.860157,
  "coverage_7m": 0.786,
  "calibration_gap_7m": 0.014,
  "dir_acc_k5_7m": 0.7218,
  "dir_fires_k5_7m": 266,
  "dir_fire_rate_k5_7m": 0.0017,
  "backtest_pnl_k5_7m": 276.5765,
  "backtest_winrate_k5_7m": 0.7218,
  "lag1_autocorr_q50_7m": 0.248
}
```