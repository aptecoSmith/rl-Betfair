# Predictor model card: conv1d_k3_s2_953dadb5b866

- session: S06_neural
- seed: 2
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V2
- train corpus: tvl_mask_29d
- horizons: ['7m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 46467,
  "train_seconds": 354.2,
  "infer_us_per_row": 2.1334271878004074,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s2_953dadb5b866.pt"
}
```

## Val metrics
```json
{
  "pinball_7m_q10": 0.790507,
  "pinball_7m_q50": 1.425898,
  "pinball_7m_q90": 0.718617,
  "mae_7m": 2.851796,
  "coverage_7m": 0.7886,
  "calibration_gap_7m": 0.0114,
  "dir_acc_k5_7m": 0.0,
  "dir_fires_k5_7m": 2,
  "dir_fire_rate_k5_7m": 0.0,
  "backtest_pnl_k5_7m": -0.101,
  "backtest_winrate_k5_7m": 0.0,
  "lag1_autocorr_q50_7m": 0.88
}
```