# Predictor model card: conv1d_k3_s1_ec4b502a2e47

- session: S05_neural
- seed: 1
- architecture: conv1d (k3)
- arch_kwargs: `{"kernel": 3, "layers": 4, "channels": 64, "dropout": 0.1}`
- feature variant: V2
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m', '15m']
- output: pinball5 (quantiles [0.1, 0.3, 0.5, 0.7, 0.9])
- training: lr=0.001, batch=1024, max_epochs=20

## Run extras
```json
{
  "param_count": 47247,
  "train_seconds": 174.6,
  "infer_us_per_row": 3.713183104991913,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\conv1d_k3_s1_ec4b502a2e47.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.532881,
  "pinball_3m_q30": 0.851291,
  "pinball_3m_q50": 0.915973,
  "pinball_3m_q70": 0.85438,
  "pinball_3m_q90": 0.516914,
  "mae_3m": 1.831946,
  "coverage_3m": 0.7966,
  "calibration_gap_3m": 0.0034,
  "dir_acc_k5_3m": 1.0,
  "dir_fires_k5_3m": 3,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 13.4206,
  "backtest_winrate_k5_3m": 1.0,
  "lag1_autocorr_q50_3m": 0.7497,
  "pinball_7m_q10": 0.787267,
  "pinball_7m_q30": 1.310762,
  "pinball_7m_q50": 1.428857,
  "pinball_7m_q70": 1.278611,
  "pinball_7m_q90": 0.723677,
  "mae_7m": 2.857714,
  "coverage_7m": 0.7889,
  "calibration_gap_7m": 0.0111,
  "dir_acc_k5_7m": 0.7793,
  "dir_fires_k5_7m": 213,
  "dir_fire_rate_k5_7m": 0.0014,
  "backtest_pnl_k5_7m": 223.7068,
  "backtest_winrate_k5_7m": 0.7793,
  "lag1_autocorr_q50_7m": 0.8931,
  "pinball_15m_q10": 1.180316,
  "pinball_15m_q30": 1.977939,
  "pinball_15m_q50": 2.149276,
  "pinball_15m_q70": 1.883793,
  "pinball_15m_q90": 1.028536,
  "mae_15m": 4.298552,
  "coverage_15m": 0.7739,
  "calibration_gap_15m": 0.0261,
  "dir_acc_k5_15m": 0.8378,
  "dir_fires_k5_15m": 1054,
  "dir_fire_rate_k5_15m": 0.0135,
  "backtest_pnl_k5_15m": 863.6493,
  "backtest_winrate_k5_15m": 0.8378,
  "lag1_autocorr_q50_15m": 0.8381
}
```