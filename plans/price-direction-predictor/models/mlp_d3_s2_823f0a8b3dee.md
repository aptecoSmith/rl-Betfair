# Predictor model card: mlp_d3_s2_823f0a8b3dee

- session: S03
- seed: 2
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
  "train_seconds": 67.6,
  "infer_us_per_row": 0.9229406714439392,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\mlp_d3_s2_823f0a8b3dee.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.542111,
  "pinball_3m_q50": 0.91839,
  "pinball_3m_q90": 0.543354,
  "mae_3m": 1.83678,
  "coverage_3m": 0.7898,
  "calibration_gap_3m": 0.0102,
  "dir_acc_k5_3m": 0.4286,
  "dir_fires_k5_3m": 7,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 2.9653,
  "backtest_winrate_k5_3m": 0.4286,
  "lag1_autocorr_q50_3m": 0.5263,
  "pinball_7m_q10": 0.806958,
  "pinball_7m_q50": 1.458161,
  "pinball_7m_q90": 0.766502,
  "mae_7m": 2.916321,
  "coverage_7m": 0.7779,
  "calibration_gap_7m": 0.0221,
  "dir_acc_k5_7m": 0.6323,
  "dir_fires_k5_7m": 699,
  "dir_fire_rate_k5_7m": 0.0046,
  "backtest_pnl_k5_7m": 504.3097,
  "backtest_winrate_k5_7m": 0.6323,
  "lag1_autocorr_q50_7m": 0.6822,
  "pinball_15m_q10": 1.206309,
  "pinball_15m_q50": 2.197583,
  "pinball_15m_q90": 1.087208,
  "mae_15m": 4.395165,
  "coverage_15m": 0.748,
  "calibration_gap_15m": 0.052,
  "dir_acc_k5_15m": 0.7054,
  "dir_fires_k5_15m": 774,
  "dir_fire_rate_k5_15m": 0.0099,
  "backtest_pnl_k5_15m": 1049.4969,
  "backtest_winrate_k5_15m": 0.7054,
  "lag1_autocorr_q50_15m": 0.615
}
```