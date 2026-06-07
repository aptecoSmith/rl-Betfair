# Predictor model card: gbm_t100d4_s1_bae8cf86dc16

- session: S04
- seed: 1
- architecture: gbm (t100d4)
- arch_kwargs: `{"n_trees": 100, "max_depth": 4, "learning_rate": 0.05}`
- feature variant: V3
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m', '15m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.05, batch=1024, max_epochs=1

## Run extras
```json
{
  "param_count": 672,
  "train_seconds": 14.7,
  "infer_us_per_row": 3.569759428501129,
  "device": "cpu",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\gbm_t100d4_s1_bae8cf86dc16.joblib"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.532349,
  "pinball_3m_q50": 0.916985,
  "pinball_3m_q90": 0.534042,
  "mae_3m": 1.83397,
  "coverage_3m": 0.7895,
  "calibration_gap_3m": 0.0105,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 1.0,
  "pinball_7m_q10": 0.787957,
  "pinball_7m_q50": 1.44768,
  "pinball_7m_q90": 0.7496,
  "mae_7m": 2.89536,
  "coverage_7m": 0.7752,
  "calibration_gap_7m": 0.0248,
  "dir_acc_k5_7m": null,
  "dir_fires_k5_7m": 0,
  "dir_fire_rate_k5_7m": 0.0,
  "backtest_pnl_k5_7m": 0.0,
  "backtest_winrate_k5_7m": null,
  "lag1_autocorr_q50_7m": 1.0,
  "pinball_15m_q10": 1.174359,
  "pinball_15m_q50": 2.192036,
  "pinball_15m_q90": 1.047079,
  "mae_15m": 4.384072,
  "coverage_15m": 0.7622,
  "calibration_gap_15m": 0.0378,
  "dir_acc_k5_15m": null,
  "dir_fires_k5_15m": 0,
  "dir_fire_rate_k5_15m": 0.0,
  "backtest_pnl_k5_15m": 0.0,
  "backtest_winrate_k5_15m": null,
  "lag1_autocorr_q50_15m": 1.0
}
```