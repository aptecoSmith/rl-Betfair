# Predictor model card: mlp_d2_s0_3b9846d0e3cd

- session: smoke
- seed: 0
- architecture: mlp (d2)
- arch_kwargs: `{"depth": 2, "hidden": 64, "dropout": 0.1}`
- feature variant: V1
- train corpus: tvl_mask_29d
- horizons: ['3m', '7m']
- output: pinball3 (quantiles [0.1, 0.5, 0.9])
- training: lr=0.001, batch=2048, max_epochs=6

## Run extras
```json
{
  "param_count": 5638,
  "train_seconds": 61.5,
  "infer_us_per_row": 0.42654573917388916,
  "device": "cuda",
  "weights_path": "C:\\Users\\jsmit\\source\\repos\\rl-betfair\\.claude\\worktrees\\affectionate-proskuriakova-108942\\registry\\predictor\\mlp_d2_s0_3b9846d0e3cd.pt"
}
```

## Val metrics
```json
{
  "pinball_3m_q10": 0.554066,
  "pinball_3m_q50": 0.916687,
  "pinball_3m_q90": 0.554825,
  "mae_3m": 1.833374,
  "coverage_3m": 0.773,
  "calibration_gap_3m": 0.027,
  "dir_acc_k5_3m": null,
  "dir_fires_k5_3m": 0,
  "dir_fire_rate_k5_3m": 0.0,
  "backtest_pnl_k5_3m": 0.0,
  "backtest_winrate_k5_3m": null,
  "lag1_autocorr_q50_3m": 0.795,
  "pinball_7m_q10": 0.815032,
  "pinball_7m_q50": 1.447386,
  "pinball_7m_q90": 0.774391,
  "mae_7m": 2.894772,
  "coverage_7m": 0.77,
  "calibration_gap_7m": 0.03,
  "dir_acc_k5_7m": 0.8378,
  "dir_fires_k5_7m": 37,
  "dir_fire_rate_k5_7m": 0.0002,
  "backtest_pnl_k5_7m": 85.7986,
  "backtest_winrate_k5_7m": 0.8378,
  "lag1_autocorr_q50_7m": 0.8822
}
```