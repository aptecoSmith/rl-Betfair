# 05 — Launch cohort

See `session_prompts/00_autonomous_full_run.md` Phase 5 for the
full driver.

Mirror predecessor launch verbatim with:

- `--race-confidence-threshold 0.50` (inherited)
- `--predictor-p-win-lay-threshold <Phase 1 value>` (expect 0.20)
- `--lay-price-max <Phase 1 value>` (expect 20)
- All else identical (12 × 8 × 6, seed 42, same 6 safety genes)

**`force_close_before_off_seconds = 0` during training.** Do
NOT set the override — preserves naked-variance signal per
`memory/project_force_close_train_vs_deploy.md`.

Arm TWO watchers (both fire at 96 rows on the same scoreboard):

1. `/tmp/auto_reeval_layq_no_forceclose.sh` — fires reeval with
   `force_close = 0` (apples-to-apples vs predecessor).
2. `/tmp/auto_reeval_layq_forceclose120.sh` — fires reeval with
   `--reward-overrides force_close_before_off_seconds=120`
   (deployment-realistic).

Heartbeat 1 h until both reevals complete (~12 h cohort +
~40 min for the two reevals).
