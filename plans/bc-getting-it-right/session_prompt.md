# BC getting-it-right — session prompt

Paste the block below into a fresh session to execute this plan.

---

Execute `plans/bc-getting-it-right/`. Read these first, in order:
1. `plans/bc-getting-it-right/purpose.md` — the thesis.
2. `plans/bc-getting-it-right/explainer_plain_english.md` — plain-English context.
3. `plans/bc-getting-it-right/hard_constraints.md` — locked decisions. Pay special attention to §1 (THE METRIC), §3 (hard negatives), §8 (success bar), §9 (no PPO until the bar passes).
4. `plans/bc-getting-it-right/master_todo.md` — the measured experiment grid.
5. Background: `plans/imitation-first/findings.md` — the gates that led here (esp. Step 1: LightGBM holdout maturation AUC 0.76; Step 1b: the full-obs input-norm finding).

GOAL: make behavioural cloning produce a SELECTIVE, maturation-aware policy — one that opens a back trade on a runner ONLY when that trade's passive lay will actually fill (mature) before the off. Judge BC by the policy's held-out maturation AUC (vs LightGBM's 0.76 reference), NOT rollout mat% alone.

START with Step A (the metric harness) — it gates everything. Then Step B (hard negatives + `mature_prob_head` BCE supervision). Steps C/D only if B underperforms.

KEY CONTEXT already in place (do not rebuild):
- Maturation-conditioned oracle: `scan_day(maturation_conditioned=True, maturation_label_out=labels)` in `training_v2/arb_oracle.py` — labels every spread-placeable candidate matured / not (these are the positives + HARD negatives).
- Policy input-norm: `DiscreteLSTMPolicy(..., input_norm=True)` + `set_input_norm_stats(mean, std)` (opt-in, landed + tested). Full obs NEEDS this.
- `maturation_reward_mode` env flag is wired (for the LATER PPO step, not BC).
- 42-day maturation-conditioned full-obs + predictor caches already scanned in `data/oracle_cache_v2/`.
- Reusable scripts in `plans/imitation-first/_step1/`: `bc_fullnet_canary.py` (full-network BC + rollout eval), `bc_overfit_diag.py`, `maturation_predictability_probe.py` (the LightGBM 0.76 reference).
- `mature_prob_head` + `mature_prob_loss_weight` + `mature_prob_open_threshold` already exist in the v2 stack (CLAUDE.md "mature_prob_head feeds actor_head"); the strict mature-prob label already treats force-closed as 0 — matches our hard-negative definition.

CONVENTIONS:
- GPU: `--device cuda`.
- Iterate STANDALONE (the `_step1/` scripts), NOT the cohort runner (hard_constraints §7).
- Data split: 42 train days (Apr 6 → May 19), 7 reserved holdout (May 20, 21, 22, 25, 27, 28, 29). NEVER train / select / threshold-tune on the holdout.
- HARD negatives = force-closing spread-placeable opens, NOT random non-opportunities (§3).
- NO PPO until the success bar (§8) passes (operator directive).

SUCCESS BAR (§8): held-out maturation AUC ≥ ~0.70 (approaching 0.76) AND, at some confidence threshold, a fully-hedged holdout rollout (fc=120, close_walk=10) shows mat% well above the ~1% random baseline with locked positive — clearly better than the imitation-first BC (4% mat%, −£1513/7d). If the head plateaus ~0.60, that is an ARCHITECTURE finding (the pooled-LSTM actor bottleneck), not a tuning problem — write it up, don't chase knobs.

IF IT WORKS → `plans/bc-getting-it-right/on_success_next_steps.md` (BC→PPO, then GA cohort, then dry-run deploy). Do not start PPO autonomously without operator sign-off.

Write results to `plans/bc-getting-it-right/findings.md` as you go. Report everything against the metric (§1) + the holdout split.
