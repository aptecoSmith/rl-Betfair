---
plan: rewrite/phase-16-ensemble-market-state
parent_purpose: ./purpose.md
---

# Hard constraints

## §1 — Headline metric is multi-day mature rate, NOT single-day pnl

Phase-14 documented (and phase-15 v8 confirmed by accident):
single-day eval pnl is dominated by ±£600 day-to-day noise.
Phase-15 v8's "+£20 on 05-06" did NOT generalise to 3-day eval
in the big run; one agent at the same threshold went to 9% mat
rate vs another at 40%.

Every phase-16 session reports as PRIMARY metrics:
- Mature rate aggregated across all eval days (matured + closed
  pairs / pairs opened, summed across days then divided)
- Force-close rate (force_closed / pairs_opened, summed)
- Number of bets per agent per day

Eval pnl is reported but treated as informative-not-load-bearing.

## §2 — Sessions ship independently

Each session lands behind a feature flag (or in S03's case, a
features-set toggle). S01 (ensemble) is the most independent —
it's a model-level change, no obs schema change. S02 and S03
both bump OBS_SCHEMA_VERSION → cache invalidation, oracle
re-scan required, etc.

Validation cohort S04 combines all three. If S04 underperforms
the individual session smokes, debug by ablation. If S04
matches the smokes' weighted sum, ship.

## §3 — Ensemble: K=5 fixed in S01

K is a hyperparameter and worth tuning. But for S01, fix K=5
to avoid combinatorial explosion of variants. K=5 is a balance:
- K=2 too few — disagreement signal weak
- K=10 too many — BC compute 10× per agent

If S01 succeeds with K=5, future plans can sweep K.

## §4 — Ensemble: independent random seeds, identical architecture

All K predictors share architecture (LayerNorm + 64-hidden MLP),
input features (per-runner slice), and labels (direction binary
at horizon=60). They differ ONLY in:
- Initialization seed
- BC mini-batch sampling order

This isolates "consensus from independent noise" as the
mechanism. Don't introduce architecture variation across the
ensemble until K=5 same-arch baseline is validated.

## §5 — Market-state features (S02) are broadcast across runners

Each market-state feature is a SINGLE scalar per (race, tick).
At obs build time it's broadcast to all max_runners slots. The
direction head reads the runner's slice (per-runner features +
the broadcast market-state values).

This is intentionally simple — one race-level statistic, copied
into every runner's slice. Don't try to compute different
market-state features per-runner; that's what cross-runner
features are for (S03).

## §6 — Cross-runner features (S03) are race-relative

Each cross-runner feature for runner i is computed from the
WHOLE race state at the current tick:
- `volume_rank_in_field_i` requires comparing all runners
- `volume_share_in_field_i` = this runner's volume / sum of all
  runners' volumes
- `ltp_velocity_zscore_i` = (this runner's velocity - field
  mean) / field std

These features must be computed in `feature_engineer.py` with
access to the full per-tick runner list. RUNNER_KEYS expands
by 4 (or 5) per-runner columns — RUNNER_DIM increments by that
many.

## §7 — OBS_SCHEMA_VERSION bump on S02 and S03

Both add new features to RUNNER_KEYS or to the obs layout.
OBS_SCHEMA_VERSION must bump on each. Effects:
- Oracle cache re-scan needed (cohort runs check schema version)
- Direction-label cache re-scan needed
- Pre-S02 / pre-S03 checkpoints fail strict load (architecture
  hash break)

Same protocol as phase-13/14/15.

## §8 — Reward magnitudes UNCHANGED

Phase 16 adds features and ensemble — neither touches the
reward function. `race_pnl`, `scalping_locked_pnl`, force-close
accounting, all preserved. Scoreboard `raw_pnl_reward` from
phase-15 cohorts remains comparable to phase-16 rows.

## §9 — Defer rolling-calibration tracker (operator option 2)

The "track recent predictions vs realized outcomes intra-day,
adjust gate threshold based on rolling calibration" mechanism is
NOT in scope for phase 16. It's documented in
`lessons_learnt.md` "Deferred: rolling calibration tracker"
with rationale (60-tick label lag, lower priority than ensemble
+ features).

If S01-S03 deliver and we still see day-to-day variance, then
phase 17 (or later) addresses calibration tracking. The
mechanism is harder to validate (delayed labels, regime-shift
detection) and bigger surgery on the trainer.

## §10 — Test coverage

New code paths require regression tests:
- S01: ensemble's `min(K back probs)` vs threshold gate test
- S01: BC loop trains all K predictors (each weight tensor
  changes between BC start and BC end)
- S02: market-state features computed correctly per (race,
  tick); broadcast to all runners
- S03: cross-runner features sum/rank correctly across runners
- S03: cross-runner features are zero when only one priceable
  runner (degenerate case)

Mirror the phase-15 test pattern (test files in `tests/test_v2_*`).
