---
plan: trajectory-retrieval-probe
status: parked
verdict: NEGATIVE (kNN loses to B1 by 12% MAE)
created: 2026-05-25
landed: 2026-05-25
---

# Findings — trajectory-retrieval-probe

## Verdict

**PARK.** Per the decision rule locked in [purpose.md](purpose.md)
before any data was looked at:

> kNN MAE = 0.15915, B1 MAE = 0.14168 → kNN is **12.3 % worse** than B1.
> "Loses to B1 across the board → Signal-to-noise so low that retrieval
> likely can't help. Park the idea."

The constant "price stays where it is" baseline beat the kNN
retrieval, and beat the linear-extrapolation and per-rank-prior
baselines too. The architecture-viability hypothesis the probe was
designed to test has its answer: under the 10 hand-engineered
features tried here, retrieval **does not** help predict the next
5 minutes of LTP.

## Phase 3 headline

(Query set, 6,934 rows, dates 2026-05-05 → 2026-05-14)

| Method | MAE | vs B1 | dir_acc | Notes |
|---|---|---|---|---|
| **B1 constant** | **0.14168** | — | (degenerate) | "price stays where it is" |
| B2 linear extrap | 0.17118 | **−20.8 %** | 0.483 | extrapolating slope is worse than random |
| B3 rank prior | 0.14292 | −0.9 % | 0.549 | per-favourite-rank mean target on index |
| **kNN k=5** | **0.15915** | **−12.3 %** | 0.517 | top-5 nearest neighbours by z-scored features |

Target log-return on the query set: |mean| ≈ 14 %, std 0.20 — prices
DO move a lot in the final 5 min pre-off. The moves just aren't
predictable from the 10 hand-engineered features the probe used.

## Why we didn't run Phases 4–5

The Phase 3 signal is decisive (kNN strictly worse than B1) and the
locked decision rule fires unambiguously. Phases 4 (per-venue /
per-rank breakdowns) and Phase 5 (validation pass) were skipped as
not-cost-effective:

- Phase 4 would reveal whether *some* slice of the data (e.g.
  high-volume favourites at major venues) is retrievable, but the
  follow-on of "narrow positive on a slice" doesn't justify the
  follow-on plan's cost. We'd still want a learned encoder to
  generalise.
- Phase 5 (validation MAE on 4,703 held-out rows) is unlikely to
  reverse a 12 % gap. It would harden the conclusion without
  changing it.

If a future thread wants to revisit this with richer features
(cross-runner, form data, learned encoder), Phases 4-5 are
re-runnable from the existing
`scratch/trajectory_retrieval/queries.parquet` in under an hour.

## What the negative result implies

### About retrieval architectures specifically

The kNN beat B2 (linear extrapolation) on MAE and on directional
accuracy — so the neighbour-aggregation IS extracting *some*
signal. It's just that the signal is small relative to the
variance of 5-min log-returns, and "the price stays here" is a
strictly better point estimate.

That maps to a known property of efficient-ish markets: short-
horizon price changes are dominated by noise; the conditional
mean given any cheap feature set is approximately zero. A kNN
with sample mean = ε plus prediction variance ≈ σ_target_std will
have higher MAE than always predicting zero, unless ε ≫ σ_neighbour.
For our setup, ε ≈ +0.023 (target mean on query set) and the
median neighbour-std is 0.130 — predicting the noisy neighbour
mean rather than zero loses MAE.

### About the current PPO+LSTM stack

The current parametric stack uses much richer features (cross-runner
context, time-attention, learned embeddings) and is optimising a
different objective (trading P&L through env mechanics, not
price prediction). So the retrieval-probe negative doesn't condemn
the parametric approach — it tells us that **pure short-horizon
LTP direction prediction is hard regardless of architecture**.

The current stack's value-add over "constant" comes from:
1. Spread capture via scalping pair lifecycle (locked PnL), NOT
   directional prediction.
2. Selective open behaviour (avoid pairs that will force-close),
   which is partially a fill-probability prediction problem — a
   different signal than the one this probe tested.
3. Risk awareness via the existing aux heads (`fill_prob_head`,
   `mature_prob_head`, `risk_head`).

None of those are tested by this probe. The negative result narrows
where retrieval could plausibly help: NOT in short-horizon LTP
direction; possibly in fill-probability / mature-probability
prediction where the signal-to-noise is better-suited to the
"this looks like that historical race" framing.

### What didn't get tested (and might be worth a separate probe)

The probe was strictly: *can we predict the next 5 min of LTP
direction from a hand-engineered embedding of the previous 25 min*?
**Three different retrieval questions remain open**:

1. **Fill-probability retrieval.** Predict whether a resting passive
   at price X will be matched within 5 min, given trajectory-so-far.
   Discrete binary target — better signal-to-noise than continuous
   log-returns. The `fill_prob_head` aux is the parametric version
   of this; a retrieval version would be cheap to run.

2. **Race-outcome retrieval.** Predict the winning runner from the
   embeddings at race start. This is what `betfair-predictors`
   already does parametrically; retrieval might be competitive
   because the signal-to-noise is higher than minute-by-minute
   prices.

3. **Mature-probability retrieval.** Predict whether a scalping pair
   opened at trajectory-state X will reach natural maturation.
   Same shape as #1 but with a different target definition.

None of these are blocked by this probe's negative; they're
distinct hypotheses with their own viability questions.

## Cost / artifacts

- **Time:** ~3 h end-to-end (data prep + features + bug fixes +
  baselines + writeup), one session.
- **Compute:** <2 min wall-clock across all phases. CPU-only.
- **Side-effects on production:** zero. Read-only against
  `data/processed/`; outputs in gitignored `scratch/`; lived on
  side branch `probe/trajectory-retrieval`. The active cohort
  (PID 46216) was unaffected.
- **Reusable artifacts** (in `scratch/trajectory_retrieval/`,
  gitignored, re-runnable from the script):
  - `ticks.parquet` — 3.55M ticks long-form. Dual-use for any
    other ad-hoc analysis of pre-off price action.
  - `queries.parquet` — 29,670 query rows with 10 z-scored
    features + target. Reusable for any other prediction
    target.
  - `results.parquet` — per-query predictions from all four
    methods + neighbour-std diagnostic.

The script itself ([scripts/trajectory_retrieval_probe.py](../../scripts/trajectory_retrieval_probe.py))
is preserved with `--phase 1/2/3` subcommands; can be re-run end-
to-end in <3 min.

## Two lessons worth keeping

Recorded in
[lessons_learnt.md](lessons_learnt.md) and promoted to operator
memory at `memory/feedback_feature_engineering_diagnostics.md`:

1. **Tick-direction bugs need value-domain checks**, not just
   shape-domain checks. The bug here survived TWO fix attempts
   because the no-lookahead smoke test and z-score sanity table
   both passed. Only `df.head(30)` on one named runner caught
   it.

2. **A single ~90σ z-score is more likely a bug than a real fat
   tail.** I almost paved over the bug by recommending robust
   normalisation. Investigate extreme rows before clipping.

These will fire any time we touch feature engineering in this
repo, regardless of probe topic.
