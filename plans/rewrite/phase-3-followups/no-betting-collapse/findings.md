---
plan: rewrite/phase-3-followups/no-betting-collapse
status: session-01-complete
opened: 2026-04-30
---

# No-betting-collapse — findings

## AMBER baseline (registry/v2_first_cohort_1777499178)

| Metric | Value |
|---|---|
| eval_day | 2026-04-29 |
| Bar 6a (mean fc_rate) | 0.308 |
| Bar 6b (ρ(entropy_coeff, fc_rate)) | −0.517 |
| Bar 6c (positive on raw P&L) | **0/12** |

Per-agent eval signature observed across all 12 agents:

- `eval_day_pnl == £0.00` exactly
- `eval_locked_pnl + eval_naked_pnl == 0.00` exactly
- `eval_winning_bets == 0`
- `eval_arbs_completed > 0` (range 2–4 per agent)

## Session 01 — verdict (a) BUG (with upstream caveat)

The `locked_pnl + naked_pnl == 0` pattern has **two root causes**, the
first of which is the deeper one:

### Root cause 1 (upstream, data) — eval-day parquet missing winners

`data/processed/2026-04-29.parquet` was processed before race results
landed: **0 of 2 markets have `winner_selection_id` populated**.
Confirmed via direct inspection on 2026-04-30:

```
2026-04-21 markets= 58 with_winner= 58
2026-04-22 markets= 66 with_winner= 66
2026-04-23 markets= 77 with_winner= 77
2026-04-24 markets= 92 with_winner= 92
2026-04-25 markets= 112 with_winner= 112
2026-04-26 markets= 28 with_winner= 28
2026-04-28 markets= 69 with_winner= 69
2026-04-29 markets=  2 with_winner=  0   ← eval day
```

When `BetfairEnv._settle_current_race` finds an empty
`winning_selection_ids` AND falsy `winner_selection_id`, it takes the
`BetManager.void_race` branch — every bet gets refunded to
`BetOutcome.VOID` with `pnl == 0`. So `race_pnl == 0` for every market
the agents traded on, and `winning_bets` (which counts
`BetOutcome.WON`) is zero by construction.

**Implication:** Bar 6c on this AMBER cohort is uninterpretable — no
agent could have produced positive cash on an eval day where every
race voided regardless of policy. The cohort's policy-collapse
diagnosis from `phase-3-cohort/findings.md` Session 04 (driven by
train-side `total_reward = −£1000…−£2200`) remains plausible, but
Bar 6c on this scoreboard does not falsify or confirm it.

### Root cause 2 (env, telemetry) — phantom locked_pnl on void

Even though the cash impact of every void was zero, the
`info["locked_pnl"]` telemetry reported the would-have-been lock
cash: a positive number ranging £3.19 to £17.35 across the 12 agents.
Mechanism:

1. `_settle_current_race` runs scalping diagnostics
   ([env/betfair_env.py:2849](env/betfair_env.py#L2849)
   pre-fix line numbering) **before** the void/settle dispatch:
   `scalping_locked_pnl += p["locked_pnl"]` is accumulated for every
   matured pair.
2. `get_paired_positions` computes `locked_pnl` from
   `matched_stake × price` — independent of outcome, valid only
   under the "if the result settles cleanly" assumption.
3. The dispatch then takes the void branch
   ([env/betfair_env.py:3002](env/betfair_env.py#L3002)) and
   `race_pnl == 0`.
4. The residual is computed as
   `naked_pnl = race_pnl − scalping_locked_pnl − scalping_closed_pnl
   − scalping_force_closed_pnl`
   ([env/betfair_env.py:3264](env/betfair_env.py#L3264)). With
   `race_pnl == 0` and only `locked` non-zero, this lands at
   `−scalping_locked_pnl` exactly.
5. `info["day_pnl"] = self._day_pnl += race_pnl` accumulates 0,
   while `info["locked_pnl"]` and `info["naked_pnl"]` accumulate
   the canceled pair, producing the cohort's load-bearing
   `locked + naked = 0` signature.

This was reproduced in unit form by
[`tests/test_v2_eval_pnl_accounting.py::TestEvalPnlAccounting::test_matured_pair_on_void_race_reports_zero_cash_buckets`](tests/test_v2_eval_pnl_accounting.py).

### Fix landed

[env/betfair_env.py:3002](env/betfair_env.py#L3002) — on the void
branch, zero the `scalping_locked_pnl` and
`scalping_early_lock_bonus` accumulators. Pair / arb COUNT
accumulators (`scalping_arbs_completed`, `scalping_arbs_naked` etc.)
are left intact — those record real market events (a passive did
fill, an aggressive did execute) that happened regardless of whether
the result eventually settled. `scalping_closed_pnl` and
`scalping_force_closed_pnl` are computed AFTER settle from
`agg.pnl + close.pnl` so are already 0 on the void path.

Regression test pinned in
[`tests/test_v2_eval_pnl_accounting.py`](tests/test_v2_eval_pnl_accounting.py):

- `test_one_matured_pair_day_pnl_equals_locked_pnl` — happy path:
  matured pair on a settle race produces `day_pnl == locked_pnl > 0,
  naked_pnl == 0`.
- `test_matured_pair_on_void_race_reports_zero_cash_buckets` —
  void path: matured pair on a void race produces `day_pnl == 0,
  locked_pnl == 0, naked_pnl == 0` (formerly `locked == +X,
  naked == −X`).

Existing scalping / arb / cohort / websocket suites pass unchanged
(217 + 213 tests, no regressions).

## Cohort re-run — deferred

The prompt's verdict-(a) recipe says: minimal fix → re-run AMBER
baseline cohort with the fix → check if Bar 6c flips.

**The re-run is not gated on the env fix alone.** Even with phantom
locked_pnl removed, the eval day's parquet still has 0/2 markets
with winners — `race_pnl` will still be 0 across the entire eval
day. Bar 6c will stay 0/12 because no agent can produce positive
cash on a day where every race voids regardless of policy. The
fix only changes `eval_locked_pnl` from "phantom +£X" to "honest
£0".

**Required next steps before any re-run:**

1. Re-process `data/processed/2026-04-29.parquet` once race results
   are available, OR
2. Pick a different eval day with full winner coverage (and update
   the protocol's seed-determinism story since the eval day is
   currently fixed by `select_days(seed=42)`).

These are out of scope for Session 01 (no env edits beyond the
accounting fix; no protocol changes). They are inputs to Session 02.

## Implications for downstream sessions

- **Session 02 (shaping ablations)** is now blocked on a clean eval
  day, not on a Session 01 verdict. The plan's ablation
  comparison-floor (`AMBER baseline 0/12 positive`) is *vacuously*
  zero — every ablation cohort that runs against an empty-winner
  eval day will also score 0/12, irrespective of the shaping
  changes.
- **Bar 6c interpretation** stands; the eval-day data must be
  honest before the metric is interpretable. The minimum-viable
  unblocker is re-processing 2026-04-29 with results, since
  `select_days(seed=42)` determinism is part of the cohort
  protocol's load-bearing comparison invariants
  (purpose.md §"What's locked").
- **Train-side P&L diagnosis from Session 04** (`total_reward
  = −£1000…−£2200` driving NOOP collapse) is unaffected by this
  finding. Train days had 100% winner coverage; the negative
  rewards came from real settlement losses on naked legs and the
  shaped-bonus penalty terms, not from voided races.

## Verdict summary

| Item | Status |
|---|---|
| Trace `day_pnl` end-to-end | ✓ done — env path traced, bug located |
| Synthetic regression test | ✓ done — `tests/test_v2_eval_pnl_accounting.py` |
| Reproduce cohort signature in unit form | ✓ done — `test_matured_pair_on_void_race_reports_zero_cash_buckets` |
| Verdict | (a) BUG — fix landed |
| Existing tests | ✓ 217 + 213 pass (no regressions) |
| AMBER cohort re-run | DEFERRED — eval-day data is the deeper blocker |

Plan status after Session 01: **session-01-complete; session-02
blocked on eval-day data fix.**
