# Open Questions

These need an operator decision before any of the proposals in
`proposals.md` can be sized properly. They are not blocking session
10 itself — they are blocking the *next* tranche of work after it.

---

## Q1 — Is the simulator's job to model passive orders at all?

The current model is "the agent can only cross the spread, and the
matcher always gives it a clean fill". That is a deliberately
optimistic regime that lets the policy focus on **selection** (which
runner, when) rather than **execution** (how to get filled cheaply).

Two reasonable positions:

- **A. Selection-only.** Live execution is a separate problem solved
  by the live wrapper in `ai-betfair`. The simulator's job is to
  pick winners, not to model microstructure. → drop P3 + P4, do
  P1 + P2 only.
- **B. Execution-aware.** A policy trained without spread cost will
  over-trade in live and bleed to friction. The simulator must
  represent both regimes for the agent to learn restraint. → do
  the full P1 → P4 sequence.

A third path: **B-lite** = do P2 (spread cost as shaped reward)
without P3/P4. This preserves the selection-only action space but
charges the agent for the trades it does take, which may be enough.

**Operator decision needed:** A, B, or B-lite?

---

## Q2 — Are we willing to invalidate existing checkpoints again?

P1 (new obs features) and P3 (new action dim) both break checkpoint
compatibility. The LSTM and transformer arch sessions already burned
through one round of "everyone re-trains from scratch", and the
operator may want a moratorium until session 10's evaluation is
complete.

**Operator decision needed:** is "must remain compatible with current
checkpoints" a constraint, or is "fresh re-train OK if the gains
justify it" the prevailing view?

---

## Q3 — What's the eval target for "did this help"?

The proposals all end with "re-train and compare", but the comparison
metric isn't specified. Options:

- Raw daily P&L on the held-out 9-day eval window.
- Sharpe-ish: P&L / std of per-race P&L.
- Bet count × spread cost as a proxy for "would this bleed in live".
- Some weighted combo.

The choice matters because P2 (spread cost in shaped reward) will
*reduce* bet count and may *reduce* raw P&L on the eval window even
when it improves expected live performance. If we judge it on raw
P&L alone we will reject it for the wrong reason.

**RESOLVED 2026-04-08 (Session 22):** Raw daily P&L on the held-out
eval window. Operator chose option A. This metric is used for all
Phase 1 comparisons. P2 results may need a supplementary note
acknowledging that raw P&L can decrease even when live performance
improves — record both numbers when P2 runs.

---

## Q4 — Where does `traded_volume_at_price` deltas come from?

P4 (queue bookkeeping) needs per-price traded-volume **deltas
between ticks**, not the cumulative-from-market-start values our
parquet currently exposes. Two paths:

- Compute deltas in-env at runtime by snapshotting on placement and
  subtracting on each subsequent tick. Simple, no data migration.
- Backfill the parquet to include per-tick deltas alongside the
  cumulative columns. Faster at training time, costs disk and one
  ETL run.

**Operator decision needed:** is the disk + one-time ETL acceptable,
or do we keep the parquet schema frozen and compute at runtime?
(Default recommendation: compute at runtime; only backfill if
profiling shows it matters.)
