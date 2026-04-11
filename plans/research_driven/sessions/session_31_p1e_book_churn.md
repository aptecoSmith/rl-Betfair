# Session 31 — P1e: order-book churn rate feature

## Before you start — read these

- `../purpose.md`
- `../hard_constraints.md` — constraints 12, 13.
- `../analysis.md` §3
- `../proposals.md` P1
- `../progress.md` — confirm session 30 has landed.
- `env/features.py` — contains OBI, microprice, traded_delta,
  mid_drift from sessions 19–21.
- `env/betfair_env.py` — observation builder.

## Origin

From `research/research.txt` line 85:

> Real money pressure depends on:
> - How fast money is being matched
> - **Orders being added/cancelled**
> - Momentum of traded volume

The first and third items are covered by `traded_delta_T` and
`mid_drift_T` (session 21). The second — the rate of order-book
churn (new orders appearing, existing orders being pulled between
ticks) — was not included in any P1 session. The audit on
2026-04-11 flagged it as the one research signal we didn't build.

## Why it matters

OBI and microprice are *snapshots* of where the book is right now.
`traded_delta` is a *flow* of matched volume. But neither captures
the *intent* signal: someone adding £500 to the back side and
pulling it 2 seconds later (spoofing / testing the market) looks
identical in the snapshot (it's gone by the next tick) and in
traded delta (nothing matched). The churn rate — how much
liquidity was *offered and withdrawn* in a window — is the signal
that captures unstable liquidity, fake walls, and market-maker
repositioning.

In live markets this is a meaningful edge signal. Whether the
historical parquet has enough tick resolution to capture it is
the main risk (see below).

## Goal

Add a per-runner feature `book_churn_T` to the observation vector.

```
book_churn_T = Σ |size_change_per_level| over the last T seconds
             / (back_volume + lay_volume)
```

where `size_change_per_level` is the absolute difference in
available size at each price level between consecutive ticks,
summed across all visible levels on both sides. Normalised by
total visible volume to make it scale-invariant across runners
with different liquidity depths.

- High churn → unstable book, likely repositioning, less
  trustworthy visible liquidity.
- Low churn → stable book, what-you-see-is-what-you-get.

`book_churn_T = 0.0` on the first tick of a race (no prior
tick to diff against).

## Inputs — constraints to obey

1. **Same vendoring rule as sessions 19–21.** Pure function in
   `env/features.py`. No numpy, no env imports. ai-betfair
   imports via `rl_bridge`.
2. **State lives on the env, not on `features.py`.** The env
   maintains a per-runner "previous tick's ladder" snapshot to
   compute the diff. The pure function takes two ladder snapshots
   and returns the churn value.
3. **Schema bump.** One bump for this feature.
4. **`T` is wall-clock seconds** (same rule as session 21).
   Config key: `features.book_churn_window_s: 60`.
5. **Normalisation denominator clamped.** If
   `(back_volume + lay_volume) == 0`, return `0.0` (no book, no
   churn).

## Risks / unknowns

- **Tick resolution in the parquet.** If the historical data has
  5-second gaps between ticks, most add/cancel cycles will be
  invisible — the churn rate will be near-zero everywhere. Check
  the typical tick cadence for a sample race before writing the
  feature. If cadence is too coarse (> 2s between ticks on
  average), document the finding in `lessons_learnt.md` and
  consider parking this session.
- **Noise.** High churn could mean "active market maker
  repositioning" (useful signal) or "Betfair exchange jittering
  sizes due to rounding" (noise). The junk filter helps but
  doesn't eliminate it. If the feature is mostly noise, the
  retrain will show it (zero gradient on the column after N
  steps).

## Steps

1. **Check tick cadence.** Before writing any code, sample 3
   races from the parquet and compute the median inter-tick gap.
   If > 2 seconds, stop and record in `lessons_learnt.md`. If
   ≤ 2 seconds, proceed.

2. **Add `compute_book_churn(prev_back, prev_lay, curr_back,
   curr_lay, n)` to `env/features.py`.** Pure function. Takes
   two ladder snapshots (previous and current, top-N each side),
   returns the normalised churn float.

3. **Add per-runner previous-tick-ladder storage to the env.**
   On each tick, after computing the churn, overwrite the stored
   snapshot with the current tick's ladder.

4. **Add windowing.** Accumulate per-tick churn values in the
   same history-buffer pattern as session 21, then return the
   mean churn over the last `T` seconds.

5. **Wire into the observation builder** after the existing P1
   features.

6. **Bump obs schema version.**

7. **Expose in `info["debug_features"]`.**

8. **Add `features.book_churn_window_s: 60` to `config.yaml`.**

## Tests to add

Create `tests/research_driven/test_p1e_book_churn.py`:

1. **Pure function: identical ladders.** Churn = 0.0.
2. **Pure function: one level added.** Churn > 0.
3. **Pure function: one level removed.** Churn > 0 (same
   magnitude as (2) for same size).
4. **Pure function: empty book.** Returns 0.0.
5. **Pure function: respects `n`.** Changes beyond top-N are
   ignored.
6. **Env smoke.** First tick = 0.0; at least one mid-race tick
   is non-zero.
7. **Env determinism.** Same race twice → identical values.
8. **Schema-bump loader refuses pre-P1e checkpoint.**

All CPU, all fast.

## Manual tests

- **Open a race with a known fast-market move.** Confirm
  `book_churn_T` spikes when the visible book visibly
  rearranges. If it stays flat during a visible shuffle, the
  tick resolution is too coarse or the computation is wrong.

## Session exit criteria

- Step 1 (tick cadence check) passed, or session parked with
  documented reason.
- All 8 tests pass.
- All existing tests pass.
- `progress.md` Session 31 entry.
- `ui_additions.md` row for churn in per-runner panel.
- `master_todo.md` updated.
- Commit.

## Do not

- Do not skip the tick-cadence check. Writing a feature that
  can't resolve the signal it's meant to capture is wasted
  work.
- Do not add state to `env/features.py`.
- Do not change the action space, reward function, or matcher.
