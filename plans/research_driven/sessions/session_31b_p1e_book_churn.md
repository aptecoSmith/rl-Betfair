# Session 31b — P1e: order-book churn rate feature (revised)

## Why this replaces session 31

Session 31 parked itself because the tick cadence (6–10s median)
failed a ≤2s threshold. That threshold was wrong — it assumed the
feature's only value was capturing sub-second spoofing cycles. At
6–10s cadence the feature still measures something the agent needs:
**how much did the visible book rearrange between the ticks it can
actually act on?**

The agent can only act on tick boundaries. A book that is stable
between ticks is trustworthy; a book that rearranged heavily is
not. The agent doesn't need sub-second resolution to use this — it
needs to know "is the book I'm looking at right now the same book
that was here last time I looked?".

The revised feature drops the cadence gate entirely and reframes
the computation for the data we actually have.

## Before you start — read these

- `../purpose.md`
- `../hard_constraints.md` — constraints 12, 13.
- `../analysis.md` §3
- `../proposals.md` P1e
- `../progress.md` — session 31 (parked) entry and session 30.
- `../lessons_learnt.md` — session 31 entry about tick cadence.
- `env/features.py` — contains OBI, microprice, traded_delta,
  mid_drift from sessions 19–21.
- `env/betfair_env.py` — observation builder.

## Goal

Add a per-runner feature `book_churn` to the observation vector
that measures how much the visible order book changed between the
current tick and the previous tick.

```
book_churn = Σ |curr_size_i − prev_size_i| across all visible levels
           / (total_visible_volume + ε)
```

where `curr_size_i` and `prev_size_i` are the available sizes at
each price level on both sides of the book, and
`total_visible_volume` is the sum of all current visible sizes
(with a small epsilon to avoid division by zero).

- High churn → the book reshuffled heavily since the last tick.
  Visible liquidity may not be reliable.
- Low churn → the book is stable. What-you-see-is-what-you-get.
- Zero on the first tick of a race (no previous tick to diff).

This is a **per-tick** feature, not a windowed one. Each tick
measures the change from the immediately preceding tick. If a
windowed average is useful, it can be added later using the same
history-buffer pattern from session 21 — but start with the raw
per-tick value and see what the network does with it.

## Inputs — constraints to obey

1. **Same vendoring rule as sessions 19–21.** Pure function in
   `env/features.py`. No numpy, no env imports. ai-betfair
   imports via `rl_bridge`.
2. **State lives on the env, not on `features.py`.** The env
   maintains a per-runner "previous tick's ladder" snapshot. The
   pure function takes two ladder snapshots and returns the
   churn value.
3. **Schema bump.** One bump for this feature.
4. **Normalisation denominator clamped.** If total visible volume
   is zero, return `0.0`.

## Steps

1. **Add `compute_book_churn(prev_back, prev_lay, curr_back,
   curr_lay, n)` to `env/features.py`.** Pure function. Takes
   two sets of ladder levels (previous and current, top-N each
   side), returns the normalised churn float.

   Implementation detail: build a dict of `{price: size}` for
   each snapshot, then iterate all prices present in either
   snapshot and sum the absolute differences. This handles levels
   that appear or disappear between ticks (their full size counts
   as churn).

2. **Add per-runner previous-tick-ladder storage to the env.**
   A dict keyed by selection_id, value is `(prev_back_levels,
   prev_lay_levels)`. On each tick, after computing churn,
   overwrite with the current tick's ladder.

3. **Wire into the observation builder** after the existing P1
   features.

4. **Bump obs schema version.**

5. **Expose in `info["debug_features"]`.**

6. **Add `features.book_churn_top_n: 3` to `config.yaml`** under
   the existing `features:` section.

## Tests to add

Create `tests/research_driven/test_p1e_book_churn.py`:

1. **Pure function: identical ladders.** Churn = 0.0.
2. **Pure function: one level's size increased.** Churn > 0,
   equals `|delta| / total_volume`.
3. **Pure function: one level disappeared.** Churn > 0, equals
   the full size of the vanished level / total.
4. **Pure function: one level appeared.** Same magnitude as (3)
   for same size.
5. **Pure function: empty book.** Returns 0.0.
6. **Pure function: respects `n`.** Changes beyond top-N are
   ignored.
7. **Env smoke.** First tick = 0.0; at least one mid-race tick
   is non-zero.
8. **Env determinism.** Same race twice → identical values.
9. **Schema-bump loader refuses pre-P1e checkpoint.**

All CPU, all fast.

## Manual tests

- **Open a race in the replay UI.** Find a sequence of ticks
  where the visible book visibly rearranges (prices appearing/
  disappearing, sizes changing). Confirm `book_churn` is higher
  on those ticks than on quiet ticks where the book barely moved.

## Session exit criteria

- All 9 tests pass.
- All existing tests pass.
- `progress.md` Session 31b entry.
- `lessons_learnt.md` entry if the feature distribution is
  surprising (e.g. if churn is near-zero even on visibly active
  markets, the top-N parameter may be too small).
- `ui_additions.md` row for churn in per-runner panel.
- `master_todo.md` updated.
- Commit.

## Do not

- Do not add state to `env/features.py`.
- Do not change the action space, reward function, or matcher.
- Do not add a windowed average in this session. Start with the
  raw per-tick value. If windowing is needed, it's a follow-up.
- Do not gate on tick cadence. The feature is meaningful at any
  cadence — it measures "how much changed between the ticks I
  can see", not "how much changed per second".
