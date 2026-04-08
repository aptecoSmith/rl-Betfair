# Session 21 — P1c: windowed features (traded delta + mid drift)

## Before you start — read these

- `../purpose.md`
- `../hard_constraints.md` — constraints 12, 13.
- `../analysis.md` §3
- `../proposals.md` P1
- `../progress.md` — confirm sessions 19 and 20 have landed.
- `../initial_testing.md`
- `env/features.py` — now contains OBI and microprice.
- `env/betfair_env.py` — observation builder.
- `data/episode_builder.py` — how ticks are read and what traded-
  volume columns exist.

## Goal

Add the two **windowed** per-runner features in one session
because they share windowing infrastructure:

- `traded_delta_T` — signed net traded volume at-or-better than
  current microprice over the last `T` seconds. Positive when
  backers are hitting lays; negative when layers are hitting backs.
- `mid_drift_T` — change in `weighted_microprice` over the last
  `T` seconds, expressed in ticks (Betfair price ticks, not env
  time ticks).

These are the first features in this folder that need **state
across ticks**. The state lives on the env, not on `features.py`
— the pure functions take a history buffer and return the value.

## Inputs — constraints to obey

1. **`features.py` stays stateless.** State lives on
   `BetfairEnv`. The feature functions take a history buffer
   (ring deque or list) as an argument and return a float. This
   keeps `features.py` vendorable into `ai-betfair`, which will
   own its own buffers from the live stream.
2. **`T` is wall-clock seconds, not tick index.** Replay ticks
   have timestamps; use them. The live stream will too. A tick-
   based window would diverge between sim and live because tick
   cadence differs.
3. **First-tick value is zero.** Before the window has any
   history, both features return `0.0` — not NaN, not a
   half-window value.
4. **Schema bump once.** Both features land in one bump; no need
   for a bump per feature when they ship in the same session.

## Steps

1. **Add two pure functions to `env/features.py`:**
    - `compute_traded_delta(history, reference_microprice,
      window_seconds, now_ts)` — iterates the history buffer,
      sums signed traded volume within window.
    - `compute_mid_drift(history, window_seconds, now_ts,
      tick_size_fn)` — finds the microprice at `now - window`
      (or nearest older), subtracts, converts difference to
      Betfair ticks using the `tick_size_fn` callback.

   The tick-size callback is passed in (not hardcoded) so the
   function stays dependency-free. `BetfairEnv` supplies the
   callback.

2. **Add per-runner history buffers to `BetfairEnv`.** A dict
   keyed by selection_id, value is a deque of
   `(timestamp, weighted_microprice, traded_volume_delta_since_
   previous_tick)` tuples. Max length bounded by the longest
   window in use, plus a margin.

3. **Update the observation builder.** On every tick, before
   building the observation, append to each runner's history.
   Then call the two feature functions.

4. **Add config keys.**
   `features.traded_delta_window_s: 5`
   `features.mid_drift_window_s: 5`
   Default 5 seconds; will be swept in session 22.

5. **Bump the obs schema version.** Loader refuses pre-P1c
   checkpoints loudly.

6. **Expose both features in `info["debug_features"]`**
   alongside the existing OBI and microprice entries.

## Tests to add

Create `tests/research_driven/test_p1c_windowed.py`:

1. **First-tick values are zero.** Fresh history buffer →
   both functions return exactly `0.0`.
2. **Traded delta sign — backers hitting lays.** Synthetic
   history with traded volume above microprice over the window
   → positive value.
3. **Traded delta sign — layers hitting backs.** Same but below
   → negative value.
4. **Traded delta window edge.** An event just inside the
   window contributes; an event just outside does not. Pin
   this exactly using `now_ts - window_seconds ± epsilon`.
5. **Mid drift — rising microprice.** Synthetic history with
   microprice rising over the window → positive tick delta
   using a mock `tick_size_fn`.
6. **Mid drift — falling microprice.** Same but falling →
   negative.
7. **Mid drift window edge.** Event just outside window is
   ignored; event just inside is used.
8. **Env smoke.** A 1-race fixture run; assert both features
   appear in `info["debug_features"]`. On the first tick of the
   race both must be `0.0`; on at least one mid-race tick at
   least one must be non-zero.
9. **Determinism.** Same race replayed twice → byte-identical
   feature values on every tick.
10. **History buffer bounded.** After a long fixture, the
    per-runner deque length is ≤ the bound, not unbounded.

All CPU, all fast. Test 8 is the slowest; keep it to one race.

## Manual tests

- **Open one race with a known fast-market move.** Confirm both
  windowed features react sensibly — `traded_delta` spikes when
  traded volume visibly surges, `mid_drift` follows the
  microprice trajectory. If either looks dead during a visible
  move, the windowing is wrong. Fix and retest.

## Session exit criteria

- All 10 new tests pass.
- Sessions 19 and 20 tests still pass.
- `env/features.py` now contains exactly four pure functions:
  OBI, microprice, traded_delta, mid_drift.
- The env has exactly one history buffer structure, shared by
  both windowed features.
- `progress.md` Session 21 entry including the config defaults
  you picked for window lengths and any surprises.
- `ui_additions.md` rows for both windowed features.
- `master_todo.md` P1 sub-bullet for P1c ticked.
- Commit.

## Do not

- Do not add state to `env/features.py`. The pure functions
  take a buffer argument; state lives on the env.
- Do not use tick indices for the window. Wall-clock seconds
  only. The live stream does not use tick indices.
- Do not put the traded_delta threshold at the raw LTP. Use
  the `weighted_microprice` from session 20 — that's the whole
  point of having it as a prior feature.
- Do not start the retrain comparison in this session. That's
  session 22.
- Do not try to merge P1a/P1b schema bumps into this one
  retroactively. Each session's bump is historical record.
