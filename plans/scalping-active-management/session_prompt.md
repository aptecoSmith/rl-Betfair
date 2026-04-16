# Scalping Active Management — Session 01 prompt

Work through session 01 fully (code + tests + progress.md
entry) before moving on. Commit after the session.

## Before you start — read these

- `plans/scalping-active-management/purpose.md` — why this
  work exists, the Gen 1 evidence, and the four changes this
  plan covers.
- `plans/scalping-active-management/hard_constraints.md` — 20
  non-negotiables. Especially:
  - §1 — invariant `raw + shaped ≈ total_reward` still holds.
  - §4 — re-quote never walks the ladder (use the same
    `PassiveOrderBook.place` path as the initial placement).
  - §5 — re-quote on a runner without an open passive is a
    no-op. Never opens a new naked position.
  - §6 — cancel returns reserved budget before re-place
    reserves new liability.
  - §7 — action-space addition is additive (new dim at the
    end; pre-existing indices don't move).
- `plans/scalping-active-management/lessons_learnt.md` — seed
  observations from the Gen 1 analysis.
- `CLAUDE.md` — especially:
  - "Order matching: single-price, no walking"
  - "Bet accounting: matched orders, not netted positions"
  - "Reward function: raw vs shaped"

## Before you touch anything — locate the code

```
grep -rn "_process_action\|SCALPING_ACTIONS_PER_RUNNER\|_maybe_place_paired" env/ agents/
```

Identify:

1. Where `_process_action` reads the per-runner action dims.
2. `SCALPING_ACTIONS_PER_RUNNER` constant in `env/betfair_env.py`
   (currently 5; Session 01 makes it 6).
3. How `_maybe_place_paired` reserves budget for the passive
   (so your cancel-and-replace symmetrically releases it).
4. The `PassiveOrderBook.place` + `PassiveOrder` model — the
   cancel path may already exist; if it does, prefer re-using
   it.
5. Observation building path — where per-runner features are
   assembled so you can add
   `seconds_since_passive_placed` and
   `passive_price_vs_current_ltp_ticks`.

Write what you find into your scratchpad before editing.

---

## Session 01 — Re-quote action + env plumbing

### Context

After an aggressive fill with scalping_mode on, the env
auto-places a passive counter-order N ticks away. Once placed,
the passive just sits — either it fills (arb_done) or it's
cancelled at race-off (arb_naked). The agent currently has no
way to say "my initial choice was wrong, move the passive
closer to market." That's what this session adds.

### What to do

1. **Bump `SCALPING_ACTIONS_PER_RUNNER` from 5 to 6.** The new
   dim is `requote_signal ∈ [-1, 1]`. Read it as the 6th
   per-runner action alongside signal/stake/aggr/cancel/arb.

2. **Env dispatch.** In `_process_action`, after the existing
   aggressive / passive / cancel handling, add a re-quote
   step:
   - For each runner, check `requote_signal > 0.5`.
   - If true, look up the open passive for this runner that
     has a `pair_id` (auto-placed by `_maybe_place_paired`).
     If none, skip (no-op).
   - Compute `arb_ticks` from the same `arb_raw` the runner
     produced this tick (use `_arb_spread_scale` gene etc).
   - Compute `new_price = current_ltp ± arb_ticks` — same
     direction rule as `_maybe_place_paired` (lay below an
     aggressive back, back above an aggressive lay).
   - Cancel the existing passive via whatever
     `PassiveOrderBook` exposes. Budget reservation returned.
   - Re-place at `new_price` using the SAME
     `PassiveOrderBook.place` call as initial placement. Same
     junk filter, same clamp, same pair_id.
   - On failure (new price in junk band, insufficient budget
     post-cancel, no opposite-side liquidity at the new
     price), leave the runner with no passive — record a
     diagnostic tag `requote_failed`. NEVER open a new naked
     position in compensation.

3. **New observation features (per-runner, 2 new floats).**
   Add to the observation builder:
   - `seconds_since_passive_placed` — `time_since_place /
     race_duration_seconds`, clamped to `[0, 1]`. `0` when
     there's no open passive.
   - `passive_price_vs_current_ltp_ticks` — signed integer
     converted to float, normalised by `MAX_ARB_TICKS`. How
     far the resting passive is from where LTP is now. `0`
     when no passive.

   `PassiveOrder` already tracks its placement tick index; if
   it doesn't track placement tick seconds, add a field that
   captures `time_to_off` at placement.

4. **Update action-space size** returned by
   `BetfairEnv.action_space` when scalping is on:
   `14 * SCALPING_ACTIONS_PER_RUNNER = 14 * 6 = 84` (was 70).
   Observation-space grows by `2 * max_runners` for the new
   features.

5. **Backward compat.** When scalping_mode is off, the
   existing 4-dim-per-runner path runs unchanged. When on,
   old checkpoints that had 5-dim-per-runner arbs still load
   — the 6th dim head is initialised fresh. Mention in the
   progress entry.

### Tests (add to tests/test_forced_arbitrage.py)

Add a new `TestScalpingRequote` class:

1. **`test_requote_noop_without_open_passive`** — fire a
   `requote_signal > 0.5` on a runner that never had an
   aggressive bet. No bets placed, `requote_failed` tag set.

2. **`test_requote_cancels_and_replaces`** — place an
   aggressive back + auto-paired lay. On a subsequent tick,
   fire `requote_signal > 0.5` with a different `arb_raw`.
   Assert: old passive is gone from `passive_book`, new
   passive is present with the same `pair_id` at the new
   price. Bet history unchanged.

3. **`test_requote_preserves_pair_id`** — same setup; after
   re-quote, the `pair_id` still matches the aggressive bet's
   `pair_id`. When the re-quoted passive fills, the pair
   completes normally.

4. **`test_requote_budget_accounting`** — snapshot
   `bm.available_budget` before the re-quote, after. The
   difference should be only the change in lay liability
   between the old and new passive price, within ± £0.01.

5. **`test_requote_into_junk_band_silent_failure`** — re-quote
   the passive to a price that the junk filter rejects (far
   outside LTP deviation cap). No new passive placed, old
   passive cancelled, `requote_failed` tag set. The aggressive
   leg is now naked. Runner's state correctly reflects "no
   passive open".

6. **`test_requote_ladder_walk_prevented`** — re-quote target
   at a level with insufficient size. Partial match only, no
   walking. Same one-price-only rule as initial placement.

7. **`test_obs_features_present`** — build an env in
   scalping mode, step through a placement + one tick of
   waiting. Assert the two new observation features are
   present at the correct indices, monotonic with elapsed
   time.

8. **`test_obs_features_zero_without_passive`** — observation
   features are exactly 0 when the runner has no open passive.

9. **`test_action_space_size_grows`** — with
   scalping_mode=True, `env.action_space.shape == (14 * 6,)`.
   With scalping_mode=False, still `(14 * 4,)` (unchanged).

10. **`test_legacy_checkpoint_loads`** — load a pre-Session-01
    checkpoint (can be a crafted fake with 5-dim-per-runner
    action head); assert the new 6th dim head initialises
    fresh, original weights preserved.

11. **`test_raw_plus_shaped_invariant_holds`** — full episode
    with re-quote actions firing. Assert
    `abs(total_reward - (info['raw_pnl_reward'] +
    info['shaped_bonus'])) < 1e-6`.

### Exit criteria

- All new tests pass.
- `python -m pytest tests/test_forced_arbitrage.py tests/test_bet_manager.py tests/test_betfair_env.py -q` — all green.
- `progress.md` updated with a Session 01 entry summarising
  what landed, noting the action-space size change and
  back-compat strategy.
- `lessons_learnt.md` appended if anything surprised you
  (e.g. budget accounting edge cases, `PassiveOrderBook`
  cancel semantics).
- Commit with a message referencing this plan and session.

---

## Cross-session rules for the whole plan

- Run `pytest tests/ -x` after each session. All tests must
  pass.
- Do NOT touch `env/exchange_matcher.py`. See CLAUDE.md
  "Order matching: single-price, no walking" — three
  independent regressions last time this was relaxed.
- Do NOT touch the existing `arb_spread` action dim or the
  `arb_spread_scale` gene. This plan adds new dims; it does
  not re-tune existing ones.
- Do NOT "improve" unrelated code you happen to read. Scope
  is tight.
- Commit after each session. Call out any reward-scale
  changes in commit messages.
- Knock-on work for `ai-betfair` — drop a note in
  `ai-betfair/incoming/` per the cross-repo postbox
  convention. The re-quote mechanic especially will need
  mirrored handling in live inference.
