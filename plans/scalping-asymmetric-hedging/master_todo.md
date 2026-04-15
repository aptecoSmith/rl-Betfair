# Master TODO — Scalping Asymmetric Hedging

Ordered session list. Tick boxes as sessions land.

When a session completes:
1. Tick its box here.
2. Add an entry to `progress.md`.
3. Append any learnings to `lessons_learnt.md`.
4. Note cross-repo follow-ups in `ai-betfair/incoming/` per the
   postbox convention.

---

## Phase 1 — Honest reward signal

- [ ] **Session 01 — Redefine `scalping_locked_pnl`**

  Change the locked-pnl accumulator to credit only the guaranteed
  floor of each back/lay pair.

  Touchpoints (verify names against current code):
  - `env/betfair_env.py::_settle_current_race` — where scalping
    accumulators are computed from settled bets.
  - Whatever helper classifies bets into pairs vs naked (may live
    in `env/` or inline).

  Logic:
  - Group matched bets by `(race_id, selection_id)` into pairs
    (FIFO: earliest open leg → next opposite-side leg).
  - For each pair, compute `win_pnl` and `lose_pnl` (the two
    outcomes that are possible given the pair — runner wins vs
    runner loses, ignoring EW place fraction at this stage; see
    note below).
  - `pair_locked = max(0, min(win_pnl, lose_pnl))`.
  - `scalping_locked_pnl = sum(pair_locked for all pairs)`.
  - Naked (unpaired) matched orders feed the existing naked-pnl
    path unchanged.

  **EW note:** For each-way races, "runner wins" and "runner
  loses" each split into sub-outcomes (won, placed, unplaced).
  Minimum over all three is still the worst case. Delegate to the
  existing EW settlement path to produce per-outcome pnl and take
  the min.

  **Tests:**
  - Equal-stake pair, runner wins: locked = 0, raw contribution
    from pair = 0 (naked windfall excluded).
  - Equal-stake pair, runner loses: locked = 0, pair nets £0, raw
    contribution = 0.
  - Properly-sized pair (back £20 @ 12.5, lay £41.67 @ 6.0):
    locked ≈ £21.66 on either outcome.
  - Unpaired back loses: feeds naked-loss path, raw gets negative
    contribution.
  - Unpaired back wins: feeds naked-windfall path, raw gets zero.
  - Raw + shaped ≈ total invariant holds for a mixed episode.
  - Previously-passing scalping tests either pass unchanged or
    have their expected values updated with a note explaining why.

- [ ] **Session 02 — Worst-case-improvement shaping term**

  Add a shaped reward component: `Δ worst_case` per closing leg.

  Definition:
  - When a bet is placed that pairs with an existing open leg
    (i.e. this is a closing leg), compute:
    - `worst_case_before` = worst outcome across open positions
      on this `(race_id, selection_id)` before this bet.
    - `worst_case_after` = worst outcome including this new leg.
    - Shaped term = `coefficient × (worst_case_after −
      worst_case_before)`.
  - Coefficient starts off (0.0) — land the plumbing, verify with
    it off, then enable at a small value and monitor.

  Touchpoints:
  - Shaping accumulator in `_settle_current_race` — or, if we
    want per-step signal, in `BetfairEnv.step` immediately after
    the matcher confirms the fill.
  - New config knob in `config.yaml` under reward shaping.
  - New per-episode diagnostic in
    `logs/training/episodes.jsonl` — e.g.
    `shaped_worst_case_improvement`.

  **Tests:**
  - Closing leg that narrows worst-case produces positive shaped
    contribution.
  - Closing leg that widens worst-case produces negative
    contribution.
  - Opening leg (no existing position) produces zero
    contribution.
  - Random policy: expected mean of shaped_worst_case_improvement
    over many episodes is near zero (use a coarse threshold —
    e.g. |mean| < 0.1 × std).
  - Raw + shaped ≈ total invariant holds.

## Phase 2 — Honest diagnostics

- [ ] **Session 03 — UI classification badge in Bet Explorer**

  Add locked / neutral / directional / naked classification to
  the pair rows in the frontend Bet Explorer.

  Touchpoints:
  - Evaluator that produces Bet Explorer data (likely under
    `frontend/`'s data-serving API, or whatever feeds
    `coverage-dashboard` / Bet Explorer views — locate via grep
    before editing).
  - Classification derived from `min(win_pnl, lose_pnl)` per pair
    (reusing Session 01's pair-grouping helper — do NOT duplicate
    the logic).
  - Frontend badge component + colour map.

  Classification:
  - `min > 0` → **locked** (green)
  - `min == 0` → **neutral** (grey)
  - `min < 0` → **directional** (amber)
  - Unpaired matched order → **naked** (red)

  **Tests:**
  - Unit test on the classifier: each of the four categories is
    exhaustive and disjoint.
  - Backend snapshot test: a canned race with one of each pair
    type produces the expected badges.
  - Frontend render test (if existing tests have a pattern to
    follow — else a manual check via preview server is fine, per
    CLAUDE.md's "Verify frontend in browser before done").

## Phase 3 — Give the agent the tool

- [ ] **Session 04 — "Close position" action**

  Add an action that closes an open position on a runner at
  market, with env-computed hedge sizing.

  Touchpoints (locate via grep before editing):
  - Action-space definition (`agents/` or `env/` — wherever the
    PPO action head is defined).
  - Env `step()` — dispatch the close action through the matcher.
  - Hedge-size computation: `back_stake × back_price / lay_price`
    (or the symmetric lay→back version), clamped to
    post-junk-filter best-level size.
  - Policy network: add the new action head without breaking
    load-compatibility for pre-Session-04 checkpoints.

  Behavioural rules:
  - If no open position → no-op (never opens a new naked
    position).
  - If hedge stake clamps, residual is NOT spilled to the next
    level (ExchangeMatcher rule is load-bearing — see CLAUDE.md).
  - Action is additive: existing open-leg actions unchanged.

  **Tests:**
  - Close action with no open position → no bet placed.
  - Close action after a back leg → lay placed at correct stake.
  - Close action after a lay leg → back placed at correct stake.
  - Clamping: close at a level with insufficient size → partial
    match, no walking, residual unmatched.
  - Load a pre-Session-04 checkpoint; new action head initialises
    fresh, existing behaviour preserved.
  - Integration: a full episode where the agent uses close
    actions produces non-zero locked_pnl.

## Phase 4 — Validation

- [ ] **Session 05 — Training run + analysis**

  - Train a fresh model with Sessions 01–04 enabled (small
    coefficient on the worst-case shaping term).
  - Compare against the current Gen 0 baselines (`94bca869`,
    `a7e9ef4f` shown in the user's screenshots).
  - Look for:
    - Ratio of locked_pnl to total_pnl trending up.
    - Naked-loss count trending down.
    - More asymmetric stake ratios in the bet log (back vs paired
      lay stake).
    - Bet Explorer shows increasing share of **locked** badges.
  - Document findings in `lessons_learnt.md`. This is analysis +
    one training run — not a code session.

---

## Summary

| Session | What | Phase |
|---------|------|-------|
| 01 | Redefine scalping_locked_pnl | 1 |
| 02 | Worst-case-improvement shaping | 1 |
| 03 | UI classification badge | 2 |
| 04 | Close-position action | 3 |
| 05 | Training run + analysis | 4 |

Total: 5 sessions. Sessions 01 and 02 change reward scale —
pre-existing training runs are no longer directly comparable after
either lands. Session 03 is diagnostic-only. Session 04 is the
action-space change and needs Sessions 01+02 already in place so
the signal points the right way.
