# Scalping Asymmetric Hedging — All Sessions (01–05)

Work through all five sessions sequentially. Complete each session
fully (code + tests + progress.md entry) before moving to the next.
Commit after each session.

## Before you start — read these

- `plans/scalping-asymmetric-hedging/purpose.md` — why this work
  exists, the four changes, and the Joyeuse worked example.
- `plans/scalping-asymmetric-hedging/hard_constraints.md` — 18
  non-negotiables. In particular: don't touch `ExchangeMatcher`,
  don't break the `raw + shaped ≈ total_reward` invariant, don't
  let locked_pnl credit lucky outcomes.
- `plans/scalping-asymmetric-hedging/lessons_learnt.md` — the seed
  observation on the Gen 0 screenshots.
- `CLAUDE.md` — especially:
  - "Bet accounting: matched orders, not netted positions"
  - "Order matching: single-price, no walking"
  - "Reward function: raw vs shaped" and the invariant
  - "`info['realised_pnl']` is last-race-only" — use
    `env.all_settled_bets` for the full day's bet history, not
    `env.bet_manager.bets`.

## Before you touch anything — locate the code

The landed scalping reward path (commit `98f834b`, 2026-04-15)
introduced `scalping_locked_pnl` and the `raw = locked + min(0,
naked)` rule. Grep for these symbols to find the current accounting
site:

```
grep -rn "scalping_locked_pnl\|locked_pnl\|naked_pnl" env/ agents/
```

Then identify:
1. Where bets are grouped into pairs vs naked (Session 01 will
   formalise this if it's inline).
2. Where the per-race accumulators are written into
   `info["raw_pnl_reward"]` / `info["shaped_bonus"]`.
3. The action-space definition for the PPO policy head (for
   Session 04).

Write what you find into the top of your working scratchpad before
editing — saves re-grepping later in the session.

---

## Session 01 — Redefine `scalping_locked_pnl`

### Context

Current locked_pnl counts any realised P&L from a bet that belongs
to a back/lay pair. Equal-stake pairs get credit when the outcome
is favourable — the +£29.12 Gold Dancer case from
`lessons_learnt.md`. That's luck, not scalping.

### What to do

1. **Extract pair-grouping into a helper.** Create (or relocate)
   a function like `env/scalping.py::group_pairs(bets: list[Bet])
   -> tuple[list[Pair], list[Bet]]` that returns `(pairs, naked)`.
   Pair definition per `hard_constraints.md` §9:
   - Group by `(race_id, selection_id)`.
   - FIFO: earliest opening leg pairs with the next opposite-side
     leg.
   - Excess stake on either side becomes a new naked bet.
   - Export `Pair` as a dataclass holding the two `Bet`s plus the
     effective paired stake/price on each side (if the closing leg
     partially hedges, only the matched portion is "paired").

2. **Add a per-pair outcome calculator.** Given a `Pair` and the
   race outcome data (winner_selection_id, winning_selection_ids,
   each_way_divisor), compute:
   - `win_pnl` = pair P&L if runner wins the race.
   - `lose_pnl` = pair P&L if runner doesn't place at all.
   - For EW races also compute `place_pnl` (runner places but
     doesn't win). Take the min across all three.
   - Delegate the per-leg P&L math to whatever BetManager /
     settlement path already does it — don't reimplement the odds
     arithmetic.

3. **Redefine locked_pnl:**
   ```python
   pair_locked = max(0.0, min(win_pnl, lose_pnl[, place_pnl]))
   scalping_locked_pnl = sum(pair_locked for pair in pairs)
   ```

4. **Naked path unchanged.** Unpaired matched orders continue to
   feed the existing naked-loss / naked-windfall accumulator.

5. **Verify the invariant.** Run an episode-level check that
   `raw + shaped ≈ total` within float tolerance.

### Tests

Add to `tests/` (locate the existing scalping tests via grep —
likely `tests/test_scalping*.py` or inside `test_betfair_env.py`):

1. **Equal-stake pair, runner wins.** Back £20 @ 10.0, lay £20 @
   5.0. Runner wins. Pair P&L: win=+£180−£80=+£100, lose=£0.
   `pair_locked = max(0, min(100, 0)) = 0`. Raw contribution from
   pair = 0. Naked windfall path excluded. Total raw from this
   race = 0.

2. **Equal-stake pair, runner loses.** Same stakes. Runner
   unplaced. win=+£100, lose=£0. `pair_locked = 0`. Raw = 0.

3. **Properly-sized pair, runner wins.** Back £20 @ 12.5, lay
   £41.67 @ 6.0. Runner wins. win = +£230 − £208.35 = +£21.65,
   lose = −£20 + £41.67 = +£21.67. `pair_locked ≈ 21.65`. Raw gets
   ≈ +£21.65.

4. **Properly-sized pair, runner loses.** Same stakes, runner
   unplaced. Raw gets ≈ +£21.67.

5. **Directional pair (lay > lock amount).** Back £20 @ 12.5, lay
   £60 @ 6.0. win = +£230 − £300 = −£70, lose = +£40. `min = −70`,
   `pair_locked = max(0, −70) = 0`. Raw from pair = 0 (and the
   −£70 loss on the win outcome feeds the naked-loss
   asymmetric-penalty path? Confirm: if it doesn't because it's
   not unpaired, document why — directional pairs shouldn't skate
   free).

6. **Unpaired back loses.** Feeds naked-loss path, raw gets
   negative contribution. Unchanged from pre-session behaviour.

7. **Unpaired back wins.** Feeds naked-windfall path, raw gets
   zero contribution. Unchanged.

8. **EW race with placed-but-not-winning outcome.** Pair the same
   runner back+lay. Compute the three per-outcome pnls and confirm
   locked uses the minimum.

9. **Raw + shaped ≈ total** over a mixed episode.

10. **Regression:** pre-existing scalping tests either pass
    unchanged or have expected values updated with a comment
    explaining the reward-scale break.

### Exit criteria

- All tests pass.
- `progress.md` updated with "Session 01 — Redefine
  scalping_locked_pnl" entry noting that reward scale has changed.
- `lessons_learnt.md` appended if anything surprised you.
- Commit with message calling out the reward-scale break.

---

## Session 02 — Worst-case-improvement shaping term

### Context

Session 01 gives an honest raw signal, but it only fires at race
settlement. Per-step shaping helps the agent learn *while the
decision is fresh*. This session adds a shaped reward for closing
legs that narrow the worst-case floor on an open position.

### What to do

1. **Compute `Δ worst_case` on each bet placement that pairs with
   an existing open leg.** Per-step — do this in `BetfairEnv.step`
   right after `ExchangeMatcher` confirms the match, NOT at
   settlement. That way the reward arrives on the same step as the
   decision.
   - `worst_case_before` = min pair-outcome P&L on this
     `(race_id, selection_id)` across open legs, *before* this bet.
     If no open position: 0 (opening legs produce zero shaping).
   - `worst_case_after` = min pair-outcome P&L including the new
     leg's matched stake/price.
   - `shaped_term = coefficient × (worst_case_after −
     worst_case_before)`.

2. **Add a config knob.** In `config.yaml` under the reward
   shaping section, add `worst_case_improvement_bonus: 0.0` (off
   by default). Wire it into the shaping accumulator. Land the
   plumbing off; flip it on in Session 05's training run.

3. **Log per-episode.** Add `shaped_worst_case_improvement` to the
   per-episode record written to `logs/training/episodes.jsonl`
   (follow whatever existing diagnostic keys do).

4. **Verify invariant.** `raw + shaped ≈ total` still holds. The
   new term contributes to `shaped`.

### Tests

1. **Narrowing trade.** Back £20 @ 12.5 (worst_case = −£20). Lay
   £41.67 @ 6.0. worst_case_after ≈ +£21.67. Δ ≈ +£41.67. Shaped
   term = coefficient × 41.67.

2. **Widening trade.** Back £20 @ 12.5, then back another £20 @
   12.5 on the same runner. worst_case was −£20, now −£40. Δ =
   −£20. Shaped term is negative.

3. **Opening leg.** First bet on a runner. No existing position.
   Shaped term = 0.

4. **Random-policy zero-mean check.** Run a fixed-seed episode
   with a fully random policy across several races. Assert
   `abs(mean(shaped_worst_case_improvement)) < 0.1 ×
   std(shaped_worst_case_improvement)` or similar coarse bound.
   This is slow — mark with whatever "slow" marker the suite
   uses, or run with low iteration count and a loose bound.

5. **Coefficient = 0 means zero contribution.** With the default
   config, shaped_worst_case_improvement is always 0. Regression
   safety.

6. **Invariant** holds with the term both off and on.

### Exit criteria

- All tests pass.
- `progress.md` updated. Flag that coefficient lands at 0 and will
  be enabled in Session 05.
- Commit.

---

## Session 03 — UI classification badge in Bet Explorer

### Context

The Bet Explorer currently shows pair P&L with green/red tint
based on realised P&L. That reads luck as skill. We add a
**locked / neutral / directional / naked** badge derived from the
pair worst-case floor (reusing Session 01's classifier — do NOT
duplicate).

### What to do

1. **Locate the evaluator data path.** Grep for "Bet Explorer",
   "bet_explorer", or the route that serves the user's screenshot
   views. Likely under `frontend/` + a backend endpoint under
   `api/` or similar. Follow the data from frontend → API → Python
   evaluator.

2. **Extend the evaluator output** to include a `pair_classification`
   field per matched order, using Session 01's `group_pairs` and
   per-pair `min(win_pnl, lose_pnl, place_pnl)`:
   - `min > 0` → `"locked"` (green)
   - `min == 0` → `"neutral"` (grey)
   - `min < 0` and paired → `"directional"` (amber)
   - unpaired → `"naked"` (red)

3. **Frontend:** render the classification as a chip/badge next to
   the existing outcome badge. Follow the existing badge component
   pattern for WON/LOST chips.

4. **Header summary counts.** Add four counters to the Bet
   Explorer header: number of locked / neutral / directional /
   naked orders. Right next to the existing TOTAL BETS / WIN BETS
   / EW BETS counters.

### Tests

1. **Unit test classifier** (Python side): each of the four
   categories is exhaustive and disjoint. Back+lay pairs with
   crafted stakes/prices land in the expected bucket.

2. **Backend snapshot test.** Canned race with one of each pair
   type — assert API response contains expected classifications.

3. **Frontend manual verify** per CLAUDE.md's "Verify frontend in
   browser before done". Use `preview_start` on the frontend, hit
   the Bet Explorer, eyeball a Gen 0 model's races — the Gold
   Dancer and Joyeuse pairs from `lessons_learnt.md` should show
   as **neutral** or **directional**, not locked.

4. Use `preview_screenshot` to capture the new badges for the
   progress.md entry.

### Exit criteria

- All tests pass.
- Frontend verified in browser.
- Screenshot saved alongside progress entry.
- `progress.md` updated.
- Commit.

---

## Session 04 — "Close position" action

### Context

Even with honest reward signal, the agent can't pick £41.67 as a
stake. This session adds a dedicated close-position action so the
env handles hedge sizing.

### What to do

1. **Locate the action space.** Grep for the PPO action head
   definition (`agents/` or `env/`). Identify:
   - How action indices map to (runner, side, stake) tuples.
   - How the policy network's final layer is shaped.
   - Checkpoint load path — the new head must init fresh without
     breaking old checkpoints.

2. **Add a "close" action per runner.** One new action index per
   runner slot. Index mapping documented in a comment block.

3. **Env dispatch.** In `BetfairEnv.step`, when the close action
   fires for runner R:
   - Look up open position on R (back outstanding vs lay
     outstanding — whichever side has more matched stake is the
     "open" direction).
   - If no open position → no-op, emit a diagnostic tag
     `close_noop`, DO NOT place any bet.
   - Compute hedge stake: `open_stake × open_price /
     target_hedge_price` where `target_hedge_price` is the
     best post-junk-filter price on the opposite side (per
     `ExchangeMatcher`'s existing rules — do not bypass them).
   - Clamp to the opposite side's available size at that level.
     Residual is NOT walked to the next level (hard constraint).
   - Place the hedge bet through `ExchangeMatcher` as normal.

4. **Policy head.** Add the new output units. Initialise their
   weights fresh. Load path: if an older checkpoint is missing the
   close-head weights, initialise those from scratch while
   preserving the original heads' weights.

5. **Config knob.** `agents.close_position_action_enabled: false`
   default. When off, the action is masked out in the policy head
   (log-prob = −∞ or equivalent). Lets Session 05 flip it on
   without rebuilding the network.

### Tests

1. **No-op:** close action with no open position → zero bets
   placed, zero P&L delta, `close_noop` diagnostic set.

2. **Back → lay close:** back £20 @ 12.5, then close. Lay placed
   at £41.67 @ best available lay price (assume ladder has enough
   size). Resulting pair has locked_pnl > 0.

3. **Lay → back close:** lay £20 @ 6.0, then close. Back placed
   at the correctly sized stake.

4. **Clamp:** close at a level with insufficient size. Partial
   match only. No walking. Residual unmatched. Pair is now
   partially hedged; locked_pnl reflects only the matched portion.

5. **Checkpoint load:** save a dummy pre-Session-04 checkpoint
   (or use an existing Gen 0 one), load it with the new action
   head, confirm original weights intact and new head initialised
   fresh.

6. **Config off:** with `close_position_action_enabled: false`,
   the policy never emits close actions. Full episode with random
   policy confirms zero close actions placed.

7. **Integration:** full episode with `close_position_action_
   enabled: true` and Session 02's coefficient > 0. Assert
   `sum(scalping_locked_pnl) > 0` — the signal chain actually
   produces locked profit.

### Exit criteria

- All tests pass.
- Pre-Session-04 checkpoints load cleanly.
- `progress.md` updated.
- Commit.

---

## Session 05 — Training run + analysis

### What to do

Analysis + one training run. No new production code changes (only
config flips and evaluation scripts).

1. **Config:**
   - Enable `worst_case_improvement_bonus` — start at 0.1 or
     whatever the existing shaping terms use as a coefficient
     scale. Justify the choice in the progress entry.
   - Enable `close_position_action_enabled: true`.

2. **Train a fresh model** from scratch or from a Gen 0 baseline,
   using the existing training pipeline. Typical episode budget
   for these plans (match whatever arb-improvements Session 10
   used).

3. **Write `scripts/scalping_hedging_comparison.py`** comparing
   the new model against Gen 0 baselines (`94bca869`, `a7e9ef4f`):
   - Ratio of locked_pnl to total_pnl per episode — should trend
     up.
   - Naked-loss count per episode — should trend down.
   - Stake-ratio distribution on paired bets (new_leg_stake /
     opening_leg_stake) — should shift away from 1.0.
   - Share of Bet Explorer badges that are **locked** vs
     directional/neutral/naked.

4. **Output:** CSV with per-episode metrics, a summary markdown
   with plots or at least tables.

5. **Write findings into `lessons_learnt.md`.** Did the agent
   actually learn to size asymmetrically? Any surprises?

### Exit criteria

- Training completed.
- Comparison script committed.
- Summary written.
- `progress.md` updated.
- Commit.

---

## Cross-session rules

- Run `pytest tests/ -x` after each session. All tests must pass.
- Do **not** touch `env/exchange_matcher.py`. See CLAUDE.md "Order
  matching: single-price, no walking" for why — three independent
  regressions last time.
- Do **not** touch observation features or training-loop plumbing
  beyond the `episodes.jsonl` additions.
- Do **not** "improve" unrelated code you happen to read. Scope is
  tight — four specific changes across five sessions.
- Commit after each session. Use clear messages that reference the
  session number and call out reward-scale breaks explicitly
  (Sessions 01 and 02).
- Knock-on work for `ai-betfair` — drop a note in
  `ai-betfair/incoming/` per the cross-repo postbox convention.
  The close-position action will need mirrored handling in live
  inference.
