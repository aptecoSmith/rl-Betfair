# Progress — Research-Driven

One entry per completed session, newest at the top. Each entry
records the factual outcome — what shipped, what files changed,
what tests were added, what did *not* ship and why.

This file is the source of truth for "what state did the last
session leave the repo in?". A new session starts by reading the
most recent entry. If the entry doesn't tell you, the previous
session under-documented and the next session is allowed to push
back.

Format:

```
## YYYY-MM-DD — Session NN — Title

**Shipped:**
- bullet of file/area changes

**Tests added:**
- bullet of test file + what it asserts

**Did not ship:**
- bullet of what was scoped but cut, with reason

**Notes for next session:**
- anything load-bearing the next reader needs

**Cross-repo follow-ups:**
- bullet for `ai-betfair` items owed (link to downstream_knockon.md
  section)
```

---

## 2026-04-08 — Session 18 — R-2 self-depletion fix

**Shipped:**
- `env/exchange_matcher.py` — added `pick_top_price` helper (returns
  post-filter best price without doing a fill); added optional
  `already_matched_at_top: float = 0.0` to `_match`, `match_back`,
  `match_lay`; adjusted fill logic to `min(stake, max(0, top.size -
  already_matched_at_top))` with a `"self-depletion exhausted level"`
  skipped_reason when adjusted size reaches zero.
- `env/bet_manager.py` — added `_matched_at_level: dict[tuple[int,
  BetSide, float], float]` accumulator (init=False, resets implicitly
  per-race via env's fresh BetManager); `place_back` and `place_lay`
  call `pick_top_price` to peek the fill price, look up the
  accumulator, pass `already_matched_at_top` to the matcher, then
  increment the accumulator after a successful match.
- `tests/research_driven/test_r2_self_depletion.py` — 9 tests (the
  6 mandated axes plus 3 sub-cases for the first axis).

**Option chosen:** (B) — small `pick_top_price` helper on
`ExchangeMatcher` so filter logic lives in one place. Matcher stays
stateless; accumulator lives exclusively on `BetManager`.

**Tests added:**
- `tests/research_driven/test_r2_self_depletion.py` — 9 tests
  covering: two backs same price same runner (3 sub-cases), two backs
  different prices same runner, two backs same price different runners,
  back+lay same price same runner, cross-race reset, skipped-reason on
  full self-exhaustion.

**Did not ship:**
- Nothing cut. All 6 axes specified in the session prompt were covered.

**Notes for next session:**
- All existing matcher (35) and bet-manager (56) tests pass unchanged —
  default-zero path is byte-identical to pre-fix behaviour.
- Reward-plumbing invariant test (`raw + shaped ≈ total_reward`) passes.
- `ai-betfair` live-side equivalent (§0a in `downstream_knockon.md`)
  still open — transient accumulator that clears on each market-data
  tick. Not in scope for session 18.

**Cross-repo follow-ups:**
- `ai-betfair` §0a: live-side self-depletion in the gap between order
  placement and next market-data tick refresh.

The first entry will be added when the first item from
`master_todo.md` lands. Until then, treat the planning files
(`purpose.md`, `analysis.md`, `proposals.md`, `open_questions.md`,
`downstream_knockon.md`, `hard_constraints.md`,
`design_decisions.md`, `not_doing.md`) as the current state.

The first session that lands here is **not** session 11. Numbering
continues from `next_steps/master_todo.md` — pick the next free
number when promoting an item, do not start over.
