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

## 2026-04-08 — Session 20 — P1b: weighted microprice feature

**Shipped:**
- `env/features.py` — `compute_microprice(back_levels, lay_levels, n, ltp_fallback)` added. Pure function, no numpy, no env imports. Uses total depth from top-N levels per side as weights applied to the best price on each side, ensuring the result is bounded by `[best_back_price, best_lay_price]`. Raises `ValueError` when both sides are empty and `ltp_fallback` is `None` or non-positive.
- `data/feature_engineer.py` — imports `compute_microprice`; `engineer_tick / engineer_race / engineer_day` accept `microprice_top_n: int = 3`; `weighted_microprice` computed per runner after `obi_topN`; falls back to `NaN` on `ValueError` (unpriceable runner is already handled upstream).
- `env/betfair_env.py` — `OBS_SCHEMA_VERSION = 3`; `"weighted_microprice"` appended to `RUNNER_KEYS`; `RUNNER_DIM` 111 → 112; `self._microprice_top_n` from config; `engineer_day` call passes `microprice_top_n`; `debug_features` per-runner dict now includes both `obi_topN` and `weighted_microprice`; imports `compute_microprice`.
- `config.yaml` — `features.microprice_top_n: 3` added.
- `tests/research_driven/test_p1a_obi.py` — updated `test_obi_in_obs_vector` to expect `RUNNER_DIM=112`.

**Tests added:**
- `tests/research_driven/test_p1b_microprice.py` — 11 tests (5 pure-function + 1 env + 4 schema-bump):
  - Pure: symmetric book equals midpoint; asymmetric sizes pull toward heavy side; empty book → LTP fallback; empty book no LTP → raises; empty book non-positive LTP → raises; bounded by best-back/best-lay (5 random books).
  - Env smoke: `weighted_microprice` appears in `info["debug_features"]` and is positive.
  - Schema: refuses P1a checkpoint (v2); refuses pre-P1a checkpoint (v1); refuses bare state-dict; accepts current version (v3).

**Did not ship:**
- `traded_delta_T`, `mid_drift_T` — deferred to session 21 per plan.

**Implementation note:**
The session spec formula uses individual level prices (`Σ back_size_i × back_price_i`), which can pull the result outside `[best_back, best_lay]` when N>1 (higher lay levels exceed best_lay). The implementation instead uses total top-N sizes as weights applied to the best price on each side, satisfying the bounded constraint while still incorporating N-level depth information. This is the standard "weighted midpoint" interpretation in market microstructure.

**Notes for next session:**
- Session 21 (P1c windowed features) adds cross-tick state to `data/feature_engineer.py`. `OBS_SCHEMA_VERSION` must be bumped again (to 4).
- Commit SHA for this session: TBD

**Cross-repo follow-ups:**
- `downstream_knockon.md` §1: `ai-betfair` now needs `weighted_microprice` in its per-runner observation assembly alongside `obi_topN`.

---

## 2026-04-08 — Session 19 — P1a: OBI feature + obs schema bump

**Shipped:**
- `env/features.py` (NEW) — pure, dependency-free `compute_obi(back_levels, lay_levels, n)` function. No numpy, no env-internal imports. Vendorable verbatim into `ai-betfair`.
- `env/betfair_env.py` — `OBS_SCHEMA_VERSION = 2` constant; `validate_obs_schema(checkpoint)` function; `"obi_topN"` appended to `RUNNER_KEYS`; `RUNNER_DIM` 110 → 111; `self._obi_top_n` from config; `debug_features` key in `_get_info` with per-runner `{"obi_topN": float}`; `engineer_day` call passes `obi_top_n`.
- `data/feature_engineer.py` — imports `compute_obi`; `engineer_tick / engineer_race / engineer_day` accept `obi_top_n: int = 3`; obi computed per runner in `engineer_tick` from raw ladder levels.
- `config.yaml` — `features: obi_top_n: 3` section added.
- `registry/model_store.py` — `save_weights` accepts optional `obs_schema_version`; `load_weights` accepts optional `expected_obs_schema_version` and validates via `validate_obs_schema` when provided; unwraps `{"obs_schema_version": N, "weights": ...}` format.
- `agents/population_manager.py` — imports `OBS_SCHEMA_VERSION`; both `save_weights` calls pass it; `load_agent` passes `expected_obs_schema_version`.

**Tests added:**
- `tests/research_driven/test_p1a_obi.py` — 13 tests (7 pure-function + 3 env + 3 schema):
  - Pure: balanced book → 0.0; all-back → 1.0; all-lay → -1.0; empty → 0.0; respects N; asymmetric; N > level count.
  - Env smoke: `obi_topN` appears in `info["debug_features"]`, in `[-1, 1]`; determinism across two runs; obs vector shape matches RUNNER_DIM=111.
  - Schema: refuses checkpoint with old version; refuses bare state-dict; accepts current version.

**Did not ship:**
- Manual test spot-check (UI not yet wired) — acceptance row already in `ui_additions.md`.
- `weighted_microprice`, `traded_delta_T`, `mid_drift_T` — deferred to sessions 20/21 per plan.

**Notes for next session:**
- Session 20 (P1b microprice) reuses `env/features.py` and the same schema-bump pattern. `OBS_SCHEMA_VERSION` must be bumped again in that session.
- Commit SHA for this session: 906ed0b

**Cross-repo follow-ups:**
- `downstream_knockon.md` §1: `ai-betfair` now needs `obi_topN` in its per-runner observation assembly. It can vendor `env/features.py` directly.

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
