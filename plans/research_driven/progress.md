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

## 2026-04-10 — Session 25 — P4a: queue-snapshot bookkeeping (state only)

**Structure choice:** **(B) — `PassiveOrderBook` as a separate class owned by `BetManager` as `self.passive_book`.** This keeps aggressive code on `BetManager` directly and passive code on `passive_book`, making the separation visible at a glance to reviewers. Fill logic in session 26 attaches cleanly to `PassiveOrderBook` without touching `BetManager`'s aggressive paths.

**Shipped:**
- `env/bet_manager.py` — `PassiveOrder` dataclass (11 fields incl. `queue_ahead_at_placement`, `traded_volume_since_placement`, `matched_stake` and `cancelled` reserved for sessions 26/29); `PassiveOrderBook` class (`place()`, `on_tick()`, `orders` property); `BetManager` gains `passive_book: PassiveOrderBook` field (initialised in `__post_init__`).
- `env/betfair_env.py` — `passive_book.on_tick(tick)` called at step 0b of `step()`, before action processing; `info["passive_orders"]` exposes serialised passive orders for the current race.

**Tests added:**
- `tests/research_driven/test_p4a_queue_snapshot.py` — 14 assertions across 8 test classes:
  - `TestPassiveBackPlacement` (2): `queue_ahead_at_placement` = top back size; order appears in book.
  - `TestPassiveLayPlacement` (1): `queue_ahead_at_placement` = top lay size.
  - `TestJunkFilteredPlacementRefused` (3): back junk refused; lay junk refused; no-LTP refused.
  - `TestTradedVolumeAccumulates` (2): delta sums across K ticks; first tick seeds baseline (delta=0).
  - `TestVolumeAtOtherRunnersIgnored` (1): runner B volume does not affect runner A's order.
  - `TestRaceResetEmptiesBook` (2): new BetManager has empty orders; empty `_last_total_matched`.
  - `TestNoAggressiveRegression` (2): aggressive bets still go through `bets`; passive has no self-depletion effect on matcher.
  - `TestBudgetUnaffectedByPassivePlacement` (2): `available_budget` and `budget` both unchanged.

**Did not ship:**
- Fill logic — session 26.
- Budget reservation — session 26.
- Cancel action — session 29.
- Action-space change — session 28.

**Notes for next session (26 — P4b passive-fill triggering + budget reservation):**
- `PassiveOrderBook.on_tick` currently only accumulates volume. Session 26 adds the fill condition: when `traded_volume_since_placement >= queue_ahead_at_placement`, the order is filled. It also adds budget reservation at placement (deducted from `BetManager.budget`) and converts (not double-subtracts) on fill.
- `PassiveOrder.matched_stake` and `cancelled` fields are already present with zero/False defaults — session 26 populates them.
- Session 26 must extend the self-depletion logic (session 18) to cover passive fills: passive fill at price P depletes available volume for aggressive bets at the same price in the same race tick.

**Cross-repo follow-ups:**
- None — no obs schema, action space, or matcher API changes.

---

## 2026-04-08 — Session 24 — P5: UI fill-side annotation

**Shipped:**
- `frontend/src/app/race-replay/race-replay.ts` — `fillSideAnnotation(action)` module-level
  export + delegate method on the component class.
- `frontend/src/app/race-replay/race-replay.html` — `<span class="fill-side-badge">` added
  to `.bet-card-header`, after the action badge, pushed right with `margin-left: auto`.
- `frontend/src/app/race-replay/race-replay.scss` — `.fill-side-badge` rule (monospace 0.65rem,
  muted #888, flex-shrink: 0).
- `frontend/src/app/bet-explorer/bet-explorer.ts` — same `fillSideAnnotation` function exported;
  delegate method on `BetExplorer` class.
- `frontend/src/app/bet-explorer/bet-explorer.html` — "Fill" column header + `<td>` cell with
  `fillSideAnnotation` call; empty-row colspan bumped 12 → 13.
- `frontend/src/app/bet-explorer/bet-explorer.scss` — `.fill-side-badge` rule (monospace 0.75rem,
  muted #888).
- `plans/research_driven/master_todo.md` — Session 24 box ticked.
- `plans/research_driven/ui_additions.md` — replay UI fill-side row ticked; live dashboard row
  remains open.

**Tests added:**
- `frontend/src/app/race-replay/race-replay.spec.ts` — `describe('fillSideAnnotation')`: two
  pure-function tests (`back` → `L→B`, `lay` → `B→L`). Both pass.
- `frontend/src/app/bet-explorer/bet-explorer.spec.ts` — same two tests. Both pass.

**Did not ship:**
- Live dashboard (`ai-betfair`) annotation — out of scope per session instructions; row in
  `ui_additions.md` remains open.
- Screenshots: no snapshot framework exists; manual check is the acceptance gate.

**Notes for next session:**
- The four `fillSideAnnotation` tests are pure-function and fast; the TestBed-based tests in
  both files have a pre-existing `initTestEnvironment` failure (not introduced by this session).

**Cross-repo follow-ups:**
- `ai-betfair` owes a matching fill-side annotation on the live dashboard bet rows (separate
  session, separate reviewer context). Tracked in `ui_additions.md` §Live dashboard P5 row.

---

## 2026-04-08 — Session 23 — P2: spread-cost shaped reward

**Shipped:**
- `plans/research_driven/sessions/session_23_p2_spread_cost.md` — design pass
  committed separately (commit `08f2fe9`) before implementation.
- `env/bet_manager.py` — `Bet` dataclass gains `ltp_at_placement: float = 0.0` field;
  `BetManager.place_back` and `place_lay` populate it from `runner.last_traded_price`.
- `env/betfair_env.py` — `_REWARD_OVERRIDE_KEYS` gains `"spread_cost_weight"`;
  `__init__` reads `self._spread_cost_weight`; `reset()` initialises `_cum_spread_cost`;
  `_settle_current_race` computes `spread_cost_term = −weight × Σ |fill−ltp|/ltp × stake`
  and adds it to `shaped`; `_get_info` exposes `info["spread_cost"]`.
- `config.yaml` — `reward.spread_cost_weight: 0.0` (off by default); gene
  `reward_spread_cost_weight: {type: float, min: 0.0, max: 1.0}` added to
  `hyperparameters.search_ranges`.
- `agents/ppo_trainer.py` — `_REWARD_GENE_MAP` entry
  `"reward_spread_cost_weight": ("spread_cost_weight",)`.

**Chosen formulation:**
`spread_cost_per_bet = matched_stake × |average_price − ltp_at_placement| / ltp_at_placement`
using LTP (not true book mid) as the fair-value reference.  See design pass §1 for
why `|·|` is used and why LTP is preferred over recomputing the book mid.
`efficiency_penalty` is unchanged — the two terms are complementary (bet count vs spread
width), not redundant.

**Tests added:**
- `tests/research_driven/test_p2_spread_cost.py` — 20 tests across 8 logical groups:
  - `TestPureComputation` (4): `ltp_at_placement` stamped correctly on back/lay;
    formula `|fill−ltp|/ltp` verified for back and lay bets.
  - `TestNoBetPolicy` (2): zero spread_cost and zero shaped_bonus for no-bet run.
  - `TestTightSpreadCost` (2): strictly negative cost when bets placed; small magnitude.
  - `TestWideSpreadCost` (1): wide spread produces larger negative cost than tight.
  - `TestRandomPolicyAsymmetry` (2): pins the intentional non-zero-mean; includes
    explicit comment that this test MUST NOT be "fixed" to allow zero spread_cost.
  - `TestRewardInvariant` (3): raw + shaped ≈ total for no-bets, with-bets, and across
    weight values 0/0.25/1.0.
  - `TestBucketing` (3): spread_cost in shaped not raw; info key always present; delta
    shaped equals info["spread_cost"].
  - `TestGenePlumbing` (5): gene in search_ranges; default 0; reward_override respected;
    ppo_trainer map entry; weight=0 gives zero cost.

**Did not ship:**
- Actual replay UI line for `spread_cost` breakdown — deferred per the session plan.
  Row recorded in `ui_additions.md`.
- `downstream_knockon.md` §2 update — no code change needed (alerting note only).

**Notes for next session:**
- Default `spread_cost_weight: 0.0` means all pre-session runs are byte-identical.
  Enable by setting `reward_spread_cost_weight` in hyperparameter search or by setting
  `reward.spread_cost_weight: X` in config.yaml.
- The intentional-asymmetry exception is recorded in `lessons_learnt.md` (Session 23
  entry) and pinned by a test (`TestRandomPolicyAsymmetry`). Do not zero-mean it.

**Cross-repo follow-ups:**
- None — no obs schema or action-space changes. `ai-betfair` is unaffected.

---

## 2026-04-08 — Session 21 — P1c: windowed features (traded_delta + mid_drift)

**Shipped:**
- `env/features.py` — `betfair_tick_size(price)`, `compute_traded_delta(history, reference_microprice, window_seconds, now_ts)`, `compute_mid_drift(history, window_seconds, now_ts, tick_size_fn)` added. All pure, no numpy, no env imports. `betfair_tick_size` implements the standard Betfair horse-racing price ladder and is passed as the `tick_size_fn` callback. `compute_traded_delta` signs volume relative to the current microprice (≤ ref → positive/backing, > ref → negative/laying). `compute_mid_drift` uses the latest history entry at-or-before (now − window) as baseline; returns 0.0 when no such entry exists.
- `data/feature_engineer.py` — imports `deque`, new functions; `TickHistory` dataclass gains `traded_delta_window_s: float = 60.0`, `mid_drift_window_s: float = 60.0`, `_windowed_history`, `_prev_total_matched`, `_windowed_maxlen` fields, `__post_init__`, `update_windowed`, `windowed_history_for`, and updated `reset()`. `engineer_tick` updates windowed history before computing `traded_delta` and `mid_drift` per runner; `NaN` microprice runners fall back to LTP for windowed history. `engineer_race` and `engineer_day` accept `traded_delta_window_s` and `mid_drift_window_s` params.
- `env/betfair_env.py` — `OBS_SCHEMA_VERSION = 4`; `"traded_delta"`, `"mid_drift"` appended to `RUNNER_KEYS`; `RUNNER_DIM` 112 → 114; `self._traded_delta_window_s`, `self._mid_drift_window_s` from config; `engineer_day` call passes window params; runtime windowed history buffers (`_windowed_history`, `_prev_total_matched_rt`, `_windowed_maxlen`) added and reset between races; `_update_runtime_windowed(tick)` helper added and called at start of `step()`; `debug_features` per-runner dict now includes `traded_delta` and `mid_drift`.
- `config.yaml` — `features.traded_delta_window_s: 60`, `features.mid_drift_window_s: 60` added. Default 60 s; will be swept in session 22.
- `tests/research_driven/test_p1a_obi.py` — updated `test_obi_in_obs_vector` to expect `RUNNER_DIM=114`.
- `tests/research_driven/test_p1b_microprice.py` — updated `test_refuses_p1a_checkpoint` to expect `OBS_SCHEMA_VERSION=4`.

**Tests added:**
- `tests/research_driven/test_p1c_windowed.py` — 29 tests across 10 logical groups:
  - `betfair_tick_size`: 6 boundary checks.
  - Test 1 (first-tick zero): 4 checks (empty history + single zero-delta entry, for both functions).
  - Test 2 (positive sign): 3 checks (at-reference, below-reference, multiple entries all positive).
  - Test 3 (negative sign): 2 checks (above-reference, mixed net result).
  - Test 4 (traded_delta window edge): 3 checks (just inside, just outside, exactly at cutoff).
  - Test 5 (mid_drift rising): 2 checks.
  - Test 6 (mid_drift falling): 1 check.
  - Test 7 (mid_drift window edge): 3 checks (just outside = IS baseline, just inside = not baseline, latest at-or-before wins).
  - Test 8 (env smoke): features appear in `debug_features`; first step = 0.0; mid-race at least one non-zero.
  - Test 9 (determinism): two independent replays yield byte-identical feature values.
  - Test 10 (bounded buffer): deque length ≤ `_windowed_maxlen` after a full race.
  - Schema: refuses P1b checkpoint (v3); accepts v4.

**Did not ship:**
- Retrain comparison — deferred to session 22.

**Implementation notes:**
- Sign convention for `traded_delta`: history entries with microprice ≤ reference are counted positively (backing pressure). This treats the reference (current) microprice as the neutral point — volume that traded when the runner was cheaper (more favoured) is net backing.
- `compute_mid_drift` baseline selection: latest entry with `ts ≤ now - window`. If no entry exists before the window boundary, returns 0.0 (not a stale baseline). This means the first ~60 s of a race have `mid_drift=0` until the window fills.
- The architecture keeps windowed features in `RUNNER_KEYS` (precomputed via `TickHistory` in `_precompute`) for the obs vector, AND maintains separate runtime deques on the env for `debug_features`. Both compute identically given the same tick sequence.
- `_windowed_maxlen = max(int(max_window * 2) + 20, 200)` — generous bound ensuring no reachable tick frequency exhausts the deque.

**Notes for next session:**
- Session 22 (P1d retrain + decision gate): train one policy with the 4-feature obs (OBI, microprice, traded_delta, mid_drift). Compare vs P1b baseline on the 9-day eval window. Sweep `traded_delta_window_s` and `mid_drift_window_s` (try 30, 60, 120).

**Cross-repo follow-ups:**
- `downstream_knockon.md` §1: `ai-betfair` now needs `traded_delta` and `mid_drift` in its per-runner obs assembly. It will own its own windowed history buffers from the live stream.

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
- Commit SHA for this session: ee03b8b

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


---

## 2026-04-08 — Session 22 — P1d: re-train and decision-gate comparison

**Shipped:**
- `scripts/session_22_p1d_compare.py` (NEW) — standalone comparison script. Defines `BaselinePPOLSTMPolicy` (RUNNER_DIM=110, schema v1), `BaselinePPOTrainer` (obs-slicing subclass of PPOTrainer), `evaluate_policy`, `check_p1_gradient_norm`, `build_baseline_obs_indices`. Both policies use identical `SHARED_HP`. Results appended to this file. Script is rerunnable.
- `plans/research_driven/open_questions.md` — Q3 resolved: raw daily P&L on held-out eval window (operator choice A, 2026-04-08).
- `plans/research_driven/sessions/session_22_p1d_retrain.md` — `## Q3 resolution` heading added at top.

**Comparison run result (2026-04-08, scripts/session_22_p1d_compare.py):**

Setup: 4 train days (2026-03-31 to 2026-04-03), 3 eval days (2026-04-04 to 2026-04-06), 5 epochs, CUDA, identical hyperparameters.

Per-day raw P&L on held-out eval window:

| Date       | Baseline P&L | P1 P&L | Delta      |
|------------|-------------|--------|------------|
| 2026-04-04 | +12044.47   | +0.00  | -12044.47  |
| 2026-04-05 | +8783.61    | +0.00  | -8783.61   |
| 2026-04-06 | +8048.47    | +0.00  | -8048.47   |
| **TOTAL**  | **+28876.55** | **+0.00** | **-28876.55** |
| MEAN/DAY   | +9625.52    | +0.00  | -9625.52   |

- Baseline total bets on eval: 1771 (high-betting strategy)
- P1 total bets on eval: 0 (collapsed to no-bet strategy during training)
- P1 gradient norm on new columns (after training): ~1e-10 (effectively zero — consistent with collapsed policy, not with wiring error)

**Diagnosis — result is confounded by training variance, not feature quality:**

The P1 policy learned to place 0 bets by epoch 2 and never recovered ("collapsed policy" problem). The baseline policy happened to discover a high-volume backing strategy. In a re-run with different random seeds, either policy could collapse. Two runs of the script show this: in run 1, baseline collapsed and P1 was profitable; in run 2, P1 collapsed and baseline was profitable. Neither outcome is reproducible.

Single-seed single-agent PPO comparison cannot discriminate feature quality in this regime. The gradient norm was ~1e-10 (near-zero) because the collapsed P1 policy's critic outputs near-constant values — confirming training collapse, not wiring error.

**Recommendation at Phase 1 gate:**

The comparison as designed cannot answer "does the P1 obs help?" in 5 epochs × 4 days × 1 seed. The session plan assumed a more stable training regime. Two options:

1. **Use the existing evolutionary infrastructure** (N=50 agents, selection, multi-generation) with half the population on P1 obs and half on baseline obs — the population averages across seeds. This is the right comparison but was out of scope for this session.
2. **Proceed to P2 anyway**: the P1 features (OBI, microprice, traded_delta, mid_drift) are correctly wired (gradient does flow at non-collapsed initialisation — confirmed in run 1), and the features are correct by unit test. The single-seed gate is not informative enough to stop the programme.

**Decision for master_todo.md:** proceed cautiously — tick the gate as "inconclusive, continuing to P2". Record in lessons_learnt.md.

**Tests added:**
- `scripts/session_22_p1d_compare.py` integration test (description in `integration_testing.md`): trains P1 policy on 1-day fixture, asserts gradient norm on new columns is non-zero at non-collapsed initialisation.

**Did not ship:**
- Window parameter sweep (`traded_delta_window_s` 30/60/120) — descoped; the single-seed result made the sweep meaningless.
- Manual spot-check via UI — no evaluator UI wired yet (deferred from session 19).

**Notes for next session (23 — P2 spread-cost shaped reward):**
- The `BaselinePPOLSTMPolicy` and comparison infrastructure in session_22 script are session-22-only. Do not import or extend them in production code.
- The evolutionary framework (PopulationManager + TrainingOrchestrator) is the right comparison tool; use it for the P2 gate rather than this script.
- Gradient check needs a non-collapsed policy as input — run it early in training (after episode 1), not after the full training run.

**Cross-repo follow-ups:**
- None new.

