# Phase −1 — env audit findings

**Question.** Does `env/` faithfully implement
`docs/betfair_market_model.md`?

**Verdict — RED (revised 2026-04-26 from initial GREEN; triage
complete same day).** Phase 0 should pause until F7 has been
fixed under one of options (1)–(3) listed below. Triage has
narrowed the fix to an architectural choice between
spec-faithful (extend the polling app) and spec-pragmatic
(redesign the passive-fill mechanic locally) — both are
fixable, but the choice is operator-level.

The initial pass found the matcher / bet_manager / settlement /
equal-profit / overdraft logic faithful to spec, with only the
spec-acknowledged minor drifts. The follow-on probe of spec §7 Q5
("does the simulator divide tv by 2 before using it?") uncovered a
load-bearing data-pipeline issue (F7 below): **per-runner
`Prices.TradedVolume` is identically zero across all processed
parquet data**, which means the documented passive-fill mechanic
(queue position + cumulative trade volume) is dormant. Paired
passives at explicit prices fill on the very next tick because
both the queue threshold and the volume delta collapse to zero.

The matcher / settle / equal-profit code is still correct given the
inputs it receives. F7 is a divergence between spec and runtime
behaviour caused by a data-ingestion gap, not a bug in `env/*`. It
is load-bearing for the rewrite because the new trainer would be
built on a fill model that doesn't match the documented one.

The remaining minor divergences (F1–F6) are unchanged and remain
spec-acknowledged.

---

## Spec conformance summary

Total spec invariants checked: **19** (§1 fundamentals, §2 matching
rules, §3 walk-through math, §4 simulator mapping, §5 approximations
table, plus the load-bearing CLAUDE.md invariants).

| Status | Count |
|---|---|
| MATCH | 14 |
| ACCEPTABLE-APPROXIMATION (spec acknowledges) | 4 |
| FLAGGED-DRIFT (spec acknowledges) | 1 |
| DIVERGE — data-pipeline cause, load-bearing | 1 (F7) |
| NOT IMPLEMENTED (un-flagged) | 0 |

---

## Findings list

### F1. `MIN_BET_STAKE = 2.00` vs Betfair's £1 minimum

- **Spec.** §2 "Minimum stake" — Betfair Exchange minimum is £1
  since Feb 2022. §5 row 8 already labels this **flagged drift**.
- **Code.** [env/bet_manager.py:45](env/bet_manager.py:45) — module
  constant `MIN_BET_STAKE = 2.00`. Enforced in
  `BetManager.place_back` and `place_lay` as a hard refusal floor on
  matched stake.
- **Severity.** `cosmetic / open-question`. Stricter than live, so
  the simulator under-trades vs reality but never over-trades. Not
  load-bearing for credit assignment, P&L correctness, or the
  scorer's labels.
- **Why this matters for the new trainer.** It doesn't. Updating
  the constant is independent of the rewrite and would only affect
  whether sub-£2 partial fills survive — the scalping policy
  doesn't operate near that floor.

### F2. LTP as single-price proxy on high-volatility ticks

- **Spec.** §1 "Best back, best lay, LTP" + §5 row 1 + §7 Q1.
  Spec explicitly accepts LTP as a lossy single-price proxy for the
  passive-fill crossability gate.
- **Code.**
  [env/bet_manager.py:701–718](env/bet_manager.py:701) — Phase 1 of
  `PassiveOrderBook.on_tick` gates volume accumulation on
  `LTP vs order.price` comparison; [env/exchange_matcher.py:97](env/exchange_matcher.py:97)
  uses LTP as junk-filter reference.
- **Severity.** `open-question` (already in spec §7 Q1).
- **Why this matters for the new trainer.** Doesn't change credit
  assignment — fills are still gated by a defensible "trade price
  consistent with this side" check. Loss of fidelity is bounded
  and known.

### F3. Queue-ahead frozen at placement

- **Spec.** §1 "Queue position" + §5 row 2 + §7 Q2.
- **Code.** [env/bet_manager.py:157](env/bet_manager.py:157) —
  `PassiveOrder.queue_ahead_at_placement` is captured once at
  placement, never refreshed.
- **Severity.** `open-question` (already in spec §7 Q2). Biases
  toward slower fills (conservative).
- **Why this matters for the new trainer.** Doesn't. Conservative
  bias is preferable for an off-policy training surface.

### F4. Cross-matching not modelled

- **Spec.** §1 "Queue position" cross-matching caveat + §5 row 9
  + §7 Q4.
- **Code.** No cross-matching anywhere in `env/`.
- **Severity.** `open-question` (already in spec §7 Q4). Material
  for small-field binaries; minor for horse-race markets which
  dominate the training data.

### F5. `tv = backers_stake × 2` doubling — superseded by F7

The original spec-§7-Q5 question was *"is `tv` halved before
being compared against single-sided `queue_ahead`?"*. The
empirical follow-on probe found a more fundamental issue: the
`tv` value is **never non-zero in the first place** for any
runner in any processed parquet. See **F7** below — the doubling
question is moot until F7 is resolved.

### F6. `Keep` / `Persist` unmatched-bet semantics not modelled

- **Spec.** §2 "Unmatched-bet cancellation" + §5 row 3 + §7 Q6.
- **Code.** [env/bet_manager.py:392](env/bet_manager.py:392) —
  `cancel_all("race-off")` lapses every unmatched passive at
  race-off. No Keep/Persist support.
- **Severity.** `open-question` (already in spec §7 Q6). Lapse is
  Betfair's default; minor for pre-race scalping.

### F7. Per-runner `total_matched` is identically zero in the data → passive-fill model is dormant

**Severity. `blocker` (load-bearing for the rewrite).**

- **Spec.** §2 "Passive orders" + §1 "Matched volume semantics".
  Passive fills require *(a)* the market to come to the order's
  price AND *(b)* enough cumulative trade volume at that price to
  clear the queue ahead. Both gates are documented as load-bearing.
- **What the simulator actually does.** Phase 1 of
  [env/bet_manager.py::PassiveOrderBook.on_tick](env/bet_manager.py:643)
  computes `delta = max(0, snap.total_matched - prev_total_matched)`
  and adds it to each open order's `traded_volume_since_placement`.
  Phase 2 fills any order whose accumulated volume meets a threshold
  of `queue_ahead_at_placement + already_filled`.
- **What the data feeds it.** Empirically tested against
  `data/processed/2026-04-06.parquet`...`2026-04-15.parquet`
  (10 days × 100 markets/day): **per-runner
  `Prices.TradedVolume` is `0.0` on every active runner on every
  tick.** Market-level `traded_volume` is populated correctly
  (£100k–£6M per race), but the per-runner field is not.
- **Where the data drops.** [data/extractor.py:1252](data/extractor.py:1252)
  — the polled-→-legacy normaliser reads
  `state.get("totalMatched", 0.0)` per runner. The polled-data
  source `state` block doesn't carry that field for any runner,
  so the default-0 path always wins. (`pms.TotalMatched`, the
  market-level scalar at extractor.py:307, is populated and ends
  up in the parquet's `traded_volume` column — but that's
  market-wide, not per-runner, so it's not consumed by the
  passive-fill code.)
- **Empirical confirmation.** Mini-simulation on a real tick
  (market `1.256488956`, sid `49515011`, LTP 9.2):
  - Place aggressive back £10 @ 9.2.
  - Place paired passive lay £10 @ 8.74 (5 % below LTP, price
    not in visible ladder).
  - `queue_ahead_at_placement = 0.0` (price not on ladder),
    `traded_volume_since_placement = 0.0`.
  - Run `on_tick(next_tick)`. `next_tick.total_matched = 0.0`,
    so delta = 0. Phase 2 threshold = 0; Phase 2 check
    `0 < 0` is False → **fill**.
  - Net: paired passive at any unique tick-offset price fills on
    the very next tick, regardless of any actual trade flow.
- **Implications.**
  1. **The crossability gate from commit `4ee9fb5` is a no-op**
     in the current data — there are no trades to gate.
  2. **The `queue_ahead_at_placement` mechanism is dormant** —
     paired-arb passives chosen via `tick_offset` rarely land on
     a visible ladder level, so queue_ahead is almost always 0.
  3. **The ONLY active fill gate is the LTP junk-band filter**
     in Phase 2 (`lo <= order.price <= hi` against ±50 % of
     current LTP) — a *price* gate, not a *volume* gate.
  4. **Fill rates and timing are decoupled from real market
     dynamics.** Whatever passive-fill rate the policy
     experiences during training reflects the LTP junk-band
     check, not the spec's volume + queue model.
- **Why this matters for the new trainer.** The rewrite intends
  to keep the env unchanged. If the env's effective passive-fill
  model is "fill on the next tick the LTP is within ±50 % of
  resting price", the new trainer is training against that model
  — not the spec. Specifically:
  - Pair completion rates may be artificially high vs. live
    (production fills depend on real volume crossing).
  - The supervised scorer's labels — which presumably use the
    same env to determine arb feasibility — inherit the same
    bias.
  - The "force-close at T−N" rate in production training (~76 %
    per recent CLAUDE.md notes) is being measured against an
    artificial passive-fill mechanic. The diagnostics that drove
    the rewrite (cohorts O / O2 / F) all reflect this baseline.
- **Triage outcome (2026-04-26 follow-on, this session): outcome
  (b) — polled source genuinely doesn't expose per-runner
  cumulative tv.** Direct inspection of the upstream
  `hotdatarefactored.polledmarketsnapshots` table shows:
  - `RunnersJson[i].state` keys are exactly:
    `{adjustmentFactor, lastPriceTraded, sortPriority, status,
    totalMatched}`.
  - `RunnersJson[i].exchange` keys are exactly:
    `{availableToBack, availableToLay}`.
  - **`state.totalMatched` is `0.0` on every runner of every row
    sampled** — including healthy pre-race rows where the row's
    market-level `TotalMatched` column is in the £100k–£1M range
    (e.g. row Id `213845`, market `1.257042447`, MarketStatus
    `OPEN`, market `TotalMatched = £146,447.45`, every runner
    `state.totalMatched = 0.0`).
  - There is no `tv` per-price array, no per-price traded-volume
    field, no alternative cumulative-volume signal in the
    `RunnersJson` payload at all.

  This rules out outcome (a) (field-rename one-liner) and rules
  out outcome (c) (legacy `ResolvedMarketSnaps` table — the table
  doesn't exist in the live `hotdatarefactored` DB).

  **Fix path therefore narrows to one of three architectural
  options:**

  1. **Extend the upstream polling app** to attach
     `runner.totalMatched` from listMarketBook to each polled
     snapshot. listMarketBook DOES expose this field. Costs N
     extra REST calls per poll cycle. Lives in StreamRecorder1
     (outside this repo).
  2. **Sum the per-price `tv` array from Stream API** and persist
     it in `RunnersJson`. Requires the polling app to subscribe
     to Stream-API `tv` updates rather than (or alongside)
     listMarketBook polling. Lives in StreamRecorder1 (outside
     this repo).
  3. **Redesign the passive-fill mechanic** in
     `env/bet_manager.py::PassiveOrderBook.on_tick` to not depend
     on `snap.total_matched`. Possible inputs that ARE populated:
     market-level `traded_volume` (cumulative, on the Tick
     itself), per-runner `lastPriceTraded` deltas (a non-zero
     LTP change implies *some* trade occurred). This is the only
     option that keeps the fix local to this repo.

  Options (1) and (2) are the spec-faithful answers — they
  populate the field the spec assumes. Option (3) is a
  spec-pragmatic answer — it adapts the simulator to the data
  that is actually available. **Operator decision required.**
- **What the existing tests miss.** The 278 tests in
  `tests/test_*.py` use synthetic `Tick` / `RunnerSnap` objects
  with explicit non-zero `total_matched` values (e.g.
  `tests/test_session_2_7a.py:117` builds a tick with
  `total_matched=5000.0`). Fill mechanics work correctly when the
  inputs are populated. The test suite has no integration test
  that asserts "real parquet input produces non-zero
  `RunnerSnap.total_matched` values", which is why this gap has
  hidden.

  **Regression guards added (this session):**
  [tests/test_per_runner_total_matched_data.py](tests/test_per_runner_total_matched_data.py).
  Two layers:
  - **Parser contract tests** (pass today, must keep passing):
    feed a synthetic polled-format runner with a known non-zero
    `state.totalMatched` through `_polled_runners_to_snap_json`
    → `parse_snap_json` and assert the value reaches
    `RunnerSnap.total_matched` intact. Locks the plumbing so a
    future fix can't silently break the parser.
  - **Real-data integration tests** (FAIL today — gate for F7
    fix): load a real `data/processed/*.parquet`, assert at
    least one active runner on at least one healthy pre-race
    tick has `total_matched > 0`. The stronger variant focuses
    on a market with £100k+ market-level volume and asserts
    per-runner volume materialises somewhere across the day.
    Skipped automatically in environments without the
    gitignored data tree (CI safety). On a developer machine
    with the data, the failure is the F7 regression guard. Once
    F7 is fixed (under any of options 1–3 above), these tests
    pass and become the lock that prevents regression.

---

## CLAUDE.md correctness facts — PASS / FAIL

| # | Invariant | Status | Pointer |
|---|---|---|---|
| 1 | Order matching: single-price, no walking | **PASS** | [env/exchange_matcher.py:308–339](env/exchange_matcher.py:308) — `top = min/max(filtered, key=...)` then `matched = min(stake, adjusted_size)`. No iteration over levels after picking the top. |
| 2 | LTP required (strict path) | **PASS** | [env/exchange_matcher.py:286–287](env/exchange_matcher.py:286) — refuses with "no LTP for runner". |
| 3 | ±50 % junk filter applied before top-pick | **PASS** | [env/exchange_matcher.py:290–301](env/exchange_matcher.py:290). |
| 4 | Hard cap inside the matcher, after junk filter | **PASS** | [env/exchange_matcher.py:314–319](env/exchange_matcher.py:314) — cap checked against post-filter `top.price`, not raw top-of-book. |
| 5 | Force-close: relaxed matcher path (no LTP req, no junk filter, hard cap kept) | **PASS** | [env/exchange_matcher.py:279–284](env/exchange_matcher.py:279) — `force_close=True` skips both, hard cap still applied at line 314. |
| 6 | Force-close: budget overdraft permitted | **PASS** | [env/bet_manager.py:923–929](env/bet_manager.py:923) (back), [env/bet_manager.py:1042–1058](env/bet_manager.py:1042) (lay) — `force_close=True` skips the `available_budget` clamp and the lay-liability scale-down. |
| 7 | Force-close: equal-profit sizing (`equal_profit_lay_stake` / `equal_profit_back_stake`) | **PASS** | [env/betfair_env.py:2492–2505](env/betfair_env.py:2492) — `_attempt_close` calls the helpers, not 1:1. |
| 8 | Equal-profit formula: `S_lay = S_back × [P_back × (1−c) + c] / (P_lay − c)` | **PASS** | [env/scalping_math.py:163–164](env/scalping_math.py:163) — exact closed form. |
| 9 | Bet count = distinct matched orders, not netted positions | **PASS** | [env/bet_manager.py:878–879](env/bet_manager.py:878) — `bet_count = len(self.bets)`; each `place_back` / `place_lay` / passive fill appends a fresh `Bet`. |
| 10 | `info["realised_pnl"]` is last-race-only; `info["day_pnl"]` is episode-true | **PASS** | [env/betfair_env.py:1209–1216](env/betfair_env.py:1209) docstring + `_settle_current_race` accumulates `self._day_pnl += race_pnl`. `bm` is replaced per race at [env/betfair_env.py:1644](env/betfair_env.py:1644). |
| 11 | `bet_manager.bets` is last-race-only; `env.all_settled_bets` is episode-cumulative | **PASS** | [env/betfair_env.py:1616](env/betfair_env.py:1616) — `_settled_bets.extend(bm.bets)` before BetManager replacement. Reset per-episode at [env/betfair_env.py:1514](env/betfair_env.py:1514). |
| 12 | Passive crossability gate (LTP vs order.price) | **PASS** | [env/bet_manager.py:701–718](env/bet_manager.py:701) — Phase 1 skips delta accumulation on LAY when `ltp > price`, on BACK when `ltp < price`. The exact fix from commit `4ee9fb5`. |
| 13 | Commission applied only at settle, only on winning leg | **PASS** | [env/bet_manager.py:1199–1229](env/bet_manager.py:1199) — `(1 - commission)` multiplied into winning gross only; losers untouched. MTM at [env/betfair_env.py:1675–1679](env/betfair_env.py:1675) does NOT deduct commission (matches §5 row 7). |
| 14 | Default commission rate = 5 % | **PASS** | [env/betfair_env.py:769](env/betfair_env.py:769) — `reward_cfg.get("commission", 0.05)`; `config.yaml:40` ships `commission: 0.05`. |
| 15 | Settlement attribution: race_pnl = locked + closed + force_closed + naked sum | **PASS** | [env/betfair_env.py:3238–3243](env/betfair_env.py:3238) — `naked_pnl = race_pnl - locked - closed - force_closed` (i.e. the four buckets sum to `race_pnl` by construction). Reward channel splits documented in CLAUDE.md "Reward function: raw vs shaped". |
| 16 | Race-off cleanup cancels passives before settlement | **PASS** | [env/betfair_env.py:2961](env/betfair_env.py:2961) — `bm.passive_book.cancel_all("race-off")` runs at the top of `_settle_current_race`. Idempotent. |

---

## Test suite status

```
python -m pytest tests/test_exchange_matcher.py \
                tests/test_bet_manager.py \
                tests/test_betfair_env.py \
                tests/test_forced_arbitrage.py \
                -q
```

**Result.** 278 passed, 0 failed, 0 skipped. 4.46 s. No regressions.

---

## High-risk behaviour probes

### a. Settlement P&L attribution

`_settle_current_race` ([env/betfair_env.py:2786](env/betfair_env.py:2786)–3140):

- **Matured pair** (both legs filled, no `close_leg`): contributes
  to `scalping_locked_pnl` via `p["locked_pnl"]` (line 2913); pair
  cash flows through `bm.settle_race` into `race_pnl`.
- **Closed pair** (any leg has `close_leg=True`, no `force_close`):
  contributes covered-share P&L to `scalping_closed_pnl`
  (line 3214). Residual (uncovered fraction) falls through into
  the naked bucket.
- **Force-closed pair** (any leg has `force_close=True`):
  contributes covered-share P&L to `scalping_force_closed_pnl`
  (line 3212).
- **Naked pair** (only aggressive leg matched at race-off):
  `bm.settle_race` settles the lone leg into `race_pnl`; the
  bucket is then derived as the residual `race_pnl - locked -
  closed - force_closed` (line 3238).

`race_pnl == locked + closed + force_closed + naked` holds by
construction because `naked` is *defined* as the residual. No
double-counting, no missing buckets. **PASS.**

### b. Single-price matching with stake > top-level size

`ExchangeMatcher._match` ([env/exchange_matcher.py:331](env/exchange_matcher.py:331)):

```python
matched = min(stake, adjusted_size)
unmatched = stake - matched
return MatchResult(matched_stake=matched, unmatched_stake=unmatched, ...)
```

A £100 back vs £20 at the top-of-book lay returns
`MatchResult(matched_stake=20, unmatched_stake=80, average_price=top.price)`.
The £80 is "conceptually cancelled" — `place_back` deducts only
`result.matched_stake` from budget, never opens a resting order
for the residual, never iterates to the next level. **PASS.**

### c. Force-close overdraft

[env/bet_manager.py:923–929](env/bet_manager.py:923) (place_back):

```python
if force_close:
    capped = stake               # NO budget clamp
else:
    capped = min(stake, self.available_budget)
```

[env/bet_manager.py:1042–1058](env/bet_manager.py:1042) (place_lay):

```python
if not force_close and liability > self.available_budget:
    # scale-down branch — skipped under force_close
```

Both overdrafts run subject to `MIN_BET_STAKE` (lines 954, 1031,
1046). **PASS** — matches CLAUDE.md "Overdraft allowed for
force-close (2026-04-21)" exactly.

### d. Equal-profit sizing

[env/scalping_math.py:163–164](env/scalping_math.py:163):

```python
numerator = back_price * (1.0 - commission) + commission
return back_stake * numerator / (lay_price - commission)
```

Spot-check with the spec's worked example (S_b=50, P_b=1.52,
P_l=1.42, c=0.05):

- numerator = 1.52 × 0.95 + 0.05 = 1.494
- denom = 1.42 − 0.05 = 1.370
- S_l = 50 × 1.494 / 1.370 ≈ 54.524

Spec quotes £54.52. **PASS.**

CLAUDE.md "Equal-profit pair sizing (scalping)" worked example
(S_b=16, P_b=8.20, P_l=6.00, c=0.05) → 16 × 7.84 / 5.95 = £21.08.
Same formula, same result. **PASS.**

### e. `env.all_settled_bets` accumulates across races

[env/betfair_env.py:1514](env/betfair_env.py:1514): `self._settled_bets = []` in `reset()`.

[env/betfair_env.py:1616](env/betfair_env.py:1616): `self._settled_bets.extend(self.bet_manager.bets)` at end of each race, before BetManager replacement.

[env/betfair_env.py:1644](env/betfair_env.py:1644): `self.bet_manager = BetManager(...)` — fresh per race.

So `all_settled_bets` is per-episode cumulative, `bm.bets` is
per-race only. Matches spec / CLAUDE.md exactly. **PASS.**

---

## What this audit did NOT cover

Per the session prompt's hard constraints, this was a read-only
audit. The following are explicitly *out of scope* and remain
open follow-on tickets (operator triage):

- **Spec §7 Q5 — `tv` doubling units check.** Requires inspecting
  StreamRecorder1's ingestion path / a sample of raw Betfair
  stream data to confirm whether `TradedVolume` enters the
  simulator already-halved or as the raw doubled field. The audit
  surfaced that no `/2` exists in the consuming code, so resolution
  is binary: either source data is already single-sided (no fix
  needed) or it's doubled (uniform 2× over-credit on passive
  fills, fix is to halve at parser ingestion).
- **Real-Betfair-data validation.** The spec is the ground truth
  for this session; cross-checking against live ladder snapshots
  is a separate exercise.

---

## Action items before Phase 0

**Triage complete (2026-04-26).** Outcome (b) confirmed —
upstream polled source has no per-runner cumulative volume in
any field. Operator decision required between three architectural
options (listed in F7 above):

1. **Extend StreamRecorder1 to attach `runner.totalMatched` from
   listMarketBook.** Spec-faithful. Adds API-call cost to the
   polling cycle. Touches code outside this repo.
2. **Subscribe StreamRecorder1 to Stream-API `tv` per-price
   arrays and sum them.** Spec-faithful. Larger change to
   StreamRecorder1's polling architecture. Touches code outside
   this repo.
3. **Redesign `PassiveOrderBook.on_tick` to not depend on
   `snap.total_matched`.** Spec-pragmatic. Available inputs:
   market-level `traded_volume` deltas (already populated),
   per-runner LTP deltas. Local to this repo. Loses some fidelity
   vs. the spec model but is implementable today.

A separate session is needed to scope and prototype the chosen
option. The session prompt should be drafted by the operator —
the choice between (1)–(3) is a business / scope call, not a
mechanical fix.

Other follow-ons (non-gating, can sequence after F7):

- **F1–F6 follow-ons.** Same as in the original audit — none of
  these block Phase 0 individually; sequencing depends on operator
  priorities.
- **`MIN_BET_STAKE` constant.** Trivial drop-in to bring it in
  line with Betfair's £1 minimum. Independent of F7.

---

## Verdict

**RED — Phase 0 paused.** F7 is a load-bearing divergence between
spec and runtime behaviour: the documented passive-fill mechanic
is dormant because per-runner `total_matched` is identically zero
in the data the env consumes. The new trainer would inherit a
fill model that doesn't match `docs/betfair_market_model.md`. F7
needs operator triage and a data-pipeline fix (or env redesign)
before Phase 0 ships.
